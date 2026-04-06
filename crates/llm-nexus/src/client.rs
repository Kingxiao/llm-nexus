//! NexusClient — the unified entry point for LLM interactions.

use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::loader;
use llm_nexus_core::pipeline::context::RequestContext;
use llm_nexus_core::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::traits::embedding::EmbeddingProvider;
use llm_nexus_core::traits::metrics::{AggregatedStats, CallRecord, MetricsBackend, StatsFilter};
use llm_nexus_core::traits::router::{RouteContext, Router};
use llm_nexus_core::types::config::NexusConfig;
use llm_nexus_core::types::embed::{EmbedRequest, EmbedResponse};
use llm_nexus_core::types::model::ModelMetadata;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};
use llm_nexus_metrics::InMemoryMetrics;
use llm_nexus_registry::StaticRegistry;
use llm_nexus_router::CostRouter;

use crate::dispatcher::ProviderDispatcher;
use crate::provider_map::ProviderMap;

/// The main client for interacting with LLM providers.
pub struct NexusClient {
    dispatcher: Arc<ProviderDispatcher>,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    embedding_providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    registry: Arc<StaticRegistry>,
    router: Arc<dyn Router>,
    metrics: Arc<dyn MetricsBackend>,
    config: NexusConfig,
}

impl NexusClient {
    /// Create a new builder.
    pub fn builder() -> NexusClientBuilder {
        NexusClientBuilder::new()
    }

    /// Load from a config directory (providers.toml + models.toml).
    /// Only registers providers whose API keys are available in the environment.
    pub fn from_config_dir(config_dir: &Path) -> NexusResult<Self> {
        let mut builder = Self::builder().config_dir(config_dir)?;
        builder = builder.auto_register_providers();
        builder.build()
    }

    /// Send a chat request to a specific model (provider resolved from registry).
    ///
    /// The request flows through the middleware pipeline before reaching the provider.
    pub async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        let mut ctx = RequestContext::new(uuid::Uuid::new_v4().to_string());

        let next = Next {
            middlewares: &self.middlewares,
            dispatcher: self.dispatcher.as_ref(),
        };
        let result = next.run(&mut ctx, request).await;

        // Record metrics (model_meta populated by dispatcher during pipeline)
        if let Some(ref model_meta) = ctx.model_meta {
            self.record_metrics(
                &ctx.request_id,
                &request.model,
                model_meta,
                ctx.elapsed_ms(),
                &result,
            )
            .await;
        }

        result
    }

    /// Send a streaming chat request.
    ///
    /// The request flows through the middleware pipeline before reaching the provider.
    /// Metrics are recorded when the stream completes (from the final chunk's usage).
    pub async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let mut ctx = RequestContext::new(uuid::Uuid::new_v4().to_string());

        let next = NextStream {
            middlewares: &self.middlewares,
            dispatcher: self.dispatcher.as_ref(),
        };
        let stream = next.run(&mut ctx, request).await?;

        // Wrap stream to record metrics from the final chunk's usage data
        let metrics = self.metrics.clone();
        let model_id = request.model.clone();
        let model_meta = ctx.model_meta.clone();
        let request_id = ctx.request_id.clone();
        let start_time = ctx.start_time;

        let metered_stream = stream.inspect(move |chunk| {
            let Some(usage) = chunk.as_ref().ok().and_then(|c| c.usage.as_ref()) else {
                return;
            };
            let Some(meta) = &model_meta else { return };

            let latency_ms = start_time.elapsed().as_millis() as u64;
            let cost = llm_nexus_metrics::cost_calculator::calculate_cost(usage, meta);
            let record = CallRecord {
                request_id: request_id.clone(),
                provider_id: meta.provider.clone(),
                model_id: model_id.clone(),
                latency_ms,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                estimated_cost_usd: cost,
                success: true,
                error: None,
                timestamp: chrono::Utc::now(),
            };
            let metrics = metrics.clone();
            tokio::spawn(async move {
                let _ = metrics.record_call(record).await;
            });
        });

        Ok(Box::pin(metered_stream))
    }

    /// Batch chat: send multiple requests through the pipeline concurrently.
    ///
    /// Each request goes through the full middleware pipeline independently.
    pub async fn chat_batch(
        &self,
        requests: &[ChatRequest],
        concurrency: usize,
    ) -> Vec<NexusResult<ChatResponse>> {
        use futures::stream::StreamExt;

        futures::stream::iter(requests)
            .map(|req| self.chat(req))
            .buffer_unordered(concurrency)
            .collect()
            .await
    }

    /// Chat with automatic routing — selects the best model for the context,
    /// applies it to the given request, and falls back through the chain on failure.
    pub async fn chat_with_routing(
        &self,
        request: &ChatRequest,
        context: &RouteContext,
    ) -> NexusResult<ChatResponse> {
        let chain = self.router.fallback_chain(context).await?;
        let mut errors = Vec::new();

        for decision in &chain {
            // Verify provider is registered before attempting
            if !self.dispatcher.providers.contains(&decision.provider_id) {
                errors.push(NexusError::ConfigError(format!(
                    "Provider '{}' not registered",
                    decision.provider_id
                )));
                continue;
            }

            let mut routed_request = request.clone();
            routed_request.model = decision.model_id.clone();

            // Route through the full middleware pipeline (cache, guardrails, budget, etc.)
            match self.chat(&routed_request).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    tracing::warn!(
                        provider = %decision.provider_id,
                        model = %decision.model_id,
                        error = %e,
                        "routing fallback: provider failed"
                    );
                    errors.push(e);
                }
            }
        }

        Err(NexusError::AllProvidersFailed(errors))
    }

    /// Generate embeddings. Routes to the appropriate EmbeddingProvider based on
    /// model metadata (provider field in registry).
    pub async fn embed(&self, request: &EmbedRequest) -> NexusResult<EmbedResponse> {
        // Try to find provider from registry
        let provider_id = self
            .resolve_model(&request.model)
            .await
            .map(|m| m.provider)?;

        let provider = self.embedding_providers.get(&provider_id).ok_or_else(|| {
            NexusError::ModelNotFound(format!(
                "no embedding provider registered for '{provider_id}'"
            ))
        })?;

        provider.embed(request).await
    }

    /// Query aggregated metrics.
    pub async fn stats(&self, filter: &StatsFilter) -> NexusResult<AggregatedStats> {
        self.metrics.query_stats(filter).await
    }

    /// Get the list of registered provider IDs.
    pub fn provider_ids(&self) -> Vec<String> {
        self.dispatcher.providers.provider_ids()
    }

    /// Get access to the model registry.
    pub fn registry(&self) -> &StaticRegistry {
        &self.registry
    }

    /// Get the metrics backend (e.g. to wrap with PrometheusExporter).
    pub fn metrics(&self) -> &Arc<dyn MetricsBackend> {
        &self.metrics
    }

    /// Get the loaded config.
    pub fn config(&self) -> &NexusConfig {
        &self.config
    }

    // -- private helpers --

    async fn resolve_model(&self, model_id: &str) -> NexusResult<ModelMetadata> {
        use llm_nexus_core::traits::registry::ModelRegistry;
        self.registry
            .get_model(model_id)
            .await?
            .ok_or_else(|| NexusError::ModelNotFound(model_id.into()))
    }

    async fn record_metrics(
        &self,
        request_id: &str,
        model_id: &str,
        model_meta: &ModelMetadata,
        latency_ms: u64,
        result: &NexusResult<ChatResponse>,
    ) {
        let (success, usage, error) = match result {
            Ok(resp) => (true, Some(&resp.usage), None),
            Err(e) => (false, None, Some(e.to_string())),
        };

        let prompt_tokens = usage.map_or(0, |u| u.prompt_tokens);
        let completion_tokens = usage.map_or(0, |u| u.completion_tokens);
        let cost = llm_nexus_metrics::cost_calculator::calculate_cost(
            &llm_nexus_core::types::response::Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            model_meta,
        );

        let record = CallRecord {
            request_id: request_id.to_string(),
            provider_id: model_meta.provider.clone(),
            model_id: model_id.into(),
            latency_ms,
            prompt_tokens,
            completion_tokens,
            estimated_cost_usd: cost,
            success,
            error,
            timestamp: chrono::Utc::now(),
        };

        if let Err(e) = self.metrics.record_call(record).await {
            tracing::warn!(error = %e, "failed to record metrics");
        }
    }
}

/// Builder for NexusClient.
pub struct NexusClientBuilder {
    config: Option<NexusConfig>,
    models: Vec<ModelMetadata>,
    providers: ProviderMap,
    embedding_providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
    metrics: Option<Arc<dyn MetricsBackend>>,
    router: Option<Arc<dyn Router>>,
}

impl NexusClientBuilder {
    pub fn new() -> Self {
        Self {
            config: None,
            models: Vec::new(),
            providers: ProviderMap::new(),
            embedding_providers: HashMap::new(),
            middlewares: Vec::new(),
            metrics: None,
            router: None,
        }
    }

    /// Load config from directory.
    pub fn config_dir(mut self, dir: &Path) -> NexusResult<Self> {
        let (mut config, models) = loader::load_config(dir)?;
        loader::apply_env_overrides(&mut config);
        self.config = Some(config);
        self.models = models;
        Ok(self)
    }

    /// Register a model in the registry.
    ///
    /// Use this when not loading from config_dir, e.g. when building
    /// the client programmatically.
    pub fn with_model(mut self, model: ModelMetadata) -> Self {
        self.models.push(model);
        self
    }

    /// Register a provider manually.
    pub fn with_provider(mut self, id: impl Into<String>, provider: Arc<dyn ChatProvider>) -> Self {
        self.providers.register(id, provider);
        self
    }

    /// Use in-memory metrics (default if none specified).
    pub fn with_in_memory_metrics(mut self) -> Self {
        self.metrics = Some(Arc::new(InMemoryMetrics::new()));
        self
    }

    /// Use a custom metrics backend.
    pub fn with_metrics(mut self, metrics: Arc<dyn MetricsBackend>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Register an embedding provider.
    pub fn with_embedding_provider(
        mut self,
        id: impl Into<String>,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        self.embedding_providers.insert(id.into(), provider);
        self
    }

    /// Add a middleware to the request pipeline.
    ///
    /// Middleware are executed in the order they are added. The first middleware
    /// added is the outermost (runs first on request, last on response).
    pub fn with_middleware(mut self, middleware: Arc<dyn ChatMiddleware>) -> Self {
        self.middlewares.push(middleware);
        self
    }

    /// Use a custom router (e.g., composite router with weighted scoring).
    pub fn with_router(mut self, router: Arc<dyn Router>) -> Self {
        self.router = Some(router);
        self
    }

    /// Auto-register providers based on config. Only registers providers
    /// whose API keys are set in the environment.
    ///
    /// Uses [`crate::provider_factory::create_provider`] to construct each
    /// provider — adding a new provider only requires a match arm there.
    pub fn auto_register_providers(mut self) -> Self {
        let Some(config) = &self.config else {
            return self;
        };

        for (id, provider_config) in &config.providers {
            // Skip if API key not set
            if std::env::var(&provider_config.api_key_env).is_err() {
                tracing::debug!(provider = %id, "skipping provider: API key not set");
                continue;
            }

            let registration = match crate::provider_factory::create_provider(id, provider_config) {
                Some(Ok(reg)) => reg,
                Some(Err(e)) => {
                    tracing::warn!(provider = %id, error = %e, "failed to register provider");
                    continue;
                }
                None => {
                    tracing::debug!(provider = %id, "no adapter available");
                    continue;
                }
            };

            if let Some(chat) = registration.chat {
                self.providers.register(id.clone(), chat);
                tracing::info!(provider = %id, "registered chat provider");
            }
            if let Some(embed) = registration.embedding {
                self.embedding_providers.insert(id.clone(), embed);
                tracing::info!(provider = %id, "registered embedding provider");
            }
        }

        self
    }

    /// Build the NexusClient.
    pub fn build(self) -> NexusResult<NexusClient> {
        let config = self.config.unwrap_or_else(|| NexusConfig {
            providers: std::collections::HashMap::new(),
        });

        let registry = Arc::new(StaticRegistry::from_models(self.models.clone()));
        let router = self
            .router
            .unwrap_or_else(|| Arc::new(CostRouter::new(self.models)));
        let metrics = self
            .metrics
            .unwrap_or_else(|| Arc::new(InMemoryMetrics::new()));

        let dispatcher = Arc::new(ProviderDispatcher {
            providers: self.providers,
            registry: registry.clone(),
        });

        // Ensure retry/timeout middleware are always innermost (closest to provider).
        // This prevents double-execution of outer middleware (budget, guardrail) on retry.
        let middlewares = reorder_builtin_middleware(self.middlewares);

        Ok(NexusClient {
            dispatcher,
            middlewares,
            embedding_providers: self.embedding_providers,
            registry,
            router,
            metrics,
            config,
        })
    }
}

/// Reorder middleware so RetryMiddleware and TimeoutMiddleware are always last (innermost).
///
/// This prevents retry from re-executing outer middleware (budget, guardrail)
/// on each retry attempt, which would cause double-deduction or duplicate checks.
///
/// Final order: [...user middleware...] → timeout → retry → provider
fn reorder_builtin_middleware(
    middlewares: Vec<Arc<dyn ChatMiddleware>>,
) -> Vec<Arc<dyn ChatMiddleware>> {
    let mut regular = Vec::new();
    let mut timeout = None;
    let mut retry = None;

    for mw in middlewares {
        match mw.name() {
            "builtin::timeout" => timeout = Some(mw),
            "builtin::retry" => retry = Some(mw),
            _ => regular.push(mw),
        }
    }

    // Append in order: timeout (outer) → retry (inner, closest to provider)
    if let Some(t) = timeout {
        regular.push(t);
    }
    if let Some(r) = retry {
        regular.push(r);
    }
    regular
}

impl Default for NexusClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::response::Usage;

    struct MockProvider {
        id: String,
        response: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for MockProvider {
        fn provider_id(&self) -> &str {
            &self.id
        }
        async fn chat(&self, _req: &ChatRequest) -> NexusResult<ChatResponse> {
            Ok(ChatResponse {
                id: "mock-id".into(),
                model: "mock-model".into(),
                content: self.response.clone(),
                finish_reason: None,
                usage: Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                },
                tool_calls: None,
            })
        }
        async fn chat_stream(
            &self,
            _req: &ChatRequest,
        ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
            Ok(Box::pin(futures::stream::empty()))
        }
        async fn list_models(&self) -> NexusResult<Vec<String>> {
            Ok(vec![])
        }
    }

    fn test_models() -> Vec<ModelMetadata> {
        vec![ModelMetadata {
            id: "test-model".into(),
            provider: "test-provider".into(),
            display_name: "Test Model".into(),
            context_window: 128000,
            max_output_tokens: Some(4096),
            input_price_per_1m: 2.50,
            output_price_per_1m: 10.00,
            capabilities: vec![llm_nexus_core::types::model::Capability::Chat],
            features: Default::default(),
            latency_baseline_ms: None,
        }]
    }

    #[tokio::test]
    async fn test_client_builder_and_chat() {
        let _unused_builder_client = NexusClient::builder()
            .with_provider(
                "test-provider",
                Arc::new(MockProvider {
                    id: "test-provider".into(),
                    response: "Hello from mock!".into(),
                }),
            )
            .with_in_memory_metrics()
            .build()
            .unwrap();

        // Manually set models so registry can resolve
        let models = test_models();
        let registry = Arc::new(StaticRegistry::from_models(models.clone()));
        let router = Arc::new(CostRouter::new(models));
        let metrics = Arc::new(InMemoryMetrics::new());

        let mut providers = ProviderMap::new();
        providers.register(
            "test-provider",
            Arc::new(MockProvider {
                id: "test-provider".into(),
                response: "Hello from mock!".into(),
            }),
        );

        let client = NexusClient {
            dispatcher: Arc::new(crate::dispatcher::ProviderDispatcher {
                providers,
                registry: registry.clone(),
            }),
            middlewares: vec![],
            embedding_providers: HashMap::new(),
            registry,
            router,
            metrics: metrics.clone(),
            config: NexusConfig {
                providers: std::collections::HashMap::new(),
            },
        };

        let request = ChatRequest {
            model: "test-model".into(),
            messages: vec![llm_nexus_core::types::request::Message::user("Hi")],
            ..Default::default()
        };

        let response = client.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from mock!");

        // Verify metrics were recorded
        let stats = client.stats(&StatsFilter::default()).await.unwrap();
        assert_eq!(stats.total_calls, 1);
        assert_eq!(stats.successful_calls, 1);
        assert!(stats.total_cost_usd > 0.0);
    }

    #[tokio::test]
    async fn test_model_not_found() {
        let client = NexusClient::builder()
            .with_in_memory_metrics()
            .build()
            .unwrap();

        let request = ChatRequest {
            model: "nonexistent".into(),
            messages: vec![],
            ..Default::default()
        };

        let result = client.chat(&request).await;
        assert!(matches!(result, Err(NexusError::ModelNotFound(_))));
    }

    #[test]
    fn test_builder_default() {
        let client = NexusClient::builder().build().unwrap();
        assert!(client.provider_ids().is_empty());
    }

    #[tokio::test]
    async fn test_from_config_dir() {
        let config_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config");
        // This will skip all providers since no API keys are set in test env
        let client = NexusClient::from_config_dir(&config_dir).unwrap();
        // Registry should have models even without providers
        let model = client.resolve_model("gpt-5.4").await;
        assert!(model.is_ok());
    }
}
