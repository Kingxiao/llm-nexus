//! Remote model registry that syncs pricing/metadata from OpenRouter's API.
//!
//! Three-level merge strategy: user override > remote > builtin TOML.
//! Caches the last successful sync to a JSON file for offline fallback.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::registry::ModelRegistry;
use llm_nexus_core::types::model::{
    Capability, ModelFeatures, ModelFilter, ModelMetadata, ModelOverride,
};

use crate::filter::matches_filter;

/// Remote registry that syncs model metadata from an OpenRouter-compatible API.
pub struct RemoteRegistry {
    /// Builtin models from TOML (lowest priority).
    builtin: Vec<ModelMetadata>,
    /// Remote models from last sync (medium priority).
    remote: RwLock<Vec<ModelMetadata>>,
    /// User overrides (highest priority).
    overrides: RwLock<HashMap<String, ModelOverride>>,
    /// HTTP client for fetching remote data.
    http: reqwest::Client,
    /// API endpoint for model listing.
    api_url: String,
    /// API key (optional — OpenRouter's model list is public).
    api_key: Option<String>,
    /// Path to cache the last successful sync result.
    cache_path: Option<PathBuf>,
}

impl RemoteRegistry {
    /// Create a new RemoteRegistry.
    ///
    /// - `builtin`: models from TOML config (fallback)
    /// - `api_url`: e.g. `https://openrouter.ai/api/v1/models`
    /// - `api_key`: optional API key for authenticated endpoints
    /// - `cache_path`: optional path to persist last sync result
    pub fn new(
        builtin: Vec<ModelMetadata>,
        api_url: impl Into<String>,
        api_key: Option<String>,
        cache_path: Option<PathBuf>,
    ) -> Self {
        // Load cached remote models if available
        let cached = cache_path
            .as_ref()
            .and_then(|p| load_cache(p).ok())
            .unwrap_or_default();

        Self {
            builtin,
            remote: RwLock::new(cached),
            overrides: RwLock::new(HashMap::new()),
            http: reqwest::Client::new(),
            api_url: api_url.into(),
            api_key,
            cache_path,
        }
    }

    /// Merge builtin + remote + overrides into a single list.
    /// Remote models with the same ID override builtin.
    fn merged_models(&self) -> Vec<ModelMetadata> {
        let remote = self.remote.read().expect("RwLock poisoned");
        let overrides = self.overrides.read().expect("RwLock poisoned");

        let mut by_id: HashMap<String, ModelMetadata> = HashMap::new();

        // Layer 1: builtin (lowest priority)
        for m in &self.builtin {
            by_id.insert(m.id.clone(), m.clone());
        }

        // Layer 2: remote (overwrites builtin)
        for m in remote.iter() {
            by_id.insert(m.id.clone(), m.clone());
        }

        // Layer 3: user overrides (highest priority, applied on top)
        let mut result: Vec<ModelMetadata> = by_id.into_values().collect();
        for model in &mut result {
            if let Some(ov) = overrides.get(&model.id) {
                apply_override(model, ov);
            }
        }

        result.sort_by(|a, b| a.id.cmp(&b.id));
        result
    }
}

#[async_trait::async_trait]
impl ModelRegistry for RemoteRegistry {
    async fn get_model(&self, model_id: &str) -> NexusResult<Option<ModelMetadata>> {
        Ok(self.merged_models().into_iter().find(|m| m.id == model_id))
    }

    async fn list_models(&self, filter: &ModelFilter) -> NexusResult<Vec<ModelMetadata>> {
        Ok(self
            .merged_models()
            .into_iter()
            .filter(|m| matches_filter(m, filter))
            .collect())
    }

    async fn refresh(&self) -> NexusResult<()> {
        let mut request = self.http.get(&self.api_url);
        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request
            .send()
            .await
            .map_err(|e| NexusError::HttpError(format!("failed to fetch remote models: {e}")))?;

        if !response.status().is_success() {
            return Err(NexusError::ProviderError {
                provider: "remote-registry".into(),
                message: format!("remote API returned status {}", response.status()),
                status_code: Some(response.status().as_u16()),
            });
        }

        let body: OpenRouterModelsResponse = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("failed to parse remote models: {e}"))
        })?;

        let models: Vec<ModelMetadata> = body
            .data
            .into_iter()
            .filter_map(|m| convert_openrouter_model(m).ok())
            .collect();

        tracing::info!(count = models.len(), "synced models from remote registry");

        // Persist to cache
        if let Some(ref path) = self.cache_path
            && let Err(e) = save_cache(path, &models)
        {
            tracing::warn!(error = %e, "failed to save remote registry cache");
        }

        // Update remote models
        let mut remote = self.remote.write().expect("RwLock poisoned");
        *remote = models;

        Ok(())
    }

    fn apply_override(&self, model_id: &str, overrides: ModelOverride) -> NexusResult<()> {
        // Verify model exists in either builtin or remote
        let exists = self.merged_models().iter().any(|m| m.id == model_id);
        if !exists {
            return Err(NexusError::ModelNotFound(model_id.to_string()));
        }
        let mut map = self.overrides.write().expect("RwLock poisoned");
        map.insert(model_id.to_string(), overrides);
        Ok(())
    }
}

// ---------- OpenRouter API types ----------

#[derive(Debug, serde::Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenRouterModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    top_provider: Option<TopProvider>,
    #[serde(default)]
    pricing: Option<OpenRouterPricing>,
    #[serde(default)]
    architecture: Option<Architecture>,
}

#[derive(Debug, serde::Deserialize)]
struct TopProvider {
    #[serde(default)]
    max_completion_tokens: Option<u32>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenRouterPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct Architecture {
    #[serde(default)]
    modality: Option<String>,
}

fn convert_openrouter_model(m: OpenRouterModel) -> NexusResult<ModelMetadata> {
    // OpenRouter pricing is per-token as a string, convert to per-1M
    let input_price = m
        .pricing
        .as_ref()
        .and_then(|p| p.prompt.as_deref())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        * 1_000_000.0;

    let output_price = m
        .pricing
        .as_ref()
        .and_then(|p| p.completion.as_deref())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        * 1_000_000.0;

    // Extract provider from model ID (e.g., "openai/gpt-5.4" -> "openai")
    let provider = m.id.split('/').next().unwrap_or("unknown").to_string();

    let modality = m
        .architecture
        .as_ref()
        .and_then(|a| a.modality.as_deref())
        .unwrap_or("");

    let vision = modality.contains("image") || modality.contains("multimodal");

    let mut capabilities = vec![Capability::Chat];
    if vision {
        capabilities.push(Capability::ImageUnderstanding);
    }
    if m.context_length.unwrap_or(0) >= 100_000 {
        capabilities.push(Capability::LongContext);
    }

    Ok(ModelMetadata {
        id: m.id,
        provider,
        display_name: m.name.unwrap_or_default(),
        context_window: m.context_length.unwrap_or(4096),
        max_output_tokens: m.top_provider.and_then(|p| p.max_completion_tokens),
        input_price_per_1m: input_price,
        output_price_per_1m: output_price,
        capabilities,
        features: ModelFeatures {
            vision,
            tool_use: true, // most models support this
            json_mode: true,
            streaming: true,
            system_prompt: true,
        },
        latency_baseline_ms: None,
    })
}

// ---------- Cache ----------

fn load_cache(path: &Path) -> NexusResult<Vec<ModelMetadata>> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| NexusError::ConfigError(format!("failed to read cache: {e}")))?;
    serde_json::from_str(&data)
        .map_err(|e| NexusError::SerializationError(format!("failed to parse cache: {e}")))
}

fn save_cache(path: &Path, models: &[ModelMetadata]) -> NexusResult<()> {
    let data = serde_json::to_string_pretty(models)
        .map_err(|e| NexusError::SerializationError(format!("failed to serialize cache: {e}")))?;
    std::fs::write(path, data)
        .map_err(|e| NexusError::ConfigError(format!("failed to write cache: {e}")))
}

fn apply_override(model: &mut ModelMetadata, ov: &ModelOverride) {
    if let Some(v) = ov.input_price_per_1m {
        model.input_price_per_1m = v;
    }
    if let Some(v) = ov.output_price_per_1m {
        model.output_price_per_1m = v;
    }
    if let Some(v) = ov.context_window {
        model.context_window = v;
    }
    if let Some(v) = ov.max_output_tokens {
        model.max_output_tokens = Some(v);
    }
    if let Some(v) = ov.latency_baseline_ms {
        model.latency_baseline_ms = Some(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn builtin_models() -> Vec<ModelMetadata> {
        vec![ModelMetadata {
            id: "builtin-model".into(),
            provider: "test".into(),
            display_name: "Builtin".into(),
            context_window: 4096,
            max_output_tokens: Some(1024),
            input_price_per_1m: 1.0,
            output_price_per_1m: 2.0,
            capabilities: vec![Capability::Chat],
            features: ModelFeatures::default(),
            latency_baseline_ms: None,
        }]
    }

    fn mock_openrouter_response() -> serde_json::Value {
        serde_json::json!({
            "data": [
                {
                    "id": "openai/gpt-5.4",
                    "name": "GPT-5.4",
                    "context_length": 1000000,
                    "top_provider": { "max_completion_tokens": 100000 },
                    "pricing": { "prompt": "0.0000025", "completion": "0.000015" },
                    "architecture": { "modality": "text+image->text" }
                },
                {
                    "id": "anthropic/claude-sonnet-4-6",
                    "name": "Claude Sonnet 4.6",
                    "context_length": 200000,
                    "pricing": { "prompt": "0.000003", "completion": "0.000015" }
                }
            ]
        })
    }

    #[tokio::test]
    async fn test_remote_sync_and_merge() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/api/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_openrouter_response()))
            .mount(&server)
            .await;

        let registry = RemoteRegistry::new(
            builtin_models(),
            format!("{}/api/v1/models", server.uri()),
            None,
            None,
        );

        // Before refresh: only builtin
        let models = registry.list_models(&ModelFilter::default()).await.unwrap();
        assert_eq!(models.len(), 1);

        // Refresh from mock server
        registry.refresh().await.unwrap();

        // After refresh: builtin + 2 remote = 3
        let models = registry.list_models(&ModelFilter::default()).await.unwrap();
        assert_eq!(models.len(), 3);

        // Verify remote model pricing converted correctly
        let gpt = registry.get_model("openai/gpt-5.4").await.unwrap().unwrap();
        assert!((gpt.input_price_per_1m - 2.5).abs() < 0.01);
        assert!((gpt.output_price_per_1m - 15.0).abs() < 0.01);
        assert_eq!(gpt.context_window, 1000000);
        assert!(gpt.capabilities.contains(&Capability::ImageUnderstanding));
        assert!(gpt.capabilities.contains(&Capability::LongContext));
    }

    #[tokio::test]
    async fn test_remote_overwrites_builtin() {
        let server = MockServer::start().await;

        // Remote has a model with the same ID as builtin
        let response = serde_json::json!({
            "data": [{
                "id": "builtin-model",
                "name": "Remote Version",
                "context_length": 128000,
                "pricing": { "prompt": "0.000005", "completion": "0.00001" }
            }]
        });

        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(response))
            .mount(&server)
            .await;

        let registry = RemoteRegistry::new(
            builtin_models(),
            format!("{}/models", server.uri()),
            None,
            None,
        );

        registry.refresh().await.unwrap();

        let model = registry.get_model("builtin-model").await.unwrap().unwrap();
        // Remote overwrites builtin
        assert_eq!(model.display_name, "Remote Version");
        assert_eq!(model.context_window, 128000);
    }

    #[tokio::test]
    async fn test_user_override_wins() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_openrouter_response()))
            .mount(&server)
            .await;

        let registry = RemoteRegistry::new(
            builtin_models(),
            format!("{}/models", server.uri()),
            None,
            None,
        );

        registry.refresh().await.unwrap();

        // Apply user override
        ModelRegistry::apply_override(
            &registry,
            "openai/gpt-5.4",
            ModelOverride {
                input_price_per_1m: Some(0.99),
                ..Default::default()
            },
        )
        .unwrap();

        let model = registry.get_model("openai/gpt-5.4").await.unwrap().unwrap();
        assert!((model.input_price_per_1m - 0.99).abs() < 0.01);
        // Output price unchanged from remote
        assert!((model.output_price_per_1m - 15.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_degraded_mode_on_failure() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let registry = RemoteRegistry::new(
            builtin_models(),
            format!("{}/models", server.uri()),
            None,
            None,
        );

        // Refresh fails
        let result = registry.refresh().await;
        assert!(result.is_err());

        // But builtin models still work
        let models = registry.list_models(&ModelFilter::default()).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "builtin-model");
    }

    #[tokio::test]
    async fn test_cache_persistence() {
        let server = MockServer::start().await;

        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_openrouter_response()))
            .mount(&server)
            .await;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cache_path = tmp.path().to_path_buf();

        let registry = RemoteRegistry::new(
            builtin_models(),
            format!("{}/models", server.uri()),
            None,
            Some(cache_path.clone()),
        );

        registry.refresh().await.unwrap();

        // Cache file should exist and be valid JSON
        let cached: Vec<ModelMetadata> =
            serde_json::from_str(&std::fs::read_to_string(&cache_path).unwrap()).unwrap();
        assert_eq!(cached.len(), 2);

        // Create new registry from cache (no refresh needed)
        let registry2 = RemoteRegistry::new(
            builtin_models(),
            "http://localhost:0/dead", // unreachable
            None,
            Some(cache_path),
        );

        let models = registry2
            .list_models(&ModelFilter::default())
            .await
            .unwrap();
        // builtin + 2 cached = 3
        assert_eq!(models.len(), 3);
    }
}
