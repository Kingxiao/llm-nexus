//! End-to-end pipeline integration tests.
//!
//! Verifies that:
//! 1. Middleware chain intercepts requests in correct order
//! 2. Cache middleware short-circuits the pipeline on hit
//! 3. Guardrail middleware blocks forbidden content
//! 4. Multiple middleware stack correctly
//! 5. Middleware-free path still works (backward compat)
//! 6. Batch chat works through the pipeline

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;
use std::pin::Pin;

use llm_nexus::client::NexusClient;
use llm_nexus_cache::CacheMiddleware;
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::pipeline::context::RequestContext;
use llm_nexus_core::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use llm_nexus_core::store::InMemoryStore;
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::request::{ChatRequest, Message};
use llm_nexus_core::types::response::{ChatResponse, StreamChunk, Usage};
use llm_nexus_guardrail::{GuardrailMiddleware, KeywordFilter};

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Provider that counts how many times chat() was called.
struct CountingProvider {
    call_count: AtomicU32,
    response: String,
}

impl CountingProvider {
    fn new(response: &str) -> Self {
        Self {
            call_count: AtomicU32::new(0),
            response: response.to_string(),
        }
    }

    fn count(&self) -> u32 {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl ChatProvider for CountingProvider {
    fn provider_id(&self) -> &str {
        "openai" // Must match the provider in config/models.toml for "gpt-5.4"
    }
    async fn chat(&self, _req: &ChatRequest) -> NexusResult<ChatResponse> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(ChatResponse {
            id: "test-id".into(),
            model: "gpt-5.4".into(),
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
        Ok(vec!["gpt-5.4".into()])
    }
}

/// Middleware that records call order for pipeline ordering verification.
struct MarkerMiddleware {
    name: String,
    log: Arc<std::sync::Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl ChatMiddleware for MarkerMiddleware {
    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        self.log.lock().unwrap().push(format!("{}-pre", self.name));
        let result = next.run(ctx, request).await;
        self.log.lock().unwrap().push(format!("{}-post", self.name));
        result
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}-stream", self.name));
        next.run(ctx, request).await
    }
}

fn config_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")
}

/// A request using gpt-5.4 which exists in config/models.toml.
fn test_request() -> ChatRequest {
    ChatRequest {
        model: "gpt-5.4".into(),
        messages: vec![Message::user("Hello")],
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_no_middleware_passthrough() {
    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .with_provider("openai", Arc::new(CountingProvider::new("direct response")))
        .build()
        .unwrap();

    let resp = client.chat(&test_request()).await.unwrap();
    assert_eq!(resp.content, "direct response");
}

#[tokio::test]
async fn test_middleware_execution_order() {
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));

    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .with_provider("openai", Arc::new(CountingProvider::new("ok")))
        .with_middleware(Arc::new(MarkerMiddleware {
            name: "outer".into(),
            log: log.clone(),
        }))
        .with_middleware(Arc::new(MarkerMiddleware {
            name: "inner".into(),
            log: log.clone(),
        }))
        .build()
        .unwrap();

    let resp = client.chat(&test_request()).await.unwrap();
    assert_eq!(resp.content, "ok");

    let entries = log.lock().unwrap();
    // Outer runs first on request, last on response (onion model)
    assert_eq!(entries[0], "outer-pre");
    assert_eq!(entries[1], "inner-pre");
    assert_eq!(entries[2], "inner-post");
    assert_eq!(entries[3], "outer-post");
}

#[tokio::test]
async fn test_cache_short_circuits_pipeline() {
    let store = Arc::new(InMemoryStore::new());
    let provider = Arc::new(CountingProvider::new("from provider"));

    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .with_provider("openai", provider.clone())
        .with_middleware(Arc::new(CacheMiddleware::new(
            store,
            Duration::from_secs(300),
        )))
        .build()
        .unwrap();

    let req = test_request();

    // First call — cache miss
    let resp1 = client.chat(&req).await.unwrap();
    assert_eq!(resp1.content, "from provider");
    assert_eq!(provider.count(), 1);

    // Second call — cache hit, provider NOT called again
    let resp2 = client.chat(&req).await.unwrap();
    assert_eq!(resp2.content, "from provider");
    assert_eq!(provider.count(), 1, "provider should not be called on cache hit");
}

#[tokio::test]
async fn test_guardrail_blocks_forbidden_request() {
    let filter = Arc::new(KeywordFilter::new(vec!["forbidden".into()]));
    let guardrail = Arc::new(GuardrailMiddleware::new(vec![filter]));

    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .with_provider("openai", Arc::new(CountingProvider::new("should not reach")))
        .with_middleware(guardrail)
        .build()
        .unwrap();

    // Safe request passes
    let safe = ChatRequest {
        model: "gpt-5.4".into(),
        messages: vec![Message::user("Hello world")],
        ..Default::default()
    };
    assert!(client.chat(&safe).await.is_ok());

    // Forbidden request blocked
    let blocked = ChatRequest {
        model: "gpt-5.4".into(),
        messages: vec![Message::user("Tell me about forbidden topics")],
        ..Default::default()
    };
    let result = client.chat(&blocked).await;
    assert!(
        matches!(result, Err(NexusError::GuardrailBlocked(_))),
        "expected GuardrailBlocked, got: {result:?}"
    );
}

#[tokio::test]
async fn test_guardrail_then_cache_stacked() {
    let store = Arc::new(InMemoryStore::new());
    let filter = Arc::new(KeywordFilter::new(vec!["blocked".into()]));
    let provider = Arc::new(CountingProvider::new("ok"));

    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .with_provider("openai", provider.clone())
        // Guardrail outermost (added first), cache innermost
        .with_middleware(Arc::new(GuardrailMiddleware::new(vec![filter])))
        .with_middleware(Arc::new(CacheMiddleware::new(
            store,
            Duration::from_secs(300),
        )))
        .build()
        .unwrap();

    // Normal request works and gets cached
    let req = test_request();
    assert_eq!(client.chat(&req).await.unwrap().content, "ok");
    assert_eq!(provider.count(), 1);

    // Same request from cache
    assert_eq!(client.chat(&req).await.unwrap().content, "ok");
    assert_eq!(provider.count(), 1);

    // Blocked request fails at guardrail (before cache)
    let blocked = ChatRequest {
        model: "gpt-5.4".into(),
        messages: vec![Message::user("blocked content")],
        ..Default::default()
    };
    assert!(matches!(
        client.chat(&blocked).await,
        Err(NexusError::GuardrailBlocked(_))
    ));
}

#[tokio::test]
async fn test_batch_chat_through_pipeline() {
    let provider = Arc::new(CountingProvider::new("batch ok"));

    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .with_provider("openai", provider.clone())
        .build()
        .unwrap();

    let requests: Vec<ChatRequest> = (0..5).map(|_| test_request()).collect();
    let results = client.chat_batch(&requests, 3).await;

    assert_eq!(results.len(), 5);
    for r in &results {
        assert_eq!(r.as_ref().unwrap().content, "batch ok");
    }
    assert_eq!(provider.count(), 5);
}

#[tokio::test]
async fn test_stream_passes_through_middleware() {
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));

    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .with_provider("openai", Arc::new(CountingProvider::new("ok")))
        .with_middleware(Arc::new(MarkerMiddleware {
            name: "M".into(),
            log: log.clone(),
        }))
        .build()
        .unwrap();

    let stream = client.chat_stream(&test_request()).await;
    assert!(stream.is_ok());

    let entries = log.lock().unwrap();
    assert_eq!(entries[0], "M-stream");
}
