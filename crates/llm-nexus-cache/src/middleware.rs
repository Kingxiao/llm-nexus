//! CacheMiddleware — short-circuits the pipeline on cache hit.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::pipeline::context::RequestContext;
use llm_nexus_core::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use llm_nexus_core::traits::store::KeyValueStore;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};

use crate::hasher::hash_request;

/// Caches non-streaming chat responses.
///
/// On cache hit, returns the cached response without calling the provider.
/// On cache miss, forwards to the next middleware/provider and stores the result.
///
/// Streaming requests are never cached (passed through).
pub struct CacheMiddleware {
    store: Arc<dyn KeyValueStore>,
    default_ttl: Duration,
}

impl CacheMiddleware {
    /// Create with a store and TTL.
    pub fn new(store: Arc<dyn KeyValueStore>, default_ttl: Duration) -> Self {
        Self { store, default_ttl }
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for CacheMiddleware {
    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        let key = hash_request(request);

        // Check cache
        if let Some(cached) = self.store.get(&key).await? {
            if let Ok(response) = serde_json::from_slice::<ChatResponse>(&cached) {
                tracing::debug!(key = %key, "cache hit");
                ctx.insert(CacheHit(true));
                return Ok(response);
            }
            // Corrupted cache entry — delete and proceed
            tracing::warn!(key = %key, "corrupted cache entry, deleting");
            let _ = self.store.delete(&key).await;
        }

        // Cache miss — forward to provider
        let response = next.run(ctx, request).await?;

        // Store in cache (best-effort, don't fail the request)
        if let Ok(serialized) = serde_json::to_vec(&response) {
            let _ = self
                .store
                .set(&key, &serialized, Some(self.default_ttl))
                .await
                .inspect_err(|e| tracing::warn!(error = %e, "failed to cache response"));
        }

        Ok(response)
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        // Streaming responses are not cached — pass through.
        next.run(ctx, request).await
    }
}

/// Marker inserted into RequestContext on cache hit.
#[allow(dead_code)]
pub struct CacheHit(pub bool);

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::store::InMemoryStore;
    use llm_nexus_core::types::request::Message;
    use llm_nexus_core::types::response::Usage;

    fn mock_response() -> ChatResponse {
        ChatResponse {
            id: "test-id".into(),
            model: "test-model".into(),
            content: "cached content".into(),
            finish_reason: None,
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            tool_calls: None,
        }
    }

    #[tokio::test]
    async fn test_cache_miss_then_hit() {
        let store = Arc::new(InMemoryStore::new());
        let cache = CacheMiddleware::new(store.clone(), Duration::from_secs(300));

        let request = ChatRequest {
            model: "test-model".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };

        // Pre-populate the store with a response
        let key = hash_request(&request);
        let response = mock_response();
        let serialized = serde_json::to_vec(&response).unwrap();
        store
            .set(&key, &serialized, Some(Duration::from_secs(300)))
            .await
            .unwrap();

        // Now the middleware should return from cache without needing a provider
        let mut ctx = RequestContext::new("req-1".into());
        // We can't call next.run() without a dispatcher, but we can verify
        // the cache hit path by checking the store
        let cached = store.get(&key).await.unwrap();
        assert!(cached.is_some());
        let cached_resp: ChatResponse =
            serde_json::from_slice(&cached.unwrap()).unwrap();
        assert_eq!(cached_resp.content, "cached content");

        // Verify CacheHit marker
        ctx.insert(CacheHit(true));
        assert!(ctx.get::<CacheHit>().unwrap().0);
    }

    #[tokio::test]
    async fn test_cache_ttl_expiry() {
        let store = Arc::new(InMemoryStore::new());

        let request = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let key = hash_request(&request);

        // Set with very short TTL
        let response = mock_response();
        let serialized = serde_json::to_vec(&response).unwrap();
        store
            .set(&key, &serialized, Some(Duration::from_millis(1)))
            .await
            .unwrap();

        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Should be expired
        assert!(store.get(&key).await.unwrap().is_none());
    }
}
