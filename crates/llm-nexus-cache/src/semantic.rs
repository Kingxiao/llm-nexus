//! SemanticCacheMiddleware — similarity-based cache using embeddings.
//!
//! Unlike the hash-based [`CacheMiddleware`], this middleware uses cosine
//! similarity on embedding vectors to match semantically equivalent queries.
//!
//! Requires an [`EmbeddingProvider`] to generate query embeddings at runtime.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::Stream;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::pipeline::context::RequestContext;
use llm_nexus_core::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use llm_nexus_core::traits::embedding::EmbeddingProvider;
use llm_nexus_core::types::request::{ChatRequest, MessageContent, Role};
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};
use llm_nexus_core::types::EmbedRequest;

use crate::vector_index::VectorIndex;

/// Semantic cache middleware that matches requests by embedding similarity.
///
/// On each request:
/// 1. Extract the last user message text
/// 2. Compute its embedding via the configured [`EmbeddingProvider`]
/// 3. Search the vector index for a match above `similarity_threshold`
/// 4. Hit → return cached response; miss → call provider, then store
///
/// Streaming requests bypass the cache (passed through).
pub struct SemanticCacheMiddleware {
    embedder: Arc<dyn EmbeddingProvider>,
    embed_model: String,
    index: Arc<VectorIndex>,
    default_ttl: Duration,
    similarity_threshold: f32,
}

/// Configuration for [`SemanticCacheMiddleware`].
pub struct SemanticCacheConfig {
    /// Embedding provider to use for vectorization.
    pub embedder: Arc<dyn EmbeddingProvider>,
    /// Model ID for embedding requests (e.g. "text-embedding-3-small").
    pub embed_model: String,
    /// Maximum number of cached entries.
    pub capacity: usize,
    /// Time-to-live for cached entries.
    pub ttl: Duration,
    /// Cosine similarity threshold (0.0-1.0). Higher = stricter matching.
    /// Recommended: 0.92-0.96 for production use.
    pub similarity_threshold: f32,
}

/// Marker inserted into RequestContext on semantic cache hit.
pub struct SemanticCacheHit {
    pub similarity: f32,
}

impl SemanticCacheMiddleware {
    pub fn new(config: SemanticCacheConfig) -> Self {
        Self {
            embedder: config.embedder,
            embed_model: config.embed_model,
            index: Arc::new(VectorIndex::new(config.capacity)),
            default_ttl: config.ttl,
            similarity_threshold: config.similarity_threshold,
        }
    }

    /// Extract the last user message as plain text for embedding.
    fn extract_query_text(request: &ChatRequest) -> Option<String> {
        request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| match &m.content {
                MessageContent::Text(t) => Some(t.clone()),
                MessageContent::Parts(parts) => {
                    // Concatenate text parts only
                    let texts: Vec<String> = parts
                        .iter()
                        .filter_map(|p| {
                            if let llm_nexus_core::types::request::ContentPart::Text {
                                text,
                            } = p
                            {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                        .collect();
                    if texts.is_empty() {
                        None
                    } else {
                        Some(texts.join(" "))
                    }
                }
            })
    }

    /// Compute embedding for a query string.
    async fn embed(&self, text: &str) -> NexusResult<Vec<f32>> {
        let response = self
            .embedder
            .embed(&EmbedRequest {
                model: self.embed_model.clone(),
                input: vec![text.to_string()],
                dimensions: None,
            })
            .await?;

        response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| NexusError::ConfigError("embedding response returned no vectors".into()))
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for SemanticCacheMiddleware {
    fn name(&self) -> &str {
        "semantic_cache"
    }

    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        // Extract query text; if none, skip cache
        let query_text = match Self::extract_query_text(request) {
            Some(t) if !t.is_empty() => t,
            _ => return next.run(ctx, request).await,
        };

        // Compute embedding (if embedding fails, fall through to provider)
        let embedding = match self.embed(&query_text).await {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "semantic cache: embedding failed, bypassing cache");
                return next.run(ctx, request).await;
            }
        };

        // Search index
        if let Some(cached) = self.index.search(&embedding, self.similarity_threshold) {
            if let Ok(response) = serde_json::from_slice::<ChatResponse>(&cached) {
                tracing::debug!("semantic cache hit");
                ctx.insert(SemanticCacheHit {
                    similarity: self.similarity_threshold,
                });
                return Ok(response);
            }
        }

        // Cache miss — call provider
        let response = next.run(ctx, request).await?;

        // Store in cache (best-effort)
        if let Ok(serialized) = serde_json::to_vec(&response) {
            self.index
                .insert(embedding, serialized, self.default_ttl);
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

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;
    use llm_nexus_core::types::response::Usage;
    use llm_nexus_core::types::EmbedResponse;

    /// Mock embedding provider that returns a fixed vector based on input text.
    struct MockEmbedder;

    #[async_trait::async_trait]
    impl EmbeddingProvider for MockEmbedder {
        fn provider_id(&self) -> &str {
            "mock"
        }

        async fn embed(&self, request: &EmbedRequest) -> NexusResult<EmbedResponse> {
            let embeddings = request
                .input
                .iter()
                .map(|text| {
                    // Simple deterministic "embedding": hash chars to 4-dim vector
                    let sum: f32 = text.bytes().map(|b| b as f32).sum();
                    let len = text.len() as f32;
                    vec![
                        (sum / 1000.0).sin(),
                        (len / 10.0).cos(),
                        (sum / 500.0).cos(),
                        (len / 5.0).sin(),
                    ]
                })
                .collect();

            Ok(EmbedResponse {
                model: "mock-embed".into(),
                embeddings,
                usage: Usage {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    total_tokens: 0,
                },
            })
        }
    }

    fn make_config() -> SemanticCacheConfig {
        SemanticCacheConfig {
            embedder: Arc::new(MockEmbedder),
            embed_model: "mock".into(),
            capacity: 100,
            ttl: Duration::from_secs(300),
            similarity_threshold: 0.95,
        }
    }

    #[test]
    fn test_extract_query_text_simple() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("What is Rust?")],
            ..Default::default()
        };
        assert_eq!(
            SemanticCacheMiddleware::extract_query_text(&req),
            Some("What is Rust?".into())
        );
    }

    #[test]
    fn test_extract_query_text_uses_last_user_message() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![
                Message::user("First question"),
                Message::assistant("Answer"),
                Message::user("Follow-up question"),
            ],
            ..Default::default()
        };
        assert_eq!(
            SemanticCacheMiddleware::extract_query_text(&req),
            Some("Follow-up question".into())
        );
    }

    #[test]
    fn test_extract_query_text_no_user_message() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![Message::system("You are a helper")],
            ..Default::default()
        };
        assert_eq!(SemanticCacheMiddleware::extract_query_text(&req), None);
    }

    #[tokio::test]
    async fn test_embed_returns_vector() {
        let cache = SemanticCacheMiddleware::new(make_config());
        let vec = cache.embed("hello world").await.unwrap();
        assert_eq!(vec.len(), 4);
    }

    #[test]
    fn test_semantic_cache_config_defaults() {
        let config = make_config();
        assert_eq!(config.capacity, 100);
        assert_eq!(config.ttl, Duration::from_secs(300));
        assert!((config.similarity_threshold - 0.95).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn test_identical_query_produces_same_embedding() {
        let cache = SemanticCacheMiddleware::new(make_config());
        let v1 = cache.embed("What is ownership in Rust?").await.unwrap();
        let v2 = cache.embed("What is ownership in Rust?").await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn test_vector_index_stores_and_retrieves() {
        let cache = SemanticCacheMiddleware::new(make_config());

        let embedding = cache.embed("test query").await.unwrap();
        let response = ChatResponse {
            id: "r1".into(),
            model: "m".into(),
            content: "cached answer".into(),
            finish_reason: None,
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
            tool_calls: None,
        };

        let serialized = serde_json::to_vec(&response).unwrap();
        cache.index.insert(embedding.clone(), serialized, Duration::from_secs(60));

        let found = cache.index.search(&embedding, 0.95);
        assert!(found.is_some());

        let deserialized: ChatResponse =
            serde_json::from_slice(&found.unwrap()).unwrap();
        assert_eq!(deserialized.content, "cached answer");
    }
}
