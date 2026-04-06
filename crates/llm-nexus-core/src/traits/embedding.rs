use crate::error::NexusResult;
use crate::types::{EmbedRequest, EmbedResponse};

/// Trait for providers that support embedding generation.
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync + 'static {
    /// Returns the unique identifier for this provider.
    fn provider_id(&self) -> &str;

    /// Generates embeddings for the given input texts.
    async fn embed(&self, request: &EmbedRequest) -> NexusResult<EmbedResponse>;

    /// Returns the maximum number of texts that can be embedded in a single request.
    fn max_batch_size(&self) -> usize {
        2048
    }
}
