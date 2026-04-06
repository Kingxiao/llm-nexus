use std::pin::Pin;

use futures::Stream;

use crate::error::NexusResult;
use crate::types::{ChatRequest, ChatResponse, StreamChunk};

/// Trait for providers that support chat completions.
#[async_trait::async_trait]
pub trait ChatProvider: Send + Sync + 'static {
    /// Returns the unique identifier for this provider (e.g. "openai", "anthropic").
    fn provider_id(&self) -> &str;

    /// Sends a chat completion request and returns the full response.
    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse>;

    /// Sends a chat completion request and returns a stream of chunks.
    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>>;

    /// Lists available model IDs from this provider.
    async fn list_models(&self) -> NexusResult<Vec<String>>;

    /// Checks if the provider is reachable and authenticated.
    async fn health_check(&self) -> NexusResult<bool> {
        Ok(true)
    }

    /// Batch chat: send multiple requests concurrently.
    ///
    /// Default: runs `chat()` concurrently with `buffer_unordered`.
    /// Providers with native batch APIs can override for better efficiency.
    async fn chat_batch(
        &self,
        requests: &[ChatRequest],
        concurrency: usize,
    ) -> Vec<NexusResult<ChatResponse>> {
        use futures::stream::StreamExt;

        // Clone requests to avoid lifetime issues with buffer_unordered
        let owned: Vec<ChatRequest> = requests.to_vec();
        let futs: Vec<_> = owned.iter().map(|req| self.chat(req)).collect();

        futures::stream::iter(futs)
            .buffer_unordered(concurrency)
            .collect()
            .await
    }
}
