//! Timeout middleware — enforces a maximum duration per request.

use std::pin::Pin;
use std::time::Duration;

use futures::Stream;

use crate::error::{NexusError, NexusResult};
use crate::pipeline::context::RequestContext;
use crate::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use crate::types::{ChatRequest, ChatResponse, StreamChunk};

/// Middleware that enforces a timeout on provider calls.
pub struct TimeoutMiddleware {
    duration: Duration,
}

impl TimeoutMiddleware {
    pub fn new(duration: Duration) -> Self {
        Self { duration }
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for TimeoutMiddleware {
    fn name(&self) -> &str {
        "builtin::timeout"
    }

    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        tokio::time::timeout(self.duration, next.run(ctx, request))
            .await
            .map_err(|_| NexusError::Timeout(self.duration.as_millis() as u64))?
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        // Timeout applies to establishing the stream, not to individual chunks.
        tokio::time::timeout(self.duration, next.run(ctx, request))
            .await
            .map_err(|_| NexusError::Timeout(self.duration.as_millis() as u64))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::test_helpers::mocks::*;

    #[tokio::test]
    async fn test_timeout_passes_when_fast() {
        let mw = TimeoutMiddleware::new(Duration::from_secs(5));
        let dispatcher = MockDispatcher::ok("fast");
        let resp = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert_eq!(resp.unwrap().content, "fast");
    }

    #[tokio::test]
    async fn test_timeout_fires_when_slow() {
        tokio::time::pause();
        let mw = TimeoutMiddleware::new(Duration::from_millis(50));
        let dispatcher = MockDispatcher::with_delay("slow", Duration::from_secs(10));
        let result = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert!(matches!(result, Err(NexusError::Timeout(50))));
    }
}
