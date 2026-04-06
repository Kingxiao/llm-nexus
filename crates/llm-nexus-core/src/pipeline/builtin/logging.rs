//! LoggingMiddleware — records per-request log entries.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;

use crate::error::NexusResult;
use crate::pipeline::context::RequestContext;
use crate::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use crate::traits::logging::{LogBackend, RequestLogEntry};
use crate::types::request::ChatRequest;
use crate::types::response::{ChatResponse, StreamChunk, Usage};

/// Middleware that logs every request/response to a [`LogBackend`].
pub struct LoggingMiddleware {
    backend: Arc<dyn LogBackend>,
    /// Whether to include the full request/response body in logs.
    /// Set to false for privacy-sensitive deployments.
    log_bodies: bool,
}

impl LoggingMiddleware {
    pub fn new(backend: Arc<dyn LogBackend>, log_bodies: bool) -> Self {
        Self {
            backend,
            log_bodies,
        }
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for LoggingMiddleware {
    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        let request_body = if self.log_bodies {
            serde_json::to_value(request).ok()
        } else {
            None
        };

        let result = next.run(ctx, request).await;

        let (success, response_body, usage, cost, error) = match &result {
            Ok(resp) => {
                let body = if self.log_bodies {
                    serde_json::to_value(resp).ok()
                } else {
                    None
                };
                (true, body, resp.usage.clone(), 0.0, None)
            }
            Err(e) => (false, None, Usage::default(), 0.0, Some(e.to_string())),
        };

        let entry = RequestLogEntry {
            request_id: ctx.request_id.clone(),
            timestamp: chrono::Utc::now(),
            provider_id: ctx.provider_id.clone().unwrap_or_default(),
            model_id: request.model.clone(),
            request_body,
            response_body,
            latency_ms: ctx.elapsed_ms(),
            usage,
            cost_usd: cost,
            success,
            error,
            metadata: std::collections::HashMap::new(),
        };

        if let Err(e) = self.backend.log_request(entry).await {
            tracing::warn!(error = %e, "failed to log request");
        }

        result
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        // Log the stream initiation (no response body for streaming)
        let result = next.run(ctx, request).await;

        if result.is_err() {
            let entry = RequestLogEntry {
                request_id: ctx.request_id.clone(),
                timestamp: chrono::Utc::now(),
                provider_id: ctx.provider_id.clone().unwrap_or_default(),
                model_id: request.model.clone(),
                request_body: None,
                response_body: None,
                latency_ms: ctx.elapsed_ms(),
                usage: Usage::default(),
                cost_usd: 0.0,
                success: false,
                error: result.as_ref().err().map(|e| e.to_string()),
                metadata: std::collections::HashMap::new(),
            };
            let _ = self.backend.log_request(entry).await;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::test_helpers::mocks::*;
    use crate::store::InMemoryLogBackend;
    use crate::traits::logging::LogFilter;

    #[tokio::test]
    async fn test_logging_records_successful_request() {
        let log_backend = Arc::new(InMemoryLogBackend::new(100));
        let mw = LoggingMiddleware::new(log_backend.clone(), false);
        let dispatcher = MockDispatcher::ok("logged");
        let resp = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert!(resp.is_ok());

        let logs = log_backend.query_logs(&LogFilter::default()).await.unwrap();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].success);
        assert_eq!(logs[0].model_id, "test-model");
    }

    #[tokio::test]
    async fn test_logging_records_body_when_enabled() {
        let log_backend = Arc::new(InMemoryLogBackend::new(100));
        let mw = LoggingMiddleware::new(log_backend.clone(), true);
        let dispatcher = MockDispatcher::ok("with body");
        let _ = run_middleware(&mw, &dispatcher, &test_request()).await;

        let logs = log_backend.query_logs(&LogFilter::default()).await.unwrap();
        assert!(logs[0].request_body.is_some());
        assert!(logs[0].response_body.is_some());
    }

    #[tokio::test]
    async fn test_logging_records_failed_request() {
        let log_backend = Arc::new(InMemoryLogBackend::new(100));
        let mw = LoggingMiddleware::new(log_backend.clone(), false);
        let dispatcher = MockDispatcher::with_error(crate::error::NexusError::ProviderError {
            provider: "mock".into(),
            message: "fail".into(),
            status_code: Some(500),
        });
        let _ = run_middleware(&mw, &dispatcher, &test_request()).await;

        let logs = log_backend.query_logs(&LogFilter::default()).await.unwrap();
        assert_eq!(logs.len(), 1);
        assert!(!logs[0].success);
        assert!(logs[0].error.is_some());
    }
}
