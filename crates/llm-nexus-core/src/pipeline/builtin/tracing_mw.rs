//! TracingMiddleware — exports LLM spans via SpanExporter.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;

use crate::error::NexusResult;
use crate::pipeline::context::RequestContext;
use crate::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use crate::traits::tracing::{LlmSpan, SpanExporter, SpanValue};
use crate::types::request::ChatRequest;
use crate::types::response::{ChatResponse, StreamChunk};

/// Middleware that exports a span for each LLM request.
///
/// Uses the [`SpanExporter`] trait — bring your own OTel/Datadog/etc implementation.
pub struct TracingMiddleware {
    exporter: Arc<dyn SpanExporter>,
}

impl TracingMiddleware {
    pub fn new(exporter: Arc<dyn SpanExporter>) -> Self {
        Self { exporter }
    }

    fn build_span(
        &self,
        ctx: &RequestContext,
        request: &ChatRequest,
        success: bool,
        error: Option<&str>,
    ) -> LlmSpan {
        let mut attrs = HashMap::new();

        // Use provider_id (set by dispatcher) as the system identifier.
        // Falls back to model_meta.provider if provider_id is not set.
        let system = ctx
            .provider_id
            .clone()
            .or_else(|| ctx.model_meta.as_ref().map(|m| m.provider.clone()));
        if let Some(sys) = system {
            attrs.insert("gen_ai.system".into(), SpanValue::String(sys));
        }
        attrs.insert(
            "gen_ai.request.model".into(),
            SpanValue::String(request.model.clone()),
        );
        if let Some(t) = request.temperature {
            attrs.insert("gen_ai.request.temperature".into(), SpanValue::Float(t as f64));
        }
        if let Some(m) = request.max_tokens {
            attrs.insert("gen_ai.request.max_tokens".into(), SpanValue::Int(m as i64));
        }
        if let Some(ref err) = error {
            attrs.insert("error.message".into(), SpanValue::String(err.to_string()));
        }

        LlmSpan {
            name: format!("chat {}", request.model),
            duration_ms: ctx.elapsed_ms(),
            success,
            attributes: attrs,
        }
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for TracingMiddleware {
    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        let result = next.run(ctx, request).await;

        let mut span = match &result {
            Ok(resp) => {
                let mut s = self.build_span(ctx, request, true, None);
                s.attributes.insert(
                    "gen_ai.usage.input_tokens".into(),
                    SpanValue::Int(resp.usage.prompt_tokens as i64),
                );
                s.attributes.insert(
                    "gen_ai.usage.output_tokens".into(),
                    SpanValue::Int(resp.usage.completion_tokens as i64),
                );
                if let Some(ref fr) = resp.finish_reason {
                    s.attributes.insert(
                        "gen_ai.response.finish_reason".into(),
                        SpanValue::String(format!("{fr:?}")),
                    );
                }
                s
            }
            Err(e) => self.build_span(ctx, request, false, Some(&e.to_string())),
        };
        span.duration_ms = ctx.elapsed_ms();

        self.exporter.export(span);
        result
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let result = next.run(ctx, request).await;

        if result.is_err() {
            let span = self.build_span(
                ctx,
                request,
                false,
                result.as_ref().err().map(|e| e.to_string()).as_deref(),
            );
            self.exporter.export(span);
        }
        // For successful streams, we can't record usage until the stream completes.
        // A future enhancement could wrap the stream to capture final usage.

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::test_helpers::mocks::*;
    use crate::traits::tracing::LlmSpan;
    use std::sync::Mutex;

    struct RecordingExporter {
        spans: Mutex<Vec<LlmSpan>>,
    }

    impl RecordingExporter {
        fn new() -> Self {
            Self {
                spans: Mutex::new(Vec::new()),
            }
        }
        fn spans(&self) -> Vec<LlmSpan> {
            self.spans.lock().unwrap().clone()
        }
    }

    impl SpanExporter for RecordingExporter {
        fn export(&self, span: LlmSpan) {
            self.spans.lock().unwrap().push(span);
        }
    }

    #[tokio::test]
    async fn test_tracing_exports_span_on_success() {
        let exporter = Arc::new(RecordingExporter::new());
        let mw = TracingMiddleware::new(exporter.clone());
        let dispatcher = MockDispatcher::ok("traced");
        let resp = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert!(resp.is_ok());

        let spans = exporter.spans();
        assert_eq!(spans.len(), 1);
        assert!(spans[0].success);
        assert!(spans[0].name.contains("test-model"));
        assert!(spans[0].attributes.contains_key("gen_ai.request.model"));
    }

    #[tokio::test]
    async fn test_tracing_exports_span_on_failure() {
        let exporter = Arc::new(RecordingExporter::new());
        let mw = TracingMiddleware::new(exporter.clone());
        let dispatcher = MockDispatcher::with_error(
            crate::error::NexusError::ProviderError {
                provider: "mock".into(),
                message: "fail".into(),
                status_code: Some(500),
            },
        );
        let _ = run_middleware(&mw, &dispatcher, &test_request()).await;

        let spans = exporter.spans();
        assert_eq!(spans.len(), 1);
        assert!(!spans[0].success);
        assert!(spans[0].attributes.contains_key("error.message"));
    }
}
