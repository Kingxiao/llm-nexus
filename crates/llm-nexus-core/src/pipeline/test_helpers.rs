//! Test helpers for pipeline middleware testing.
//!
//! Provides mock dispatchers and providers for unit testing middleware
//! in isolation without requiring the full NexusClient stack.

#[cfg(test)]
pub(crate) mod mocks {
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    use futures::Stream;

    use crate::error::{NexusError, NexusResult};
    use crate::pipeline::context::RequestContext;
    use crate::pipeline::middleware::Dispatcher;
    use crate::types::request::ChatRequest;
    use crate::types::response::{ChatResponse, StreamChunk, Usage};

    /// Mock dispatcher that returns a fixed response.
    /// Error factory function for mocks (NexusError is not Clone).
    type ErrorFactory = Box<dyn Fn() -> NexusError + Send + Sync>;

    pub struct MockDispatcher {
        pub response: String,
        pub call_count: AtomicU32,
        pub delay: Option<Duration>,
        pub error_factory: Option<ErrorFactory>,
    }

    impl MockDispatcher {
        pub fn ok(response: &str) -> Self {
            Self {
                response: response.to_string(),
                call_count: AtomicU32::new(0),
                delay: None,
                error_factory: None,
            }
        }

        pub fn with_delay(response: &str, delay: Duration) -> Self {
            Self {
                response: response.to_string(),
                call_count: AtomicU32::new(0),
                delay: Some(delay),
                error_factory: None,
            }
        }

        pub fn with_error(error: NexusError) -> Self {
            // Convert the error into a factory that recreates it
            let msg = error.to_string();
            let factory: ErrorFactory = match error {
                NexusError::AuthError(_) => Box::new(move || NexusError::AuthError(msg.clone())),
                NexusError::ProviderError {
                    provider,
                    status_code,
                    ..
                } => {
                    let p = provider.clone();
                    let m = msg.clone();
                    Box::new(move || NexusError::ProviderError {
                        provider: p.clone(),
                        message: m.clone(),
                        status_code,
                    })
                }
                _ => Box::new(move || NexusError::HttpError(msg.clone())),
            };
            Self {
                response: String::new(),
                call_count: AtomicU32::new(0),
                delay: None,
                error_factory: Some(factory),
            }
        }

        pub fn count(&self) -> u32 {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait::async_trait]
    impl Dispatcher for MockDispatcher {
        async fn dispatch(
            &self,
            ctx: &mut RequestContext,
            _request: &ChatRequest,
        ) -> NexusResult<ChatResponse> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            ctx.provider_id = Some("mock".into());

            if let Some(ref factory) = self.error_factory {
                return Err(factory());
            }

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

        async fn dispatch_stream(
            &self,
            _ctx: &mut RequestContext,
            _request: &ChatRequest,
        ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    /// Helper: run a single middleware with a mock dispatcher.
    pub async fn run_middleware(
        middleware: &dyn crate::pipeline::middleware::ChatMiddleware,
        dispatcher: &dyn Dispatcher,
        request: &ChatRequest,
    ) -> NexusResult<ChatResponse> {
        let mut ctx = RequestContext::new("test-req".into());
        let middlewares: Vec<Arc<dyn crate::pipeline::middleware::ChatMiddleware>> = vec![];
        let next = crate::pipeline::middleware::Next {
            middlewares: &middlewares,
            dispatcher,
        };
        middleware.process(&mut ctx, request, next).await
    }

    /// Dispatcher that fails N times then succeeds (for retry tests).
    pub struct FailThenSucceedDispatcher {
        pub fail_count: u32,
        pub call_count: AtomicU32,
        pub error: NexusError,
    }

    impl FailThenSucceedDispatcher {
        pub fn new(fail_count: u32) -> Self {
            Self {
                fail_count,
                call_count: AtomicU32::new(0),
                error: NexusError::ProviderError {
                    provider: "mock".into(),
                    message: "transient".into(),
                    status_code: Some(500),
                },
            }
        }
    }

    #[async_trait::async_trait]
    impl Dispatcher for FailThenSucceedDispatcher {
        async fn dispatch(
            &self,
            _ctx: &mut RequestContext,
            _request: &ChatRequest,
        ) -> NexusResult<ChatResponse> {
            let n = self.call_count.fetch_add(1, Ordering::Relaxed);
            if n < self.fail_count {
                return Err(NexusError::ProviderError {
                    provider: "mock".into(),
                    message: "transient".into(),
                    status_code: Some(500),
                });
            }
            Ok(ChatResponse {
                id: "ok".into(),
                model: "m".into(),
                content: "recovered".into(),
                finish_reason: None,
                usage: Usage::default(),
                tool_calls: None,
            })
        }

        async fn dispatch_stream(
            &self,
            _ctx: &mut RequestContext,
            _request: &ChatRequest,
        ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    pub fn test_request() -> ChatRequest {
        ChatRequest {
            model: "test-model".into(),
            messages: vec![crate::types::request::Message::user("hello")],
            ..Default::default()
        }
    }
}
