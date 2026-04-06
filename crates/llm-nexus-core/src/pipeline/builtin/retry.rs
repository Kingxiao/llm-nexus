//! Retry middleware — wraps provider calls with exponential backoff.
//!
//! Delegates to the existing [`crate::middleware::retry::with_retry`] logic,
//! now integrated into the pipeline instead of requiring manual wrapping.

use crate::error::NexusResult;
use crate::middleware::retry::RetryConfig;
use crate::pipeline::context::RequestContext;
use crate::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use crate::types::{ChatRequest, ChatResponse, StreamChunk};

use std::pin::Pin;

use futures::Stream;

/// Middleware that retries failed requests with exponential backoff.
///
/// Only retries on transient errors (429, 5xx, timeout, network).
/// Non-retryable errors (4xx, auth) are returned immediately.
pub struct RetryMiddleware {
    config: RetryConfig,
}

impl RetryMiddleware {
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self {
            config: RetryConfig::default(),
        }
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for RetryMiddleware {
    fn name(&self) -> &str {
        "builtin::retry"
    }

    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        // For retry, we need to be able to call next multiple times.
        // But Next is consumed on use. So we reconstruct it each time
        // using the same middleware slice and dispatcher reference.
        //
        // On the first (and hopefully only) attempt, we use the provided next.
        // The retry logic from middleware::retry handles backoff.
        //
        // Since middleware pipeline Next is single-use by design (each layer
        // peels one middleware off), retry must be the innermost middleware
        // to work correctly — it retries only the provider dispatch, not the
        // entire middleware chain.
        //
        // For simplicity in v1, retry wraps the single next.run() call.
        let config = &self.config;
        let mut delay = config.initial_delay_ms;
        let mut attempts = 0;

        // We need to reconstruct next for each retry. Store the parts.
        let middlewares = next.middlewares;
        let dispatcher = next.dispatcher;

        loop {
            let next_inner = Next {
                middlewares,
                dispatcher,
            };
            match next_inner.run(ctx, request).await {
                Ok(resp) => return Ok(resp),
                Err(err) => {
                    attempts += 1;
                    if attempts > config.max_retries
                        || !crate::middleware::retry::is_retryable(&err)
                    {
                        return Err(err);
                    }

                    if let crate::error::NexusError::RateLimited {
                        retry_after_ms: Some(ms),
                    } = &err
                    {
                        tokio::time::sleep(std::time::Duration::from_millis(*ms)).await;
                    } else {
                        let jitter = delay / 4;
                        let actual_delay =
                            delay + crate::middleware::retry::rand_simple(jitter);
                        tokio::time::sleep(std::time::Duration::from_millis(actual_delay)).await;
                    }

                    delay = ((delay as f64) * config.backoff_factor) as u64;
                    delay = delay.min(config.max_delay_ms);
                }
            }
        }
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        // Streaming: no retry (can't replay a stream). Just pass through.
        next.run(ctx, request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::test_helpers::mocks::*;

    #[tokio::test]
    async fn test_retry_succeeds_first_try() {
        let mw = RetryMiddleware::with_defaults();
        let dispatcher = MockDispatcher::ok("first try");
        let resp = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert_eq!(resp.unwrap().content, "first try");
        assert_eq!(dispatcher.count(), 1);
    }

    #[tokio::test]
    async fn test_retry_recovers_after_failures() {
        tokio::time::pause();
        let mw = RetryMiddleware::new(RetryConfig {
            max_retries: 3,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            backoff_factor: 2.0,
        });
        let dispatcher = FailThenSucceedDispatcher::new(2); // fail 2x, succeed on 3rd
        let resp = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert_eq!(resp.unwrap().content, "recovered");
        assert_eq!(dispatcher.call_count.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_retry_gives_up_after_max() {
        tokio::time::pause();
        let mw = RetryMiddleware::new(RetryConfig {
            max_retries: 2,
            initial_delay_ms: 10,
            max_delay_ms: 100,
            backoff_factor: 2.0,
        });
        let dispatcher = FailThenSucceedDispatcher::new(10); // always fail
        let result = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_retry_no_retry_on_auth_error() {
        tokio::time::pause();
        let mw = RetryMiddleware::with_defaults();
        let dispatcher = MockDispatcher::with_error(
            crate::error::NexusError::AuthError("bad key".into()),
        );
        let result = run_middleware(&mw, &dispatcher, &test_request()).await;
        assert!(result.is_err());
        // Auth errors are not retryable — should fail on first attempt only
        assert_eq!(dispatcher.count(), 1, "auth error should not be retried");
    }
}
