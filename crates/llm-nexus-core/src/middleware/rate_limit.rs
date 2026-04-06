use std::sync::Arc;

use tokio::sync::Mutex;

use crate::error::{NexusError, NexusResult};

pub struct TokenBucketLimiter {
    inner: Arc<Mutex<TokenBucketState>>,
}

struct TokenBucketState {
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: tokio::time::Instant,
}

impl TokenBucketLimiter {
    /// Create a new limiter that allows `max_requests_per_second` requests per second.
    pub fn new(max_requests_per_second: f64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TokenBucketState {
                tokens: max_requests_per_second,
                max_tokens: max_requests_per_second,
                refill_rate: max_requests_per_second,
                last_refill: tokio::time::Instant::now(),
            })),
        }
    }

    /// Wait until a token is available, then consume it.
    pub async fn acquire(&self) -> NexusResult<()> {
        loop {
            {
                let mut state = self.inner.lock().await;
                Self::refill(&mut state);

                if state.tokens >= 1.0 {
                    state.tokens -= 1.0;
                    return Ok(());
                }
            }
            // Wait a short interval before retrying.
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    /// Try to acquire a token immediately. Returns `RateLimited` if none available.
    pub async fn try_acquire(&self) -> NexusResult<()> {
        let mut state = self.inner.lock().await;
        Self::refill(&mut state);

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            Ok(())
        } else {
            let wait_ms = ((1.0 - state.tokens) / state.refill_rate * 1000.0) as u64;
            Err(NexusError::RateLimited {
                retry_after_ms: Some(wait_ms),
            })
        }
    }

    fn refill(state: &mut TokenBucketState) {
        let now = tokio::time::Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.tokens = (state.tokens + elapsed * state.refill_rate).min(state.max_tokens);
        state.last_refill = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_allows_within_limit() {
        tokio::time::pause();
        let limiter = TokenBucketLimiter::new(5.0);

        // Should allow up to 5 requests immediately.
        for _ in 0..5 {
            assert!(limiter.try_acquire().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_try_acquire_blocks_excess() {
        tokio::time::pause();
        let limiter = TokenBucketLimiter::new(2.0);

        assert!(limiter.try_acquire().await.is_ok());
        assert!(limiter.try_acquire().await.is_ok());
        // Third should fail.
        let result = limiter.try_acquire().await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            NexusError::RateLimited { .. }
        ));
    }

    #[tokio::test]
    async fn test_refill_over_time() {
        tokio::time::pause();
        let limiter = TokenBucketLimiter::new(1.0);

        // Consume the single token.
        assert!(limiter.try_acquire().await.is_ok());
        assert!(limiter.try_acquire().await.is_err());

        // Advance time by 1 second — should refill 1 token.
        tokio::time::advance(std::time::Duration::from_secs(1)).await;

        assert!(limiter.try_acquire().await.is_ok());
    }
}
