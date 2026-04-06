use std::future::Future;

use crate::error::{NexusError, NexusResult};

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 500,
            max_delay_ms: 30_000,
            backoff_factor: 2.0,
        }
    }
}

/// Determine whether an error is retryable.
pub fn is_retryable(err: &NexusError) -> bool {
    matches!(
        err,
        NexusError::RateLimited { .. }
            | NexusError::Timeout(_)
            | NexusError::HttpError(_)
            | NexusError::StreamError(_)
    ) || matches!(
        err,
        NexusError::ProviderError { status_code: Some(code), .. } if *code >= 500
    )
}

/// Retry wrapper with exponential backoff and jitter.
pub async fn with_retry<F, Fut, T>(config: &RetryConfig, mut operation: F) -> NexusResult<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = NexusResult<T>>,
{
    let mut delay = config.initial_delay_ms;
    let mut attempts = 0;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                attempts += 1;
                if attempts > config.max_retries || !is_retryable(&err) {
                    return Err(err);
                }

                if let NexusError::RateLimited {
                    retry_after_ms: Some(ms),
                } = &err
                {
                    tokio::time::sleep(std::time::Duration::from_millis(*ms)).await;
                } else {
                    let jitter = delay / 4;
                    let actual_delay = delay + rand_simple(jitter);
                    tokio::time::sleep(std::time::Duration::from_millis(actual_delay)).await;
                }

                delay = ((delay as f64) * config.backoff_factor) as u64;
                delay = delay.min(config.max_delay_ms);
            }
        }
    }
}

/// Simple pseudo-random to avoid pulling in the `rand` crate.
pub fn rand_simple(max: u64) -> u64 {
    if max == 0 {
        return 0;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    nanos % max
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[tokio::test]
    async fn test_retry_on_rate_limit() {
        tokio::time::pause();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = Arc::clone(&attempts);

        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 1000,
            backoff_factor: 2.0,
        };

        let result = with_retry(&config, || {
            let attempts = Arc::clone(&attempts_clone);
            async move {
                let n = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 3 {
                    Err(NexusError::RateLimited {
                        retry_after_ms: Some(50),
                    })
                } else {
                    Ok("success")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_no_retry_on_auth_error() {
        tokio::time::pause();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = Arc::clone(&attempts);

        let config = RetryConfig::default();

        let result: NexusResult<()> = with_retry(&config, || {
            let attempts = Arc::clone(&attempts_clone);
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(NexusError::AuthError("invalid key".into()))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_on_server_error() {
        tokio::time::pause();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = Arc::clone(&attempts);

        let config = RetryConfig {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 1000,
            backoff_factor: 2.0,
        };

        let result = with_retry(&config, || {
            let attempts = Arc::clone(&attempts_clone);
            async move {
                let n = attempts.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 2 {
                    Err(NexusError::ProviderError {
                        provider: "test".into(),
                        message: "internal server error".into(),
                        status_code: Some(500),
                    })
                } else {
                    Ok("recovered")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert!(attempts.load(Ordering::SeqCst) >= 2);
    }

    #[tokio::test]
    async fn test_no_retry_on_400() {
        tokio::time::pause();
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = Arc::clone(&attempts);

        let config = RetryConfig::default();

        let result: NexusResult<()> = with_retry(&config, || {
            let attempts = Arc::clone(&attempts_clone);
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(NexusError::ProviderError {
                    provider: "test".into(),
                    message: "bad request".into(),
                    status_code: Some(400),
                })
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
