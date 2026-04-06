use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::router::RouteDecision;

/// Executes a request across a fallback chain, returning the first success.
///
/// Tries each decision in order. On failure, logs a warning and moves to the next.
/// If all fail, returns `AllProvidersFailed` with the collected errors.
pub async fn execute_with_fallback<F, Fut, T>(
    chain: &[RouteDecision],
    mut execute: F,
) -> NexusResult<T>
where
    F: FnMut(&RouteDecision) -> Fut,
    Fut: std::future::Future<Output = NexusResult<T>>,
{
    let mut errors = Vec::new();
    for decision in chain {
        match execute(decision).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                tracing::warn!(
                    provider = %decision.provider_id,
                    model = %decision.model_id,
                    error = %e,
                    "fallback: provider failed, trying next"
                );
                errors.push(e);
            }
        }
    }
    Err(NexusError::AllProvidersFailed(errors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn decision(provider: &str, model: &str) -> RouteDecision {
        RouteDecision {
            provider_id: provider.into(),
            model_id: model.into(),
            estimated_cost_per_1k: None,
            estimated_latency_ms: None,
        }
    }

    #[tokio::test]
    async fn test_first_succeeds() {
        let chain = vec![
            decision("openai", "gpt-4o"),
            decision("anthropic", "claude"),
        ];
        let call_count = AtomicUsize::new(0);

        let result = execute_with_fallback(&chain, |_d| {
            call_count.fetch_add(1, Ordering::SeqCst);
            async { Ok::<&str, NexusError>("ok") }
        })
        .await;

        assert_eq!(result.unwrap(), "ok");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_fallback_to_second() {
        let chain = vec![
            decision("openai", "gpt-4o"),
            decision("anthropic", "claude"),
        ];
        let call_count = AtomicUsize::new(0);

        let result = execute_with_fallback(&chain, |d| {
            let idx = call_count.fetch_add(1, Ordering::SeqCst);
            let provider = d.provider_id.clone();
            async move {
                if idx == 0 {
                    Err(NexusError::Timeout(5000))
                } else {
                    Ok(provider)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), "anthropic");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_all_fail() {
        let chain = vec![
            decision("openai", "gpt-4o"),
            decision("anthropic", "claude"),
        ];

        let result: NexusResult<String> =
            execute_with_fallback(&chain, |_d| async { Err(NexusError::Timeout(1000)) }).await;

        match result.unwrap_err() {
            NexusError::AllProvidersFailed(errors) => {
                assert_eq!(errors.len(), 2);
            }
            other => panic!("expected AllProvidersFailed, got: {other}"),
        }
    }
}
