//! Rate-limit aware routing — cooldown providers that return 429.
//!
//! `CooldownRouter` wraps any [`Router`] and filters out providers
//! currently in cooldown from the fallback chain.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::traits::router::{RouteContext, RouteDecision, Router};

/// Tracks which providers are in cooldown after rate limiting.
///
/// Thread-safe — designed to be shared across the NexusClient and middleware.
pub struct ProviderHealthState {
    cooldowns: Mutex<HashMap<String, Instant>>,
    default_cooldown: Duration,
}

impl ProviderHealthState {
    pub fn new(default_cooldown: Duration) -> Self {
        Self {
            cooldowns: Mutex::new(HashMap::new()),
            default_cooldown,
        }
    }

    /// Mark a provider as rate-limited. It will be excluded from routing
    /// until the cooldown expires.
    pub fn mark_rate_limited(&self, provider_id: &str) {
        let mut map = self.cooldowns.lock().unwrap_or_else(|e| e.into_inner());
        let until = Instant::now() + self.default_cooldown;
        map.insert(provider_id.to_string(), until);
        tracing::info!(
            provider = %provider_id,
            cooldown_secs = self.default_cooldown.as_secs(),
            "provider entering cooldown"
        );
    }

    /// Mark a provider as rate-limited with a specific duration
    /// (e.g. from a `Retry-After` header).
    pub fn mark_rate_limited_for(&self, provider_id: &str, duration: Duration) {
        let mut map = self.cooldowns.lock().unwrap_or_else(|e| e.into_inner());
        let until = Instant::now() + duration;
        map.insert(provider_id.to_string(), until);
    }

    /// Check if a provider is currently in cooldown.
    pub fn is_in_cooldown(&self, provider_id: &str) -> bool {
        let mut map = self.cooldowns.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&until) = map.get(provider_id) {
            if Instant::now() < until {
                return true;
            }
            // Cooldown expired — remove
            map.remove(provider_id);
        }
        false
    }
}

/// Router decorator that filters out rate-limited providers.
pub struct CooldownRouter {
    inner: Arc<dyn Router>,
    health: Arc<ProviderHealthState>,
}

impl CooldownRouter {
    pub fn new(inner: Arc<dyn Router>, health: Arc<ProviderHealthState>) -> Self {
        Self { inner, health }
    }
}

#[async_trait::async_trait]
impl Router for CooldownRouter {
    async fn route(&self, context: &RouteContext) -> NexusResult<RouteDecision> {
        // Get the full chain and return the first non-cooldown provider
        let chain = self.fallback_chain(context).await?;
        chain
            .into_iter()
            .next()
            .ok_or(llm_nexus_core::error::NexusError::NoRouteAvailable)
    }

    async fn fallback_chain(&self, context: &RouteContext) -> NexusResult<Vec<RouteDecision>> {
        let chain = self.inner.fallback_chain(context).await?;
        let filtered: Vec<_> = chain
            .into_iter()
            .filter(|d| !self.health.is_in_cooldown(&d.provider_id))
            .collect();

        if filtered.is_empty() {
            tracing::warn!("all providers in cooldown, returning full chain as fallback");
            // Fall back to the original chain — better to try a rate-limited
            // provider than return no route at all.
            return self.inner.fallback_chain(context).await;
        }

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::traits::router::TaskType;

    struct MockRouter {
        decisions: Vec<RouteDecision>,
    }

    #[async_trait::async_trait]
    impl Router for MockRouter {
        async fn route(&self, _ctx: &RouteContext) -> NexusResult<RouteDecision> {
            Ok(self.decisions[0].clone())
        }
        async fn fallback_chain(&self, _ctx: &RouteContext) -> NexusResult<Vec<RouteDecision>> {
            Ok(self.decisions.clone())
        }
    }

    fn test_context() -> RouteContext {
        RouteContext {
            task_type: Some(TaskType::Chat),
            ..Default::default()
        }
    }

    fn decision(provider: &str, model: &str) -> RouteDecision {
        RouteDecision {
            provider_id: provider.into(),
            model_id: model.into(),
            estimated_cost_per_1k: Some(1.0),
            estimated_latency_ms: None,
        }
    }

    #[tokio::test]
    async fn test_no_cooldown_passthrough() {
        let inner = Arc::new(MockRouter {
            decisions: vec![decision("openai", "gpt-5"), decision("anthropic", "claude")],
        });
        let health = Arc::new(ProviderHealthState::new(Duration::from_secs(60)));
        let router = CooldownRouter::new(inner, health);

        let chain = router.fallback_chain(&test_context()).await.unwrap();
        assert_eq!(chain.len(), 2);
    }

    #[tokio::test]
    async fn test_cooldown_filters_provider() {
        let inner = Arc::new(MockRouter {
            decisions: vec![decision("openai", "gpt-5"), decision("anthropic", "claude")],
        });
        let health = Arc::new(ProviderHealthState::new(Duration::from_secs(60)));
        health.mark_rate_limited("openai");

        let router = CooldownRouter::new(inner, health);
        let chain = router.fallback_chain(&test_context()).await.unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider_id, "anthropic");
    }

    #[tokio::test]
    async fn test_all_cooldown_falls_back_to_full_chain() {
        let inner = Arc::new(MockRouter {
            decisions: vec![decision("openai", "gpt-5")],
        });
        let health = Arc::new(ProviderHealthState::new(Duration::from_secs(60)));
        health.mark_rate_limited("openai");

        let router = CooldownRouter::new(inner, health);
        let chain = router.fallback_chain(&test_context()).await.unwrap();
        // Should return full chain as fallback
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provider_id, "openai");
    }

    #[tokio::test]
    async fn test_cooldown_expires() {
        let health = ProviderHealthState::new(Duration::from_millis(1));
        health.mark_rate_limited("openai");

        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!health.is_in_cooldown("openai"));
    }

    #[tokio::test]
    async fn test_route_skips_cooldown() {
        let inner = Arc::new(MockRouter {
            decisions: vec![decision("openai", "gpt-5"), decision("anthropic", "claude")],
        });
        let health = Arc::new(ProviderHealthState::new(Duration::from_secs(60)));
        health.mark_rate_limited("openai");

        let router = CooldownRouter::new(inner, health);
        let best = router.route(&test_context()).await.unwrap();
        assert_eq!(best.provider_id, "anthropic");
    }
}
