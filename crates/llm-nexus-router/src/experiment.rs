//! A/B experiment routing — split traffic across model variants.
//!
//! Uses consistent hashing on request_id to ensure the same user/request
//! always hits the same variant within an experiment.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::traits::router::{RouteContext, RouteDecision, Router};

/// A model variant in an experiment.
#[derive(Debug, Clone)]
pub struct Variant {
    pub provider_id: String,
    pub model_id: String,
    /// Relative weight (e.g. 70 and 30 for a 70/30 split).
    pub weight: u32,
}

/// An experiment definition with weighted variants.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub name: String,
    pub variants: Vec<Variant>,
}

impl Experiment {
    /// Select a variant based on consistent hashing of a key (e.g. request_id).
    pub fn select_variant(&self, hash_key: &str) -> Option<&Variant> {
        if self.variants.is_empty() {
            return None;
        }

        let total_weight: u32 = self.variants.iter().map(|v| v.weight).sum();
        if total_weight == 0 {
            return self.variants.first();
        }

        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        hash_key.hash(&mut hasher);
        let hash = hasher.finish();
        let bucket = (hash % total_weight as u64) as u32;

        let mut cumulative = 0;
        for variant in &self.variants {
            cumulative += variant.weight;
            if bucket < cumulative {
                return Some(variant);
            }
        }

        self.variants.last()
    }
}

/// Router that assigns requests to experiment variants.
///
/// If no experiment matches the context, delegates to the inner router.
/// The experiment is selected by matching `RouteContext.task_type` or
/// by always applying the first experiment (simple mode).
pub struct ExperimentRouter {
    inner: Arc<dyn Router>,
    experiment: Experiment,
}

impl ExperimentRouter {
    /// Create with a single experiment and a fallback router.
    pub fn new(inner: Arc<dyn Router>, experiment: Experiment) -> Self {
        Self { inner, experiment }
    }
}

#[async_trait::async_trait]
impl Router for ExperimentRouter {
    async fn route(&self, context: &RouteContext) -> NexusResult<RouteDecision> {
        let hash_key = context.experiment_key.clone().unwrap_or_else(|| {
            // Fallback: use nanos (not stable across calls)
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .to_string()
        });
        if let Some(variant) = self.experiment.select_variant(&hash_key) {
            return Ok(RouteDecision {
                provider_id: variant.provider_id.clone(),
                model_id: variant.model_id.clone(),
                estimated_cost_per_1k: None,
                estimated_latency_ms: None,
            });
        }
        self.inner.route(context).await
    }

    async fn fallback_chain(
        &self,
        context: &RouteContext,
    ) -> NexusResult<Vec<RouteDecision>> {
        // For fallback chain, put the experiment variant first, then inner chain
        let hash_key = context.experiment_key.clone().unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .to_string()
        });
        let mut chain = Vec::new();

        if let Some(variant) = self.experiment.select_variant(&hash_key) {
            chain.push(RouteDecision {
                provider_id: variant.provider_id.clone(),
                model_id: variant.model_id.clone(),
                estimated_cost_per_1k: None,
                estimated_latency_ms: None,
            });
        }

        // Add inner chain as fallback
        chain.extend(self.inner.fallback_chain(context).await?);
        Ok(chain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::traits::router::TaskType;
    use std::collections::HashMap;

    struct MockRouter;

    #[async_trait::async_trait]
    impl Router for MockRouter {
        async fn route(&self, _ctx: &RouteContext) -> NexusResult<RouteDecision> {
            Ok(RouteDecision {
                provider_id: "fallback".into(),
                model_id: "fallback-model".into(),
                estimated_cost_per_1k: None,
                estimated_latency_ms: None,
            })
        }
        async fn fallback_chain(&self, _ctx: &RouteContext) -> NexusResult<Vec<RouteDecision>> {
            Ok(vec![RouteDecision {
                provider_id: "fallback".into(),
                model_id: "fallback-model".into(),
                estimated_cost_per_1k: None,
                estimated_latency_ms: None,
            }])
        }
    }

    fn test_context() -> RouteContext {
        RouteContext {
            task_type: Some(TaskType::Chat),
            ..Default::default()
        }
    }

    #[test]
    fn test_consistent_hashing_deterministic() {
        let exp = Experiment {
            name: "test-exp".into(),
            variants: vec![
                Variant { provider_id: "a".into(), model_id: "m-a".into(), weight: 50 },
                Variant { provider_id: "b".into(), model_id: "m-b".into(), weight: 50 },
            ],
        };

        // Same key always selects the same variant
        let v1 = exp.select_variant("req-123").unwrap();
        let v2 = exp.select_variant("req-123").unwrap();
        assert_eq!(v1.provider_id, v2.provider_id);
    }

    #[test]
    fn test_weight_distribution_approximate() {
        let exp = Experiment {
            name: "split-test".into(),
            variants: vec![
                Variant { provider_id: "a".into(), model_id: "m-a".into(), weight: 70 },
                Variant { provider_id: "b".into(), model_id: "m-b".into(), weight: 30 },
            ],
        };

        let mut counts: HashMap<String, u32> = HashMap::new();
        for i in 0..1000 {
            let key = format!("req-{i}");
            let v = exp.select_variant(&key).unwrap();
            *counts.entry(v.provider_id.clone()).or_default() += 1;
        }

        let a_count = *counts.get("a").unwrap_or(&0);
        // Should be roughly 700 ± 50
        assert!(a_count > 600 && a_count < 800, "a_count={a_count}, expected ~700");
    }

    #[test]
    fn test_single_variant_always_selected() {
        let exp = Experiment {
            name: "single".into(),
            variants: vec![
                Variant { provider_id: "only".into(), model_id: "m".into(), weight: 100 },
            ],
        };

        let v = exp.select_variant("any-key").unwrap();
        assert_eq!(v.provider_id, "only");
    }

    #[tokio::test]
    async fn test_experiment_router_uses_variant() {
        let exp = Experiment {
            name: "test".into(),
            variants: vec![
                Variant { provider_id: "exp-provider".into(), model_id: "exp-model".into(), weight: 100 },
            ],
        };
        let router = ExperimentRouter::new(Arc::new(MockRouter), exp);

        let decision = router.route(&test_context()).await.unwrap();
        assert_eq!(decision.provider_id, "exp-provider");
    }

    #[tokio::test]
    async fn test_fallback_chain_includes_inner() {
        let exp = Experiment {
            name: "test".into(),
            variants: vec![
                Variant { provider_id: "exp".into(), model_id: "m".into(), weight: 100 },
            ],
        };
        let router = ExperimentRouter::new(Arc::new(MockRouter), exp);

        let chain = router.fallback_chain(&test_context()).await.unwrap();
        assert_eq!(chain.len(), 2); // experiment variant + fallback
        assert_eq!(chain[0].provider_id, "exp");
        assert_eq!(chain[1].provider_id, "fallback");
    }
}
