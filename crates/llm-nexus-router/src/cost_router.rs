use async_trait::async_trait;
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::router::{RouteContext, RouteDecision, Router};
use llm_nexus_core::types::model::ModelMetadata;

use crate::scorer::{CostScorer, ScoringFunction};

/// Cost-aware router that selects the cheapest model satisfying constraints.
pub struct CostRouter {
    models: Vec<ModelMetadata>,
    scorer: Box<dyn ScoringFunction>,
}

impl CostRouter {
    pub fn new(models: Vec<ModelMetadata>) -> Self {
        Self {
            models,
            scorer: Box::new(CostScorer),
        }
    }

    pub fn with_scorer(models: Vec<ModelMetadata>, scorer: Box<dyn ScoringFunction>) -> Self {
        Self { models, scorer }
    }

    /// Filters models that satisfy all constraints in the route context.
    fn filter_candidates(&self, context: &RouteContext) -> Vec<&ModelMetadata> {
        self.models
            .iter()
            .filter(|m| {
                // Required capabilities
                for cap in &context.required_capabilities {
                    if !m.capabilities.contains(cap) {
                        return false;
                    }
                }
                // Cost ceiling
                if let Some(max_cost) = context.max_cost_per_1k_tokens {
                    let cost_per_1k = (m.input_price_per_1m + m.output_price_per_1m) / 2000.0;
                    if cost_per_1k > max_cost {
                        return false;
                    }
                }
                // Provider preferences
                if !context.preferred_providers.is_empty()
                    && !context.preferred_providers.contains(&m.provider)
                {
                    return false;
                }
                // Provider exclusions
                if context.excluded_providers.contains(&m.provider) {
                    return false;
                }
                true
            })
            .collect()
    }

    fn to_decision(m: &ModelMetadata) -> RouteDecision {
        RouteDecision {
            provider_id: m.provider.clone(),
            model_id: m.id.clone(),
            estimated_cost_per_1k: Some((m.input_price_per_1m + m.output_price_per_1m) / 2000.0),
            estimated_latency_ms: m.latency_baseline_ms,
        }
    }
}

#[async_trait]
impl Router for CostRouter {
    async fn route(&self, context: &RouteContext) -> NexusResult<RouteDecision> {
        let candidates = self.filter_candidates(context);
        candidates
            .iter()
            .min_by(|a, b| {
                self.scorer
                    .score(a, context)
                    .partial_cmp(&self.scorer.score(b, context))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| Self::to_decision(m))
            .ok_or(NexusError::NoRouteAvailable)
    }

    async fn fallback_chain(&self, context: &RouteContext) -> NexusResult<Vec<RouteDecision>> {
        let candidates = self.filter_candidates(context);
        let mut scored: Vec<_> = candidates
            .iter()
            .map(|m| (*m, self.scorer.score(m, context)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let chain: Vec<RouteDecision> = scored.iter().map(|(m, _)| Self::to_decision(m)).collect();

        if chain.is_empty() {
            Err(NexusError::NoRouteAvailable)
        } else {
            Ok(chain)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::model::{Capability, ModelFeatures};

    fn model(id: &str, provider: &str, input_price: f64, output_price: f64) -> ModelMetadata {
        ModelMetadata {
            id: id.into(),
            provider: provider.into(),
            display_name: id.into(),
            context_window: 128000,
            max_output_tokens: Some(4096),
            input_price_per_1m: input_price,
            output_price_per_1m: output_price,
            capabilities: vec![Capability::Chat, Capability::CodeGeneration],
            features: ModelFeatures::default(),
            latency_baseline_ms: None,
        }
    }

    #[tokio::test]
    async fn test_selects_cheapest() {
        let models = vec![
            model("expensive", "openai", 30.0, 60.0),
            model("cheap", "anthropic", 1.0, 2.0),
            model("mid", "google", 10.0, 20.0),
        ];
        let router = CostRouter::new(models);
        let ctx = RouteContext::default();
        let decision = router.route(&ctx).await.unwrap();
        assert_eq!(decision.model_id, "cheap");
    }

    #[tokio::test]
    async fn test_respects_capability_filter() {
        let mut cheap = model("cheap", "anthropic", 1.0, 2.0);
        // cheap model lacks ImageUnderstanding
        cheap.capabilities = vec![Capability::Chat];

        let mut expensive = model("expensive", "openai", 30.0, 60.0);
        expensive.capabilities.push(Capability::ImageUnderstanding);

        let router = CostRouter::new(vec![cheap, expensive]);
        let ctx = RouteContext {
            required_capabilities: vec![Capability::ImageUnderstanding],
            ..Default::default()
        };
        let decision = router.route(&ctx).await.unwrap();
        assert_eq!(decision.model_id, "expensive");
    }

    #[tokio::test]
    async fn test_no_route_available() {
        let router = CostRouter::new(vec![model("only", "openai", 1.0, 2.0)]);
        let ctx = RouteContext {
            required_capabilities: vec![Capability::Embedding],
            ..Default::default()
        };
        let result = router.route(&ctx).await;
        assert!(matches!(result.unwrap_err(), NexusError::NoRouteAvailable));
    }

    #[tokio::test]
    async fn test_excluded_providers() {
        let models = vec![
            model("cheap", "openai", 1.0, 2.0),
            model("mid", "anthropic", 10.0, 20.0),
        ];
        let router = CostRouter::new(models);
        let ctx = RouteContext {
            excluded_providers: vec!["openai".into()],
            ..Default::default()
        };
        let decision = router.route(&ctx).await.unwrap();
        assert_eq!(decision.model_id, "mid");
        assert_eq!(decision.provider_id, "anthropic");
    }

    #[tokio::test]
    async fn test_fallback_chain_ordered_by_cost() {
        let models = vec![
            model("expensive", "openai", 30.0, 60.0),
            model("cheap", "anthropic", 1.0, 2.0),
            model("mid", "google", 10.0, 20.0),
        ];
        let router = CostRouter::new(models);
        let ctx = RouteContext::default();
        let chain = router.fallback_chain(&ctx).await.unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].model_id, "cheap");
        assert_eq!(chain[1].model_id, "mid");
        assert_eq!(chain[2].model_id, "expensive");
    }
}
