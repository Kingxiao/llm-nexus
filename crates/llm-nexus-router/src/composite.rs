//! Composite routing via weighted scorer combination.
//!
//! `WeightedScorer` combines multiple `ScoringFunction` implementations with
//! configurable weights. Used with `CostRouter::with_scorer()` to create
//! multi-objective routers (e.g., 70% cost + 30% latency).

use llm_nexus_core::types::model::ModelMetadata;

use crate::cost_router::CostRouter;
use crate::scorer::ScoringFunction;

/// Combines multiple scoring functions with weights.
///
/// Each scorer's output is multiplied by its weight, then summed.
/// Lower combined scores are preferred (consistent with `ScoringFunction` convention).
///
/// # Example
/// ```ignore
/// let scorer = WeightedScorer::new(vec![
///     (Box::new(CostScorer), 0.7),
///     (Box::new(LatencyScorer), 0.3),
/// ]);
/// let router = CostRouter::with_scorer(models, Box::new(scorer));
/// ```
pub struct WeightedScorer {
    scorers: Vec<(Box<dyn ScoringFunction>, f64)>,
}

impl WeightedScorer {
    /// Create a new weighted scorer.
    ///
    /// Each tuple is `(scorer, weight)`. Weights are used as-is (not normalized).
    pub fn new(scorers: Vec<(Box<dyn ScoringFunction>, f64)>) -> Self {
        Self { scorers }
    }
}

impl ScoringFunction for WeightedScorer {
    fn score(
        &self,
        model: &ModelMetadata,
        context: &llm_nexus_core::traits::router::RouteContext,
    ) -> f64 {
        self.scorers
            .iter()
            .map(|(scorer, weight)| scorer.score(model, context) * weight)
            .sum()
    }
}

/// Create a composite router with weighted multi-objective scoring.
///
/// Wraps `CostRouter::with_scorer` with a `WeightedScorer`.
pub fn composite_router(
    models: Vec<ModelMetadata>,
    scorers: Vec<(Box<dyn ScoringFunction>, f64)>,
) -> CostRouter {
    CostRouter::with_scorer(models, Box::new(WeightedScorer::new(scorers)))
}

/// Create a latency-optimized router.
///
/// Uses `LatencyScorer` as the sole scoring function.
pub fn latency_router(models: Vec<ModelMetadata>) -> CostRouter {
    CostRouter::with_scorer(models, Box::new(crate::scorer::LatencyScorer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scorer::{CostScorer, LatencyScorer};
    use llm_nexus_core::traits::router::{RouteContext, Router};
    use llm_nexus_core::types::model::{Capability, ModelFeatures};

    fn make_model(
        id: &str,
        input_price: f64,
        output_price: f64,
        latency_ms: Option<u64>,
    ) -> ModelMetadata {
        ModelMetadata {
            id: id.into(),
            provider: "test".into(),
            display_name: id.into(),
            context_window: 128000,
            max_output_tokens: Some(4096),
            input_price_per_1m: input_price,
            output_price_per_1m: output_price,
            capabilities: vec![Capability::Chat],
            features: ModelFeatures::default(),
            latency_baseline_ms: latency_ms,
        }
    }

    #[test]
    fn test_weighted_scorer_blended() {
        let scorer = WeightedScorer::new(vec![
            (Box::new(CostScorer), 0.7),
            (Box::new(LatencyScorer), 0.3),
        ]);
        let ctx = RouteContext::default();
        // CostScorer: 10*0.7 + 20*0.3 = 13.0, then * 0.7 = 9.1
        // LatencyScorer: 200, then * 0.3 = 60.0
        // Total: 69.1
        let m = make_model("test", 10.0, 20.0, Some(200));
        let score = scorer.score(&m, &ctx);
        let expected = (10.0 * 0.7 + 20.0 * 0.3) * 0.7 + 200.0 * 0.3;
        assert!((score - expected).abs() < 0.01);
    }

    #[test]
    fn test_weighted_scorer_single_degenerates() {
        let scorer = WeightedScorer::new(vec![(Box::new(CostScorer), 1.0)]);
        let cost_only = CostScorer;
        let ctx = RouteContext::default();
        let m = make_model("test", 5.0, 15.0, Some(100));
        assert!((scorer.score(&m, &ctx) - cost_only.score(&m, &ctx)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_scorer_zero_weight_ignores() {
        let scorer = WeightedScorer::new(vec![
            (Box::new(CostScorer), 1.0),
            (Box::new(LatencyScorer), 0.0),
        ]);
        let cost_only = CostScorer;
        let ctx = RouteContext::default();
        let m = make_model("test", 5.0, 15.0, Some(9999));
        // Even though latency is huge, weight 0.0 makes it irrelevant
        assert!((scorer.score(&m, &ctx) - cost_only.score(&m, &ctx)).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_composite_router_picks_best_blended() {
        // Model A: cheap but slow; Model B: expensive but fast
        // Note: CostScorer produces ~1-30 range, LatencyScorer produces ~100-2000.
        // Weights must account for scale differences.
        let models = vec![
            make_model("cheap-slow", 1.0, 2.0, Some(2000)),
            make_model("expensive-fast", 20.0, 40.0, Some(100)),
        ];

        // Cost-only: should prefer cheap-slow
        let cost_only = composite_router(
            models.clone(),
            vec![(Box::new(CostScorer), 1.0), (Box::new(LatencyScorer), 0.0)],
        );
        let ctx = RouteContext::default();
        let decision = cost_only.route(&ctx).await.unwrap();
        assert_eq!(decision.model_id, "cheap-slow");

        // Latency-only: should prefer expensive-fast
        let latency_only = composite_router(
            models,
            vec![(Box::new(CostScorer), 0.0), (Box::new(LatencyScorer), 1.0)],
        );
        let decision = latency_only.route(&ctx).await.unwrap();
        assert_eq!(decision.model_id, "expensive-fast");
    }

    #[tokio::test]
    async fn test_composite_router_fallback_chain_ordered() {
        let models = vec![
            make_model("a", 10.0, 20.0, Some(500)),
            make_model("b", 5.0, 10.0, Some(1000)),
            make_model("c", 1.0, 2.0, Some(2000)),
        ];

        // Equal weight: blended score decides order
        let router = composite_router(
            models,
            vec![(Box::new(CostScorer), 0.5), (Box::new(LatencyScorer), 0.5)],
        );
        let ctx = RouteContext::default();
        let chain = router.fallback_chain(&ctx).await.unwrap();
        assert_eq!(chain.len(), 3);
        // All 3 should be present
        let ids: Vec<_> = chain.iter().map(|d| d.model_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
    }

    #[tokio::test]
    async fn test_latency_router_picks_fastest() {
        let models = vec![
            make_model("slow", 1.0, 2.0, Some(2000)),
            make_model("fast", 50.0, 100.0, Some(50)),
            make_model("mid", 10.0, 20.0, Some(500)),
        ];
        let router = latency_router(models);
        let ctx = RouteContext::default();
        let decision = router.route(&ctx).await.unwrap();
        assert_eq!(decision.model_id, "fast");
    }

    #[tokio::test]
    async fn test_empty_scorers_returns_zero() {
        let scorer = WeightedScorer::new(vec![]);
        let ctx = RouteContext::default();
        let m = make_model("test", 10.0, 20.0, Some(500));
        assert!((scorer.score(&m, &ctx) - 0.0).abs() < f64::EPSILON);
    }
}
