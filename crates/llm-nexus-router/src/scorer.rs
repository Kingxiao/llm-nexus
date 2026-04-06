use llm_nexus_core::traits::router::RouteContext;
use llm_nexus_core::types::model::ModelMetadata;

/// Scoring function for ranking models. Lower scores are preferred.
pub trait ScoringFunction: Send + Sync {
    fn score(&self, model: &ModelMetadata, context: &RouteContext) -> f64;
}

/// Ranks models by weighted cost (input 70%, output 30%).
pub struct CostScorer;

impl ScoringFunction for CostScorer {
    fn score(&self, model: &ModelMetadata, _context: &RouteContext) -> f64 {
        model.input_price_per_1m * 0.7 + model.output_price_per_1m * 0.3
    }
}

/// Ranks models by baseline latency.
pub struct LatencyScorer;

impl ScoringFunction for LatencyScorer {
    fn score(&self, model: &ModelMetadata, _context: &RouteContext) -> f64 {
        model.latency_baseline_ms.unwrap_or(500) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::model::{Capability, ModelFeatures};

    fn make_model(input_price: f64, output_price: f64, latency_ms: Option<u64>) -> ModelMetadata {
        ModelMetadata {
            id: "test".into(),
            provider: "test".into(),
            display_name: "test".into(),
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
    fn test_cost_scorer_weighted() {
        let scorer = CostScorer;
        let ctx = RouteContext::default();
        let m = make_model(10.0, 30.0, None);
        let score = scorer.score(&m, &ctx);
        // 10.0 * 0.7 + 30.0 * 0.3 = 7.0 + 9.0 = 16.0
        assert!((score - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_latency_scorer_with_value() {
        let scorer = LatencyScorer;
        let ctx = RouteContext::default();
        let m = make_model(1.0, 2.0, Some(200));
        assert!((scorer.score(&m, &ctx) - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_latency_scorer_default() {
        let scorer = LatencyScorer;
        let ctx = RouteContext::default();
        let m = make_model(1.0, 2.0, None);
        assert!((scorer.score(&m, &ctx) - 500.0).abs() < f64::EPSILON);
    }
}
