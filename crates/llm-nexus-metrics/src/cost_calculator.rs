use llm_nexus_core::types::model::ModelMetadata;
use llm_nexus_core::types::response::Usage;

/// Calculate estimated cost in USD based on token usage and model pricing.
pub fn calculate_cost(usage: &Usage, model: &ModelMetadata) -> f64 {
    let input_cost = (usage.prompt_tokens as f64 / 1_000_000.0) * model.input_price_per_1m;
    let output_cost = (usage.completion_tokens as f64 / 1_000_000.0) * model.output_price_per_1m;
    input_cost + output_cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::model::{Capability, ModelMetadata};

    fn make_model(input_price: f64, output_price: f64) -> ModelMetadata {
        ModelMetadata {
            id: "test-model".to_string(),
            provider: "test".to_string(),
            display_name: "Test Model".to_string(),
            context_window: 128_000,
            max_output_tokens: Some(16_384),
            input_price_per_1m: input_price,
            output_price_per_1m: output_price,
            capabilities: vec![Capability::Chat],
            features: Default::default(),
            latency_baseline_ms: None,
        }
    }

    fn make_usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    #[test]
    fn test_cost_calculation() {
        let model = make_model(2.50, 10.00);
        let usage = make_usage(1_000, 500);
        let cost = calculate_cost(&usage, &model);
        // input: 1000/1M * 2.50 = 0.0025
        // output: 500/1M * 10.00 = 0.005
        let expected = 0.0075;
        assert!(
            (cost - expected).abs() < 1e-10,
            "expected {expected}, got {cost}"
        );
    }

    #[test]
    fn test_zero_tokens() {
        let model = make_model(2.50, 10.00);
        let usage = make_usage(0, 0);
        let cost = calculate_cost(&usage, &model);
        assert!((cost - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_volume() {
        let model = make_model(2.50, 10.00);
        let usage = make_usage(1_000_000, 500_000);
        let cost = calculate_cost(&usage, &model);
        // input: 1M/1M * 2.50 = 2.50
        // output: 500K/1M * 10.00 = 5.00
        let expected = 7.50;
        assert!(
            (cost - expected).abs() < 1e-10,
            "expected {expected}, got {cost}"
        );
    }
}
