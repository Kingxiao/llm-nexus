//! Model filtering logic for registry queries.

use llm_nexus_core::types::model::{ModelFeatures, ModelFilter, ModelMetadata};

/// Returns `true` if the model matches all criteria in the filter.
///
/// An empty/default filter matches every model.
pub fn matches_filter(model: &ModelMetadata, filter: &ModelFilter) -> bool {
    // Capability check: model must have ALL requested capabilities
    if !filter.capabilities.is_empty()
        && !filter
            .capabilities
            .iter()
            .all(|cap| model.capabilities.contains(cap))
    {
        return false;
    }

    // Price ceiling checks
    if let Some(max_in) = filter.max_input_price_per_1m
        && model.input_price_per_1m > max_in
    {
        return false;
    }
    if let Some(max_out) = filter.max_output_price_per_1m
        && model.output_price_per_1m > max_out
    {
        return false;
    }

    // Minimum context window
    if let Some(min_ctx) = filter.min_context_window
        && model.context_window < min_ctx
    {
        return false;
    }

    // Provider allow-list
    if !filter.providers.is_empty() && !filter.providers.contains(&model.provider) {
        return false;
    }

    // Required features
    if let Some(ref required) = filter.required_features
        && !features_satisfied(&model.features, required)
    {
        return false;
    }

    true
}

/// Checks that `actual` features satisfy all `true` fields in `required`.
fn features_satisfied(actual: &ModelFeatures, required: &ModelFeatures) -> bool {
    if required.vision && !actual.vision {
        return false;
    }
    if required.tool_use && !actual.tool_use {
        return false;
    }
    if required.json_mode && !actual.json_mode {
        return false;
    }
    if required.streaming && !actual.streaming {
        return false;
    }
    if required.system_prompt && !actual.system_prompt {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::model::*;

    fn test_model() -> ModelMetadata {
        ModelMetadata {
            id: "test-model".into(),
            provider: "openai".into(),
            display_name: "Test Model".into(),
            context_window: 128000,
            max_output_tokens: Some(16384),
            input_price_per_1m: 2.50,
            output_price_per_1m: 10.00,
            capabilities: vec![Capability::Chat, Capability::CodeGeneration],
            features: ModelFeatures {
                vision: true,
                tool_use: true,
                json_mode: true,
                streaming: true,
                system_prompt: true,
            },
            latency_baseline_ms: None,
        }
    }

    #[test]
    fn test_empty_filter_matches_all() {
        let model = test_model();
        assert!(matches_filter(&model, &ModelFilter::default()));
    }

    #[test]
    fn test_capability_filter_match() {
        let model = test_model();
        let filter = ModelFilter {
            capabilities: vec![Capability::Chat],
            ..Default::default()
        };
        assert!(matches_filter(&model, &filter));
    }

    #[test]
    fn test_capability_filter_miss() {
        let model = test_model();
        let filter = ModelFilter {
            capabilities: vec![Capability::Embedding],
            ..Default::default()
        };
        assert!(!matches_filter(&model, &filter));
    }

    #[test]
    fn test_price_filter_under() {
        let model = test_model(); // input_price = 2.50
        let filter = ModelFilter {
            max_input_price_per_1m: Some(3.0),
            ..Default::default()
        };
        assert!(matches_filter(&model, &filter));
    }

    #[test]
    fn test_price_filter_over() {
        let model = test_model(); // input_price = 2.50
        let filter = ModelFilter {
            max_input_price_per_1m: Some(1.0),
            ..Default::default()
        };
        assert!(!matches_filter(&model, &filter));
    }

    #[test]
    fn test_output_price_filter() {
        let model = test_model(); // output_price = 10.00
        let filter = ModelFilter {
            max_output_price_per_1m: Some(5.0),
            ..Default::default()
        };
        assert!(!matches_filter(&model, &filter));
    }

    #[test]
    fn test_context_window_filter() {
        let model = test_model(); // context_window = 128000
        let filter = ModelFilter {
            min_context_window: Some(200000),
            ..Default::default()
        };
        assert!(!matches_filter(&model, &filter));
    }

    #[test]
    fn test_provider_filter_match() {
        let model = test_model(); // provider = "openai"
        let filter = ModelFilter {
            providers: vec!["openai".into()],
            ..Default::default()
        };
        assert!(matches_filter(&model, &filter));
    }

    #[test]
    fn test_provider_filter_miss() {
        let model = test_model();
        let filter = ModelFilter {
            providers: vec!["anthropic".into()],
            ..Default::default()
        };
        assert!(!matches_filter(&model, &filter));
    }

    #[test]
    fn test_required_features_match() {
        let model = test_model();
        let filter = ModelFilter {
            required_features: Some(ModelFeatures {
                vision: true,
                tool_use: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(matches_filter(&model, &filter));
    }

    #[test]
    fn test_required_features_miss() {
        let mut model = test_model();
        model.features.vision = false;
        let filter = ModelFilter {
            required_features: Some(ModelFeatures {
                vision: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(!matches_filter(&model, &filter));
    }

    #[test]
    fn test_combined_filters() {
        let model = test_model();
        let filter = ModelFilter {
            capabilities: vec![Capability::Chat],
            max_input_price_per_1m: Some(5.0),
            providers: vec!["openai".into()],
            ..Default::default()
        };
        assert!(matches_filter(&model, &filter));
    }
}
