use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub provider: String,
    pub display_name: String,
    pub context_window: u32,
    pub max_output_tokens: Option<u32>,
    pub input_price_per_1m: f64,
    pub output_price_per_1m: f64,
    pub capabilities: Vec<Capability>,
    #[serde(default)]
    pub features: ModelFeatures,
    pub latency_baseline_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelFeatures {
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub tool_use: bool,
    #[serde(default)]
    pub json_mode: bool,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub system_prompt: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Capability {
    Chat,
    CodeGeneration,
    Reasoning,
    ImageUnderstanding,
    LongContext,
    Embedding,
}

#[derive(Debug, Clone, Default)]
pub struct ModelFilter {
    pub capabilities: Vec<Capability>,
    pub max_input_price_per_1m: Option<f64>,
    pub max_output_price_per_1m: Option<f64>,
    pub min_context_window: Option<u32>,
    pub providers: Vec<String>,
    pub required_features: Option<ModelFeatures>,
}

#[derive(Debug, Clone, Default)]
pub struct ModelOverride {
    pub input_price_per_1m: Option<f64>,
    pub output_price_per_1m: Option<f64>,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub latency_baseline_ms: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_metadata_deserialize() {
        let json = r#"{
            "id": "gpt-4o",
            "provider": "openai",
            "display_name": "GPT-4o",
            "context_window": 128000,
            "max_output_tokens": 16384,
            "input_price_per_1m": 2.5,
            "output_price_per_1m": 10.0,
            "capabilities": ["Chat", "CodeGeneration", "ImageUnderstanding"],
            "latency_baseline_ms": 800
        }"#;
        let meta: ModelMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.id, "gpt-4o");
        assert_eq!(meta.context_window, 128000);
        assert_eq!(meta.capabilities.len(), 3);
        assert!(!meta.features.vision); // default false
    }

    #[test]
    fn test_capability_serialization() {
        let cap = Capability::CodeGeneration;
        let json = serde_json::to_string(&cap).unwrap();
        assert_eq!(json, r#""CodeGeneration""#);

        let deserialized: Capability = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, Capability::CodeGeneration);
    }

    #[test]
    fn test_model_filter_default() {
        let filter = ModelFilter::default();
        assert!(filter.capabilities.is_empty());
        assert!(filter.max_input_price_per_1m.is_none());
        assert!(filter.providers.is_empty());
    }

    #[test]
    fn test_model_features_default() {
        let features = ModelFeatures::default();
        assert!(!features.vision);
        assert!(!features.tool_use);
        assert!(!features.json_mode);
        assert!(!features.streaming);
        assert!(!features.system_prompt);
    }
}
