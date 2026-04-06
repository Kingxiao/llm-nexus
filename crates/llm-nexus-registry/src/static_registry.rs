//! Static in-memory model registry backed by TOML configuration files.

use std::collections::HashMap;
use std::path::Path;
use std::sync::RwLock;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::loader;
use llm_nexus_core::traits::registry::ModelRegistry;
use llm_nexus_core::types::model::{ModelFilter, ModelMetadata, ModelOverride};

use crate::filter::matches_filter;

/// A static model registry that loads model metadata from TOML config
/// and supports runtime overrides (e.g. custom pricing).
pub struct StaticRegistry {
    models: Vec<ModelMetadata>,
    overrides: RwLock<HashMap<String, ModelOverride>>,
}

impl StaticRegistry {
    /// Creates a registry from an already-loaded list of models.
    pub fn from_models(models: Vec<ModelMetadata>) -> Self {
        Self {
            models,
            overrides: RwLock::new(HashMap::new()),
        }
    }

    /// Loads providers and models from a config directory containing
    /// `providers.toml` and `models.toml`.
    pub fn from_config_dir(config_dir: &Path) -> NexusResult<Self> {
        let (_providers, models) = loader::load_config(config_dir)?;
        Ok(Self::from_models(models))
    }

    /// Returns a model with any active overrides applied.
    fn resolve_model(&self, model: &ModelMetadata) -> ModelMetadata {
        let overrides = self.overrides.read().expect("RwLock poisoned");
        match overrides.get(&model.id) {
            Some(ov) => apply_override(model.clone(), ov),
            None => model.clone(),
        }
    }
}

/// Applies an override to a cloned model, replacing only the fields that
/// are `Some` in the override.
fn apply_override(mut model: ModelMetadata, ov: &ModelOverride) -> ModelMetadata {
    if let Some(v) = ov.input_price_per_1m {
        model.input_price_per_1m = v;
    }
    if let Some(v) = ov.output_price_per_1m {
        model.output_price_per_1m = v;
    }
    if let Some(v) = ov.context_window {
        model.context_window = v;
    }
    if let Some(v) = ov.max_output_tokens {
        model.max_output_tokens = Some(v);
    }
    if let Some(v) = ov.latency_baseline_ms {
        model.latency_baseline_ms = Some(v);
    }
    model
}

#[async_trait::async_trait]
impl ModelRegistry for StaticRegistry {
    async fn get_model(&self, model_id: &str) -> NexusResult<Option<ModelMetadata>> {
        Ok(self
            .models
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| self.resolve_model(m)))
    }

    async fn list_models(&self, filter: &ModelFilter) -> NexusResult<Vec<ModelMetadata>> {
        Ok(self
            .models
            .iter()
            .map(|m| self.resolve_model(m))
            .filter(|m| matches_filter(m, filter))
            .collect())
    }

    async fn refresh(&self) -> NexusResult<()> {
        // Phase 1: no-op. Phase 2 will add remote catalog sync.
        tracing::debug!("StaticRegistry::refresh is a no-op in Phase 1");
        Ok(())
    }

    fn apply_override(&self, model_id: &str, overrides: ModelOverride) -> NexusResult<()> {
        // Verify the model exists before storing the override
        if !self.models.iter().any(|m| m.id == model_id) {
            return Err(NexusError::ModelNotFound(model_id.to_string()));
        }
        let mut map = self.overrides.write().expect("RwLock poisoned");
        map.insert(model_id.to_string(), overrides);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::model::*;

    fn test_models() -> Vec<ModelMetadata> {
        vec![
            ModelMetadata {
                id: "gpt-5.4".into(),
                provider: "openai".into(),
                display_name: "GPT-5.4".into(),
                context_window: 1000000,
                max_output_tokens: Some(100000),
                input_price_per_1m: 2.50,
                output_price_per_1m: 15.00,
                capabilities: vec![
                    Capability::Chat,
                    Capability::CodeGeneration,
                    Capability::Reasoning,
                    Capability::ImageUnderstanding,
                    Capability::LongContext,
                ],
                features: ModelFeatures {
                    vision: true,
                    tool_use: true,
                    json_mode: true,
                    streaming: true,
                    system_prompt: true,
                },
                latency_baseline_ms: None,
            },
            ModelMetadata {
                id: "deepseek-chat".into(),
                provider: "deepseek".into(),
                display_name: "DeepSeek Chat".into(),
                context_window: 65536,
                max_output_tokens: Some(8192),
                input_price_per_1m: 0.28,
                output_price_per_1m: 0.42,
                capabilities: vec![
                    Capability::Chat,
                    Capability::CodeGeneration,
                    Capability::Reasoning,
                ],
                features: ModelFeatures {
                    vision: false,
                    tool_use: true,
                    json_mode: true,
                    streaming: true,
                    system_prompt: true,
                },
                latency_baseline_ms: None,
            },
            ModelMetadata {
                id: "claude-sonnet-4-6".into(),
                provider: "anthropic".into(),
                display_name: "Claude Sonnet 4.6".into(),
                context_window: 1000000,
                max_output_tokens: Some(64000),
                input_price_per_1m: 3.00,
                output_price_per_1m: 15.00,
                capabilities: vec![
                    Capability::Chat,
                    Capability::CodeGeneration,
                    Capability::Reasoning,
                    Capability::ImageUnderstanding,
                    Capability::LongContext,
                ],
                features: ModelFeatures {
                    vision: true,
                    tool_use: true,
                    json_mode: true,
                    streaming: true,
                    system_prompt: true,
                },
                latency_baseline_ms: None,
            },
        ]
    }

    #[tokio::test]
    async fn test_get_model() {
        let registry = StaticRegistry::from_models(test_models());
        let model = registry.get_model("gpt-5.4").await.unwrap().unwrap();
        assert_eq!(model.provider, "openai");
        assert_eq!(model.context_window, 1000000);
    }

    #[tokio::test]
    async fn test_get_model_not_found() {
        let registry = StaticRegistry::from_models(test_models());
        let model = registry.get_model("nonexistent").await.unwrap();
        assert!(model.is_none());
    }

    #[tokio::test]
    async fn test_list_all_models() {
        let registry = StaticRegistry::from_models(test_models());
        let models = registry.list_models(&ModelFilter::default()).await.unwrap();
        assert_eq!(models.len(), 3);
    }

    #[tokio::test]
    async fn test_filter_by_capability() {
        let registry = StaticRegistry::from_models(test_models());
        let filter = ModelFilter {
            capabilities: vec![Capability::ImageUnderstanding],
            ..Default::default()
        };
        let models = registry.list_models(&filter).await.unwrap();
        assert_eq!(models.len(), 2); // gpt-5.4 and claude-sonnet
        assert!(
            models
                .iter()
                .all(|m| m.capabilities.contains(&Capability::ImageUnderstanding))
        );
    }

    #[tokio::test]
    async fn test_filter_by_price() {
        let registry = StaticRegistry::from_models(test_models());
        let filter = ModelFilter {
            max_input_price_per_1m: Some(1.0),
            ..Default::default()
        };
        let models = registry.list_models(&filter).await.unwrap();
        assert_eq!(models.len(), 1); // only deepseek-chat at 0.28
        assert!(models.iter().all(|m| m.input_price_per_1m <= 1.0));
    }

    #[tokio::test]
    async fn test_filter_by_provider() {
        let registry = StaticRegistry::from_models(test_models());
        let filter = ModelFilter {
            providers: vec!["anthropic".into()],
            ..Default::default()
        };
        let models = registry.list_models(&filter).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider, "anthropic");
    }

    #[tokio::test]
    async fn test_user_override() {
        let registry = StaticRegistry::from_models(test_models());
        ModelRegistry::apply_override(
            &registry,
            "gpt-5.4",
            ModelOverride {
                input_price_per_1m: Some(2.00),
                ..Default::default()
            },
        )
        .unwrap();
        let model = registry.get_model("gpt-5.4").await.unwrap().unwrap();
        assert!((model.input_price_per_1m - 2.00).abs() < f64::EPSILON);
        // Other fields unchanged
        assert!((model.output_price_per_1m - 15.00).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_override_nonexistent_model() {
        let registry = StaticRegistry::from_models(test_models());
        let result =
            ModelRegistry::apply_override(&registry, "nonexistent", ModelOverride::default());
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_refresh_noop() {
        let registry = StaticRegistry::from_models(test_models());
        assert!(registry.refresh().await.is_ok());
    }

    #[tokio::test]
    async fn test_from_config_dir() {
        let config_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config");
        let registry = StaticRegistry::from_config_dir(&config_dir).unwrap();
        let model = registry
            .get_model("gemini-2.5-flash")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(model.provider, "gemini");
        assert_eq!(model.context_window, 1048576);
    }

    #[tokio::test]
    async fn test_override_affects_filter() {
        let registry = StaticRegistry::from_models(test_models());
        // gpt-5.4 costs 2.50, override to 0.50
        ModelRegistry::apply_override(
            &registry,
            "gpt-5.4",
            ModelOverride {
                input_price_per_1m: Some(0.50),
                ..Default::default()
            },
        )
        .unwrap();
        let filter = ModelFilter {
            max_input_price_per_1m: Some(1.0),
            ..Default::default()
        };
        let models = registry.list_models(&filter).await.unwrap();
        // Now both deepseek-chat (0.28) and gpt-5.4 (overridden to 0.50) match
        assert_eq!(models.len(), 2);
    }
}
