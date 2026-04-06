use crate::error::NexusResult;
use crate::types::{ModelFilter, ModelMetadata, ModelOverride};

/// Trait for accessing and managing model metadata.
#[async_trait::async_trait]
pub trait ModelRegistry: Send + Sync + 'static {
    /// Retrieves metadata for a specific model by ID.
    async fn get_model(&self, model_id: &str) -> NexusResult<Option<ModelMetadata>>;

    /// Lists models matching the given filter criteria.
    async fn list_models(&self, filter: &ModelFilter) -> NexusResult<Vec<ModelMetadata>>;

    /// Refreshes the model registry from its data source.
    async fn refresh(&self) -> NexusResult<()>;

    /// Applies runtime overrides to a model's metadata (e.g. custom pricing).
    fn apply_override(&self, model_id: &str, overrides: ModelOverride) -> NexusResult<()>;
}
