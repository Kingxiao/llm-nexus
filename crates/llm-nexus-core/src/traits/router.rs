use serde::{Deserialize, Serialize};

use crate::error::NexusResult;
use crate::types::Capability;

/// Context for making a routing decision.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouteContext {
    pub task_type: Option<TaskType>,
    pub max_cost_per_1k_tokens: Option<f64>,
    pub max_latency_ms: Option<u64>,
    pub required_capabilities: Vec<Capability>,
    pub preferred_providers: Vec<String>,
    pub excluded_providers: Vec<String>,
    /// Stable key for A/B experiment assignment (e.g. user_id, session_id).
    /// If set, ExperimentRouter uses this for consistent variant selection.
    /// If unset, a random key is generated per request.
    pub experiment_key: Option<String>,
}

/// High-level task categories that influence model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Chat,
    CodeGeneration,
    Reasoning,
    Embedding,
    ImageUnderstanding,
}

/// The result of a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub provider_id: String,
    pub model_id: String,
    pub estimated_cost_per_1k: Option<f64>,
    pub estimated_latency_ms: Option<u64>,
}

/// Trait for selecting the best provider/model for a given request context.
#[async_trait::async_trait]
pub trait Router: Send + Sync + 'static {
    /// Selects the best provider/model for the given context.
    async fn route(&self, context: &RouteContext) -> NexusResult<RouteDecision>;

    /// Returns an ordered list of fallback options for the given context.
    async fn fallback_chain(&self, context: &RouteContext) -> NexusResult<Vec<RouteDecision>>;
}
