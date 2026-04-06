use serde::{Deserialize, Serialize};

use crate::error::NexusResult;

/// A single LLM API call record for metrics tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRecord {
    pub request_id: String,
    pub provider_id: String,
    pub model_id: String,
    pub latency_ms: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub estimated_cost_usd: f64,
    pub success: bool,
    pub error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Filter criteria for querying aggregated stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsFilter {
    pub provider_id: Option<String>,
    pub model_id: Option<String>,
    pub since: Option<chrono::DateTime<chrono::Utc>>,
    pub until: Option<chrono::DateTime<chrono::Utc>>,
}

/// Aggregated statistics over a set of call records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_cost_usd: f64,
    pub avg_latency_ms: f64,
    pub p99_latency_ms: Option<u64>,
}

/// Trait for recording and querying LLM API call metrics.
#[async_trait::async_trait]
pub trait MetricsBackend: Send + Sync + 'static {
    /// Records a single API call.
    async fn record_call(&self, record: CallRecord) -> NexusResult<()>;

    /// Queries aggregated statistics matching the given filter.
    async fn query_stats(&self, filter: &StatsFilter) -> NexusResult<AggregatedStats>;
}
