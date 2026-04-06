//! Per-request logging abstraction.
//!
//! Unlike [`MetricsBackend`](super::metrics::MetricsBackend) which stores aggregated
//! stats, `LogBackend` stores individual request/response records for debugging,
//! auditing, and analytics.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::NexusResult;
use crate::types::response::Usage;

/// A complete log entry for a single request/response cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLogEntry {
    pub request_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub provider_id: String,
    pub model_id: String,
    /// The request body (optional — may be omitted for privacy).
    pub request_body: Option<serde_json::Value>,
    /// The response body (optional — may be omitted for privacy).
    pub response_body: Option<serde_json::Value>,
    pub latency_ms: u64,
    pub usage: Usage,
    pub cost_usd: f64,
    pub success: bool,
    pub error: Option<String>,
    /// Arbitrary metadata from RequestContext extensions.
    pub metadata: HashMap<String, String>,
}

/// Filter for querying log entries.
#[derive(Debug, Default)]
pub struct LogFilter {
    pub provider_id: Option<String>,
    pub model_id: Option<String>,
    pub since: Option<chrono::DateTime<chrono::Utc>>,
    pub until: Option<chrono::DateTime<chrono::Utc>>,
    pub limit: Option<usize>,
}

/// Backend for per-request logging.
#[async_trait::async_trait]
pub trait LogBackend: Send + Sync + 'static {
    /// Log a request/response entry.
    async fn log_request(&self, entry: RequestLogEntry) -> NexusResult<()>;

    /// Query log entries with optional filters.
    async fn query_logs(&self, filter: &LogFilter) -> NexusResult<Vec<RequestLogEntry>>;
}
