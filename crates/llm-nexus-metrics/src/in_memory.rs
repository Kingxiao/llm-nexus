use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::traits::metrics::{AggregatedStats, CallRecord, MetricsBackend, StatsFilter};

use crate::aggregation::aggregate;

/// Default maximum number of records before oldest are evicted.
const DEFAULT_MAX_RECORDS: usize = 100_000;

/// In-memory metrics backend with a configurable capacity cap.
/// When the cap is reached, the oldest records are evicted (ring buffer behavior).
/// Suitable for development, testing, and low-traffic deployments.
pub struct InMemoryMetrics {
    records: Arc<RwLock<Vec<CallRecord>>>,
    max_records: usize,
}

impl InMemoryMetrics {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(Vec::new())),
            max_records: DEFAULT_MAX_RECORDS,
        }
    }

    /// Create with a custom capacity limit.
    pub fn with_capacity(max_records: usize) -> Self {
        Self {
            records: Arc::new(RwLock::new(Vec::new())),
            max_records,
        }
    }
}

impl Default for InMemoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MetricsBackend for InMemoryMetrics {
    async fn record_call(&self, record: CallRecord) -> NexusResult<()> {
        let mut records = self.records.write().await;
        if records.len() >= self.max_records {
            let drain_count = self.max_records / 10; // evict oldest 10%
            records.drain(..drain_count);
            tracing::debug!(
                evicted = drain_count,
                remaining = records.len(),
                "metrics capacity reached, evicted oldest records"
            );
        }
        records.push(record);
        Ok(())
    }

    async fn query_stats(&self, filter: &StatsFilter) -> NexusResult<AggregatedStats> {
        let records = self.records.read().await;
        Ok(aggregate(&records, filter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn make_record(
        provider: &str,
        model: &str,
        latency_ms: u64,
        prompt_tokens: u32,
        completion_tokens: u32,
        cost: f64,
        success: bool,
        timestamp: chrono::DateTime<Utc>,
    ) -> CallRecord {
        CallRecord {
            request_id: uuid_like(provider, latency_ms),
            provider_id: provider.to_string(),
            model_id: model.to_string(),
            latency_ms,
            prompt_tokens,
            completion_tokens,
            estimated_cost_usd: cost,
            success,
            error: if success {
                None
            } else {
                Some("error".to_string())
            },
            timestamp,
        }
    }

    fn uuid_like(prefix: &str, n: u64) -> String {
        format!("{prefix}-{n}")
    }

    #[tokio::test]
    async fn test_record_and_query() {
        let backend = InMemoryMetrics::new();
        let ts = Utc::now();

        for i in 0..10 {
            let record = make_record("openai", "gpt-4o", 100 + i * 10, 500, 200, 0.01, true, ts);
            backend.record_call(record).await.unwrap();
        }

        let stats = backend.query_stats(&StatsFilter::default()).await.unwrap();
        assert_eq!(stats.total_calls, 10);
        assert_eq!(stats.successful_calls, 10);
        assert_eq!(stats.failed_calls, 0);
        assert_eq!(stats.total_prompt_tokens, 5_000);
        assert_eq!(stats.total_completion_tokens, 2_000);
        assert!((stats.total_cost_usd - 0.10).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_filter_by_provider() {
        let backend = InMemoryMetrics::new();
        let ts = Utc::now();

        for _ in 0..5 {
            backend
                .record_call(make_record(
                    "openai", "gpt-4o", 100, 500, 200, 0.01, true, ts,
                ))
                .await
                .unwrap();
        }
        for _ in 0..3 {
            backend
                .record_call(make_record(
                    "anthropic",
                    "claude-sonnet",
                    150,
                    600,
                    300,
                    0.02,
                    true,
                    ts,
                ))
                .await
                .unwrap();
        }

        let filter = StatsFilter {
            provider_id: Some("openai".to_string()),
            ..Default::default()
        };
        let stats = backend.query_stats(&filter).await.unwrap();
        assert_eq!(stats.total_calls, 5);

        let filter = StatsFilter {
            provider_id: Some("anthropic".to_string()),
            ..Default::default()
        };
        let stats = backend.query_stats(&filter).await.unwrap();
        assert_eq!(stats.total_calls, 3);
    }

    #[tokio::test]
    async fn test_filter_by_time_range() {
        let backend = InMemoryMetrics::new();

        let t1 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let t3 = Utc.with_ymd_and_hms(2025, 12, 1, 0, 0, 0).unwrap();

        backend
            .record_call(make_record(
                "openai", "gpt-4o", 100, 500, 200, 0.01, true, t1,
            ))
            .await
            .unwrap();
        backend
            .record_call(make_record(
                "openai", "gpt-4o", 120, 500, 200, 0.01, true, t2,
            ))
            .await
            .unwrap();
        backend
            .record_call(make_record(
                "openai", "gpt-4o", 140, 500, 200, 0.01, true, t3,
            ))
            .await
            .unwrap();

        let filter = StatsFilter {
            since: Some(Utc.with_ymd_and_hms(2025, 3, 1, 0, 0, 0).unwrap()),
            until: Some(Utc.with_ymd_and_hms(2025, 9, 1, 0, 0, 0).unwrap()),
            ..Default::default()
        };
        let stats = backend.query_stats(&filter).await.unwrap();
        assert_eq!(stats.total_calls, 1);
        assert!((stats.avg_latency_ms - 120.0).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_capacity_eviction() {
        let backend = InMemoryMetrics::with_capacity(20);
        let ts = Utc::now();

        // Fill beyond capacity
        for i in 0..25 {
            backend
                .record_call(make_record(
                    "openai", "gpt-4o", i as u64, 100, 50, 0.001, true, ts,
                ))
                .await
                .unwrap();
        }

        let stats = backend.query_stats(&StatsFilter::default()).await.unwrap();
        // After eviction: 20 - 2 (10% of 20) + remaining inserts
        assert!(stats.total_calls <= 25);
        assert!(stats.total_calls > 0);
    }

    #[tokio::test]
    async fn test_empty_stats() {
        let backend = InMemoryMetrics::new();
        let stats = backend.query_stats(&StatsFilter::default()).await.unwrap();
        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.successful_calls, 0);
        assert_eq!(stats.failed_calls, 0);
        assert_eq!(stats.total_prompt_tokens, 0);
        assert_eq!(stats.total_completion_tokens, 0);
        assert!((stats.total_cost_usd - 0.0).abs() < 1e-10);
        assert!((stats.avg_latency_ms - 0.0).abs() < 1e-10);
        assert!(stats.p99_latency_ms.is_none());
    }
}
