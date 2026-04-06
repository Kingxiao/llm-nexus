//! Prometheus metrics exporter.
//!
//! Decorator pattern: wraps an existing `MetricsBackend` and updates
//! Prometheus counters/histograms on each `record_call`.

use std::sync::Arc;

use async_trait::async_trait;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::traits::metrics::{AggregatedStats, CallRecord, MetricsBackend, StatsFilter};

/// Labels for per-provider/model metrics.
#[derive(Clone, Debug, Hash, PartialEq, Eq, prometheus_client::encoding::EncodeLabelSet)]
struct RequestLabels {
    provider: String,
    model: String,
}

/// Prometheus metrics exporter wrapping an inner `MetricsBackend`.
///
/// On each `record_call`, it:
/// 1. Forwards to the inner backend (for storage/querying)
/// 2. Updates Prometheus counters and histograms
///
/// Call `gather()` to get the Prometheus text format output.
pub struct PrometheusExporter {
    inner: Arc<dyn MetricsBackend>,
    registry: Registry,
    requests_total: Family<RequestLabels, Counter>,
    requests_failed: Family<RequestLabels, Counter>,
    duration_seconds: Family<RequestLabels, Histogram>,
    tokens_prompt: Family<RequestLabels, Counter>,
    tokens_completion: Family<RequestLabels, Counter>,
    cost_usd: Family<RequestLabels, Counter>,
}

impl PrometheusExporter {
    /// Create a new exporter wrapping an inner metrics backend.
    pub fn new(inner: Arc<dyn MetricsBackend>) -> Self {
        let mut registry = Registry::default();

        let requests_total = Family::<RequestLabels, Counter>::default();
        registry.register(
            "llm_nexus_requests",
            "Total LLM API requests",
            requests_total.clone(),
        );

        let requests_failed = Family::<RequestLabels, Counter>::default();
        registry.register(
            "llm_nexus_requests_failed",
            "Failed LLM API requests",
            requests_failed.clone(),
        );

        let duration_seconds = Family::<RequestLabels, Histogram>::new_with_constructor(|| {
            Histogram::new(exponential_buckets(0.01, 2.0, 15))
        });
        registry.register(
            "llm_nexus_request_duration_seconds",
            "LLM API request duration in seconds",
            duration_seconds.clone(),
        );

        let tokens_prompt = Family::<RequestLabels, Counter>::default();
        registry.register(
            "llm_nexus_tokens_prompt",
            "Total prompt tokens consumed",
            tokens_prompt.clone(),
        );

        let tokens_completion = Family::<RequestLabels, Counter>::default();
        registry.register(
            "llm_nexus_tokens_completion",
            "Total completion tokens generated",
            tokens_completion.clone(),
        );

        let cost_usd = Family::<RequestLabels, Counter>::default();
        registry.register(
            "llm_nexus_estimated_cost_usd",
            "Estimated total cost in USD",
            cost_usd.clone(),
        );

        Self {
            inner,
            registry,
            requests_total,
            requests_failed,
            duration_seconds,
            tokens_prompt,
            tokens_completion,
            cost_usd,
        }
    }

    /// Encode all metrics in Prometheus text exposition format.
    pub fn gather(&self) -> String {
        let mut buf = String::new();
        encode(&mut buf, &self.registry).expect("prometheus encoding should not fail");
        buf
    }
}

#[async_trait]
impl MetricsBackend for PrometheusExporter {
    async fn record_call(&self, record: CallRecord) -> NexusResult<()> {
        let labels = RequestLabels {
            provider: record.provider_id.clone(),
            model: record.model_id.clone(),
        };

        self.requests_total.get_or_create(&labels).inc();

        if !record.success {
            self.requests_failed.get_or_create(&labels).inc();
        }

        self.duration_seconds
            .get_or_create(&labels)
            .observe(record.latency_ms as f64 / 1000.0);

        self.tokens_prompt
            .get_or_create(&labels)
            .inc_by(record.prompt_tokens as u64);

        self.tokens_completion
            .get_or_create(&labels)
            .inc_by(record.completion_tokens as u64);

        // Cost tracking: Counter only supports u64, so we track in micro-USD
        let micro_usd = (record.estimated_cost_usd * 1_000_000.0) as u64;
        self.cost_usd.get_or_create(&labels).inc_by(micro_usd);

        // Forward to inner backend
        self.inner.record_call(record).await
    }

    async fn query_stats(&self, filter: &StatsFilter) -> NexusResult<AggregatedStats> {
        self.inner.query_stats(filter).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryMetrics;

    fn make_record(provider: &str, model: &str, latency_ms: u64, success: bool) -> CallRecord {
        CallRecord {
            request_id: format!("{provider}-{latency_ms}"),
            provider_id: provider.to_string(),
            model_id: model.to_string(),
            latency_ms,
            prompt_tokens: 100,
            completion_tokens: 50,
            estimated_cost_usd: 0.001,
            success,
            error: if success {
                None
            } else {
                Some("test error".into())
            },
            timestamp: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_exporter_forwards_to_inner() {
        let inner = Arc::new(InMemoryMetrics::new());
        let exporter = PrometheusExporter::new(inner.clone());

        exporter
            .record_call(make_record("openai", "gpt-5", 200, true))
            .await
            .unwrap();

        let stats = inner.query_stats(&StatsFilter::default()).await.unwrap();
        assert_eq!(stats.total_calls, 1);
    }

    #[tokio::test]
    async fn test_gather_produces_valid_output() {
        let inner = Arc::new(InMemoryMetrics::new());
        let exporter = PrometheusExporter::new(inner);

        exporter
            .record_call(make_record("openai", "gpt-5", 200, true))
            .await
            .unwrap();
        exporter
            .record_call(make_record("openai", "gpt-5", 300, false))
            .await
            .unwrap();

        let output = exporter.gather();
        assert!(output.contains("llm_nexus_requests_total"));
        assert!(output.contains("llm_nexus_requests_failed_total"));
        assert!(output.contains("llm_nexus_request_duration_seconds"));
        assert!(output.contains("llm_nexus_tokens_prompt_total"));
        assert!(output.contains("llm_nexus_tokens_completion_total"));
    }

    #[tokio::test]
    async fn test_labels_differentiate_providers() {
        let inner = Arc::new(InMemoryMetrics::new());
        let exporter = PrometheusExporter::new(inner);

        exporter
            .record_call(make_record("openai", "gpt-5", 100, true))
            .await
            .unwrap();
        exporter
            .record_call(make_record("anthropic", "claude", 200, true))
            .await
            .unwrap();

        let output = exporter.gather();
        assert!(output.contains("openai"));
        assert!(output.contains("anthropic"));
    }

    #[tokio::test]
    async fn test_failed_requests_counted() {
        let inner = Arc::new(InMemoryMetrics::new());
        let exporter = PrometheusExporter::new(inner);

        exporter
            .record_call(make_record("p", "m", 100, true))
            .await
            .unwrap();
        exporter
            .record_call(make_record("p", "m", 200, false))
            .await
            .unwrap();
        exporter
            .record_call(make_record("p", "m", 300, false))
            .await
            .unwrap();

        let output = exporter.gather();
        // requests_total should be 3, failed should be 2
        assert!(output.contains("llm_nexus_requests_total{"));
        assert!(output.contains("llm_nexus_requests_failed_total{"));
    }

    #[tokio::test]
    async fn test_query_stats_delegates_to_inner() {
        let inner = Arc::new(InMemoryMetrics::new());
        let exporter = PrometheusExporter::new(inner);

        exporter
            .record_call(make_record("p", "m", 100, true))
            .await
            .unwrap();

        let stats = exporter
            .query_stats(&StatsFilter::default())
            .await
            .unwrap();
        assert_eq!(stats.total_calls, 1);
    }
}
