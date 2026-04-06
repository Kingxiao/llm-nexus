//! In-memory LogBackend implementation.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::error::NexusResult;
use crate::traits::logging::{LogBackend, LogFilter, RequestLogEntry};

/// In-memory log storage. Suitable for development and testing.
///
/// Uses `VecDeque` for O(1) eviction of oldest entries.
pub struct InMemoryLogBackend {
    entries: Mutex<VecDeque<RequestLogEntry>>,
    /// Maximum entries to retain. Oldest entries are evicted when exceeded.
    capacity: usize,
}

impl InMemoryLogBackend {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            capacity,
        }
    }
}

impl Default for InMemoryLogBackend {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[async_trait::async_trait]
impl LogBackend for InMemoryLogBackend {
    async fn log_request(&self, entry: RequestLogEntry) -> NexusResult<()> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if entries.len() >= self.capacity {
            entries.pop_front(); // O(1) instead of Vec::remove(0) which is O(n)
        }
        entries.push_back(entry);
        Ok(())
    }

    async fn query_logs(&self, filter: &LogFilter) -> NexusResult<Vec<RequestLogEntry>> {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let iter = entries.iter().filter(|e| {
            if filter.provider_id.as_ref().is_some_and(|pid| &e.provider_id != pid) {
                return false;
            }
            if filter.model_id.as_ref().is_some_and(|mid| &e.model_id != mid) {
                return false;
            }
            if filter.since.is_some_and(|since| e.timestamp < since) {
                return false;
            }
            if filter.until.is_some_and(|until| e.timestamp > until) {
                return false;
            }
            true
        });

        let results: Vec<_> = match filter.limit {
            Some(limit) => iter.rev().take(limit).cloned().collect(),
            None => iter.cloned().collect(),
        };

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::response::Usage;
    use std::collections::HashMap;

    fn make_entry(provider: &str, model: &str) -> RequestLogEntry {
        RequestLogEntry {
            request_id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            provider_id: provider.into(),
            model_id: model.into(),
            request_body: None,
            response_body: None,
            latency_ms: 100,
            usage: Usage::default(),
            cost_usd: 0.01,
            success: true,
            error: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_log_and_query() {
        let backend = InMemoryLogBackend::new(100);
        backend.log_request(make_entry("openai", "gpt-5")).await.unwrap();
        backend.log_request(make_entry("anthropic", "claude")).await.unwrap();

        let all = backend.query_logs(&LogFilter::default()).await.unwrap();
        assert_eq!(all.len(), 2);

        let filtered = backend
            .query_logs(&LogFilter {
                provider_id: Some("openai".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].provider_id, "openai");
    }

    #[tokio::test]
    async fn test_capacity_eviction() {
        let backend = InMemoryLogBackend::new(3);
        for i in 0..5 {
            backend
                .log_request(make_entry("p", &format!("m{i}")))
                .await
                .unwrap();
        }
        let all = backend.query_logs(&LogFilter::default()).await.unwrap();
        assert_eq!(all.len(), 3);
        // Oldest should be evicted
        assert_eq!(all[0].model_id, "m2");
    }

    #[tokio::test]
    async fn test_query_with_limit() {
        let backend = InMemoryLogBackend::new(100);
        for i in 0..10 {
            backend
                .log_request(make_entry("p", &format!("m{i}")))
                .await
                .unwrap();
        }
        let limited = backend
            .query_logs(&LogFilter {
                limit: Some(3),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(limited.len(), 3);
    }
}
