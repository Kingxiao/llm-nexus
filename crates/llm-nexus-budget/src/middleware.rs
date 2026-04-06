//! BudgetMiddleware — pre-checks and post-records cost against budget.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use tokio::sync::Mutex;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::pipeline::context::RequestContext;
use llm_nexus_core::pipeline::middleware::{ChatMiddleware, Next, NextStream};
use llm_nexus_core::traits::store::KeyValueStore;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};

use crate::types::{BudgetConfig, BudgetStatus};

/// Middleware that enforces spend limits.
///
/// Thread-safe: per-key locks prevent TOCTOU races on concurrent requests.
pub struct BudgetMiddleware {
    store: Arc<dyn KeyValueStore>,
    default_config: BudgetConfig,
    /// Per-key locks to serialize budget check + spend recording.
    key_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

/// Budget key identifier inserted into RequestContext.
pub struct BudgetKey(pub String);

impl BudgetMiddleware {
    pub fn new(store: Arc<dyn KeyValueStore>, default_config: BudgetConfig) -> Self {
        Self {
            store,
            default_config,
            key_locks: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a per-key lock.
    async fn key_lock(&self, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.key_locks.lock().await;
        // Sweep idle locks: if strong_count == 1, only the HashMap holds it,
        // meaning no request is currently using it. Safe to remove.
        locks.retain(|_, v| Arc::strong_count(v) > 1);
        locks
            .entry(key.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    async fn get_status(&self, key: &str) -> NexusResult<BudgetStatus> {
        let Some(data) = self.store.get(key).await? else {
            return Ok(BudgetStatus::new(self.default_config.limit_usd));
        };
        let Ok(status) = serde_json::from_slice::<BudgetStatus>(&data) else {
            return Ok(BudgetStatus::new(self.default_config.limit_usd));
        };

        if self.default_config.period.should_reset(status.period_start) {
            let fresh = BudgetStatus::new(self.default_config.limit_usd);
            self.save_status(key, &fresh).await?;
            return Ok(fresh);
        }
        Ok(status)
    }

    async fn record_spend(&self, key: &str, cost_usd: f64) -> NexusResult<()> {
        // Lock is already held by caller (process method)
        let mut status = self.get_status(key).await?;
        status.spent_usd += cost_usd;
        status.remaining_usd = (self.default_config.limit_usd - status.spent_usd).max(0.0);
        status.exceeded = status.spent_usd >= self.default_config.limit_usd;
        self.save_status(key, &status).await
    }

    async fn save_status(&self, key: &str, status: &BudgetStatus) -> NexusResult<()> {
        let serialized = serde_json::to_vec(status)
            .map_err(|e| NexusError::SerializationError(e.to_string()))?;
        self.store.set(key, &serialized, None).await
    }
}

#[async_trait::async_trait]
impl ChatMiddleware for BudgetMiddleware {
    async fn process(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: Next<'_>,
    ) -> NexusResult<ChatResponse> {
        let budget_key = ctx
            .get::<BudgetKey>()
            .map(|k| k.0.clone())
            .unwrap_or_else(|| "budget:default".to_string());

        // Acquire per-key lock to prevent TOCTOU race
        let lock = self.key_lock(&budget_key).await;
        let _guard = lock.lock().await;

        let status = self.get_status(&budget_key).await?;
        if status.exceeded {
            return Err(NexusError::BudgetExceeded(format!(
                "budget exceeded: spent ${:.4} of ${:.4} limit",
                status.spent_usd, self.default_config.limit_usd
            )));
        }

        // Release lock during the actual LLM call (don't block other budget keys)
        drop(_guard);

        let response = next.run(ctx, request).await?;

        // Re-acquire lock to record spend atomically
        let _guard = lock.lock().await;
        if let Some(ref model_meta) = ctx.model_meta {
            let cost =
                llm_nexus_metrics::cost_calculator::calculate_cost(&response.usage, model_meta);
            if let Err(e) = self.record_spend(&budget_key, cost).await {
                tracing::warn!(error = %e, "failed to record budget spend");
            }
        }

        Ok(response)
    }

    async fn process_stream(
        &self,
        ctx: &mut RequestContext,
        request: &ChatRequest,
        next: NextStream<'_>,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let budget_key = ctx
            .get::<BudgetKey>()
            .map(|k| k.0.clone())
            .unwrap_or_else(|| "budget:default".to_string());

        let lock = self.key_lock(&budget_key).await;
        let _guard = lock.lock().await;

        let status = self.get_status(&budget_key).await?;
        if status.exceeded {
            return Err(NexusError::BudgetExceeded(format!(
                "budget exceeded: spent ${:.4} of ${:.4} limit",
                status.spent_usd, self.default_config.limit_usd
            )));
        }

        drop(_guard);

        let stream = next.run(ctx, request).await?;

        // Wrap stream to record spend when final chunk with usage arrives
        let store = self.store.clone();
        let limit = self.default_config.limit_usd;
        let model_meta = ctx.model_meta.clone();

        use futures::StreamExt;
        let metered = stream.inspect(move |chunk| {
            let Some(usage) = chunk.as_ref().ok().and_then(|c| c.usage.as_ref()) else {
                return;
            };
            let Some(meta) = &model_meta else { return };

            let cost = llm_nexus_metrics::cost_calculator::calculate_cost(usage, meta);
            let store = store.clone();
            let key = budget_key.clone();
            tokio::spawn(async move {
                // Best-effort spend recording for streaming
                let Ok(Some(data)) = store.get(&key).await else {
                    return;
                };
                let Ok(mut status) = serde_json::from_slice::<BudgetStatus>(&data) else {
                    return;
                };
                status.spent_usd += cost;
                status.remaining_usd = (limit - status.spent_usd).max(0.0);
                status.exceeded = status.spent_usd >= limit;
                if let Ok(serialized) = serde_json::to_vec(&status) {
                    let _ = store.set(&key, &serialized, None).await;
                }
            });
        });

        Ok(Box::pin(metered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BudgetPeriod;
    use llm_nexus_core::store::InMemoryStore;

    #[tokio::test]
    async fn test_budget_status_starts_at_zero() {
        let store = Arc::new(InMemoryStore::new());
        let config = BudgetConfig {
            limit_usd: 10.0,
            period: BudgetPeriod::Daily,
        };
        let middleware = BudgetMiddleware::new(store, config);

        let status = middleware.get_status("budget:test").await.unwrap();
        assert!((status.spent_usd).abs() < f64::EPSILON);
        assert!((status.remaining_usd - 10.0).abs() < f64::EPSILON);
        assert!(!status.exceeded);
    }

    #[tokio::test]
    async fn test_record_spend_accumulates() {
        let store = Arc::new(InMemoryStore::new());
        let config = BudgetConfig {
            limit_usd: 10.0,
            period: BudgetPeriod::Total,
        };
        let middleware = BudgetMiddleware::new(store, config);

        middleware.record_spend("budget:test", 3.5).await.unwrap();
        middleware.record_spend("budget:test", 2.5).await.unwrap();

        let status = middleware.get_status("budget:test").await.unwrap();
        assert!((status.spent_usd - 6.0).abs() < 0.01);
        assert!((status.remaining_usd - 4.0).abs() < 0.01);
        assert!(!status.exceeded);
    }

    #[tokio::test]
    async fn test_budget_exceeded() {
        let store = Arc::new(InMemoryStore::new());
        let config = BudgetConfig {
            limit_usd: 5.0,
            period: BudgetPeriod::Total,
        };
        let middleware = BudgetMiddleware::new(store, config);

        middleware.record_spend("budget:test", 5.0).await.unwrap();

        let status = middleware.get_status("budget:test").await.unwrap();
        assert!(status.exceeded);
    }

    #[tokio::test]
    async fn test_separate_budget_keys() {
        let store = Arc::new(InMemoryStore::new());
        let config = BudgetConfig {
            limit_usd: 10.0,
            period: BudgetPeriod::Total,
        };
        let middleware = BudgetMiddleware::new(store, config);

        middleware.record_spend("budget:team-a", 7.0).await.unwrap();
        middleware.record_spend("budget:team-b", 2.0).await.unwrap();

        let a = middleware.get_status("budget:team-a").await.unwrap();
        let b = middleware.get_status("budget:team-b").await.unwrap();
        assert!((a.spent_usd - 7.0).abs() < 0.01);
        assert!((b.spent_usd - 2.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_daily_period_reset() {
        let store = Arc::new(InMemoryStore::new());
        let config = BudgetConfig {
            limit_usd: 10.0,
            period: BudgetPeriod::Daily,
        };
        let middleware = BudgetMiddleware::new(store.clone(), config);

        middleware.record_spend("budget:daily", 8.0).await.unwrap();

        let mut status = middleware.get_status("budget:daily").await.unwrap();
        status.period_start = chrono::Utc::now() - chrono::Duration::days(1);
        middleware
            .save_status("budget:daily", &status)
            .await
            .unwrap();

        let fresh = middleware.get_status("budget:daily").await.unwrap();
        assert!((fresh.spent_usd).abs() < f64::EPSILON);
        assert!(!fresh.exceeded);
    }

    #[tokio::test]
    async fn test_total_period_no_reset() {
        let store = Arc::new(InMemoryStore::new());
        let config = BudgetConfig {
            limit_usd: 10.0,
            period: BudgetPeriod::Total,
        };
        let middleware = BudgetMiddleware::new(store.clone(), config);

        middleware.record_spend("budget:total", 8.0).await.unwrap();

        let mut status = middleware.get_status("budget:total").await.unwrap();
        status.period_start = chrono::Utc::now() - chrono::Duration::days(365);
        middleware
            .save_status("budget:total", &status)
            .await
            .unwrap();

        let same = middleware.get_status("budget:total").await.unwrap();
        assert!((same.spent_usd - 8.0).abs() < 0.01);
    }
}
