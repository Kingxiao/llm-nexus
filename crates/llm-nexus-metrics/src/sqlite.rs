//! SQLite-backed MetricsBackend.
//!
//! Persists call records to a SQLite database with auto-migration.
//! Aggregation uses SQL queries for efficient large-dataset handling.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use rusqlite::Connection;
use tokio::sync::Mutex;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::metrics::{AggregatedStats, CallRecord, MetricsBackend, StatsFilter};

/// SQLite metrics backend with auto-migration.
pub struct SqliteMetrics {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteMetrics {
    /// Open (or create) a SQLite database at the given path.
    pub fn open(path: impl AsRef<Path>) -> NexusResult<Self> {
        let conn = Connection::open(path).map_err(|e| {
            NexusError::ConfigError(format!("failed to open metrics database: {e}"))
        })?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create an in-memory SQLite database (useful for testing).
    pub fn in_memory() -> NexusResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            NexusError::ConfigError(format!("failed to open in-memory metrics db: {e}"))
        })?;
        Self::migrate(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn migrate(conn: &Connection) -> NexusResult<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS call_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                provider_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                estimated_cost_usd REAL NOT NULL,
                success INTEGER NOT NULL,
                error TEXT,
                timestamp TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_call_records_timestamp ON call_records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_call_records_provider ON call_records(provider_id);
            CREATE INDEX IF NOT EXISTS idx_call_records_model ON call_records(model_id);",
        )
        .map_err(|e| NexusError::ConfigError(format!("metrics migration failed: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl MetricsBackend for SqliteMetrics {
    async fn record_call(&self, record: CallRecord) -> NexusResult<()> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO call_records
                (request_id, provider_id, model_id, latency_ms, prompt_tokens,
                 completion_tokens, estimated_cost_usd, success, error, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                record.request_id,
                record.provider_id,
                record.model_id,
                record.latency_ms,
                record.prompt_tokens,
                record.completion_tokens,
                record.estimated_cost_usd,
                record.success as i32,
                record.error,
                record.timestamp.to_rfc3339(),
            ],
        )
        .map_err(|e| {
            NexusError::ProviderError {
                provider: "metrics-sqlite".into(),
                message: format!("failed to insert call record: {e}"),
                status_code: None,
            }
        })?;
        Ok(())
    }

    async fn query_stats(&self, filter: &StatsFilter) -> NexusResult<AggregatedStats> {
        let conn = self.conn.lock().await;

        let mut sql = String::from(
            "SELECT
                COUNT(*) as total_calls,
                COALESCE(SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), 0) as successful_calls,
                COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0) as failed_calls,
                COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                COALESCE(SUM(estimated_cost_usd), 0.0) as total_cost_usd,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms
             FROM call_records WHERE 1=1",
        );

        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(ref provider) = filter.provider_id {
            sql.push_str(&format!(" AND provider_id = ?{param_idx}"));
            params.push(Box::new(provider.clone()));
            param_idx += 1;
        }
        if let Some(ref model) = filter.model_id {
            sql.push_str(&format!(" AND model_id = ?{param_idx}"));
            params.push(Box::new(model.clone()));
            param_idx += 1;
        }
        if let Some(since) = filter.since {
            sql.push_str(&format!(" AND timestamp >= ?{param_idx}"));
            params.push(Box::new(since.to_rfc3339()));
            param_idx += 1;
        }
        if let Some(until) = filter.until {
            sql.push_str(&format!(" AND timestamp <= ?{param_idx}"));
            params.push(Box::new(until.to_rfc3339()));
        }
        let _ = param_idx;

        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

        let stats = conn
            .query_row(&sql, param_refs.as_slice(), |row| {
                Ok(AggregatedStats {
                    total_calls: row.get::<_, i64>(0)? as u64,
                    successful_calls: row.get::<_, i64>(1)? as u64,
                    failed_calls: row.get::<_, i64>(2)? as u64,
                    total_prompt_tokens: row.get::<_, i64>(3)? as u64,
                    total_completion_tokens: row.get::<_, i64>(4)? as u64,
                    total_cost_usd: row.get(5)?,
                    avg_latency_ms: row.get(6)?,
                    p99_latency_ms: None, // computed below
                })
            })
            .map_err(|e| NexusError::ProviderError {
                provider: "metrics-sqlite".into(),
                message: format!("stats query failed: {e}"),
                status_code: None,
            })?;

        // P99 latency via a separate query (SQLite has no built-in percentile)
        let p99 = if stats.total_calls > 0 {
            let mut p99_sql = String::from(
                "SELECT latency_ms FROM call_records WHERE 1=1",
            );

            // Reuse same filters
            let mut p99_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
            let mut pidx = 1;
            if let Some(ref provider) = filter.provider_id {
                p99_sql.push_str(&format!(" AND provider_id = ?{pidx}"));
                p99_params.push(Box::new(provider.clone()));
                pidx += 1;
            }
            if let Some(ref model) = filter.model_id {
                p99_sql.push_str(&format!(" AND model_id = ?{pidx}"));
                p99_params.push(Box::new(model.clone()));
                pidx += 1;
            }
            if let Some(since) = filter.since {
                p99_sql.push_str(&format!(" AND timestamp >= ?{pidx}"));
                p99_params.push(Box::new(since.to_rfc3339()));
                pidx += 1;
            }
            if let Some(until) = filter.until {
                p99_sql.push_str(&format!(" AND timestamp <= ?{pidx}"));
                p99_params.push(Box::new(until.to_rfc3339()));
                pidx += 1;
            }
            let _ = pidx; // suppress unused warning

            p99_sql.push_str(" ORDER BY latency_ms ASC");

            let p99_refs: Vec<&dyn rusqlite::types::ToSql> =
                p99_params.iter().map(|p| p.as_ref()).collect();

            let mut stmt = conn.prepare(&p99_sql).map_err(|e| NexusError::ProviderError {
                provider: "metrics-sqlite".into(),
                message: format!("p99 query prepare failed: {e}"),
                status_code: None,
            })?;

            let latencies: Vec<u64> = stmt
                .query_map(p99_refs.as_slice(), |row| row.get::<_, i64>(0))
                .map_err(|e| NexusError::ProviderError {
                    provider: "metrics-sqlite".into(),
                    message: format!("p99 query failed: {e}"),
                    status_code: None,
                })?
                .filter_map(|r| r.ok())
                .map(|v| v as u64)
                .collect();

            if !latencies.is_empty() {
                let idx = ((latencies.len() as f64) * 0.99).ceil() as usize;
                latencies.get(idx.saturating_sub(1)).copied()
            } else {
                None
            }
        } else {
            None
        };

        Ok(AggregatedStats {
            p99_latency_ms: p99,
            ..stats
        })
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
            request_id: format!("{provider}-{latency_ms}"),
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
                Some("test error".to_string())
            },
            timestamp,
        }
    }

    #[tokio::test]
    async fn test_sqlite_record_and_query() {
        let db = SqliteMetrics::in_memory().unwrap();
        let ts = Utc::now();

        for i in 0..10 {
            let record =
                make_record("test-provider", "test-model", 100 + i * 10, 500, 200, 0.01, true, ts);
            db.record_call(record).await.unwrap();
        }

        let stats = db.query_stats(&StatsFilter::default()).await.unwrap();
        assert_eq!(stats.total_calls, 10);
        assert_eq!(stats.successful_calls, 10);
        assert_eq!(stats.failed_calls, 0);
        assert_eq!(stats.total_prompt_tokens, 5_000);
        assert_eq!(stats.total_completion_tokens, 2_000);
        assert!((stats.total_cost_usd - 0.10).abs() < 1e-6);
        assert!(stats.p99_latency_ms.is_some());
    }

    #[tokio::test]
    async fn test_sqlite_filter_by_provider() {
        let db = SqliteMetrics::in_memory().unwrap();
        let ts = Utc::now();

        for _ in 0..5 {
            db.record_call(make_record("provider-a", "model-a", 100, 500, 200, 0.01, true, ts))
                .await
                .unwrap();
        }
        for _ in 0..3 {
            db.record_call(make_record("provider-b", "model-b", 150, 600, 300, 0.02, true, ts))
                .await
                .unwrap();
        }

        let stats = db
            .query_stats(&StatsFilter {
                provider_id: Some("provider-a".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(stats.total_calls, 5);

        let stats = db
            .query_stats(&StatsFilter {
                provider_id: Some("provider-b".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(stats.total_calls, 3);
    }

    #[tokio::test]
    async fn test_sqlite_filter_by_time_range() {
        let db = SqliteMetrics::in_memory().unwrap();

        let t1 = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap();
        let t2 = Utc.with_ymd_and_hms(2025, 6, 1, 0, 0, 0).unwrap();
        let t3 = Utc.with_ymd_and_hms(2025, 12, 1, 0, 0, 0).unwrap();

        db.record_call(make_record("p", "m", 100, 500, 200, 0.01, true, t1))
            .await
            .unwrap();
        db.record_call(make_record("p", "m", 120, 500, 200, 0.01, true, t2))
            .await
            .unwrap();
        db.record_call(make_record("p", "m", 140, 500, 200, 0.01, true, t3))
            .await
            .unwrap();

        let stats = db
            .query_stats(&StatsFilter {
                since: Some(Utc.with_ymd_and_hms(2025, 3, 1, 0, 0, 0).unwrap()),
                until: Some(Utc.with_ymd_and_hms(2025, 9, 1, 0, 0, 0).unwrap()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(stats.total_calls, 1);
        assert!((stats.avg_latency_ms - 120.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_sqlite_empty_stats() {
        let db = SqliteMetrics::in_memory().unwrap();
        let stats = db.query_stats(&StatsFilter::default()).await.unwrap();
        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.successful_calls, 0);
        assert!((stats.total_cost_usd - 0.0).abs() < 1e-10);
        assert!(stats.p99_latency_ms.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_with_failures() {
        let db = SqliteMetrics::in_memory().unwrap();
        let ts = Utc::now();

        db.record_call(make_record("p", "m", 100, 500, 200, 0.01, true, ts))
            .await
            .unwrap();
        db.record_call(make_record("p", "m", 200, 0, 0, 0.0, false, ts))
            .await
            .unwrap();

        let stats = db.query_stats(&StatsFilter::default()).await.unwrap();
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.successful_calls, 1);
        assert_eq!(stats.failed_calls, 1);
    }
}
