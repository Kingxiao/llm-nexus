//! Metrics collection and export backends for llm-nexus.
//!
//! Records per-request telemetry (latency, token counts, cost) and exposes
//! aggregated statistics via the [`MetricsCollector`](llm_nexus_core::traits::metrics::MetricsCollector)
//! trait. The default [`InMemoryMetrics`] backend requires no external dependencies.
//!
//! # Feature Flags
//!
//! | Feature | Backend |
//! |---------|---------|
//! | _(default)_ | [`InMemoryMetrics`] — bounded in-memory ring buffer |
//! | `sqlite` | [`SqliteMetrics`] — persistent SQLite storage |
//! | `prometheus` | [`PrometheusExporter`] — Prometheus exposition format |

pub mod aggregation;
pub mod cost_calculator;
pub mod in_memory;
#[cfg(feature = "prometheus")]
pub mod prom;
#[cfg(feature = "sqlite")]
pub mod sqlite;

pub use in_memory::InMemoryMetrics;
#[cfg(feature = "prometheus")]
pub use prom::PrometheusExporter;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteMetrics;
