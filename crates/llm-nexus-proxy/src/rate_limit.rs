//! Simple rate limiting configuration for the proxy.
//!
//! Uses tower's ConcurrencyLimit for request-level throttling.
//! Configure via `NEXUS_MAX_CONCURRENT_REQUESTS` (default: 100).

/// Get the configured max concurrent requests from environment.
pub fn max_concurrent_requests() -> usize {
    std::env::var("NEXUS_MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100)
}
