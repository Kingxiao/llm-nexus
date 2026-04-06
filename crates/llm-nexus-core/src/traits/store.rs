//! Key-value store abstraction for caching, budget state, etc.

use std::time::Duration;

use crate::error::NexusResult;

/// Generic async key-value store.
///
/// Used by cache middleware, budget middleware, and any component that
/// needs persistent or semi-persistent storage.
///
/// Implementations: [`InMemoryStore`](crate::store::InMemoryStore) (default).
#[async_trait::async_trait]
pub trait KeyValueStore: Send + Sync + 'static {
    /// Get a value by key. Returns `None` if not found or expired.
    async fn get(&self, key: &str) -> NexusResult<Option<Vec<u8>>>;

    /// Set a value with optional TTL. Overwrites existing value.
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> NexusResult<()>;

    /// Delete a key. No-op if not found.
    async fn delete(&self, key: &str) -> NexusResult<()>;
}
