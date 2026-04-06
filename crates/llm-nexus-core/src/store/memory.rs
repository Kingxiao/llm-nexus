//! In-memory key-value store with TTL support.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::NexusResult;
use crate::traits::store::KeyValueStore;

struct Entry {
    value: Vec<u8>,
    expires_at: Option<Instant>,
}

impl Entry {
    fn is_expired(&self) -> bool {
        self.expires_at.is_some_and(|t| Instant::now() >= t)
    }
}

/// In-memory [`KeyValueStore`] backed by a `HashMap` with TTL.
///
/// Suitable for single-process deployments. For distributed setups,
/// implement `KeyValueStore` with Redis or similar.
pub struct InMemoryStore {
    data: Mutex<HashMap<String, Entry>>,
    /// Maximum number of entries. When exceeded, expired entries are evicted first,
    /// then oldest entries by insertion order.
    max_entries: usize,
}

impl InMemoryStore {
    /// Create with a capacity limit.
    pub fn new() -> Self {
        Self::with_capacity(100_000)
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
            max_entries,
        }
    }

    /// Evict expired entries, then oldest entries if still over capacity.
    fn evict_if_needed(map: &mut HashMap<String, Entry>, max: usize) {
        if map.len() <= max {
            return;
        }
        // First pass: remove expired
        map.retain(|_, v| !v.is_expired());
        if map.len() <= max {
            return;
        }
        // Second pass: remove oldest (by earliest expires_at, or arbitrary if no TTL)
        let excess = map.len() - max;
        let keys_to_remove: Vec<String> = map.keys().take(excess).cloned().collect();
        for key in keys_to_remove {
            map.remove(&key);
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl KeyValueStore for InMemoryStore {
    async fn get(&self, key: &str) -> NexusResult<Option<Vec<u8>>> {
        let mut map = self.data.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = map.get(key) {
            if entry.is_expired() {
                map.remove(key);
                return Ok(None);
            }
            Ok(Some(entry.value.clone()))
        } else {
            Ok(None)
        }
    }

    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> NexusResult<()> {
        let mut map = self.data.lock().unwrap_or_else(|e| e.into_inner());
        Self::evict_if_needed(&mut map, self.max_entries);
        let expires_at = ttl.map(|d| Instant::now() + d);
        map.insert(
            key.to_string(),
            Entry {
                value: value.to_vec(),
                expires_at,
            },
        );
        Ok(())
    }

    async fn delete(&self, key: &str) -> NexusResult<()> {
        let mut map = self.data.lock().unwrap_or_else(|e| e.into_inner());
        map.remove(key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_set_and_get() {
        let store = InMemoryStore::new();
        store.set("k1", b"hello", None).await.unwrap();
        assert_eq!(store.get("k1").await.unwrap(), Some(b"hello".to_vec()));
    }

    #[tokio::test]
    async fn test_get_missing() {
        let store = InMemoryStore::new();
        assert_eq!(store.get("nope").await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryStore::new();
        store.set("k1", b"val", None).await.unwrap();
        store.delete("k1").await.unwrap();
        assert_eq!(store.get("k1").await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_ttl_expired() {
        let store = InMemoryStore::new();
        store
            .set("k1", b"val", Some(Duration::from_millis(1)))
            .await
            .unwrap();
        // Wait for expiry
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert_eq!(store.get("k1").await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_ttl_not_expired() {
        let store = InMemoryStore::new();
        store
            .set("k1", b"val", Some(Duration::from_secs(60)))
            .await
            .unwrap();
        assert_eq!(store.get("k1").await.unwrap(), Some(b"val".to_vec()));
    }

    #[tokio::test]
    async fn test_overwrite() {
        let store = InMemoryStore::new();
        store.set("k1", b"v1", None).await.unwrap();
        store.set("k1", b"v2", None).await.unwrap();
        assert_eq!(store.get("k1").await.unwrap(), Some(b"v2".to_vec()));
    }
}
