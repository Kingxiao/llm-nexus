//! Response caching middleware for llm-nexus.
//!
//! Caches non-streaming chat responses using a [`KeyValueStore`](llm_nexus_core::store::KeyValueStore)
//! backend. Cache key is derived from `hash(model + messages + params)`.
//!
//! # Examples
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use std::time::Duration;
//! use llm_nexus_cache::CacheMiddleware;
//! use llm_nexus_core::store::InMemoryStore;
//!
//! let store = Arc::new(InMemoryStore::new());
//! let cache = CacheMiddleware::new(store, Duration::from_secs(300));
//! ```

mod hasher;
mod middleware;
mod semantic;
mod vector_index;

pub use middleware::CacheMiddleware;
pub use semantic::{SemanticCacheConfig, SemanticCacheMiddleware};
