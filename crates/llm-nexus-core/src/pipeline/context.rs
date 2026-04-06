//! Per-request context that flows through the middleware pipeline.
//!
//! Each middleware can read/write typed data via the [`Extensions`] map,
//! enabling loose coupling between middleware (e.g. Identity, BudgetRef).

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::time::Instant;

use crate::types::model::ModelMetadata;

/// Per-request context flowing through the middleware pipeline.
///
/// Created once per `chat()`/`chat_stream()` call. Middleware can store
/// arbitrary typed data in [`extensions`](Self::extensions) without
/// modifying this struct.
pub struct RequestContext {
    /// Unique request identifier.
    pub request_id: String,
    /// Resolved model metadata (set by the pipeline before dispatch).
    pub model_meta: Option<ModelMetadata>,
    /// Resolved provider ID.
    pub provider_id: Option<String>,
    /// When the request entered the pipeline.
    pub start_time: Instant,
    /// Type-safe key-value storage for middleware-specific data.
    extensions: Extensions,
}

impl RequestContext {
    /// Create a new context for a request.
    pub fn new(request_id: String) -> Self {
        Self {
            request_id,
            model_meta: None,
            provider_id: None,
            start_time: Instant::now(),
            extensions: Extensions::new(),
        }
    }

    /// Insert a typed value into the extensions map.
    pub fn insert<T: Send + Sync + 'static>(&mut self, val: T) {
        self.extensions.insert(val);
    }

    /// Get a reference to a typed value from extensions.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.extensions.get::<T>()
    }

    /// Get a mutable reference to a typed value from extensions.
    pub fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.extensions.get_mut::<T>()
    }

    /// Elapsed time since the request entered the pipeline.
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }
}

/// Type-safe map for storing arbitrary data, keyed by [`TypeId`].
///
/// Similar to `http::Extensions` but simplified for our use case.
struct Extensions {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Extensions {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn insert<T: Send + Sync + 'static>(&mut self, val: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(val));
    }

    fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|v| v.downcast_ref())
    }

    fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|v| v.downcast_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extensions_insert_and_get() {
        let mut ctx = RequestContext::new("req-1".into());
        ctx.insert(42u32);
        ctx.insert("hello".to_string());

        assert_eq!(ctx.get::<u32>(), Some(&42));
        assert_eq!(ctx.get::<String>(), Some(&"hello".to_string()));
        assert_eq!(ctx.get::<f64>(), None);
    }

    #[test]
    fn test_extensions_get_mut() {
        let mut ctx = RequestContext::new("req-2".into());
        ctx.insert(10u64);

        if let Some(val) = ctx.get_mut::<u64>() {
            *val += 5;
        }
        assert_eq!(ctx.get::<u64>(), Some(&15));
    }

    #[test]
    fn test_context_elapsed() {
        let ctx = RequestContext::new("req-3".into());
        // Just verify it doesn't panic and returns a reasonable value
        assert!(ctx.elapsed_ms() < 1000);
    }

    #[test]
    fn test_context_fields() {
        let mut ctx = RequestContext::new("req-4".into());
        assert_eq!(ctx.request_id, "req-4");
        assert!(ctx.model_meta.is_none());
        assert!(ctx.provider_id.is_none());

        ctx.provider_id = Some("openai".into());
        assert_eq!(ctx.provider_id.as_deref(), Some("openai"));
    }
}
