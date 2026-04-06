//! Runtime provider registry — maps provider IDs to ChatProvider implementations.

use std::collections::HashMap;
use std::sync::Arc;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::chat::ChatProvider;

/// Thread-safe map of provider_id -> ChatProvider implementation.
pub struct ProviderMap {
    providers: HashMap<String, Arc<dyn ChatProvider>>,
}

impl ProviderMap {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider. Overwrites any existing provider with the same ID.
    pub fn register(&mut self, id: impl Into<String>, provider: Arc<dyn ChatProvider>) {
        self.providers.insert(id.into(), provider);
    }

    /// Get a provider by ID.
    pub fn get(&self, id: &str) -> NexusResult<Arc<dyn ChatProvider>> {
        self.providers
            .get(id)
            .cloned()
            .ok_or_else(|| NexusError::ConfigError(format!("Provider '{}' not registered", id)))
    }

    /// Check if a provider is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.providers.contains_key(id)
    }

    /// List all registered provider IDs.
    pub fn provider_ids(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Resolve which provider handles a given model ID by checking the registry.
    pub fn resolve_provider_for_model(
        &self,
        _model_id: &str,
        model_provider: &str,
    ) -> NexusResult<Arc<dyn ChatProvider>> {
        self.get(model_provider)
    }
}

impl Default for ProviderMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::ChatRequest;
    use llm_nexus_core::types::response::{ChatResponse, StreamChunk, Usage};
    use std::pin::Pin;

    struct MockProvider {
        id: String,
    }

    #[async_trait::async_trait]
    impl ChatProvider for MockProvider {
        fn provider_id(&self) -> &str {
            &self.id
        }
        async fn chat(&self, _request: &ChatRequest) -> NexusResult<ChatResponse> {
            Ok(ChatResponse {
                id: "mock".into(),
                model: "mock-model".into(),
                content: "mock response".into(),
                finish_reason: None,
                usage: Usage::default(),
                tool_calls: None,
            })
        }
        async fn chat_stream(
            &self,
            _request: &ChatRequest,
        ) -> NexusResult<Pin<Box<dyn futures::Stream<Item = NexusResult<StreamChunk>> + Send>>>
        {
            Ok(Box::pin(futures::stream::empty()))
        }
        async fn list_models(&self) -> NexusResult<Vec<String>> {
            Ok(vec!["mock-model".into()])
        }
    }

    #[test]
    fn test_register_and_get() {
        let mut map = ProviderMap::new();
        map.register("test", Arc::new(MockProvider { id: "test".into() }));
        assert!(map.contains("test"));
        assert!(!map.contains("other"));
        let provider = map.get("test").unwrap();
        assert_eq!(provider.provider_id(), "test");
    }

    #[test]
    fn test_get_missing_provider() {
        let map = ProviderMap::new();
        assert!(map.get("nonexistent").is_err());
    }

    #[test]
    fn test_provider_ids() {
        let mut map = ProviderMap::new();
        map.register("a", Arc::new(MockProvider { id: "a".into() }));
        map.register("b", Arc::new(MockProvider { id: "b".into() }));
        let ids = map.provider_ids();
        assert_eq!(ids.len(), 2);
    }
}
