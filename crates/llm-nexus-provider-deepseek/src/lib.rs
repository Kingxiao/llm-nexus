//! DeepSeek provider for llm-nexus.
//!
//! Delegates all calls to the OpenAI-compatible adapter ([`llm_nexus_provider_openai`])
//! with DeepSeek-specific configuration. The DeepSeek API is fully compatible with
//! the OpenAI chat completions format.
//!
//! - Default base URL: `https://api.deepseek.com` (verified: 2026-04-04)
//! - Endpoint: `/chat/completions`
//!
//! # Examples
//!
//! ```rust,ignore
//! use llm_nexus_provider_deepseek::DeepSeekProvider;
//! use llm_nexus_core::types::config::ProviderConfig;
//!
//! let config = ProviderConfig { /* ... */ };
//! let provider = DeepSeekProvider::from_config(&config)?;
//! ```

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use llm_nexus_core::error::NexusResult;
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::config::ProviderConfig;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};
use llm_nexus_provider_openai::OpenAiProvider;

/// DeepSeek provider that delegates all calls to the OpenAI-compatible adapter.
pub struct DeepSeekProvider {
    inner: OpenAiProvider,
}

impl DeepSeekProvider {
    /// Create from a [`ProviderConfig`]. Reads API key from environment variable.
    pub fn from_config(config: &ProviderConfig) -> NexusResult<Self> {
        let inner = OpenAiProvider::from_config(config, "deepseek")?;
        Ok(Self { inner })
    }

    /// Create with explicit base_url and api_key (useful for testing).
    pub fn with_base_url_and_key(base_url: String, api_key: String) -> Self {
        Self {
            inner: OpenAiProvider::with_base_url_and_key(base_url, api_key)
                .with_provider_id("deepseek"),
        }
    }
}

#[async_trait]
impl ChatProvider for DeepSeekProvider {
    fn provider_id(&self) -> &str {
        "deepseek"
    }

    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        self.inner.chat(request).await
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        self.inner.chat_stream(request).await
    }

    async fn list_models(&self) -> NexusResult<Vec<String>> {
        self.inner.list_models().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_deepseek_chat() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-ds-test",
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
            })))
            .mount(&mock_server)
            .await;

        let provider =
            DeepSeekProvider::with_base_url_and_key(mock_server.uri(), "ds-test-key".into());

        let request = ChatRequest {
            model: "deepseek-chat".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from DeepSeek!");
        assert_eq!(response.usage.total_tokens, 12);
    }

    #[tokio::test]
    async fn test_deepseek_provider_id() {
        let provider =
            DeepSeekProvider::with_base_url_and_key("http://localhost".into(), "key".into());
        assert_eq!(provider.provider_id(), "deepseek");
    }
}
