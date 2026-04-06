//! Google Cloud Vertex AI provider for llm-nexus.
//!
//! Implements [`ChatProvider`](llm_nexus_core::traits::chat::ChatProvider) using the
//! Vertex AI OpenAI-compatible endpoint with OAuth2 token management.
//!
//! Zero new crypto dependencies — uses `gcloud auth print-access-token` for tokens.
//!
//! # Configuration
//!
//! Environment variables:
//! - `VERTEX_PROJECT_ID` — GCP project ID (required)
//! - `VERTEX_LOCATION` — Region (default: `us-central1`)
//! - `VERTEX_ACCESS_TOKEN` — Static access token (optional, overrides gcloud)
//!
//! # Examples
//!
//! ```rust,ignore
//! use llm_nexus_provider_vertex::VertexAiProvider;
//! use llm_nexus_core::types::config::ProviderConfig;
//!
//! # async fn run() -> llm_nexus_core::error::NexusResult<()> {
//! let config = ProviderConfig::default();
//! let provider = VertexAiProvider::from_config(&config)?;
//! # Ok(())
//! # }
//! ```

pub mod auth;

use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::config::ProviderConfig;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};
use llm_nexus_provider_openai::convert::{from_openai_response, to_openai_request};
use llm_nexus_provider_openai::stream::parse_sse_line;
use llm_nexus_provider_openai::types::OpenAiResponse;

use auth::TokenCache;

/// Vertex AI provider with OAuth2 token management.
pub struct VertexAiProvider {
    client: reqwest::Client,
    base_url: String,
    token_cache: TokenCache,
}

impl VertexAiProvider {
    /// Create from ProviderConfig + environment variables.
    ///
    /// Config `base_url` is used if set to a real URL.
    /// If `base_url` is "auto" or empty, constructs from `VERTEX_PROJECT_ID` + `VERTEX_LOCATION`.
    pub fn from_config(config: &ProviderConfig) -> NexusResult<Self> {
        let base_url = if config.base_url.is_empty() || config.base_url == "auto" {
            let project_id = std::env::var("VERTEX_PROJECT_ID").map_err(|_| {
                NexusError::AuthError("VERTEX_PROJECT_ID not set".into())
            })?;
            let location = std::env::var("VERTEX_LOCATION")
                .unwrap_or_else(|_| "us-central1".to_string());

            // Vertex AI OpenAI-compatible endpoint — verified: 2026-04-04
            format!(
                "https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi"
            )
        } else {
            config.base_url.trim_end_matches('/').to_string()
        };

        let token_cache = TokenCache::new().map_err(|e| {
            NexusError::AuthError(format!("Vertex AI token init failed: {e}"))
        })?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| NexusError::HttpError(e.to_string()))?;

        Ok(Self {
            client,
            base_url,
            token_cache,
        })
    }

    async fn get_token(&self) -> NexusResult<String> {
        self.token_cache
            .get_token()
            .await
            .map_err(NexusError::AuthError)
    }
}

#[async_trait]
impl ChatProvider for VertexAiProvider {
    fn provider_id(&self) -> &str {
        "vertex"
    }

    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        let openai_req = to_openai_request(request, false);
        let token = self.get_token().await?;
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .json(&openai_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 401 {
                return Err(NexusError::AuthError(
                    "Vertex AI: 401 Unauthorized — token may have expired".into(),
                ));
            }
            return Err(NexusError::ProviderError {
                provider: "vertex".into(),
                message: body,
                status_code: Some(status.as_u16()),
            });
        }

        let resp: OpenAiResponse = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("Failed to parse Vertex AI response: {e}"))
        })?;

        from_openai_response(resp)
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let openai_req = to_openai_request(request, true);
        let token = self.get_token().await?;
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json")
            .json(&openai_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 401 {
                return Err(NexusError::AuthError(
                    "Vertex AI: 401 Unauthorized — token may have expired".into(),
                ));
            }
            return Err(NexusError::ProviderError {
                provider: "vertex".into(),
                message: body,
                status_code: Some(status.as_u16()),
            });
        }

        let byte_stream = response.bytes_stream();
        let chunk_stream = byte_stream
            .map(|result| match result {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    let chunks: Vec<NexusResult<StreamChunk>> = text
                        .lines()
                        .filter_map(|line| match parse_sse_line(line) {
                            Ok(Some(chunk)) => Some(Ok(chunk)),
                            Ok(None) => None,
                            Err(e) => Some(Err(e)),
                        })
                        .collect();
                    stream::iter(chunks)
                }
                Err(e) => stream::iter(vec![Err(NexusError::StreamError(e.to_string()))]),
            })
            .flatten();

        Ok(Box::pin(chunk_stream))
    }

    async fn list_models(&self) -> NexusResult<Vec<String>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;
    use wiremock::matchers::{header, method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn mock_vertex_response() -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-vertex-123",
            "model": "gemini-2.5-pro",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Vertex!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        })
    }

    #[tokio::test]
    async fn test_vertex_chat_with_mock() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex("/chat/completions"))
            .and(header("Authorization", "Bearer test-vertex-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_vertex_response()))
            .mount(&server)
            .await;

        // Set static token for test
        unsafe { std::env::set_var("VERTEX_ACCESS_TOKEN", "test-vertex-token") };

        let token_cache = TokenCache::new().unwrap();
        let provider = VertexAiProvider {
            client: reqwest::Client::new(),
            base_url: server.uri(),
            token_cache,
        };

        let request = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from Vertex!");
        assert_eq!(response.usage.total_tokens, 9);

        unsafe { std::env::remove_var("VERTEX_ACCESS_TOKEN") };
    }

    #[tokio::test]
    async fn test_vertex_auth_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
            .mount(&server)
            .await;

        unsafe { std::env::set_var("VERTEX_ACCESS_TOKEN", "bad-token") };

        let token_cache = TokenCache::new().unwrap();
        let provider = VertexAiProvider {
            client: reqwest::Client::new(),
            base_url: server.uri(),
            token_cache,
        };

        let request = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let result = provider.chat(&request).await;
        assert!(matches!(result, Err(NexusError::AuthError(_))));

        unsafe { std::env::remove_var("VERTEX_ACCESS_TOKEN") };
    }
}
