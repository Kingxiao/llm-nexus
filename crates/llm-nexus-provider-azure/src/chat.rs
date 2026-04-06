//! ChatProvider implementation for Azure OpenAI.

use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};
use llm_nexus_provider_openai::convert::{from_openai_response, to_openai_request};
use llm_nexus_provider_openai::stream::parse_sse_line;
use llm_nexus_provider_openai::types::OpenAiResponse;

use crate::AzureOpenAiProvider;

#[async_trait]
impl ChatProvider for AzureOpenAiProvider {
    fn provider_id(&self) -> &str {
        "azure_openai"
    }

    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        let openai_req = to_openai_request(request, false);
        let url = self.chat_url(&request.model);

        let req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&openai_req);
        let req = self.apply_auth(req);

        let response = req.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(NexusError::RateLimited {
                    retry_after_ms: None,
                });
            }
            return Err(NexusError::ProviderError {
                provider: "azure_openai".into(),
                message: body,
                status_code: Some(status.as_u16()),
            });
        }

        let resp: OpenAiResponse = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("Failed to parse Azure response: {e}"))
        })?;

        from_openai_response(resp)
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let openai_req = to_openai_request(request, true);
        let url = self.chat_url(&request.model);

        let req = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&openai_req);
        let req = self.apply_auth(req);

        let response = req.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(NexusError::ProviderError {
                provider: "azure_openai".into(),
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
        // Azure doesn't have a standard model listing endpoint
        // Return empty — models are known from the registry
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AzureOpenAiProvider;
    use llm_nexus_core::types::request::Message;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn mock_chat_response() -> serde_json::Value {
        serde_json::json!({
            "id": "chatcmpl-azure-123",
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Azure!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        })
    }

    #[tokio::test]
    async fn test_v1_endpoint_with_api_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/chat/completions"))
            .and(header("api-key", "test-azure-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_chat_response()))
            .mount(&server)
            .await;

        let provider = AzureOpenAiProvider::new(server.uri(), "test-azure-key", false, None);

        let request = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from Azure!");
        assert_eq!(response.usage.total_tokens, 9);
    }

    #[tokio::test]
    async fn test_classic_deployment_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/my-gpt5/chat/completions"))
            .and(query_param("api-version", "2024-10-21"))
            .and(header("api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_chat_response()))
            .mount(&server)
            .await;

        let provider =
            AzureOpenAiProvider::new(server.uri(), "test-key", false, Some("2024-10-21".into()))
                .with_default_deployment("my-gpt5");

        let request = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from Azure!");
    }

    #[tokio::test]
    async fn test_bearer_auth_mode() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/chat/completions"))
            .and(header("Authorization", "Bearer aad-token-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_chat_response()))
            .mount(&server)
            .await;

        let provider = AzureOpenAiProvider::new(server.uri(), "aad-token-123", true, None);

        let request = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from Azure!");
    }

    #[tokio::test]
    async fn test_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429))
            .mount(&server)
            .await;

        let provider = AzureOpenAiProvider::new(server.uri(), "key", false, None);

        let request = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };
        let result = provider.chat(&request).await;
        assert!(matches!(result, Err(NexusError::RateLimited { .. })));
    }

    #[tokio::test]
    async fn test_classic_endpoint_model_as_deployment() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/openai/deployments/gpt-5.4/chat/completions"))
            .and(query_param("api-version", "2024-10-21"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_chat_response()))
            .mount(&server)
            .await;

        let provider =
            AzureOpenAiProvider::new(server.uri(), "key", false, Some("2024-10-21".into()));

        let request = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from Azure!");
    }
}
