//! ChatProvider implementation for OpenAI-compatible APIs.

use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};

use crate::OpenAiProvider;
use crate::convert::{from_openai_response, to_openai_request};
use crate::stream::parse_sse_line;
use crate::types::OpenAiResponse;

#[async_trait]
impl ChatProvider for OpenAiProvider {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        let openai_req = to_openai_request(request, false);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(NexusError::RateLimited {
                    retry_after_ms: None,
                });
            }
            return Err(NexusError::ProviderError {
                provider: self.provider_id.clone(),
                message: body,
                status_code: Some(status.as_u16()),
            });
        }

        let resp: OpenAiResponse = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("Failed to parse OpenAI response: {e}"))
        })?;

        from_openai_response(resp)
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let openai_req = to_openai_request(request, true);

        let response = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                return Err(NexusError::RateLimited {
                    retry_after_ms: None,
                });
            }
            return Err(NexusError::ProviderError {
                provider: self.provider_id.clone(),
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
        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let body: serde_json::Value = response.json().await.unwrap_or_default();
        let models = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_chat_with_mock() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-test",
                "model": "gpt-5.4",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            })))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(mock_server.uri(), "test-key".into());

        let request = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hi there!");
        assert_eq!(response.usage.total_tokens, 8);
    }

    #[tokio::test]
    async fn test_chat_rate_limited() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(mock_server.uri(), "test-key".into());

        let request = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let result = provider.chat(&request).await;
        assert!(matches!(result, Err(NexusError::RateLimited { .. })));
    }

    #[tokio::test]
    async fn test_chat_server_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal server error"))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(mock_server.uri(), "test-key".into());

        let request = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let result = provider.chat(&request).await;
        assert!(matches!(
            result,
            Err(NexusError::ProviderError {
                status_code: Some(500),
                ..
            })
        ));
    }

    #[tokio::test]
    async fn test_list_models_with_mock() {
        let mock_server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"id": "gpt-5.4", "object": "model"},
                    {"id": "gpt-5.4-mini", "object": "model"}
                ]
            })))
            .mount(&mock_server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(mock_server.uri(), "test-key".into());

        let models = provider.list_models().await.unwrap();
        assert_eq!(models.len(), 2);
        assert!(models.contains(&"gpt-5.4".to_string()));
    }
}
