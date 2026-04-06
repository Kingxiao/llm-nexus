//! `ChatProvider` trait implementation for Anthropic.

use std::pin::Pin;

use futures::stream::{self, Stream, StreamExt};
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};

use crate::AnthropicProvider;
use crate::convert::{from_anthropic_response, to_anthropic_request};
use crate::stream::{parse_anthropic_event, parse_sse_lines};
use crate::types::AnthropicResponse;

#[async_trait::async_trait]
impl llm_nexus_core::traits::chat::ChatProvider for AnthropicProvider {
    fn provider_id(&self) -> &str {
        "anthropic"
    }

    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        let anthropic_req = to_anthropic_request(request);

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json")
            .json(&anthropic_req)
            .send()
            .await
            .map_err(|e| NexusError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body = response.text().await.unwrap_or_default();

            if status_code == 429 {
                return Err(NexusError::RateLimited {
                    retry_after_ms: None,
                });
            }

            return Err(NexusError::ProviderError {
                provider: "anthropic".into(),
                message: body,
                status_code: Some(status_code),
            });
        }

        let resp: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| NexusError::SerializationError(e.to_string()))?;

        from_anthropic_response(resp)
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let mut anthropic_req = to_anthropic_request(request);
        anthropic_req.stream = true;

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json")
            .json(&anthropic_req)
            .send()
            .await
            .map_err(|e| NexusError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();

            if status_code == 429 {
                return Err(NexusError::RateLimited {
                    retry_after_ms: None,
                });
            }

            return Err(NexusError::ProviderError {
                provider: "anthropic".into(),
                message: body,
                status_code: Some(status_code),
            });
        }

        let byte_stream = response.bytes_stream();

        let chunk_stream = byte_stream
            .scan(String::new(), |buffer, chunk_result| {
                let chunk_result = chunk_result.map_err(|e| NexusError::StreamError(e.to_string()));

                match chunk_result {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&text);

                        let mut results: Vec<NexusResult<StreamChunk>> = Vec::new();

                        // Process complete SSE events (terminated by double newline)
                        while let Some(pos) = buffer.find("\n\n") {
                            let event_block = buffer[..pos].to_string();
                            *buffer = buffer[pos + 2..].to_string();

                            let events = parse_sse_lines(&event_block);
                            for (event_type, data) in events {
                                match parse_anthropic_event(&event_type, &data) {
                                    Ok(Some(stream_chunk)) => {
                                        results.push(Ok(stream_chunk));
                                    }
                                    Ok(None) => {
                                        // message_stop or unrecognized — skip
                                    }
                                    Err(e) => {
                                        results.push(Err(e));
                                    }
                                }
                            }
                        }

                        futures::future::ready(Some(results))
                    }
                    Err(e) => futures::future::ready(Some(vec![Err(e)])),
                }
            })
            .flat_map(stream::iter);

        Ok(Box::pin(chunk_stream))
    }

    async fn list_models(&self) -> NexusResult<Vec<String>> {
        // Anthropic does not expose a public models list API.
        // Model discovery is handled by llm-nexus-registry.
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use llm_nexus_core::traits::chat::ChatProvider;
    use llm_nexus_core::types::request::Message;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn mock_anthropic_response() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_mock_01",
            "type": "message",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "Hello from mock!"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8
            }
        })
    }

    #[tokio::test]
    async fn test_chat_with_mock() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_anthropic_response()))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url_and_key(server.uri(), "test-key".into());

        let req = ChatRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![Message::system("Be concise."), Message::user("Hi")],
            max_tokens: Some(256),
            ..Default::default()
        };

        let resp = provider.chat(&req).await.unwrap();
        assert_eq!(resp.id, "msg_mock_01");
        assert_eq!(resp.content, "Hello from mock!");
        assert_eq!(resp.usage.prompt_tokens, 12);
        assert_eq!(resp.usage.completion_tokens, 8);
        assert_eq!(resp.usage.total_tokens, 20);
    }

    #[tokio::test]
    async fn test_chat_rate_limited() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "Too many requests"}
            })))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url_and_key(server.uri(), "test-key".into());

        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };

        let err = provider.chat(&req).await.unwrap_err();
        assert!(matches!(err, NexusError::RateLimited { .. }));
    }

    #[tokio::test]
    async fn test_chat_error_response() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url_and_key(server.uri(), "test-key".into());

        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };

        let err = provider.chat(&req).await.unwrap_err();
        match err {
            NexusError::ProviderError {
                provider,
                status_code,
                ..
            } => {
                assert_eq!(provider, "anthropic");
                assert_eq!(status_code, Some(500));
            }
            other => panic!("expected ProviderError, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_auth_header_sent() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "secret-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_anthropic_response()))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url_and_key(server.uri(), "secret-123".into());

        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };

        // If the header doesn't match, wiremock returns 404
        let result = provider.chat(&req).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_provider_id() {
        let provider = AnthropicProvider::with_base_url_and_key("http://unused".into(), "k".into());
        assert_eq!(provider.provider_id(), "anthropic");
    }

    #[tokio::test]
    async fn test_list_models_empty() {
        let provider = AnthropicProvider::with_base_url_and_key("http://unused".into(), "k".into());
        let models = provider.list_models().await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_chat_stream_with_mock() {
        use futures::StreamExt;

        let server = MockServer::start().await;

        let sse_body = "\
event: message_start\n\
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s01\",\"model\":\"claude-sonnet-4-20250514\"}}\n\
\n\
event: content_block_start\n\
data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\
\n\
event: content_block_delta\n\
data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\
\n\
event: content_block_stop\n\
data: {\"type\":\"content_block_stop\",\"index\":0}\n\
\n\
event: message_delta\n\
data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\
\n\
event: message_stop\n\
data: {\"type\":\"message_stop\"}\n\
\n";

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(sse_body)
                    .insert_header("content-type", "text/event-stream"),
            )
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url_and_key(server.uri(), "test-key".into());

        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };

        let mut stream = provider.chat_stream(&req).await.unwrap();
        let mut collected_text = String::new();
        let mut got_finish = false;

        while let Some(result) = stream.next().await {
            let chunk = result.unwrap();
            if let Some(text) = chunk.delta_content {
                collected_text.push_str(&text);
            }
            if chunk.finish_reason.is_some() {
                got_finish = true;
            }
        }

        assert_eq!(collected_text, "Hi there");
        assert!(got_finish);
    }
}
