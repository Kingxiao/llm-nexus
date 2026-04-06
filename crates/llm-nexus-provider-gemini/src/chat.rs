//! ChatProvider implementation for the Gemini API.

use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, StreamChunk};

use crate::GeminiProvider;
use crate::convert::{from_gemini_response, to_gemini_request};
use crate::stream::parse_gemini_sse_line;
use crate::types::GeminiResponse;

#[async_trait]
impl ChatProvider for GeminiProvider {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    async fn chat(&self, request: &ChatRequest) -> NexusResult<ChatResponse> {
        let gemini_req = to_gemini_request(request);

        // Gemini uses model name in URL path, API key as query param
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, request.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&gemini_req)
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
                provider: "gemini".into(),
                message: body,
                status_code: Some(status.as_u16()),
            });
        }

        let resp: GeminiResponse = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("Failed to parse Gemini response: {e}"))
        })?;

        from_gemini_response(resp, &request.model)
    }

    async fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let gemini_req = to_gemini_request(request);

        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, request.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&gemini_req)
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
                provider: "gemini".into(),
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
                        .filter_map(|line| match parse_gemini_sse_line(line) {
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
        let url = format!("{}/v1beta/models?key={}", self.base_url, self.api_key);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Ok(vec![]);
        }

        let body: serde_json::Value = response.json().await.unwrap_or_default();
        let models = body["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        m["name"]
                            .as_str()
                            .and_then(|s| s.strip_prefix("models/"))
                            .map(|s| s.to_string())
                    })
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
    use wiremock::matchers::{method, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_gemini_chat_mock() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(query_param("key", "test-gemini-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello from Gemini!"}]
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 12
                }
            })))
            .mount(&mock_server)
            .await;

        let provider =
            GeminiProvider::with_base_url_and_key(mock_server.uri(), "test-gemini-key".into());

        let request = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let response = provider.chat(&request).await.unwrap();
        assert_eq!(response.content, "Hello from Gemini!");
        assert_eq!(response.usage.prompt_tokens, 8);
        assert_eq!(response.usage.completion_tokens, 4);
        assert_eq!(response.usage.total_tokens, 12);
    }

    #[tokio::test]
    async fn test_gemini_auth_query_param() {
        let mock_server = MockServer::start().await;

        // Only match if key is in query params (not header)
        Mock::given(method("POST"))
            .and(query_param("key", "my-secret-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "ok"}]
                    },
                    "finishReason": "STOP"
                }]
            })))
            .mount(&mock_server)
            .await;

        let provider =
            GeminiProvider::with_base_url_and_key(mock_server.uri(), "my-secret-key".into());

        let request = ChatRequest {
            model: "gemini-2.5-flash".into(),
            messages: vec![Message::user("test")],
            ..Default::default()
        };
        let response = provider.chat(&request).await;
        assert!(response.is_ok(), "API key should be passed as query param");
    }

    #[tokio::test]
    async fn test_gemini_error() {
        let mock_server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_string(r#"{"error":{"message":"Invalid model"}}"#),
            )
            .mount(&mock_server)
            .await;

        let provider = GeminiProvider::with_base_url_and_key(mock_server.uri(), "test-key".into());

        let request = ChatRequest {
            model: "nonexistent-model".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let result = provider.chat(&request).await;
        assert!(matches!(
            result,
            Err(NexusError::ProviderError {
                status_code: Some(400),
                ..
            })
        ));
    }
}
