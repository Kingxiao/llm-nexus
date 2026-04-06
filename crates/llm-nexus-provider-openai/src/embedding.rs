//! EmbeddingProvider implementation for OpenAI-compatible APIs.

use async_trait::async_trait;
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::embedding::EmbeddingProvider;
use llm_nexus_core::types::embed::{EmbedRequest, EmbedResponse};
use llm_nexus_core::types::response::Usage;

use crate::OpenAiProvider;

#[async_trait]
impl EmbeddingProvider for OpenAiProvider {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    async fn embed(&self, request: &EmbedRequest) -> NexusResult<EmbedResponse> {
        let mut body = serde_json::json!({
            "model": request.model,
            "input": request.input,
        });

        if let Some(dims) = request.dimensions {
            body["dimensions"] = serde_json::json!(dims);
        }

        let response = self
            .client
            .post(format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
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

        let resp: serde_json::Value = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("Failed to parse embeddings response: {e}"))
        })?;

        let embeddings: Vec<Vec<f32>> = resp["data"]
            .as_array()
            .ok_or_else(|| {
                NexusError::SerializationError("missing data array in embeddings response".into())
            })?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .collect();

        let usage = Usage {
            prompt_tokens: resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: 0,
            total_tokens: resp["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32,
        };

        Ok(EmbedResponse {
            model: resp["model"].as_str().unwrap_or(&request.model).to_string(),
            embeddings,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_embed_single_input() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(server.uri(), "test-key".into());
        let request = EmbedRequest {
            model: "text-embedding-3-small".into(),
            input: vec!["hello".into()],
            dimensions: None,
        };
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.embeddings[0].len(), 3);
        assert_eq!(response.usage.prompt_tokens, 5);
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0},
                    {"embedding": [0.3, 0.4], "index": 1}
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 10, "total_tokens": 10}
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(server.uri(), "test-key".into());
        let request = EmbedRequest {
            model: "text-embedding-3-small".into(),
            input: vec!["hello".into(), "world".into()],
            dimensions: None,
        };
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings.len(), 2);
    }

    #[tokio::test]
    async fn test_embed_with_dimensions() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{"embedding": [0.1, 0.2, 0.3, 0.4], "index": 0}],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            })))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(server.uri(), "test-key".into());
        let request = EmbedRequest {
            model: "text-embedding-3-large".into(),
            input: vec!["hello".into()],
            dimensions: Some(4),
        };
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings[0].len(), 4);
    }

    #[tokio::test]
    async fn test_embed_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(429))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::with_base_url_and_key(server.uri(), "test-key".into());
        let request = EmbedRequest {
            model: "m".into(),
            input: vec!["hi".into()],
            dimensions: None,
        };
        let result = provider.embed(&request).await;
        assert!(matches!(result, Err(NexusError::RateLimited { .. })));
    }
}
