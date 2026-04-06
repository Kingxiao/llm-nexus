//! EmbeddingProvider implementation for Gemini.
//!
//! Gemini embedding endpoint (verified: 2026-04-04):
//! POST {base_url}/v1beta/models/{model}:embedContent?key={api_key}
//! Batch: POST {base_url}/v1beta/models/{model}:batchEmbedContents?key={api_key}

use async_trait::async_trait;
use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::traits::embedding::EmbeddingProvider;
use llm_nexus_core::types::embed::{EmbedRequest, EmbedResponse};
use llm_nexus_core::types::response::Usage;

use crate::GeminiProvider;

#[async_trait]
impl EmbeddingProvider for GeminiProvider {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    async fn embed(&self, request: &EmbedRequest) -> NexusResult<EmbedResponse> {
        if request.input.len() == 1 {
            return self.embed_single(request).await;
        }
        self.embed_batch(request).await
    }

    fn max_batch_size(&self) -> usize {
        100 // Gemini batchEmbedContents limit
    }
}

impl GeminiProvider {
    async fn embed_single(&self, request: &EmbedRequest) -> NexusResult<EmbedResponse> {
        let url = format!(
            "{}/v1beta/models/{}:embedContent?key={}",
            self.base_url, request.model, self.api_key
        );

        let mut body = serde_json::json!({
            "content": {
                "parts": [{"text": &request.input[0]}]
            }
        });

        if let Some(dims) = request.dimensions {
            body["outputDimensionality"] = serde_json::json!(dims);
        }

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(NexusError::ProviderError {
                provider: "gemini".into(),
                message: body,
                status_code: Some(status.as_u16()),
            });
        }

        let resp: serde_json::Value = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("failed to parse Gemini embedding: {e}"))
        })?;

        let values = resp["embedding"]["values"]
            .as_array()
            .ok_or_else(|| {
                NexusError::SerializationError("missing embedding.values in response".into())
            })?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<f32>>();

        Ok(EmbedResponse {
            model: request.model.clone(),
            embeddings: vec![values],
            usage: Usage {
                prompt_tokens: 0, // Gemini embedContent doesn't return token counts
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }

    async fn embed_batch(&self, request: &EmbedRequest) -> NexusResult<EmbedResponse> {
        let url = format!(
            "{}/v1beta/models/{}:batchEmbedContents?key={}",
            self.base_url, request.model, self.api_key
        );

        let requests: Vec<serde_json::Value> = request
            .input
            .iter()
            .map(|text| {
                let mut req = serde_json::json!({
                    "model": format!("models/{}", request.model),
                    "content": {
                        "parts": [{"text": text}]
                    }
                });
                if let Some(dims) = request.dimensions {
                    req["outputDimensionality"] = serde_json::json!(dims);
                }
                req
            })
            .collect();

        let body = serde_json::json!({ "requests": requests });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(NexusError::ProviderError {
                provider: "gemini".into(),
                message: body,
                status_code: Some(status.as_u16()),
            });
        }

        let resp: serde_json::Value = response.json().await.map_err(|e| {
            NexusError::SerializationError(format!("failed to parse Gemini batch embedding: {e}"))
        })?;

        let embeddings: Vec<Vec<f32>> = resp["embeddings"]
            .as_array()
            .ok_or_else(|| {
                NexusError::SerializationError("missing embeddings array in response".into())
            })?
            .iter()
            .map(|emb| {
                emb["values"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .collect();

        Ok(EmbedResponse {
            model: request.model.clone(),
            embeddings,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_embed_single() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/text-embedding-004:embedContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embedding": {
                    "values": [0.1, 0.2, 0.3]
                }
            })))
            .mount(&server)
            .await;

        let provider = GeminiProvider::with_base_url_and_key(server.uri(), "test-key".into());
        let request = EmbedRequest {
            model: "text-embedding-004".into(),
            input: vec!["hello".into()],
            dimensions: None,
        };
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings.len(), 1);
        assert_eq!(response.embeddings[0].len(), 3);
    }

    #[tokio::test]
    async fn test_embed_batch() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/text-embedding-004:batchEmbedContents",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [
                    {"values": [0.1, 0.2]},
                    {"values": [0.3, 0.4]}
                ]
            })))
            .mount(&server)
            .await;

        let provider = GeminiProvider::with_base_url_and_key(server.uri(), "test-key".into());
        let request = EmbedRequest {
            model: "text-embedding-004".into(),
            input: vec!["hello".into(), "world".into()],
            dimensions: None,
        };
        let response = provider.embed(&request).await.unwrap();
        assert_eq!(response.embeddings.len(), 2);
    }

    #[tokio::test]
    async fn test_embed_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/bad-model:embedContent"))
            .respond_with(ResponseTemplate::new(404).set_body_string("model not found"))
            .mount(&server)
            .await;

        let provider = GeminiProvider::with_base_url_and_key(server.uri(), "test-key".into());
        let request = EmbedRequest {
            model: "bad-model".into(),
            input: vec!["hello".into()],
            dimensions: None,
        };
        let result = provider.embed(&request).await;
        assert!(matches!(
            result,
            Err(NexusError::ProviderError {
                status_code: Some(404),
                ..
            })
        ));
    }
}
