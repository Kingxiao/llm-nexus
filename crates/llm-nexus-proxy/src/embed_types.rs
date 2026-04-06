//! OpenAI-compatible embedding request/response types for the proxy layer.

use llm_nexus::types::embed::{EmbedRequest, EmbedResponse};
use serde::{Deserialize, Serialize};

/// OpenAI-compatible embedding request.
#[derive(Debug, Deserialize)]
pub struct OaiEmbedRequest {
    pub model: String,
    pub input: OaiEmbedInput,
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// Input can be a single string or array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OaiEmbedInput {
    Single(String),
    Batch(Vec<String>),
}

impl OaiEmbedRequest {
    pub fn into_embed_request(self) -> EmbedRequest {
        let input = match self.input {
            OaiEmbedInput::Single(s) => vec![s],
            OaiEmbedInput::Batch(v) => v,
        };
        EmbedRequest {
            model: self.model,
            input,
            dimensions: self.dimensions,
        }
    }
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Serialize)]
pub struct OaiEmbedResponse {
    pub object: &'static str,
    pub data: Vec<OaiEmbeddingData>,
    pub model: String,
    pub usage: OaiEmbedUsage,
}

#[derive(Debug, Serialize)]
pub struct OaiEmbeddingData {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct OaiEmbedUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

impl From<EmbedResponse> for OaiEmbedResponse {
    fn from(resp: EmbedResponse) -> Self {
        let data: Vec<OaiEmbeddingData> = resp
            .embeddings
            .into_iter()
            .enumerate()
            .map(|(i, embedding)| OaiEmbeddingData {
                object: "embedding",
                embedding,
                index: i,
            })
            .collect();

        Self {
            object: "list",
            data,
            model: resp.model,
            usage: OaiEmbedUsage {
                prompt_tokens: resp.usage.prompt_tokens,
                total_tokens: resp.usage.total_tokens,
            },
        }
    }
}
