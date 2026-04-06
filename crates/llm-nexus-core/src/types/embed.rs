use serde::{Deserialize, Serialize};

use super::response::Usage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_request_serialize_roundtrip() {
        let request = EmbedRequest {
            model: "text-embedding-3-small".into(),
            input: vec!["hello world".into(), "foo bar".into()],
            dimensions: Some(1536),
        };
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: EmbedRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, "text-embedding-3-small");
        assert_eq!(deserialized.input.len(), 2);
        assert_eq!(deserialized.dimensions, Some(1536));
    }

    #[test]
    fn test_embed_response_deserialize() {
        let json = r#"{
            "model": "text-embedding-3-small",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 0,
                "total_tokens": 8
            }
        }"#;
        let resp: EmbedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.embeddings.len(), 2);
        assert_eq!(resp.usage.prompt_tokens, 8);
    }
}
