//! Gemini API serde types.
//!
//! Gemini REST API (verified: 2026-04-04):
//! - Endpoint: {base_url}/v1beta/models/{model}:generateContent
//! - Auth: API key as query parameter `?key={api_key}`
//! - System prompt: top-level `system_instruction` field
//! - Messages: `contents` array, roles are "user" / "model"

use serde::{Deserialize, Serialize};

/// Top-level request body for Gemini generateContent.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

/// A single content block (user turn, model turn, or system instruction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

/// A part within a content block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GeminiPart {
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    Text {
        text: String,
    },
    InlineData {
        inline_data: InlineData,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: FileData,
    },
}

/// A function call returned by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

/// Base64-encoded inline media.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

/// Remote file reference (Gemini downloads the URL server-side).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileData {
    pub mime_type: String,
    pub file_uri: String,
}

/// Generation parameters.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// MIME type for structured output (e.g. "application/json").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_mime_type: Option<String>,
    /// JSON Schema for structured output (Gemini's native format).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_schema: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Top-level response from Gemini generateContent.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Option<Vec<GeminiCandidate>>,
    pub usage_metadata: Option<GeminiUsage>,
}

/// A single candidate in the response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCandidate {
    pub content: Option<GeminiContent>,
    pub finish_reason: Option<String>,
}

/// Token usage metadata.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiUsage {
    pub prompt_token_count: Option<u32>,
    pub candidates_token_count: Option<u32>,
    pub total_token_count: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_request_serialize() {
        let req = GeminiRequest {
            contents: vec![GeminiContent {
                role: "user".into(),
                parts: vec![GeminiPart::Text {
                    text: "Hello".into(),
                }],
            }],
            system_instruction: None,
            generation_config: Some(GenerationConfig {
                temperature: Some(0.7),
                max_output_tokens: Some(1024),
                top_p: None,
                stop_sequences: None,
                response_mime_type: None,
                response_schema: None,
            }),
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["contents"][0]["role"], "user");
        assert!(json.get("systemInstruction").is_none());
        let temp = json["generationConfig"]["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.01);
        assert_eq!(json["generationConfig"]["maxOutputTokens"], 1024);
    }

    #[test]
    fn test_gemini_response_deserialize() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hi there!"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });
        let resp: GeminiResponse = serde_json::from_value(json).unwrap();
        let candidate = &resp.candidates.as_ref().unwrap()[0];
        let content = candidate.content.as_ref().unwrap();
        assert_eq!(content.role, "model");
        if let GeminiPart::Text { text } = &content.parts[0] {
            assert_eq!(text, "Hi there!");
        } else {
            panic!("expected Text part");
        }
        let usage = resp.usage_metadata.as_ref().unwrap();
        assert_eq!(usage.prompt_token_count, Some(10));
        assert_eq!(usage.candidates_token_count, Some(5));
    }

    #[test]
    fn test_gemini_part_inline_data() {
        let part = GeminiPart::InlineData {
            inline_data: InlineData {
                mime_type: "image/png".into(),
                data: "base64data".into(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["inline_data"]["mimeType"], "image/png");
    }

    #[test]
    fn test_gemini_response_deserialize_function_call() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "Shanghai"}
                        }
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 8,
                "totalTokenCount": 23
            }
        });
        let resp: GeminiResponse = serde_json::from_value(json).unwrap();
        let candidate = &resp.candidates.as_ref().unwrap()[0];
        let content = candidate.content.as_ref().unwrap();
        if let GeminiPart::FunctionCall { function_call } = &content.parts[0] {
            assert_eq!(function_call.name, "get_weather");
            assert_eq!(function_call.args["location"], "Shanghai");
        } else {
            panic!("expected FunctionCall part");
        }
    }

    #[test]
    fn test_gemini_response_deserialize_mixed_parts() {
        let json = serde_json::json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Calling function..."},
                        {"functionCall": {"name": "search", "args": {"q": "rust"}}}
                    ]
                },
                "finishReason": "STOP"
            }]
        });
        let resp: GeminiResponse = serde_json::from_value(json).unwrap();
        let parts = &resp.candidates.as_ref().unwrap()[0]
            .content
            .as_ref()
            .unwrap()
            .parts;
        assert_eq!(parts.len(), 2);
        assert!(matches!(&parts[0], GeminiPart::Text { .. }));
        assert!(matches!(&parts[1], GeminiPart::FunctionCall { .. }));
    }
}
