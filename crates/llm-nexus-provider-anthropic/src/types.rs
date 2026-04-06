//! Anthropic Messages API request/response types.
//! Reference: <https://docs.anthropic.com/en/api/messages>
//! verified: 2026-04-04

use serde::{Deserialize, Serialize};

// ── Request ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    pub stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

// ── Response ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ── Error ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

// ── Tools ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

// ── Streaming ──────────────────────────────────────────────────────

/// Raw SSE data payload — fields vary by event type, so we parse
/// selectively in `stream::parse_anthropic_event`.
#[derive(Debug, Deserialize)]
pub struct StreamEventData {
    #[serde(rename = "type")]
    pub event_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_response_deserialize() {
        let json = r#"{
            "id": "msg_01",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "Hello!"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_01");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(resp.content.len(), 1);
    }

    #[test]
    fn test_content_block_text() {
        let json = r#"{"type": "text", "text": "hi"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::Text { text } => assert_eq!(text, "hi"),
            _ => panic!("expected Text block"),
        }
    }

    #[test]
    fn test_content_block_tool_use() {
        let json = r#"{
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_weather",
            "input": {"city": "Shanghai"}
        }"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "Shanghai");
            }
            _ => panic!("expected ToolUse block"),
        }
    }

    #[test]
    fn test_anthropic_request_serialize() {
        let req = AnthropicRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: serde_json::Value::String("hi".into()),
            }],
            system: Some("You are helpful.".into()),
            max_tokens: 1024,
            temperature: Some(0.7),
            top_p: None,
            stop_sequences: None,
            tools: None,
            stream: false,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"system\":\"You are helpful.\""));
        assert!(!json.contains("top_p"));
        assert!(!json.contains("stop_sequences"));
    }

    #[test]
    fn test_anthropic_error_deserialize() {
        let json = r#"{
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "max_tokens: must be positive"
            }
        }"#;
        let err: AnthropicError = serde_json::from_str(json).unwrap();
        assert_eq!(err.error.error_type, "invalid_request_error");
        assert!(err.error.message.contains("max_tokens"));
    }
}
