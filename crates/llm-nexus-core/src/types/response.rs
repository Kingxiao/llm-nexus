use serde::{Deserialize, Serialize};

use super::request::{ToolCall, ToolCallDelta};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub content: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_tool_call: Option<ToolCallDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_response_deserialize() {
        let json = r#"{
            "id": "resp-1",
            "model": "test-model",
            "content": "Hello!",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "resp-1");
        assert_eq!(resp.model, "test-model");
        assert_eq!(resp.content, "Hello!");
        assert_eq!(resp.finish_reason, Some(FinishReason::Stop));
        assert_eq!(resp.usage.total_tokens, 15);
        assert!(resp.tool_calls.is_none());
    }

    #[test]
    fn test_usage_default() {
        let usage = Usage::default();
        assert_eq!(usage.prompt_tokens, 0);
        assert_eq!(usage.completion_tokens, 0);
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn test_finish_reason_serialization() {
        let reason = FinishReason::ToolCalls;
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, r#""tool_calls""#);

        let deserialized: FinishReason = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, FinishReason::ToolCalls);
    }
}
