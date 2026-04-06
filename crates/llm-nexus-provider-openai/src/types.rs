//! OpenAI-specific API types for serialization/deserialization.

use serde::{Deserialize, Serialize};

/// OpenAI chat completion request body.
#[derive(Debug, Serialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
}

/// OpenAI message format.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI chat completion response.
#[derive(Debug, Deserialize)]
pub struct OpenAiResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChoice {
    pub index: u32,
    pub message: Option<OpenAiResponseMessage>,
    pub delta: Option<OpenAiResponseMessage>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponseMessage {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_response_deserialize() {
        let json = serde_json::json!({
            "id": "chatcmpl-abc",
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let resp: OpenAiResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "chatcmpl-abc");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.as_ref().unwrap().content.as_deref(),
            Some("Hello!")
        );
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 15);
    }

    #[test]
    fn test_openai_request_serialize() {
        let req = OpenAiRequest {
            model: "gpt-5.4".into(),
            messages: vec![OpenAiMessage {
                role: "user".into(),
                content: serde_json::Value::String("hi".into()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: Some(0.7),
            max_tokens: None,
            top_p: None,
            stop: None,
            tools: None,
            response_format: None,
            stream: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "gpt-5.4");
        assert_eq!(json["messages"][0]["role"], "user");
        assert!(json.get("stream").is_none()); // false => skipped
        assert!(json.get("max_tokens").is_none()); // None => skipped
    }
}
