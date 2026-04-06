use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Default for Message {
    fn default() -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(String::new()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(text.into()),
            ..Default::default()
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(text.into()),
            ..Default::default()
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(text.into()),
            ..Default::default()
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(text.into()),
            tool_call_id: Some(tool_call_id.into()),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Audio content (URL or base64 data URI).
    #[serde(rename = "input_audio")]
    AudioUrl { input_audio: AudioContent },
    /// Document/file content (base64 or URL).
    #[serde(rename = "file")]
    Document { file: DocumentContent },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Audio content for multimodal requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContent {
    /// Base64-encoded audio data or URL.
    pub data: String,
    /// Audio format (e.g. "wav", "mp3", "ogg").
    pub format: String,
}

/// Document/file content for multimodal requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentContent {
    /// Base64-encoded file data or URL.
    pub data: String,
    /// MIME type (e.g. "application/pdf", "text/plain").
    pub mime_type: String,
    /// Optional filename.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    /// Structured output with a JSON Schema constraint.
    /// Providers that support it will enforce the schema on generation.
    #[serde(rename = "json_schema")]
    JsonSchema {
        /// Schema name (used by some providers for caching).
        name: String,
        /// The JSON Schema definition.
        schema: serde_json::Value,
        /// Whether to enforce strict schema adherence.
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_request_serialize_roundtrip() {
        let request = ChatRequest {
            model: "test-model".into(),
            messages: vec![Message::user("hello")],
            temperature: Some(0.7),
            ..Default::default()
        };
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, "test-model");
        assert_eq!(deserialized.messages.len(), 1);
        assert_eq!(deserialized.temperature, Some(0.7));
        assert!(deserialized.max_tokens.is_none());
    }

    #[test]
    fn test_message_convenience_constructors() {
        let user = Message::user("hi");
        assert_eq!(user.role, Role::User);

        let system = Message::system("you are helpful");
        assert_eq!(system.role, Role::System);

        let assistant = Message::assistant("sure");
        assert_eq!(assistant.role, Role::Assistant);

        let tool = Message::tool("call_123", r#"{"result": 42}"#);
        assert_eq!(tool.role, Role::Tool);
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_123"));
    }

    #[test]
    fn test_multimodal_message_serialize() {
        let msg = Message {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "What is this?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/img.png".into(),
                        detail: Some("high".into()),
                    },
                },
            ]),
            ..Default::default()
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("image_url"));
        assert!(json.contains("What is this?"));

        let roundtrip: Message = serde_json::from_str(&json).unwrap();
        if let MessageContent::Parts(parts) = &roundtrip.content {
            assert_eq!(parts.len(), 2);
        } else {
            panic!("expected Parts variant");
        }
    }

    #[test]
    fn test_optional_fields_skipped() {
        let request = ChatRequest {
            model: "m".into(),
            messages: vec![],
            ..Default::default()
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(!json.contains("temperature"));
        assert!(!json.contains("max_tokens"));
        assert!(!json.contains("tools"));
    }
}
