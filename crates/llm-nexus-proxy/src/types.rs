//! OpenAI-compatible request/response types for the proxy layer.
//!
//! These are thin wrappers that convert between the OpenAI HTTP API format
//! and the internal NexusClient types.

use llm_nexus::types::request::{ChatRequest, Message, ResponseFormat, ToolDefinition};
use llm_nexus::types::response::{ChatResponse, FinishReason, StreamChunk};
use serde::{Deserialize, Serialize};

// ---------- Request ----------

/// OpenAI-compatible chat completion request body.
///
/// Accepts the standard OpenAI format and converts to NexusClient's ChatRequest.
#[derive(Debug, Deserialize)]
pub struct OaiChatRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub stream: Option<bool>,
}

impl OaiChatRequest {
    /// Convert to the internal ChatRequest type.
    ///
    /// Messages are deserialized via serde_json::from_value to handle the
    /// polymorphic content field (string vs array of content parts).
    pub fn into_chat_request(self) -> Result<ChatRequest, serde_json::Error> {
        let messages: Vec<Message> = self
            .messages
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ChatRequest {
            model: self.model,
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            stop: self.stop,
            tools: self.tools,
            response_format: self.response_format,
            extra: None,
        })
    }

    pub fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

// ---------- Response ----------

/// OpenAI-compatible chat completion response.
#[derive(Debug, Serialize)]
pub struct OaiChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OaiChoice>,
    pub usage: OaiUsage,
}

#[derive(Debug, Serialize)]
pub struct OaiChoice {
    pub index: u32,
    pub message: OaiMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OaiMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct OaiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl From<ChatResponse> for OaiChatResponse {
    fn from(resp: ChatResponse) -> Self {
        let content = if resp.content.is_empty() {
            None
        } else {
            Some(resp.content)
        };

        let tool_calls = resp.tool_calls.map(|tcs| {
            tcs.into_iter()
                .map(|tc| serde_json::to_value(tc).unwrap_or_default())
                .collect()
        });

        Self {
            id: resp.id,
            object: "chat.completion",
            created: chrono::Utc::now().timestamp(),
            model: resp.model,
            choices: vec![OaiChoice {
                index: 0,
                message: OaiMessage {
                    role: "assistant",
                    content,
                    tool_calls,
                },
                finish_reason: resp.finish_reason.map(finish_reason_str),
            }],
            usage: OaiUsage {
                prompt_tokens: resp.usage.prompt_tokens,
                completion_tokens: resp.usage.completion_tokens,
                total_tokens: resp.usage.total_tokens,
            },
        }
    }
}

// ---------- Streaming ----------

/// OpenAI-compatible streaming chunk.
#[derive(Debug, Serialize)]
pub struct OaiStreamChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OaiStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OaiUsage>,
}

#[derive(Debug, Serialize)]
pub struct OaiStreamChoice {
    pub index: u32,
    pub delta: OaiDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OaiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

impl OaiStreamChunk {
    pub fn from_nexus(chunk: &StreamChunk, request_id: &str, model: &str) -> Self {
        let tool_calls = chunk
            .delta_tool_call
            .as_ref()
            .map(|tc| vec![serde_json::to_value(tc).unwrap_or_default()]);

        Self {
            id: request_id.to_owned(),
            object: "chat.completion.chunk",
            created: chrono::Utc::now().timestamp(),
            model: model.to_owned(),
            choices: vec![OaiStreamChoice {
                index: 0,
                delta: OaiDelta {
                    role: None,
                    content: chunk.delta_content.clone(),
                    tool_calls,
                },
                finish_reason: chunk
                    .finish_reason
                    .as_ref()
                    .map(|r| finish_reason_str(r.clone())),
            }],
            usage: chunk.usage.as_ref().map(|u| OaiUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        }
    }
}

// ---------- Models ----------

/// OpenAI-compatible model list response.
#[derive(Debug, Serialize)]
pub struct OaiModelList {
    pub object: &'static str,
    pub data: Vec<OaiModel>,
}

#[derive(Debug, Serialize)]
pub struct OaiModel {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

// ---------- Error ----------

/// OpenAI-compatible error response.
#[derive(Debug, Serialize)]
pub struct OaiErrorResponse {
    pub error: OaiErrorBody,
}

#[derive(Debug, Serialize)]
pub struct OaiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}

// ---------- Helpers ----------

fn finish_reason_str(reason: FinishReason) -> String {
    match reason {
        FinishReason::Stop => "stop".into(),
        FinishReason::Length => "length".into(),
        FinishReason::ToolCalls => "tool_calls".into(),
        FinishReason::ContentFilter => "content_filter".into(),
    }
}
