//! Conversion between unified Nexus types and OpenAI API types.

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::request::{ChatRequest, Message, MessageContent, Role};
use llm_nexus_core::types::response::{ChatResponse, FinishReason, Usage};

use crate::types::{OpenAiMessage, OpenAiRequest, OpenAiResponse};

/// Convert a unified ChatRequest into an OpenAI API request.
pub fn to_openai_request(req: &ChatRequest, stream: bool) -> OpenAiRequest {
    let messages = req.messages.iter().map(convert_message).collect();

    let tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|t| serde_json::to_value(t).unwrap_or_default())
            .collect()
    });

    let response_format = req.response_format.as_ref().map(convert_response_format);

    OpenAiRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        top_p: req.top_p,
        stop: req.stop.clone(),
        tools,
        response_format,
        stream,
    }
}

use llm_nexus_core::types::request::ResponseFormat;

/// Convert ResponseFormat to OpenAI's expected JSON structure.
///
/// OpenAI's json_schema format wraps the schema in a nested object:
/// `{"type":"json_schema","json_schema":{"name":"...","schema":{...},"strict":true}}`
fn convert_response_format(rf: &ResponseFormat) -> serde_json::Value {
    match rf {
        ResponseFormat::Text => serde_json::json!({"type": "text"}),
        ResponseFormat::JsonObject => serde_json::json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            schema,
            strict,
        } => {
            let mut inner = serde_json::json!({
                "name": name,
                "schema": schema,
            });
            if let Some(s) = strict {
                inner["strict"] = serde_json::json!(s);
            }
            serde_json::json!({
                "type": "json_schema",
                "json_schema": inner,
            })
        }
    }
}

fn convert_message(msg: &Message) -> OpenAiMessage {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    let content = match &msg.content {
        MessageContent::Text(t) => serde_json::Value::String(t.clone()),
        MessageContent::Parts(parts) => {
            serde_json::to_value(parts).unwrap_or(serde_json::Value::Null)
        }
    };

    let tool_calls = msg.tool_calls.as_ref().map(|tc| {
        tc.iter()
            .map(|t| serde_json::to_value(t).unwrap_or_default())
            .collect()
    });

    OpenAiMessage {
        role: role.into(),
        content,
        name: msg.name.clone(),
        tool_calls,
        tool_call_id: msg.tool_call_id.clone(),
    }
}

/// Convert an OpenAI API response into a unified ChatResponse.
pub fn from_openai_response(resp: OpenAiResponse) -> NexusResult<ChatResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| NexusError::ProviderError {
            provider: "openai".into(),
            message: "empty choices array".into(),
            status_code: None,
        })?;

    let content = choice
        .message
        .as_ref()
        .and_then(|m| m.content.clone())
        .unwrap_or_default();

    let finish_reason = choice.finish_reason.as_deref().map(parse_finish_reason);

    let tool_calls = choice
        .message
        .as_ref()
        .and_then(|m| m.tool_calls.as_ref())
        .and_then(|tc| {
            let parsed: Vec<_> = tc
                .iter()
                .filter_map(|v| serde_json::from_value(v.clone()).ok())
                .collect();
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        });

    let usage = resp.usage.map_or_else(Usage::default, |u| Usage {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
    });

    Ok(ChatResponse {
        id: resp.id,
        model: resp.model,
        content,
        finish_reason,
        usage,
        tool_calls,
    })
}

fn parse_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::ContentPart;
    use llm_nexus_core::types::request::ImageUrl;

    #[test]
    fn test_to_openai_request() {
        let req = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::system("You are helpful"), Message::user("Hello")],
            temperature: Some(0.7),
            max_tokens: Some(1024),
            ..Default::default()
        };
        let openai_req = to_openai_request(&req, false);
        let json = serde_json::to_value(&openai_req).unwrap();
        assert_eq!(json["model"], "gpt-5.4");
        assert_eq!(json["messages"].as_array().unwrap().len(), 2);
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][1]["role"], "user");
        assert!(json["temperature"].as_f64().unwrap() - 0.7 < 0.01);
        assert_eq!(json["max_tokens"], 1024);
    }

    #[test]
    fn test_multimodal_message_conversion() {
        let req = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message {
                role: Role::User,
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "What is this?".into(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/img.png".into(),
                            detail: None,
                        },
                    },
                ]),
                ..Default::default()
            }],
            ..Default::default()
        };
        let openai_req = to_openai_request(&req, false);
        let json = serde_json::to_value(&openai_req).unwrap();
        assert!(json["messages"][0]["content"].is_array());
    }

    #[test]
    fn test_from_openai_response() {
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
        let result = from_openai_response(resp).unwrap();
        assert_eq!(result.content, "Hello!");
        assert_eq!(result.finish_reason, Some(FinishReason::Stop));
        assert_eq!(result.usage.total_tokens, 15);
    }

    #[test]
    fn test_from_openai_response_empty_choices() {
        let json = serde_json::json!({
            "id": "chatcmpl-abc",
            "model": "gpt-5.4",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        });
        let resp: OpenAiResponse = serde_json::from_value(json).unwrap();
        let result = from_openai_response(resp);
        assert!(result.is_err());
    }

    #[test]
    fn test_finish_reason_mapping() {
        assert_eq!(parse_finish_reason("stop"), FinishReason::Stop);
        assert_eq!(parse_finish_reason("length"), FinishReason::Length);
        assert_eq!(parse_finish_reason("tool_calls"), FinishReason::ToolCalls);
        assert_eq!(
            parse_finish_reason("content_filter"),
            FinishReason::ContentFilter
        );
        assert_eq!(parse_finish_reason("unknown"), FinishReason::Stop);
    }
}
