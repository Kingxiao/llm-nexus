//! Conversion between unified Nexus types and Anthropic-specific types.

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::types::request::{
    ChatRequest, FunctionCall, Message, MessageContent, Role, ToolCall,
};
use llm_nexus_core::types::response::{ChatResponse, FinishReason, Usage};

use crate::types::{
    AnthropicMessage, AnthropicRequest, AnthropicResponse, AnthropicTool, ContentBlock,
};

/// Convert a unified `ChatRequest` into an `AnthropicRequest`.
///
/// Key difference: Anthropic expects system prompts as a top-level `system`
/// field, not inside the `messages` array. This function extracts all
/// `Role::System` messages and concatenates them.
pub fn to_anthropic_request(req: &ChatRequest) -> AnthropicRequest {
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<AnthropicMessage> = Vec::new();

    for msg in &req.messages {
        if msg.role == Role::System {
            if let Some(text) = extract_text(&msg.content) {
                system_parts.push(text);
            }
            continue;
        }
        messages.push(convert_message(msg));
    }

    // Anthropic doesn't have native json_schema response_format.
    // Inject schema constraint into system prompt as best-effort.
    if let Some(llm_nexus_core::types::request::ResponseFormat::JsonSchema {
        name, schema, ..
    }) = &req.response_format
    {
        system_parts.push(format!(
            "You MUST respond with valid JSON matching this schema (name: {name}):\n```json\n{}\n```\nRespond ONLY with the JSON object, no other text.",
            serde_json::to_string_pretty(schema).unwrap_or_default()
        ));
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    let tools = req.tools.as_ref().map(|tool_defs| {
        tool_defs
            .iter()
            .map(|td| AnthropicTool {
                name: td.function.name.clone(),
                description: td.function.description.clone(),
                input_schema: td
                    .function
                    .parameters
                    .clone()
                    .unwrap_or(serde_json::json!({"type": "object", "properties": {}})),
            })
            .collect()
    });

    AnthropicRequest {
        model: req.model.clone(),
        messages,
        system,
        max_tokens: req.max_tokens.unwrap_or(4096),
        temperature: req.temperature,
        top_p: req.top_p,
        stop_sequences: req.stop.clone(),
        tools,
        stream: false,
    }
}

/// Convert an `AnthropicResponse` into a unified `ChatResponse`.
pub fn from_anthropic_response(resp: AnthropicResponse) -> NexusResult<ChatResponse> {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in &resp.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: input.to_string(),
                    },
                });
            }
        }
    }

    let content = text_parts.join("");
    let finish_reason = resp.stop_reason.as_deref().map(map_stop_reason);
    let tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    Ok(ChatResponse {
        id: resp.id,
        model: resp.model,
        content,
        finish_reason,
        usage: Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        },
        tool_calls,
    })
}

/// Map Anthropic stop_reason strings to the unified `FinishReason`.
pub fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}

fn convert_message(msg: &Message) -> AnthropicMessage {
    let role = match msg.role {
        Role::User | Role::Tool => "user",
        Role::Assistant => "assistant",
        Role::System => unreachable!("system messages filtered before conversion"),
    };

    let content = match &msg.content {
        MessageContent::Text(text) => serde_json::Value::String(text.clone()),
        MessageContent::Parts(parts) => {
            let blocks: Vec<serde_json::Value> = parts
                .iter()
                .map(|p| serde_json::to_value(p).unwrap_or_default())
                .collect();
            serde_json::Value::Array(blocks)
        }
    };

    AnthropicMessage {
        role: role.into(),
        content,
    }
}

fn extract_text(content: &MessageContent) -> Option<String> {
    match content {
        MessageContent::Text(text) => Some(text.clone()),
        MessageContent::Parts(parts) => {
            let texts: Vec<String> = parts
                .iter()
                .filter_map(|p| match p {
                    llm_nexus_core::types::request::ContentPart::Text { text } => {
                        Some(text.clone())
                    }
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;

    #[test]
    fn test_system_prompt_extraction() {
        let req = ChatRequest {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![Message::system("You are helpful."), Message::user("Hello")],
            max_tokens: Some(1024),
            ..Default::default()
        };

        let anthropic_req = to_anthropic_request(&req);

        assert_eq!(anthropic_req.system.as_deref(), Some("You are helpful."));
        assert_eq!(anthropic_req.messages.len(), 1);
        assert_eq!(anthropic_req.messages[0].role, "user");
    }

    #[test]
    fn test_multiple_system_messages_concatenated() {
        let req = ChatRequest {
            model: "test".into(),
            messages: vec![
                Message::system("Rule 1"),
                Message::system("Rule 2"),
                Message::user("Hi"),
            ],
            ..Default::default()
        };

        let anthropic_req = to_anthropic_request(&req);
        assert_eq!(anthropic_req.system.as_deref(), Some("Rule 1\nRule 2"));
        assert_eq!(anthropic_req.messages.len(), 1);
    }

    #[test]
    fn test_no_system_message() {
        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Hi")],
            ..Default::default()
        };

        let anthropic_req = to_anthropic_request(&req);
        assert!(anthropic_req.system.is_none());
    }

    #[test]
    fn test_from_anthropic_response() {
        let resp = AnthropicResponse {
            id: "msg_01".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![ContentBlock::Text {
                text: "Hello!".into(),
            }],
            stop_reason: Some("end_turn".into()),
            usage: crate::types::AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let chat_resp = from_anthropic_response(resp).unwrap();
        assert_eq!(chat_resp.id, "msg_01");
        assert_eq!(chat_resp.content, "Hello!");
        assert_eq!(chat_resp.finish_reason, Some(FinishReason::Stop));
        assert_eq!(chat_resp.usage.prompt_tokens, 10);
        assert_eq!(chat_resp.usage.completion_tokens, 5);
        assert_eq!(chat_resp.usage.total_tokens, 15);
    }

    #[test]
    fn test_multiple_content_blocks_joined() {
        let resp = AnthropicResponse {
            id: "msg_02".into(),
            model: "test".into(),
            content: vec![
                ContentBlock::Text {
                    text: "Part 1".into(),
                },
                ContentBlock::Text {
                    text: "Part 2".into(),
                },
            ],
            stop_reason: None,
            usage: crate::types::AnthropicUsage {
                input_tokens: 5,
                output_tokens: 10,
            },
        };

        let chat_resp = from_anthropic_response(resp).unwrap();
        assert_eq!(chat_resp.content, "Part 1Part 2");
    }

    #[test]
    fn test_tool_use_block_mapped_to_tool_calls() {
        let resp = AnthropicResponse {
            id: "msg_03".into(),
            model: "test".into(),
            content: vec![
                ContentBlock::Text {
                    text: "Let me check.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_01".into(),
                    name: "search".into(),
                    input: serde_json::json!({"q": "test"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: crate::types::AnthropicUsage {
                input_tokens: 8,
                output_tokens: 12,
            },
        };

        let chat_resp = from_anthropic_response(resp).unwrap();
        assert_eq!(chat_resp.content, "Let me check.");
        assert_eq!(chat_resp.finish_reason, Some(FinishReason::ToolCalls));

        let tool_calls = chat_resp.tool_calls.expect("tool_calls should be Some");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "toolu_01");
        assert_eq!(tool_calls[0].call_type, "function");
        assert_eq!(tool_calls[0].function.name, "search");
        let args: serde_json::Value =
            serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["q"], "test");
    }

    #[test]
    fn test_multiple_tool_use_blocks() {
        let resp = AnthropicResponse {
            id: "msg_04".into(),
            model: "test".into(),
            content: vec![
                ContentBlock::ToolUse {
                    id: "toolu_01".into(),
                    name: "search".into(),
                    input: serde_json::json!({"q": "rust"}),
                },
                ContentBlock::ToolUse {
                    id: "toolu_02".into(),
                    name: "calculate".into(),
                    input: serde_json::json!({"expr": "1+1"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: crate::types::AnthropicUsage {
                input_tokens: 10,
                output_tokens: 20,
            },
        };

        let chat_resp = from_anthropic_response(resp).unwrap();
        assert_eq!(chat_resp.content, "");
        let tool_calls = chat_resp.tool_calls.expect("tool_calls should be Some");
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].function.name, "search");
        assert_eq!(tool_calls[1].function.name, "calculate");
    }

    #[test]
    fn test_no_tool_use_returns_none() {
        let resp = AnthropicResponse {
            id: "msg_05".into(),
            model: "test".into(),
            content: vec![ContentBlock::Text {
                text: "Just text.".into(),
            }],
            stop_reason: Some("end_turn".into()),
            usage: crate::types::AnthropicUsage {
                input_tokens: 5,
                output_tokens: 3,
            },
        };

        let chat_resp = from_anthropic_response(resp).unwrap();
        assert!(chat_resp.tool_calls.is_none());
    }

    #[test]
    fn test_map_stop_reason_variants() {
        assert_eq!(map_stop_reason("end_turn"), FinishReason::Stop);
        assert_eq!(map_stop_reason("stop_sequence"), FinishReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), FinishReason::Length);
        assert_eq!(map_stop_reason("tool_use"), FinishReason::ToolCalls);
        assert_eq!(map_stop_reason("unknown"), FinishReason::Stop);
    }

    #[test]
    fn test_tools_converted_to_anthropic_format() {
        use llm_nexus_core::types::request::{FunctionDefinition, ToolDefinition};

        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Use the tool")],
            tools: Some(vec![ToolDefinition {
                tool_type: "function".into(),
                function: FunctionDefinition {
                    name: "get_weather".into(),
                    description: Some("Get the weather for a city".into()),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        },
                        "required": ["city"]
                    })),
                },
            }]),
            ..Default::default()
        };

        let anthropic_req = to_anthropic_request(&req);
        let tools = anthropic_req.tools.expect("tools should be Some");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(
            tools[0].description.as_deref(),
            Some("Get the weather for a city")
        );
        assert_eq!(tools[0].input_schema["type"], "object");
        assert!(tools[0].input_schema["properties"]["city"].is_object());
    }

    #[test]
    fn test_no_tools_remains_none() {
        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Hi")],
            tools: None,
            ..Default::default()
        };
        let anthropic_req = to_anthropic_request(&req);
        assert!(anthropic_req.tools.is_none());
    }

    #[test]
    fn test_default_max_tokens() {
        let req = ChatRequest {
            model: "test".into(),
            messages: vec![Message::user("Hi")],
            max_tokens: None,
            ..Default::default()
        };
        let anthropic_req = to_anthropic_request(&req);
        assert_eq!(anthropic_req.max_tokens, 4096);
    }
}
