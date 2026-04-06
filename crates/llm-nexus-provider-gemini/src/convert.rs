//! Conversion between unified Nexus types and Gemini API types.

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::request::{
    ChatRequest, ContentPart, FunctionCall, MessageContent, Role, ToolCall,
};
use llm_nexus_core::types::response::{ChatResponse, FinishReason, Usage};

use crate::types::{
    GeminiCandidate, GeminiContent, GeminiPart, GeminiRequest, GeminiResponse, GeminiUsage,
    GenerationConfig,
};

/// Convert a unified [`ChatRequest`] into a Gemini API request body.
///
/// Key differences from OpenAI:
/// - System messages become the top-level `system_instruction` field.
/// - Role "assistant" maps to "model".
/// - Role "tool" maps to "user" (Gemini has no dedicated tool role for simple text).
pub fn to_gemini_request(req: &ChatRequest) -> GeminiRequest {
    let mut system_instruction = None;
    let mut contents = Vec::new();

    for msg in &req.messages {
        if msg.role == Role::System {
            if let MessageContent::Text(text) = &msg.content {
                system_instruction = Some(GeminiContent {
                    role: "user".into(),
                    parts: vec![GeminiPart::Text { text: text.clone() }],
                });
            }
            continue;
        }

        let role = match msg.role {
            Role::User | Role::Tool => "user",
            Role::Assistant => "model",
            Role::System => unreachable!(),
        };

        let parts = match &msg.content {
            MessageContent::Text(t) => vec![GeminiPart::Text { text: t.clone() }],
            MessageContent::Parts(parts) => convert_parts(parts),
        };

        contents.push(GeminiContent {
            role: role.into(),
            parts,
        });
    }

    let generation_config = build_generation_config(req);

    GeminiRequest {
        contents,
        system_instruction,
        generation_config,
    }
}

fn convert_parts(parts: &[ContentPart]) -> Vec<GeminiPart> {
    parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text } => Some(GeminiPart::Text { text: text.clone() }),
            ContentPart::ImageUrl { image_url } => Some(convert_image_url(&image_url.url)),
            // Audio and document types: skip unsupported content parts
            _ => None,
        })
        .collect()
}

/// Convert an image URL to the appropriate Gemini part.
///
/// - `data:image/png;base64,...` → InlineData (parsed)
/// - `https://...` → FileData (Gemini downloads server-side)
fn convert_image_url(url: &str) -> GeminiPart {
    if let Some(rest) = url.strip_prefix("data:") {
        // Parse data URI: data:{mime};base64,{data}
        if let Some((header, data)) = rest.split_once(',') {
            let mime_type = header.strip_suffix(";base64").unwrap_or(header).to_string();
            return GeminiPart::InlineData {
                inline_data: crate::types::InlineData {
                    mime_type,
                    data: data.to_string(),
                },
            };
        }
    }

    // HTTP/HTTPS URL → Gemini fileData (server-side download)
    let mime_type = guess_mime_from_url(url);
    GeminiPart::FileData {
        file_data: crate::types::FileData {
            mime_type,
            file_uri: url.to_string(),
        },
    }
}

fn guess_mime_from_url(url: &str) -> String {
    let lower = url.to_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else {
        // Default to JPEG for unknown extensions
        "image/jpeg"
    }
    .to_string()
}

fn build_generation_config(req: &ChatRequest) -> Option<GenerationConfig> {
    use llm_nexus_core::types::request::ResponseFormat;

    let (response_mime_type, response_schema) = match &req.response_format {
        Some(ResponseFormat::JsonObject) => (Some("application/json".to_string()), None),
        Some(ResponseFormat::JsonSchema { schema, .. }) => {
            (Some("application/json".to_string()), Some(schema.clone()))
        }
        _ => (None, None),
    };

    if req.temperature.is_none()
        && req.max_tokens.is_none()
        && req.top_p.is_none()
        && req.stop.is_none()
        && response_mime_type.is_none()
    {
        return None;
    }
    Some(GenerationConfig {
        temperature: req.temperature,
        max_output_tokens: req.max_tokens,
        top_p: req.top_p,
        stop_sequences: req.stop.clone(),
        response_mime_type,
        response_schema,
    })
}

/// Convert a Gemini API response into a unified [`ChatResponse`].
pub fn from_gemini_response(resp: GeminiResponse, model: &str) -> NexusResult<ChatResponse> {
    let candidate = resp
        .candidates
        .as_ref()
        .and_then(|c| c.first())
        .ok_or_else(|| NexusError::ProviderError {
            provider: "gemini".into(),
            message: "empty candidates array".into(),
            status_code: None,
        })?;

    let content = extract_text(candidate);
    let tool_calls = extract_tool_calls(candidate);
    let finish_reason = candidate
        .finish_reason
        .as_deref()
        .map(parse_gemini_finish_reason);
    let usage = resp
        .usage_metadata
        .as_ref()
        .map(convert_usage)
        .unwrap_or_default();

    Ok(ChatResponse {
        id: uuid::Uuid::new_v4().to_string(),
        model: model.to_string(),
        content,
        finish_reason,
        usage,
        tool_calls,
    })
}

fn extract_text(candidate: &GeminiCandidate) -> String {
    candidate
        .content
        .as_ref()
        .map(|c| {
            c.parts
                .iter()
                .filter_map(|p| match p {
                    GeminiPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn extract_tool_calls(candidate: &GeminiCandidate) -> Option<Vec<ToolCall>> {
    let calls: Vec<ToolCall> = candidate
        .content
        .as_ref()
        .map(|c| {
            c.parts
                .iter()
                .filter_map(|p| match p {
                    GeminiPart::FunctionCall { function_call } => {
                        let arguments = serde_json::to_string(&function_call.args)
                            .unwrap_or_else(|_| "{}".to_string());
                        Some(ToolCall {
                            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: function_call.name.clone(),
                                arguments,
                            },
                        })
                    }
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

fn parse_gemini_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "STOP" => FinishReason::Stop,
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn convert_usage(u: &GeminiUsage) -> Usage {
    let prompt = u.prompt_token_count.unwrap_or(0);
    let completion = u.candidates_token_count.unwrap_or(0);
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: u.total_token_count.unwrap_or(prompt + completion),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;

    #[test]
    fn test_system_instruction_extraction() {
        let req = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![
                Message::system("You are a helpful assistant"),
                Message::user("Hello"),
            ],
            ..Default::default()
        };
        let gemini_req = to_gemini_request(&req);

        // System message extracted to system_instruction
        assert!(gemini_req.system_instruction.is_some());
        let si = gemini_req.system_instruction.unwrap();
        assert_eq!(si.role, "user");
        if let GeminiPart::Text { text } = &si.parts[0] {
            assert_eq!(text, "You are a helpful assistant");
        } else {
            panic!("expected Text part");
        }

        // Only user message in contents (system excluded)
        assert_eq!(gemini_req.contents.len(), 1);
        assert_eq!(gemini_req.contents[0].role, "user");
    }

    #[test]
    fn test_role_mapping() {
        let req = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![
                Message::user("Hello"),
                Message::assistant("Hi there"),
                Message::user("How are you?"),
            ],
            ..Default::default()
        };
        let gemini_req = to_gemini_request(&req);

        assert_eq!(gemini_req.contents.len(), 3);
        assert_eq!(gemini_req.contents[0].role, "user");
        assert_eq!(gemini_req.contents[1].role, "model"); // assistant -> model
        assert_eq!(gemini_req.contents[2].role, "user");
        assert!(gemini_req.system_instruction.is_none());
    }

    #[test]
    fn test_from_gemini_response() {
        let resp = GeminiResponse {
            candidates: Some(vec![crate::types::GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart::Text {
                        text: "Hello!".into(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
            }]),
            usage_metadata: Some(GeminiUsage {
                prompt_token_count: Some(10),
                candidates_token_count: Some(5),
                total_token_count: Some(15),
            }),
        };

        let result = from_gemini_response(resp, "gemini-2.5-pro").unwrap();
        assert_eq!(result.content, "Hello!");
        assert_eq!(result.model, "gemini-2.5-pro");
        assert_eq!(result.finish_reason, Some(FinishReason::Stop));
        assert_eq!(result.usage.prompt_tokens, 10);
        assert_eq!(result.usage.completion_tokens, 5);
        assert_eq!(result.usage.total_tokens, 15);
    }

    #[test]
    fn test_from_gemini_response_empty_candidates() {
        let resp = GeminiResponse {
            candidates: Some(vec![]),
            usage_metadata: None,
        };
        let result = from_gemini_response(resp, "gemini-2.5-pro");
        assert!(result.is_err());
    }

    #[test]
    fn test_generation_config_built_when_params_present() {
        let req = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![Message::user("Hello")],
            temperature: Some(0.5),
            max_tokens: Some(256),
            ..Default::default()
        };
        let gemini_req = to_gemini_request(&req);
        let gc = gemini_req.generation_config.unwrap();
        assert_eq!(gc.temperature, Some(0.5));
        assert_eq!(gc.max_output_tokens, Some(256));
    }

    #[test]
    fn test_generation_config_none_when_no_params() {
        let req = ChatRequest {
            model: "gemini-2.5-pro".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let gemini_req = to_gemini_request(&req);
        assert!(gemini_req.generation_config.is_none());
    }

    #[test]
    fn test_from_gemini_response_with_tool_calls() {
        use crate::types::GeminiFunctionCall;

        let resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: "get_weather".into(),
                            args: serde_json::json!({"location": "Shanghai", "unit": "celsius"}),
                        },
                    }],
                }),
                finish_reason: Some("STOP".into()),
            }]),
            usage_metadata: Some(GeminiUsage {
                prompt_token_count: Some(20),
                candidates_token_count: Some(10),
                total_token_count: Some(30),
            }),
        };

        let result = from_gemini_response(resp, "gemini-2.5-pro").unwrap();
        assert!(result.tool_calls.is_some());
        let calls = result.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_type, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].id.starts_with("call_"));
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "Shanghai");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_from_gemini_response_mixed_text_and_tool_call() {
        use crate::types::GeminiFunctionCall;

        let resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".into(),
                    parts: vec![
                        GeminiPart::Text {
                            text: "Let me check the weather.".into(),
                        },
                        GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: "get_weather".into(),
                                args: serde_json::json!({"location": "Beijing"}),
                            },
                        },
                    ],
                }),
                finish_reason: Some("STOP".into()),
            }]),
            usage_metadata: None,
        };

        let result = from_gemini_response(resp, "gemini-2.5-pro").unwrap();
        assert_eq!(result.content, "Let me check the weather.");
        assert!(result.tool_calls.is_some());
        let calls = result.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_from_gemini_response_no_tool_calls() {
        let resp = GeminiResponse {
            candidates: Some(vec![GeminiCandidate {
                content: Some(GeminiContent {
                    role: "model".into(),
                    parts: vec![GeminiPart::Text {
                        text: "Just text.".into(),
                    }],
                }),
                finish_reason: Some("STOP".into()),
            }]),
            usage_metadata: None,
        };

        let result = from_gemini_response(resp, "gemini-2.5-pro").unwrap();
        assert!(result.tool_calls.is_none());
    }
}
