//! Conversion between NexusClient types and Bedrock Converse API types.

use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, InferenceConfiguration, Message as BedrockMessage,
    SystemContentBlock,
};

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::request::{ChatRequest, FunctionCall, MessageContent, Role, ToolCall};
use llm_nexus_core::types::response::{ChatResponse, FinishReason, Usage};

/// Convert a NexusClient ChatRequest into Bedrock Converse API inputs.
///
/// Returns: (messages, optional system prompt, optional inference config)
pub fn to_converse_input(
    request: &ChatRequest,
) -> NexusResult<(
    Vec<BedrockMessage>,
    Option<SystemContentBlock>,
    Option<InferenceConfiguration>,
)> {
    let mut messages = Vec::new();
    let mut system_prompt: Option<SystemContentBlock> = None;

    for msg in &request.messages {
        match msg.role {
            Role::System => {
                let text = extract_text(&msg.content);
                system_prompt = Some(SystemContentBlock::Text(text));
            }
            Role::User => {
                let text = extract_text(&msg.content);
                messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(text))
                        .build()
                        .map_err(|e| {
                            NexusError::SerializationError(format!(
                                "failed to build user message: {e}"
                            ))
                        })?,
                );
            }
            Role::Assistant => {
                let text = extract_text(&msg.content);
                messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::Assistant)
                        .content(ContentBlock::Text(text))
                        .build()
                        .map_err(|e| {
                            NexusError::SerializationError(format!(
                                "failed to build assistant message: {e}"
                            ))
                        })?,
                );
            }
            Role::Tool => {
                // Tool results go as user messages with tool_result content blocks
                // For now, send as plain text user message
                let text = extract_text(&msg.content);
                messages.push(
                    BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(text))
                        .build()
                        .map_err(|e| {
                            NexusError::SerializationError(format!(
                                "failed to build tool message: {e}"
                            ))
                        })?,
                );
            }
        }
    }

    // Build inference config
    let mut config_builder = InferenceConfiguration::builder();
    let mut has_config = false;

    if let Some(temp) = request.temperature {
        config_builder = config_builder.temperature(temp);
        has_config = true;
    }
    if let Some(max_tokens) = request.max_tokens {
        config_builder = config_builder.max_tokens(max_tokens as i32);
        has_config = true;
    }
    if let Some(top_p) = request.top_p {
        config_builder = config_builder.top_p(top_p);
        has_config = true;
    }
    if let Some(ref stops) = request.stop {
        for s in stops {
            config_builder = config_builder.stop_sequences(s.clone());
        }
        has_config = true;
    }

    let inference_config = if has_config {
        Some(config_builder.build())
    } else {
        None
    };

    Ok((messages, system_prompt, inference_config))
}

/// Convert Bedrock Converse output to NexusClient ChatResponse.
pub fn from_converse_output(
    output: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
    model: &str,
) -> NexusResult<ChatResponse> {
    // Extract text from output content blocks
    let content = output
        .output()
        .and_then(|o| {
            if let aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) = o {
                Some(
                    msg.content()
                        .iter()
                        .filter_map(|block| {
                            if let ContentBlock::Text(text) = block {
                                Some(text.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                )
            } else {
                None
            }
        })
        .unwrap_or_default();

    let finish_reason = Some(match output.stop_reason() {
        aws_sdk_bedrockruntime::types::StopReason::EndTurn => FinishReason::Stop,
        aws_sdk_bedrockruntime::types::StopReason::MaxTokens => FinishReason::Length,
        aws_sdk_bedrockruntime::types::StopReason::StopSequence => FinishReason::Stop,
        aws_sdk_bedrockruntime::types::StopReason::ToolUse => FinishReason::ToolCalls,
        aws_sdk_bedrockruntime::types::StopReason::ContentFiltered => FinishReason::ContentFilter,
        aws_sdk_bedrockruntime::types::StopReason::GuardrailIntervened => {
            FinishReason::ContentFilter
        }
        _ => FinishReason::Stop,
    });

    let usage = output
        .usage()
        .map(|u| {
            let input = u.input_tokens() as u32;
            let output = u.output_tokens() as u32;
            Usage {
                prompt_tokens: input,
                completion_tokens: output,
                total_tokens: input + output,
            }
        })
        .unwrap_or_default();

    // Extract tool_use content blocks
    let tool_calls: Option<Vec<ToolCall>> = output.output().and_then(|o| {
        if let aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) = o {
            let calls: Vec<ToolCall> = msg
                .content()
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::ToolUse(tu) = block {
                        let input_doc = tu.input();
                        let arguments =
                            serde_json::to_string(&document_to_json(input_doc)).unwrap_or_default();
                        Some(ToolCall {
                            id: tu.tool_use_id().to_string(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: tu.name().to_string(),
                                arguments,
                            },
                        })
                    } else {
                        None
                    }
                })
                .collect();
            if calls.is_empty() {
                None
            } else {
                Some(calls)
            }
        } else {
            None
        }
    });

    Ok(ChatResponse {
        id: format!("bedrock-{}", uuid::Uuid::new_v4()),
        model: model.to_string(),
        content,
        finish_reason,
        usage,
        tool_calls,
    })
}

fn extract_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(s) => s.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                llm_nexus_core::types::request::ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// Convert AWS Smithy Document to serde_json::Value.
fn document_to_json(doc: &aws_smithy_types::Document) -> serde_json::Value {
    match doc {
        aws_smithy_types::Document::Object(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), document_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        aws_smithy_types::Document::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(document_to_json).collect())
        }
        aws_smithy_types::Document::Number(n) => {
            if let Ok(i) = i64::try_from(*n) {
                serde_json::Value::Number(i.into())
            } else {
                let f: f64 = n.to_f64_lossy();
                serde_json::Number::from_f64(f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }
        }
        aws_smithy_types::Document::String(s) => serde_json::Value::String(s.clone()),
        aws_smithy_types::Document::Bool(b) => serde_json::Value::Bool(*b),
        aws_smithy_types::Document::Null => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_sdk_bedrockruntime::types::{
        ContentBlock, ConverseOutput as ConverseOutputType, StopReason, TokenUsage, ToolUseBlock,
    };
    use llm_nexus_core::types::request::Message;

    #[test]
    fn test_to_converse_input_basic_messages() {
        let request = ChatRequest {
            model: "test-model".into(),
            messages: vec![
                Message::system("You are helpful"),
                Message::user("Hello"),
                Message::assistant("Hi there"),
            ],
            ..Default::default()
        };

        let (messages, system, config) = to_converse_input(&request).unwrap();
        assert_eq!(messages.len(), 2); // user + assistant (system is separate)
        assert!(system.is_some());
        assert!(config.is_none());
    }

    #[test]
    fn test_to_converse_input_with_inference_config() {
        let request = ChatRequest {
            model: "test-model".into(),
            messages: vec![Message::user("Hi")],
            temperature: Some(0.7),
            max_tokens: Some(1000),
            top_p: Some(0.9),
            stop: Some(vec!["END".into()]),
            ..Default::default()
        };

        let (messages, _, config) = to_converse_input(&request).unwrap();
        assert_eq!(messages.len(), 1);
        assert!(config.is_some());
    }

    #[test]
    fn test_from_converse_output_text() {
        let output = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
            .output(ConverseOutputType::Message(
                aws_sdk_bedrockruntime::types::Message::builder()
                    .role(aws_sdk_bedrockruntime::types::ConversationRole::Assistant)
                    .content(ContentBlock::Text("Hello from Bedrock!".into()))
                    .build()
                    .unwrap(),
            ))
            .stop_reason(StopReason::EndTurn)
            .usage(
                TokenUsage::builder()
                    .input_tokens(10)
                    .output_tokens(5)
                    .total_tokens(15)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        let response = from_converse_output(output, "test-model").unwrap();
        assert_eq!(response.content, "Hello from Bedrock!");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 5);
        assert!(response.tool_calls.is_none());
    }

    #[test]
    fn test_from_converse_output_tool_use() {
        let tool_input = aws_smithy_types::Document::Object({
            let mut map = std::collections::HashMap::new();
            map.insert(
                "location".to_string(),
                aws_smithy_types::Document::String("Tokyo".to_string()),
            );
            map
        });

        let output = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
            .output(ConverseOutputType::Message(
                aws_sdk_bedrockruntime::types::Message::builder()
                    .role(aws_sdk_bedrockruntime::types::ConversationRole::Assistant)
                    .content(ContentBlock::ToolUse(
                        ToolUseBlock::builder()
                            .tool_use_id("call_123")
                            .name("get_weather")
                            .input(tool_input)
                            .build()
                            .unwrap(),
                    ))
                    .build()
                    .unwrap(),
            ))
            .stop_reason(StopReason::ToolUse)
            .usage(
                TokenUsage::builder()
                    .input_tokens(20)
                    .output_tokens(10)
                    .total_tokens(30)
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();

        let response = from_converse_output(output, "test-model").unwrap();
        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        let tool_calls = response.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_123");
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert!(tool_calls[0].function.arguments.contains("Tokyo"));
    }

    #[test]
    fn test_from_converse_output_stop_reasons() {
        let make_output = |reason: StopReason| {
            aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
                .output(ConverseOutputType::Message(
                    aws_sdk_bedrockruntime::types::Message::builder()
                        .role(aws_sdk_bedrockruntime::types::ConversationRole::Assistant)
                        .content(ContentBlock::Text("ok".into()))
                        .build()
                        .unwrap(),
                ))
                .stop_reason(reason)
                .usage(
                    TokenUsage::builder()
                        .input_tokens(1)
                        .output_tokens(1)
                        .total_tokens(2)
                        .build()
                        .unwrap(),
                )
                .build()
                .unwrap()
        };

        assert_eq!(
            from_converse_output(make_output(StopReason::EndTurn), "m")
                .unwrap()
                .finish_reason,
            Some(FinishReason::Stop)
        );
        assert_eq!(
            from_converse_output(make_output(StopReason::MaxTokens), "m")
                .unwrap()
                .finish_reason,
            Some(FinishReason::Length)
        );
        assert_eq!(
            from_converse_output(make_output(StopReason::ContentFiltered), "m")
                .unwrap()
                .finish_reason,
            Some(FinishReason::ContentFilter)
        );
    }

    #[test]
    fn test_document_to_json_nested() {
        let doc = aws_smithy_types::Document::Object({
            let mut map = std::collections::HashMap::new();
            map.insert(
                "key".to_string(),
                aws_smithy_types::Document::String("value".to_string()),
            );
            map.insert(
                "num".to_string(),
                aws_smithy_types::Document::Number(aws_smithy_types::Number::PosInt(42)),
            );
            map.insert("flag".to_string(), aws_smithy_types::Document::Bool(true));
            map
        });

        let json = document_to_json(&doc);
        assert_eq!(json["key"], "value");
        assert_eq!(json["num"], 42);
        assert_eq!(json["flag"], true);
    }
}
