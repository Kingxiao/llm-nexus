//! Anthropic SSE stream parser.
//!
//! Anthropic streaming uses these event types:
//! - `message_start` — initial message metadata
//! - `content_block_start` — new content block begins
//! - `content_block_delta` — incremental text/tool delta
//! - `content_block_stop` — content block ends
//! - `message_delta` — final stop_reason + usage
//! - `message_stop` — stream complete
//! - `ping` — keep-alive

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::types::request::{FunctionCallDelta, ToolCallDelta};
use llm_nexus_core::types::response::{FinishReason, StreamChunk, Usage};

use crate::convert::map_stop_reason;

/// Parse a single SSE event into an optional `StreamChunk`.
///
/// Returns `Ok(None)` when the stream is finished (`message_stop`) or
/// the event type is unrecognized.
pub fn parse_anthropic_event(event_type: &str, data: &str) -> NexusResult<Option<StreamChunk>> {
    match event_type {
        "content_block_start" => parse_content_block_start(data),
        "content_block_delta" => parse_content_block_delta(data),
        "message_delta" => parse_message_delta(data),
        "message_stop" => Ok(None),
        "message_start" | "content_block_stop" | "ping" => Ok(Some(StreamChunk {
            delta_content: None,
            delta_tool_call: None,
            finish_reason: None,
            usage: None,
        })),
        _ => Ok(None),
    }
}

fn parse_content_block_start(data: &str) -> NexusResult<Option<StreamChunk>> {
    let parsed: serde_json::Value = serde_json::from_str(data)?;
    let block = &parsed["content_block"];

    // Anthropic sends content_block_start with type "tool_use" containing id + name
    if block["type"].as_str() == Some("tool_use") {
        let id = block["id"].as_str().map(|s| s.to_string());
        let name = block["name"].as_str().map(|s| s.to_string());
        let index = parsed["index"].as_u64().map(|i| i as u32);

        return Ok(Some(StreamChunk {
            delta_content: None,
            delta_tool_call: Some(ToolCallDelta {
                index,
                id,
                function: Some(FunctionCallDelta {
                    name,
                    arguments: None,
                }),
            }),
            finish_reason: None,
            usage: None,
        }));
    }

    Ok(Some(StreamChunk {
        delta_content: None,
        delta_tool_call: None,
        finish_reason: None,
        usage: None,
    }))
}

fn parse_content_block_delta(data: &str) -> NexusResult<Option<StreamChunk>> {
    let parsed: serde_json::Value = serde_json::from_str(data)?;
    let delta = &parsed["delta"];

    // Text delta
    if delta["type"].as_str() == Some("text_delta") {
        let text = delta["text"].as_str().map(|s| s.to_string());
        return Ok(Some(StreamChunk {
            delta_content: text,
            delta_tool_call: None,
            finish_reason: None,
            usage: None,
        }));
    }

    // Tool use input JSON delta
    if delta["type"].as_str() == Some("input_json_delta") {
        let partial_json = delta["partial_json"].as_str().map(|s| s.to_string());
        let index = parsed["index"].as_u64().map(|i| i as u32);
        return Ok(Some(StreamChunk {
            delta_content: None,
            delta_tool_call: Some(ToolCallDelta {
                index,
                id: None,
                function: Some(FunctionCallDelta {
                    name: None,
                    arguments: partial_json,
                }),
            }),
            finish_reason: None,
            usage: None,
        }));
    }

    // Fallback: try plain text
    let text = delta["text"].as_str().map(|s| s.to_string());
    Ok(Some(StreamChunk {
        delta_content: text,
        delta_tool_call: None,
        finish_reason: None,
        usage: None,
    }))
}

fn parse_message_delta(data: &str) -> NexusResult<Option<StreamChunk>> {
    let parsed: serde_json::Value = serde_json::from_str(data)?;

    let finish_reason: Option<FinishReason> =
        parsed["delta"]["stop_reason"].as_str().map(map_stop_reason);

    let usage: Option<Usage> = parsed["usage"].as_object().map(|u| {
        let output = u.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        Usage {
            prompt_tokens: 0,
            completion_tokens: output,
            total_tokens: output,
        }
    });

    Ok(Some(StreamChunk {
        delta_content: None,
        delta_tool_call: None,
        finish_reason,
        usage,
    }))
}

/// Parse raw SSE bytes into (event_type, data) pairs.
///
/// SSE format:
/// ```text
/// event: content_block_delta
/// data: {"type":"content_block_delta",...}
///
/// ```
pub fn parse_sse_lines(buf: &str) -> Vec<(String, String)> {
    let mut events = Vec::new();
    let mut current_event = String::new();
    let mut current_data = String::new();

    for line in buf.lines() {
        if let Some(value) = line.strip_prefix("event: ") {
            current_event = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("data: ") {
            current_data = value.to_string();
        } else if line.is_empty() && !current_event.is_empty() {
            events.push((
                std::mem::take(&mut current_event),
                std::mem::take(&mut current_data),
            ));
        }
    }

    // Handle trailing event without final blank line
    if !current_event.is_empty() {
        events.push((current_event, current_data));
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_content_block_delta() {
        let data =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let chunk = parse_anthropic_event("content_block_delta", data)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.delta_content.as_deref(), Some("Hi"));
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_parse_message_delta_with_stop_reason() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}"#;
        let chunk = parse_anthropic_event("message_delta", data)
            .unwrap()
            .unwrap();
        assert!(chunk.delta_content.is_none());
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.completion_tokens, 42);
        assert_eq!(usage.total_tokens, 42);
    }

    #[test]
    fn test_parse_message_stop_returns_none() {
        let result = parse_anthropic_event("message_stop", "{}").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_ping_returns_empty_chunk() {
        let chunk = parse_anthropic_event("ping", "{}").unwrap().unwrap();
        assert!(chunk.delta_content.is_none());
        assert!(chunk.finish_reason.is_none());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_parse_message_start_returns_empty_chunk() {
        let chunk = parse_anthropic_event("message_start", "{}")
            .unwrap()
            .unwrap();
        assert!(chunk.delta_content.is_none());
    }

    #[test]
    fn test_parse_unknown_event_returns_none() {
        let result = parse_anthropic_event("unknown_event", "{}").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_sse_lines() {
        let raw = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"Hi\"}}\n\n";
        let events = parse_sse_lines(raw);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "message_start");
        assert_eq!(events[1].0, "content_block_delta");
    }

    #[test]
    fn test_parse_sse_lines_trailing_event() {
        let raw = "event: ping\ndata: {}";
        let events = parse_sse_lines(raw);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "ping");
    }

    #[test]
    fn test_parse_message_delta_max_tokens() {
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":100}}"#;
        let chunk = parse_anthropic_event("message_delta", data)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.finish_reason, Some(FinishReason::Length));
    }

    #[test]
    fn test_parse_content_block_start_tool_use() {
        let data = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{}}}"#;
        let chunk = parse_anthropic_event("content_block_start", data)
            .unwrap()
            .unwrap();
        let tc = chunk.delta_tool_call.unwrap();
        assert_eq!(tc.index, Some(1));
        assert_eq!(tc.id.as_deref(), Some("toolu_123"));
        let func = tc.function.unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
    }

    #[test]
    fn test_parse_content_block_delta_input_json() {
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"San"}}"#;
        let chunk = parse_anthropic_event("content_block_delta", data)
            .unwrap()
            .unwrap();
        assert!(chunk.delta_content.is_none());
        let tc = chunk.delta_tool_call.unwrap();
        assert_eq!(tc.index, Some(1));
        let func = tc.function.unwrap();
        assert_eq!(func.arguments.as_deref(), Some("{\"location\":\"San"));
    }

    #[test]
    fn test_parse_content_block_start_text() {
        let data =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let chunk = parse_anthropic_event("content_block_start", data)
            .unwrap()
            .unwrap();
        assert!(chunk.delta_tool_call.is_none());
        assert!(chunk.delta_content.is_none());
    }
}
