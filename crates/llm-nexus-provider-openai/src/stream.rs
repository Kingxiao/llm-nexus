//! OpenAI SSE stream parsing.

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::request::{FunctionCallDelta, ToolCallDelta};
use llm_nexus_core::types::response::{FinishReason, StreamChunk, Usage};

/// Parse a single SSE `data:` line from OpenAI streaming response.
///
/// Returns `Ok(Some(chunk))` for content, `Ok(None)` for stream-end or empty lines.
pub fn parse_sse_line(line: &str) -> NexusResult<Option<StreamChunk>> {
    let line = line.trim();

    if line.is_empty() || line.starts_with(':') {
        return Ok(None);
    }

    let data = if let Some(stripped) = line.strip_prefix("data: ") {
        stripped.trim()
    } else {
        return Ok(None);
    };

    if data == "[DONE]" {
        return Ok(None);
    }

    let parsed: serde_json::Value =
        serde_json::from_str(data).map_err(|e| NexusError::StreamError(e.to_string()))?;

    let choice = parsed["choices"].as_array().and_then(|c| c.first());

    let Some(choice) = choice else {
        // usage-only chunk (OpenAI sends usage in final chunk with empty choices)
        let usage = parse_usage(&parsed);
        return Ok(Some(StreamChunk {
            delta_content: None,
            delta_tool_call: None,
            finish_reason: None,
            usage,
        }));
    };

    let delta_content = choice["delta"]["content"].as_str().map(|s| s.to_string());

    let finish_reason = choice["finish_reason"].as_str().map(|r| match r {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    });

    let usage = parse_usage(&parsed);

    // Parse streaming tool_calls from delta
    let delta_tool_call = choice["delta"]["tool_calls"]
        .as_array()
        .and_then(|tcs| tcs.first())
        .map(|tc| ToolCallDelta {
            index: tc["index"].as_u64().map(|i| i as u32),
            id: tc["id"].as_str().map(|s| s.to_string()),
            function: Some(FunctionCallDelta {
                name: tc["function"]["name"].as_str().map(|s| s.to_string()),
                arguments: tc["function"]["arguments"].as_str().map(|s| s.to_string()),
            }),
        });

    Ok(Some(StreamChunk {
        delta_content,
        delta_tool_call,
        finish_reason,
        usage,
    }))
}

fn parse_usage(parsed: &serde_json::Value) -> Option<Usage> {
    let u = parsed.get("usage")?;
    if u.is_null() {
        return None;
    }
    Some(Usage {
        prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_content_delta() {
        let line = r#"data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hello"}}]}"#;
        let chunk = parse_sse_line(line).unwrap().unwrap();
        assert_eq!(chunk.delta_content, Some("Hello".into()));
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_parse_sse_done() {
        let line = "data: [DONE]";
        let chunk = parse_sse_line(line).unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_parse_sse_empty_line() {
        let chunk = parse_sse_line("").unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_parse_sse_comment() {
        let chunk = parse_sse_line(": keep-alive").unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_parse_sse_finish_reason() {
        let line = r#"data: {"id":"chatcmpl-abc","choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let chunk = parse_sse_line(line).unwrap().unwrap();
        assert!(chunk.delta_content.is_none());
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_parse_sse_with_usage() {
        let line = r#"data: {"id":"chatcmpl-abc","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let chunk = parse_sse_line(line).unwrap().unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_parse_sse_invalid_json() {
        let line = "data: {invalid json}";
        let result = parse_sse_line(line);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_sse_tool_call_start() {
        let line = r#"data: {"id":"chatcmpl-abc","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#;
        let chunk = parse_sse_line(line).unwrap().unwrap();
        let tc = chunk.delta_tool_call.unwrap();
        assert_eq!(tc.index, Some(0));
        assert_eq!(tc.id.as_deref(), Some("call_123"));
        let func = tc.function.unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert_eq!(func.arguments.as_deref(), Some(""));
    }

    #[test]
    fn test_parse_sse_tool_call_arguments_delta() {
        let line = r#"data: {"id":"chatcmpl-abc","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}"#;
        let chunk = parse_sse_line(line).unwrap().unwrap();
        let tc = chunk.delta_tool_call.unwrap();
        assert_eq!(tc.index, Some(0));
        assert!(tc.id.is_none()); // only sent on first chunk
        let func = tc.function.unwrap();
        assert!(func.name.is_none());
        assert_eq!(func.arguments.as_deref(), Some("{\"loc"));
    }
}
