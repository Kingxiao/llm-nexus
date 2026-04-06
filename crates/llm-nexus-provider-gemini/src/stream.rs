//! Gemini streaming response parser.
//!
//! Gemini streaming uses SSE format with `data:` prefixed JSON lines.
//! Each chunk is a complete GeminiResponse JSON object.

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::request::{FunctionCallDelta, ToolCallDelta};
use llm_nexus_core::types::response::{FinishReason, StreamChunk, Usage};

use crate::types::{GeminiPart, GeminiResponse};

/// Parse a single SSE `data:` line from Gemini streaming response.
///
/// Returns `Ok(Some(chunk))` for content, `Ok(None)` for empty/comment lines.
pub fn parse_gemini_sse_line(line: &str) -> NexusResult<Option<StreamChunk>> {
    let line = line.trim();

    if line.is_empty() || line.starts_with(':') {
        return Ok(None);
    }

    let data = if let Some(stripped) = line.strip_prefix("data: ") {
        stripped.trim()
    } else {
        return Ok(None);
    };

    let resp: GeminiResponse =
        serde_json::from_str(data).map_err(|e| NexusError::StreamError(e.to_string()))?;

    let candidate = resp.candidates.as_ref().and_then(|c| c.first());

    let delta_content = candidate.and_then(|c| {
        c.content.as_ref().and_then(|content| {
            let text: String = content
                .parts
                .iter()
                .filter_map(|p| match p {
                    GeminiPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if text.is_empty() { None } else { Some(text) }
        })
    });

    let finish_reason = candidate
        .and_then(|c| c.finish_reason.as_deref())
        .map(|r| match r {
            "STOP" => FinishReason::Stop,
            "MAX_TOKENS" => FinishReason::Length,
            "SAFETY" => FinishReason::ContentFilter,
            _ => FinishReason::Stop,
        });

    let usage = resp.usage_metadata.as_ref().map(|u| {
        let prompt = u.prompt_token_count.unwrap_or(0);
        let completion = u.candidates_token_count.unwrap_or(0);
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: u.total_token_count.unwrap_or(prompt + completion),
        }
    });

    // Extract function call from streaming parts (Gemini sends full function call per chunk)
    let delta_tool_call = candidate.and_then(|c| {
        c.content.as_ref().and_then(|content| {
            content.parts.iter().find_map(|p| match p {
                GeminiPart::FunctionCall { function_call } => Some(ToolCallDelta {
                    index: Some(0),
                    id: None, // Gemini doesn't provide tool call IDs in streaming
                    function: Some(FunctionCallDelta {
                        name: Some(function_call.name.clone()),
                        arguments: Some(function_call.args.to_string()),
                    }),
                }),
                _ => None,
            })
        })
    });

    Ok(Some(StreamChunk {
        delta_content,
        delta_tool_call,
        finish_reason,
        usage,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gemini_sse_content() {
        let line =
            r#"data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}"#;
        let chunk = parse_gemini_sse_line(line).unwrap().unwrap();
        assert_eq!(chunk.delta_content, Some("Hello".into()));
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_parse_gemini_sse_with_finish() {
        let line = r#"data: {"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}"#;
        let chunk = parse_gemini_sse_line(line).unwrap().unwrap();
        assert_eq!(chunk.delta_content, Some("!".into()));
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.total_tokens, 8);
    }

    #[test]
    fn test_parse_gemini_sse_empty_line() {
        let chunk = parse_gemini_sse_line("").unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_parse_gemini_sse_comment() {
        let chunk = parse_gemini_sse_line(": keep-alive").unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_parse_gemini_sse_invalid_json() {
        let result = parse_gemini_sse_line("data: {invalid}");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_gemini_sse_function_call() {
        let line = r#"data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Tokyo"}}}],"role":"model"}}]}"#;
        let chunk = parse_gemini_sse_line(line).unwrap().unwrap();
        assert!(chunk.delta_content.is_none());
        let tc = chunk.delta_tool_call.unwrap();
        assert_eq!(tc.index, Some(0));
        let func = tc.function.unwrap();
        assert_eq!(func.name.as_deref(), Some("get_weather"));
        assert!(func.arguments.as_deref().unwrap().contains("Tokyo"));
    }
}
