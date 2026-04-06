//! Built-in guardrail checks.

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::types::request::{ChatRequest, MessageContent};
use llm_nexus_core::types::response::ChatResponse;

use crate::middleware::{GuardrailCheck, GuardrailVerdict};

/// Blocks requests/responses containing any of the specified keywords.
pub struct KeywordFilter {
    blocked_keywords: Vec<String>,
}

impl KeywordFilter {
    pub fn new(keywords: Vec<String>) -> Self {
        Self {
            blocked_keywords: keywords.into_iter().map(|k| k.to_lowercase()).collect(),
        }
    }

    fn contains_keyword(&self, text: &str) -> Option<&str> {
        let lower = text.to_lowercase();
        self.blocked_keywords
            .iter()
            .find(|kw| lower.contains(kw.as_str()))
            .map(|s| s.as_str())
    }
}

#[async_trait::async_trait]
impl GuardrailCheck for KeywordFilter {
    async fn check_request(&self, request: &ChatRequest) -> NexusResult<GuardrailVerdict> {
        for msg in &request.messages {
            let text = match &msg.content {
                MessageContent::Text(t) => t.clone(),
                MessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| match p {
                        llm_nexus_core::types::request::ContentPart::Text { text } => {
                            Some(text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            };
            if let Some(kw) = self.contains_keyword(&text) {
                return Ok(GuardrailVerdict::Block {
                    reason: format!("blocked keyword: {kw}"),
                });
            }
        }
        Ok(GuardrailVerdict::Allow)
    }

    async fn check_response(&self, response: &ChatResponse) -> NexusResult<GuardrailVerdict> {
        if let Some(kw) = self.contains_keyword(&response.content) {
            return Ok(GuardrailVerdict::Block {
                reason: format!("response contains blocked keyword: {kw}"),
            });
        }
        Ok(GuardrailVerdict::Allow)
    }
}

/// Blocks requests/responses matching a regex pattern.
pub struct RegexFilter {
    patterns: Vec<regex_lite::Regex>,
}

impl RegexFilter {
    /// Create from pattern strings. Invalid patterns are logged and skipped.
    pub fn new(patterns: Vec<String>) -> Self {
        Self {
            patterns: patterns
                .into_iter()
                .filter_map(|p| match regex_lite::Regex::new(&p) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        tracing::warn!(pattern = %p, error = %e, "guardrail regex pattern failed to compile, skipping");
                        None
                    }
                })
                .collect(),
        }
    }

    fn matches(&self, text: &str) -> Option<String> {
        self.patterns
            .iter()
            .find(|p| p.is_match(text))
            .map(|p| p.as_str().to_string())
    }
}

#[async_trait::async_trait]
impl GuardrailCheck for RegexFilter {
    async fn check_request(&self, request: &ChatRequest) -> NexusResult<GuardrailVerdict> {
        for msg in &request.messages {
            let text = match &msg.content {
                MessageContent::Text(t) => t.as_str(),
                _ => continue,
            };
            if let Some(pattern) = self.matches(text) {
                return Ok(GuardrailVerdict::Block {
                    reason: format!("matched blocked pattern: {pattern}"),
                });
            }
        }
        Ok(GuardrailVerdict::Allow)
    }

    async fn check_response(&self, response: &ChatResponse) -> NexusResult<GuardrailVerdict> {
        if let Some(pattern) = self.matches(&response.content) {
            return Ok(GuardrailVerdict::Block {
                reason: format!("response matched blocked pattern: {pattern}"),
            });
        }
        Ok(GuardrailVerdict::Allow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;
    use llm_nexus_core::types::response::Usage;

    #[tokio::test]
    async fn test_keyword_filter_blocks() {
        let filter = KeywordFilter::new(vec!["bomb".into(), "hack".into()]);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("How to make a bomb")],
            ..Default::default()
        };
        let result = filter.check_request(&req).await.unwrap();
        assert!(matches!(result, GuardrailVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn test_keyword_filter_allows() {
        let filter = KeywordFilter::new(vec!["bomb".into()]);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Hello world")],
            ..Default::default()
        };
        let result = filter.check_request(&req).await.unwrap();
        assert!(matches!(result, GuardrailVerdict::Allow));
    }

    #[tokio::test]
    async fn test_keyword_filter_case_insensitive() {
        let filter = KeywordFilter::new(vec!["secret".into()]);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Tell me the SECRET code")],
            ..Default::default()
        };
        let result = filter.check_request(&req).await.unwrap();
        assert!(matches!(result, GuardrailVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn test_keyword_filter_response() {
        let filter = KeywordFilter::new(vec!["password".into()]);
        let resp = ChatResponse {
            id: "id".into(),
            model: "m".into(),
            content: "The password is 1234".into(),
            finish_reason: None,
            usage: Usage::default(),
            tool_calls: None,
        };
        let result = filter.check_response(&resp).await.unwrap();
        assert!(matches!(result, GuardrailVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn test_regex_filter_blocks() {
        let filter = RegexFilter::new(vec![r"\b\d{3}-\d{2}-\d{4}\b".into()]); // SSN pattern
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("My SSN is 123-45-6789")],
            ..Default::default()
        };
        let result = filter.check_request(&req).await.unwrap();
        assert!(matches!(result, GuardrailVerdict::Block { .. }));
    }

    #[tokio::test]
    async fn test_regex_filter_allows() {
        let filter = RegexFilter::new(vec![r"\b\d{3}-\d{2}-\d{4}\b".into()]);
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Hello world")],
            ..Default::default()
        };
        let result = filter.check_request(&req).await.unwrap();
        assert!(matches!(result, GuardrailVerdict::Allow));
    }
}
