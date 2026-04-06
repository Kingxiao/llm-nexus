//! Request hashing for cache key generation.

use sha2::{Digest, Sha256};

use llm_nexus_core::types::request::ChatRequest;

/// Generate a deterministic cache key from a chat request.
///
/// Hashes: model + messages (role + content) + temperature + max_tokens + top_p + stop.
/// Tool definitions and response_format are included to avoid serving cached responses
/// for structurally different requests.
pub fn hash_request(request: &ChatRequest) -> String {
    let mut hasher = Sha256::new();

    // Use serde_json for deterministic serialization (not Debug which may vary
    // across Rust versions). ChatRequest derives Serialize, so this is stable.
    if let Ok(serialized) = serde_json::to_vec(request) {
        hasher.update(&serialized);
    } else {
        // Fallback: hash model + message count (should never happen since
        // ChatRequest is always serializable)
        hasher.update(request.model.as_bytes());
        hasher.update(request.messages.len().to_be_bytes());
    }

    let result = hasher.finalize();
    format!("nexus:cache:{:x}", result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_nexus_core::types::request::Message;

    #[test]
    fn test_same_request_same_hash() {
        let req = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        assert_eq!(hash_request(&req), hash_request(&req));
    }

    #[test]
    fn test_different_model_different_hash() {
        let req1 = ChatRequest {
            model: "gpt-5.4".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let req2 = ChatRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        assert_ne!(hash_request(&req1), hash_request(&req2));
    }

    #[test]
    fn test_different_message_different_hash() {
        let req1 = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Hello")],
            ..Default::default()
        };
        let req2 = ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("Goodbye")],
            ..Default::default()
        };
        assert_ne!(hash_request(&req1), hash_request(&req2));
    }

    #[test]
    fn test_hash_has_prefix() {
        let req = ChatRequest {
            model: "m".into(),
            messages: vec![],
            ..Default::default()
        };
        assert!(hash_request(&req).starts_with("nexus:cache:"));
    }
}
