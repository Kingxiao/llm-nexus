//! Integration tests for chat providers against real APIs.
//!
//! Run with: `NEXUS_INTEGRATION=1 cargo test --test chat_providers --features full`

mod common;

use common::{build_client, skip_unless_integration};
use llm_nexus::types::request::{ChatRequest, Message};

#[tokio::test]
async fn test_openai_chat() {
    skip_unless_integration!();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("OPENAI_API_KEY not set, skipping");
        return;
    }

    let client = build_client();
    let request = ChatRequest {
        model: "gpt-4.1-nano".into(),
        messages: vec![Message::user("Say 'hello' and nothing else.")],
        max_tokens: Some(10),
        ..Default::default()
    };

    let response = client.chat(&request).await.unwrap();
    assert!(!response.content.is_empty());
    assert!(response.usage.total_tokens > 0);
}

#[tokio::test]
async fn test_anthropic_chat() {
    skip_unless_integration!();
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("ANTHROPIC_API_KEY not set, skipping");
        return;
    }

    let client = build_client();
    let request = ChatRequest {
        model: "claude-haiku-4-5-20251001".into(),
        messages: vec![Message::user("Say 'hello' and nothing else.")],
        max_tokens: Some(10),
        ..Default::default()
    };

    let response = client.chat(&request).await.unwrap();
    assert!(!response.content.is_empty());
}

#[tokio::test]
async fn test_gemini_chat() {
    skip_unless_integration!();
    if std::env::var("GEMINI_API_KEY").is_err() {
        eprintln!("GEMINI_API_KEY not set, skipping");
        return;
    }

    let client = build_client();
    let request = ChatRequest {
        model: "gemini-2.5-flash".into(),
        messages: vec![Message::user("Say 'hello' and nothing else.")],
        max_tokens: Some(10),
        ..Default::default()
    };

    let response = client.chat(&request).await.unwrap();
    assert!(!response.content.is_empty());
}

#[tokio::test]
async fn test_deepseek_chat() {
    skip_unless_integration!();
    if std::env::var("DEEPSEEK_API_KEY").is_err() {
        eprintln!("DEEPSEEK_API_KEY not set, skipping");
        return;
    }

    let client = build_client();
    let request = ChatRequest {
        model: "deepseek-chat".into(),
        messages: vec![Message::user("Say 'hello' and nothing else.")],
        max_tokens: Some(10),
        ..Default::default()
    };

    let response = client.chat(&request).await.unwrap();
    assert!(!response.content.is_empty());
}

#[tokio::test]
async fn test_streaming_openai() {
    skip_unless_integration!();
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("OPENAI_API_KEY not set, skipping");
        return;
    }

    use futures::StreamExt;

    let client = build_client();
    let request = ChatRequest {
        model: "gpt-4.1-nano".into(),
        messages: vec![Message::user("Count from 1 to 3.")],
        max_tokens: Some(20),
        ..Default::default()
    };

    let mut stream = client.chat_stream(&request).await.unwrap();
    let mut chunks = 0;
    let mut text = String::new();

    while let Some(result) = stream.next().await {
        let chunk = result.unwrap();
        if let Some(content) = chunk.delta_content {
            text.push_str(&content);
        }
        chunks += 1;
    }

    assert!(chunks > 0, "expected at least one chunk");
    assert!(!text.is_empty(), "expected non-empty streamed text");
}
