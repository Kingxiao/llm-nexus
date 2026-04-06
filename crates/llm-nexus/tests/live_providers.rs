//! Live API integration tests for real providers.
//!
//! Requires real API keys in environment variables.
//! Gated behind `NEXUS_INTEGRATION=1`.
//!
//! Run with:
//! ```
//! NEXUS_INTEGRATION=1 cargo test -p llm-nexus --test live_providers --all-features -- --nocapture
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use llm_nexus::client::NexusClient;
use llm_nexus::NexusError;
use llm_nexus_cache::CacheMiddleware;
use llm_nexus_core::store::InMemoryStore;
use llm_nexus_core::types::request::{ChatRequest, Message};

macro_rules! skip_unless_integration {
    () => {
        if std::env::var("NEXUS_INTEGRATION").is_err() {
            eprintln!("skipping integration test (set NEXUS_INTEGRATION=1)");
            return;
        }
    };
}

fn config_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")
}

fn build_client() -> NexusClient {
    NexusClient::from_config_dir(&config_dir()).expect("failed to load config")
}

// ---------------------------------------------------------------------------
// Provider registration
// ---------------------------------------------------------------------------

#[test]
fn test_provider_registration() {
    skip_unless_integration!();
    let client = build_client();
    let providers = client.provider_ids();
    println!("registered providers: {providers:?}");
    assert!(!providers.is_empty(), "at least one provider should register");
}

// ---------------------------------------------------------------------------
// OpenRouter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_openrouter_chat() {
    skip_unless_integration!();
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("skipping: OPENROUTER_API_KEY not set");
        return;
    }
    let client = build_client();
    let req = ChatRequest {
        model: "openrouter/auto".into(),
        messages: vec![Message::user("Say 'hello' and nothing else.")],
        max_tokens: Some(10),
        ..Default::default()
    };
    let resp = client.chat(&req).await;
    println!("openrouter response: {resp:?}");
    assert!(resp.is_ok(), "openrouter chat failed: {resp:?}");
    let resp = resp.unwrap();
    assert!(!resp.content.is_empty());
    println!("openrouter: model={} content='{}' tokens={}", resp.model, resp.content, resp.usage.total_tokens);
}

// ---------------------------------------------------------------------------
// MiniMax
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_minimax_chat() {
    skip_unless_integration!();
    if std::env::var("MINIMAX_API_KEY").is_err() {
        eprintln!("skipping: MINIMAX_API_KEY not set");
        return;
    }
    let client = build_client();
    let req = ChatRequest {
        model: "MiniMax-M2".into(),
        messages: vec![Message::user("Reply with just the word 'pong'.")],
        max_tokens: Some(10),
        ..Default::default()
    };
    let resp = client.chat(&req).await;
    match &resp {
        Err(NexusError::ProviderError { status_code: Some(401), .. }) => {
            eprintln!("minimax: 401 auth error — check MINIMAX_API_KEY format (may need group_id.api_key)");
            return; // skip, not fail
        }
        _ => {}
    }
    println!("minimax response: {resp:?}");
    assert!(resp.is_ok(), "minimax chat failed: {resp:?}");
    let resp = resp.unwrap();
    assert!(!resp.content.is_empty());
    println!("minimax: model={} content='{}' tokens={}", resp.model, resp.content, resp.usage.total_tokens);
}

// ---------------------------------------------------------------------------
// 302.AI
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_302ai_chat() {
    skip_unless_integration!();
    if std::env::var("API_302_KEY").is_err() {
        eprintln!("skipping: API_302_KEY not set");
        return;
    }
    let client = build_client();
    let req = ChatRequest {
        model: "gpt-4o".into(),
        messages: vec![Message::user("Reply with just 'ok'.")],
        max_tokens: Some(10),
        ..Default::default()
    };
    let resp = client.chat(&req).await;
    match &resp {
        Ok(r) => println!("302.ai: model={} content='{}'", r.model, r.content),
        Err(e) => println!("302.ai error: {e}"),
    }
    assert!(resp.is_ok(), "302.ai chat failed");
    let resp = resp.unwrap();
    assert!(!resp.content.is_empty());
    println!("302.ai: model={} content='{}' tokens={}", resp.model, resp.content, resp.usage.total_tokens);
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_openrouter_streaming() {
    skip_unless_integration!();
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("skipping: OPENROUTER_API_KEY not set");
        return;
    }
    let client = build_client();
    let req = ChatRequest {
        model: "openrouter/auto".into(),
        messages: vec![Message::user("Count from 1 to 3.")],
        max_tokens: Some(20),
        ..Default::default()
    };
    let stream = client.chat_stream(&req).await;
    assert!(stream.is_ok(), "stream setup failed");

    let mut stream = stream.unwrap();
    let mut chunks = 0;
    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(c) => {
                if let Some(text) = &c.delta_content {
                    content.push_str(text);
                }
                chunks += 1;
            }
            Err(e) => {
                eprintln!("stream error: {e}");
                break;
            }
        }
    }
    println!("openrouter stream: {chunks} chunks, content='{content}'");
    assert!(chunks > 0, "expected at least one chunk");
}

// ---------------------------------------------------------------------------
// Cache integration with real provider
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_cache_with_real_provider() {
    skip_unless_integration!();
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("skipping: OPENROUTER_API_KEY not set");
        return;
    }

    let store = Arc::new(InMemoryStore::new());

    let client = NexusClient::builder()
        .config_dir(&config_dir())
        .unwrap()
        .auto_register_providers()
        .with_middleware(Arc::new(CacheMiddleware::new(
            store.clone(),
            Duration::from_secs(60),
        )))
        .build()
        .unwrap();

    let req = ChatRequest {
        model: "openrouter/auto".into(),
        messages: vec![Message::user("What is 2+2? Answer with just the number.")],
        max_tokens: Some(5),
        ..Default::default()
    };

    // First call — hits the real API
    let start1 = std::time::Instant::now();
    let resp1 = client.chat(&req).await.unwrap();
    let dur1 = start1.elapsed();
    println!("first call: {}ms, content='{}'", dur1.as_millis(), resp1.content);

    // Second call — should hit cache (much faster)
    let start2 = std::time::Instant::now();
    let resp2 = client.chat(&req).await.unwrap();
    let dur2 = start2.elapsed();
    println!("second call: {}ms, content='{}'", dur2.as_millis(), resp2.content);

    assert_eq!(resp1.content, resp2.content, "cached response should match");
    assert!(dur2 < dur1, "cache hit should be faster than API call");
    println!("cache speedup: {:.1}x", dur1.as_millis() as f64 / dur2.as_millis().max(1) as f64);
}
