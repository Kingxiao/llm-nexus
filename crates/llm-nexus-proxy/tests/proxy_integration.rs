//! Integration tests for the proxy server.
//!
//! Uses axum's built-in test utilities with mock providers — no real HTTP server needed.

use std::pin::Pin;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use futures::Stream;
use llm_nexus::NexusClient;
use llm_nexus_core::error::NexusResult;
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::types::request::ChatRequest;
use llm_nexus_core::types::response::{ChatResponse, FinishReason, StreamChunk, Usage};
use llm_nexus_proxy::build_router;
use tower::ServiceExt;

// ---------- Mock provider ----------

struct MockProvider {
    id: String,
}

#[async_trait::async_trait]
impl ChatProvider for MockProvider {
    fn provider_id(&self) -> &str {
        &self.id
    }

    async fn chat(&self, req: &ChatRequest) -> NexusResult<ChatResponse> {
        Ok(ChatResponse {
            id: "chatcmpl-mock-123".into(),
            model: req.model.clone(),
            content: "Hello from mock proxy!".into(),
            finish_reason: Some(FinishReason::Stop),
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 8,
                total_tokens: 18,
            },
            tool_calls: None,
        })
    }

    async fn chat_stream(
        &self,
        _req: &ChatRequest,
    ) -> NexusResult<Pin<Box<dyn Stream<Item = NexusResult<StreamChunk>> + Send>>> {
        let chunks = vec![
            Ok(StreamChunk {
                delta_content: Some("Hello".into()),
                delta_tool_call: None,
                finish_reason: None,
                usage: None,
            }),
            Ok(StreamChunk {
                delta_content: Some(" world".into()),
                delta_tool_call: None,
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    prompt_tokens: 5,
                    completion_tokens: 2,
                    total_tokens: 7,
                }),
            }),
        ];
        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    async fn list_models(&self) -> NexusResult<Vec<String>> {
        Ok(vec!["mock-model".into()])
    }
}

// ---------- Test config helper ----------

/// Write minimal TOML configs to a temp directory and return the client.
fn build_test_client_with_mock() -> Arc<NexusClient> {
    let tmp = tempfile::tempdir().unwrap();
    let config_dir = tmp.path();

    std::fs::write(
        config_dir.join("providers.toml"),
        concat!(
            "[providers.mock]\n",
            "display_name = \"Mock\"\n",
            "base_url = \"http://localhost:0\"\n",
            "api_key_env = \"MOCK_API_KEY\"\n",
            "auth_header = \"Authorization\"\n",
            "auth_scheme = \"Bearer\"\n",
            "timeout_secs = 30\n",
            "max_retries = 0\n",
        ),
    )
    .unwrap();

    std::fs::write(
        config_dir.join("models.toml"),
        concat!(
            "[[models]]\n",
            "id = \"mock-model\"\n",
            "provider = \"mock\"\n",
            "display_name = \"Mock Model\"\n",
            "context_window = 128000\n",
            "max_output_tokens = 4096\n",
            "input_price_per_1m = 1.0\n",
            "output_price_per_1m = 3.0\n",
            "capabilities = [\"Chat\"]\n",
            "\n",
            "[models.features]\n",
            "vision = false\n",
            "tool_use = false\n",
            "json_mode = false\n",
            "streaming = true\n",
            "system_prompt = true\n",
        ),
    )
    .unwrap();

    unsafe { std::env::set_var("MOCK_API_KEY", "test-key") };

    // NOTE: tempdir is dropped here but the config is already loaded into memory.
    // The client holds all parsed data, not file references.
    let client = NexusClient::builder()
        .config_dir(config_dir)
        .unwrap()
        .with_provider("mock", Arc::new(MockProvider { id: "mock".into() }))
        .with_in_memory_metrics()
        .build()
        .unwrap();

    // Keep tmpdir alive by leaking — tests are short-lived processes.
    std::mem::forget(tmp);

    Arc::new(client)
}

// ---------- Tests ----------

#[tokio::test]
async fn test_health_endpoint() {
    let client = Arc::new(
        NexusClient::builder()
            .with_in_memory_metrics()
            .build()
            .unwrap(),
    );
    let app = build_router(client);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn test_models_endpoint_empty() {
    let client = Arc::new(
        NexusClient::builder()
            .with_in_memory_metrics()
            .build()
            .unwrap(),
    );
    let app = build_router(client);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    assert!(json["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_chat_completions_model_not_found() {
    let client = Arc::new(
        NexusClient::builder()
            .with_in_memory_metrics()
            .build()
            .unwrap(),
    );
    let app = build_router(client);

    let body = serde_json::json!({
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "hi"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("nonexistent"));
}

#[tokio::test]
async fn test_chat_completions_invalid_body() {
    let client = Arc::new(
        NexusClient::builder()
            .with_in_memory_metrics()
            .build()
            .unwrap(),
    );
    let app = build_router(client);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"not": "valid"}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    // axum returns 422 for deserialization failure (missing required fields)
    assert!(
        response.status() == StatusCode::BAD_REQUEST
            || response.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn test_chat_completions_with_mock_provider() {
    let client = build_test_client_with_mock();
    let app = build_router(client);

    let body = serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["object"], "chat.completion");
    assert_eq!(
        json["choices"][0]["message"]["content"],
        "Hello from mock proxy!"
    );
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert_eq!(json["usage"]["prompt_tokens"], 10);
    assert_eq!(json["usage"]["completion_tokens"], 8);
    assert_eq!(json["usage"]["total_tokens"], 18);
}

#[tokio::test]
async fn test_models_endpoint_with_data() {
    let client = build_test_client_with_mock();
    let app = build_router(client);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "mock-model");
    assert_eq!(data[0]["owned_by"], "mock");
}

#[tokio::test]
async fn test_streaming_chat_completions() {
    let client = build_test_client_with_mock();
    let app = build_router(client);

    let body = serde_json::json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // SSE content-type
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.contains("text/event-stream"));

    // Read body and verify SSE events
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_str = String::from_utf8(body.to_vec()).unwrap();

    // Should contain data lines with JSON chunks and [DONE]
    assert!(body_str.contains("data: {"), "SSE data missing JSON");
    assert!(body_str.contains("data: [DONE]"), "SSE [DONE] missing");
    assert!(body_str.contains("Hello"), "missing 'Hello' chunk");
    assert!(body_str.contains(" world"), "missing ' world' chunk");
}

// Virtual key tests are in virtual_key module tests (virtual_key.rs)
// to avoid env var pollution with NEXUS_PROXY_AUTH_TOKEN.
