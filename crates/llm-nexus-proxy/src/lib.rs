//! OpenAI-compatible HTTP proxy powered by [`NexusClient`](llm_nexus::NexusClient).
//!
//! Exposes `/v1/chat/completions`, `/v1/embeddings`, and `/v1/models` endpoints
//! that any OpenAI SDK can target. Includes bearer-token auth, virtual key
//! multi-tenancy, concurrency limiting, and optional Prometheus metrics.
//!
//! # Examples
//!
//! ```rust,no_run
//! use llm_nexus_proxy::build_router;
//! use std::sync::Arc;
//!
//! # async fn run(client: Arc<llm_nexus::NexusClient>) {
//! let router = build_router(client);
//! let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
//! axum::serve(listener, router).await.unwrap();
//! # }
//! ```

pub mod auth;
pub mod embed_types;
pub mod handlers;
pub mod rate_limit;
pub mod sse;
pub mod types;
pub mod virtual_key;

use std::sync::Arc;

use axum::Router;
use llm_nexus::NexusClient;
use tower::limit::ConcurrencyLimitLayer;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::virtual_key::VirtualKeyStore;

/// A function that gathers metrics in text format (e.g. Prometheus exposition).
pub type MetricsGatherer = dyn Fn() -> String + Send + Sync;

/// Shared state accessible to all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub client: Arc<NexusClient>,
    /// Optional metrics gatherer for `/metrics` endpoint.
    pub metrics_gatherer: Option<Arc<MetricsGatherer>>,
    /// Optional virtual key store for multi-tenant auth.
    pub virtual_keys: Option<Arc<VirtualKeyStore>>,
}

/// Build the axum router with all proxy endpoints.
pub fn build_router(client: Arc<NexusClient>) -> Router {
    let state = AppState {
        client,
        metrics_gatherer: None,
        virtual_keys: None,
    };

    build_router_with_state(state)
}

/// Build the axum router with a custom AppState (for Prometheus, virtual keys, etc.).
///
/// Includes:
/// - Bearer token auth (master token + virtual keys)
/// - Concurrency limiting (from `NEXUS_MAX_CONCURRENT_REQUESTS`, default 100)
/// - CORS (permissive)
/// - Request tracing
pub fn build_router_with_state(state: AppState) -> Router {
    let auth_state = auth::AuthState {
        master_token: std::env::var("NEXUS_PROXY_AUTH_TOKEN").ok(),
        virtual_keys: state.virtual_keys.clone(),
    };
    let max_concurrent = rate_limit::max_concurrent_requests();

    Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(handlers::chat_completions),
        )
        .route(
            "/v1/embeddings",
            axum::routing::post(handlers::embeddings),
        )
        .route("/v1/models", axum::routing::get(handlers::list_models))
        .route("/health", axum::routing::get(handlers::health))
        .route("/metrics", axum::routing::get(handlers::metrics))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(build_cors_layer())
                .layer(ConcurrencyLimitLayer::new(max_concurrent))
                .layer(axum::middleware::from_fn_with_state(
                    auth_state,
                    auth::auth_middleware,
                )),
        )
        .with_state(state)
}

/// Build CORS layer from `NEXUS_CORS_ORIGINS` env var.
///
/// - If set: only listed origins are allowed (comma-separated).
/// - If unset: permissive (all origins allowed, suitable for development).
fn build_cors_layer() -> CorsLayer {
    match std::env::var("NEXUS_CORS_ORIGINS") {
        Ok(origins) if !origins.is_empty() => {
            let allowed: Vec<_> = origins
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(allowed)
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any)
        }
        _ => CorsLayer::permissive(),
    }
}
