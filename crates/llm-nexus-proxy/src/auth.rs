//! Bearer token authentication middleware for the proxy.
//!
//! Auth flow:
//! 1. Skip auth for /health and /metrics
//! 2. If no master token configured AND no virtual key store → open proxy
//! 3. Check master token → if match, proceed (admin access)
//! 4. Check virtual key store → if match, inject Identity into request extensions
//! 5. Otherwise → 401

use std::sync::Arc;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::virtual_key::VirtualKeyStore;

/// Constant-time comparison to prevent timing side-channel attacks.
/// No external dependency — simple XOR accumulation.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let result = a
        .as_bytes()
        .iter()
        .zip(b.as_bytes())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y));
    result == 0
}

/// Auth state passed to the middleware.
#[derive(Clone)]
pub struct AuthState {
    /// Master API token (from `NEXUS_PROXY_AUTH_TOKEN` env var).
    pub master_token: Option<String>,
    /// Virtual key store for multi-tenant access.
    pub virtual_keys: Option<Arc<VirtualKeyStore>>,
}

pub async fn auth_middleware(
    axum::extract::State(auth): axum::extract::State<AuthState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Skip auth for non-API routes
    let path = request.uri().path();
    if path == "/health" || path == "/metrics" {
        return next.run(request).await;
    }

    // If no auth configured at all, allow all requests
    if auth.master_token.is_none() && auth.virtual_keys.is_none() {
        return next.run(request).await;
    }

    // Extract Bearer token
    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "));

    let Some(token) = token else {
        return unauthorized();
    };

    // Check master token first
    if auth
        .master_token
        .as_deref()
        .is_some_and(|m| constant_time_eq(token, m))
    {
        return next.run(request).await;
    }

    // Check virtual key store
    if let Some(identity) = auth.virtual_keys.as_ref().and_then(|s| s.resolve(token)) {
        request.extensions_mut().insert(identity);
        return next.run(request).await;
    }

    unauthorized()
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        axum::Json(serde_json::json!({
            "error": {
                "message": "invalid or missing authorization token",
                "type": "authentication_error"
            }
        })),
    )
        .into_response()
}
