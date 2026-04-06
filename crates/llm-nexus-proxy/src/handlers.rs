//! Route handlers for the OpenAI-compatible proxy endpoints.

use axum::extract::{FromRequest, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use llm_nexus::types::model::ModelFilter;
use llm_nexus_core::error::NexusError;
use llm_nexus_core::traits::registry::ModelRegistry;

use crate::embed_types::{OaiEmbedRequest, OaiEmbedResponse};
use crate::sse::stream_to_sse;
use crate::types::{
    OaiChatRequest, OaiChatResponse, OaiErrorBody, OaiErrorResponse, OaiModel, OaiModelList,
};
use crate::AppState;

/// POST /v1/chat/completions
///
/// Accepts an OpenAI-format request, dispatches through NexusClient,
/// and returns either a JSON response or an SSE stream.
pub async fn chat_completions(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Result<impl IntoResponse, impl IntoResponse> {
    // Check virtual key model restrictions if Identity is present
    let identity = request
        .extensions()
        .get::<crate::virtual_key::Identity>()
        .cloned();

    let body: OaiChatRequest = match axum::Json::from_request(request, &state).await {
        Ok(axum::Json(b)) => b,
        Err(e) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid request: {e}"),
                "invalid_request_error",
            ));
        }
    };

    // Enforce allowed_models from virtual key
    let model_allowed = identity
        .as_ref()
        .and_then(|id| id.allowed_models.as_ref())
        .map(|allowed| allowed.iter().any(|m| m == &body.model))
        .unwrap_or(true);

    if !model_allowed {
        return Err(error_response(
            StatusCode::FORBIDDEN,
            &format!("model '{}' not allowed for this key", body.model),
            "permission_error",
        ));
    }

    let is_stream = body.is_stream();
    let model_name = body.model.clone();
    let request_id = uuid::Uuid::new_v4().to_string();

    let chat_request = match body.into_chat_request() {
        Ok(r) => r,
        Err(e) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                &format!("invalid request: {e}"),
                "invalid_request_error",
            ));
        }
    };

    if is_stream {
        match state.client.chat_stream(&chat_request).await {
            Ok(stream) => Ok(stream_to_sse(stream, request_id, model_name).into_response()),
            Err(e) => Err(nexus_error_response(&e)),
        }
    } else {
        match state.client.chat(&chat_request).await {
            Ok(resp) => {
                let oai_resp: OaiChatResponse = resp.into();
                Ok(Json(oai_resp).into_response())
            }
            Err(e) => Err(nexus_error_response(&e)),
        }
    }
}

/// GET /v1/models
///
/// Lists all available models from the registry in OpenAI format.
pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let filter = ModelFilter::default();
    match state.client.registry().list_models(&filter).await {
        Ok(models) => {
            let data: Vec<OaiModel> = models
                .into_iter()
                .map(|m| OaiModel {
                    id: m.id,
                    object: "model",
                    created: 0,
                    owned_by: m.provider,
                })
                .collect();
            Json(OaiModelList {
                object: "list",
                data,
            })
            .into_response()
        }
        Err(e) => nexus_error_response(&e).into_response(),
    }
}

/// POST /v1/embeddings
///
/// Accepts an OpenAI-format embedding request, dispatches through NexusClient.
pub async fn embeddings(
    State(state): State<AppState>,
    Json(body): Json<OaiEmbedRequest>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let embed_request = body.into_embed_request();

    match state.client.embed(&embed_request).await {
        Ok(resp) => {
            let oai_resp: OaiEmbedResponse = resp.into();
            Ok(Json(oai_resp).into_response())
        }
        Err(e) => Err(nexus_error_response(&e)),
    }
}

/// GET /health
pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET /metrics (text format — Prometheus or any registered exporter)
pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    match &state.metrics_gatherer {
        Some(gatherer) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            gatherer(),
        )
            .into_response(),
        None => (StatusCode::SERVICE_UNAVAILABLE, "metrics not configured").into_response(),
    }
}

// ---------- Error mapping ----------

fn nexus_error_response(err: &NexusError) -> (StatusCode, Json<OaiErrorResponse>) {
    let (status, error_type) = match err {
        NexusError::AuthError(_) => (StatusCode::UNAUTHORIZED, "authentication_error"),
        NexusError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
        NexusError::RateLimited { .. } => (StatusCode::TOO_MANY_REQUESTS, "rate_limit_error"),
        NexusError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, "timeout_error"),
        NexusError::NoRouteAvailable => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        NexusError::ProviderError { status_code, .. } => {
            let code = status_code
                .and_then(|c| StatusCode::from_u16(c).ok())
                .unwrap_or(StatusCode::BAD_GATEWAY);
            (code, "upstream_error")
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
    };
    error_response(status, &err.to_string(), error_type)
}

fn error_response(
    status: StatusCode,
    message: &str,
    error_type: &str,
) -> (StatusCode, Json<OaiErrorResponse>) {
    (
        status,
        Json(OaiErrorResponse {
            error: OaiErrorBody {
                message: message.to_owned(),
                error_type: error_type.to_owned(),
                code: None,
            },
        }),
    )
}
