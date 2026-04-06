//! Azure OpenAI provider adapter for llm-nexus.
//!
//! Implements [`ChatProvider`](llm_nexus_core::traits::chat::ChatProvider) with
//! Azure-specific URL construction and authentication.
//!
//! Supports two URL formats:
//! - **v1** (recommended): `{resource_endpoint}/openai/v1/chat/completions`
//! - **Classic**: `{resource_endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}`
//!
//! Auth: `api-key` header (API key) or `Authorization: Bearer` (AAD/Entra token).
//!
//! # Examples
//!
//! ```rust,ignore
//! use llm_nexus_provider_azure::AzureOpenAiProvider;
//!
//! let provider = AzureOpenAiProvider::new(
//!     "https://my-resource.openai.azure.com",
//!     "my-api-key",
//!     false, // use api-key auth, not bearer
//!     None,  // no api_version = v1 format
//! );
//! ```

pub mod chat;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::config::ProviderConfig;

/// Azure OpenAI provider.
pub struct AzureOpenAiProvider {
    client: reqwest::Client,
    /// Resource endpoint (e.g. `https://my-resource.openai.azure.com`)
    resource_endpoint: String,
    /// API key or AAD bearer token
    api_key: String,
    /// Auth mode: "api-key" or "bearer"
    auth_mode: AuthMode,
    /// API version for classic endpoint format
    api_version: Option<String>,
    /// Default deployment name (used when model isn't a deployment ID)
    default_deployment: Option<String>,
}

#[derive(Debug, Clone)]
enum AuthMode {
    /// Azure API key auth (header: `api-key`)
    ApiKey,
    /// AAD/Entra ID bearer token (header: `Authorization: Bearer`)
    Bearer,
}

impl AzureOpenAiProvider {
    /// Create from ProviderConfig.
    ///
    /// Config fields:
    /// - `base_url`: Azure resource endpoint
    /// - `api_key_env`: env var for API key
    /// - `api_version`: optional, enables classic deployment URL format
    /// - `auth_scheme`: "Bearer" for AAD, anything else for api-key
    pub fn from_config(config: &ProviderConfig) -> NexusResult<Self> {
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            NexusError::AuthError(format!(
                "Environment variable {} not set",
                config.api_key_env
            ))
        })?;

        let auth_mode = if config.auth_scheme == "Bearer" {
            AuthMode::Bearer
        } else {
            AuthMode::ApiKey
        };

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| NexusError::HttpError(e.to_string()))?;

        Ok(Self {
            client,
            resource_endpoint: config.base_url.trim_end_matches('/').to_string(),
            api_key,
            auth_mode,
            api_version: config.api_version.clone(),
            default_deployment: None,
        })
    }

    /// Create with explicit parameters (for testing).
    pub fn new(
        resource_endpoint: impl Into<String>,
        api_key: impl Into<String>,
        auth_mode_bearer: bool,
        api_version: Option<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            resource_endpoint: resource_endpoint.into(),
            api_key: api_key.into(),
            auth_mode: if auth_mode_bearer {
                AuthMode::Bearer
            } else {
                AuthMode::ApiKey
            },
            api_version,
            default_deployment: None,
        }
    }

    /// Set a default deployment name.
    pub fn with_default_deployment(mut self, deployment: impl Into<String>) -> Self {
        self.default_deployment = Some(deployment.into());
        self
    }

    /// Build the chat completions URL.
    ///
    /// If `api_version` is set, uses classic deployment format.
    /// Otherwise uses v1 format.
    pub(crate) fn chat_url(&self, model: &str) -> String {
        if let Some(ref version) = self.api_version {
            // Classic: /openai/deployments/{deployment}/chat/completions?api-version={v}
            let deployment = self.default_deployment.as_deref().unwrap_or(model);
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                self.resource_endpoint, deployment, version
            )
        } else {
            // v1: /openai/v1/chat/completions
            format!("{}/openai/v1/chat/completions", self.resource_endpoint)
        }
    }

    pub(crate) fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth_mode {
            AuthMode::ApiKey => req.header("api-key", &self.api_key),
            AuthMode::Bearer => req.header("Authorization", format!("Bearer {}", self.api_key)),
        }
    }
}
