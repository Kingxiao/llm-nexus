//! Google Gemini provider adapter for llm-nexus.
//!
//! Implements [`ChatProvider`](llm_nexus_core::traits::chat::ChatProvider) and
//! [`EmbeddingProvider`](llm_nexus_core::traits::embedding::EmbeddingProvider) against
//! the Gemini `generateContent` API (verified: 2026-04-04).
//!
//! Key differences from OpenAI-style providers:
//! - Auth via query parameter `?key=` instead of `Authorization` header
//! - Model name embedded in URL path
//! - System prompt via top-level `system_instruction` field
//! - Role mapping: `assistant` -> `model`
//!
//! # Examples
//!
//! ```rust,ignore
//! use llm_nexus_provider_gemini::GeminiProvider;
//! use llm_nexus_core::types::config::ProviderConfig;
//!
//! let config = ProviderConfig { /* ... */ };
//! let provider = GeminiProvider::from_config(&config)?;
//! ```

pub mod chat;
pub mod convert;
pub mod embedding;
pub mod stream;
pub mod types;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::config::ProviderConfig;

/// Gemini API provider.
pub struct GeminiProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
}

impl GeminiProvider {
    /// Create from a [`ProviderConfig`]. Reads API key from environment variable.
    pub fn from_config(config: &ProviderConfig) -> NexusResult<Self> {
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            NexusError::AuthError(format!(
                "Environment variable {} not set",
                config.api_key_env
            ))
        })?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| NexusError::HttpError(e.to_string()))?;

        Ok(Self {
            client,
            base_url: config.base_url.clone(),
            api_key,
        })
    }

    /// Create with explicit base_url and api_key (useful for testing).
    pub fn with_base_url_and_key(base_url: String, api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
        }
    }
}
