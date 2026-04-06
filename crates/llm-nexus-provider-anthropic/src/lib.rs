//! Anthropic Messages API provider adapter for llm-nexus.
//!
//! Implements [`ChatProvider`](llm_nexus_core::traits::chat::ChatProvider) against
//! the Anthropic `/v1/messages` endpoint, including streaming and tool-use support.
//!
//! # Examples
//!
//! ```rust,ignore
//! use llm_nexus_provider_anthropic::AnthropicProvider;
//! use llm_nexus_core::types::config::ProviderConfig;
//!
//! let config = ProviderConfig { /* ... */ };
//! let provider = AnthropicProvider::new(&config)?;
//! ```

pub mod chat;
pub mod convert;
pub mod stream;
pub mod types;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::config::ProviderConfig;

pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    api_version: String,
}

impl AnthropicProvider {
    /// Create from a `ProviderConfig`. The API key is read from the environment
    /// variable named in `config.api_key_env`.
    pub fn new(config: &ProviderConfig) -> NexusResult<Self> {
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            NexusError::AuthError(format!(
                "Environment variable {} not set",
                config.api_key_env
            ))
        })?;

        // verified: 2026-04-04
        let api_version = config
            .api_version
            .clone()
            .unwrap_or_else(|| "2023-06-01".into());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| NexusError::HttpError(e.to_string()))?;

        Ok(Self {
            client,
            base_url: config.base_url.clone(),
            api_key,
            api_version,
        })
    }

    /// Test helper: create with explicit base_url and api_key (no env lookup).
    pub fn with_base_url_and_key(base_url: String, api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            api_version: "2023-06-01".into(), // verified: 2026-04-04
        }
    }
}
