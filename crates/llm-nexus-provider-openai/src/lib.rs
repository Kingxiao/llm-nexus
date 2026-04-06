//! OpenAI provider adapter for llm-nexus.
//!
//! Implements [`ChatProvider`](llm_nexus_core::traits::chat::ChatProvider) and
//! [`EmbeddingProvider`](llm_nexus_core::traits::embedding::EmbeddingProvider) against
//! the OpenAI chat/completions and embeddings APIs.
//!
//! Also serves as the base for any OpenAI-compatible provider
//! (DeepSeek, OpenRouter, 302.AI, etc.) by swapping the `base_url`.
//!
//! # Examples
//!
//! ```rust,ignore
//! use llm_nexus_provider_openai::OpenAiProvider;
//!
//! let provider = OpenAiProvider::with_base_url_and_key(
//!     "https://api.openai.com/v1".into(),
//!     std::env::var("OPENAI_API_KEY").unwrap(),
//! );
//! ```

pub mod chat;
pub mod convert;
pub mod embedding;
pub mod stream;
pub mod types;

use llm_nexus_core::error::{NexusError, NexusResult};
use llm_nexus_core::types::config::ProviderConfig;

/// OpenAI-compatible provider.
pub struct OpenAiProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    pub(crate) provider_id: String,
}

impl OpenAiProvider {
    /// Create from a ProviderConfig. Reads API key from environment.
    pub fn from_config(config: &ProviderConfig, provider_id: &str) -> NexusResult<Self> {
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
            provider_id: provider_id.into(),
        })
    }

    /// Create with explicit base_url and api_key (for testing and OpenAI-compatible providers).
    pub fn with_base_url_and_key(base_url: String, api_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            api_key,
            provider_id: "openai".into(),
        }
    }

    /// Create with a custom provider_id (for DeepSeek, OpenRouter, etc.).
    pub fn with_provider_id(mut self, id: impl Into<String>) -> Self {
        self.provider_id = id.into();
        self
    }
}
