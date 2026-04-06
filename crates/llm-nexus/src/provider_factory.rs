//! Declarative provider factory — creates chat + embedding providers from config.
//!
//! Each provider registers its constructor here. Adding a new provider only
//! requires adding a single match arm + constructor function, without touching
//! client.rs or the builder logic.

use std::sync::Arc;

use llm_nexus_core::error::NexusResult;
use llm_nexus_core::traits::chat::ChatProvider;
use llm_nexus_core::traits::embedding::EmbeddingProvider;
use llm_nexus_core::types::config::ProviderConfig;

/// Result of creating a provider — may include both chat and embedding capabilities.
pub struct ProviderRegistration {
    pub chat: Option<Arc<dyn ChatProvider>>,
    pub embedding: Option<Arc<dyn EmbeddingProvider>>,
}

/// Attempt to create provider(s) for the given provider ID and config.
///
/// Returns `None` if no adapter is available (unknown ID, feature not compiled).
/// Returns `Some(Err(...))` if the adapter exists but construction failed (e.g. missing API key).
pub fn create_provider(
    id: &str,
    config: &ProviderConfig,
) -> Option<NexusResult<ProviderRegistration>> {
    match id {
        #[cfg(feature = "openai")]
        "openai" => Some(create_openai(config)),

        #[cfg(feature = "anthropic")]
        "anthropic" => Some(create_anthropic(config)),

        #[cfg(feature = "gemini")]
        "gemini" => Some(create_gemini(config)),

        #[cfg(feature = "deepseek")]
        "deepseek" => Some(create_deepseek(config)),

        #[cfg(feature = "azure")]
        "azure_openai" => Some(create_azure(config)),

        #[cfg(feature = "vertex")]
        "vertex" => Some(create_vertex(config)),

        _ if config.openai_compatible => create_openai_compatible(config, id),

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Per-provider constructors
// ---------------------------------------------------------------------------

#[cfg(feature = "openai")]
fn create_openai(config: &ProviderConfig) -> NexusResult<ProviderRegistration> {
    let provider = Arc::new(llm_nexus_provider_openai::OpenAiProvider::from_config(
        config, "openai",
    )?);
    Ok(ProviderRegistration {
        chat: Some(provider.clone()),
        embedding: Some(provider),
    })
}

#[cfg(feature = "anthropic")]
fn create_anthropic(config: &ProviderConfig) -> NexusResult<ProviderRegistration> {
    let provider = Arc::new(llm_nexus_provider_anthropic::AnthropicProvider::new(
        config,
    )?);
    Ok(ProviderRegistration {
        chat: Some(provider),
        embedding: None,
    })
}

#[cfg(feature = "gemini")]
fn create_gemini(config: &ProviderConfig) -> NexusResult<ProviderRegistration> {
    let provider = Arc::new(llm_nexus_provider_gemini::GeminiProvider::from_config(
        config,
    )?);
    Ok(ProviderRegistration {
        chat: Some(provider.clone()),
        embedding: Some(provider),
    })
}

#[cfg(feature = "deepseek")]
fn create_deepseek(config: &ProviderConfig) -> NexusResult<ProviderRegistration> {
    let provider = Arc::new(llm_nexus_provider_deepseek::DeepSeekProvider::from_config(
        config,
    )?);
    Ok(ProviderRegistration {
        chat: Some(provider),
        embedding: None,
    })
}

#[cfg(feature = "azure")]
fn create_azure(config: &ProviderConfig) -> NexusResult<ProviderRegistration> {
    let provider = Arc::new(llm_nexus_provider_azure::AzureOpenAiProvider::from_config(
        config,
    )?);
    Ok(ProviderRegistration {
        chat: Some(provider),
        embedding: None,
    })
}

#[cfg(feature = "vertex")]
fn create_vertex(config: &ProviderConfig) -> NexusResult<ProviderRegistration> {
    let provider = Arc::new(llm_nexus_provider_vertex::VertexAiProvider::from_config(
        config,
    )?);
    Ok(ProviderRegistration {
        chat: Some(provider),
        embedding: None,
    })
}

fn create_openai_compatible(
    config: &ProviderConfig,
    id: &str,
) -> Option<NexusResult<ProviderRegistration>> {
    #[cfg(feature = "openai")]
    {
        let result = llm_nexus_provider_openai::OpenAiProvider::from_config(config, id).map(|p| {
            ProviderRegistration {
                chat: Some(Arc::new(p)),
                embedding: None,
            }
        });
        Some(result)
    }
    #[cfg(not(feature = "openai"))]
    {
        let _ = (config, id);
        tracing::warn!(provider = %id, "openai feature required for compatible providers");
        None
    }
}
