//! Configuration loading with TOML files and environment variable overrides.

use std::path::Path;

use crate::error::{NexusError, NexusResult};
use crate::types::config::NexusConfig;
use crate::types::model::ModelMetadata;

/// Wrapper for the models TOML array.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelsConfig {
    pub models: Vec<ModelMetadata>,
}

/// Loads `providers.toml` and `models.toml` from the given directory.
pub fn load_config(config_dir: &Path) -> NexusResult<(NexusConfig, Vec<ModelMetadata>)> {
    let providers_path = config_dir.join("providers.toml");
    let models_path = config_dir.join("models.toml");

    let providers_str = std::fs::read_to_string(&providers_path).map_err(|e| {
        NexusError::ConfigError(format!(
            "Failed to read {}: {}",
            providers_path.display(),
            e
        ))
    })?;
    let models_str = std::fs::read_to_string(&models_path).map_err(|e| {
        NexusError::ConfigError(format!("Failed to read {}: {}", models_path.display(), e))
    })?;

    let providers_config: NexusConfig = toml::from_str(&providers_str)?;
    let models_config: ModelsConfig = toml::from_str(&models_str)?;

    Ok((providers_config, models_config.models))
}

/// Applies environment variable overrides to provider configuration.
///
/// Pattern: `NEXUS_PROVIDERS_{NAME}_{FIELD}` overrides `providers.{name}.{field}`.
///
/// Supported fields: `BASE_URL`, `TIMEOUT_SECS`, `MAX_RETRIES`.
pub fn apply_env_overrides(config: &mut NexusConfig) {
    for (name, provider) in config.providers.iter_mut() {
        let prefix = format!("NEXUS_PROVIDERS_{}", name.to_uppercase());

        if let Ok(val) = std::env::var(format!("{prefix}_BASE_URL")) {
            provider.base_url = val;
        }
        if let Ok(val) = std::env::var(format!("{prefix}_TIMEOUT_SECS"))
            && let Ok(v) = val.parse()
        {
            provider.timeout_secs = v;
        }
        if let Ok(val) = std::env::var(format!("{prefix}_MAX_RETRIES"))
            && let Ok(v) = val.parse()
        {
            provider.max_retries = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config")
    }

    #[test]
    fn test_load_providers_config() {
        let (config, _models) = load_config(&config_dir()).unwrap();
        assert!(config.providers.contains_key("openai"));
        assert_eq!(config.providers["openai"].api_key_env, "OPENAI_API_KEY");
        assert!(config.providers.contains_key("anthropic"));
        assert_eq!(config.providers["anthropic"].auth_header, "x-api-key");
        assert!(config.providers.contains_key("gemini"));
        assert!(config.providers.contains_key("deepseek"));
        assert!(config.providers.contains_key("openrouter"));
    }

    #[test]
    fn test_load_models_config() {
        let (_config, models) = load_config(&config_dir()).unwrap();
        assert!(models.len() >= 10);
        let gpt = models.iter().find(|m| m.id == "gpt-5.4").unwrap();
        assert_eq!(gpt.provider, "openai");
        assert_eq!(gpt.context_window, 1000000);
    }

    #[test]
    fn test_env_override() {
        let (mut config, _) = load_config(&config_dir()).unwrap();
        // SAFETY: test runs single-threaded via `cargo test -- --test-threads=1` or
        // env mutation is scoped and restored immediately.
        unsafe {
            std::env::set_var("NEXUS_PROVIDERS_OPENAI_BASE_URL", "http://localhost:8080");
        }
        apply_env_overrides(&mut config);
        assert_eq!(config.providers["openai"].base_url, "http://localhost:8080");
        unsafe {
            std::env::remove_var("NEXUS_PROVIDERS_OPENAI_BASE_URL");
        }
    }

    #[test]
    fn test_env_override_timeout() {
        let (mut config, _) = load_config(&config_dir()).unwrap();
        unsafe {
            std::env::set_var("NEXUS_PROVIDERS_ANTHROPIC_TIMEOUT_SECS", "30");
        }
        apply_env_overrides(&mut config);
        assert_eq!(config.providers["anthropic"].timeout_secs, 30);
        unsafe {
            std::env::remove_var("NEXUS_PROVIDERS_ANTHROPIC_TIMEOUT_SECS");
        }
    }

    #[test]
    fn test_load_config_missing_dir() {
        let result = load_config(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }
}
