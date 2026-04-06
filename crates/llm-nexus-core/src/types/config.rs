use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexusConfig {
    pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub display_name: String,
    pub base_url: String,
    pub api_key_env: String,
    #[serde(default = "default_auth_header")]
    pub auth_header: String,
    #[serde(default = "default_auth_scheme")]
    pub auth_scheme: String,
    #[serde(default)]
    pub api_version: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "default_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub openai_compatible: bool,
}

fn default_auth_header() -> String {
    "Authorization".into()
}

fn default_auth_scheme() -> String {
    "Bearer".into()
}

fn default_timeout() -> u64 {
    120
}

fn default_retries() -> u32 {
    3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_config_deserialize_with_defaults() {
        let json = r#"{
            "display_name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY"
        }"#;
        let config: ProviderConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.display_name, "OpenAI");
        assert_eq!(config.auth_header, "Authorization");
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_retries, 3);
        assert!(!config.openai_compatible);
        assert!(config.api_version.is_none());
    }

    #[test]
    fn test_nexus_config_deserialize() {
        let json = r#"{
            "providers": {
                "openai": {
                    "display_name": "OpenAI",
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                    "openai_compatible": true
                },
                "anthropic": {
                    "display_name": "Anthropic",
                    "base_url": "https://api.anthropic.com/v1",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "auth_header": "x-api-key",
                    "auth_scheme": ""
                }
            }
        }"#;
        let config: NexusConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.providers.len(), 2);
        assert!(config.providers.contains_key("openai"));
        assert!(config.providers["openai"].openai_compatible);
        assert_eq!(config.providers["anthropic"].auth_header, "x-api-key");
    }

    #[test]
    fn test_provider_config_serialize_roundtrip() {
        let config = ProviderConfig {
            display_name: "Test".into(),
            base_url: "https://test.com".into(),
            api_key_env: "TEST_KEY".into(),
            auth_header: default_auth_header(),
            auth_scheme: "Bearer".into(),
            api_version: Some("2024-01-01".into()),
            timeout_secs: 60,
            max_retries: 5,
            openai_compatible: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ProviderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.timeout_secs, 60);
        assert_eq!(deserialized.max_retries, 5);
        assert_eq!(deserialized.api_version.as_deref(), Some("2024-01-01"));
    }
}
