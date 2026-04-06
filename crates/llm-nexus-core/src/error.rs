use thiserror::Error;

pub type NexusResult<T> = Result<T, NexusError>;

#[derive(Error, Debug)]
pub enum NexusError {
    #[error("Provider '{provider}' error (status {status_code:?}): {message}")]
    ProviderError {
        provider: String,
        message: String,
        status_code: Option<u16>,
    },
    #[error("HTTP error: {0}")]
    HttpError(String),
    #[error("Authentication error: {0}")]
    AuthError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("No suitable model found for routing context")]
    NoRouteAvailable,
    #[error("Timeout after {0}ms")]
    Timeout(u64),
    #[error("All providers in fallback chain failed")]
    AllProvidersFailed(Vec<NexusError>),
    #[error("Rate limited, retry after {retry_after_ms:?}ms")]
    RateLimited { retry_after_ms: Option<u64> },
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Guardrail blocked: {0}")]
    GuardrailBlocked(String),
    #[error("Budget exceeded: {0}")]
    BudgetExceeded(String),
}

impl From<reqwest::Error> for NexusError {
    fn from(e: reqwest::Error) -> Self {
        NexusError::HttpError(e.to_string())
    }
}

impl From<serde_json::Error> for NexusError {
    fn from(e: serde_json::Error) -> Self {
        NexusError::SerializationError(e.to_string())
    }
}

impl From<toml::de::Error> for NexusError {
    fn from(e: toml::de::Error) -> Self {
        NexusError::ConfigError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NexusError::ProviderError {
            provider: "openai".into(),
            message: "rate limited".into(),
            status_code: Some(429),
        };
        assert!(err.to_string().contains("openai"));
        assert!(err.to_string().contains("429"));
    }

    #[test]
    fn test_error_from_reqwest() {
        let result: NexusResult<()> = Err(NexusError::HttpError("connection refused".into()));
        assert!(result.is_err());
    }

    #[test]
    fn test_error_from_serde() {
        let bad_json = "not json";
        let err: Result<serde_json::Value, _> = serde_json::from_str(bad_json);
        let nexus_err: NexusError = err.unwrap_err().into();
        assert!(matches!(nexus_err, NexusError::SerializationError(_)));
    }

    #[test]
    fn test_rate_limited_display() {
        let err = NexusError::RateLimited {
            retry_after_ms: Some(5000),
        };
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_all_providers_failed() {
        let errors = vec![
            NexusError::Timeout(3000),
            NexusError::AuthError("invalid key".into()),
        ];
        let err = NexusError::AllProvidersFailed(errors);
        assert!(err.to_string().contains("All providers"));
    }
}
