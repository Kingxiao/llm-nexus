use crate::error::{NexusError, NexusResult};

/// Resolve an API key from the given environment variable name.
pub fn resolve_api_key(env_var: &str) -> NexusResult<String> {
    std::env::var(env_var)
        .map_err(|_| NexusError::AuthError(format!("Environment variable {} not set", env_var)))
}

/// Build an authentication header pair `(header_name, header_value)` for the given scheme.
///
/// Supported schemes:
/// - `"Bearer"` — standard `Authorization: Bearer <key>`
/// - `""` — Anthropic-style `x-api-key: <key>`
/// - `"query_param"` — key passed as query parameter (caller handles placement)
/// - anything else — `Authorization: <scheme> <key>`
pub fn build_auth_header(scheme: &str, api_key: &str) -> (String, String) {
    match scheme {
        "Bearer" => ("Authorization".into(), format!("Bearer {api_key}")),
        "" => ("x-api-key".into(), api_key.into()),
        "query_param" => (String::new(), api_key.into()),
        _ => ("Authorization".into(), format!("{scheme} {api_key}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_api_key_missing() {
        let result = resolve_api_key("NEXUS_TEST_NONEXISTENT_KEY_XYZ");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NexusError::AuthError(_)));
    }

    #[test]
    fn test_resolve_api_key_present() {
        unsafe {
            std::env::set_var("NEXUS_TEST_AUTH_KEY", "sk-test-123");
        }
        let result = resolve_api_key("NEXUS_TEST_AUTH_KEY");
        assert_eq!(result.unwrap(), "sk-test-123");
        unsafe {
            std::env::remove_var("NEXUS_TEST_AUTH_KEY");
        }
    }

    #[test]
    fn test_build_auth_header_bearer() {
        let (name, value) = build_auth_header("Bearer", "sk-abc");
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer sk-abc");
    }

    #[test]
    fn test_build_auth_header_anthropic() {
        let (name, value) = build_auth_header("", "sk-ant-xyz");
        assert_eq!(name, "x-api-key");
        assert_eq!(value, "sk-ant-xyz");
    }

    #[test]
    fn test_build_auth_header_query_param() {
        let (name, value) = build_auth_header("query_param", "AIza-key");
        assert_eq!(name, "");
        assert_eq!(value, "AIza-key");
    }
}
