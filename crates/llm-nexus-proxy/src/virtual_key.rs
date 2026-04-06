//! Virtual key management for multi-tenant proxy deployments.
//!
//! Virtual keys are scoped API keys that map to an identity (user, team, org)
//! with optional model restrictions and budget limits.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

/// Identity resolved from a virtual key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// Unique key identifier.
    pub key_id: String,
    /// Optional user identifier.
    pub user_id: Option<String>,
    /// Optional team identifier.
    pub team_id: Option<String>,
    /// If set, only these models can be used with this key.
    pub allowed_models: Option<Vec<String>>,
    /// Budget limit in USD (per period, enforced by BudgetMiddleware).
    pub budget_limit_usd: Option<f64>,
}

/// In-memory store for virtual keys.
///
/// For production, replace with a database-backed implementation.
pub struct VirtualKeyStore {
    keys: Mutex<HashMap<String, Identity>>,
}

impl VirtualKeyStore {
    pub fn new() -> Self {
        Self {
            keys: Mutex::new(HashMap::new()),
        }
    }

    /// Register a virtual key.
    pub fn add_key(&self, token: impl Into<String>, identity: Identity) {
        self.keys
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(token.into(), identity);
    }

    /// Resolve a Bearer token to an Identity.
    /// Returns None if the token is not a registered virtual key.
    pub fn resolve(&self, token: &str) -> Option<Identity> {
        self.keys
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(token)
            .cloned()
    }
}

impl Default for VirtualKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_resolve() {
        let store = VirtualKeyStore::new();
        store.add_key(
            "vk-test-123",
            Identity {
                key_id: "k1".into(),
                user_id: Some("user-42".into()),
                team_id: None,
                allowed_models: Some(vec!["gpt-5.4".into()]),
                budget_limit_usd: Some(10.0),
            },
        );

        let id = store.resolve("vk-test-123").unwrap();
        assert_eq!(id.key_id, "k1");
        assert_eq!(id.user_id, Some("user-42".into()));
        assert_eq!(id.allowed_models.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_resolve_unknown_key() {
        let store = VirtualKeyStore::new();
        assert!(store.resolve("unknown").is_none());
    }

    #[test]
    fn test_multiple_keys() {
        let store = VirtualKeyStore::new();
        store.add_key(
            "vk-a",
            Identity {
                key_id: "a".into(),
                user_id: None,
                team_id: Some("team-frontend".into()),
                allowed_models: None,
                budget_limit_usd: None,
            },
        );
        store.add_key(
            "vk-b",
            Identity {
                key_id: "b".into(),
                user_id: None,
                team_id: Some("team-backend".into()),
                allowed_models: None,
                budget_limit_usd: None,
            },
        );

        assert_eq!(
            store.resolve("vk-a").unwrap().team_id,
            Some("team-frontend".into())
        );
        assert_eq!(
            store.resolve("vk-b").unwrap().team_id,
            Some("team-backend".into())
        );
    }
}
