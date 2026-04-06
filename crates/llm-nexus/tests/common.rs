//! Shared utilities for integration tests.
//!
//! All integration tests require `NEXUS_INTEGRATION=1` to run.
//! They hit real provider APIs and require valid API keys in the environment.

use std::path::Path;
use std::sync::Arc;

use llm_nexus::NexusClient;

/// Skip the test if NEXUS_INTEGRATION is not set.
macro_rules! skip_unless_integration {
    () => {
        if std::env::var("NEXUS_INTEGRATION").is_err() {
            eprintln!("NEXUS_INTEGRATION not set, skipping integration test");
            return;
        }
    };
}

pub(crate) use skip_unless_integration;

/// Build a NexusClient from the project's config directory.
pub fn build_client() -> Arc<NexusClient> {
    let config_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("config");
    let client =
        NexusClient::from_config_dir(&config_dir).expect("failed to build client from config dir");
    Arc::new(client)
}
