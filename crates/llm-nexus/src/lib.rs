//! Unified LLM adapter with intelligent routing and cost tracking.
//!
//! `llm-nexus` is the facade crate — the only dependency you need. It re-exports
//! core types, traits, and the [`NexusClient`] that dispatches requests to any
//! configured provider through a middleware pipeline.
//!
//! # Examples
//!
//! ```rust,no_run
//! use llm_nexus::prelude::*;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let client = NexusClient::from_config_dir(Path::new("config"))?;
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Flags
//!
//! Provider crates are gated behind feature flags (default: `openai`, `anthropic`):
//!
//! | Feature | Provider |
//! |---------|----------|
//! | `openai` | OpenAI |
//! | `anthropic` | Anthropic |
//! | `gemini` | Google Gemini |
//! | `deepseek` | DeepSeek (implies `openai`) |
//! | `azure` | Azure OpenAI (implies `openai`) |
//! | `bedrock` | AWS Bedrock |
//! | `vertex` | Google Vertex AI (implies `openai`) |
//! | `full` | All providers |
//! | `sqlite-metrics` | SQLite metrics backend |
//! | `prometheus-metrics` | Prometheus metrics exporter |
//! | `remote-registry` | Remote model registry sync |

pub mod client;
pub(crate) mod dispatcher;
pub mod provider_factory;
pub mod provider_map;

// Re-export core types for convenience
pub use llm_nexus_core::error::{NexusError, NexusResult};
pub use llm_nexus_core::traits;
pub use llm_nexus_core::types;

// Re-export key types at top level
pub use llm_nexus_core::traits::metrics::StatsFilter;
pub use llm_nexus_core::traits::router::RouteContext;
pub use llm_nexus_core::types::model::{Capability, ModelMetadata};
pub use llm_nexus_core::types::request::{ChatRequest, Message};
pub use llm_nexus_core::types::embed::{EmbedRequest, EmbedResponse};
pub use llm_nexus_core::types::response::{ChatResponse, StreamChunk, Usage};

// Re-export the client
pub use client::{NexusClient, NexusClientBuilder};

/// Convenience prelude for common imports.
pub mod prelude {
    pub use crate::client::NexusClient;
    pub use crate::{Capability, ModelMetadata, RouteContext, StatsFilter};
    pub use crate::{ChatRequest, ChatResponse, Message, StreamChunk, Usage};
    pub use crate::{NexusError, NexusResult};
}
