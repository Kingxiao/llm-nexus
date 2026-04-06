//! Core types, traits, and middleware primitives for llm-nexus.
//!
//! This crate defines the shared vocabulary used by all provider and infrastructure
//! crates: [`ChatRequest`](types::request::ChatRequest),
//! [`ChatResponse`](types::response::ChatResponse), the [`ChatProvider`](traits::chat::ChatProvider)
//! trait, the middleware pipeline, and the unified error type.
//!
//! Application code should depend on the `llm-nexus` facade instead of this crate directly.

pub mod error;
pub mod loader;
pub mod middleware;
pub mod pipeline;
pub mod store;
pub mod traits;
pub mod types;

pub use error::{NexusError, NexusResult};
pub use types::*;
