//! Middleware pipeline for intercepting chat requests.
//!
//! The pipeline sits between [`NexusClient`] and the provider, enabling
//! cross-cutting concerns (caching, guardrails, logging, retry, etc.)
//! without modifying provider or client code.

pub mod builtin;
pub mod context;
pub mod middleware;
#[cfg(test)]
pub(crate) mod test_helpers;

pub use context::RequestContext;
pub use middleware::{ChatMiddleware, Next, NextStream};
