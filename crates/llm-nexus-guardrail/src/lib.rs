//! Guardrails middleware for llm-nexus.
//!
//! Pluggable pre/post checks on chat requests and responses.
//! Ships with [`KeywordFilter`] and [`RegexFilter`]; implement
//! [`GuardrailCheck`] for custom validation logic.
//!
//! # Examples
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use llm_nexus_guardrail::{GuardrailMiddleware, KeywordFilter};
//!
//! let filter = Arc::new(KeywordFilter::new(vec!["password".into()]));
//! let guardrail = GuardrailMiddleware::new(vec![filter]);
//! ```

mod checks;
mod middleware;

pub use checks::{KeywordFilter, RegexFilter};
pub use middleware::{GuardrailCheck, GuardrailMiddleware, GuardrailVerdict};
