//! Request routing strategies for llm-nexus.
//!
//! Provides pluggable routers that select the best provider/model for each request
//! based on cost, latency, health, or custom scoring. Includes automatic fallback
//! and cooldown on provider errors.
//!
//! # Examples
//!
//! ```rust,no_run
//! use llm_nexus_router::{CostRouter, CooldownRouter};
//! ```

pub mod composite;
pub mod cooldown;
pub mod cost_router;
pub mod experiment;
pub mod fallback;
pub mod scorer;

pub use composite::{WeightedScorer, composite_router, latency_router};
pub use cooldown::{CooldownRouter, ProviderHealthState};
pub use cost_router::CostRouter;
pub use experiment::ExperimentRouter;
