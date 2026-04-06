//! Budget management middleware for llm-nexus.
//!
//! Tracks per-identity spend against configurable limits and rejects requests
//! that would exceed the budget. Supports daily, weekly, and monthly periods.
//!
//! # Examples
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use llm_nexus_budget::{BudgetMiddleware, BudgetConfig, BudgetPeriod};
//! use llm_nexus_core::store::InMemoryStore;
//!
//! let store = Arc::new(InMemoryStore::new());
//! let config = BudgetConfig {
//!     limit_usd: 100.0,
//!     period: BudgetPeriod::Monthly,
//! };
//! let budget = BudgetMiddleware::new(store, config);
//! ```

mod middleware;
mod types;

pub use middleware::BudgetMiddleware;
pub use types::{BudgetConfig, BudgetPeriod, BudgetStatus};
