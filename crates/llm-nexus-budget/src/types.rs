//! Budget configuration and status types.

use chrono::Datelike;
use serde::{Deserialize, Serialize};

/// Budget limit configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Maximum spend in USD for the period.
    pub limit_usd: f64,
    /// Budget period (when the counter resets).
    pub period: BudgetPeriod,
}

/// Budget period for cost accumulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetPeriod {
    /// Never resets (lifetime budget).
    Total,
    /// Resets daily at midnight UTC.
    Daily,
    /// Resets monthly on the 1st.
    Monthly,
}

/// Current budget status for an identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    /// Accumulated spend in this period (USD).
    pub spent_usd: f64,
    /// Remaining budget (USD).
    pub remaining_usd: f64,
    /// Whether the budget is exceeded.
    pub exceeded: bool,
    /// Start of the current budget period (UTC timestamp).
    /// Used by Daily/Monthly to determine when to reset.
    pub period_start: chrono::DateTime<chrono::Utc>,
}

impl BudgetStatus {
    /// Create a fresh status for a new budget period.
    pub fn new(limit_usd: f64) -> Self {
        Self {
            spent_usd: 0.0,
            remaining_usd: limit_usd,
            exceeded: false,
            period_start: chrono::Utc::now(),
        }
    }
}

impl BudgetPeriod {
    /// Check if the current period has elapsed since `period_start`.
    pub fn should_reset(&self, period_start: chrono::DateTime<chrono::Utc>) -> bool {
        let now = chrono::Utc::now();
        match self {
            BudgetPeriod::Total => false,
            BudgetPeriod::Daily => period_start.date_naive() < now.date_naive(),
            BudgetPeriod::Monthly => {
                period_start.date_naive().month() != now.date_naive().month()
                    || period_start.date_naive().year() != now.date_naive().year()
            }
        }
    }
}
