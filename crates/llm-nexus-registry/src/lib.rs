//! Model metadata registry for llm-nexus.
//!
//! Stores model capabilities, pricing, and context window sizes. The
//! [`StaticRegistry`] loads from TOML config at startup; enable the `remote-sync`
//! feature for periodic sync from a remote endpoint.
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | _(default)_ | [`StaticRegistry`] — TOML-based, loaded at init |
//! | `remote-sync` | [`RemoteRegistry`] — periodic HTTP refresh |

pub mod filter;
#[cfg(feature = "remote-sync")]
pub mod remote;
pub mod static_registry;

#[cfg(feature = "remote-sync")]
pub use remote::RemoteRegistry;
pub use static_registry::StaticRegistry;
