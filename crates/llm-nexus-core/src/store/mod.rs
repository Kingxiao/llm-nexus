//! Store implementations.

pub mod log_memory;
pub mod memory;

pub use log_memory::InMemoryLogBackend;
pub use memory::InMemoryStore;
