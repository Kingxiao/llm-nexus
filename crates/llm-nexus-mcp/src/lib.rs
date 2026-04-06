//! MCP (Model Context Protocol) Gateway for llm-nexus.
//!
//! Connects to MCP servers, discovers their tools, and converts them into
//! LLM tool definitions. When a model returns a tool call targeting an MCP
//! tool, the gateway executes it and returns the result.
//!
//! JSON-RPC is implemented inline — zero external dependencies beyond tokio/serde.
//!
//! # Examples
//!
//! ```rust,no_run
//! use llm_nexus_mcp::{McpClient, McpServerConfig, McpTransport};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let config = McpServerConfig {
//!     name: "my-server".into(),
//!     transport: McpTransport::Stdio {
//!         command: "mcp-server".into(),
//!         args: vec![],
//!         env: Default::default(),
//!     },
//! };
//! let mut client = McpClient::connect(&config).await.map_err(|e| e.to_string())?;
//! let tools = client.list_tools().await.map_err(|e| e.to_string())?;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod types;

pub use client::McpClient;
pub use types::{McpServerConfig, McpTool, McpTransport};
