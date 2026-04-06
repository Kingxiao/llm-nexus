//! MCP client — communicates with MCP servers via stdio transport.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::types::{JsonRpcRequest, JsonRpcResponse, McpServerConfig, McpTool, McpTransport};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// A client connected to a single MCP server process.
pub struct McpClient {
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    _child: Child,
    request_id: AtomicU64,
    timeout: Duration,
}

impl McpClient {
    /// Spawn an MCP server process and initialize the connection.
    pub async fn connect(config: &McpServerConfig) -> Result<Self, String> {
        Self::connect_with_timeout(config, DEFAULT_TIMEOUT).await
    }

    pub async fn connect_with_timeout(
        config: &McpServerConfig,
        timeout: Duration,
    ) -> Result<Self, String> {
        let McpTransport::Stdio {
            ref command,
            ref args,
            ref env,
        } = config.transport;

        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(env.iter())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn MCP server: {e}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or("MCP server stdin not available")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("MCP server stdout not available")?;

        let mut client = Self {
            stdin,
            reader: BufReader::new(stdout),
            _child: child,
            request_id: AtomicU64::new(1),
            timeout,
        };

        client.initialize().await?;
        Ok(client)
    }

    async fn initialize(&mut self) -> Result<(), String> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "llm-nexus",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let _resp = self.send_request("initialize", Some(params)).await?;
        self.send_notification("notifications/initialized", None)
            .await?;
        Ok(())
    }

    /// Discover available tools from the MCP server.
    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>, String> {
        let resp = self.send_request("tools/list", None).await?;
        let tools = resp
            .result
            .and_then(|r| r.get("tools").cloned())
            .and_then(|t| serde_json::from_value::<Vec<McpTool>>(t).ok())
            .unwrap_or_default();
        Ok(tools)
    }

    /// Call a tool on the MCP server.
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });
        let resp = self.send_request("tools/call", Some(params)).await?;
        if let Some(err) = resp.error {
            return Err(format!("MCP tool error: {}", err.message));
        }
        Ok(resp.result.unwrap_or(serde_json::Value::Null))
    }

    async fn send_request(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<JsonRpcResponse, String> {
        let id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(id, method, params);
        let mut line =
            serde_json::to_string(&request).map_err(|e| format!("serialize error: {e}"))?;
        line.push('\n');

        self.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| format!("write error: {e}"))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| format!("flush error: {e}"))?;

        // Read responses until we find one matching our request ID.
        // Skip notifications (lines without "id" or with non-matching id).
        loop {
            let mut response_line = String::new();
            let read_result = tokio::time::timeout(
                self.timeout,
                self.reader.read_line(&mut response_line),
            )
            .await
            .map_err(|_| format!("MCP server response timeout after {:?}", self.timeout))?
            .map_err(|e| format!("read error: {e}"))?;

            if read_result == 0 {
                return Err("MCP server closed stdout".into());
            }

            // Try to parse as JSON-RPC response with matching ID
            match serde_json::from_str::<JsonRpcResponse>(&response_line) {
                Ok(resp) if resp.id == id => return Ok(resp),
                _ => {} // notification, wrong ID, or parse error — skip
            }
            // Not a valid response (notification or garbage) — skip and read next line
        }
    }

    async fn send_notification(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params.unwrap_or(serde_json::json!({})),
        });
        let mut line =
            serde_json::to_string(&notif).map_err(|e| format!("serialize error: {e}"))?;
        line.push('\n');

        self.stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| format!("write error: {e}"))?;
        self.stdin
            .flush()
            .await
            .map_err(|e| format!("flush error: {e}"))?;
        Ok(())
    }
}
