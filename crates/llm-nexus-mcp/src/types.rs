//! MCP protocol types (JSON-RPC 2.0 subset).
//!
//! Implements just enough of the MCP protocol to support tool discovery
//! and execution. No external JSON-RPC dependencies.

use serde::{Deserialize, Serialize};

/// JSON-RPC 2.0 request.
#[derive(Debug, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: &'static str,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Deserialize)]
pub struct JsonRpcResponse {
    pub id: u64,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error.
#[derive(Debug, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

/// MCP tool definition (from tools/list response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the tool's input parameters.
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Option<serde_json::Value>,
}

/// MCP tool call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    pub content: Vec<McpContent>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

/// MCP content block (text or other).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
}

/// Configuration for connecting to an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Display name for the server.
    pub name: String,
    /// Transport type.
    pub transport: McpTransport,
}

/// MCP transport configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpTransport {
    /// Stdio transport: launch a subprocess and communicate via stdin/stdout.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: std::collections::HashMap<String, String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_request_serialize() {
        let req = JsonRpcRequest::new(1, "tools/list", None);
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["id"], 1);
        assert_eq!(json["method"], "tools/list");
        assert!(json.get("params").is_none());
    }

    #[test]
    fn test_jsonrpc_request_with_params() {
        let params = serde_json::json!({"name": "test_tool", "arguments": {}});
        let req = JsonRpcRequest::new(42, "tools/call", Some(params));
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["params"]["name"], "test_tool");
    }

    #[test]
    fn test_jsonrpc_response_deserialize_success() {
        let json = r#"{"id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_jsonrpc_response_deserialize_error() {
        let json = r#"{"id":1,"error":{"code":-32601,"message":"method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_mcp_tool_deserialize() {
        let json = r#"{"name":"get_weather","description":"Get weather","inputSchema":{"type":"object","properties":{"location":{"type":"string"}}}}"#;
        let tool: McpTool = serde_json::from_str(json).unwrap();
        assert_eq!(tool.name, "get_weather");
        assert!(tool.input_schema.is_some());
    }

    #[test]
    fn test_mcp_tool_result_deserialize() {
        let json = r#"{"content":[{"type":"text","text":"sunny"}],"isError":false}"#;
        let result: McpToolResult = serde_json::from_str(json).unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn test_server_config_serialize() {
        let config = McpServerConfig {
            name: "test-server".into(),
            transport: McpTransport::Stdio {
                command: "node".into(),
                args: vec!["server.js".into()],
                env: std::collections::HashMap::new(),
            },
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["name"], "test-server");
    }
}
