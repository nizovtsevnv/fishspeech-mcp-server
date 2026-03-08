use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct JsonRpcRequest {
    pub id: Option<Value>,
    pub method: String,
    pub params: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

// ---------------------------------------------------------------------------
// MCP tool result helpers
// ---------------------------------------------------------------------------

pub fn tool_result(id: Value, text: &str) -> JsonRpcResponse {
    JsonRpcResponse::ok(
        id,
        serde_json::json!({
            "content": [{"type": "text", "text": text}]
        }),
    )
}

pub fn tool_error(id: Value, message: &str) -> JsonRpcResponse {
    JsonRpcResponse::ok(
        id,
        serde_json::json!({
            "content": [{"type": "text", "text": message}],
            "isError": true
        }),
    )
}

pub struct ToolCallParams {
    pub tool_name: String,
    pub arguments: serde_json::Map<String, Value>,
}

pub fn extract_tool_params(
    id: Value,
    params: Option<Value>,
) -> Result<ToolCallParams, JsonRpcResponse> {
    let params = match params {
        Some(p) => p,
        None => return Err(JsonRpcResponse::err(id, -32602, "missing params".into())),
    };
    let tool_name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let arguments = params
        .get("arguments")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    Ok(ToolCallParams {
        tool_name,
        arguments,
    })
}

pub fn handle_initialize(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::ok(
        id,
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "fishspeech-mcp-server",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

/// Dispatch a single JSON-RPC request string. Returns None for notifications.
pub fn dispatch_request<F>(
    request_json: &str,
    tools_list_response: &Value,
    mut handle_call: F,
) -> Option<String>
where
    F: FnMut(Value, Option<Value>) -> JsonRpcResponse,
{
    let request: JsonRpcRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(error = %e, "invalid JSON-RPC request");
            let resp = JsonRpcResponse::err(Value::Null, -32700, "Parse error".into());
            return serde_json::to_string(&resp).ok();
        }
    };

    // Notifications have no id — don't respond.
    let id = match request.id {
        Some(id) if !id.is_null() => id,
        _ => {
            tracing::debug!(method = %request.method, "received notification");
            return None;
        }
    };

    let response = match request.method.as_str() {
        "initialize" => handle_initialize(id),
        "tools/list" => JsonRpcResponse::ok(id, tools_list_response.clone()),
        "tools/call" => handle_call(id, request.params),
        _ => JsonRpcResponse::err(id, -32601, format!("method not found: {}", request.method)),
    };

    serde_json::to_string(&response).ok()
}

// ---------------------------------------------------------------------------
// MCP stdio loop
// ---------------------------------------------------------------------------

pub fn run_stdio_loop<F>(tools_list_response: Value, mut handle_call: F)
where
    F: FnMut(Value, Option<Value>) -> JsonRpcResponse,
{
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(error = %e, "stdin read failed");
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        if let Some(mut json) = dispatch_request(&line, &tools_list_response, |id, params| {
            handle_call(id, params)
        }) {
            json.push('\n');

            if let Err(e) = stdout.write_all(json.as_bytes()) {
                tracing::error!(error = %e, "stdout write failed");
                break;
            }
            if let Err(e) = stdout.flush() {
                tracing::error!(error = %e, "stdout flush failed");
                break;
            }
        }
    }

    tracing::info!("MCP server shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonrpc_request_deserialize() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "initialize");
    }

    #[test]
    fn jsonrpc_notification_no_id() {
        let json = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert!(req.id.is_none());
    }

    #[test]
    fn jsonrpc_response_serialize() {
        let resp = JsonRpcResponse::ok(Value::from(1), serde_json::json!({"result": true}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn jsonrpc_error_response() {
        let resp = JsonRpcResponse::err(Value::from(1), -32602, "missing params".into());
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32602"));
    }

    #[test]
    fn tool_result_format() {
        let resp = tool_result(Value::from(1), "hello world");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("hello world"));
        assert!(!json.contains("isError"));
    }

    #[test]
    fn tool_error_format() {
        let resp = tool_error(Value::from(1), "something broke");
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("something broke"));
        assert!(json.contains("isError"));
    }

    #[test]
    fn extract_tool_params_success() {
        let params = serde_json::json!({
            "name": "synthesize",
            "arguments": {"text": "hello"}
        });
        let tcp = extract_tool_params(Value::from(1), Some(params)).unwrap();
        assert_eq!(tcp.tool_name, "synthesize");
        assert_eq!(
            tcp.arguments.get("text").and_then(|v| v.as_str()),
            Some("hello")
        );
    }

    #[test]
    fn extract_tool_params_missing() {
        let result = extract_tool_params(Value::from(1), None);
        assert!(result.is_err());
    }

    #[test]
    fn initialize_response() {
        let resp = handle_initialize(Value::from(1));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("fishspeech-mcp-server"));
        assert!(json.contains("protocolVersion"));
    }
}
