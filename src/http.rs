use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use candle_core::Tensor;
use serde_json::Value;

use crate::mcp;
use crate::synthesize::SynthesisConfig;
use crate::voice::VoiceStore;
use crate::FishSpeechModel;

pub struct AppState {
    pub model: Mutex<FishSpeechModel>,
    pub voices: VoiceStore,
    pub synthesis_config: SynthesisConfig,
    pub tools: Value,
    pub auth_token: Option<String>,
    pub sessions: Mutex<HashSet<String>>,
}

pub async fn run_http_server(state: Arc<AppState>, host: &str, port: u16) {
    let app = Router::new()
        .route("/mcp", post(handle_post).delete(handle_delete))
        .with_state(state);

    let addr = format!("{host}:{port}");
    tracing::info!("HTTP server listening on {addr}");

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(error = %e, addr = %addr, "failed to bind HTTP listener");
            std::process::exit(1);
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!(error = %e, "HTTP server error");
        std::process::exit(1);
    }
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), StatusCode> {
    let expected = match &state.auth_token {
        Some(t) => t,
        None => return Ok(()),
    };

    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if let Some(token) = auth.strip_prefix("Bearer ") {
        if token == expected.as_str() {
            return Ok(());
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

fn get_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

async fn handle_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    if let Err(status) = check_auth(&state, &headers) {
        return status.into_response();
    }

    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.contains("application/json") {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        )
            .into_response();
    }

    // Peek at the method to determine if this is an initialize request
    let is_initialize = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("method").and_then(|m| m.as_str()).map(String::from))
        .as_deref()
        == Some("initialize");

    if !is_initialize {
        let session_id = match get_session_id(&headers) {
            Some(id) => id,
            None => {
                return (StatusCode::BAD_REQUEST, "Missing Mcp-Session-Id header").into_response()
            }
        };

        let sessions = state.sessions.lock().expect("session lock poisoned");
        if !sessions.contains(&session_id) {
            return (StatusCode::BAD_REQUEST, "Invalid session ID").into_response();
        }
    }

    let state_clone = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        mcp::dispatch_request(&body, &state_clone.tools, |id, params| {
            handle_tools_call(id, params, &state_clone)
        })
    })
    .await
    .expect("dispatch task panicked");

    match result {
        Some(response_json) => {
            let mut builder = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json");

            if is_initialize {
                let session_id = uuid::Uuid::new_v4().to_string();
                state
                    .sessions
                    .lock()
                    .expect("session lock poisoned")
                    .insert(session_id.clone());
                builder = builder.header("mcp-session-id", &session_id);
            }

            builder.body(Body::from(response_json)).unwrap()
        }
        None => (StatusCode::ACCEPTED, "").into_response(),
    }
}

fn handle_tools_call(id: Value, params: Option<Value>, state: &AppState) -> mcp::JsonRpcResponse {
    let tcp = match mcp::extract_tool_params(id.clone(), params) {
        Ok(p) => p,
        Err(resp) => return resp,
    };
    let arguments = Value::Object(tcp.arguments);

    match tcp.tool_name.as_str() {
        "synthesize" => {
            let text = match arguments.get("text").and_then(|v| v.as_str()) {
                Some(t) => t,
                None => return mcp::tool_error(id, "missing required argument: text"),
            };

            let voice_name = arguments.get("voice").and_then(|v| v.as_str());
            let voice_tensor: Option<&Tensor> = match voice_name {
                Some(name) => match state.voices.get(name) {
                    Some(t) => Some(t),
                    None => {
                        let available = state.voices.names();
                        return mcp::tool_error(
                            id,
                            &format!(
                                "unknown voice: {name}. Available: {}",
                                available
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        );
                    }
                },
                None => None,
            };

            let mut model = state.model.lock().expect("model lock poisoned");
            match crate::synthesize::synthesize(
                text,
                voice_tensor,
                &mut model,
                &state.synthesis_config,
            ) {
                Ok(b64) => mcp::tool_result(id, &b64),
                Err(e) => mcp::tool_error(id, &e),
            }
        }
        _ => mcp::JsonRpcResponse::err(id, -32602, format!("unknown tool: {}", tcp.tool_name)),
    }
}

async fn handle_delete(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(status) = check_auth(&state, &headers) {
        return status.into_response();
    }

    let session_id = match get_session_id(&headers) {
        Some(id) => id,
        None => return (StatusCode::BAD_REQUEST, "Missing Mcp-Session-Id header").into_response(),
    };

    let mut sessions = state.sessions.lock().expect("session lock poisoned");
    if sessions.remove(&session_id) {
        (StatusCode::OK, "Session terminated").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Session not found").into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_auth_valid() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer secret123".parse().unwrap());
        assert!(check_auth_standalone(Some("secret123"), &headers).is_ok());
    }

    #[test]
    fn check_auth_invalid() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer wrong".parse().unwrap());
        assert!(check_auth_standalone(Some("secret123"), &headers).is_err());
    }

    #[test]
    fn check_auth_missing() {
        let headers = HeaderMap::new();
        assert!(check_auth_standalone(Some("secret123"), &headers).is_err());
    }

    #[test]
    fn no_token_configured() {
        let headers = HeaderMap::new();
        assert!(check_auth_standalone(None, &headers).is_ok());
    }

    #[test]
    fn session_lifecycle() {
        let sessions: Mutex<HashSet<String>> = Mutex::new(HashSet::new());

        let session_id = uuid::Uuid::new_v4().to_string();
        sessions.lock().unwrap().insert(session_id.clone());
        assert!(sessions.lock().unwrap().contains(&session_id));

        assert!(sessions.lock().unwrap().remove(&session_id));
        assert!(!sessions.lock().unwrap().contains(&session_id));
    }

    #[test]
    fn missing_session_id() {
        let headers = HeaderMap::new();
        assert!(get_session_id(&headers).is_none());
    }

    #[test]
    fn get_session_id_present() {
        let mut headers = HeaderMap::new();
        headers.insert("mcp-session-id", "test-id-123".parse().unwrap());
        assert_eq!(get_session_id(&headers), Some("test-id-123".to_string()));
    }

    fn check_auth_standalone(token: Option<&str>, headers: &HeaderMap) -> Result<(), StatusCode> {
        let expected = match token {
            Some(t) => t,
            None => return Ok(()),
        };

        let auth = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if let Some(bearer_token) = auth.strip_prefix("Bearer ") {
            if bearer_token == expected {
                return Ok(());
            }
        }

        Err(StatusCode::UNAUTHORIZED)
    }
}
