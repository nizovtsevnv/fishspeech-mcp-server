use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::Parser;
use serde_json::Value;

use fish_speech_core::codec::{FireflyCodec, FireflyConfig};
use fish_speech_core::config::{WhichFishVersion, WhichLM, WhichModel};
use fish_speech_core::lm::dual_ar::TokenConfig;
use fish_speech_core::lm::{BaseModelArgs, DualARTransformer};

mod audio;
mod http;
mod mcp;
mod synthesize;
mod voice;

use synthesize::SynthesisConfig;
use voice::VoiceStore;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "fishspeech-mcp-server", version)]
struct Args {
    /// Path to model checkpoint directory (auto-downloads from HF if not set)
    #[arg(long)]
    checkpoint_dir: Option<String>,

    /// HuggingFace repo ID for model auto-download
    #[arg(long, default_value = "jkeisling/fish-speech-1.5")]
    hf_repo: String,

    /// Directory with voice .npy files for voice cloning
    #[arg(long)]
    voices_dir: Option<String>,

    /// Device: "cpu" or "cuda"
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.7)]
    temp: f64,

    /// Top-p (nucleus) sampling
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,

    /// Top-k sampling
    #[arg(long, default_value_t = 256)]
    top_k: usize,

    /// Repetition penalty
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f32,

    /// Maximum new tokens to generate
    #[arg(long, default_value_t = 2048)]
    max_new_tokens: usize,

    /// Transport mode: stdio or http
    #[arg(long, default_value = "stdio")]
    transport: String,

    /// Host to bind HTTP server (http transport only)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port for HTTP server (http transport only)
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Bearer token for HTTP authentication (http transport only)
    #[arg(long)]
    auth: Option<String>,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct FishSpeechModel {
    pub lm: DualARTransformer,
    pub codec: FireflyCodec,
    pub tokenizer: tokenizers::Tokenizer,
    pub config: BaseModelArgs,
    pub lm_version: WhichLM,
    pub device: Device,
}

fn resolve_checkpoint_dir(explicit_path: Option<&str>, hf_repo: &str) -> Result<PathBuf, String> {
    if let Some(path) = explicit_path {
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(format!("checkpoint directory not found: {}", p.display()));
        }
        return Ok(p);
    }

    tracing::info!(repo = %hf_repo, "resolving model from HuggingFace Hub");

    let api =
        hf_hub::api::sync::Api::new().map_err(|e| format!("failed to create HF Hub API: {e}"))?;
    let repo = api.model(hf_repo.to_string());

    let required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors",
    ];

    let mut resolved_dir = None;
    for filename in &required_files {
        let path = repo
            .get(filename)
            .map_err(|e| format!("failed to download {filename}: {e}"))?;

        if resolved_dir.is_none() {
            resolved_dir = path.parent().map(|p| p.to_path_buf());
        }
    }

    resolved_dir.ok_or_else(|| "failed to resolve checkpoint directory from HF Hub".into())
}

fn load_model(checkpoint_dir: &Path, device: &Device) -> Result<FishSpeechModel, String> {
    let lm_version = WhichLM::from_model(WhichModel::Fish1_5);
    let fish_version = WhichFishVersion::Fish1_5;

    let config = BaseModelArgs::from_file(checkpoint_dir.join("config.json"))
        .map_err(|e| format!("failed to load config.json: {e}"))?;

    let tokenizer = tokenizers::Tokenizer::from_file(checkpoint_dir.join("tokenizer.json"))
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;

    let token_config = TokenConfig::new(lm_version, &tokenizer, &config)
        .map_err(|e| format!("failed to create token config: {e}"))?;

    let lm_path = checkpoint_dir.join("model.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[lm_path], DType::F32, device)
            .map_err(|e| format!("failed to load LM weights: {e}"))?
    };
    let lm = DualARTransformer::load(&vb, &config, &token_config, lm_version)
        .map_err(|e| format!("failed to load DualARTransformer: {e}"))?;

    let codec_path = checkpoint_dir.join("firefly-gan-vq-fsq-8x1024-21hz-generator.safetensors");
    let codec_config = FireflyConfig::get_config_for(fish_version);
    let codec_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[codec_path], DType::F32, device)
            .map_err(|e| format!("failed to load codec weights: {e}"))?
    };
    let codec = FireflyCodec::load(codec_config, codec_vb, fish_version)
        .map_err(|e| format!("failed to load FireflyCodec: {e}"))?;

    Ok(FishSpeechModel {
        lm,
        codec,
        tokenizer,
        config,
        lm_version,
        device: device.clone(),
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(
                "fishspeech_mcp_server=info"
                    .parse()
                    .expect("valid directive"),
            ),
        )
        .init();

    let args = Args::parse();

    // Resolve device
    let device = if args.device == "cuda" {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    tracing::info!(device = ?device, "selected compute device");

    // Resolve checkpoint directory (explicit path or HF Hub download)
    let checkpoint_dir = match resolve_checkpoint_dir(args.checkpoint_dir.as_deref(), &args.hf_repo)
    {
        Ok(dir) => dir,
        Err(e) => {
            tracing::error!(error = %e, "failed to resolve model checkpoint");
            std::process::exit(1);
        }
    };

    tracing::info!(path = %checkpoint_dir.display(), "checkpoint directory resolved");

    // Load model
    let mut model = match load_model(&checkpoint_dir, &device) {
        Ok(m) => m,
        Err(e) => {
            tracing::error!(error = %e, "failed to load Fish Speech model");
            std::process::exit(1);
        }
    };

    tracing::info!("Fish Speech model loaded");

    // Load voice prompts
    let voices = match &args.voices_dir {
        Some(dir) => match VoiceStore::load(Path::new(dir), &device, model.config.num_codebooks) {
            Ok(store) => {
                tracing::info!(count = store.len(), "voice store loaded");
                store
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to load voices, continuing without");
                VoiceStore::empty()
            }
        },
        None => VoiceStore::empty(),
    };

    let synthesis_config = SynthesisConfig {
        temp: args.temp,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        max_new_tokens: args.max_new_tokens,
    };

    tracing::info!("Fish Speech MCP server ready");

    let tools = serde_json::json!({
        "tools": [{
            "name": "synthesize",
            "description": "Synthesize speech from text using Fish Speech 1.5",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to synthesize into speech"
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice name (filename without .npy extension from voices directory)"
                    }
                },
                "required": ["text"]
            }
        }]
    });

    match args.transport.as_str() {
        "stdio" => {
            mcp::run_stdio_loop(tools, |id: Value, params: Option<Value>| {
                handle_tools_call(id, params, &mut model, &voices, &synthesis_config)
            });
        }
        "http" => {
            let state = Arc::new(http::AppState {
                model: Mutex::new(model),
                voices,
                synthesis_config,
                tools,
                auth_token: args.auth.clone(),
                sessions: Mutex::new(HashSet::new()),
            });

            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::error!(error = %e, "failed to create tokio runtime");
                    std::process::exit(1);
                }
            };
            rt.block_on(http::run_http_server(state, &args.host, args.port));
        }
        other => {
            eprintln!("Unknown transport: {other}");
            std::process::exit(1);
        }
    }
}

fn handle_tools_call(
    id: Value,
    params: Option<Value>,
    model: &mut FishSpeechModel,
    voices: &VoiceStore,
    config: &SynthesisConfig,
) -> mcp::JsonRpcResponse {
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

            let voice_tensor = match voice_name {
                Some(name) => match voices.get(name) {
                    Some(t) => Some(t),
                    None => {
                        let available = voices.names();
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

            match synthesize::synthesize(text, voice_tensor, model, config) {
                Ok(b64) => mcp::tool_result(id, &b64),
                Err(e) => mcp::tool_error(id, &e),
            }
        }
        _ => mcp::JsonRpcResponse::err(id, -32602, format!("unknown tool: {}", tcp.tool_name)),
    }
}
