# fishspeech-mcp-server

[![CI](https://github.com/nizovtsevnv/fishspeech-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/nizovtsevnv/fishspeech-mcp-server/actions/workflows/ci.yml)
[![Release](https://github.com/nizovtsevnv/fishspeech-mcp-server/actions/workflows/release.yml/badge.svg)](https://github.com/nizovtsevnv/fishspeech-mcp-server/actions/workflows/release.yml)

Text-to-speech MCP server powered by [Fish Speech 1.5](https://github.com/fishaudio/fish-speech).

Standalone binary that exposes a `synthesize` tool over MCP (Model Context Protocol) via stdio or HTTP transport using JSON-RPC 2.0.

## Features

- **Tool `synthesize`** — accepts text, returns base64-encoded OGG/Opus audio
- **Voice cloning** — optional voice prompt via pre-encoded `.npy` files
- **Native audio encoding** — OGG/Opus via `opus` + `ogg` crates (no ffmpeg required)
- **Auto-download** — model weights auto-download from HuggingFace Hub
- **Optional CUDA** — GPU acceleration via candle CUDA backend
- **HTTP transport** — MCP Streamable HTTP with Bearer token authentication and session management
- **Dual transport** — stdio (default) or HTTP mode via `--transport` flag

## Architecture

Single crate, six source modules:

| Module | Responsibility |
|---|---|
| `src/main.rs` | CLI parsing (clap), model loading, entry point, transport selection |
| `src/mcp.rs` | JSON-RPC 2.0 dispatch, MCP tool result helpers, stdio read/write loop |
| `src/http.rs` | HTTP transport: axum server, Bearer auth, session management |
| `src/synthesize.rs` | Fish Speech inference: text preprocessing, VQ code generation, codec decoding |
| `src/audio.rs` | Audio encoding: f32→i16 conversion, OGG/Opus encoding, resampling |
| `src/voice.rs` | Voice store: loading and managing `.npy` voice prompt files |

## CLI Arguments

```
fishspeech-mcp-server [OPTIONS]

Options:
  --checkpoint-dir <PATH>    Path to model checkpoint directory (auto-downloads from HF if not set)
  --hf-repo <REPO>           HuggingFace repo ID [default: jkeisling/fish-speech-1.5]
  --voices-dir <PATH>        Directory with voice .npy files for voice cloning
  --device <DEVICE>          Device: cpu or cuda [default: cpu]
  --temp <FLOAT>             Sampling temperature [default: 0.7]
  --top-p <FLOAT>            Top-p (nucleus) sampling [default: 0.8]
  --top-k <INT>              Top-k sampling [default: 256]
  --repetition-penalty <F>   Repetition penalty [default: 1.2]
  --max-new-tokens <INT>     Maximum new tokens to generate [default: 2048]
  --transport <MODE>         Transport mode: stdio or http [default: stdio]
  --host <HOST>              Host to bind HTTP server [default: 127.0.0.1]
  --port <PORT>              Port for HTTP server [default: 8080]
  --auth <TOKEN>             Bearer token for HTTP authentication (optional)
  --version                  Print version and exit
```

## Build

### Prerequisites

- Rust toolchain (stable)
- CMake (tokenizers/candle bindgen dependency)
- libclang (bindgen dependency)
- libopus (OGG/Opus encoding)

### Standard build

```bash
cargo build --release
```

### CUDA build

CUDA requires NVIDIA CUDA toolkit (nvcc, cudart, cuBLAS):

```bash
cargo build --release --features cuda
```

### Nix

```bash
nix build
```

## Runtime Dependencies

- **Fish Speech model** — auto-downloads from [HuggingFace Hub](https://huggingface.co/jkeisling/fish-speech-1.5), or provide via `--checkpoint-dir`

## Voice Encoding

Use the `encode-voice` utility to create voice prompt files from reference audio:

```bash
encode-voice --checkpoint-dir /path/to/model --input reference.wav --output voice.npy
```

Then use `--voices-dir` to load all `.npy` files and reference them by name in the `synthesize` tool.

## Rust Dependencies

| Crate | Purpose |
|---|---|
| `fish_speech_core` | Fish Speech 1.5 inference engine (candle-based) |
| `candle-core`, `candle-nn` | Tensor computation and neural network operations |
| `hf-hub` | HuggingFace Hub API for model auto-download |
| `tokenizers` | Tokenizer for text preprocessing |
| `clap` | CLI argument parsing |
| `serde`, `serde_json` | JSON serialization for MCP protocol |
| `base64` | Encoding audio output as base64 |
| `hound` | WAV encoding (test-only) |
| `opus`, `ogg` | Native OGG/Opus audio encoding |
| `tracing`, `tracing-subscriber` | Structured logging to stderr |
| `axum` | HTTP server framework for MCP HTTP transport |
| `tokio` | Async runtime for HTTP transport |
| `uuid` | Session ID generation (UUID v4) |
| `libc` | fd redirection for stdout suppression |

## MCP Protocol

The server supports two transport modes:

- **stdio** (default) — communicates over stdin/stdout, one JSON object per line
- **HTTP** — MCP Streamable HTTP on `POST /mcp` and `DELETE /mcp`

### HTTP Transport

Start the server in HTTP mode:

```bash
fishspeech-mcp-server --transport http --port 8080 --auth secret123
```

**Authentication**: when `--auth` is set, all requests must include `Authorization: Bearer <token>`. Without `--auth`, authentication is disabled.

**Sessions**: the `initialize` request returns an `Mcp-Session-Id` header. All subsequent requests must include this header. Sessions are terminated via `DELETE /mcp`.

### Stdio Transport

The server communicates over stdin/stdout using JSON-RPC 2.0, one JSON object per line.

### Initialize

Request:
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
```

Response:
```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"fishspeech-mcp-server","version":"<version>"}}}
```

### List tools

Request:
```json
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
```

Response:
```json
{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"synthesize","description":"Synthesize speech from text using Fish Speech 1.5","inputSchema":{"type":"object","properties":{"text":{"type":"string","description":"Text to synthesize into speech"},"voice":{"type":"string","description":"Voice name (filename without .npy extension from voices directory)"}},"required":["text"]}}]}}
```

### Synthesize

Request:
```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"synthesize","arguments":{"text":"Hello, world!","voice":"speaker1"}}}
```

Response:
```json
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"<base64-encoded OGG/Opus audio>"}]}}
```

## CI/CD

GitHub Actions workflows:

- **CI** (`ci.yml`) — runs `cargo fmt`, `cargo clippy`, `cargo test` on every push/PR to `main`/`develop`
- **Release** (`release.yml`) — builds binaries for 4 targets on tag push (`v*`), uploads as release assets

Release targets:

| Artifact | Build method | Notes |
|---|---|---|
| `linux-x86_64` | nix (default) | glibc, CPU only |
| `linux-x86_64-cuda` | cargo + CUDA toolkit | glibc, GPU acceleration |
| `macos-x86_64` | nix (default) | Intel Mac |
| `macos-arm64` | nix (default) | Apple Silicon |

Release process:
1. Create a git tag: `git tag vX.Y.Z && git push --tags`
2. CI builds binaries for all targets
3. Create a GitHub release from the tag — CI attaches build artifacts automatically

To update `cargoHash` in `flake.nix` after changing dependencies:
```bash
./scripts/update-cargo-hash.sh
```

## Usage

### Claude Desktop (stdio)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fishspeech": {
      "command": "/path/to/fishspeech-mcp-server",
      "args": ["--checkpoint-dir", "/path/to/fish-speech-1.5"]
    }
  }
}
```

### HTTP mode

```bash
fishspeech-mcp-server --transport http --port 8080 --auth mytoken
```

Connect any HTTP-capable MCP client to `http://127.0.0.1:8080/mcp`. All requests require `Authorization: Bearer mytoken` and `Content-Type: application/json`. See [HTTP Transport](#http-transport) for protocol details.

### Any MCP client (stdio)

The server reads JSON-RPC requests from stdin and writes responses to stdout. Logs go to stderr. Connect any MCP-compatible client using stdio transport.
