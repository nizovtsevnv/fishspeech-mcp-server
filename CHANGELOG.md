# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0] - 2026-03-10

Initial release.

### Added

- MCP server with `synthesize` tool (text-to-speech via Fish Speech 1.5)
- Voice cloning via pre-encoded `.npy` voice prompt files
- Native OGG/Opus audio encoding (libopus statically linked)
- Dual transport: stdio (default) and HTTP with Bearer auth
- Session management for HTTP transport
- Auto-download of model weights from HuggingFace Hub
- Optional CUDA GPU acceleration (`--features cuda`)
- `encode-voice` utility for creating voice prompts from reference audio
- Nix flake for reproducible builds
- CI/CD: fmt/clippy/test on push, release builds for 4 targets

[v0.1.0]: https://github.com/nizovtsevnv/fishspeech-mcp-server/releases/tag/v0.1.0
