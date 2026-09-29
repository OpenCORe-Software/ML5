# Changelog

All notable changes to ML5 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-28

### Added
- Initial release of ML5: local GGUF inference with a Rust CLI and daemon.
- `ml5` CLI with commands: `run`, `pull`, `list`, `status`, `unload`, `delete`, `hf`, `backend`.
- `ml5d` daemon hosting native (`/api/*`) and OpenAI-compatible (`/v1/*`) endpoints.
- Native streaming via SSE with status events and usage metadata.
- OpenAI-style streaming and non-streaming responses.
- Hugging Face model pulls with file selection, progress reporting, and GGUF verification.
- Interactive chat with `/help`, `/clear`, `/model`, `/exit`, `/quit` commands.
- GPU backend detection and dynamic backend download support.
- Memory budgeting with configurable `max_memory_fraction`.
- Windows installer script (`install.ps1`).

### Known Limitations
- CORe registry pulls are not yet implemented.
- No resumable downloads.
- No KV cache reuse or continuous batching.
- No split-GGUF download support.
- No cryptographic checksum verification (content length + GGUF magic only).
- `keep_alive_secs` is not yet enforced by a background reaper.
