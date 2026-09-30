# Changelog

All notable changes to ML5 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-09-30

### Changed
- Release binaries are now statically linked (no `gpu`/dynamic-link feature), so
  `ml5`/`ml5d` run standalone on every platform with no shared-library hunting.
  GPU offload via dynamic backends is an opt-in local build (`--features gpu`).

### Removed
- Legacy-CUDA backend and installer. CUDA 11.7 (the last Pascal-capable toolkit)
  cannot be built on modern Windows toolchains, and its upstream DLL predates the
  bindings' API. Pascal-era NVIDIA GPUs (GTX 10-series) use the Vulkan backend.

### Fixed
- Installer places core llama/ggml runtime libraries beside the binaries on
  dynamic (gpu-feature) installs, so the daemon starts instead of failing with a
  missing-library error.

## [0.2.0] - 2026-09-29

### Massive overhaul

### Added
- Continuous batching: `ml5d --parallel [--n-parallel N]` runs a shared-context,
  multi-sequence decode loop so concurrent requests are batched together instead of
  serialized. Each slot gets the full `--ctx-size` of KV cache; total is `ctx × N`.
  Defaults to CPU core count (capped at 8) when N is omitted.
- Prefix caching (KV reuse across requests):
  - `--manual-kv`: a request's prompt prefix is kept in cache only when the request
    sets `cache: true` (native API) or `options.cache` / `keep_in_cache` (Ollama).
    Follow-up requests with a matching prefix skip re-ingesting it.
  - `--auto-kv`: cache every request's prefix automatically.
  - Works in both serial and `--parallel` mode (per-slot prefix matching).
- Ollama-compatible model management endpoints under `/ollama`:
  `POST /api/pull` (streaming progress), `DELETE /api/delete`, `POST /api/copy`,
  and unload via `keep_alive: 0` on `/api/generate` and `/api/chat`.
- Linux installer (`install.sh`) with systemd / OpenRC / runit / sysvinit detection.
- macOS installer (`install-macos.sh`) registering a launchd LaunchDaemon.
- Installer falls back to the GitHub release CDN for daemon binaries when the
  update server is unreachable.

### Changed
- CI release builds now compile with `--features gpu,safetensors`, so released
  `ml5d` binaries can load CUDA/Vulkan/CPU llama.cpp backends at runtime
  (dynamic backends). One `ml5d` binary serves all backends.

### Fixed
- Parallel worker no longer dies on a single bad request; the loop is restartable
  and a shared batch token budget prevents llama.cpp batch overflows under
  concurrent load.

### Performance
- `last_used` tracking switched from a per-token `Mutex` to a relaxed atomic.
- Harmony (gpt-oss) output filter no longer allocates a `String` per token.
- Stop-sequence filter fast path when no stop sequences are configured, and the
  suffix scan is capped to the longest stop length.

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
