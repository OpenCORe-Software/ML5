# Changelog

All notable changes to ML5 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-10-14

### Added
- Repeat `ml5d --models-dir` to combine model stores from multiple drives in every
  model list. The first directory remains the download destination. Duplicate
  model names receive distinct store-prefixed names, repeated locations are
  deduplicated, and unavailable additional drives can be reconnected and rescanned.
- Status output lists all model directories; native model listings include each
  model's original path.
- `ml5 rescan` refreshes every configured model directory without restarting the
  daemon, including drives reconnected after startup.

### Fixed
- Windows GPU detection now finds NVIDIA cards even when `nvidia-smi` is
  unavailable, and locates the Vulkan loader using the Windows system directory.
- Windows prefers the installer's selected runtime over other bundled runtimes.
  Backend repair and swaps replace the runtime as a complete set, preventing
  stale CUDA, Vulkan, or CPU libraries from mixing. Repair preserves the existing
  backend selection.
- Daemon logs show the actual GPU layer count reported by llama.cpp and warn
  when requested GPU offload loads no model layers onto the GPU.

## [0.3.2] - 2026-10-14

### Added
- `--opt-cache-prompt`: prompt caching becomes per-request opt-in. Requests on
  the native and OpenAI chat/completions endpoints set `cache_prompt: true` to
  reuse a cached prefix; the CLI has a matching `--cache-prompt` flag.

### Fixed
- Ollama `/api/tags` now reports model capabilities (including `vision`), so
  clients can detect image-capable models from the model list.

## [0.3.1] - 2026-10-14

### Added
- `--no-cache-prompt` and `--cache-reuse N` daemon options controlling
  llama-server's prompt (prefix) caching.

### Fixed
- The memory budget is a fraction of total RAM+VRAM (previously free memory),
  so `--max-memory-fraction` means what it says.
- `ml5` normalizes wildcard hosts (`0.0.0.0`, `[::]`) in `--host`/`ML5_HOST` to
  loopback, fixing "Cannot reach ML5" on Windows when the daemon binds all
  interfaces.
- Windows: llama-server children are assigned to a kill-on-close job object, so
  they no longer outlive ml5d when it is force-stopped (`ml5d --stop`,
  taskkill, crash). Model unload on Windows also no longer waits out a 10 s
  grace period that could never elapse.

## [0.3.0] - 2026-10-11

### llama-server engine

ML5 now runs GGUF models in llama.cpp's own `llama-server` (pinned release b10687,
`--jinja`), one child process per loaded model, instead of in-process bindings.
Builds no longer need CMake, libclang, or a C++ toolchain.

### Added
- Tool calling on every API: OpenAI `tools`/`tool_choice`/`parallel_tool_calls`,
  Anthropic `tool_use`/`tool_result`, Ollama `tools`, and Responses
  `function_call`/`function_call_output`, streamed in each API's native shape.
- Vision through `mmproj` projectors: images on OpenAI, Anthropic, Ollama, Responses
  and native requests. Pulls download the best projector (f16 > bf16 > q8_0 > f32)
  next to the model and pair it automatically.
- Reasoning output separated from the answer (`reasoning_content`, `thinking`
  blocks, Responses reasoning items) with per-request `think`, `reasoning_effort`
  and Anthropic `thinking` controls.
- Structured output: `response_format` JSON mode / JSON schema, Ollama `format`,
  Responses `text.format`.
- KV-cache reuse on every API, reported as `cached_tokens` /
  `cache_read_input_tokens`.
- Several loaded models at once (`--max-loaded-models`, default 2) with LRU
  eviction, per-request and default keep-alive (`--keep-alive 15m`, `-1` = never),
  crash detection, and in-flight protection so a busy model is never evicted.
- Automatic context size from free RAM/VRAM and the model's KV footprint
  (`--ctx-size 0`, `--auto-ctx-max`), and `--gpu-layers auto|all|N` with
  llama.cpp's fit-to-VRAM placement.
- Runtime management: `llama-server` is located (`--llama-server`,
  `ML5_LLAMA_SERVER`, beside the binary, `/usr/lib/ml5/llama.cpp`,
  `~/.ml5/runtime`, `PATH`) or downloaded and SHA-256-verified on first use.
- Pull improvements: resumable downloads, SHA-256 verification against Hugging
  Face LFS hashes, split GGUF sets, `owner/repo:QUANT`, aggregate progress,
  cancellation (`/api/pull/cancel`, `ml5 cancel`), listing (`/api/pulls`,
  `ml5 pulls`), and re-attaching to a pull already in progress.
- Model details from GGUF metadata: architecture, parameter size, quantization,
  trained context, and capabilities (completion, embedding, tools, thinking,
  vision) in `/api/models`, `/v1/models`, `/ollama/api/tags|show|ps`.
- Endpoints: `/api/load`, `/api/rescan`, `/v1/models/{id}`, `/ollama/api/embed`.
- Exact `/v1/messages/count_tokens` using the model's tokenizer and template.
- CLI: `ml5 load`, `ml5 ps`, `ml5 pulls`, `ml5 cancel`, `--image`, `--think`,
  `--keep-alive`, interactive `/think` and `/image`, dimmed reasoning output,
  richer `list` and `status`, and `ml5 backend detect|status|download` for the
  llama.cpp runtime.
- Safetensors models use the repository's chat template and stream valid UTF-8.
- `ml5d` features `safetensors` and `cuda`.

### Changed
- Linux installer sets `ml5d` up as a systemd **user** service
  (`/usr/lib/systemd/user/ml5d.service`, enabled globally, lingering for the
  installing user) and removes the old system unit.
- Ollama `/ollama/api/pull` streams NDJSON like Ollama.
- `ml5d` logs to the journal when run by systemd, binds before writing its pid
  file, and on SIGTERM/Ctrl+C drains requests for up to 10 s and unloads models.
- `ml5d --background`/`--stop` work on Linux and macOS (previously Windows only).
- `--parallel N` maps to llama-server slots with continuous batching.

### Removed
- In-process llama.cpp bindings and the `gpu`/`vulkan`/`cpu` build features.
- `--manual-kv`/`--auto-kv` and the per-request `cache` flag; prompt caching is
  always on.

### Fixed
- Deleting a safetensors model removes its directory.
- Safetensors load failures are reported instead of appearing to succeed.
- Native `/api/models` includes `friendly_name`.

## [0.2.3] - 2026-09-30

### Multi-node support for GGUF

### Added
- `ml5d --tensor-split 0.5,0.5`: raw llama.cpp tensor split ratios, distributing
  model layers across multiple GPUs or RPC cluster nodes.
- `ml5d --rpc-nodes node1:50052,node2:50052`: registers llama.cpp RPC cluster
  workers (ggml-rpc-server endpoints) as offload devices for GGUF inference.
- `ml5 backend download` now also extracts `ggml-rpc-server.exe` so worker nodes
  can be set up from the same backend pack.

## [0.2.2] - 2026-09-30

### Hotfix 0.2.2

### Fixed
- Ollama-compatible endpoints (`/ollama/api/chat`, `/ollama/api/generate`) now stream
  newline-delimited JSON instead of SSE, matching real Ollama. Fixes "Expecting value"
  parse failures in clients like Open WebUI.
- `/ollama/api/show` now returns a `capabilities` array (completion, embedding, and
  `thinking` when the model emits think/channel markers) so clients can detect
  thinking-capable models.

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
