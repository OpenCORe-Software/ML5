# ML5

<p align="center"><img src="ML5.png" alt="ML5" width="480"></p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Local LLM inference with a Rust CLI and daemon, serving OpenAI, Anthropic, Ollama
and native APIs on `http://127.0.0.1:11435`.

ML5 manages models and requests; inference for GGUF models runs in
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` (`--jinja`),
one child process per loaded model. Safetensors models run in-process on
[candle](https://github.com/huggingface/candle) when built with the `safetensors`
feature.

## Features

- **Real tool calling** on every API surface (OpenAI `tools`, Anthropic `tool_use`,
  Ollama `tools`, Responses `function_call`) using each model's own chat template.
- **Vision**: models with an `mmproj` projector accept images (OpenAI `image_url`,
  Anthropic image blocks, Ollama `images`, native `images`). The projector is
  downloaded automatically with the model.
- **Reasoning**: thinking output is separated into `reasoning_content` / `thinking`
  blocks; turn it on or off per request (`think`, `reasoning_effort`,
  Anthropic `thinking`).
- **Structured output**: JSON mode and JSON-schema constrained output
  (`response_format`, Ollama `format`, Responses `text.format`).
- **KV-cache reuse** across requests on every API, with cached-token counts in usage.
- **Continuous batching**: `--parallel N` serves N requests per model at once.
- **Several models at once** with LRU eviction, idle keep-alive, memory budgeting, and
  automatic context sizing from free RAM/VRAM.
- **Model store**: Hugging Face pulls with quantization picking, split GGUFs,
  resumable downloads, SHA-256 verification, cancellation, and progress events.
- **Runtime management**: the pinned llama.cpp release is located on disk or
  downloaded and verified on first use (CPU, Vulkan, CUDA, ROCm, Metal builds).

## Install

### Linux

```sh
curl -fsSL https://raw.githubusercontent.com/OpenCORe-Software/ML5/master/install-linux.sh | sudo sh
```

On systemd machines this installs `ml5d` as a **user service**
(`/usr/lib/systemd/user/ml5d.service`, enabled globally) so it runs in each user's
session with that user's GPU access and model store, and enables lingering for the
installing user so the daemon also runs without a login. Manage it with
`systemctl --user start|stop|status ml5d` and read logs with
`journalctl --user -u ml5d`.

For the GPU, install the Vulkan loader and drivers (Debian/Ubuntu:
`sudo apt install libvulkan1 mesa-vulkan-drivers`).

### macOS

```sh
curl -fsSL https://raw.githubusercontent.com/OpenCORe-Software/ML5/master/install-macos.sh | sudo sh
```

### Windows

Run the `ml5-installer` from the latest release, or install a local build:

```powershell
cargo build --release
powershell -ExecutionPolicy Bypass -File install.ps1 -SourceDir .\target\release
```

### From source

```sh
cargo build --release
```

See [BUILD.md](BUILD.md) for features and packaging.

## Usage

```sh
ml5 pull hf:unsloth/Qwen3-8B-GGUF               # best quant that fits, plus mmproj if any
ml5 pull hf:unsloth/Qwen3-8B-GGUF:Q5_K_M        # pick a quantization
ml5 pull hf:owner/repo/model-Q4_K_M.gguf        # a specific file
ml5 pulls                                       # downloads in progress
ml5 cancel <id>                                 # cancel one (resumes on the next pull)

ml5 list                                        # size, params, quant, context, capabilities
ml5 run <model>                                 # interactive chat
ml5 run <model> --message "Explain RAID levels" --system "Be concise"
ml5 run <model> --message "What is in this picture?" --image photo.jpg
ml5 run <model> --think off --max-tokens 512 --temperature 0.7
echo "Summarize this" | ml5 run <model> --quiet

ml5 load <model> --keep-alive 1h                # preload
ml5 ps                                          # loaded models
ml5 unload <model>
ml5 status                                      # daemon, runtime, memory, loaded models
ml5 rename <model> <friendly-name>
ml5 delete <model>

ml5 backend detect                              # GPUs and recommended runtime
ml5 backend status                              # which llama-server is used
ml5 backend download --kind vulkan              # fetch a runtime build ahead of time
```

Interactive chat commands: `/help`, `/clear`, `/model <name>`,
`/set system <text|show|clear>`, `/think on|off`, `/image <path>`, `/exit`.

## llama-server runtime

ML5 is pinned to llama.cpp release **b10687**. It looks for `llama-server` in this
order and uses the first one found:

1. `ml5d --llama-server PATH`, then the `ML5_LLAMA_SERVER` environment variable
   (a binary or a directory containing one)
2. `llama.cpp/` beside the `ml5d` executable, the executable's own directory, and
   `<prefix>/lib/ml5/llama.cpp`
3. `/usr/lib/ml5/llama.cpp`, `/usr/local/lib/ml5/llama.cpp`, `/opt/ml5/llama.cpp`
4. `~/.ml5/backends` (Windows installer) and `~/.ml5/runtime/<kind>-b10687`
   (downloaded by ML5)
5. `PATH`

If none is found, the matching release archive is downloaded from GitHub on the
first model load, checked against a pinned SHA-256, and unpacked to
`~/.ml5/runtime`.

## Daemon options

```sh
ml5d --help
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--host`, `--port` | `127.0.0.1:11435` | Bind address |
| `--models-dir` | `~/.ml5/models` | Model store (`$ML5_HOME/models` if set) |
| `--llama-server` | auto | llama-server binary or directory |
| `--ctx-size` | `0` (auto) | Context per slot; auto sizes from free memory |
| `--auto-ctx-max` | `32768` | Ceiling for the automatic context size |
| `--gpu-layers` | `auto` | `auto` fits to free VRAM, `all`, or a number (`0` = CPU) |
| `--parallel [N]` | `1` | Concurrent requests per model; bare flag means 4 |
| `--max-loaded-models` | `2` | Models kept loaded at once (LRU eviction) |
| `--keep-alive` | `15m` | Idle unload delay (`30s`, `15m`, `2h`; `-1` = never) |
| `--max-memory-fraction` | `0.85` | Share of RAM+VRAM a load may claim |
| `--flash-attn` | `auto` | `auto`, `on`, `off` |
| `--cache-type-k/-v` | `f16` | KV cache types (`q8_0` halves KV memory) |
| `--no-cache-prompt` | off | Disable prefix (prompt) caching in model runtimes |
| `--cache-reuse N` | `0` | Min chunk size for KV-shift prefix reuse (`0` = off) |
| `--no-kv-offload` | off | Keep the KV cache in system RAM |
| `--batch-size`, `--ubatch-size` | `2048`, `512` | Prompt batch sizes |
| `--threads`, `--threads-batch` | auto | CPU threads |
| `--main-gpu`, `--tensor-split` | `0`, none | Multi-GPU placement |
| `--rpc-nodes` | none | llama.cpp RPC workers (`host:port,...`) |
| `--no-mmap`, `--mlock` | off | Memory mapping / pinning |
| `--background`, `--stop` | | Detach as a background process / stop it |

Requests may override `n_ctx`, `n_gpu_layers`, `flash_attn`, `cache_type_k`,
`cache_type_v`, and `keep_alive`; a model reloads only when a load-affecting value
changes.

## API

### Native (`/api/*`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Chat (SSE with load/prompt-progress status events) |
| `/api/generate` | POST | Raw completion |
| `/api/embed` | POST | Embeddings |
| `/api/models` | GET | Installed models with details and capabilities |
| `/api/status` | GET | Daemon, runtime, memory, loaded models, pulls |
| `/api/metrics` | GET | Compact counters (models, active requests, memory, pulls) |
| `/api/load` | POST | Preload a model |
| `/api/unload` | POST | Unload a model |
| `/api/pull` | POST | Pull a model (SSE progress; re-attaches to a running pull) |
| `/api/pulls` | GET | Pulls in progress |
| `/api/pull/cancel` | POST | Cancel a pull by id |
| `/api/delete` | DELETE | Delete a model |
| `/api/rename` | POST | Set a friendly name |
| `/api/rescan` | POST | Rescan the model store |

### OpenAI-compatible (`/v1/*`)

`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`,
`/v1/models/{id}`, `/v1/responses`.

### Anthropic-compatible (`/v1/*`)

`/v1/messages`, `/v1/messages/count_tokens` (exact counts from the model's
tokenizer and template).

### Ollama-compatible (`/ollama/api/*`)

`generate`, `chat`, `tags`, `show`, `ps`, `version`, `embed`, `embeddings`, `pull`,
`delete`, `copy` — NDJSON streaming, like Ollama.

## Model formats

| Format | Engine | Notes |
|--------|--------|-------|
| GGUF | llama-server | Single or split files, optional `mmproj` for vision |
| Safetensors | candle | Llama-family architectures; build with `--features ml5d/safetensors` |

## Development

```sh
cargo build --workspace
cargo clippy --workspace --all-targets
cargo test --workspace
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [candle](https://github.com/huggingface/candle)
- [axum](https://github.com/tokio-rs/axum)
- Inspired by [Ollama](https://ollama.com/)
