# Building ML5

ML5 is pure Rust. GGUF inference runs in llama.cpp's prebuilt `llama-server`, so no
C/C++ toolchain, CMake, or libclang is needed to build it.

## Prerequisites

- Rust stable (via [rustup](https://rustup.rs))
- Linux: a C linker (`build-essential` on Debian/Ubuntu) and `pkg-config`
- Windows: Visual Studio 2022 Build Tools (C++ workload) for the MSVC linker

## Build

```sh
cargo build --release
```

Binaries land in `target/release/`:

- `ml5` — CLI client
- `ml5d` — daemon / API server
- `ml5-installer` — Windows installer

### Features

| Feature | Effect |
|---------|--------|
| `ml5d/safetensors` | Serve full safetensors models in-process with candle (CPU) |
| `ml5d/cuda` | `safetensors` with candle's CUDA backend (needs the CUDA toolkit) |

```sh
cargo build --release --workspace --features ml5d/safetensors
```

## llama-server

`ml5d` needs llama.cpp release **b10687** at runtime. Without a packaged copy it
downloads the matching build on the first model load. To ship it with a package,
unpack the release archive for your platform into `/usr/lib/ml5/llama.cpp` (Linux),
or into `llama.cpp/` beside the `ml5d` binary:

```sh
sudo mkdir -p /usr/lib/ml5/llama.cpp
curl -fLO https://github.com/ggml-org/llama.cpp/releases/download/b10687/llama-b10687-bin-ubuntu-vulkan-x64.tar.gz
echo "f38f493e7d137f1f3cd352d5cfdfabc945676775b235feb1b0dbb2dadba43d2d  llama-b10687-bin-ubuntu-vulkan-x64.tar.gz" | sha256sum -c
sudo tar -xzf llama-b10687-bin-ubuntu-vulkan-x64.tar.gz -C /usr/lib/ml5/llama.cpp --strip-components=1
```

The Vulkan build needs `libvulkan1` and a Vulkan driver (`mesa-vulkan-drivers`)
plus `libgomp1`. The child process is started with its own directory on
`LD_LIBRARY_PATH`, so the shared libraries in the archive are found without
installing them system-wide. `ml5d --llama-server PATH` or `ML5_LLAMA_SERVER`
points at any other build.

## systemd

Run `ml5d` as a user service so it gets the session's GPU access:

```ini
# /usr/lib/systemd/user/ml5d.service
[Unit]
Description=CORe ML5 inference daemon
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/ml5d
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
```

```sh
sudo systemctl --global enable ml5d.service
systemctl --user start ml5d
```

On headless servers, `sudo loginctl enable-linger <user>` starts the user manager
(and `ml5d`) at boot without a login.

## Notes

- Stop the daemon before overwriting its binary
  (`systemctl --user stop ml5d`, `ml5d --stop`, or `Stop-Process -Name ml5d` on Windows).
- Pascal-era NVIDIA GPUs (GTX 10-series) are not supported by current CUDA builds;
  ML5 uses the Vulkan runtime for them.
