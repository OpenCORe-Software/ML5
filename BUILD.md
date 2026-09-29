# Building ML5

## Prerequisites

- **Rust** (stable toolchain via rustup)
- **CMake 3.31** (not 4.x — llama-cpp-sys-2 is incompatible with 4.x)
- **Visual Studio 2022 Build Tools** with C++ workload
- **Git**

## Environment Setup

Before building, set these environment variables so CMake finds the correct generator:

```powershell
$env:CMAKE_GENERATOR = "Visual Studio 17 2022"
$env:CMAKE_GENERATOR_INSTANCE = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
```

## Build

```powershell
cargo build --release
```

Binaries are produced in `target/release/`:
- `ml5.exe` — CLI client
- `ml5d.exe` — daemon/server

## Install

Copy to your install directory (e.g. `%LOCALAPPDATA%\Programs\ML5\bin`):

```powershell
Copy-Item target\release\ml5.exe "$env:LOCALAPPDATA\Programs\ML5\bin\ml5.exe" -Force
Copy-Item target\release\ml5d.exe "$env:LOCALAPPDATA\Programs\ML5\bin\ml5d.exe" -Force
```

## Notes

- **CMake 4.x is not supported** — `llama-cpp-sys-2` fails with "JSON flag table not found" errors
- If you get `CMAKE_DETERMINE_COMPILER_ID` errors, your CMake installation is corrupted — reinstall 3.31
- The daemon must be stopped before overwriting the binaries:
  ```powershell
  Stop-Process -Name ml5d -Force
  ```
