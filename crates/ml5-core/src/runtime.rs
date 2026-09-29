use crate::error::{Ml5Error, Result};
use std::path::PathBuf;

pub const LLAMA_RELEASE_TAG: &str = "b10687";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuKind {
    NvidiaCuda,
    Vulkan,
    Rocm,
    Metal,
    Cpu,
}

pub fn runtime_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ml5")
        .join("runtime")
}

pub fn detect_gpu() -> GpuKind {
    #[cfg(target_os = "macos")]
    {
        return GpuKind::Metal;
    }

    let nvidia = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false);
    if nvidia {
        return GpuKind::Vulkan;
    }

    #[cfg(windows)]
    {
        if let Ok(out) = std::process::Command::new("wmic")
            .args(["path", "win32_VideoController", "get", "name"])
            .output()
        {
            let text = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if text.contains("amd") || text.contains("radeon") || text.contains("intel") {
                return GpuKind::Vulkan;
            }
        }
    }

    GpuKind::Cpu
}

pub fn asset_name(kind: GpuKind) -> (&'static str, &'static str) {
    #[cfg(windows)]
    {
        match kind {
            GpuKind::NvidiaCuda => ("llama-{}-bin-win-cuda-12.4-x64.zip", "zip"),
            GpuKind::Vulkan => ("llama-{}-bin-win-vulkan-x64.zip", "zip"),
            GpuKind::Rocm => ("llama-{}-bin-win-rocm-7.14-x64.zip", "zip"),
            _ => ("llama-{}-bin-win-cpu-x64.zip", "zip"),
        }
    }
    #[cfg(target_os = "linux")]
    {
        match kind {
            GpuKind::Vulkan => ("llama-{}-bin-ubuntu-vulkan-x64.tar.gz", "tar.gz"),
            GpuKind::Rocm => ("llama-{}-bin-ubuntu-rocm-7.14-x64.tar.gz", "tar.gz"),
            _ => ("llama-{}-bin-ubuntu-x64.tar.gz", "tar.gz"),
        }
    }
    #[cfg(target_os = "macos")]
    {
        ("llama-{}-bin-macos-arm64.tar.gz", "tar.gz")
    }
}

pub fn download_url(kind: GpuKind) -> String {
    let (name, _) = asset_name(kind);
    format!(
        "https://github.com/ggml-org/llama.cpp/releases/download/{}/{}",
        LLAMA_RELEASE_TAG,
        name.replace("{}", LLAMA_RELEASE_TAG)
    )
}

pub fn server_binary_path(kind: GpuKind) -> PathBuf {
    let dir = runtime_dir().join(format!("{:?}", kind).to_lowercase());
    #[cfg(windows)]
    {
        dir.join("llama-server.exe")
    }
    #[cfg(not(windows))]
    {
        dir.join("llama-server")
    }
}

pub fn runtime_installed(kind: GpuKind) -> bool {
    server_binary_path(kind).exists()
}

pub async fn ensure_runtime<F>(kind: GpuKind, mut on_event: F) -> Result<PathBuf>
where
    F: FnMut(String),
{
    let bin = server_binary_path(kind);
    if bin.exists() {
        return Ok(bin);
    }

    let dir = bin.parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&dir)?;
    let url = download_url(kind);
    on_event(format!(
        "downloading llama.cpp runtime ({kind:?}) from {url}"
    ));

    let bytes = reqwest::get(&url)
        .await
        .map_err(|e| Ml5Error::Download(format!("runtime download failed: {e}")))?
        .bytes()
        .await
        .map_err(|e| Ml5Error::Download(format!("runtime read failed: {e}")))?;

    let (_, ext) = asset_name(kind);
    let archive = dir.join(format!("runtime.{ext}"));
    std::fs::write(&archive, &bytes)?;

    on_event("extracting runtime".into());
    if ext == "zip" {
        let file = std::fs::File::open(&archive)?;
        let mut zip =
            zip::ZipArchive::new(file).map_err(|e| Ml5Error::Download(format!("bad zip: {e}")))?;
        for i in 0..zip.len() {
            let mut f = zip
                .by_index(i)
                .map_err(|e| Ml5Error::Download(e.to_string()))?;
            let name = f.name().to_string();
            if name.ends_with(".dll") || name.ends_with(".exe") {
                let out = dir.join(PathBuf::from(&name).file_name().unwrap());
                let mut outf = std::fs::File::create(&out)?;
                std::io::copy(&mut f, &mut outf)?;
            }
        }
    } else {
        let file = std::fs::File::open(&archive)?;
        let gz = flate2::read::GzDecoder::new(file);
        let mut tar = tar::Archive::new(gz);
        tar.unpack(&dir)
            .map_err(|e| Ml5Error::Download(format!("untar failed: {e}")))?;
    }
    let _ = std::fs::remove_file(&archive);

    if !bin.exists() {
        return Err(Ml5Error::Download(format!(
            "runtime extracted but {} not found",
            bin.display()
        )));
    }
    on_event(format!("runtime ready: {}", bin.display()));
    Ok(bin)
}
