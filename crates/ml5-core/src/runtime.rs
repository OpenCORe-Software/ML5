use crate::error::{Ml5Error, Result};
use futures::StreamExt;
use std::path::{Path, PathBuf};

pub const LLAMA_RELEASE_TAG: &str = "b10687";

#[cfg(unix)]
pub const BUNDLED_DIR: &str = "/usr/lib/ml5/llama.cpp";

#[cfg(windows)]
pub const SERVER_EXE: &str = "llama-server.exe";
#[cfg(not(windows))]
pub const SERVER_EXE: &str = "llama-server";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuKind {
    NvidiaCuda,
    Vulkan,
    Rocm,
    Metal,
    Cpu,
}

impl GpuKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::NvidiaCuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Rocm => "rocm",
            Self::Metal => "metal",
            Self::Cpu => "cpu",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "cuda" | "nvidia" => Some(Self::NvidiaCuda),
            "vulkan" => Some(Self::Vulkan),
            "rocm" | "hip" => Some(Self::Rocm),
            "metal" => Some(Self::Metal),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuDevice {
    pub vendor: String,
    pub name: String,
}

pub fn runtime_dir() -> PathBuf {
    crate::config::Config::ml5_home().join("runtime")
}

pub fn backends_dir() -> PathBuf {
    crate::config::Config::ml5_home().join("backends")
}

pub fn detect_gpus() -> Vec<GpuDevice> {
    let mut out = Vec::new();
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .stderr(std::process::Stdio::null())
        .output()
    {
        if o.status.success() {
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                let line = line.trim();
                if !line.is_empty() {
                    out.push(GpuDevice {
                        vendor: "NVIDIA".into(),
                        name: line.to_string(),
                    });
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
            return out;
        };
        let mut seen = std::collections::HashSet::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let dev = entry.path().join("device");
            let canonical = std::fs::canonicalize(&dev).unwrap_or(dev.clone());
            if !seen.insert(canonical) {
                continue;
            }
            let vendor_id = std::fs::read_to_string(dev.join("vendor"))
                .map(|s| s.trim().to_ascii_lowercase())
                .unwrap_or_default();
            let vendor = match vendor_id.as_str() {
                "0x10de" => "NVIDIA",
                "0x1002" => "AMD",
                "0x8086" => "Intel",
                _ => continue,
            };
            if vendor == "NVIDIA" && out.iter().any(|g| g.vendor == "NVIDIA") {
                continue;
            }
            let device_id = std::fs::read_to_string(dev.join("device"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            out.push(GpuDevice {
                vendor: vendor.into(),
                name: format!("{vendor} GPU {device_id}"),
            });
        }
    }
    #[cfg(windows)]
    {
        if let Ok(o) = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ])
            .output()
        {
			if !o.status.success() {
				return out;
			}
			let nvidia = out.iter().any(|g| g.vendor == "NVIDIA");
			for line in String::from_utf8_lossy(&o.stdout).lines() {
				let line = line.trim();
				let lower = line.to_ascii_lowercase();
				if line.is_empty() || out.iter().any(|g| g.name.eq_ignore_ascii_case(line)) {
					continue;
				}
				let vendor = if lower.contains("nvidia") {
					if nvidia {
						continue;
					}
					"NVIDIA"
				} else if lower.contains("amd") || lower.contains("radeon") {
					"AMD"
				} else if lower.contains("intel") {
					"Intel"
				} else {
					continue;
				};
				out.push(GpuDevice {
					vendor: vendor.into(),
					name: line.to_string(),
				});
			}
        }
    }
    out
}

pub fn vulkan_loader_present() -> bool {
    #[cfg(target_os = "linux")]
    {
        let candidates = [
            "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
            "/usr/lib64/libvulkan.so.1",
            "/usr/lib/libvulkan.so.1",
            "/lib/x86_64-linux-gnu/libvulkan.so.1",
        ];
        if candidates.iter().any(|p| Path::new(p).exists()) {
            return true;
        }
        std::process::Command::new("ldconfig")
            .arg("-p")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("libvulkan.so.1"))
            .unwrap_or(false)
    }
    #[cfg(windows)]
    {
		std::env::var_os("SystemRoot")
			.map(PathBuf::from)
			.unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
			.join("System32")
			.join("vulkan-1.dll")
			.is_file()
    }
    #[cfg(target_os = "macos")]
    {
        false
    }
}

pub fn detect_gpu() -> GpuKind {
    static DETECTED: std::sync::OnceLock<GpuKind> = std::sync::OnceLock::new();
    *DETECTED.get_or_init(|| {
        if cfg!(target_os = "macos") {
            GpuKind::Metal
        } else if detect_gpus().is_empty() || !vulkan_loader_present() {
            GpuKind::Cpu
        } else {
            GpuKind::Vulkan
        }
    })
}

pub fn resolved_kind(kind: GpuKind) -> GpuKind {
    let arm = cfg!(target_arch = "aarch64");
    if cfg!(target_os = "macos") {
        return GpuKind::Metal;
    }
    match kind {
        GpuKind::Metal => GpuKind::Cpu,
        GpuKind::NvidiaCuda if cfg!(target_os = "linux") => GpuKind::Vulkan,
        GpuKind::NvidiaCuda | GpuKind::Vulkan | GpuKind::Rocm if arm && cfg!(windows) => GpuKind::Cpu,
        GpuKind::Rocm if arm => GpuKind::Cpu,
        other => other,
    }
}

pub fn asset_name(kind: GpuKind) -> Option<(String, &'static str)> {
    let kind = resolved_kind(kind);
    let tag = LLAMA_RELEASE_TAG;
    #[cfg(windows)]
    {
        let name = if cfg!(target_arch = "aarch64") {
            "cpu-arm64"
        } else if cfg!(target_arch = "x86_64") {
            match kind {
                GpuKind::NvidiaCuda => "cuda-12.4-x64",
                GpuKind::Vulkan => "vulkan-x64",
                GpuKind::Rocm => "rocm-7.14-x64",
                _ => "cpu-x64",
            }
        } else {
            return None;
        };
        Some((format!("llama-{tag}-bin-win-{name}.zip"), "zip"))
    }
    #[cfg(target_os = "linux")]
    {
        let gpu = matches!(kind, GpuKind::Vulkan | GpuKind::NvidiaCuda);
        let name = if cfg!(target_arch = "aarch64") {
            if gpu {
                "vulkan-arm64"
            } else {
                "arm64"
            }
        } else if cfg!(target_arch = "x86_64") {
            match kind {
                GpuKind::Vulkan | GpuKind::NvidiaCuda => "vulkan-x64",
                GpuKind::Rocm => "rocm-7.14-x64",
                _ => "x64",
            }
        } else if cfg!(target_arch = "s390x") {
            "s390x"
        } else {
            return None;
        };
        Some((format!("llama-{tag}-bin-ubuntu-{name}.tar.gz"), "tar.gz"))
    }
    #[cfg(target_os = "macos")]
    {
        let _ = kind;
        let name = if cfg!(target_arch = "aarch64") {
            "arm64"
        } else if cfg!(target_arch = "x86_64") {
            "x64"
        } else {
            return None;
        };
        Some((format!("llama-{tag}-bin-macos-{name}.tar.gz"), "tar.gz"))
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = (kind, tag);
        None
    }
}

pub fn extra_assets(kind: GpuKind) -> Vec<String> {
    let kind = resolved_kind(kind);
    if cfg!(windows) && kind == GpuKind::NvidiaCuda && !cfg!(target_arch = "aarch64") {
        vec!["cudart-llama-bin-win-cuda-12.4-x64.zip".into()]
    } else {
        Vec::new()
    }
}

pub fn asset_sha256(name: &str) -> Option<&'static str> {
    const DIGESTS: &[(&str, &str)] = &[
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
        ("llama-b10687-bin-macos-arm64.tar.gz", "ad1f407db2b21eb636779ef90c493b327ce55b59ab4fc43ed8197c5c61839b0a"),
        ("llama-b10687-bin-macos-x64.tar.gz", "9250762748c64398477725ceb43ed43bfac452a2bb13b6abcada7389bff83fb5"),
        ("llama-b10687-bin-ubuntu-arm64.tar.gz", "5c0c8a0cee780d3024d3b3df0c817171a8ae2d50db70ac8f3d19af8fda1126a8"),
        ("llama-b10687-bin-ubuntu-rocm-7.14-x64.tar.gz", "5ade1220ee888417e211f81f56289f7f2147dd162b29495cf21913db0f562a4d"),
        ("llama-b10687-bin-ubuntu-s390x.tar.gz", "8df3acf5be6ce022984c94b37eecd512d688978f954332f8022c98a1201ab1ec"),
        ("llama-b10687-bin-ubuntu-vulkan-arm64.tar.gz", "47892e844104523ebfa476c89a9e079a56b115a79b375dd238c055f4961b1d86"),
        ("llama-b10687-bin-ubuntu-vulkan-x64.tar.gz", "f38f493e7d137f1f3cd352d5cfdfabc945676775b235feb1b0dbb2dadba43d2d"),
        ("llama-b10687-bin-ubuntu-x64.tar.gz", "3ea69deaab84792fca5982c975c6bebed4c2c664294b662852602e0be11f09e7"),
        ("llama-b10687-bin-win-cpu-arm64.zip", "24bdd131f066627587a4121c1578e2f8cdaaa4ac3bb5b71c23f109bc42422086"),
        ("llama-b10687-bin-win-cpu-x64.zip", "7671db956d077e7f055c7b6b0cf48e58726785212497468d0fc3b40c6aa9adae"),
        ("llama-b10687-bin-win-cuda-12.4-x64.zip", "63b52f4eea95a09a32df360b28b928597d07375a380b8d79f20ba9f1cbfe1f59"),
        ("llama-b10687-bin-win-rocm-7.14-x64.zip", "47c11b8d43ea2aa4b735652ac2c2a019e627d7b93f631f051f6fd502f1a12105"),
        ("llama-b10687-bin-win-vulkan-x64.zip", "7fe1f90294d95f4cb8937867162e6962664b92902271aa2b9a023a55f35a9de3"),
    ];
    DIGESTS.iter().find(|(n, _)| *n == name).map(|(_, d)| *d)
}

pub fn asset_url(name: &str) -> String {
    format!("https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE_TAG}/{name}")
}

pub fn download_url(kind: GpuKind) -> Option<String> {
    asset_name(kind).map(|(name, _)| asset_url(&name))
}

fn no_prebuilt_error() -> Ml5Error {
    Ml5Error::Download(format!(
        "llama.cpp {LLAMA_RELEASE_TAG} has no prebuilt runtime for this platform ({} {}); install {SERVER_EXE} and pass its path with --llama-server or ML5_LLAMA_SERVER",
        std::env::consts::OS,
        std::env::consts::ARCH
    ))
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

fn find_file_in(dir: &Path, depth: usize, accept: fn(&Path) -> bool) -> Option<PathBuf> {
    let direct = dir.join(SERVER_EXE);
    if accept(&direct) {
        return Some(direct);
    }
    if depth == 0 {
        return None;
    }
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    subdirs.sort();
    subdirs.into_iter().find_map(|d| find_file_in(&d, depth - 1, accept))
}

fn find_server_in(dir: &Path, depth: usize) -> Option<PathBuf> {
    find_file_in(dir, depth, is_executable)
}

fn path_lookup(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|d| d.join(name))
        .find(|p| is_executable(p))
}

pub fn runtime_install_dir(kind: GpuKind) -> PathBuf {
    let kind = resolved_kind(kind);
    runtime_dir().join(format!("{}-{}", kind.label(), LLAMA_RELEASE_TAG))
}

pub fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
	#[cfg(windows)]
	dirs.push(backends_dir());
    let mut exe_dir = None;
    if let Ok(exe) = std::env::current_exe() {
        let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
        if let Some(bin) = exe.parent() {
            dirs.push(bin.join("llama.cpp"));
            if let Some(prefix) = bin.parent() {
                dirs.push(prefix.join("lib").join("ml5").join("llama.cpp"));
            }
            exe_dir = Some(bin.to_path_buf());
        }
    }
    #[cfg(unix)]
    {
        dirs.push(PathBuf::from(BUNDLED_DIR));
        dirs.push(PathBuf::from("/usr/local/lib/ml5/llama.cpp"));
        dirs.push(PathBuf::from("/opt/ml5/llama.cpp"));
    }
    dirs.extend(exe_dir);
    dirs.push(backends_dir());
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|d| seen.insert(d.clone()));
    dirs
}

pub fn locate_server(configured: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = configured {
        return if p.is_dir() { find_server_in(p, 2) } else { Some(p.to_path_buf()).filter(|p| is_executable(p)) };
    }
    if let Some(env) = std::env::var_os("ML5_LLAMA_SERVER").filter(|v| !v.is_empty()) {
        let p = PathBuf::from(env);
        let found = if p.is_dir() { find_server_in(&p, 2) } else { Some(p).filter(|p| is_executable(p)) };
        if found.is_some() {
            return found;
        }
    }
    for dir in candidate_dirs() {
        if let Some(found) = find_server_in(&dir, 0) {
            return Some(found);
        }
    }
    let preferred = detect_gpu();
    for kind in [preferred, GpuKind::Vulkan, GpuKind::Cpu, GpuKind::NvidiaCuda, GpuKind::Rocm, GpuKind::Metal] {
        if let Some(found) = find_server_in(&runtime_install_dir(kind), 2) {
            return Some(found);
        }
    }
    path_lookup(SERVER_EXE)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RuntimeInfo {
    pub server: Option<PathBuf>,
    pub release_tag: &'static str,
    pub detected: GpuKind,
    pub gpus: Vec<GpuDevice>,
    pub installed: Vec<PathBuf>,
}

pub fn runtime_info(configured: Option<&Path>) -> RuntimeInfo {
    let mut installed = Vec::new();
    if let Ok(entries) = std::fs::read_dir(runtime_dir()) {
        for e in entries.flatten() {
            if let Some(server) = find_server_in(&e.path(), 2) {
                installed.push(server);
            }
        }
    }
    installed.sort();
    RuntimeInfo {
        server: locate_server(configured),
        release_tag: LLAMA_RELEASE_TAG,
        detected: detect_gpu(),
        gpus: detect_gpus(),
        installed,
    }
}

pub async fn ensure_runtime<F>(configured: Option<&Path>, mut on_event: F) -> Result<PathBuf>
where
    F: FnMut(String),
{
    if let Some(found) = locate_server(configured) {
        return Ok(found);
    }
    if configured.is_some() {
        return Err(Ml5Error::Backend(format!(
            "the configured llama-server path does not contain {SERVER_EXE}"
        )));
    }
    download_runtime(detect_gpu(), &mut on_event).await
}

pub async fn download_runtime<F>(kind: GpuKind, on_event: &mut F) -> Result<PathBuf>
where
    F: FnMut(String),
{
    let kind = resolved_kind(kind);
    let dest = runtime_install_dir(kind);
    if let Some(found) = find_server_in(&dest, 2) {
        return Ok(found);
    }
    if asset_name(kind).is_none() {
        return Err(no_prebuilt_error());
    }
    std::fs::create_dir_all(runtime_dir())?;
    let staging = runtime_dir().join(format!(".staging-{}-{}", kind.label(), std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging)?;
    let result = fetch_into(kind, &staging, on_event).await;
    let outcome = match result {
        Ok(extract_dir) => {
            let _ = std::fs::remove_dir_all(&dest);
            std::fs::rename(&extract_dir, &dest)
                .map_err(Ml5Error::from)
                .and_then(|_| {
                    find_server_in(&dest, 2)
                        .ok_or_else(|| Ml5Error::Download(format!("{SERVER_EXE} missing after install")))
                })
        }
        Err(e) => Err(e),
    };
    let _ = std::fs::remove_dir_all(&staging);
    let server = outcome?;
    on_event(format!("runtime ready: {}", server.display()));
    Ok(server)
}

async fn fetch_into<F>(kind: GpuKind, staging: &Path, on_event: &mut F) -> Result<PathBuf>
where
    F: FnMut(String),
{
    let client = reqwest::Client::builder()
        .user_agent(concat!("ml5/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(std::time::Duration::from_secs(20))
        .read_timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| Ml5Error::Download(e.to_string()))?;
    let extract_dir = staging.join("extract");
    std::fs::create_dir_all(&extract_dir)?;
    let (main, _) = asset_name(kind).ok_or_else(no_prebuilt_error)?;
    let mut assets = vec![main];
    assets.extend(extra_assets(kind));
    for (i, name) in assets.iter().enumerate() {
        let expected = asset_sha256(name).ok_or_else(|| {
            Ml5Error::Download(format!("{name} has no pinned SHA-256 checksum for llama.cpp {LLAMA_RELEASE_TAG}"))
        })?;
        let url = asset_url(name);
        on_event(format!("downloading llama.cpp {LLAMA_RELEASE_TAG} ({}) from {url}", kind.label()));
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| Ml5Error::Download(format!("runtime download failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(Ml5Error::Download(format!(
                "runtime download returned {} for {url}",
                resp.status()
            )));
        }
        let total = resp.content_length();
        let ext = if name.ends_with(".zip") { "zip" } else { "tar.gz" };
        let archive = staging.join(format!("asset-{i}.{ext}"));
        let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
        let mut file = tokio::fs::File::create(&archive).await?;
        let mut stream = resp.bytes_stream();
        let mut done: u64 = 0;
        let mut last = std::time::Instant::now();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Ml5Error::Download(format!("runtime stream error: {e}")))?;
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
            sha2::Digest::update(&mut hasher, &chunk);
            done += chunk.len() as u64;
            if last.elapsed() >= std::time::Duration::from_secs(2) {
                on_event(match total {
                    Some(t) => format!("runtime download {:.0}%", done as f64 * 100.0 / t as f64),
                    None => format!("runtime download {}", crate::config::humansize(done)),
                });
                last = std::time::Instant::now();
            }
        }
        tokio::io::AsyncWriteExt::flush(&mut file).await?;
        drop(file);
        if total.is_some_and(|t| t != done) {
            return Err(Ml5Error::Download("runtime download was incomplete; retry".into()));
        }
        let actual = hex::encode(sha2::Digest::finalize(hasher));
        if actual != expected {
            return Err(Ml5Error::Download(format!(
                "{name} failed its SHA-256 check (expected {expected}, got {actual})"
            )));
        }
        on_event(format!("extracting {name}"));
        let target = extract_dir.clone();
        let ext = ext.to_string();
        tokio::task::spawn_blocking(move || extract_archive(&archive, &target, &ext))
            .await
            .map_err(|e| Ml5Error::Download(format!("extraction task failed: {e}")))??;
    }
    if find_server_in(&extract_dir, 2).is_none() {
        return Err(Ml5Error::Download(format!("runtime archive did not contain {SERVER_EXE}")));
    }
    Ok(extract_dir)
}

fn extract_archive(archive: &Path, dest: &Path, ext: &str) -> Result<()> {
    let file = std::fs::File::open(archive)?;
    if ext == "zip" {
        let mut zip = zip::ZipArchive::new(file).map_err(|e| Ml5Error::Download(format!("bad zip: {e}")))?;
        for i in 0..zip.len() {
            let mut f = zip.by_index(i).map_err(|e| Ml5Error::Download(e.to_string()))?;
            let Some(rel) = f.enclosed_name() else { continue };
            let out = dest.join(rel);
            if f.is_dir() {
                std::fs::create_dir_all(&out)?;
                continue;
            }
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut outf = std::fs::File::create(&out)?;
            std::io::copy(&mut f, &mut outf)?;
        }
    } else {
        let gz = flate2::read::GzDecoder::new(file);
        let mut tar = tar::Archive::new(gz);
        tar.set_preserve_permissions(true);
        for entry in tar.entries().map_err(|e| Ml5Error::Download(format!("bad tar: {e}")))? {
            let mut entry = entry.map_err(|e| Ml5Error::Download(format!("bad tar entry: {e}")))?;
            entry
                .unpack_in(dest)
                .map_err(|e| Ml5Error::Download(format!("untar failed: {e}")))?;
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Some(server) = find_file_in(dest, 2, Path::is_file) {
            let mut perm = std::fs::metadata(&server)?.permissions();
            perm.set_mode(perm.mode() | 0o755);
            std::fs::set_permissions(&server, perm)?;
        }
    }
    Ok(())
}
