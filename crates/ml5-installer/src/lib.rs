use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;
use std::path::{Path, PathBuf};

const UPDATE_SERVER: &str = "https://updates.ml5.opencore.one";
const GITHUB_RELEASE: &str = "https://github.com/OpenCORe-Software/ML5/releases/latest/download";
const LLAMA_RELEASE: &str = "b10687";
const LLAMA_LEGACY_CUDA_RELEASE: &str = "b3617";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    Cuda,
    LegacyCuda,
    Vulkan,
    Cpu,
}

impl Backend {
    pub fn label(&self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA",
            Backend::LegacyCuda => "Legacy-CUDA (GTX 10-series)",
            Backend::Vulkan => "Vulkan",
            Backend::Cpu => "CPU only",
        }
    }

    pub fn marker(&self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::LegacyCuda => "legacy-cuda",
            Backend::Vulkan => "vulkan",
            Backend::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExistingAction {
    Repair,
    Uninstall,
    Swap,
    Cancel,
}

#[derive(Parser)]
#[command(name = "ml5-installer", about = "CORe ML5 installer")]
struct Cli {
    #[arg(long, help = "Non-interactive: assume yes, auto-pick defaults")]
    yes: bool,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    #[command(about = "Install ML5 (default)")]
    Install,
    #[command(about = "Uninstall ML5")]
    Uninstall,
}

#[derive(Debug, Clone)]
struct GpuInfo {
    name: String,
    is_nvidia: bool,
    nvidia_cc: Option<(u32, u32)>,
}

#[cfg(windows)]
fn detect_gpus() -> Vec<GpuInfo> {
    let mut out = Vec::new();
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,compute_cap", "--format=csv,noheader"])
        .output()
    {
        if o.status.success() {
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let mut parts = line.rsplitn(2, ',');
                let cc_str = parts.next().unwrap_or("").trim().to_string();
                let name = parts.next().unwrap_or("").trim().to_string();
                let cc = cc_str
                    .split('.')
                    .filter_map(|p| p.parse::<u32>().ok())
                    .collect::<Vec<_>>();
                let nvidia_cc = if cc.len() == 2 { Some((cc[0], cc[1])) } else { None };
                out.push(GpuInfo {
                    name,
                    is_nvidia: true,
                    nvidia_cc,
                });
            }
        }
    }
    if let Ok(o) = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join \"`n\"",
        ])
        .output()
    {
        if o.status.success() {
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                let line = line.trim();
                if line.is_empty()
                    || line.to_lowercase().contains("nvidia")
                    || out.iter().any(|g| g.name == line)
                {
                    continue;
                }
                out.push(GpuInfo {
                    name: line.to_string(),
                    is_nvidia: false,
                    nvidia_cc: None,
                });
            }
        }
    }
    out
}

#[cfg(not(windows))]
fn detect_gpus() -> Vec<GpuInfo> {
    Vec::new()
}

#[allow(dead_code)]
fn pick_backend(gpus: &[GpuInfo]) -> Backend {
    for g in gpus {
        if g.is_nvidia {
            if let Some((major, _)) = g.nvidia_cc {
                if major >= 7 {
                    return Backend::Cuda;
                }
                if major == 6 {
                    return Backend::LegacyCuda;
                }
            }
            return Backend::Cuda;
        }
    }
    if !gpus.is_empty() {
        return Backend::Vulkan;
    }
    Backend::Cpu
}

fn install_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("AppData")
        .join("Local")
        .join("Programs")
        .join("ML5")
}

fn bin_dir() -> PathBuf {
    install_dir().join("bin")
}

fn backends_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ml5")
        .join("backends")
}

fn models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ml5")
        .join("models")
}

fn marker_path() -> PathBuf {
    install_dir().join("backend.marker")
}

fn read_marker() -> Option<Backend> {
    let s = std::fs::read_to_string(marker_path()).ok()?;
    match s.trim() {
        "cuda" => Some(Backend::Cuda),
        "legacy-cuda" => Some(Backend::LegacyCuda),
        "vulkan" => Some(Backend::Vulkan),
        "cpu" => Some(Backend::Cpu),
        _ => None,
    }
}

fn write_marker(b: Backend) -> Result<()> {
    std::fs::create_dir_all(install_dir())?;
    std::fs::write(marker_path(), b.marker())?;
    Ok(())
}

fn is_installed() -> bool {
    bin_dir().join("ml5.exe").exists() && bin_dir().join("ml5d.exe").exists()
}

fn stop_daemon() {
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/IM", "ml5d.exe"])
        .output();
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/IM", "ml5.exe"])
        .output();
    std::thread::sleep(std::time::Duration::from_millis(500));
}

fn ask(prompt: &str, options: &[&str], default: usize, non_interactive: bool) -> Result<usize> {
    if non_interactive {
        return Ok(default);
    }
    loop {
        print!("{prompt} ");
        for (i, o) in options.iter().enumerate() {
            print!("[{}] {o}  ", i + 1);
        }
        print!("(default {}): ", default + 1);
        std::io::stdout().flush()?;
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            return Ok(default);
        }
        let line = line.trim();
        if line.is_empty() {
            return Ok(default);
        }
        if let Ok(n) = line.parse::<usize>() {
            if (1..=options.len()).contains(&n) {
                return Ok(n - 1);
            }
        }
        println!("Enter 1-{}.", options.len());
    }
}

#[allow(dead_code)]
fn ask_backend(default: Backend, non_interactive: bool) -> Result<Backend> {
    let options = ["CUDA", "Legacy-CUDA (GTX 10-series)", "Vulkan", "CPU only"];
    let default_idx = match default {
        Backend::Cuda => 0,
        Backend::LegacyCuda => 1,
        Backend::Vulkan => 2,
        Backend::Cpu => 3,
    };
    let idx = ask("Choose backend:", &options, default_idx, non_interactive)?;
    Ok(match idx {
        0 => Backend::Cuda,
        1 => Backend::LegacyCuda,
        2 => Backend::Vulkan,
        _ => Backend::Cpu,
    })
}

fn llama_backend_url(b: Backend) -> (&'static str, String) {
    match b {
        Backend::Cuda => (
            LLAMA_RELEASE,
            format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{}/llama-{}-bin-win-cuda-12.4-x64.zip",
                LLAMA_RELEASE, LLAMA_RELEASE
            ),
        ),
        Backend::LegacyCuda => (
            LLAMA_LEGACY_CUDA_RELEASE,
            format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{}/llama-{}-bin-win-cuda-cu11.7.1-x64.zip",
                LLAMA_LEGACY_CUDA_RELEASE, LLAMA_LEGACY_CUDA_RELEASE
            ),
        ),
        Backend::Vulkan => (
            LLAMA_RELEASE,
            format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{}/llama-{}-bin-win-vulkan-x64.zip",
                LLAMA_RELEASE, LLAMA_RELEASE
            ),
        ),
        Backend::Cpu => (
            LLAMA_RELEASE,
            format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{}/llama-{}-bin-win-cpu-x64.zip",
                LLAMA_RELEASE, LLAMA_RELEASE
            ),
        ),
    }
}

fn llama_extra_urls(b: Backend) -> Vec<String> {
    match b {
        Backend::LegacyCuda => vec![format!(
            "https://github.com/ggml-org/llama.cpp/releases/download/{}/cudart-llama-bin-win-cu11.7.1-x64.zip",
            LLAMA_LEGACY_CUDA_RELEASE
        )],
        _ => Vec::new(),
    }
}

async fn download_with_progress(client: &reqwest::Client, url: &str, label: &str) -> Result<Vec<u8>> {
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow!("download failed: {e}"))?;
    if !resp.status().is_success() {
        bail!("download of {url} returned {}", resp.status());
    }
    let total = resp.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{msg} {bytes}/{total_bytes} [{bar:40.cyan/blue}] {bytes_per_sec}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    pb.set_message(label.to_string());

    let mut data = Vec::with_capacity(total as usize);
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        pb.inc(chunk.len() as u64);
        data.extend_from_slice(&chunk);
    }
    pb.finish_and_clear();
    Ok(data)
}

fn extract_backend_zip(data: &[u8], dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    let cursor = std::io::Cursor::new(data);
    let mut zip = zip::ZipArchive::new(cursor)?;
    for i in 0..zip.len() {
        let mut f = zip.by_index(i)?;
        let name = f.name().to_string();
        if name.ends_with(".dll") || name.ends_with(".exe") {
            let fname = PathBuf::from(&name)
                .file_name()
                .ok_or_else(|| anyhow!("bad path"))?
                .to_os_string();
            let out = dest.join(fname);
            let mut outf = std::fs::File::create(&out)?;
            std::io::copy(&mut f, &mut outf)?;
        }
    }
    Ok(())
}

async fn fetch_binary(
    client: &reqwest::Client,
    name: &str,
    dest: &Path,
) -> Result<()> {
    let from_server = async {
        let check: serde_json::Value = client
            .get(format!("{UPDATE_SERVER}/api/check?version=0.0.0&binary={name}"))
            .send()
            .await
            .context("update server check failed")?
            .json()
            .await
            .context("bad update server response")?;

        let url = check["download_url"]
            .as_str()
            .ok_or_else(|| anyhow!("no download URL for {name} on update server"))?;
        let expected_sha = check["sha256"].as_str().unwrap_or("").to_string();

        let full_url = if url.starts_with("http") {
            url.to_string()
        } else {
            format!("{UPDATE_SERVER}{url}")
        };

        let data = download_with_progress(client, &full_url, name).await?;

        if !expected_sha.is_empty() {
            use sha2::Digest;
            let actual = hex::encode(sha2::Sha256::digest(&data));
            if actual != expected_sha {
                bail!("SHA256 mismatch for {name}");
            }
        }
        Ok::<Vec<u8>, anyhow::Error>(data)
    };

    let data = match from_server.await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[ml5] update server unreachable ({e}); falling back to GitHub release");
            let url = format!("{GITHUB_RELEASE}/{name}");
            download_with_progress(client, &url, name).await?
        }
    };

    std::fs::create_dir_all(dest)?;
    std::fs::write(dest.join(name), &data)?;
    Ok(())
}

fn add_to_user_path(dir: &Path) -> Result<()> {
    #[cfg(windows)]
    {
        let dir_str = dir.to_string_lossy().to_string();
        let script = format!(
            "$p=[Environment]::GetEnvironmentVariable('Path','User');\
             $e=@($p -split ';' | Where-Object {{ $_ -ne '' }});\
             if (-not ($e | Where-Object {{ $_.TrimEnd('\\') -ieq '{}' }})) {{\
               [Environment]::SetEnvironmentVariable('Path', (($e + '{}') -join ';'), 'User')\
             }}",
            dir_str.trim_end_matches('\\'),
            dir_str
        );
        std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .output()?;
    }
    Ok(())
}

async fn install(backend: Backend, yes: bool) -> Result<()> {
    let existing = read_marker();
    let installed = is_installed();

    if installed {
        let existing_label = existing.map(|b| b.label()).unwrap_or("unknown");
        println!("ML5 is already installed ({existing_label}).");
        let same_backend = existing == Some(backend);
        let choice = if same_backend {
            ask(
                "Action:",
                &["Repair", "Uninstall", "Cancel"],
                0,
                yes,
            )?
        } else {
            ask(
                "Action:",
                &["Repair", "Uninstall", "Swap installation", "Cancel"],
                2,
                yes,
            )?
        };
        let action = if same_backend {
            match choice {
                0 => ExistingAction::Repair,
                1 => ExistingAction::Uninstall,
                _ => ExistingAction::Cancel,
            }
        } else {
            match choice {
                0 => ExistingAction::Repair,
                1 => ExistingAction::Uninstall,
                2 => ExistingAction::Swap,
                _ => ExistingAction::Cancel,
            }
        };

        match action {
            ExistingAction::Cancel => {
                println!("Cancelled.");
                return Ok(());
            }
            ExistingAction::Uninstall => {
                uninstall(yes).await?;
                return Ok(());
            }
            ExistingAction::Repair => {
                println!("Repairing {existing_label} install...");
            }
            ExistingAction::Swap => {
                println!(
                    "Swapping {} -> {} (models preserved)",
                    existing_label,
                    backend.label()
                );
            }
        }
    }

    println!("Backend: {}", backend.label());
    stop_daemon();

    let client = reqwest::Client::builder()
        .user_agent("ml5-installer/0.1")
        .build()?;

    let bindir = bin_dir();
    std::fs::create_dir_all(&bindir)?;

    fetch_binary(&client, "ml5.exe", &bindir).await?;
    fetch_binary(&client, "ml5d.exe", &bindir).await?;

    let (_, url) = llama_backend_url(backend);
    println!("Downloading llama.cpp {} backend...", backend.label());
    let zip_data = download_with_progress(&client, &url, "backend").await?;
    extract_backend_zip(&zip_data, &backends_dir())?;

    for extra in llama_extra_urls(backend) {
        println!("Downloading runtime support archive...");
        let data = download_with_progress(&client, &extra, "runtime").await?;
        extract_backend_zip(&data, &backends_dir())?;
    }

    write_marker(backend)?;
    std::fs::create_dir_all(models_dir())?;

    add_to_user_path(&bindir)?;

    println!();
    println!("ML5 installed to {}", install_dir().display());
    println!("  Binaries: {}", bindir.display());
    println!("  Models:   {}", models_dir().display());
    println!("  Backend:  {} ({})", backend.label(), backends_dir().display());
    println!();
    println!("Open a new terminal and try: ml5 status");
    Ok(())
}

async fn uninstall(yes: bool) -> Result<()> {
    if !is_installed() {
        println!("ML5 is not installed.");
        return Ok(());
    }
    let keep = ask(
        "Keep downloaded models?",
        &["Keep models", "Delete everything"],
        0,
        yes,
    )? == 0;

    stop_daemon();
    let dir = install_dir();
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    let backends = backends_dir();
    if backends.exists() {
        std::fs::remove_dir_all(&backends)?;
    }
    if !keep {
        let models = models_dir();
        if models.exists() {
            std::fs::remove_dir_all(&models)?;
        }
    }

    #[cfg(windows)]
    {
        let bin_str = bin_dir().to_string_lossy().to_string();
        let script = format!(
            "$p=[Environment]::GetEnvironmentVariable('Path','User');\
             $e=@($p -split ';' | Where-Object {{ $_ -ne '' -and ($_.TrimEnd('\\') -ine '{}') }});\
             [Environment]::SetEnvironmentVariable('Path', ($e -join ';'), 'User')",
            bin_str.trim_end_matches('\\')
        );
        let _ = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .output();
    }

    println!("ML5 uninstalled.");
    if keep {
        println!("Models kept in {}", models_dir().display());
    }
    Ok(())
}

pub async fn run_installer(backend: Backend) -> Result<()> {
    let cli = Cli::parse();

    if matches!(cli.cmd, Some(Cmd::Uninstall)) {
        return uninstall(cli.yes).await;
    }

    let gpus = detect_gpus();
    if !gpus.is_empty() {
        println!("Detected GPUs:");
        for g in &gpus {
            let cc = g
                .nvidia_cc
                .map(|(a, b)| format!(" (compute {a}.{b})"))
                .unwrap_or_default();
            println!("  {}{cc}", g.name);
        }
        println!();
    }

    install(backend, cli.yes).await
}
