use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const MAX_MEMORY_FRACTION: f32 = 0.85;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelParams {
    #[serde(default = "d_ctx")]
    pub n_ctx: u32,
    #[serde(default = "d_batch")]
    pub n_batch: u32,
    #[serde(default = "d_ubatch")]
    pub n_ubatch: u32,
    #[serde(default)]
    pub n_threads: i32,
    #[serde(default)]
    pub n_threads_batch: i32,
    #[serde(default = "d_gpu_layers")]
    pub n_gpu_layers: i32,
    #[serde(default)]
    pub main_gpu: i32,
    #[serde(default = "d_true")]
    pub use_mmap: bool,
    #[serde(default)]
    pub use_mlock: bool,
    #[serde(default = "d_flash")]
    pub flash_attn: String,
    #[serde(default = "d_kv")]
    pub cache_type_k: String,
    #[serde(default = "d_kv")]
    pub cache_type_v: String,
    #[serde(default = "d_true")]
    pub offload_kqv: bool,
    #[serde(default)]
    pub embeddings: bool,
}

fn d_ctx() -> u32 {
    2048
}
fn d_batch() -> u32 {
    2048
}
fn d_ubatch() -> u32 {
    512
}
fn d_gpu_layers() -> i32 {
    0
}
fn d_true() -> bool {
    true
}
fn d_flash() -> String {
    "auto".into()
}
fn d_kv() -> String {
    "f16".into()
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_ctx: d_ctx(),
            n_batch: d_batch(),
            n_ubatch: d_ubatch(),
            n_threads: 0,
            n_threads_batch: 0,
            n_gpu_layers: d_gpu_layers(),
            main_gpu: 0,
            use_mmap: true,
            use_mlock: false,
            flash_attn: d_flash(),
            cache_type_k: d_kv(),
            cache_type_v: d_kv(),
            offload_kqv: true,
            embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub models_dir: PathBuf,
    pub keep_alive_secs: u64,
    pub max_loaded_models: usize,
    pub model: ModelParams,
    pub max_memory_fraction: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 11435,
            models_dir: Self::models_dir_default(),
            keep_alive_secs: 300,
            max_loaded_models: 2,
            model: ModelParams::default(),
            max_memory_fraction: MAX_MEMORY_FRACTION,
        }
    }
}

impl Config {
    pub fn validate(&self) -> crate::Result<()> {
        crate::types::RequestOverrides {
            n_ctx: Some(self.model.n_ctx),
            n_batch: Some(self.model.n_batch),
            n_gpu_layers: Some(self.model.n_gpu_layers),
            n_threads: Some(self.model.n_threads),
            flash_attn: Some(self.model.flash_attn.clone()),
            cache_type_k: Some(self.model.cache_type_k.clone()),
            cache_type_v: Some(self.model.cache_type_v.clone()),
        }
        .validate()?;
        if self.model.n_ubatch == 0
            || self.max_loaded_models == 0
            || !self.max_memory_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_memory_fraction)
            || self.max_memory_fraction == 0.0
        {
            return Err(crate::error::Ml5Error::InvalidRequest(
                "Batch/model limits must be positive and max_memory_fraction must be in (0, 1]."
                    .into(),
            ));
        }
        Ok(())
    }
    pub fn models_dir_default() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ml5")
            .join("models")
    }

    pub fn log_dir(&self) -> PathBuf {
        self.models_dir
            .parent()
            .map(|p| p.join("logs"))
            .unwrap_or_else(|| PathBuf::from("./logs"))
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.models_dir)?;
        std::fs::create_dir_all(self.log_dir())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemInfo {
    pub total_bytes: u64,
    pub avail_bytes: u64,
}

#[cfg(windows)]
pub fn system_memory() -> Option<MemInfo> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
    }
    let mut st = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut st) };
    if ok == 0 {
        return None;
    }
    Some(MemInfo {
        total_bytes: st.total_phys,
        avail_bytes: st.avail_phys,
    })
}

#[cfg(not(windows))]
pub fn system_memory() -> Option<MemInfo> {
    None
}

#[allow(dead_code)]
pub fn free_gpu_memory_bytes() -> Option<u64> {
    if let Some(v) = nvidia_smi_memory("memory.free") {
        return Some(v);
    }
    #[cfg(windows)]
    {
        wmi_gpu_free_memory()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

pub fn total_gpu_memory_bytes() -> Option<u64> {
    if let Some(v) = nvidia_smi_memory("memory.total") {
        return Some(v);
    }
    #[cfg(windows)]
    {
        wmi_gpu_total_memory()
    }
    #[cfg(not(windows))]
    {
        None
    }
}

fn nvidia_smi_memory(field: &str) -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            format!("--query-gpu={field}"),
            "--format=csv,noheader,nounits".into(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mib: u64 = text
        .lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .sum();
    if mib == 0 {
        None
    } else {
        Some(mib * 1024 * 1024)
    }
}

#[cfg(windows)]
fn wmi_gpu_free_memory() -> Option<u64> {
    let out = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance -ClassName Win32_VideoController | Measure-Object -Property AdapterRAM -Maximum).Maximum",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let bytes: u64 = text.trim().parse().ok()?;
    if bytes == 0 {
        None
    } else {
        Some(bytes)
    }
}

#[cfg(windows)]
fn wmi_gpu_total_memory() -> Option<u64> {
    wmi_gpu_free_memory()
}

pub struct MemoryBudget {
    pub budget_bytes: u64,
    pub sys_total: u64,
    pub sys_avail: u64,
    pub gpu_total: u64,
    pub gpu_offload: bool,
    pub max_fraction: f32,
}

impl MemoryBudget {
    pub fn check(&self, model_size: u64, est_ctx_bytes: u64) -> Result<(), String> {
        if self.sys_total == 0 && self.gpu_total == 0 {
            return Ok(());
        }
        let need = model_size.saturating_add(est_ctx_bytes);
        if need > self.budget_bytes {
            return Err(format!(
                "model + KV cache (~{:.1} GB) would exceed {:.0}% of total memory ({:.1} GB budget: {} RAM total [{} avail] + {} VRAM total). Lower ctx, use a smaller quant, or reduce gpu layers.",
                need as f64 / 1e9,
                self.max_fraction * 100.0,
                self.budget_bytes as f64 / 1e9,
                humansize(self.sys_total),
                humansize(self.sys_avail),
                humansize(self.gpu_total),
            ));
        }
        Ok(())
    }
}

fn humansize(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

pub fn memory_budget(max_fraction: f32, gpu_offload: bool) -> MemoryBudget {
    let sys = system_memory().unwrap_or(MemInfo {
        total_bytes: 0,
        avail_bytes: 0,
    });
    let gpu = if gpu_offload {
        total_gpu_memory_bytes().unwrap_or(0)
    } else {
        0
    };
    let gpu_counted = if gpu_offload { gpu } else { 0 };
    let budget = (sys.total_bytes as f64 * max_fraction as f64) as u64
        + (gpu_counted as f64 * max_fraction as f64) as u64;
    MemoryBudget {
        budget_bytes: budget,
        sys_total: sys.total_bytes,
        sys_avail: sys.avail_bytes,
        gpu_total: gpu,
        gpu_offload,
        max_fraction,
    }
}
