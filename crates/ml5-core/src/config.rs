use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const MAX_MEMORY_FRACTION: f32 = 0.85;
pub const AUTO_CTX_MAX: u32 = 32768;
pub const AUTO_CTX_MIN: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuLayers {
    #[default]
    Auto,
    All,
    Count(u32),
}

impl GpuLayers {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "all" | "-1" | "max" => Ok(Self::All),
            other => other
                .parse::<u32>()
                .map(Self::Count)
                .map_err(|_| format!("invalid GPU layer value '{s}': use auto, all, -1, or a number")),
        }
    }

    pub fn from_request(n: i32) -> Self {
        if n < 0 {
            Self::All
        } else {
            Self::Count(n as u32)
        }
    }

    pub fn as_arg(&self) -> String {
        match self {
            Self::Auto => "auto".into(),
            Self::All => "all".into(),
            Self::Count(n) => n.to_string(),
        }
    }

    pub fn uses_gpu(&self) -> bool {
        !matches!(self, Self::Count(0))
    }
}

impl std::fmt::Display for GpuLayers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.as_arg())
    }
}

impl Serialize for GpuLayers {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Count(n) => s.serialize_u32(*n),
            other => s.serialize_str(&other.as_arg()),
        }
    }
}

impl<'de> Deserialize<'de> for GpuLayers {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        match v {
            serde_json::Value::Number(n) => Ok(Self::from_request(n.as_i64().unwrap_or(0) as i32)),
            serde_json::Value::String(s) => Self::parse(&s).map_err(serde::de::Error::custom),
            _ => Ok(Self::Auto),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: GpuLayers,
    pub main_gpu: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub flash_attn: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub cache_prompt: bool,
    pub cache_reuse: u32,
    pub opt_cache_prompt: bool,
    pub offload_kqv: bool,
    pub tensor_split: Option<String>,
    pub rpc_nodes: Option<String>,
}

impl ModelParams {
    pub fn with_overrides(&self, o: &crate::types::RequestOverrides) -> Self {
        let mut p = self.clone();
        if let Some(v) = o.n_ctx {
            p.n_ctx = v;
        }
        if let Some(v) = o.n_batch {
            p.n_batch = v;
            p.n_ubatch = p.n_ubatch.min(v);
        }
        if let Some(v) = o.n_gpu_layers {
            p.n_gpu_layers = v;
        }
        if let Some(v) = &o.flash_attn {
            p.flash_attn = v.clone();
        }
        if let Some(v) = &o.cache_type_k {
            p.cache_type_k = v.clone();
        }
        if let Some(v) = &o.cache_type_v {
            p.cache_type_v = v.clone();
        }
        if let Some(v) = o.n_threads {
            p.n_threads = v;
        }
        p
    }

    pub fn same_load(&self, other: &Self) -> bool {
        self.n_batch == other.n_batch
            && self.n_ubatch == other.n_ubatch
            && self.n_threads == other.n_threads
            && self.n_threads_batch == other.n_threads_batch
            && self.n_gpu_layers == other.n_gpu_layers
            && self.main_gpu == other.main_gpu
            && self.use_mmap == other.use_mmap
            && self.use_mlock == other.use_mlock
            && self.flash_attn == other.flash_attn
            && self.cache_type_k == other.cache_type_k
            && self.cache_type_v == other.cache_type_v
            && self.cache_prompt == other.cache_prompt
            && self.cache_reuse == other.cache_reuse
            && self.opt_cache_prompt == other.opt_cache_prompt
            && self.offload_kqv == other.offload_kqv
            && self.tensor_split == other.tensor_split
            && self.rpc_nodes == other.rpc_nodes
    }
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            n_ctx: 0,
            n_batch: 2048,
            n_ubatch: 512,
            n_threads: 0,
            n_threads_batch: 0,
            n_gpu_layers: GpuLayers::Auto,
            main_gpu: 0,
            use_mmap: true,
            use_mlock: false,
            flash_attn: "auto".into(),
            cache_type_k: "f16".into(),
            cache_type_v: "f16".into(),
            cache_prompt: true,
            cache_reuse: 0,
            opt_cache_prompt: false,
            offload_kqv: true,
            tensor_split: None,
            rpc_nodes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub models_dir: PathBuf,
	pub extra_model_dirs: Vec<PathBuf>,
    pub keep_alive_secs: i64,
    pub max_loaded_models: usize,
    pub model: ModelParams,
    pub max_memory_fraction: f32,
    pub n_parallel: usize,
    pub llama_server: Option<PathBuf>,
    pub auto_ctx_max: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 11435,
            models_dir: Self::models_dir_default(),
			extra_model_dirs: Vec::new(),
            keep_alive_secs: 900,
            max_loaded_models: 2,
            model: ModelParams::default(),
            max_memory_fraction: MAX_MEMORY_FRACTION,
            n_parallel: 1,
            llama_server: None,
            auto_ctx_max: AUTO_CTX_MAX,
        }
    }
}

impl Config {
    pub fn validate(&self) -> crate::Result<()> {
		if self.model_dirs().iter().any(|path| path.as_os_str().is_empty()) {
			return Err(crate::error::Ml5Error::InvalidRequest("model directories must not be empty".into()));
		}
        crate::types::RequestOverrides {
            n_ctx: Some(self.model.n_ctx),
            n_batch: Some(self.model.n_batch),
            n_gpu_layers: None,
            n_threads: Some(self.model.n_threads),
            flash_attn: Some(self.model.flash_attn.clone()),
            cache_type_k: Some(self.model.cache_type_k.clone()),
            cache_type_v: Some(self.model.cache_type_v.clone()),
        }
        .validate()?;
        if self.model.n_ubatch == 0
            || self.max_loaded_models == 0
            || self.n_parallel == 0
            || self.auto_ctx_max < AUTO_CTX_MIN
            || !self.max_memory_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.max_memory_fraction)
            || self.max_memory_fraction == 0.0
        {
            return Err(crate::error::Ml5Error::InvalidRequest(format!(
                "Batch/model/slot limits must be positive, the automatic context cap must be at least {AUTO_CTX_MIN}, and max_memory_fraction must be in (0, 1]."
            )));
        }
        Ok(())
    }

    pub fn ml5_home() -> PathBuf {
        if let Some(dir) = std::env::var_os("ML5_HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(dir);
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ml5")
    }

    pub fn models_dir_default() -> PathBuf {
        Self::ml5_home().join("models")
    }

	pub fn model_dirs(&self) -> Vec<PathBuf> {
		let mut dirs = vec![self.models_dir.clone()];
		for dir in &self.extra_model_dirs {
			if !dirs.contains(dir) {
				dirs.push(dir.clone());
			}
		}
		dirs
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

#[cfg(target_os = "linux")]
pub fn system_memory() -> Option<MemInfo> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<u64> {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kib| kib * 1024)
    };
    let total = field("MemTotal:")?;
    let avail = field("MemAvailable:").or_else(|| {
        Some(field("MemFree:")? + field("Cached:").unwrap_or(0) + field("Buffers:").unwrap_or(0))
    })?;
    let cgroup_limit = cgroup_memory_limit();
    Some(match cgroup_limit {
        Some(limit) if limit < total => MemInfo {
            total_bytes: limit,
            avail_bytes: avail.min(limit),
        },
        _ => MemInfo {
            total_bytes: total,
            avail_bytes: avail,
        },
    })
}

#[cfg(target_os = "linux")]
fn cgroup_memory_limit() -> Option<u64> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = cgroup.lines().find_map(|l| l.strip_prefix("0::"))?;
    let path = format!("/sys/fs/cgroup{}/memory.max", rel.trim());
    let raw = std::fs::read_to_string(path).ok()?;
    raw.trim().parse::<u64>().ok()
}

#[cfg(target_os = "macos")]
pub fn system_memory() -> Option<MemInfo> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let total: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    let vm = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&vm.stdout);
    let page: u64 = text
        .lines()
        .next()
        .and_then(|l| l.split("page size of ").nth(1))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);
    let pages = |name: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().trim_end_matches('.').parse::<u64>().ok())
            .unwrap_or(0)
    };
    let avail = (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")) * page;
    Some(MemInfo {
        total_bytes: total,
        avail_bytes: avail.min(total),
    })
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub fn system_memory() -> Option<MemInfo> {
    None
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemory {
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub unified: bool,
}

pub fn gpu_memory_cached(max_age: std::time::Duration) -> Option<GpuMemory> {
    static CACHE: std::sync::Mutex<Option<(std::time::Instant, Option<GpuMemory>)>> =
        std::sync::Mutex::new(None);
    if let Ok(cache) = CACHE.lock() {
        if let Some((at, value)) = *cache {
            if at.elapsed() < max_age {
                return value;
            }
        }
    }
    let value = gpu_memory();
    if let Ok(mut cache) = CACHE.lock() {
        *cache = Some((std::time::Instant::now(), value));
    }
    value
}

pub fn gpu_memory() -> Option<GpuMemory> {
    if let Some((total, free)) = nvidia_smi_memory() {
        return Some(GpuMemory {
            total_bytes: total,
            free_bytes: free.unwrap_or(total),
            unified: false,
        });
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(mem) = linux_drm_vram() {
            return Some(mem);
        }
    }
    #[cfg(target_os = "macos")]
    {
        return system_memory().map(|m| GpuMemory {
            total_bytes: m.total_bytes,
            free_bytes: m.avail_bytes,
            unified: true,
        });
    }
    #[cfg(windows)]
    {
        return wmi_gpu_total_memory().map(|t| GpuMemory {
            total_bytes: t,
            free_bytes: t,
            unified: false,
        });
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(target_os = "linux")]
fn linux_drm_vram() -> Option<GpuMemory> {
    let mut best: Option<GpuMemory> = None;
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = entry.path().join("device");
        let read = |f: &str| -> Option<u64> {
            std::fs::read_to_string(dev.join(f))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
        };
        if let Some(total) = read("mem_info_vram_total") {
            let used = read("mem_info_vram_used").unwrap_or(0);
            let candidate = GpuMemory {
                total_bytes: total,
                free_bytes: total.saturating_sub(used),
                unified: total < 2 * 1024 * 1024 * 1024,
            };
            if best.is_none_or(|b| candidate.total_bytes > b.total_bytes) {
                best = Some(candidate);
            }
        }
    }
    best
}

fn nvidia_smi_memory() -> Option<(u64, Option<u64>)> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut total_mib: u64 = 0;
    let mut free_mib: Option<u64> = None;
    for line in text.lines() {
        let mut fields = line.split(',').map(|f| f.trim().parse::<u64>().ok());
        let Some(Some(total)) = fields.next() else { continue };
        total_mib += total;
        if let Some(Some(free)) = fields.next() {
            free_mib = Some(free_mib.unwrap_or(0) + free);
        }
    }
    if total_mib == 0 {
        None
    } else {
        Some((total_mib * 1024 * 1024, free_mib.map(|f| f * 1024 * 1024)))
    }
}

#[cfg(windows)]
fn wmi_gpu_total_memory() -> Option<u64> {
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

#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    pub budget_bytes: u64,
    pub sys_total: u64,
    pub sys_avail: u64,
    pub gpu_total: u64,
    pub gpu_free: u64,
    pub max_fraction: f32,
}

impl MemoryBudget {
    pub fn known(&self) -> bool {
        self.sys_total > 0
    }

    pub fn fits(&self, need: u64) -> bool {
        !self.known() || need <= self.budget_bytes
    }

    pub fn describe_failure(&self, need: u64) -> String {
        format!(
            "model + context (~{}) would not fit the memory budget ({} = {:.0}% of {} total RAM{}). Use a smaller quantization, a smaller context (--ctx-size), or unload other models.",
            humansize(need),
            humansize(self.budget_bytes),
            self.max_fraction * 100.0,
            humansize(self.sys_total),
            if self.gpu_total > 0 {
                format!(" + {} VRAM", humansize(self.gpu_total))
            } else {
                String::new()
            },
        )
    }
}

pub fn humansize(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1e9)
    } else {
        format!("{:.0} MB", b as f64 / 1e6)
    }
}

pub fn memory_budget(max_fraction: f32, gpu_offload: bool) -> MemoryBudget {
    let sys = system_memory().unwrap_or(MemInfo {
        total_bytes: 0,
        avail_bytes: 0,
    });
    let gpu = if gpu_offload {
        gpu_memory().filter(|g| !g.unified).unwrap_or_default()
    } else {
        GpuMemory::default()
    };
    let budget = (sys.total_bytes as f64 * max_fraction as f64) as u64
        + (gpu.total_bytes as f64 * max_fraction as f64) as u64;
    MemoryBudget {
        budget_bytes: budget,
        sys_total: sys.total_bytes,
        sys_avail: sys.avail_bytes,
        gpu_total: gpu.total_bytes,
        gpu_free: gpu.free_bytes,
        max_fraction,
    }
}
