use crate::backend::{Backend, LoadSpec, TokenStream};
use crate::config::{Config, ModelParams, AUTO_CTX_MIN};
use crate::error::{Ml5Error, Result};
use crate::gguf::{self, GgufInfo};
use crate::model::{self, ModelDetails, ModelInfo, ModelMeta, ModelType};
use crate::pull::PullEvent;
use crate::types::*;
use futures::StreamExt;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, Mutex, RwLock};

const REAPER_INTERVAL: Duration = Duration::from_secs(5);
const RUNTIME_OVERHEAD: u64 = 512 * 1024 * 1024;
const VISION_OVERHEAD: u64 = 256 * 1024 * 1024;
const UNKNOWN_MEMORY_CTX: u32 = 8192;

fn now_millis() -> u64 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64
}

pub struct LoadedModel {
    pub info: ModelInfo,
    pub backend: Arc<dyn Backend>,
    pub params: ModelParams,
    pub n_parallel: usize,
    pub embeddings: bool,
    pub need_bytes: u64,
    pub loaded_at: chrono::DateTime<chrono::Utc>,
    last_used: AtomicU64,
    active: AtomicUsize,
    keep_alive: AtomicI64,
}

impl LoadedModel {
    fn touch(&self) {
        self.last_used.store(now_millis(), Ordering::Relaxed);
    }

    pub fn active_requests(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }

    fn idle_millis(&self) -> u64 {
        now_millis().saturating_sub(self.last_used.load(Ordering::Relaxed))
    }

    fn expired(&self) -> bool {
        let keep = self.keep_alive.load(Ordering::Relaxed);
        keep >= 0 && self.idle_millis() >= keep as u64 * 1000
    }
}

pub struct ActiveGuard(Arc<LoadedModel>);

impl ActiveGuard {
    fn new(model: Arc<LoadedModel>) -> Self {
        model.active.fetch_add(1, Ordering::AcqRel);
        model.touch();
        Self(model)
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.touch();
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl std::ops::Deref for ActiveGuard {
    type Target = LoadedModel;
    fn deref(&self) -> &LoadedModel {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PullState {
    pub id: String,
    pub target: String,
    pub file: Option<String>,
    pub stage: String,
    pub downloaded: u64,
    pub total: Option<u64>,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone)]
pub struct PullHandle {
    pub id: String,
    pub key: String,
    pub tx: broadcast::Sender<PullEvent>,
    pub cancel: Arc<AtomicBool>,
    state: Arc<std::sync::Mutex<PullState>>,
}

impl PullHandle {
    pub fn record(&self, ev: &PullEvent) {
        if let Ok(mut s) = self.state.lock() {
            match ev {
                PullEvent::Started { .. } => {}
                PullEvent::Resolving { target } => {
                    s.stage = "resolving".into();
                    s.target = target.clone();
                }
                PullEvent::Downloading { file } => {
                    s.stage = "downloading".into();
                    s.file = Some(file.clone());
                }
                PullEvent::Progress { downloaded, total } => {
                    s.downloaded = *downloaded;
                    s.total = *total;
                }
                PullEvent::Verifying { file } => {
                    s.stage = "verifying".into();
                    s.file = Some(file.clone());
                }
                PullEvent::Done { .. } => s.stage = "done".into(),
                PullEvent::Error { .. } => s.stage = "error".into(),
            }
        }
    }

    pub fn snapshot(&self) -> Option<PullState> {
        self.state.lock().ok().map(|s| s.clone())
    }

    pub fn send(&self, ev: PullEvent) {
        self.record(&ev);
        let _ = self.tx.send(ev);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadedStatus {
    pub name: String,
    pub friendly_name: Option<String>,
    pub backend: String,
    pub base_url: Option<String>,
    pub n_ctx: u32,
    pub n_parallel: usize,
    pub gpu_layers: String,
    pub embeddings: bool,
    pub vision: bool,
    pub active_requests: usize,
    pub idle_secs: u64,
    pub expires_in_secs: Option<i64>,
    pub loaded_at: chrono::DateTime<chrono::Utc>,
    pub memory_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeStatus {
    pub server: Option<String>,
    pub release_tag: &'static str,
    pub configured: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryStatus {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub gpu_total_bytes: u64,
    pub gpu_free_bytes: u64,
    pub committed_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineStatus {
    pub version: String,
    pub model_count: usize,
    pub loaded_models: Vec<String>,
    pub loaded: Vec<LoadedStatus>,
    pub loading: Vec<String>,
    pub pulls: Vec<PullState>,
    pub uptime_secs: u64,
    pub models_dir: String,
	pub model_dirs: Vec<String>,
    pub log_dir: String,
    pub defaults: ModelParams,
    pub max_loaded_models: usize,
    pub keep_alive_secs: i64,
    pub n_parallel: usize,
    pub auto_ctx_max: u32,
    pub runtime: RuntimeStatus,
    pub memory: MemoryStatus,
    pub safetensors: bool,
}

type GgufCache = HashMap<PathBuf, (u64, Option<SystemTime>, Option<GgufInfo>)>;

pub struct Engine {
    pub config: Config,
    models: RwLock<HashMap<String, ModelInfo>>,
    loaded: RwLock<HashMap<String, Arc<LoadedModel>>>,
    loading: RwLock<HashSet<String>>,
    pulls: RwLock<HashMap<String, PullHandle>>,
    gguf_cache: std::sync::Mutex<GgufCache>,
    lifecycle: Mutex<()>,
    started: Instant,
}

impl Engine {
    pub fn new(config: Config) -> Result<Arc<Self>> {
        config.validate()?;
        config.ensure_dirs()?;
        let engine = Arc::new(Self {
            config,
            models: RwLock::new(HashMap::new()),
            loaded: RwLock::new(HashMap::new()),
            loading: RwLock::new(HashSet::new()),
            pulls: RwLock::new(HashMap::new()),
            gguf_cache: std::sync::Mutex::new(HashMap::new()),
            lifecycle: Mutex::new(()),
            started: Instant::now(),
        });
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let weak = Arc::downgrade(&engine);
            handle.spawn(reaper(weak));
        }
        Ok(engine)
    }

    pub async fn register_pull(&self, key: &str, target: &str) -> (PullHandle, bool) {
        let mut pulls = self.pulls.write().await;
        if let Some(existing) = pulls.get(key) {
            return (existing.clone(), false);
        }
        let (tx, _) = broadcast::channel(256);
        let id = uuid::Uuid::new_v4().simple().to_string();
        let handle = PullHandle {
            id: id.clone(),
            key: key.to_string(),
            tx,
            cancel: Arc::new(AtomicBool::new(false)),
            state: Arc::new(std::sync::Mutex::new(PullState {
                id,
                target: target.to_string(),
                file: None,
                stage: "starting".into(),
                downloaded: 0,
                total: None,
                started_at: chrono::Utc::now(),
            })),
        };
        pulls.insert(key.to_string(), handle.clone());
        (handle, true)
    }

    pub async fn finish_pull(&self, key: &str) {
        self.pulls.write().await.remove(key);
    }

    pub async fn cancel_pull(&self, id: &str) -> bool {
        let pulls = self.pulls.read().await;
        match pulls.values().find(|p| p.id == id) {
            Some(p) => {
                p.cancel.store(true, Ordering::Release);
                true
            }
            None => false,
        }
    }

    pub async fn list_pulls(&self) -> Vec<PullState> {
        let mut out: Vec<PullState> = self
            .pulls
            .read()
            .await
            .values()
            .filter_map(PullHandle::snapshot)
            .collect();
        out.sort_by_key(|p| p.started_at);
        out
    }

    pub async fn scan_models(&self) -> Result<()> {
		let roots = self.config.model_dirs();
        let cache = self
            .gguf_cache
            .lock()
            .map(|c| c.clone())
            .unwrap_or_default();
		let (models, cache) = tokio::task::spawn_blocking(move || {
			let mut stores = Vec::new();
			let mut next_cache = GgufCache::new();
			let mut directories = HashSet::new();
			let mut paths = HashSet::new();
			for (index, root) in roots.iter().enumerate() {
				let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.clone());
				if !directories.insert(canonical) {
					continue;
				}
				let (found, updated) = match scan_store(root, &cache) {
					Ok(scanned) => scanned,
					Err(error) if index > 0 => {
						tracing::warn!(path = %root.display(), %error, "could not scan additional model directory; rescan after the drive becomes available");
						continue;
					}
					Err(error) => return Err(error),
				};
				next_cache.extend(updated);
				let mut found: Vec<ModelInfo> = found.into_values().collect();
				found.sort_by(|a, b| a.name.cmp(&b.name));
				for info in found {
					let canonical = std::fs::canonicalize(&info.path).unwrap_or_else(|_| info.path.clone());
					if paths.insert(canonical) {
						stores.push((index + 1, info));
					}
				}
			}
			let reserved: HashSet<String> = stores.iter().map(|(_, info)| info.name.clone()).collect();
			let mut models = HashMap::new();
			for (store, mut info) in stores {
				if models.contains_key(&info.name) {
					let original = info.name.clone();
					let mut alias = format!("store{store}/{original}");
					let mut suffix = 2;
					while models.contains_key(&alias) || reserved.contains(&alias) {
						alias = format!("store{store}-{suffix}/{original}");
						suffix += 1;
					}
					tracing::warn!(model = %original, alias = %alias, path = %info.path.display(), "duplicate model name across stores; using a distinct name");
					info.name = alias;
				}
				models.insert(info.name.clone(), info);
			}
			Ok::<_, Ml5Error>((models, next_cache))
		})
            .await
            .map_err(|e| Ml5Error::Backend(format!("model scan task failed: {e}")))??;
        if let Ok(mut c) = self.gguf_cache.lock() {
            *c = cache;
        }
        tracing::info!(count = models.len(), "scanned model store");
        *self.models.write().await = models;
        Ok(())
    }

    pub async fn list_models(&self) -> Vec<ModelInfo> {
        let mut models: Vec<_> = self.models.read().await.values().cloned().collect();
        models.sort_by(|a, b| a.name.cmp(&b.name));
        models
    }

    pub async fn get_model(&self, name: &str) -> Result<ModelInfo> {
        let models = self.models.read().await;
        let wanted = name.trim();
        let candidates = [
            wanted,
            wanted.strip_suffix(":latest").unwrap_or(wanted),
            wanted.strip_suffix(".gguf").unwrap_or(wanted),
        ];
        for c in candidates {
            if let Some(info) = models.get(c) {
                return Ok(info.clone());
            }
        }
        let normalize = |s: &str| -> String {
            s.chars()
                .map(|c| {
                    if c.is_alphanumeric() {
                        c.to_ascii_lowercase()
                    } else {
                        '-'
                    }
                })
                .collect()
        };
        let target = normalize(candidates[1]);
		let mut matches: Vec<&ModelInfo> = models.values().filter(|m| normalize(&m.name) == target).collect();
		if matches.is_empty() {
			matches = models.values().filter(|m| m.friendly_name.as_deref().is_some_and(|f| normalize(f) == target)).collect();
		}
		if matches.len() == 1 {
			return Ok(matches[0].clone());
		}
		if matches.len() > 1 {
			let mut names: Vec<&str> = matches.iter().map(|m| m.name.as_str()).collect();
			names.sort_unstable();
			return Err(Ml5Error::InvalidRequest(format!("model name '{name}' is ambiguous; use an exact name from `ml5 list`: {}", names.join(", "))));
		}
        Err(Ml5Error::ModelNotFound(format!(
            "{name}. Run `ml5 list` for installed names or `ml5 pull hf:owner/repo` to download one."
        )))
    }

    pub async fn rename_model(&self, name: &str, new_friendly_name: &str) -> Result<()> {
        let info = self.get_model(name).await?;
        let friendly = new_friendly_name.trim();
        if friendly.is_empty() {
            return Err(Ml5Error::InvalidRequest("the new name must not be empty".into()));
        }
        let meta_path = info
            .meta_path
            .clone()
            .unwrap_or_else(|| model::gguf_meta_path(&info.path));
        let mut meta = ModelMeta::load(&meta_path);
        meta.friendly_name = Some(friendly.to_string());
        meta.save(&meta_path)?;
        if let Some(m) = self.models.write().await.get_mut(&info.name) {
            m.friendly_name = Some(friendly.to_string());
            m.meta_path = Some(meta_path);
        }
        Ok(())
    }

    pub async fn delete_model(&self, name: &str) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        let info = self.get_model(name).await?;
		let loaded: Vec<String> = self.loaded.read().await.values()
			.filter(|m| m.info.name == info.name || m.info.path == info.path)
			.map(|m| m.info.name.clone()).collect();
		for name in loaded {
			self.remove_loaded(&name, true).await?;
		}
        match info.model_type {
            ModelType::Safetensors => tokio::fs::remove_dir_all(&info.path).await?,
            ModelType::Gguf => {
                let files = if info.files.is_empty() {
                    vec![info.path.clone()]
                } else {
                    info.files.clone()
                };
                for f in files {
                    match tokio::fs::remove_file(&f).await {
                        Ok(()) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                        Err(e) => return Err(e.into()),
                    }
                }
                if let Some(meta) = &info.meta_path {
                    let _ = tokio::fs::remove_file(meta).await;
                }
                if let Some(mmproj) = &info.mmproj {
                    let shared = self
                        .models
                        .read()
                        .await
                        .values()
                        .any(|m| m.name != info.name && m.mmproj.as_ref() == Some(mmproj));
                    if !shared {
                        let _ = tokio::fs::remove_file(mmproj).await;
                    }
                }
            }
        }
        self.models.write().await.remove(&info.name);
        if let Ok(mut c) = self.gguf_cache.lock() {
            c.retain(|p, _| p.exists());
        }
        Ok(())
    }

    pub async fn unload_model(&self, name: &str) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        let info = self.get_model(name).await?;
        self.remove_loaded(&info.name, false).await
    }

    pub async fn unload_all(&self) {
        let _guard = self.lifecycle.lock().await;
        let all: Vec<Arc<LoadedModel>> = self.loaded.write().await.drain().map(|(_, m)| m).collect();
        for m in all {
            if let Err(e) = m.backend.unload().await {
                tracing::warn!(model = %m.info.name, %e, "unload during shutdown failed");
            }
        }
    }

    async fn remove_loaded(&self, name: &str, force: bool) -> Result<()> {
        let removed = {
            let mut loaded = self.loaded.write().await;
            match loaded.get(name) {
                Some(m) if m.active_requests() > 0 && !force => {
                    return Err(Ml5Error::InvalidRequest(format!(
                        "Model '{name}' is busy; cancel or finish its requests first."
                    )));
                }
                Some(_) => loaded.remove(name),
                None => None,
            }
        };
        if let Some(m) = removed {
            m.backend.unload().await?;
            tracing::info!(model = %name, "model unloaded");
        }
        Ok(())
    }

    pub async fn status(&self) -> EngineStatus {
        let (sys, gpu, server) = {
            let configured = self.config.llama_server.clone();
            tokio::task::spawn_blocking(move || {
                (
                    crate::config::system_memory(),
                    crate::config::gpu_memory_cached(std::time::Duration::from_secs(5))
                        .filter(|g| !g.unified)
                        .unwrap_or_default(),
                    crate::runtime::locate_server(configured.as_deref()).map(|p| p.display().to_string()),
                )
            })
            .await
            .unwrap_or_default()
        };
        let models = self.models.read().await;
        let loaded = self.loaded.read().await;
        let mut loaded_status: Vec<LoadedStatus> = loaded
            .values()
            .map(|m| {
                let keep = m.keep_alive.load(Ordering::Relaxed);
                let idle = m.idle_millis() / 1000;
                LoadedStatus {
                    name: m.info.name.clone(),
                    friendly_name: m.info.friendly_name.clone(),
                    backend: m.backend.name().to_string(),
                    base_url: m.backend.base_url(),
                    n_ctx: m.params.n_ctx,
                    n_parallel: m.n_parallel,
                    gpu_layers: m.params.n_gpu_layers.as_arg(),
                    embeddings: m.embeddings,
                    vision: m.info.mmproj.is_some() && !m.embeddings,
                    active_requests: m.active_requests(),
                    idle_secs: idle,
                    expires_in_secs: (keep >= 0 && m.active_requests() == 0)
                        .then(|| (keep - idle as i64).max(0)),
                    loaded_at: m.loaded_at,
                    memory_bytes: m.need_bytes,
                }
            })
            .collect();
        loaded_status.sort_by(|a, b| a.name.cmp(&b.name));
        let committed: u64 = loaded.values().map(|m| m.need_bytes).sum();
        let mut loading: Vec<String> = self.loading.read().await.iter().cloned().collect();
        loading.sort();
        EngineStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_count: models.len(),
            loaded_models: loaded_status.iter().map(|m| m.name.clone()).collect(),
            loaded: loaded_status,
            loading,
            pulls: self.list_pulls().await,
            uptime_secs: self.started.elapsed().as_secs(),
            models_dir: self.config.models_dir.display().to_string(),
			model_dirs: self.config.model_dirs().iter().map(|p| p.display().to_string()).collect(),
            log_dir: self.config.log_dir().display().to_string(),
            defaults: self.config.model.clone(),
            max_loaded_models: self.config.max_loaded_models,
            keep_alive_secs: self.config.keep_alive_secs,
            n_parallel: self.config.n_parallel,
            auto_ctx_max: self.config.auto_ctx_max,
            runtime: RuntimeStatus {
                server,
                release_tag: crate::runtime::LLAMA_RELEASE_TAG,
                configured: self
                    .config
                    .llama_server
                    .as_ref()
                    .map(|p| p.display().to_string()),
            },
            memory: MemoryStatus {
                total_bytes: sys.map(|s| s.total_bytes).unwrap_or(0),
                available_bytes: sys.map(|s| s.avail_bytes).unwrap_or(0),
                gpu_total_bytes: gpu.total_bytes,
                gpu_free_bytes: gpu.free_bytes,
                committed_bytes: committed,
            },
            safetensors: cfg!(feature = "safetensors"),
        }
    }

    pub async fn loaded_contexts(&self) -> HashMap<String, u32> {
		let models = self.models.read().await;
        self.loaded
            .read()
            .await
            .values()
			.filter(|m| models.get(&m.info.name).is_some_and(|info| info.path == m.info.path))
            .map(|m| (m.info.name.clone(), m.params.n_ctx))
            .collect()
    }

    pub async fn is_loaded(&self, name: &str) -> bool {
        match self.get_model(name).await {
			Ok(info) => self.loaded.read().await.get(&info.name).is_some_and(|m| m.info.path == info.path),
            Err(_) => false,
        }
    }

    pub async fn preload(
        &self,
        name: &str,
        overrides: &RequestOverrides,
        keep_alive: Option<i64>,
    ) -> Result<String> {
        overrides.validate()?;
        let info = self.get_model(name).await?;
        let embeddings = !info.has(Capability::Chat);
        let _guard = self.acquire(&info, overrides, embeddings, keep_alive).await?;
        Ok(info.name)
    }

    pub async fn chat(&self, mut req: ChatRequest) -> Result<TokenStream> {
        req.params.validate()?;
        req.overrides.validate()?;
        if req.messages.is_empty() {
            return Err(Ml5Error::InvalidRequest("messages must not be empty".into()));
        }
        let info = self.get_model(&req.model).await?;
        if !info.has(Capability::Chat) {
            return Err(Ml5Error::InvalidRequest(format!(
                "'{}' is an embedding model and cannot chat; use it with /v1/embeddings.",
                info.name
            )));
        }
        if req.messages.iter().any(|m| !m.images.is_empty()) && !info.has(Capability::Vision) {
            return Err(Ml5Error::InvalidRequest(format!(
                "'{}' cannot read images: it has no vision projector (mmproj). Pull a vision model, or place its mmproj-*.gguf next to the model.",
                info.name
            )));
        }
        let guard = self
            .acquire(&info, &req.overrides, false, req.keep_alive_secs)
            .await?;
        req.model = info.name.clone();
        let stream = guard.backend.chat(req).await?;
        Ok(guarded(stream, guard))
    }

    pub async fn generate(&self, mut req: GenerateRequest) -> Result<TokenStream> {
        req.params.validate()?;
        req.overrides.validate()?;
        if req.prompt.is_empty() {
            return Err(Ml5Error::InvalidRequest("prompt must not be empty".into()));
        }
        let info = self.get_model(&req.model).await?;
        if !info.has(Capability::Generate) {
            return Err(Ml5Error::InvalidRequest(format!(
                "'{}' is an embedding model and cannot generate text.",
                info.name
            )));
        }
        let guard = self
            .acquire(&info, &req.overrides, false, req.keep_alive_secs)
            .await?;
        req.model = info.name.clone();
        let stream = guard.backend.generate(req).await?;
        Ok(guarded(stream, guard))
    }

    pub async fn count_tokens(&self, mut req: ChatRequest) -> Result<u32> {
        let info = self.get_model(&req.model).await?;
        if !info.has(Capability::Chat) {
            return Err(Ml5Error::InvalidRequest(format!(
                "'{}' is an embedding model and has no chat template.",
                info.name
            )));
        }
        let guard = self.acquire(&info, &req.overrides, false, None).await?;
        req.model = info.name.clone();
        guard.backend.count_tokens(&req).await
    }

    pub async fn embed(&self, mut req: EmbedRequest) -> Result<EmbedResponse> {
        if req.input.is_empty() || req.input.iter().all(|s| s.is_empty()) {
            return Err(Ml5Error::InvalidRequest("input must not be empty".into()));
        }
        let info = self.get_model(&req.model).await?;
        if !info.has(Capability::Embed) {
            return Err(Ml5Error::InvalidRequest(format!(
                "'{}' is a chat model, not an embedding model. Pull an embedding model (for example hf:nomic-ai/nomic-embed-text-v1.5-GGUF) for embeddings.",
                info.name
            )));
        }
        let guard = self
            .acquire(&info, &RequestOverrides::default(), true, req.keep_alive_secs)
            .await?;
        req.model = info.name.clone();
        guard.backend.embed(req).await
    }

    fn compatible(
        m: &LoadedModel,
        wanted: &ModelParams,
        overrides: &RequestOverrides,
        embeddings: bool,
    ) -> bool {
        if m.embeddings != embeddings {
            return false;
        }
        if !overrides.affects_load() {
            return true;
        }
        let ctx_ok = match overrides.n_ctx {
            None | Some(0) => true,
            Some(n) => m.params.n_ctx >= n,
        };
        ctx_ok && m.params.same_load(wanted)
    }

    async fn acquire(
        &self,
        info: &ModelInfo,
        overrides: &RequestOverrides,
        embeddings: bool,
        keep_alive: Option<i64>,
    ) -> Result<ActiveGuard> {
        let wanted = self.config.model.with_overrides(overrides);
        let fast = {
            let loaded = self.loaded.read().await;
            loaded
                .get(&info.name)
				.filter(|m| m.info.path == info.path && Self::compatible(m, &wanted, overrides, embeddings))
                .map(|m| ActiveGuard::new(m.clone()))
        };
        if let Some(guard) = fast {
            if guard.backend.is_alive().await {
                if let Some(k) = keep_alive {
                    guard.keep_alive.store(k, Ordering::Relaxed);
                }
                return Ok(guard);
            }
        }

        let _lifecycle = self.lifecycle.lock().await;
        let existing = self.loaded.read().await.get(&info.name).cloned();
        if let Some(m) = existing {
            let alive = m.backend.is_alive().await;
			if alive && m.info.path == info.path && Self::compatible(&m, &wanted, overrides, embeddings) {
                let guard = ActiveGuard::new(m);
                if let Some(k) = keep_alive {
                    guard.keep_alive.store(k, Ordering::Relaxed);
                }
                return Ok(guard);
            }
            let removed = {
                let mut loaded = self.loaded.write().await;
                if alive && m.active_requests() > 0 {
                    return Err(Ml5Error::InvalidRequest(format!(
						"Model '{}' is serving other requests from a different file or with different load settings (context, GPU layers, KV cache, or embedding mode); retry when they finish.",
                        info.name
                    )));
                }
                loaded.remove(&info.name)
            };
            if !alive {
                tracing::warn!(model = %info.name, "model runtime exited unexpectedly; restarting it");
            } else {
                tracing::info!(model = %info.name, "reloading model with new settings");
            }
            if let Some(old) = removed {
                old.backend.unload().await.ok();
            }
        }

        let n_parallel = if embeddings { 1 } else { self.config.n_parallel.max(1) };
        while self.loaded.read().await.len() >= self.config.max_loaded_models {
            if !self.evict_one(&info.name).await {
                return Err(Ml5Error::InvalidRequest(format!(
                    "All {} model slots are busy; retry when a request finishes, or raise --max-loaded-models.",
                    self.config.max_loaded_models
                )));
            }
        }

        let spec = loop {
            let committed: u64 = self.loaded.read().await.values().map(|m| m.need_bytes).sum();
            match self.plan_load(info, &wanted, embeddings, n_parallel, committed) {
                Ok(spec) => break spec,
                Err(e) => {
                    if !self.evict_one(&info.name).await {
                        return Err(e);
                    }
                }
            }
        };

        let (spec, need) = spec;
        let backend = self.make_backend(info);
        self.loading.write().await.insert(info.name.clone());
        tracing::info!(
            model = %info.name,
            ctx = spec.params.n_ctx,
            parallel = spec.n_parallel,
            embeddings,
            gpu_layers = %spec.params.n_gpu_layers,
            "loading model"
        );
        let result = backend.load(&spec).await;
        self.loading.write().await.remove(&info.name);
        result?;

        let entry = Arc::new(LoadedModel {
            info: info.clone(),
            backend,
            params: spec.params.clone(),
            n_parallel: spec.n_parallel,
            embeddings,
            need_bytes: need,
            loaded_at: chrono::Utc::now(),
            last_used: AtomicU64::new(now_millis()),
            active: AtomicUsize::new(0),
            keep_alive: AtomicI64::new(keep_alive.unwrap_or(self.config.keep_alive_secs)),
        });
        let guard = {
            let mut loaded = self.loaded.write().await;
            loaded.insert(info.name.clone(), entry.clone());
            ActiveGuard::new(entry)
        };
        tracing::info!(model = %info.name, "model ready");
        Ok(guard)
    }

    async fn evict_one(&self, keep: &str) -> bool {
        let victim = {
            let mut loaded = self.loaded.write().await;
            let name = loaded
                .iter()
                .filter(|(n, m)| n.as_str() != keep && m.active_requests() == 0)
                .min_by_key(|(_, m)| m.last_used.load(Ordering::Relaxed))
                .map(|(n, _)| n.clone());
            name.and_then(|n| loaded.remove(&n))
        };
        match victim {
            Some(m) => {
                tracing::info!(model = %m.info.name, "evicting least recently used model");
                m.backend.unload().await.ok();
                true
            }
            None => false,
        }
    }

    fn make_backend(&self, info: &ModelInfo) -> Arc<dyn Backend> {
        match info.model_type {
            ModelType::Gguf => Arc::new(crate::backends::llamaserver::LlamaServerBackend::new(
                self.config.llama_server.clone(),
                self.config.log_dir(),
            )),
            ModelType::Safetensors => Arc::new(crate::backends::candle::CandleBackend::new()),
        }
    }

    fn plan_load(
        &self,
        info: &ModelInfo,
        wanted: &ModelParams,
        embeddings: bool,
        n_parallel: usize,
        committed: u64,
    ) -> Result<(LoadSpec, u64)> {
        let mut params = wanted.clone();
        let details = info.details.clone().unwrap_or_default();
        let gpu = params.n_gpu_layers.uses_gpu();
        let budget = crate::config::memory_budget(self.config.max_memory_fraction, gpu);
        let free = budget
            .known()
            .then(|| budget.budget_bytes.saturating_sub(committed));

        let mmproj_bytes = if embeddings {
            0
        } else {
            info.mmproj
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len() + VISION_OVERHEAD)
                .unwrap_or(0)
        };
        let weights = info.size_bytes + mmproj_bytes;
        let overhead = RUNTIME_OVERHEAD;
        let kv_scale = (gguf::cache_type_bytes(&params.cache_type_k)
            + gguf::cache_type_bytes(&params.cache_type_v))
            / 4.0;
        let kv_per_token = (details.kv_bytes_per_token_f16 as f64 * kv_scale).ceil() as u64;
        let trained = details
            .context_length
            .map(|c| c.min(u32::MAX as u64) as u32)
            .filter(|c| *c > 0)
            .unwrap_or(AUTO_CTX_MIN);
        let upper = trained.min(self.config.auto_ctx_max).max(256);

        if params.n_ctx == 0 {
            params.n_ctx = if embeddings || info.model_type == ModelType::Safetensors {
                upper
            } else if let (Some(free), true) = (free, kv_per_token > 0) {
                let room = free.saturating_sub(weights + overhead);
                let per_slot = room / (kv_per_token * n_parallel as u64);
                let lower = AUTO_CTX_MIN.min(upper);
                let fitted = (per_slot.min(u32::MAX as u64) as u32) / 1024 * 1024;
                fitted.clamp(lower, upper)
            } else if kv_per_token == 0 {
                upper
            } else {
                upper.min(UNKNOWN_MEMORY_CTX)
            };
        }

        let need = weights + overhead + kv_per_token * params.n_ctx as u64 * n_parallel as u64;
        if let Some(free) = free {
            if need > free {
                let mut message = budget.describe_failure(need);
                if committed > 0 {
                    message.push_str(&format!(
                        " Other loaded models hold about {}.",
                        crate::config::humansize(committed)
                    ));
                }
                return Err(Ml5Error::Backend(message));
            }
        }

        Ok((
            LoadSpec {
                model: info.clone(),
                params,
                n_parallel,
                embeddings,
            },
            need,
        ))
    }

    async fn reap(&self) {
        let candidates: Vec<Arc<LoadedModel>> = self
            .loaded
            .read()
            .await
            .values()
            .filter(|m| m.active_requests() == 0)
            .cloned()
            .collect();
        let mut doomed = Vec::new();
        for m in candidates {
            if !m.backend.is_alive().await {
                tracing::warn!(model = %m.info.name, "model runtime exited; releasing it");
                doomed.push(m);
            } else if m.expired() {
                tracing::info!(model = %m.info.name, "keep-alive expired; unloading");
                doomed.push(m);
            }
        }
        if doomed.is_empty() {
            return;
        }
        let Ok(_lifecycle) = self.lifecycle.try_lock() else { return };
        let removed: Vec<Arc<LoadedModel>> = {
            let mut loaded = self.loaded.write().await;
            let names: Vec<String> = doomed
                .iter()
                .filter(|m| {
                    m.active_requests() == 0
                        && loaded
                            .get(&m.info.name)
                            .is_some_and(|cur| Arc::ptr_eq(cur, m))
                })
                .map(|m| m.info.name.clone())
                .collect();
            names.iter().filter_map(|n| loaded.remove(n)).collect()
        };
        for m in removed {
            m.backend.unload().await.ok();
        }
    }
}

async fn reaper(engine: Weak<Engine>) {
    let mut tick = tokio::time::interval(REAPER_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        let Some(engine) = engine.upgrade() else { return };
        engine.reap().await;
    }
}

fn guarded(mut stream: TokenStream, guard: ActiveGuard) -> TokenStream {
    Box::pin(async_stream::stream! {
        let _guard = guard;
        while let Some(chunk) = stream.next().await {
            yield chunk;
        }
    })
}

fn modified(meta: &std::fs::Metadata) -> Option<chrono::DateTime<chrono::Utc>> {
    meta.modified().ok().map(chrono::DateTime::from)
}

fn scan_store(
    root: &Path,
	old_cache: &GgufCache,
) -> Result<(HashMap<String, ModelInfo>, GgufCache)> {
    let mut models = HashMap::new();
    let mut cache = GgufCache::new();
    let mut root_ggufs = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(root)?
        .flatten()
        .map(|e| e.path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_file() {
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                root_ggufs.push(path);
            }
            continue;
        }
        if !path.is_dir() {
            continue;
        }
        let dir_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        if dir_name.starts_with('.') {
            continue;
        }
        let mut children: Vec<PathBuf> = std::fs::read_dir(&path)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        children.sort();
        let has_config = path.join("config.json").is_file();
        let has_st = children
            .iter()
            .any(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"));
        if has_config && has_st {
            if let Some(info) = safetensors_model(&path, &dir_name, &children) {
                models.insert(info.name.clone(), info);
            }
            continue;
        }
        let ggufs: Vec<PathBuf> = children
            .into_iter()
            .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf"))
            .collect();
        if !ggufs.is_empty() {
            for info in gguf_group(&path, &ggufs, Some(&dir_name), old_cache, &mut cache) {
                models.insert(info.name.clone(), info);
            }
        }
    }
    for info in gguf_group(root, &root_ggufs, None, old_cache, &mut cache) {
        models.insert(info.name.clone(), info);
    }
    Ok((models, cache))
}

fn safetensors_model(path: &Path, name: &str, children: &[PathBuf]) -> Option<ModelInfo> {
    let meta_path = path.join("model.meta.json");
    let sidecar = ModelMeta::load(&meta_path);
    let size = children
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .filter(|m| m.is_file())
        .map(|m| m.len())
        .sum();
    let dir_meta = std::fs::metadata(path).ok()?;
    let context_length = std::fs::read_to_string(path.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v["max_position_embeddings"].as_u64());
    Some(ModelInfo {
        name: name.to_string(),
        path: path.to_path_buf(),
        size_bytes: size,
        capabilities: vec![Capability::Chat, Capability::Generate],
        digest: None,
        modified_at: modified(&dir_meta),
        friendly_name: sidecar.friendly_name,
        model_type: ModelType::Safetensors,
        mmproj: None,
        files: Vec::new(),
        meta_path: Some(meta_path),
        details: Some(ModelDetails {
            architecture: "llama".into(),
            parameter_size: String::new(),
            quantization: "safetensors".into(),
            context_length,
            ..Default::default()
        }),
    })
}

fn read_gguf_cached(
    path: &Path,
    meta: &std::fs::Metadata,
    old: &GgufCache,
    cache: &mut GgufCache,
) -> Option<GgufInfo> {
    let mtime = meta.modified().ok();
    if let Some((len, time, info)) = old.get(path) {
        if *len == meta.len() && *time == mtime {
            cache.insert(path.to_path_buf(), (*len, *time, info.clone()));
            return info.clone();
        }
    }
    let info = match gguf::read_info(path) {
        Ok(info) => Some(info),
        Err(e) => {
            tracing::warn!(path = %path.display(), %e, "could not read GGUF metadata");
            None
        }
    };
    cache.insert(path.to_path_buf(), (meta.len(), mtime, info.clone()));
    info
}

fn gguf_group(
    dir: &Path,
    files: &[PathBuf],
    subdir: Option<&str>,
    old: &GgufCache,
    cache: &mut GgufCache,
) -> Vec<ModelInfo> {
    let file_name = |p: &PathBuf| -> String {
        p.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string()
    };
    let mmprojs: Vec<&PathBuf> = files
        .iter()
        .filter(|p| model::is_mmproj_file(&file_name(p)))
        .collect();
    let mut out = Vec::new();
    for first in files.iter().filter(|p| !model::is_mmproj_file(&file_name(p))) {
        let fname = file_name(first);
        let parts: Vec<PathBuf> = match model::split_part(&fname) {
            Some((_, idx, _)) if idx != 1 => continue,
            Some((base, _, total)) => {
                let parts: Vec<PathBuf> = (1..=total)
                    .map(|i| dir.join(model::split_part_name(&base, i, total)))
                    .collect();
                if let Some(missing) = parts.iter().find(|p| !p.is_file()) {
                    tracing::warn!(missing = %missing.display(), "skipping incomplete split GGUF model");
                    continue;
                }
                parts
            }
            None => vec![first.clone()],
        };
        let Ok(meta) = std::fs::metadata(first) else { continue };
        let stem = model::gguf_model_name(&fname);
        let name = match subdir {
            Some(d) => format!("{d}/{stem}"),
            None => stem.clone(),
        };
        let meta_path = model::gguf_meta_path(first);
        let mut sidecar = ModelMeta::load(&meta_path);
        let mut mmproj = sidecar
            .mmproj_file
            .as_ref()
            .map(|f| dir.join(f))
            .filter(|p| p.is_file());
        if mmproj.is_none() {
            let prefix = stem.split_once("__").map(|(p, _)| format!("{p}__"));
            let paired = prefix
                .as_ref()
                .and_then(|pre| mmprojs.iter().find(|m| file_name(m).starts_with(pre.as_str())))
                .or_else(|| (subdir.is_some() && mmprojs.len() == 1).then(|| &mmprojs[0]))
                .map(|p| (*p).clone());
            if let Some(found) = paired {
                sidecar.mmproj_file = Some(file_name(&found));
                if let Err(e) = sidecar.save(&meta_path) {
                    tracing::debug!(%e, "could not record mmproj pairing");
                }
                mmproj = Some(found);
            }
        }
        let size: u64 = parts
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        let ginfo = read_gguf_cached(first, &meta, old, cache);
        let mut capabilities = Vec::new();
        let details = match &ginfo {
            Some(g) if g.is_embedding_model() => {
                capabilities.push(Capability::Embed);
                Some(details_from(g, size))
            }
            Some(g) => {
                capabilities.extend([Capability::Chat, Capability::Generate]);
                if g.supports_tools() {
                    capabilities.push(Capability::Tools);
                }
                if g.supports_thinking() {
                    capabilities.push(Capability::Thinking);
                }
                if mmproj.is_some() {
                    capabilities.push(Capability::Vision);
                }
                Some(details_from(g, size))
            }
            None => {
                capabilities.extend([Capability::Chat, Capability::Generate]);
                if mmproj.is_some() {
                    capabilities.push(Capability::Vision);
                }
                None
            }
        };
        out.push(ModelInfo {
            name,
            path: first.clone(),
            size_bytes: size,
            capabilities,
            digest: None,
            modified_at: modified(&meta),
            friendly_name: sidecar.friendly_name.clone(),
            model_type: ModelType::Gguf,
            mmproj,
            files: parts,
            meta_path: Some(meta_path),
            details,
        });
    }
    out
}

fn details_from(g: &GgufInfo, size: u64) -> ModelDetails {
    ModelDetails {
        architecture: g.architecture.clone(),
        parameter_size: g.parameter_label(size),
        quantization: g.quantization().to_string(),
        context_length: g.context_length,
        block_count: g.block_count,
        embedding_length: g.embedding_length,
        kv_bytes_per_token_f16: g.kv_bytes_per_token(2.0),
    }
}
