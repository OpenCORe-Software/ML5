use crate::backend::Backend;
use crate::config::Config;
use crate::error::{Ml5Error, Result};

fn now_millis() -> u64 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_millis() as u64
}
use crate::model::ModelInfo;
use crate::types::*;
use futures::Stream;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, Mutex, RwLock};

pub struct LoadedModel {
    pub info: ModelInfo,
    pub backend: Arc<dyn Backend>,
    pub last_used: std::sync::atomic::AtomicU64,
    pub gpu_layers: i32,
}

pub struct Engine {
    pub config: Config,
    models: RwLock<HashMap<String, ModelInfo>>,
    loaded: RwLock<HashMap<String, Arc<LoadedModel>>>,
    pulls: RwLock<HashMap<String, broadcast::Sender<crate::pull::PullEvent>>>,
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
            pulls: RwLock::new(HashMap::new()),
            lifecycle: Mutex::new(()),
            started: Instant::now(),
        });
        Ok(engine)
    }

    pub async fn register_pull(
        &self,
        key: &str,
    ) -> Result<broadcast::Sender<crate::pull::PullEvent>> {
        let mut pulls = self.pulls.write().await;
        if pulls.contains_key(key) {
            return Err(Ml5Error::InvalidRequest(
                "This model is already being downloaded; wait for that pull to finish.".into(),
            ));
        }
        let (tx, _) = broadcast::channel(128);
        pulls.insert(key.to_string(), tx.clone());
        Ok(tx)
    }

    pub async fn finish_pull(&self, key: &str) {
        self.pulls.write().await.remove(key);
    }

    pub async fn subscribe_pull(
        &self,
        key: &str,
    ) -> Option<broadcast::Receiver<crate::pull::PullEvent>> {
        self.pulls.read().await.get(key).map(|tx| tx.subscribe())
    }

    pub async fn scan_models(&self) -> Result<()> {
        let mut models = HashMap::new();
        let dir = self.config.models_dir.clone();
        let mut entries = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                let meta = entry.metadata().await?;
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let meta_path = path.with_extension("meta.json");
                let sidecar: crate::model::ModelMeta = if meta_path.exists() {
                    tokio::fs::read_to_string(&meta_path)
                        .await
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default()
                } else {
                    crate::model::ModelMeta::default()
                };

                models.insert(
                    name.clone(),
                    ModelInfo {
                        name,
                        path: path.clone(),
                        size_bytes: meta.len(),
                        capabilities: vec![
                            Capability::Chat,
                            Capability::Generate,
                            Capability::Embed,
                        ],
                        digest: None,
                        modified_at: meta.modified().ok().map(chrono::DateTime::from),
                        friendly_name: sidecar.friendly_name,
                        model_type: crate::model::ModelType::Gguf,
                    },
                );
            } else if path.is_dir() && path.join("config.json").exists() {
                let has_st = std::fs::read_dir(&path)
                    .map(|rd| {
                        rd.filter_map(|e| e.ok())
                            .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
                    })
                    .unwrap_or(false);
                if !has_st {
                    continue;
                }
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let meta_path = path.join("model.meta.json");
                let sidecar: crate::model::ModelMeta = if meta_path.exists() {
                    tokio::fs::read_to_string(&meta_path)
                        .await
                        .ok()
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default()
                } else {
                    crate::model::ModelMeta::default()
                };

                let mut size = 0u64;
                if let Ok(rd) = std::fs::read_dir(&path) {
                    for e in rd.filter_map(|e| e.ok()) {
                        if let Ok(m) = e.metadata() {
                            size += m.len();
                        }
                    }
                }

                models.insert(
                    name.clone(),
                    ModelInfo {
                        name,
                        path: path.clone(),
                        size_bytes: size,
                        capabilities: vec![Capability::Chat, Capability::Generate],
                        digest: None,
                        modified_at: entry.metadata().await.ok().and_then(|m| m.modified().ok()).map(chrono::DateTime::from),
                        friendly_name: sidecar.friendly_name,
                        model_type: crate::model::ModelType::Safetensors,
                    },
                );
            }
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
        if let Some(info) = models.get(name) {
            return Ok(info.clone());
        }
        let normalized: String = name
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();
        if let Some(info) = models.values().find(|m| {
            let m_norm: String = m
                .name
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '-' })
                .collect();
            m_norm == normalized
        }) {
            return Ok(info.clone());
        }
        if let Some(info) = models.values().find(|m| {
            m.friendly_name
                .as_deref()
                .map(|f| {
                    let f_norm: String = f
                        .chars()
                        .map(|c| if c.is_alphanumeric() { c } else { '-' })
                        .collect();
                    f_norm == normalized
                })
                .unwrap_or(false)
        }) {
            return Ok(info.clone());
        }
        Err(Ml5Error::ModelNotFound(format!("{name}. Run `ml5 list` for installed names or `ml5 pull hf:owner/repo` to download one.")))
    }

    pub async fn rename_model(&self, name: &str, new_friendly_name: &str) -> Result<()> {
        let info = self.get_model(name).await?;
        let meta_path = info.path.with_extension("meta.json");

        let mut meta: crate::model::ModelMeta = if meta_path.exists() {
            tokio::fs::read_to_string(&meta_path)
                .await
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            crate::model::ModelMeta::default()
        };

        meta.friendly_name = Some(new_friendly_name.to_string());
        let json = serde_json::to_string_pretty(&meta)?;
        tokio::fs::write(&meta_path, json).await?;

        if let Some(m) = self.models.write().await.get_mut(&info.name) {
            m.friendly_name = Some(new_friendly_name.to_string());
        }

        Ok(())
    }

    pub async fn delete_model(&self, name: &str) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        let info = self.get_model(name).await?;
        self.remove_loaded(&info.name).await?;
        tokio::fs::remove_file(&info.path).await?;
        self.models.write().await.remove(&info.name);
        Ok(())
    }

    pub async fn unload_model(&self, name: &str) -> Result<()> {
        let _guard = self.lifecycle.lock().await;
        let info = self.get_model(name).await?;
        self.remove_loaded(&info.name).await
    }

    async fn remove_loaded(&self, name: &str) -> Result<()> {
        let mut loaded = self.loaded.write().await;
        if let Some(m) = loaded.get(name) {
            if Arc::strong_count(m) > 1 {
                return Err(Ml5Error::InvalidRequest(format!(
                    "Model '{name}' is busy; cancel or finish its requests first."
                )));
            }
            m.backend.unload().await?;
            loaded.remove(name);
            tracing::info!(model = %name, "model unloaded");
        }
        Ok(())
    }

    pub async fn status(&self) -> EngineStatus {
        let models = self.models.read().await;
        let loaded = self.loaded.read().await;
        EngineStatus {
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_count: models.len(),
            loaded_models: loaded.keys().cloned().collect(),
            uptime_secs: self.started.elapsed().as_secs(),
            models_dir: self.config.models_dir.display().to_string(),
            defaults: self.config.model.clone(),
            dynamic_backends: cfg!(feature = "gpu"),
            safetensors: cfg!(feature = "safetensors"),
        }
    }

    pub async fn is_loaded(&self, name: &str) -> bool {
        match self.get_model(name).await {
            Ok(info) => self.loaded.read().await.contains_key(&info.name),
            Err(_) => false,
        }
    }

    pub async fn chat(
        &self,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        req.params.validate()?;
        req.overrides.validate()?;
        if req.messages.is_empty() {
            return Err(Ml5Error::InvalidRequest(
                "messages must not be empty".into(),
            ));
        }
        let model = self.ensure_loaded(&req.model, &req.overrides).await?;
        let mut stream = model.backend.chat(req).await?;
        Ok(Box::pin(async_stream::stream! {
            while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
                model.last_used.store(now_millis(), std::sync::atomic::Ordering::Relaxed);
                yield chunk;
            }
            model.last_used.store(now_millis(), std::sync::atomic::Ordering::Relaxed);
        }))
    }

    pub async fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        req.params.validate()?;
        req.overrides.validate()?;
        if req.prompt.is_empty() {
            return Err(Ml5Error::InvalidRequest("prompt must not be empty".into()));
        }
        let model = self.ensure_loaded(&req.model, &req.overrides).await?;
        let mut stream = model.backend.generate(req).await?;
        Ok(Box::pin(async_stream::stream! {
            while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
                model.last_used.store(now_millis(), std::sync::atomic::Ordering::Relaxed);
                yield chunk;
            }
            model.last_used.store(now_millis(), std::sync::atomic::Ordering::Relaxed);
        }))
    }

    pub async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
        let model = self
            .ensure_loaded(&req.model, &RequestOverrides::default())
            .await?;
        model.backend.embed(req).await
    }

    async fn ensure_loaded(
        &self,
        name: &str,
        overrides: &RequestOverrides,
    ) -> Result<Arc<LoadedModel>> {
        let _guard = self.lifecycle.lock().await;
        let info = self.get_model(name).await?;
        let mut loaded = self.loaded.write().await;
        if let Some(m) = loaded.get(&info.name) {
            if overrides.n_gpu_layers.is_some_and(|g| g != m.gpu_layers) {
                return Err(Ml5Error::InvalidRequest(format!("GPU layers differ from the loaded model. Run `ml5 unload {}` before changing --gpu-layers.", info.name)));
            }
            m.last_used.store(now_millis(), std::sync::atomic::Ordering::Relaxed);
            return Ok(m.clone());
        }
        if loaded.len() >= self.config.max_loaded_models {
            if let Some(evict_name) = loaded
                .iter()
                .filter(|(_, m)| Arc::strong_count(m) == 1)
                .min_by_key(|(_, m)| m.last_used.load(std::sync::atomic::Ordering::Relaxed))
                .map(|(n, _)| n.clone())
            {
                tracing::info!(model = %evict_name, "evicting LRU model");
                if let Some(m) = loaded.remove(&evict_name) {
                    let _ = m.backend.unload().await;
                }
            } else {
                return Err(Ml5Error::InvalidRequest(
                    "All loaded models are busy; retry when a request finishes.".into(),
                ));
            }
        }
        drop(loaded);

        let mut params = self.config.model.clone();
        if let Some(g) = overrides.n_gpu_layers {
            params.n_gpu_layers = g;
        }

        let n_parallel = if self.config.parallel {
            self.config.n_parallel.max(1)
        } else {
            1
        };
        let backend: Arc<dyn crate::backend::Backend> = match info.model_type {
            crate::model::ModelType::Gguf => Arc::new(crate::backends::llama::LlamaCppBackend::new(
                params.clone(),
                self.config.max_memory_fraction,
                n_parallel,
                self.config.manual_kv,
                self.config.auto_kv,
            )),
            crate::model::ModelType::Safetensors => Arc::new(crate::backends::candle::CandleBackend::new(
                params.clone(),
            )),
        };
        backend.load(&info.path, &params).await?;

        let name = info.name.clone();
        let entry = Arc::new(LoadedModel {
            info,
            backend,
            last_used: std::sync::atomic::AtomicU64::new(now_millis()),
            gpu_layers: params.n_gpu_layers,
        });
        self.loaded.write().await.insert(name, entry.clone());
        Ok(entry)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineStatus {
    pub version: String,
    pub model_count: usize,
    pub loaded_models: Vec<String>,
    pub uptime_secs: u64,
    pub models_dir: String,
    pub defaults: crate::config::ModelParams,
    pub dynamic_backends: bool,
    pub safetensors: bool,
}
