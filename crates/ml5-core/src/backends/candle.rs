use crate::backend::{Backend, TokenStream};
use crate::config::ModelParams;
use crate::error::{Ml5Error, Result};
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use tokio::sync::{mpsc, Mutex};

#[cfg(feature = "safetensors")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "safetensors")]
use candle_nn::VarBuilder;
#[cfg(feature = "safetensors")]
use candle_transformers::generation::LogitsProcessor;
#[cfg(feature = "safetensors")]
use candle_transformers::models::llama::{Cache, Config, LlamaConfig, Llama};

#[cfg(feature = "safetensors")]
static DEVICE: std::sync::OnceLock<Device> = std::sync::OnceLock::new();

#[cfg(feature = "safetensors")]
fn device() -> Result<&'static Device> {
    if let Some(d) = DEVICE.get() {
        return Ok(d);
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            let _ = DEVICE.set(d);
            return Ok(DEVICE.get().unwrap());
        }
    }
    let d = Device::Cpu;
    let _ = DEVICE.set(d);
    Ok(DEVICE.get().unwrap())
}

#[cfg(not(feature = "safetensors"))]
fn device() -> Result<()> {
    Err(Ml5Error::Backend(
        "safetensors support not compiled in (build with --features safetensors)".into(),
    ))
}

enum Job {
    Generate(GenerateJob),
    Unload,
}

struct GenerateJob {
    prompt: String,
    params: SamplingParams,
    out: mpsc::UnboundedSender<Result<StreamChunk>>,
}

pub struct CandleBackend {
    worker: Mutex<Option<WorkerHandle>>,
    path: Mutex<Option<PathBuf>>,
    params: Mutex<ModelParams>,
}

struct WorkerHandle {
    tx: std_mpsc::Sender<Job>,
}

impl CandleBackend {
    pub fn new(params: ModelParams) -> Self {
        Self {
            worker: Mutex::new(None),
            path: Mutex::new(None),
            params: Mutex::new(params),
        }
    }

    async fn worker(&self) -> Result<tokio::sync::MutexGuard<'_, Option<WorkerHandle>>> {
        let guard = self.worker.lock().await;
        if guard.is_none() {
            return Err(Ml5Error::Backend("backend not loaded".into()));
        }
        Ok(guard)
    }
}

#[async_trait]
impl Backend for CandleBackend {
    fn name(&self) -> &str {
        "candle"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Chat, Capability::Generate]
    }

    async fn load(&self, model_path: &Path, params: &ModelParams) -> Result<()> {
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = (model_path, params);
            return Err(Ml5Error::Backend(
                "safetensors support not compiled in (build with --features safetensors)".into(),
            ));
        }

        #[cfg(feature = "safetensors")]
        {
            let path = model_path.to_path_buf();
            let params = params.clone();
            let handle = tokio::task::spawn_blocking(move || {
                let (tx, rx) = std_mpsc::channel::<Job>();
                std::thread::Builder::new()
                    .name("ml5-candle".into())
                    .spawn(move || candle_worker(path, params, rx))
                    .map_err(|e| Ml5Error::Backend(format!("failed to spawn candle worker: {e}")))?;
                Ok::<WorkerHandle, Ml5Error>(WorkerHandle { tx })
            })
            .await
            .map_err(|e| Ml5Error::Backend(format!("load join failed: {e}")))??;

            *self.worker.lock().await = Some(handle);
            *self.path.lock().await = Some(model_path.to_path_buf());
            Ok(())
        }
    }

    async fn unload(&self) -> Result<()> {
        let mut guard = self.worker.lock().await;
        if let Some(w) = guard.take() {
            let _ = w.tx.send(Job::Unload);
        }
        *self.path.lock().await = None;
        Ok(())
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream> {
        let prompt = req
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
            + "\nassistant:";
        self.generate(GenerateRequest {
            model: req.model,
            prompt,
            params: req.params,
            overrides: req.overrides,
            stream: req.stream,
            cache: req.cache,
        })
        .await
    }

    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream> {
        let guard = self.worker().await?;
        let worker = guard.as_ref().expect("checked above");
        let (tx, rx) = mpsc::unbounded_channel();
        worker
            .tx
            .send(Job::Generate(GenerateJob {
                prompt: req.prompt,
                params: req.params,
                out: tx,
            }))
            .map_err(|_| Ml5Error::Backend("candle worker died".into()))?;
        drop(guard);
        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).boxed())
    }

    async fn embed(&self, _req: EmbedRequest) -> Result<EmbedResponse> {
        Err(Ml5Error::Backend(
            "embed not supported on candle backend yet".into(),
        ))
    }
}

#[cfg(feature = "safetensors")]
fn candle_worker(path: PathBuf, params: ModelParams, rx: std_mpsc::Receiver<Job>) {
    if let Err(e) = candle_worker_inner(path, params, rx) {
        tracing::error!("candle worker failed: {e}");
    }
}

#[cfg(feature = "safetensors")]
fn candle_worker_inner(path: PathBuf, _params: ModelParams, rx: std_mpsc::Receiver<Job>) -> Result<()> {
    let dev = device()?.clone();

    let config_path = path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| Ml5Error::Backend(format!("no config.json in {}: {e}", path.display())))?;
    let llama_config: LlamaConfig = serde_json::from_str(&config_str)
        .map_err(|e| Ml5Error::Backend(format!("bad config.json: {e}")))?;
    let config = llama_config.into_config(false);

    let tokenizer_path = path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Ml5Error::Backend(format!("no tokenizer.json in {}: {e}", path.display())))?;

    let mut safetensor_files: Vec<PathBuf> = std::fs::read_dir(&path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    safetensor_files.sort();
    if safetensor_files.is_empty() {
        return Err(Ml5Error::Backend(format!(
            "no .safetensors files in {}",
            path.display()
        )));
    }

    let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &dev)
            .map_err(|e| Ml5Error::Backend(format!("failed to load safetensors: {e}")))?
    };
    let model = Llama::load(vb, &config)
        .map_err(|e| Ml5Error::Backend(format!("failed to build model: {e}")))?;

    while let Ok(job) = rx.recv() {
        match job {
            Job::Unload => break,
            Job::Generate(j) => {
                if let Err(e) = run_generate(&model, &tokenizer, &config, &dev, j) {
                    tracing::error!("candle generate failed: {e}");
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "safetensors")]
fn run_generate(
    model: &Llama,
    tokenizer: &tokenizers::Tokenizer,
    config: &Config,
    dev: &Device,
    job: GenerateJob,
) -> Result<()> {
    let enc = tokenizer
        .encode(job.prompt.as_str(), true)
        .map_err(|e| Ml5Error::Backend(format!("tokenization failed: {e}")))?;
    let tokens: Vec<u32> = enc.get_ids().to_vec();
    if tokens.is_empty() {
        return Err(Ml5Error::InvalidRequest("empty prompt".into()));
    }

    let mut cache = Cache::new(true, DType::F32, config, dev)
        .map_err(|e| Ml5Error::Backend(format!("cache init failed: {e}")))?;

    let mut sampler = LogitsProcessor::new(
        job.params.seed.unwrap_or(0x4C4C) as u64,
        job.params.temperature.map(|t| t as f64),
        job.params.top_p.map(|t| t as f64),
    );

    let max_tokens = job.params.max_tokens.unwrap_or(512) as usize;
    let prompt_len = tokens.len();
    let mut completion_tokens: u32 = 0;

    let _ = job.out.send(Ok(StreamChunk {
        text: String::new(),
        done: false,
        usage: None,
        progress: Some(InferenceProgress {
            stage: "processing_prompt".into(),
            completed: 0,
            total: prompt_len,
        }),
        finish_reason: None,
    }));

    let input = Tensor::new(tokens.as_slice(), dev)
        .map_err(|e| Ml5Error::Backend(format!("tensor failed: {e}")))?
        .unsqueeze(0)
        .map_err(|e| Ml5Error::Backend(format!("unsqueeze failed: {e}")))?;
    let logits = model
        .forward(&input, 0, &mut cache)
        .map_err(|e| Ml5Error::Backend(format!("forward failed: {e}")))?;
    let logits = logits
        .squeeze(0)
        .map_err(|e| Ml5Error::Backend(format!("squeeze failed: {e}")))?;

    let mut next_token = sampler
        .sample(&logits)
        .map_err(|e| Ml5Error::Backend(format!("sample failed: {e}")))?;

    let eos_ids: Vec<u32> = match &config.eos_token_id {
        Some(candle_transformers::models::llama::LlamaEosToks::Single(t)) => vec![*t],
        Some(candle_transformers::models::llama::LlamaEosToks::Multiple(ts)) => ts.clone(),
        None => vec![2],
    };
    let mut generated = vec![next_token];
    completion_tokens += 1;

    let piece = tokenizer
        .decode(&[next_token], true)
        .map_err(|e| Ml5Error::Backend(format!("decode failed: {e}")))?;
    if job
        .out
        .send(Ok(StreamChunk {
            text: piece,
            done: false,
            usage: None,
            progress: None,
            finish_reason: None,
        }))
        .is_err()
    {
        return Ok(());
    }

    for _ in 1..max_tokens {
        if eos_ids.contains(&next_token) {
            break;
        }
        let input = Tensor::new(&[next_token], dev)
            .map_err(|e| Ml5Error::Backend(format!("tensor failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| Ml5Error::Backend(format!("unsqueeze failed: {e}")))?;
        let logits = model
            .forward(&input, prompt_len + generated.len() - 1, &mut cache)
            .map_err(|e| Ml5Error::Backend(format!("forward failed: {e}")))?;
        let logits = logits
            .squeeze(0)
            .map_err(|e| Ml5Error::Backend(format!("squeeze failed: {e}")))?;
        next_token = sampler
            .sample(&logits)
            .map_err(|e| Ml5Error::Backend(format!("sample failed: {e}")))?;
        generated.push(next_token);
        completion_tokens += 1;

        let piece = tokenizer
            .decode(&[next_token], true)
            .map_err(|e| Ml5Error::Backend(format!("decode failed: {e}")))?;
        if job
            .out
            .send(Ok(StreamChunk {
                text: piece,
                done: false,
                usage: None,
                progress: None,
                finish_reason: None,
            }))
            .is_err()
        {
            return Ok(());
        }
    }

    let _ = job.out.send(Ok(StreamChunk {
        text: String::new(),
        done: true,
        usage: Some(Usage {
            prompt_tokens: prompt_len as u32,
            completion_tokens,
            prompt_ms: 0,
            generation_ms: 0,
        }),
        progress: None,
        finish_reason: Some(if generated.last().is_some_and(|t| eos_ids.contains(t)) {
            "stop".into()
        } else {
            "length".into()
        }),
    }));
    Ok(())
}

#[cfg(not(feature = "safetensors"))]
fn candle_worker(_path: PathBuf, _params: ModelParams, _rx: std_mpsc::Receiver<Job>) {}
