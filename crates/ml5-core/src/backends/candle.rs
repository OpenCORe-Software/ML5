use crate::backend::{Backend, LoadSpec, TokenStream};
use crate::error::{Ml5Error, Result};
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::mpsc as std_mpsc;
use tokio::sync::{mpsc, Mutex};

#[cfg(feature = "safetensors")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "safetensors")]
use candle_nn::VarBuilder;
#[cfg(feature = "safetensors")]
use candle_transformers::generation::LogitsProcessor;
#[cfg(feature = "safetensors")]
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks};
#[cfg(feature = "safetensors")]
use std::path::{Path, PathBuf};

#[cfg_attr(not(feature = "safetensors"), allow(dead_code))]
enum Job {
    Generate(GenerateJob),
    Unload,
}

#[cfg_attr(not(feature = "safetensors"), allow(dead_code))]
struct GenerateJob {
    prompt: String,
    templated: bool,
    params: SamplingParams,
    out: mpsc::UnboundedSender<Result<StreamChunk>>,
}

struct WorkerHandle {
    tx: std_mpsc::Sender<Job>,
    template: Option<ChatTemplate>,
    #[cfg(feature = "safetensors")]
    tokenizer: tokenizers::Tokenizer,
}

#[derive(Clone)]
#[cfg_attr(not(feature = "safetensors"), allow(dead_code))]
struct ChatTemplate {
    source: String,
    bos: String,
    eos: String,
}

pub struct CandleBackend {
    worker: Mutex<Option<WorkerHandle>>,
}

impl CandleBackend {
    pub fn new() -> Self {
        Self {
            worker: Mutex::new(None),
        }
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "safetensors")]
fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

#[cfg(feature = "safetensors")]
fn token_text(v: &serde_json::Value) -> String {
    v.as_str()
        .map(str::to_string)
        .or_else(|| v["content"].as_str().map(str::to_string))
        .unwrap_or_default()
}

#[cfg(feature = "safetensors")]
fn load_template(dir: &Path) -> Option<ChatTemplate> {
    let raw = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let source = match &cfg["chat_template"] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(items) => items
            .iter()
            .find(|t| t["name"].as_str() == Some("default"))
            .or_else(|| items.first())
            .and_then(|t| t["template"].as_str())
            .map(str::to_string)?,
        _ => std::fs::read_to_string(dir.join("chat_template.jinja")).ok()?,
    };
    Some(ChatTemplate {
        source,
        bos: token_text(&cfg["bos_token"]),
        eos: token_text(&cfg["eos_token"]),
    })
}

#[cfg(feature = "safetensors")]
fn render_template(t: &ChatTemplate, messages: &[Message]) -> Result<String> {
    let mut env = minijinja::Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        },
    );
    env.add_function("strftime_now", |fmt: String| -> String {
        chrono::Local::now().format(&fmt).to_string()
    });
    env.add_template("chat", &t.source)
        .map_err(|e| Ml5Error::Backend(format!("invalid chat template: {e}")))?;
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
        .collect();
    env.get_template("chat")
        .and_then(|tpl| {
            tpl.render(minijinja::context! {
                messages => msgs,
                add_generation_prompt => true,
                bos_token => t.bos.clone(),
                eos_token => t.eos.clone(),
            })
        })
        .map_err(|e| Ml5Error::InvalidRequest(format!("chat template failed: {e}")))
}

#[cfg(not(feature = "safetensors"))]
fn render_template(_t: &ChatTemplate, _messages: &[Message]) -> Result<String> {
    Err(Ml5Error::Backend(
        "safetensors support not compiled in (build with --features safetensors)".into(),
    ))
}

#[async_trait]
impl Backend for CandleBackend {
    fn name(&self) -> &str {
        "candle"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Chat, Capability::Generate]
    }

    async fn load(&self, spec: &LoadSpec) -> Result<()> {
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = spec;
            Err(Ml5Error::Backend(
                "safetensors support not compiled in (build with --features safetensors)".into(),
            ))
        }

        #[cfg(feature = "safetensors")]
        {
            self.unload().await?;
            let path = spec.model.path.clone();
            let template = load_template(&path);
            let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
                .map_err(|e| Ml5Error::Backend(format!("no usable tokenizer.json in {}: {e}", path.display())))?;
            let (tx, rx) = std_mpsc::channel::<Job>();
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<()>>();
            std::thread::Builder::new()
                .name("ml5-candle".into())
                .spawn(move || candle_worker(path, rx, ready_tx))
                .map_err(|e| Ml5Error::Backend(format!("failed to spawn candle worker: {e}")))?;
            ready_rx
                .await
                .map_err(|_| Ml5Error::Backend("candle worker exited during load".into()))??;
            *self.worker.lock().await = Some(WorkerHandle { tx, template, tokenizer });
            Ok(())
        }
    }

    async fn unload(&self) -> Result<()> {
        if let Some(w) = self.worker.lock().await.take() {
            let _ = w.tx.send(Job::Unload);
        }
        Ok(())
    }

    async fn is_alive(&self) -> bool {
        self.worker.lock().await.is_some()
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream> {
        if req.tools.as_ref().is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
            return Err(Ml5Error::InvalidRequest(
                "tool calling needs a GGUF model; safetensors models run on the candle backend without tool support".into(),
            ));
        }
        let template = self
            .worker
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| Ml5Error::Backend("backend not loaded".into()))?
            .template
            .clone();
        let (prompt, templated) = match template {
            Some(t) => (render_template(&t, &req.messages)?, true),
            None => (
                req.messages
                    .iter()
                    .map(|m| format!("{}: {}", m.role, m.content))
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\nassistant:",
                false,
            ),
        };
        self.submit(prompt, templated, req.params).await
    }

    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream> {
        self.submit(req.prompt, false, req.params).await
    }

    async fn count_tokens(&self, req: &ChatRequest) -> Result<u32> {
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = req;
            Err(Ml5Error::Backend(
                "safetensors support not compiled in (build with --features safetensors)".into(),
            ))
        }
        #[cfg(feature = "safetensors")]
        {
            let guard = self.worker.lock().await;
            let worker = guard
                .as_ref()
                .ok_or_else(|| Ml5Error::Backend("backend not loaded".into()))?;
            let (prompt, templated) = match &worker.template {
                Some(t) => (render_template(t, &req.messages)?, true),
                None => (
                    req.messages
                        .iter()
                        .map(|m| format!("{}: {}", m.role, m.content))
                        .collect::<Vec<_>>()
                        .join("\n")
                        + "\nassistant:",
                    false,
                ),
            };
            let enc = worker
                .tokenizer
                .encode(prompt.as_str(), !templated)
                .map_err(|e| Ml5Error::Backend(format!("tokenization failed: {e}")))?;
            Ok(enc.get_ids().len() as u32)
        }
    }

    async fn embed(&self, _req: EmbedRequest) -> Result<EmbedResponse> {
        Err(Ml5Error::InvalidRequest(
            "embeddings are not available for safetensors models; use a GGUF embedding model".into(),
        ))
    }
}

impl CandleBackend {
    async fn submit(&self, prompt: String, templated: bool, params: SamplingParams) -> Result<TokenStream> {
        let guard = self.worker.lock().await;
        let worker = guard
            .as_ref()
            .ok_or_else(|| Ml5Error::Backend("backend not loaded".into()))?;
        let (tx, rx) = mpsc::unbounded_channel();
        worker
            .tx
            .send(Job::Generate(GenerateJob {
                prompt,
                templated,
                params,
                out: tx,
            }))
            .map_err(|_| Ml5Error::Backend("candle worker died".into()))?;
        drop(guard);
        Ok(tokio_stream::wrappers::UnboundedReceiverStream::new(rx).boxed())
    }
}

#[cfg(feature = "safetensors")]
struct LoadedLlama {
    model: Llama,
    tokenizer: tokenizers::Tokenizer,
    config: Config,
    device: Device,
    eos_ids: Vec<u32>,
}

#[cfg(feature = "safetensors")]
fn candle_worker(
    path: PathBuf,
    rx: std_mpsc::Receiver<Job>,
    ready: tokio::sync::oneshot::Sender<Result<()>>,
) {
    let loaded = match load_llama(&path) {
        Ok(l) => {
            let _ = ready.send(Ok(()));
            l
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    while let Ok(job) = rx.recv() {
        match job {
            Job::Unload => break,
            Job::Generate(j) => {
                let out = j.out.clone();
                if let Err(e) = run_generate(&loaded, j) {
                    let _ = out.send(Err(e));
                }
            }
        }
    }
}

#[cfg(feature = "safetensors")]
fn load_llama(path: &Path) -> Result<LoadedLlama> {
    let device = device();
    let config_str = std::fs::read_to_string(path.join("config.json"))
        .map_err(|e| Ml5Error::Backend(format!("no config.json in {}: {e}", path.display())))?;
    let llama_config: LlamaConfig = serde_json::from_str(&config_str)
        .map_err(|e| Ml5Error::Backend(format!("unsupported config.json (Llama-family models only): {e}")))?;
    let config = llama_config.into_config(false);
    let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
        .map_err(|e| Ml5Error::Backend(format!("no usable tokenizer.json in {}: {e}", path.display())))?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Ml5Error::Backend(format!("no .safetensors files in {}", path.display())));
    }
    let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&files, dtype, &device)
            .map_err(|e| Ml5Error::Backend(format!("failed to load safetensors: {e}")))?
    };
    let model = Llama::load(vb, &config)
        .map_err(|e| Ml5Error::Backend(format!("failed to build model: {e}")))?;
    let mut eos_ids: Vec<u32> = match &config.eos_token_id {
        Some(LlamaEosToks::Single(t)) => vec![*t],
        Some(LlamaEosToks::Multiple(ts)) => ts.clone(),
        None => Vec::new(),
    };
    if let Some(t) = load_template(path) {
        if let Some(id) = tokenizer.token_to_id(&t.eos) {
            eos_ids.push(id);
        }
    }
    for name in ["</s>", "<|eot_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(name) {
            eos_ids.push(id);
        }
    }
    eos_ids.sort_unstable();
    eos_ids.dedup();
    Ok(LoadedLlama {
        model,
        tokenizer,
        config,
        device,
        eos_ids,
    })
}

#[cfg(feature = "safetensors")]
fn candle_err(what: &str) -> impl Fn(candle_core::Error) -> Ml5Error + '_ {
    move |e| Ml5Error::Backend(format!("{what} failed: {e}"))
}

#[cfg(feature = "safetensors")]
fn run_generate(l: &LoadedLlama, job: GenerateJob) -> Result<()> {
    let started = std::time::Instant::now();
    let enc = l
        .tokenizer
        .encode(job.prompt.as_str(), !job.templated)
        .map_err(|e| Ml5Error::Backend(format!("tokenization failed: {e}")))?;
    let tokens: Vec<u32> = enc.get_ids().to_vec();
    if tokens.is_empty() {
        return Err(Ml5Error::InvalidRequest("empty prompt".into()));
    }
    let prompt_len = tokens.len();
    if prompt_len >= l.config.max_position_embeddings {
        return Err(Ml5Error::InvalidRequest(format!(
            "prompt has {prompt_len} tokens but the model supports {}",
            l.config.max_position_embeddings
        )));
    }
    let max_tokens = (job.params.max_tokens.unwrap_or(2048) as usize)
        .min(l.config.max_position_embeddings - prompt_len);
    let mut cache = Cache::new(true, DType::F32, &l.config, &l.device).map_err(candle_err("cache init"))?;
    let temperature = job.params.temperature.unwrap_or(0.7) as f64;
    let mut sampler = LogitsProcessor::new(
        job.params.seed.map(u64::from).unwrap_or_else(rand_seed),
        (temperature > 0.0).then_some(temperature),
        job.params.top_p.map(f64::from),
    );
    let _ = job.out.send(Ok(StreamChunk {
        progress: Some(InferenceProgress {
            stage: "processing_prompt".into(),
            completed: 0,
            total: prompt_len,
        }),
        ..Default::default()
    }));

    let stops: Vec<&str> = job.params.stop.iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
    let mut generated: Vec<u32> = Vec::new();
    let mut text = String::new();
    let mut emitted = 0usize;
    let mut prompt_ms = 0u64;
    let mut finish = "length";
    let mut input_tokens = tokens;
    let mut pos = 0usize;
    for step in 0..max_tokens {
        let input = Tensor::new(input_tokens.as_slice(), &l.device)
            .map_err(candle_err("tensor"))?
            .unsqueeze(0)
            .map_err(candle_err("unsqueeze"))?;
        let logits = l
            .model
            .forward(&input, pos, &mut cache)
            .map_err(candle_err("forward"))?
            .squeeze(0)
            .map_err(candle_err("squeeze"))?;
        pos += input_tokens.len();
        if step == 0 {
            prompt_ms = started.elapsed().as_millis() as u64;
        }
        let next = sampler.sample(&logits).map_err(candle_err("sample"))?;
        if l.eos_ids.contains(&next) {
            finish = "stop";
            break;
        }
        generated.push(next);
        text = l
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| Ml5Error::Backend(format!("decode failed: {e}")))?;
        if text.ends_with('\u{FFFD}') {
            input_tokens = vec![next];
            continue;
        }
        let stop_at = stops.iter().filter_map(|s| text.find(s)).min();
        let limit = stop_at.unwrap_or_else(|| text.len() - held_back(&text, &stops));
        if !emit(&job.out, &text, &mut emitted, limit) {
            return Ok(());
        }
        if let Some(at) = stop_at {
            text.truncate(at);
            finish = "stop";
            break;
        }
        input_tokens = vec![next];
    }
    if finish == "length" && !text.ends_with('\u{FFFD}') && !emit(&job.out, &text, &mut emitted, text.len()) {
        return Ok(());
    }
    let total_ms = started.elapsed().as_millis() as u64;
    let _ = job.out.send(Ok(StreamChunk {
        done: true,
        usage: Some(Usage {
            prompt_tokens: prompt_len as u32,
            completion_tokens: generated.len() as u32,
            cached_tokens: 0,
            prompt_ms,
            generation_ms: total_ms.saturating_sub(prompt_ms),
        }),
        finish_reason: Some(finish.into()),
        ..Default::default()
    }));
    Ok(())
}

#[cfg(feature = "safetensors")]
fn held_back(text: &str, stops: &[&str]) -> usize {
    stops
        .iter()
        .flat_map(|s| (1..s.len()).filter(|k| s.is_char_boundary(*k)).map(move |k| &s[..k]))
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

#[cfg(feature = "safetensors")]
fn emit(
    out: &mpsc::UnboundedSender<Result<StreamChunk>>,
    text: &str,
    emitted: &mut usize,
    limit: usize,
) -> bool {
    if limit <= *emitted || !text.is_char_boundary(limit) || !text.is_char_boundary(*emitted) {
        return true;
    }
    let piece = text[*emitted..limit].to_string();
    *emitted = limit;
    out.send(Ok(StreamChunk {
        text: piece,
        ..Default::default()
    }))
    .is_ok()
}

#[cfg(feature = "safetensors")]
fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x4C4C)
}
