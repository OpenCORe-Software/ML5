use crate::backend::{Backend, TokenStream};
use crate::config::{memory_budget, ModelParams};
use crate::error::{Ml5Error, Result};
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc as std_mpsc, Arc, OnceLock};
use tokio::sync::{mpsc, oneshot, Mutex};

const DEFAULT_MAX_TOKENS: u32 = 512;

struct HarmonyOutput {
    header: bool,
    channel: bool,
    metadata: String,
    thinking: bool,
}

impl HarmonyOutput {
    fn new(prompt: &str) -> Self {
        let tail = prompt.rsplit("<|start|>").next().unwrap_or("");
        Self {
            header: !tail.contains("<|message|>"),
            channel: false,
            metadata: String::new(),
            thinking: false,
        }
    }

    fn push(&mut self, piece: &str) -> String {
        match piece {
            "<|start|>" => {
                self.header = true;
                self.channel = false;
                self.metadata.clear();
                String::new()
            }
            "<|channel|>" | "<|meta_sep|>" => {
                self.header = true;
                self.channel = true;
                self.metadata.clear();
                String::new()
            }
            "<|message|>" | "<|im_sep|>" => {
                self.header = false;
                if self.metadata.trim() == "analysis" {
                    self.thinking = true;
                    "<think>\n".into()
                } else {
                    String::new()
                }
            }
            "<|end|>" | "<|im_end|>" => self.finish(),
            "<|return|>" | "<|fim_suffix|>" | "<|call|>" | "<|ghissue|>" => self.finish(),
            _ if self.header => {
                if self.channel {
                    self.metadata.push_str(piece);
                }
                String::new()
            }
            _ => piece.into(),
        }
    }

    fn finish(&mut self) -> String {
        if std::mem::take(&mut self.thinking) {
            "\n</think>\n\n".into()
        } else {
            String::new()
        }
    }
}

static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
static NEXT_SEQ_ID: AtomicUsize = AtomicUsize::new(0);

fn backend() -> Result<&'static LlamaBackend> {
    if let Some(b) = BACKEND.get() {
        return Ok(b);
    }

    #[cfg(feature = "gpu")]
    {
        let backends_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ml5")
            .join("backends");
        eprintln!(
            "[ml5] dynamic-backends enabled; backends dir: {}",
            backends_dir.display()
        );
        if backends_dir.exists() {
            let count = std::fs::read_dir(&backends_dir)
                .map(|d| {
                    d.flatten()
                        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("dll"))
                        .count()
                })
                .unwrap_or(0);
            eprintln!("[ml5] found {count} backend dll(s)");
            llama_cpp_2::llama_backend::load_backends_from_path(&backends_dir);
            tracing::info!(dir = %backends_dir.display(), "loaded dynamic backends");
        } else {
            eprintln!(
                "[ml5] backends dir does not exist: {}",
                backends_dir.display()
            );
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("[ml5] WARNING: built WITHOUT 'gpu' feature - dynamic backends disabled");
    }
    let b = LlamaBackend::init().map_err(|e| Ml5Error::Backend(format!("llama init: {e}")))?;
    let _ = BACKEND.set(b);
    Ok(BACKEND.get().expect("just initialized"))
}

fn flash_attn_policy(s: &str) -> llama_cpp_sys_2::llama_flash_attn_type {
    match s.to_ascii_lowercase().as_str() {
        "on" | "enabled" | "true" | "1" => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED,
        "off" | "disabled" | "false" | "0" => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED,
        _ => llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_AUTO,
    }
}

fn kv_cache_type(s: &str) -> KvCacheType {
    match s.to_ascii_lowercase().as_str() {
        "f32" => KvCacheType::F32,
        "f16" => KvCacheType::F16,
        "bf16" => KvCacheType::BF16,
        "q8_0" => KvCacheType::Q8_0,
        "q8_1" => KvCacheType::Q8_1,
        "q4_0" => KvCacheType::Q4_0,
        "q4_1" => KvCacheType::Q4_1,
        "q5_0" => KvCacheType::Q5_0,
        "q5_1" => KvCacheType::Q5_1,
        "q4_k" => KvCacheType::Q4_K,
        "q5_k" => KvCacheType::Q5_K,
        "q6_k" => KvCacheType::Q6_K,
        "q2_k" => KvCacheType::Q2_K,
        "q3_k" => KvCacheType::Q3_K,
        "iq4_nl" => KvCacheType::IQ4_NL,
        _ => KvCacheType::F16,
    }
}

fn model_params_from(p: &ModelParams) -> LlamaModelParams {
    let gpu_layers: u32 = if p.n_gpu_layers < 0 {
        u32::MAX
    } else {
        p.n_gpu_layers as u32
    };
    LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_main_gpu(p.main_gpu)
        .with_use_mmap(p.use_mmap)
        .with_use_mlock(p.use_mlock)
}

fn build_model_params(p: &ModelParams, n_gpu_layers: u32) -> LlamaModelParams {
    let mut mp = p.clone();
    mp.n_gpu_layers = n_gpu_layers as i32;
    model_params_from(&mp)
}

fn ctx_params_from(p: &ModelParams, o: &RequestOverrides, embeddings: bool) -> LlamaContextParams {
    let n_ctx = o.n_ctx.unwrap_or(p.n_ctx);
    let n_batch = o.n_batch.unwrap_or(p.n_batch);
    let flash = o.flash_attn.as_deref().unwrap_or(&p.flash_attn);
    let type_k = o.cache_type_k.as_deref().unwrap_or(&p.cache_type_k);
    let type_v = o.cache_type_v.as_deref().unwrap_or(&p.cache_type_v);
    let n_threads = o.n_threads.unwrap_or(p.n_threads);

    let mut cp = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx.max(1)))
        .with_n_batch(n_batch.max(1))
        .with_n_ubatch(p.n_ubatch.max(1))
        .with_flash_attention_policy(flash_attn_policy(flash))
        .with_type_k(kv_cache_type(type_k))
        .with_type_v(kv_cache_type(type_v))
        .with_offload_kqv(p.offload_kqv)
        .with_embeddings(embeddings);

    if n_threads > 0 {
        cp = cp.with_n_threads(n_threads);
    }
    if p.n_threads_batch > 0 {
        cp = cp.with_n_threads_batch(p.n_threads_batch);
    }
    cp
}

struct StopFilter {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopFilter {
    fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: String::new(),
            stopped: false,
        }
    }

    fn push(&mut self, text: &str) -> (String, bool) {
        self.pending.push_str(text);
        if let Some(pos) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            let output = self.pending[..pos].to_string();
            self.pending.clear();
            self.stopped = true;
            return (output, true);
        }
        let keep = self
            .pending
            .char_indices()
            .map(|(i, _)| &self.pending[i..])
            .filter(|suffix| self.stops.iter().any(|s| s.starts_with(suffix)))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let output = self.pending.drain(..self.pending.len() - keep).collect();
        (output, false)
    }

    fn finish(&mut self) -> String {
        if self.stopped {
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }
}

fn estimate_ctx_bytes(n_ctx: u32, n_embd: u64) -> u64 {
    2u64 * n_ctx as u64 * n_embd.max(1) * 2
}

#[allow(dead_code)]
fn compute_auto_gpu_layers(
    model_size: u64,
    n_layer: u32,
    n_ctx: u32,
    n_embd: u64,
    max_mem_fraction: f32,
) -> u32 {
    let total_vram = match crate::config::total_gpu_memory_bytes() {
        Some(v) => v,
        None => return 0,
    };
    let budget = (total_vram as f64 * max_mem_fraction as f64) as u64;

    let overhead = model_size / 20;
    let per_layer = if n_layer > 0 {
        model_size.saturating_sub(overhead) / n_layer as u64
    } else {
        0
    };
    if per_layer == 0 {
        return 0;
    }

    let ctx_cost = estimate_ctx_bytes(n_ctx, n_embd);
    let usable = budget.saturating_sub(overhead + ctx_cost);
    let layers = (usable / per_layer) as u32;
    layers.min(n_layer)
}

struct WorkerHandle {
    model: Arc<LlamaModel>,
    tx: std_mpsc::Sender<Job>,
}

impl WorkerHandle {
    fn spawn(path: &Path, params: ModelParams, max_mem_fraction: f32) -> Result<Self> {
        let be = backend()?;

        let meta = std::fs::metadata(path)?;
        let model_size = meta.len();

        let mut params = params;

        if params.n_gpu_layers < 0 {
            tracing::debug!("requesting maximum GPU offload");
            params.n_gpu_layers = i32::MAX;
        }

        let gpu_offload = params.n_gpu_layers != 0;
        let est_ctx = estimate_ctx_bytes(params.n_ctx, 2048);
        let budget = memory_budget(max_mem_fraction, gpu_offload);
        if let Err(msg) = budget.check(model_size, est_ctx) {
            return Err(Ml5Error::Backend(format!("memory guard: {msg}")));
        }

        let mp = build_model_params(&params, params.n_gpu_layers.max(0) as u32);
        let model = LlamaModel::load_from_file(be, path, &mp)
            .map_err(|e| Ml5Error::Backend(format!("failed to load {}: {e}", path.display())))?;
        let model = Arc::new(model);

        let (tx, rx) = std_mpsc::channel::<Job>();
        let worker_model = model.clone();
        let worker_params = params.clone();
        std::thread::Builder::new()
            .name(format!(
                "ml5-infer-{}",
                NEXT_SEQ_ID.fetch_add(1, Ordering::Relaxed)
            ))
            .spawn(move || inference_loop(worker_model, worker_params, rx))
            .map_err(|e| Ml5Error::Backend(format!("failed to spawn inference thread: {e}")))?;

        Ok(Self { model, tx })
    }

    fn submit(&self, job: Job) -> Result<()> {
        self.tx
            .send(job)
            .map_err(|_| Ml5Error::Backend("inference worker died".into()))
    }
}

enum Job {
    Generate(GenerateJob),
    Embed {
        overrides: RequestOverrides,
        input: Vec<String>,
        reply: oneshot::Sender<Result<EmbedResponse>>,
    },
    Unload(oneshot::Sender<()>),
}

struct GenerateJob {
    prompt: String,
    params: SamplingParams,
    overrides: RequestOverrides,
    out: mpsc::Sender<Result<StreamChunk>>,
}

fn build_sampler(params: &SamplingParams, n_vocab: i32) -> LlamaSampler {
    let mut chain: Vec<LlamaSampler> = Vec::new();

    let last_n = params.penalty_last_n.unwrap_or(64);
    let repeat = params.repeat_penalty.unwrap_or(1.0);
    let freq = params.frequency_penalty.unwrap_or(0.0);
    let pres = params.presence_penalty.unwrap_or(0.0);
    if repeat != 1.0 || freq != 0.0 || pres != 0.0 {
        chain.push(LlamaSampler::penalties(n_vocab, last_n, repeat, freq, pres));
    }

    if let Some(k) = params.top_k {
        if k > 0 {
            chain.push(LlamaSampler::top_k(k));
        }
    }
    if let Some(p) = params.top_p {
        if (0.0..=1.0).contains(&p) {
            chain.push(LlamaSampler::top_p(p, 1));
        }
    }
    if let Some(mp) = params.min_p {
        if mp > 0.0 {
            chain.push(LlamaSampler::min_p(mp, 1));
        }
    }
    chain.push(LlamaSampler::temp(params.temperature.unwrap_or(0.8)));
    chain.push(LlamaSampler::dist(params.seed.unwrap_or(0xFFFFFFFF)));
    LlamaSampler::chain_simple(chain)
}

fn run_inference(model: &LlamaModel, base: &ModelParams, job: GenerateJob) -> Result<()> {
    // dear fuck
    if job.out.is_closed() {
        return Ok(());
    }
    let prompt_started = std::time::Instant::now();
    let be = backend()?;
    let ctx_params = ctx_params_from(base, &job.overrides, false);
    let n_ctx = job.overrides.n_ctx.unwrap_or(base.n_ctx) as usize;

    let mut ctx = model
        .new_context(be, ctx_params)
        .map_err(|e| Ml5Error::Backend(format!("failed to create context: {e}")))?;

    let tokens = model
        .str_to_token(&job.prompt, llama_cpp_2::model::AddBos::Always)
        .map_err(|e| Ml5Error::Backend(format!("tokenization failed: {e}")))?;

    if tokens.len() >= n_ctx.saturating_sub(1) {
        return Err(Ml5Error::InvalidRequest(format!(
            "prompt ({} tokens) exceeds context size ({n_ctx})",
            tokens.len()
        )));
    }

    let batch_size = (job.overrides.n_batch.unwrap_or(base.n_batch) as usize)
        .max(1)
        .min(n_ctx);
    let mut batch = LlamaBatch::new(batch_size, 1);
    for (index, part) in tokens.chunks(batch_size).enumerate() {
        if job.out.is_closed() {
            return Ok(());
        }
        batch.clear();
        let offset = index * batch_size;
        for (i, token) in part.iter().enumerate() {
            let pos = offset + i;
            batch
                .add(*token, pos as i32, &[0], pos + 1 == tokens.len())
                .map_err(|e| Ml5Error::Backend(format!("batch failed: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| Ml5Error::Backend(format!("prompt decode failed: {e}")))?;
        if job
            .out
            .blocking_send(Ok(StreamChunk {
                progress: Some(InferenceProgress {
                    stage: "processing_prompt".into(),
                    completed: offset + part.len(),
                    total: tokens.len(),
                }),
                ..Default::default()
            }))
            .is_err()
        {
            return Ok(());
        }
    }
    let prompt_ms = prompt_started.elapsed().as_millis() as u64;

    let mut sampler = build_sampler(&job.params, model.n_vocab());
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let max_tokens =
        (job.params.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS) as usize).min(n_ctx - tokens.len());
    let mut completion_tokens: u32 = 0;

    let mut stop_filter = StopFilter::new(job.params.stop.clone());
    let mut finish_reason = "length";
    let generation_started = std::time::Instant::now();
    let mut harmony = (model.meta_val_str("general.architecture").as_deref() == Ok("gpt-oss"))
        .then(|| HarmonyOutput::new(&job.prompt));

    for n_cur in (tokens.len()..).take(max_tokens) {
        if job.out.is_closed() {
            return Ok(());
        }
        let token: LlamaToken = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| Ml5Error::Backend(format!("Cannot decode generated token: {e}")))?;
        let piece = if let Some(harmony) = &mut harmony {
            harmony.push(&piece)
        } else {
            piece
        };
        let (filtered, matched_stop) = stop_filter.push(&piece);

        if !filtered.is_empty()
            && job
                .out
                .blocking_send(Ok(StreamChunk {
                    text: filtered,
                    ..Default::default()
                }))
                .is_err()
        {
            return Ok(());
        }

        if matched_stop {
            finish_reason = "stop";
            break;
        }
        if completion_tokens as usize == max_tokens {
            break;
        }

        batch.clear();
        batch
            .add(token, n_cur as i32, &[0], true)
            .map_err(|e| Ml5Error::Backend(format!("batch failed: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| Ml5Error::Backend(format!("decode failed: {e}")))?;
    }

    let _ = job.out.blocking_send(Ok(StreamChunk {
        text: stop_filter.finish()
            + &harmony
                .as_mut()
                .map(HarmonyOutput::finish)
                .unwrap_or_default(),
        done: true,
        usage: Some(Usage {
            prompt_tokens: tokens.len() as u32,
            completion_tokens,
            prompt_ms,
            generation_ms: generation_started.elapsed().as_millis() as u64,
        }),
        finish_reason: Some(finish_reason.into()),
        ..Default::default()
    }));
    Ok(())
}

fn run_embed(
    model: &LlamaModel,
    base: &ModelParams,
    overrides: &RequestOverrides,
    input: Vec<String>,
) -> Result<EmbedResponse> {
    let be = backend()?;
    let ctx_params = ctx_params_from(base, overrides, true);
    let n_ctx = overrides.n_ctx.unwrap_or(base.n_ctx) as usize;

    let mut ctx = model
        .new_context(be, ctx_params)
        .map_err(|e| Ml5Error::Backend(format!("failed to create context: {e}")))?;

    let mut embeddings = Vec::with_capacity(input.len());
    let mut total_prompt_tokens = 0u32;

    for text in &input {
        let tokens = model
            .str_to_token(text, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Ml5Error::Backend(format!("tokenization failed: {e}")))?;
        total_prompt_tokens += tokens.len() as u32;

        let mut batch = LlamaBatch::new(tokens.len().max(1).min(n_ctx), 1);
        batch
            .add_sequence(&tokens, 0, true)
            .map_err(|e| Ml5Error::Backend(format!("batch failed: {e}")))?;
        ctx.clear_kv_cache();
        ctx.decode(&mut batch)
            .map_err(|e| Ml5Error::Backend(format!("decode failed: {e}")))?;

        let pooled = match ctx.embeddings_seq_ith(0) {
            Ok(seq) => seq.to_vec(),
            Err(_) => {
                let n = tokens.len();
                let dim = model.n_embd() as usize;
                let mut acc = vec![0f32; dim];
                let mut count = 0usize;
                for i in 0..n {
                    if let Ok(e) = ctx.embeddings_ith(i as i32) {
                        for (a, v) in acc.iter_mut().zip(e.iter()) {
                            *a += v;
                        }
                        count += 1;
                    }
                }
                if count == 0 {
                    return Err(Ml5Error::Backend(
                        "model exposes neither sequence nor token embeddings".into(),
                    ));
                }
                for a in &mut acc {
                    *a /= count as f32;
                }
                acc
            }
        };
        embeddings.push(pooled);
    }

    Ok(EmbedResponse {
        embeddings,
        usage: Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: 0,
            ..Default::default()
        },
    })
}

fn inference_loop(model: Arc<LlamaModel>, params: ModelParams, rx: std_mpsc::Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        match job {
            Job::Generate(j) => {
                let out = j.out.clone();
                if let Err(e) = run_inference(&model, &params, j) {
                    let _ = out.blocking_send(Err(e));
                }
            }
            Job::Embed {
                overrides,
                input,
                reply,
            } => {
                let _ = reply.send(run_embed(&model, &params, &overrides, input));
            }
            Job::Unload(reply) => {
                drop(model);
                let _ = reply.send(());
                return;
            }
        }
    }
    tracing::debug!("inference worker exited");
}

const CHATML_TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

fn apply_chat_template(model: &LlamaModel, messages: &[Message]) -> Result<String> {
    let tmpl = model
        .chat_template(None)
        .or_else(|_| {
            llama_cpp_2::model::LlamaChatTemplate::new(CHATML_TEMPLATE)
                .map_err(|e| Ml5Error::Backend(format!("chatml template: {e}")))
        })
        .map_err(|e| Ml5Error::Backend(format!("no chat template: {e}")))?;

    let msgs: Vec<llama_cpp_2::model::LlamaChatMessage> = messages
        .iter()
        .map(|m| llama_cpp_2::model::LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| Ml5Error::InvalidRequest(format!("bad chat message: {e}")))?;

    model
        .apply_chat_template(&tmpl, &msgs, true)
        .map_err(|e| Ml5Error::Backend(format!("apply chat template: {e}")))
}

pub struct LlamaCppBackend {
    worker: Mutex<Option<WorkerHandle>>,
    path: Mutex<Option<PathBuf>>,
    params: Mutex<ModelParams>,
    max_mem_fraction: f32,
}

impl LlamaCppBackend {
    pub fn new(params: ModelParams, max_mem_fraction: f32) -> Self {
        Self {
            worker: Mutex::new(None),
            path: Mutex::new(None),
            params: Mutex::new(params),
            max_mem_fraction,
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
impl Backend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Chat, Capability::Generate, Capability::Embed]
    }

    async fn load(&self, model_path: &Path, params: &ModelParams) -> Result<()> {
        let mut guard = self.worker.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        *self.params.lock().await = params.clone();
        let path = model_path.to_path_buf();
        let p = params.clone();
        let frac = self.max_mem_fraction;
        let handle = tokio::task::spawn_blocking(move || WorkerHandle::spawn(&path, p, frac))
            .await
            .map_err(|e| Ml5Error::Backend(format!("load task failed: {e}")))??;
        *self.path.lock().await = Some(model_path.to_path_buf());
        *guard = Some(handle);
        tracing::info!(path = %model_path.display(), "model loaded");
        Ok(())
    }

    async fn unload(&self) -> Result<()> {
        let mut guard = self.worker.lock().await;
        if let Some(w) = guard.take() {
            let (tx, rx) = oneshot::channel();
            w.submit(Job::Unload(tx))?;
            drop(w);
            rx.await.map_err(|_| {
                Ml5Error::Backend("Worker stopped before unloading completed".into())
            })?;
        }
        *self.path.lock().await = None;
        Ok(())
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream> {
        let guard = self.worker().await?;
        let worker = guard.as_ref().expect("checked above");
        let prompt = apply_chat_template(&worker.model, &req.messages)?;
        let (tx, rx) = mpsc::channel(32);
        worker.submit(Job::Generate(GenerateJob {
            prompt,
            params: req.params,
            overrides: req.overrides,
            out: tx,
        }))?;
        drop(guard);
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }

    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream> {
        let guard = self.worker().await?;
        let worker = guard.as_ref().expect("checked above");
        let (tx, rx) = mpsc::channel(32);
        worker.submit(Job::Generate(GenerateJob {
            prompt: req.prompt,
            params: req.params,
            overrides: req.overrides,
            out: tx,
        }))?;
        drop(guard);
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }

    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
        let guard = self.worker().await?;
        let worker = guard.as_ref().expect("checked above");
        let (tx, rx) = oneshot::channel();
        worker.submit(Job::Embed {
            overrides: RequestOverrides::default(),
            input: req.input,
            reply: tx,
        })?;
        drop(guard);
        rx.await
            .map_err(|_| Ml5Error::Backend("embed reply dropped".into()))?
    }
}

#[cfg(test)]
mod tests {
    use super::{HarmonyOutput, StopFilter};

    #[test]
    fn harmony_analysis_boundary_continues_into_visible_final_answer() {
        let mut output = HarmonyOutput::new("<|start|>user<|message|>hi<|end|><|start|>assistant");
        let pieces = [
            "<|channel|>",
            "analysis",
            "<|message|>",
            "Let me think.",
            "<|end|>",
            "<|start|>",
            "assistant",
            "<|channel|>",
            "final",
            "<|message|>",
            "Hello!",
            "<|return|>",
        ];
        let text: String = pieces.into_iter().map(|p| output.push(p)).collect();
        assert_eq!(text, "<think>\nLet me think.\n</think>\n\nHello!");
    }

    #[test]
    fn harmony_message_marker_is_not_a_stop_signal() {
        let mut output = HarmonyOutput::new("<|start|>assistant");
        assert_eq!(output.push("<|message|>"), "");
        assert_eq!(output.push("Hi!"), "Hi!");
        let mut output = HarmonyOutput::new("<|start|>assistant<|message|>");
        assert_eq!(output.push("Hi!"), "Hi!");
    }

    #[test]
    fn stop_sequences_cross_tokens_without_leaking() {
        let mut filter = StopFilter::new(vec!["<stop>".into(), "終わり".into()]);
        assert_eq!(filter.push("hello <st"), ("hello ".into(), false));
        assert_eq!(filter.push("op>secret"), (String::new(), true));
        assert_eq!(filter.finish(), "");
        let mut filter = StopFilter::new(vec!["終わり".into()]);
        assert_eq!(filter.push("日本語終"), ("日本語".into(), false));
        assert_eq!(filter.push("わり後"), (String::new(), true));
    }

    #[test]
    fn unmatched_prefix_is_preserved() {
        let mut filter = StopFilter::new(vec!["stop".into(), String::new()]);
        assert_eq!(filter.push("test st"), ("test ".into(), false));
        assert_eq!(filter.push("uff"), ("stuff".into(), false));
        assert_eq!(filter.push("st"), (String::new(), false));
        assert_eq!(filter.finish(), "st");
    }
}
