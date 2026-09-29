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
use std::time::Duration;
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

    fn push<'a>(&mut self, piece: &'a str) -> std::borrow::Cow<'a, str> {
        use std::borrow::Cow;
        match piece {
            "<|start|>" => {
                self.header = true;
                self.channel = false;
                self.metadata.clear();
                Cow::Borrowed("")
            }
            "<|channel|>" | "<|meta_sep|>" => {
                self.header = true;
                self.channel = true;
                self.metadata.clear();
                Cow::Borrowed("")
            }
            "<|message|>" | "<|im_sep|>" => {
                self.header = false;
                if self.metadata.trim() == "analysis" {
                    self.thinking = true;
                    Cow::Borrowed("<think>\n")
                } else {
                    Cow::Borrowed("")
                }
            }
            "<|end|>" | "<|im_end|>" => Cow::Owned(self.finish()),
            "<|return|>" | "<|fim_suffix|>" | "<|call|>" | "<|ghissue|>" => {
                Cow::Owned(self.finish())
            }
            _ if self.header => {
                if self.channel {
                    self.metadata.push_str(piece);
                }
                Cow::Borrowed("")
            }
            _ => Cow::Borrowed(piece),
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
    max_stop_len: usize,
}

impl StopFilter {
    fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let max_stop_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            stops,
            pending: String::new(),
            stopped: false,
            max_stop_len,
        }
    }

    fn push(&mut self, text: &str) -> (String, bool) {
        self.pending.push_str(text);
        if self.stops.is_empty() {
            return (std::mem::take(&mut self.pending), false);
        }
        if let Some(pos) = self.stops.iter().filter_map(|s| self.pending.find(s)).min() {
            let output = self.pending[..pos].to_string();
            self.pending.clear();
            self.stopped = true;
            return (output, true);
        }
        let floor = self.pending.len().saturating_sub(self.max_stop_len);
        let keep = self
            .pending
            .char_indices()
            .rev()
            .take_while(|(i, _)| *i >= floor)
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
    fn spawn(
        path: &Path,
        params: ModelParams,
        max_mem_fraction: f32,
        parallel: usize,
        kv_cache: bool,
    ) -> Result<Self> {
        let be = backend()?;

        let meta = std::fs::metadata(path)?;
        let model_size = meta.len();

        let mut params = params;

        if params.n_gpu_layers < 0 {
            tracing::debug!("requesting maximum GPU offload");
            params.n_gpu_layers = i32::MAX;
        }

        let gpu_offload = params.n_gpu_layers != 0;
        let est_ctx = estimate_ctx_bytes(params.n_ctx, 2048).saturating_mul(parallel.max(1) as u64);
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
            .spawn(move || inference_loop(worker_model, worker_params, rx, parallel, kv_cache))
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
    cache: bool,
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

fn run_inference<'m>(
    model: &'m LlamaModel,
    base: &ModelParams,
    job: GenerateJob,
    cache: &mut Option<PrefixCache<'m>>,
    kv_cache_enabled: bool,
) -> Result<()> {
    // dear fuck
    if job.out.is_closed() {
        return Ok(());
    }
    let prompt_started = std::time::Instant::now();
    let be = backend()?;
    let n_ctx = job.overrides.n_ctx.unwrap_or(base.n_ctx) as usize;

    let tokens = model
        .str_to_token(&job.prompt, llama_cpp_2::model::AddBos::Always)
        .map_err(|e| Ml5Error::Backend(format!("tokenization failed: {e}")))?;

    if tokens.len() >= n_ctx.saturating_sub(1) {
        return Err(Ml5Error::InvalidRequest(format!(
            "prompt ({} tokens) exceeds context size ({n_ctx})",
            tokens.len()
        )));
    }

    let use_cache = kv_cache_enabled && job.cache;

    let mut start_pos = 0usize;
    if use_cache && cache.is_some() {
        let cached = &cache.as_ref().unwrap().tokens;
        start_pos = cached
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
    }

    let mut owned_ctx;
    let ctx: &mut llama_cpp_2::context::LlamaContext = if use_cache {
        if cache.is_none() {
            let cp = ctx_params_from(base, &job.overrides, false);
            let c = model
                .new_context(be, cp)
                .map_err(|e| Ml5Error::Backend(format!("failed to create context: {e}")))?;
            *cache = Some(PrefixCache {
                ctx: c,
                tokens: Vec::new(),
            });
        }
        let pc = cache.as_mut().unwrap();
        let _ = pc.ctx.kv_cache_seq_rm(0, Some(start_pos as u32), None);
        pc.tokens.clone_from(&tokens);
        &mut pc.ctx
    } else {
        let cp = ctx_params_from(base, &job.overrides, false);
        owned_ctx = model
            .new_context(be, cp)
            .map_err(|e| Ml5Error::Backend(format!("failed to create context: {e}")))?;
        &mut owned_ctx
    };

    let batch_size = (job.overrides.n_batch.unwrap_or(base.n_batch) as usize)
        .max(1)
        .min(n_ctx);
    let mut batch = LlamaBatch::new(batch_size, 1);

    if start_pos < tokens.len() {
        let suffix = &tokens[start_pos..];
        for (index, part) in suffix.chunks(batch_size).enumerate() {
            if job.out.is_closed() {
                return Ok(());
            }
            batch.clear();
            let offset = start_pos + index * batch_size;
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
        let token: LlamaToken = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;

        let piece = model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| Ml5Error::Backend(format!("Cannot decode generated token: {e}")))?;
        let piece: std::borrow::Cow<str> = if let Some(harmony) = &mut harmony {
            harmony.push(&piece)
        } else {
            std::borrow::Cow::Owned(piece)
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

struct PrefixCache<'m> {
    ctx: llama_cpp_2::context::LlamaContext<'m>,
    tokens: Vec<LlamaToken>,
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

struct Slot {
    seq_id: i32,
    prompt: Vec<LlamaToken>,
    prompt_pos: usize,
    generated: u32,
    max_tokens: usize,
    sampler: LlamaSampler,
    decoder: encoding_rs::Decoder,
    stop: StopFilter,
    harmony: Option<HarmonyOutput>,
    out: mpsc::Sender<Result<StreamChunk>>,
    prompt_started: std::time::Instant,
    generation_started: Option<std::time::Instant>,
    prompt_ms: u64,
    finish_reason: &'static str,
    last_token: Option<LlamaToken>,
    pending_logits: bool,
    cache: bool,
}

impl Slot {
    fn new(
        seq_id: i32,
        job: GenerateJob,
        model: &LlamaModel,
        n_ctx: usize,
        start_pos: usize,
    ) -> Result<Self> {
        let tokens = model
            .str_to_token(&job.prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| Ml5Error::Backend(format!("tokenization failed: {e}")))?;
        if tokens.len() >= n_ctx.saturating_sub(1) {
            return Err(Ml5Error::InvalidRequest(format!(
                "prompt ({} tokens) exceeds context size ({n_ctx})",
                tokens.len()
            )));
        }
        let max_tokens = (job.params.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS) as usize)
            .min(n_ctx - tokens.len());
        let harmony = (model.meta_val_str("general.architecture").as_deref() == Ok("gpt-oss"))
            .then(|| HarmonyOutput::new(&job.prompt));
        Ok(Self {
            seq_id,
            prompt: tokens,
            prompt_pos: start_pos,
            generated: 0,
            max_tokens,
            sampler: build_sampler(&job.params, model.n_vocab()),
            decoder: encoding_rs::UTF_8.new_decoder(),
            stop: StopFilter::new(job.params.stop.clone()),
            harmony,
            out: job.out,
            prompt_started: std::time::Instant::now(),
            generation_started: None,
            prompt_ms: 0,
            finish_reason: "length",
            last_token: None,
            pending_logits: false,
            cache: job.cache,
        })
    }

    fn prompt_done(&self) -> bool {
        self.prompt_pos >= self.prompt.len()
    }
}

fn inference_loop(
    model: Arc<LlamaModel>,
    params: ModelParams,
    rx: std_mpsc::Receiver<Job>,
    parallel: usize,
    kv_cache: bool,
) {
    if parallel <= 1 {
        serial_loop(model, params, rx, kv_cache);
    } else {
        parallel_loop(model, params, rx, parallel, kv_cache);
    }
}

fn serial_loop(
    model: Arc<LlamaModel>,
    params: ModelParams,
    rx: std_mpsc::Receiver<Job>,
    kv_cache: bool,
) {
    let mut cache: Option<PrefixCache> = None;
    while let Ok(job) = rx.recv() {
        match job {
            Job::Generate(j) => {
                let out = j.out.clone();
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_inference(&model, &params, j, &mut cache, kv_cache)
                }));
                match res {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        let _ = out.blocking_send(Err(e));
                    }
                    Err(_) => {
                        tracing::error!("inference panicked; dropping KV cache state");
                        cache = None;
                        let _ = out.blocking_send(Err(Ml5Error::Backend(
                            "inference panicked (request aborted; worker survived)".into(),
                        )));
                    }
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
                drop(cache);
                drop(model);
                let _ = reply.send(());
                return;
            }
        }
    }
    tracing::debug!("inference worker exited");
}

fn parallel_loop(
    model: Arc<LlamaModel>,
    params: ModelParams,
    rx: std_mpsc::Receiver<Job>,
    parallel: usize,
    kv_cache: bool,
) {
    loop {
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            parallel_loop_inner(&model, &params, &rx, parallel, kv_cache)
        }));
        match res {
            Ok(Ok(())) => break,
            Ok(Err(e)) => {
                tracing::error!("parallel inference loop error: {e}; restarting with fresh context");
            }
            Err(_) => {
                tracing::error!("parallel inference panicked; restarting with fresh context");
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    tracing::debug!("inference worker exited");
}

fn parallel_loop_inner(
    model: &Arc<LlamaModel>,
    params: &ModelParams,
    rx: &std_mpsc::Receiver<Job>,
    parallel: usize,
    kv_cache: bool,
) -> Result<()> {
    let be = backend()?;
    let per_slot_ctx = params.n_ctx as usize;
    let total_ctx = per_slot_ctx.saturating_mul(parallel);
    let n_batch = (params.n_batch as usize).max(1);
    let n_ubatch = (params.n_ubatch as usize).max(1);

    let mut ctx_params = ctx_params_from(params, &RequestOverrides::default(), false)
        .with_n_seq_max(parallel as u32);
    ctx_params = ctx_params.with_n_ctx(NonZeroU32::new(total_ctx.max(1) as u32));
    let mut ctx = model
        .new_context(be, ctx_params)
        .map_err(|e| Ml5Error::Backend(format!(
            "failed to create context ({total_ctx} cells = {per_slot_ctx} x {parallel} slots): {e}"
        )))?;

    let mut slots: Vec<Option<Slot>> = (0..parallel).map(|_| None).collect();
    let mut slot_prefix: Vec<Vec<LlamaToken>> = (0..parallel).map(|_| Vec::new()).collect();
    let mut batch = LlamaBatch::new(n_batch.max(parallel), parallel as i32);
    let mut next_seq: i32 = 0;
    let mut pending_unload: Option<oneshot::Sender<()>> = None;

    loop {
        loop {
            let idle = slots.iter().all(Option::is_none);
            let job = if idle {
                match rx.recv() {
                    Ok(j) => Some(j),
                    Err(_) => None,
                }
            } else {
                match rx.try_recv() {
                    Ok(j) => Some(j),
                    Err(std_mpsc::TryRecvError::Empty) => {
                        match rx.recv_timeout(Duration::from_millis(1)) {
                            Ok(j) => Some(j),
                            Err(_) => None,
                        }
                    }
                    Err(std_mpsc::TryRecvError::Disconnected) => None,
                }
            };
            let job = match job {
                Some(j) => j,
                None if idle => return Ok(()),
                None => break,
            };
            match job {
                Job::Unload(reply) => {
                    if slots.iter().all(Option::is_none) {
                        let _ = reply.send(());
                        return Ok(());
                    }
                    pending_unload = Some(reply);
                }
                Job::Embed { reply, .. } => {
                    let _ = reply.send(Err(Ml5Error::InvalidRequest(
                        "embed is not available while --parallel is active; unload the model first".into(),
                    )));
                }
                Job::Generate(j) => {
                    if let Some(idx) = slots.iter().position(Option::is_none) {
                        let out = j.out.clone();
                        let want_cache = kv_cache && j.cache;
                        let mut start_pos = 0usize;
                        let mut best_idx = idx;
                        if want_cache {
                            let probe = model
                                .str_to_token(&j.prompt, llama_cpp_2::model::AddBos::Always)
                                .unwrap_or_default();
                            let mut best_len = 0usize;
                            for (i, p) in slot_prefix.iter().enumerate() {
                                if slots[i].is_some() || p.is_empty() {
                                    continue;
                                }
                                let common = p
                                    .iter()
                                    .zip(probe.iter())
                                    .take_while(|(a, b)| a == b)
                                    .count();
                                if common > best_len {
                                    best_len = common;
                                    best_idx = i;
                                }
                            }
                            start_pos = best_len;
                            if start_pos > 0 {
                                let _ = ctx.kv_cache_seq_rm(
                                    best_idx as i32,
                                    Some(start_pos as u32),
                                    None,
                                );
                            }
                        }
                        match Slot::new(best_idx as i32, j, model, per_slot_ctx, start_pos) {
                            Ok(s) => {
                                slots[best_idx] = Some(s);
                                next_seq = next_seq.wrapping_add(1);
                            }
                            Err(e) => {
                                let _ = out.blocking_send(Err(e));
                            }
                        }
                    } else {
                        let _ = j.out.blocking_send(Err(Ml5Error::InvalidRequest(
                            "all parallel slots are busy; retry shortly".into(),
                        )));
                    }
                }
            }
            if slots.iter().all(Option::is_none) && rx.try_recv().is_err() {
                break;
            }
        }

        batch.clear();
        let mut logit_idx_of_slot: Vec<Option<i32>> = vec![None; parallel];
        let batch_budget = n_batch;

        for (i, slot_opt) in slots.iter_mut().enumerate() {
            let slot = match slot_opt {
                Some(s) => s,
                None => continue,
            };
            if slot.out.is_closed() {
                slot_prefix[i] = Vec::new();
                let _ = ctx.kv_cache_seq_rm(slot.seq_id, None, None);
                *slot_opt = None;
                continue;
            }
            if slot.prompt_done() {
                continue;
            }

            let used = batch.n_tokens() as usize;
            if used >= batch_budget {
                break;
            }
            let room = batch_budget - used;

            let remaining = slot.prompt.len() - slot.prompt_pos;
            let take = remaining.min(n_ubatch).min(room);
            let chunk_end = slot.prompt_pos + take;
            for (j, t) in slot.prompt[slot.prompt_pos..chunk_end].iter().enumerate() {
                let pos = slot.prompt_pos + j;
                let is_last = pos + 1 == slot.prompt.len();
                let off = batch.n_tokens();
                batch
                    .add(*t, pos as i32, &[slot.seq_id], is_last)
                    .map_err(|e| Ml5Error::Backend(format!("batch failed: {e}")))?;
                if is_last {
                    logit_idx_of_slot[i] = Some(off);
                    slot.pending_logits = true;
                }
            }
            slot.prompt_pos = chunk_end;

            let _ = slot.out.blocking_send(Ok(StreamChunk {
                progress: Some(InferenceProgress {
                    stage: "processing_prompt".into(),
                    completed: slot.prompt_pos,
                    total: slot.prompt.len(),
                }),
                ..Default::default()
            }));
        }

        for (i, slot_opt) in slots.iter_mut().enumerate() {
            let slot = match slot_opt {
                Some(s) => s,
                None => continue,
            };
            if slot.pending_logits {
                continue;
            }
            let Some(tok) = slot.last_token else { continue };
            if slot.generated as usize >= slot.max_tokens || slot.finish_reason == "stop" {
                continue;
            }
            if slot.out.is_closed() {
                slot_prefix[i] = Vec::new();
                let _ = ctx.kv_cache_seq_rm(slot.seq_id, None, None);
                *slot_opt = None;
                continue;
            }
            let used = batch.n_tokens() as usize;
            if used >= batch_budget {
                break;
            }
            let pos = slot.prompt.len() + slot.generated as usize - 1;
            let off = batch.n_tokens();
            batch
                .add(tok, pos as i32, &[slot.seq_id], true)
                .map_err(|e| Ml5Error::Backend(format!("batch failed: {e}")))?;
            slot.pending_logits = true;
            logit_idx_of_slot[i] = Some(off);
        }

        if batch.n_tokens() > 0 {
            ctx.decode(&mut batch)
                .map_err(|e| Ml5Error::Backend(format!("decode failed: {e}")))?;
        }

        for (i, slot_opt) in slots.iter_mut().enumerate() {
            let Some(logit_idx) = logit_idx_of_slot[i] else { continue };
            let slot = match slot_opt {
                Some(s) => s,
                None => continue,
            };
            slot.pending_logits = false;

            if slot.generation_started.is_none() {
                slot.prompt_ms = slot.prompt_started.elapsed().as_millis() as u64;
                slot.generation_started = Some(std::time::Instant::now());
            }

            let token = slot.sampler.sample(&ctx, logit_idx);
            slot.sampler.accept(token);
            slot.last_token = Some(token);

            let mut done = false;
            if model.is_eog_token(token) {
                slot.finish_reason = "stop";
                done = true;
            } else {
                slot.generated += 1;
                let piece = model
                    .token_to_piece(token, &mut slot.decoder, true, None)
                    .map_err(|e| Ml5Error::Backend(format!("Cannot decode generated token: {e}")))?;
                let piece: std::borrow::Cow<str> = if let Some(h) = &mut slot.harmony {
                    h.push(&piece)
                } else {
                    std::borrow::Cow::Owned(piece)
                };
                let (filtered, matched_stop) = slot.stop.push(&piece);
                if !filtered.is_empty()
                    && slot
                        .out
                        .blocking_send(Ok(StreamChunk {
                            text: filtered,
                            ..Default::default()
                        }))
                        .is_err()
                {
                    slot_prefix[i] = Vec::new();
                    let _ = ctx.kv_cache_seq_rm(slot.seq_id, None, None);
                    *slot_opt = None;
                    continue;
                }
                if matched_stop {
                    slot.finish_reason = "stop";
                    done = true;
                }
                if slot.generated as usize >= slot.max_tokens {
                    done = true;
                }
            }

            if done {
                let tail = slot.stop.finish()
                    + &slot
                        .harmony
                        .as_mut()
                        .map(HarmonyOutput::finish)
                        .unwrap_or_default();
                let _ = slot.out.blocking_send(Ok(StreamChunk {
                    text: tail,
                    done: true,
                    usage: Some(Usage {
                        prompt_tokens: slot.prompt.len() as u32,
                        completion_tokens: slot.generated,
                        prompt_ms: slot.prompt_ms,
                        generation_ms: slot
                            .generation_started
                            .map(|t| t.elapsed().as_millis() as u64)
                            .unwrap_or(0),
                    }),
                    finish_reason: Some(slot.finish_reason.into()),
                    ..Default::default()
                }));
                if slot.cache && kv_cache {
                    let keep = slot.prompt.len();
                    slot_prefix[i] = slot.prompt.clone();
                    let _ = ctx.kv_cache_seq_rm(slot.seq_id, Some(keep as u32), None);
                } else {
                    slot_prefix[i] = Vec::new();
                    let _ = ctx.kv_cache_seq_rm(slot.seq_id, None, None);
                }
                *slot_opt = None;
            }
        }

        if slots.iter().all(Option::is_none) {
            if let Some(reply) = pending_unload.take() {
                let _ = reply.send(());
                return Ok(());
            }
        }
    }
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
    parallel: usize,
    manual_kv: bool,
    auto_kv: bool,
}

impl LlamaCppBackend {
    pub fn new(
        params: ModelParams,
        max_mem_fraction: f32,
        parallel: usize,
        manual_kv: bool,
        auto_kv: bool,
    ) -> Self {
        Self {
            worker: Mutex::new(None),
            path: Mutex::new(None),
            params: Mutex::new(params),
            max_mem_fraction,
            parallel,
            manual_kv,
            auto_kv,
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
        let parallel = self.parallel;
        let kv_cache = self.manual_kv || self.auto_kv;
        let handle = tokio::task::spawn_blocking(move || {
            WorkerHandle::spawn(&path, p, frac, parallel, kv_cache)
        })
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
        if self.parallel > 1 && req.overrides.n_ctx.is_some() {
            return Err(Ml5Error::InvalidRequest(
                "per-request n_ctx is not supported with --parallel (slots share one context); set --ctx-size on ml5d".into(),
            ));
        }
        let prompt = apply_chat_template(&worker.model, &req.messages)?;
        let (tx, rx) = mpsc::channel(32);
        worker.submit(Job::Generate(GenerateJob {
            prompt,
            params: req.params,
            overrides: req.overrides,
            cache: self.auto_kv || req.cache,
            out: tx,
        }))?;
        drop(guard);
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx).boxed())
    }

    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream> {
        let guard = self.worker().await?;
        let worker = guard.as_ref().expect("checked above");
        if self.parallel > 1 && req.overrides.n_ctx.is_some() {
            return Err(Ml5Error::InvalidRequest(
                "per-request n_ctx is not supported with --parallel (slots share one context); set --ctx-size on ml5d".into(),
            ));
        }
        let (tx, rx) = mpsc::channel(32);
        worker.submit(Job::Generate(GenerateJob {
            prompt: req.prompt,
            params: req.params,
            overrides: req.overrides,
            cache: self.auto_kv || req.cache,
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