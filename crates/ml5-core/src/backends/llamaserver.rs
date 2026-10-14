use crate::backend::{Backend, LoadSpec, TokenStream};
use crate::error::{Ml5Error, Result};
use crate::runtime;
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const READY_TIMEOUT: Duration = Duration::from_secs(1800);
const STOP_GRACE: Duration = Duration::from_secs(10);

struct Instance {
    child: Child,
    port: u16,
    log_path: PathBuf,
    #[cfg(windows)]
    _job: Option<JobGuard>,
}

pub struct LlamaServerBackend {
    instance: Mutex<Option<Instance>>,
    base: RwLock<Option<String>>,
    server_override: Option<PathBuf>,
    log_dir: PathBuf,
    client: reqwest::Client,
    opt_cache_prompt: AtomicBool,
}

impl LlamaServerBackend {
    pub fn new(server_override: Option<PathBuf>, log_dir: PathBuf) -> Self {
        Self {
            instance: Mutex::new(None),
            base: RwLock::new(None),
            server_override,
            log_dir,
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .pool_idle_timeout(Duration::from_secs(90))
                .build()
                .expect("reqwest client"),
            opt_cache_prompt: AtomicBool::new(false),
        }
    }

    fn free_port() -> Result<u16> {
        TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr().map(|a| a.port()))
            .map_err(|e| Ml5Error::Backend(format!("no free local port for llama-server: {e}")))
    }

    fn url(&self, path: &str) -> Result<String> {
        let base = self
            .base
            .read()
            .ok()
            .and_then(|b| b.clone())
            .ok_or_else(|| Ml5Error::Backend("model runtime is not running".into()))?;
        Ok(format!("{base}{path}"))
    }

    fn build_command(server: &Path, spec: &LoadSpec, port: u16) -> Command {
        let p = &spec.params;
        let mut cmd = Command::new(server);
        let ctx_total = p.n_ctx.max(256) as usize * spec.n_parallel.max(1);
        let (batch, ubatch) = if spec.embeddings {
            let n = (ctx_total as u32).max(p.n_batch);
            (n, n)
        } else {
            (p.n_batch, p.n_ubatch.min(p.n_batch))
        };
        cmd.arg("--model")
            .arg(&spec.model.path)
            .arg("--alias")
            .arg(&spec.model.name)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(ctx_total.to_string())
            .arg("--parallel")
            .arg(spec.n_parallel.max(1).to_string())
            .arg("--batch-size")
            .arg(batch.to_string())
            .arg("--ubatch-size")
            .arg(ubatch.to_string())
            .arg("--n-gpu-layers")
            .arg(p.n_gpu_layers.as_arg())
            .arg("--fit")
            .arg(if p.n_gpu_layers == crate::config::GpuLayers::Auto { "on" } else { "off" })
            .arg("--flash-attn")
            .arg(match p.flash_attn.to_ascii_lowercase().as_str() {
                "on" | "enabled" | "true" | "1" => "on",
                "off" | "disabled" | "false" | "0" => "off",
                _ => "auto",
            })
            .arg("--cache-type-k")
            .arg(&p.cache_type_k)
            .arg("--cache-type-v")
            .arg(&p.cache_type_v)
            .arg("--load-mode")
            .arg(match (p.use_mmap, p.use_mlock) {
                (true, true) => "mmap+mlock",
                (true, false) => "mmap",
                (false, true) => "mlock",
                (false, false) => "none",
            })
            .arg("--no-webui")
            .arg("--log-colors")
            .arg("off");
        if spec.embeddings {
            cmd.arg("--embeddings");
        } else {
            cmd.arg("--jinja").arg("--reasoning-format").arg("deepseek");
            if let Some(mmproj) = &spec.model.mmproj {
                cmd.arg("--mmproj").arg(mmproj);
                if !p.n_gpu_layers.uses_gpu() {
                    cmd.arg("--no-mmproj-offload");
                }
            }
        }
        if !p.offload_kqv {
            cmd.arg("--no-kv-offload");
        }
        if !p.cache_prompt || p.opt_cache_prompt {
            cmd.arg("--no-cache-prompt");
        }
        if p.cache_reuse > 0 {
            cmd.arg("--cache-reuse").arg(p.cache_reuse.to_string());
        }
        if p.main_gpu > 0 {
            cmd.arg("--main-gpu").arg(p.main_gpu.to_string());
        }
        if let Some(ts) = p.tensor_split.as_ref().filter(|s| !s.is_empty()) {
            cmd.arg("--tensor-split").arg(ts);
        }
        if let Some(nodes) = p.rpc_nodes.as_ref().filter(|s| !s.is_empty()) {
            cmd.arg("--rpc").arg(nodes);
        }
        if p.n_threads > 0 {
            cmd.arg("--threads").arg(p.n_threads.to_string());
        }
        if p.n_threads_batch > 0 {
            cmd.arg("--threads-batch").arg(p.n_threads_batch.to_string());
        }
        if let Some(dir) = server.parent() {
            cmd.current_dir(dir);
            #[cfg(target_os = "linux")]
            {
                let mut ld = std::ffi::OsString::from(dir);
                if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH").filter(|v| !v.is_empty()) {
                    ld.push(":");
                    ld.push(existing);
                }
                cmd.env("LD_LIBRARY_PATH", ld);
            }
        }
        for (key, _) in std::env::vars_os() {
            let k = key.to_string_lossy();
            if k.starts_with("LLAMA_ARG_") || k == "LLAMA_API_KEY" || k == "MTMD_BACKEND_DEVICE" {
                cmd.env_remove(&key);
            }
        }
        cmd.stdin(Stdio::null()).kill_on_drop(true);
        #[cfg(target_os = "linux")]
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                if libc::getppid() == 1 {
                    libc::_exit(1);
                }
                Ok(())
            });
        }
        #[cfg(windows)]
        {
            cmd.creation_flags(0x08000000);
        }
        cmd
    }

    async fn wait_ready(&self, port: u16, log_path: &Path) -> Result<()> {
        let url = format!("http://127.0.0.1:{port}/health");
        let started = Instant::now();
        loop {
            {
                let mut guard = self.instance.lock().await;
                let Some(inst) = guard.as_mut() else {
                    return Err(Ml5Error::Backend("model load was cancelled".into()));
                };
                if let Ok(Some(status)) = inst.child.try_wait() {
                    *guard = None;
                    return Err(Ml5Error::Backend(format!(
                        "llama-server exited during load ({status}): {}",
                        explain_log(log_path)
                    )));
                }
            }
            if let Ok(r) = self
                .client
                .get(&url)
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                if r.status().is_success() {
                    return Ok(());
                }
            }
            if started.elapsed() > READY_TIMEOUT {
                return Err(Ml5Error::Backend(format!(
                    "llama-server did not become ready within {} minutes: {}",
                    READY_TIMEOUT.as_secs() / 60,
                    explain_log(log_path)
                )));
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    async fn post_stream(&self, path: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
        let resp = self
            .client
            .post(self.url(path)?)
            .json(body)
            .send()
            .await
            .map_err(|e| Ml5Error::Backend(format!("model runtime request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(runtime_error(status, &text));
        }
        Ok(resp)
    }

    fn apply_cache_prompt(&self, body: &mut serde_json::Value, requested: Option<bool>) {
        if self.opt_cache_prompt.load(Ordering::Relaxed) {
            if let Some(o) = body.as_object_mut() {
                o.insert("cache_prompt".into(), requested.unwrap_or(false).into());
            }
        }
    }
}

pub fn runtime_error(status: reqwest::StatusCode, text: &str) -> Ml5Error {
    let message = serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|v| {
            v["error"]["message"]
                .as_str()
                .or_else(|| v["error"].as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| text.trim().to_string());
    if status.is_client_error() {
        Ml5Error::InvalidRequest(message)
    } else {
        Ml5Error::Backend(format!("llama-server returned {status}: {message}"))
    }
}

fn explain_log(path: &Path) -> String {
    let text = std::fs::read(path)
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let tail = &lines[lines.len().saturating_sub(60)..];
    let important: Vec<&str> = tail
        .iter()
        .copied()
        .filter(|l| {
            let l = l.to_ascii_lowercase();
            l.contains("error") || l.contains("failed") || l.contains("unable") || l.contains("out of memory") || l.contains("not enough")
        })
        .collect();
    let chosen = if important.is_empty() {
        tail[tail.len().saturating_sub(3)..].to_vec()
    } else {
        important[important.len().saturating_sub(4)..].to_vec()
    };
    if chosen.is_empty() {
        format!("no output (see {})", path.display())
    } else {
        format!("{} (full log: {})", chosen.join(" | "), path.display())
    }
}

fn log_file_name(model: &str) -> String {
    let safe: String = model
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect();
    format!("llama-server-{}.log", safe.chars().take(120).collect::<String>())
}

#[cfg(unix)]
fn terminate(child: &Child) -> bool {
    if let Some(pid) = child.id() {
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
    }
    true
}

#[cfg(not(unix))]
fn terminate(_child: &Child) -> bool {
    false
}

#[cfg(windows)]
struct JobGuard(winapi::um::winnt::HANDLE);

#[cfg(windows)]
impl Drop for JobGuard {
    fn drop(&mut self) {
        unsafe {
            winapi::um::handleapi::CloseHandle(self.0);
        }
    }
}

#[cfg(windows)]
fn job_guard(child: &Child) -> Option<JobGuard> {
    use winapi::um::jobapi2::{AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject};
    use winapi::um::winnt::{
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    const WARN: &str = "llama-server is not tied to ml5d's lifetime; it may keep running if ml5d is force-killed";
    let job = unsafe { CreateJobObjectW(std::ptr::null_mut(), std::ptr::null()) };
    if job.is_null() {
        tracing::warn!("CreateJobObject failed; {WARN}");
        return None;
    }
    let guard = JobGuard(job);
    let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    let ok = unsafe {
        SetInformationJobObject(
            guard.0,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *mut _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    };
    if ok == 0 {
        tracing::warn!("SetInformationJobObject failed; {WARN}");
        return None;
    }
    let Some(handle) = child.raw_handle() else {
        tracing::warn!("no process handle for llama-server; {WARN}");
        return None;
    };
    if unsafe { AssignProcessToJobObject(guard.0, handle as _) } == 0 {
        tracing::warn!("AssignProcessToJobObject failed; {WARN}");
        return None;
    }
    Some(guard)
}

pub struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some((end, sep)) = find_event_end(&self.buf) {
            let block: Vec<u8> = self.buf.drain(..end + sep).collect();
            let text = String::from_utf8_lossy(&block[..end]);
            let mut data = String::new();
            for line in text.lines() {
                let line = line.trim_end_matches('\r');
                if let Some(rest) = line.strip_prefix("data:") {
                    if !data.is_empty() {
                        data.push('\n');
                    }
                    data.push_str(rest.strip_prefix(' ').unwrap_or(rest));
                }
            }
            if !data.is_empty() {
                out.push(data);
            }
        }
        out
    }

    pub fn finish(&mut self) -> Vec<String> {
        if self.buf.is_empty() {
            return Vec::new();
        }
        self.buf.extend_from_slice(b"\n\n");
        self.push(&[])
    }
}

impl Default for SseDecoder {
    fn default() -> Self {
        Self::new()
    }
}

fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, 2));
        }
        if i + 3 < buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

#[derive(Default)]
struct ToolAccumulator {
    calls: Vec<ToolCall>,
}

impl ToolAccumulator {
    fn push(&mut self, deltas: &serde_json::Value) {
        let Some(items) = deltas.as_array() else { return };
        for (pos, d) in items.iter().enumerate() {
            let index = d["index"].as_u64().map(|i| i as usize).unwrap_or(pos);
            while self.calls.len() <= index {
                self.calls.push(ToolCall {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                });
            }
            let call = &mut self.calls[index];
            if let Some(id) = d["id"].as_str().filter(|s| !s.is_empty()) {
                call.id = id.to_string();
            }
            if let Some(name) = d["function"]["name"].as_str() {
                call.name.push_str(name);
            }
            match &d["function"]["arguments"] {
                serde_json::Value::String(s) => call.arguments.push_str(s),
                serde_json::Value::Null => {}
                other => call.arguments.push_str(&other.to_string()),
            }
        }
    }

    fn finish(mut self) -> Vec<ToolCall> {
        self.calls.retain(|c| !c.name.is_empty());
        for (i, c) in self.calls.iter_mut().enumerate() {
            if c.id.is_empty() {
                c.id = format!("call_{}_{}", uuid::Uuid::new_v4().simple(), i);
            }
        }
        self.calls
    }
}

fn usage_from(v: &serde_json::Value, usage: &mut Usage) {
    if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
        if let Some(p) = u["prompt_tokens"].as_u64() {
            usage.prompt_tokens = p as u32;
        }
        if let Some(c) = u["completion_tokens"].as_u64() {
            usage.completion_tokens = c as u32;
        }
        if let Some(c) = u["prompt_tokens_details"]["cached_tokens"].as_u64() {
            usage.cached_tokens = c as u32;
        }
    }
    if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
        if let Some(ms) = t["prompt_ms"].as_f64() {
            usage.prompt_ms = ms.max(0.0) as u64;
        }
        if let Some(ms) = t["predicted_ms"].as_f64() {
            usage.generation_ms = ms.max(0.0) as u64;
        }
        if usage.completion_tokens == 0 {
            if let Some(n) = t["predicted_n"].as_u64() {
                usage.completion_tokens = n as u32;
            }
        }
        if usage.prompt_tokens == 0 {
            let n = t["prompt_n"].as_u64().unwrap_or(0) + t["cache_n"].as_u64().unwrap_or(0);
            usage.prompt_tokens = n as u32;
        }
        if usage.cached_tokens == 0 {
            if let Some(n) = t["cache_n"].as_u64() {
                usage.cached_tokens = n as u32;
            }
        }
    }
}

fn openai_stream(resp: reqwest::Response, chat: bool) -> TokenStream {
    let out = async_stream::stream! {
        let mut decoder = SseDecoder::new();
        let mut bytes = resp.bytes_stream();
        let mut tools = ToolAccumulator::default();
        let mut usage = Usage::default();
        let mut finish: Option<String> = None;
        let mut finished = false;
        let mut ended = false;
        'outer: while !ended {
            let events = match bytes.next().await {
                Some(Ok(chunk)) => decoder.push(&chunk),
                Some(Err(e)) => { yield Err(Ml5Error::Backend(format!("model runtime stream error: {e}"))); return; }
                None => { ended = true; decoder.finish() }
            };
            for data in events {
                let data = data.trim();
                if data == "[DONE]" { finished = true; break 'outer; }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else { continue };
                if let Some(err) = v.get("error").filter(|e| !e.is_null()) {
                    let msg = err["message"].as_str().or_else(|| err.as_str()).unwrap_or("inference failed").to_string();
                    yield Err(Ml5Error::Backend(msg));
                    return;
                }
                if let Some(p) = v.get("prompt_progress").filter(|p| p.is_object()) {
                    let total = p["total"].as_u64().unwrap_or(0) as usize;
                    let processed = p["processed"].as_u64().unwrap_or(0) as usize;
                    yield Ok(StreamChunk {
                        progress: Some(InferenceProgress { stage: "processing_prompt".into(), completed: processed, total }),
                        ..Default::default()
                    });
                }
                usage_from(&v, &mut usage);
                let Some(choice) = v["choices"].get(0) else { continue };
                let (text, thinking) = if chat {
                    let delta = &choice["delta"];
                    tools.push(&delta["tool_calls"]);
                    (
                        delta["content"].as_str().unwrap_or("").to_string(),
                        delta["reasoning_content"].as_str().unwrap_or("").to_string(),
                    )
                } else {
                    (choice["text"].as_str().unwrap_or("").to_string(), String::new())
                };
                if let Some(reason) = choice["finish_reason"].as_str() {
                    finish = Some(reason.to_string());
                }
                if !text.is_empty() || !thinking.is_empty() {
                    yield Ok(StreamChunk { text, thinking, ..Default::default() });
                }
            }
        }
        if !finished && finish.is_none() {
            yield Err(Ml5Error::Backend("the model runtime ended the response early; retry the request".into()));
            return;
        }
        let tool_calls = tools.finish();
        let finish_reason = if !tool_calls.is_empty() {
            Some("tool_calls".to_string())
        } else {
            Some(finish.unwrap_or_else(|| "stop".into()))
        };
        yield Ok(StreamChunk {
            done: true,
            usage: Some(usage),
            tool_calls,
            finish_reason,
            ..Default::default()
        });
    };
    out.boxed()
}

#[async_trait]
impl Backend for LlamaServerBackend {
    fn name(&self) -> &str {
        "llama-server"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Chat, Capability::Generate, Capability::Embed, Capability::Vision, Capability::Tools, Capability::Thinking]
    }

    async fn load(&self, spec: &LoadSpec) -> Result<()> {
        self.unload().await?;
        self.opt_cache_prompt
            .store(spec.params.opt_cache_prompt && spec.params.cache_prompt, Ordering::Relaxed);
        let server = runtime::ensure_runtime(self.server_override.as_deref(), |msg| tracing::info!("{msg}")).await?;
        std::fs::create_dir_all(&self.log_dir)?;
        let log_path = self.log_dir.join(log_file_name(&spec.model.name));
        let log = std::fs::File::create(&log_path)?;
        let log_err = log.try_clone()?;
        let port = Self::free_port()?;
        let mut cmd = Self::build_command(&server, spec, port);
        cmd.stdout(Stdio::from(log)).stderr(Stdio::from(log_err));
        tracing::info!(
            model = %spec.model.name,
            server = %server.display(),
            port,
            ctx = spec.params.n_ctx,
            gpu_layers = %spec.params.n_gpu_layers,
            mmproj = spec.model.mmproj.is_some(),
            "starting llama-server"
        );
        let child = cmd
            .spawn()
            .map_err(|e| Ml5Error::Backend(format!("failed to start {}: {e}", server.display())))?;
        #[cfg(windows)]
        let job = job_guard(&child);
        *self.instance.lock().await = Some(Instance {
            child,
            port,
            log_path: log_path.clone(),
            #[cfg(windows)]
            _job: job,
        });
        if let Err(e) = self.wait_ready(port, &log_path).await {
            self.unload().await.ok();
            return Err(e);
        }
        if let Ok(mut b) = self.base.write() {
            *b = Some(format!("http://127.0.0.1:{port}"));
        }
        tracing::info!(model = %spec.model.name, port, "llama-server ready");
        Ok(())
    }

    async fn unload(&self) -> Result<()> {
        if let Ok(mut b) = self.base.write() {
            *b = None;
        }
        let Some(mut inst) = self.instance.lock().await.take() else {
            return Ok(());
        };
        if terminate(&inst.child) {
            match tokio::time::timeout(STOP_GRACE, inst.child.wait()).await {
                Ok(_) => {}
                Err(_) => {
                    let _ = inst.child.kill().await;
                    let _ = inst.child.wait().await;
                }
            }
        } else {
            let _ = inst.child.kill().await;
            let _ = inst.child.wait().await;
        }
        tracing::info!(port = inst.port, log = %inst.log_path.display(), "llama-server stopped");
        Ok(())
    }

    async fn is_alive(&self) -> bool {
        let mut guard = self.instance.lock().await;
        match guard.as_mut() {
            Some(inst) => matches!(inst.child.try_wait(), Ok(None)),
            None => false,
        }
    }

    fn base_url(&self) -> Option<String> {
        self.base.read().ok().and_then(|b| b.clone())
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream> {
        let mut body = req.to_openai();
        self.apply_cache_prompt(&mut body, req.cache_prompt);
        let resp = self.post_stream("/v1/chat/completions", &body).await?;
        Ok(openai_stream(resp, true))
    }

    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream> {
        let mut body = serde_json::Map::new();
        body.insert("model".into(), req.model.clone().into());
        body.insert("prompt".into(), req.prompt.clone().into());
        body.insert("stream".into(), true.into());
        body.insert("stream_options".into(), serde_json::json!({ "include_usage": true }));
        body.insert("return_progress".into(), true.into());
        req.params.apply_openai(&mut body);
        let mut body = serde_json::Value::Object(body);
        self.apply_cache_prompt(&mut body, req.cache_prompt);
        let resp = self
            .post_stream("/v1/completions", &body)
            .await?;
        Ok(openai_stream(resp, false))
    }

    async fn count_tokens(&self, req: &ChatRequest) -> Result<u32> {
        let mut body = req.to_openai();
        if let Some(o) = body.as_object_mut() {
            o.remove("stream");
            o.remove("stream_options");
            o.remove("return_progress");
        }
        let templated: serde_json::Value = self
            .post_stream("/apply-template", &body)
            .await?
            .json()
            .await
            .map_err(|e| Ml5Error::Backend(format!("bad template response: {e}")))?;
        let prompt = templated["prompt"].as_str().unwrap_or_default();
        let tokens: serde_json::Value = self
            .post_stream(
                "/tokenize",
                &serde_json::json!({ "content": prompt, "add_special": true, "parse_special": true }),
            )
            .await?
            .json()
            .await
            .map_err(|e| Ml5Error::Backend(format!("bad tokenize response: {e}")))?;
        Ok(tokens["tokens"].as_array().map(|t| t.len()).unwrap_or(0) as u32)
    }

    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
        let body = serde_json::json!({ "model": req.model, "input": req.input });
        let resp = self.post_stream("/v1/embeddings", &body).await?;
        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| Ml5Error::Backend(format!("bad embedding response: {e}")))?;
        let mut items: Vec<(usize, Vec<f32>)> = v["data"]
            .as_array()
            .map(|data| {
                data.iter()
                    .enumerate()
                    .map(|(i, item)| {
                        (
                            item["index"].as_u64().map(|x| x as usize).unwrap_or(i),
                            item["embedding"]
                                .as_array()
                                .map(|e| e.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                                .unwrap_or_default(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        items.sort_by_key(|(i, _)| *i);
        Ok(EmbedResponse {
            embeddings: items.into_iter().map(|(_, e)| e).collect(),
            usage: Usage {
                prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                ..Default::default()
            },
        })
    }
}
