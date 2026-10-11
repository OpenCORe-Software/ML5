use super::common::*;
use crate::state::AppState;
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use ml5_core::error::Ml5Error;
use ml5_core::model::{ModelInfo, ModelType};
use ml5_core::pull::PullEvent;
use ml5_core::types::*;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::convert::Infallible;

type ApiError = (StatusCode, Json<Value>);

fn err_response(e: Ml5Error) -> ApiError {
    (status_of(&e), Json(json!({ "error": e.to_string() })))
}

fn ndjson_response<S>(stream: S) -> axum::response::Response
where
    S: futures::Stream<Item = Result<Value, Infallible>> + Send + 'static,
{
    let body = Body::from_stream(stream.map(|r| r.map(|v| format!("{v}\n"))));
    let mut resp = axum::response::Response::new(body);
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    resp
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

fn with_usage(mut v: Value, u: &Usage, reason: &str) -> Value {
    if let Some(o) = v.as_object_mut() {
        o.insert("done_reason".into(), reason.into());
        o.insert("prompt_eval_count".into(), u.prompt_tokens.into());
        o.insert("eval_count".into(), u.completion_tokens.into());
        o.insert("prompt_eval_duration".into(), (u.prompt_ms * 1_000_000).into());
        o.insert("eval_duration".into(), (u.generation_ms * 1_000_000).into());
        o.insert("load_duration".into(), 0.into());
        o.insert("total_duration".into(), ((u.prompt_ms + u.generation_ms) * 1_000_000).into());
    }
    v
}

fn done_reason(finish: &str) -> &'static str {
    match finish {
        "length" => "length",
        _ => "stop",
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub num_predict: Option<i64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<i64>,
    pub stop: Option<Vec<String>>,
    pub num_ctx: Option<u32>,
    pub num_batch: Option<u32>,
    pub num_gpu: Option<i32>,
    pub num_thread: Option<i32>,
}

impl OllamaOptions {
    fn to_sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            max_tokens: self.num_predict.filter(|n| *n > 0).map(|n| n.min(u32::MAX as i64) as u32),
            stop: self.stop.clone().unwrap_or_default().into_iter().filter(|s| !s.is_empty()).collect(),
            seed: self.seed.filter(|s| *s >= 0).map(|s| s as u32),
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            penalty_last_n: self.repeat_last_n,
        }
    }

    fn to_overrides(&self) -> RequestOverrides {
        RequestOverrides {
            n_ctx: self.num_ctx,
            n_batch: self.num_batch,
            n_gpu_layers: self.num_gpu.map(ml5_core::config::GpuLayers::from_request),
            n_threads: self.num_thread,
            ..Default::default()
        }
    }
}

fn think_of(v: &Option<Value>) -> (Option<bool>, Option<String>) {
    match v {
        Some(Value::Bool(b)) => (Some(*b), None),
        Some(Value::String(level)) => (Some(true), Some(level.clone())),
        _ => (None, None),
    }
}

fn wants_unload(keep_alive: &Option<Value>) -> bool {
    keep_alive.as_ref().and_then(parse_keep_alive) == Some(0)
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateBody {
    pub model: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub images: Option<Vec<String>>,
    #[serde(default)]
    pub raw: bool,
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub think: Option<Value>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatBody {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub format: Option<Value>,
    #[serde(default)]
    pub think: Option<Value>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
}

fn ollama_message(v: &Value) -> Result<Message, Ml5Error> {
    let mut m = openai_message(v)?;
    if let Some(images) = v["images"].as_array() {
        m.images
            .extend(images.iter().filter_map(|i| i.as_str()).map(str::to_string));
    }
    if m.role == "tool" && m.name.is_none() {
        m.name = v["tool_name"].as_str().map(str::to_string);
    }
    Ok(m)
}

fn ollama_tool_calls(calls: &[ToolCall]) -> Value {
    calls
        .iter()
        .map(|t| json!({ "id": t.id, "function": { "name": t.name, "arguments": t.arguments_value() } }))
        .collect()
}

pub async fn generate(
    State(state): State<AppState>,
    Json(body): Json<OllamaGenerateBody>,
) -> Result<axum::response::Response, ApiError> {
    let prompt = body.prompt.clone().unwrap_or_default();
    if prompt.is_empty() {
        if wants_unload(&body.keep_alive) {
            state.engine.unload_model(&body.model).await.map_err(err_response)?;
        } else {
            state
                .engine
                .preload(&body.model, &RequestOverrides::default(), body.keep_alive.as_ref().and_then(parse_keep_alive))
                .await
                .map_err(err_response)?;
        }
        return Ok(Json(json!({
            "model": body.model,
            "created_at": now_rfc3339(),
            "response": "",
            "done": true,
            "done_reason": if wants_unload(&body.keep_alive) { "unload" } else { "load" },
        }))
        .into_response());
    }

    let streaming = body.stream.unwrap_or(true);
    let opts = body.options.unwrap_or_default();
    let keep_alive_secs = body.keep_alive.as_ref().and_then(parse_keep_alive);
    let images = body.images.unwrap_or_default();
    let model = body.model.clone();
    let mut stream = if body.raw {
        state
            .engine
            .generate(GenerateRequest {
                model: body.model.clone(),
                prompt,
                params: opts.to_sampling(),
                overrides: opts.to_overrides(),
                keep_alive_secs,
            })
            .await
            .map_err(err_response)?
    } else {
        let (think, effort) = think_of(&body.think);
        let mut messages = Vec::new();
        if let Some(sys) = body.system.filter(|s| !s.is_empty()) {
            messages.push(Message::text("system", sys));
        }
        messages.push(Message {
            role: "user".into(),
            content: prompt,
            images,
            ..Default::default()
        });
        state
            .engine
            .chat(ChatRequest {
                model: body.model.clone(),
                messages,
                params: opts.to_sampling(),
                overrides: opts.to_overrides(),
                think,
                reasoning_effort: effort,
                response_format: body.format.as_ref().and_then(response_format_from_ollama),
                keep_alive_secs,
                ..Default::default()
            })
            .await
            .map_err(err_response)?
    };

    if !streaming {
        let c = collect(&mut stream).await.map_err(err_response)?;
        let mut v = json!({ "model": model, "created_at": now_rfc3339(), "response": c.text, "done": true });
        if !c.thinking.is_empty() {
            v["thinking"] = c.thinking.into();
        }
        return Ok(Json(with_usage(v, &c.usage, done_reason(&c.finish_reason))).into_response());
    }

    let lines = async_stream::stream! {
        while let Some(item) = stream.next().await {
            match item {
                Ok(c) if c.progress.is_some() => continue,
                Ok(c) => {
                    let mut v = json!({ "model": model, "created_at": now_rfc3339(), "response": c.text, "done": c.done });
                    if !c.thinking.is_empty() {
                        v["thinking"] = c.thinking.into();
                    }
                    if c.done {
                        let reason = done_reason(c.finish_reason.as_deref().unwrap_or("stop"));
                        yield Ok(with_usage(v, &c.usage.unwrap_or_default(), reason));
                        return;
                    }
                    yield Ok(v);
                }
                Err(e) => {
                    yield Ok(json!({ "error": e.to_string() }));
                    return;
                }
            }
        }
        yield Ok(json!({ "error": "Inference ended unexpectedly; retry the request." }));
    };
    Ok(ndjson_response(lines))
}

pub async fn chat(
    State(state): State<AppState>,
    Json(body): Json<OllamaChatBody>,
) -> Result<axum::response::Response, ApiError> {
    if body.messages.is_empty() {
        if wants_unload(&body.keep_alive) {
            state.engine.unload_model(&body.model).await.map_err(err_response)?;
        } else {
            state
                .engine
                .preload(&body.model, &RequestOverrides::default(), body.keep_alive.as_ref().and_then(parse_keep_alive))
                .await
                .map_err(err_response)?;
        }
        return Ok(Json(json!({
            "model": body.model,
            "created_at": now_rfc3339(),
            "message": { "role": "assistant", "content": "" },
            "done": true,
            "done_reason": if wants_unload(&body.keep_alive) { "unload" } else { "load" },
        }))
        .into_response());
    }

    let streaming = body.stream.unwrap_or(true);
    let opts = body.options.unwrap_or_default();
    let messages = body
        .messages
        .iter()
        .map(ollama_message)
        .collect::<Result<Vec<_>, _>>()
        .map_err(err_response)?;
    let (think, effort) = think_of(&body.think);
    let req = ChatRequest {
        model: body.model.clone(),
        messages,
        params: opts.to_sampling(),
        overrides: opts.to_overrides(),
        tools: non_empty_tools(body.tools),
        think,
        reasoning_effort: effort,
        response_format: body.format.as_ref().and_then(response_format_from_ollama),
        keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
        ..Default::default()
    };
    let mut stream = state.engine.chat(req).await.map_err(err_response)?;
    let model = body.model.clone();

    if !streaming {
        let c = collect(&mut stream).await.map_err(err_response)?;
        let mut message = json!({ "role": "assistant", "content": c.text });
        if !c.thinking.is_empty() {
            message["thinking"] = c.thinking.into();
        }
        if !c.tool_calls.is_empty() {
            message["tool_calls"] = ollama_tool_calls(&c.tool_calls);
        }
        let v = json!({ "model": model, "created_at": now_rfc3339(), "message": message, "done": true });
        return Ok(Json(with_usage(v, &c.usage, done_reason(&c.finish_reason))).into_response());
    }

    let lines = async_stream::stream! {
        while let Some(item) = stream.next().await {
            match item {
                Ok(c) if c.progress.is_some() => continue,
                Ok(c) => {
                    if c.done && !c.tool_calls.is_empty() {
                        yield Ok(json!({
                            "model": model, "created_at": now_rfc3339(),
                            "message": { "role": "assistant", "content": "", "tool_calls": ollama_tool_calls(&c.tool_calls) },
                            "done": false
                        }));
                    }
                    let mut message = json!({ "role": "assistant", "content": c.text });
                    if !c.thinking.is_empty() {
                        message["thinking"] = c.thinking.into();
                    }
                    let v = json!({ "model": model, "created_at": now_rfc3339(), "message": message, "done": c.done });
                    if c.done {
                        let reason = done_reason(c.finish_reason.as_deref().unwrap_or("stop"));
                        yield Ok(with_usage(v, &c.usage.unwrap_or_default(), reason));
                        return;
                    }
                    yield Ok(v);
                }
                Err(e) => {
                    yield Ok(json!({ "error": e.to_string() }));
                    return;
                }
            }
        }
        yield Ok(json!({ "error": "Inference ended unexpectedly; retry the request." }));
    };
    Ok(ndjson_response(lines))
}

fn digest_of(m: &ModelInfo) -> String {
    use sha2::Digest;
    let modified = m.modified_at.map(|t| t.timestamp()).unwrap_or(0);
    let raw = format!("{}|{}|{}", m.path.display(), m.size_bytes, modified);
    hex::encode(sha2::Sha256::digest(raw.as_bytes()))
}

fn details_of(m: &ModelInfo) -> Value {
    let d = m.details.clone().unwrap_or_default();
    json!({
        "parent_model": "",
        "format": match m.model_type { ModelType::Gguf => "gguf", ModelType::Safetensors => "safetensors" },
        "family": d.architecture,
        "families": if d.architecture.is_empty() { Value::Null } else { json!([d.architecture]) },
        "parameter_size": d.parameter_size,
        "quantization_level": d.quantization
    })
}

fn ollama_caps(m: &ModelInfo) -> Vec<&'static str> {
    let mut caps = Vec::new();
    if m.has(Capability::Chat) {
        caps.push("completion");
    }
    if m.has(Capability::Embed) {
        caps.push("embedding");
    }
    if m.has(Capability::Tools) {
        caps.push("tools");
    }
    if m.has(Capability::Thinking) {
        caps.push("thinking");
    }
    if m.has(Capability::Vision) {
        caps.push("vision");
    }
    caps
}

pub async fn tags(State(state): State<AppState>) -> impl IntoResponse {
    let models: Vec<Value> = state
        .engine
        .list_models()
        .await
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "model": m.name,
                "modified_at": m.modified_at.map(|t| t.to_rfc3339()).unwrap_or_default(),
                "size": m.size_bytes,
                "digest": digest_of(m),
                "details": details_of(m),
            })
        })
        .collect();
    Json(json!({ "models": models }))
}

#[derive(Debug, Deserialize)]
pub struct ShowBody {
    pub model: Option<String>,
    pub name: Option<String>,
}

pub async fn show(
    State(state): State<AppState>,
    Json(body): Json<ShowBody>,
) -> Result<impl IntoResponse, ApiError> {
    let name = body
        .model
        .or(body.name)
        .ok_or_else(|| err_response(Ml5Error::InvalidRequest("missing model".into())))?;
    let m = state.engine.get_model(&name).await.map_err(err_response)?;
    let d = m.details.clone().unwrap_or_default();
    let mut info = Map::new();
    info.insert("general.architecture".into(), d.architecture.clone().into());
    info.insert("general.parameter_count".into(), Value::Null);
    if let Some(n) = d.context_length {
        info.insert(format!("{}.context_length", d.architecture), n.into());
    }
    if let Some(n) = d.block_count {
        info.insert(format!("{}.block_count", d.architecture), n.into());
    }
    if let Some(n) = d.embedding_length {
        info.insert(format!("{}.embedding_length", d.architecture), n.into());
    }
    Ok(Json(json!({
        "license": "",
        "modelfile": format!("FROM {}", m.path.display()),
        "parameters": "",
        "template": "{{ .Prompt }}",
        "modified_at": m.modified_at.map(|t| t.to_rfc3339()).unwrap_or_default(),
        "details": details_of(&m),
        "capabilities": ollama_caps(&m),
        "model_info": info,
    })))
}

pub async fn version() -> impl IntoResponse {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

#[derive(Debug, Deserialize)]
pub struct OllamaEmbedBody {
    pub model: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
}

async fn run_embed(state: &AppState, body: OllamaEmbedBody) -> Result<EmbedResponse, ApiError> {
    let input: Vec<String> = match body.input {
        Some(Value::String(s)) => vec![s],
        Some(Value::Array(a)) => a.into_iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => body.prompt.into_iter().collect(),
    };
    state
        .engine
        .embed(EmbedRequest {
            model: body.model,
            input,
            keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
        })
        .await
        .map_err(err_response)
}

pub async fn embeddings(
    State(state): State<AppState>,
    Json(body): Json<OllamaEmbedBody>,
) -> Result<impl IntoResponse, ApiError> {
    let resp = run_embed(&state, body).await?;
    Ok(Json(json!({ "embedding": resp.embeddings.into_iter().next().unwrap_or_default() })))
}

pub async fn embed(
    State(state): State<AppState>,
    Json(body): Json<OllamaEmbedBody>,
) -> Result<impl IntoResponse, ApiError> {
    let model = body.model.clone();
    let resp = run_embed(&state, body).await?;
    Ok(Json(json!({
        "model": model,
        "embeddings": resp.embeddings,
        "prompt_eval_count": resp.usage.prompt_tokens
    })))
}

pub async fn ps(State(state): State<AppState>) -> impl IntoResponse {
    let status = state.engine.status().await;
    let mut models = Vec::new();
    for l in &status.loaded {
        let Ok(m) = state.engine.get_model(&l.name).await else { continue };
        let expires = match l.expires_in_secs {
            Some(s) => (chrono::Utc::now() + chrono::Duration::seconds(s)).to_rfc3339(),
            None => "2318-01-01T00:00:00Z".to_string(),
        };
        models.push(json!({
            "name": l.name,
            "model": l.name,
            "size": l.memory_bytes,
            "digest": digest_of(&m),
            "details": details_of(&m),
            "expires_at": expires,
            "size_vram": if l.gpu_layers == "0" { 0 } else { m.size_bytes },
            "context_length": l.n_ctx,
        }));
    }
    Json(json!({ "models": models }))
}

#[derive(Debug, Deserialize)]
pub struct OllamaPullBody {
    pub model: Option<String>,
    pub name: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
}

fn pull_status(ev: &PullEvent) -> Option<Value> {
    Some(match ev {
        PullEvent::Started { .. } => json!({ "status": "pulling manifest" }),
        PullEvent::Resolving { target } => json!({ "status": format!("pulling manifest for {target}") }),
        PullEvent::Downloading { file } => json!({ "status": format!("pulling {file}"), "digest": file }),
        PullEvent::Progress { downloaded, total } => json!({
            "status": "downloading",
            "completed": downloaded,
            "total": total.unwrap_or(0),
        }),
        PullEvent::Verifying { .. } => json!({ "status": "verifying sha256 digest" }),
        PullEvent::Done { .. } => json!({ "status": "success" }),
        PullEvent::Error { error } => json!({ "error": error }),
    })
}

pub async fn pull(
    State(state): State<AppState>,
    Json(body): Json<OllamaPullBody>,
) -> Result<axum::response::Response, ApiError> {
    let name = body
        .model
        .or(body.name)
        .ok_or_else(|| err_response(Ml5Error::InvalidRequest("missing model name".into())))?;
    let (registry, reference) = match name
        .strip_prefix("hf.co/")
        .or_else(|| name.strip_prefix("huggingface.co/"))
    {
        Some(rest) => ("hf", rest.to_string()),
        None if name.contains('/') => ("hf", name.clone()),
        None => ("routes", name.clone()),
    };
    let source = super::native::resolve_source(registry, &reference)
        .await
        .map_err(err_response)?;
    let handle = super::native::start_pull(state.engine.clone(), source, None, None).await;
    let mut rx = handle.tx.subscribe();

    if !body.stream.unwrap_or(true) {
        loop {
            match rx.recv().await {
                Ok(PullEvent::Done { .. }) => return Ok(Json(json!({ "status": "success" })).into_response()),
                Ok(PullEvent::Error { error }) => {
                    return Err(err_response(Ml5Error::Download(error)));
                }
                Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => return Err(err_response(Ml5Error::Download("pull ended unexpectedly".into()))),
            }
        }
    }

    let lines = async_stream::stream! {
        yield Ok(json!({ "status": "pulling manifest" }));
        loop {
            let ev = match rx.recv().await {
                Ok(ev) => ev,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => {
                    yield Ok(json!({ "error": "pull ended unexpectedly; retry" }));
                    return;
                }
            };
            let done = matches!(ev, PullEvent::Done { .. } | PullEvent::Error { .. });
            if let Some(v) = pull_status(&ev) {
                yield Ok(v);
            }
            if done {
                return;
            }
        }
    };
    Ok(ndjson_response(lines))
}

#[derive(Debug, Deserialize)]
pub struct OllamaDeleteBody {
    pub model: Option<String>,
    pub name: Option<String>,
}

pub async fn delete(
    State(state): State<AppState>,
    Json(body): Json<OllamaDeleteBody>,
) -> Result<impl IntoResponse, ApiError> {
    let name = body
        .model
        .or(body.name)
        .ok_or_else(|| err_response(Ml5Error::InvalidRequest("missing model name".into())))?;
    state.engine.delete_model(&name).await.map_err(err_response)?;
    Ok(StatusCode::OK.into_response())
}

#[derive(Debug, Deserialize)]
pub struct OllamaCopyBody {
    pub source: String,
    pub destination: String,
}

pub async fn copy(
    State(state): State<AppState>,
    Json(body): Json<OllamaCopyBody>,
) -> Result<impl IntoResponse, ApiError> {
    let info = state.engine.get_model(&body.source).await.map_err(err_response)?;
    if info.model_type != ModelType::Gguf || info.files.len() > 1 {
        return Err(err_response(Ml5Error::InvalidRequest(
            "copy is only supported for single-file GGUF models".into(),
        )));
    }
    let safe: String = body
        .destination
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '-' })
        .collect();
    let dir = info.path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let dest = dir.join(format!("{safe}.gguf"));
    if dest.exists() {
        return Err(err_response(Ml5Error::InvalidRequest(format!(
            "'{}' already exists",
            body.destination
        ))));
    }
    tokio::fs::copy(&info.path, &dest)
        .await
        .map_err(|e| err_response(Ml5Error::Backend(format!("copy failed: {e}"))))?;
    if let Some(src_meta) = &info.meta_path {
        let mut meta = ml5_core::model::ModelMeta::load(src_meta);
        meta.friendly_name = None;
        if meta.mmproj_file.is_none() {
            meta.mmproj_file = info
                .mmproj
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().into_owned());
        }
        let _ = meta.save(&ml5_core::model::gguf_meta_path(&dest));
    }
    state.engine.scan_models().await.map_err(err_response)?;
    Ok(StatusCode::OK.into_response())
}
