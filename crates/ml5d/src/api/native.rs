use super::common::*;
use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use ml5_core::engine::PullHandle;
use ml5_core::error::Ml5Error;
use ml5_core::pull::{PullEvent, PullSource};
use ml5_core::types::*;
use ml5_core::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;

type ApiError = (StatusCode, Json<Value>);

fn err_response(e: Ml5Error) -> ApiError {
    (status_of(&e), Json(json!({ "error": e.to_string() })))
}

fn event(value: Value) -> Result<Event, Infallible> {
    Ok(Event::default().data(value.to_string()))
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct ChatBody {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub think: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default, alias = "format")]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
    #[serde(default)]
    pub cache_prompt: Option<bool>,
    #[serde(flatten)]
    pub overrides: RequestOverrides,
    #[serde(default = "default_true")]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub model: String,
    pub message: Message,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct SamplingFields {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub stop: Vec<String>,
}

impl SamplingFields {
    fn into_params(self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            max_tokens: self.max_tokens,
            stop: self.stop.into_iter().filter(|s| !s.is_empty()).collect(),
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            penalty_last_n: None,
        }
    }
}

pub async fn chat(
    State(state): State<AppState>,
    Json(body): Json<ChatBody>,
) -> Result<axum::response::Response, ApiError> {
    state.engine.get_model(&body.model).await.map_err(err_response)?;
    let response_format = body.response_format.as_ref().and_then(|f| match f {
        Value::String(_) => response_format_from_ollama(f),
        Value::Object(o) if o.contains_key("type") => Some(f.clone()),
        Value::Object(_) => response_format_from_ollama(f),
        _ => None,
    });
    let req = ChatRequest {
        model: body.model.clone(),
        messages: body.messages,
        params: body.sampling.into_params(),
        overrides: body.overrides,
        tools: non_empty_tools(body.tools),
        tool_choice: body.tool_choice,
        parallel_tool_calls: None,
        think: body.think,
        reasoning_effort: body.reasoning_effort,
        response_format,
        keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
        cache_prompt: body.cache_prompt,
    };
    native_response(state.engine, body.model, body.stream, Work::Chat(req)).await
}

enum Work {
    Chat(ChatRequest),
    Generate(GenerateRequest),
}

fn chunk_value(model: &str, chat: bool, c: StreamChunk) -> Value {
    if let Some(progress) = c.progress {
        return json!({ "status": progress.stage, "completed": progress.completed, "total": progress.total });
    }
    if chat {
        serde_json::to_value(ChatChunk {
            model: model.into(),
            message: Message {
                role: "assistant".into(),
                content: c.text,
                tool_calls: c.tool_calls,
                thinking: (!c.thinking.is_empty()).then_some(c.thinking),
                ..Default::default()
            },
            done: c.done,
            usage: c.usage,
            finish_reason: c.finish_reason,
        })
        .unwrap_or_else(|_| json!({ "error": "serialization failed" }))
    } else {
        json!({ "model": model, "response": c.text, "done": c.done, "usage": c.usage, "finish_reason": c.finish_reason })
    }
}

async fn native_response(
    engine: Arc<Engine>,
    model: String,
    streaming: bool,
    work: Work,
) -> Result<axum::response::Response, ApiError> {
    let is_chat = matches!(work, Work::Chat(_));
    let was_loaded = engine.is_loaded(&model).await;
    let defaults = engine.config.model.clone();
    let start = {
        let engine = engine.clone();
        async move {
            match work {
                Work::Chat(req) => engine.chat(req).await,
                Work::Generate(req) => engine.generate(req).await,
            }
        }
    };
    if !streaming {
        let mut stream = start.await.map_err(err_response)?;
        let c = collect(&mut stream).await.map_err(err_response)?;
        let done = StreamChunk {
            text: c.text,
            thinking: c.thinking,
            tool_calls: c.tool_calls,
            done: true,
            usage: Some(c.usage),
            progress: None,
            finish_reason: Some(c.finish_reason),
        };
        return Ok(Json(chunk_value(&model, is_chat, done)).into_response());
    }
    let stream = async_stream::stream! {
        yield event(json!({ "status": if was_loaded { "queued" } else { "loading_model" }, "model": model, "defaults": defaults }));
        let mut stream = match start.await {
            Ok(stream) => stream,
            Err(e) => { yield event(json!({ "error": e.to_string() })); return; }
        };
        yield event(json!({ "status": "queued", "message": "Model ready; waiting for prompt processing" }));
        let mut done = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(c) => {
                    done = c.done;
                    yield event(chunk_value(&model, is_chat, c));
                    if done { break; }
                }
                Err(e) => { yield event(json!({ "error": e.to_string() })); return; }
            }
        }
        if !done {
            yield event(json!({ "error": "Inference ended unexpectedly; retry the request." }));
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

#[derive(Debug, Deserialize)]
pub struct GenerateBody {
    pub model: String,
    pub prompt: String,
    #[serde(flatten)]
    pub sampling: SamplingFields,
    #[serde(default)]
    pub keep_alive: Option<Value>,
    #[serde(default)]
    pub cache_prompt: Option<bool>,
    #[serde(flatten)]
    pub overrides: RequestOverrides,
    #[serde(default = "default_true")]
    pub stream: bool,
}

pub async fn generate(
    State(state): State<AppState>,
    Json(body): Json<GenerateBody>,
) -> Result<axum::response::Response, ApiError> {
    state.engine.get_model(&body.model).await.map_err(err_response)?;
    let req = GenerateRequest {
        model: body.model.clone(),
        prompt: body.prompt,
        params: body.sampling.into_params(),
        overrides: body.overrides,
        keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
        cache_prompt: body.cache_prompt,
    };
    native_response(state.engine, body.model, body.stream, Work::Generate(req)).await
}

#[derive(Debug, Deserialize)]
pub struct EmbedBody {
    pub model: String,
    pub input: Vec<String>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
}

pub async fn embed(
    State(state): State<AppState>,
    Json(body): Json<EmbedBody>,
) -> Result<impl IntoResponse, ApiError> {
    let model = body.model.clone();
    let resp = state
        .engine
        .embed(EmbedRequest {
            model: body.model,
            input: body.input,
            keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
        })
        .await
        .map_err(err_response)?;
    Ok(Json(json!({ "model": model, "embeddings": resp.embeddings, "usage": resp.usage })))
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let loaded_contexts = state.engine.loaded_contexts().await;
    let models: Vec<Value> = state
        .engine
        .list_models()
        .await
        .into_iter()
        .map(|m| {
            let loaded = loaded_contexts.get(&m.name);
            json!({
                "name": m.name,
                "friendly_name": m.friendly_name,
                "display_name": m.display_name(),
                "size_bytes": m.size_bytes,
                "capabilities": m.capabilities,
                "format": m.model_type,
                "modified_at": m.modified_at,
                "vision": m.mmproj.is_some(),
                "mmproj": m.mmproj.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()),
                "details": m.details,
                "loaded": loaded.is_some(),
                "n_ctx": loaded,
            })
        })
        .collect();
    Json(json!({ "models": models }))
}

#[derive(Debug, Deserialize)]
pub struct PullBody {
    pub model: String,
    #[serde(default = "default_registry")]
    pub registry: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub quant: Option<String>,
}

fn default_registry() -> String {
    "hf".into()
}

pub async fn resolve_source(registry: &str, model: &str) -> Result<PullSource, Ml5Error> {
    if registry != "routes" {
        return ml5_core::pull::parse_pull_target(registry, model);
    }
    match ml5_core::pull::resolve_via_routes(model).await {
        Ok(Some(source)) => Ok(source),
        Ok(None) => ml5_core::pull::parse_pull_target("core", model),
        Err(e) => {
            tracing::warn!("routes resolution failed, falling back to core: {e}");
            ml5_core::pull::parse_pull_target("core", model)
        }
    }
}

pub async fn start_pull(
    engine: Arc<Engine>,
    source: PullSource,
    token: Option<String>,
    quant: Option<String>,
) -> PullHandle {
    let key = format!("{source:?}|{}", quant.as_deref().unwrap_or(""));
    let (handle, fresh) = engine.register_pull(&key, &source.describe()).await;
    if !fresh {
        return handle;
    }
    let task = handle.clone();
    tokio::spawn(async move {
        let models_dir = engine.config.models_dir.clone();
        let result = ml5_core::pull::pull(
            &source,
            &models_dir,
            token.as_deref(),
            quant.as_deref(),
            &task.cancel,
            |ev| {
                match &ev {
                    PullEvent::Resolving { target } => tracing::info!(%target, "resolving model"),
                    PullEvent::Downloading { file } => tracing::info!(%file, "downloading"),
                    PullEvent::Verifying { file } => tracing::info!(%file, "verifying"),
                    _ => {}
                }
                task.send(ev);
            },
        )
        .await;
        match result {
            Ok((name, path)) => {
                tracing::info!(%name, path = %path.display(), "model pulled");
                match engine.scan_models().await {
                    Ok(()) => task.send(PullEvent::Done { model: name }),
                    Err(e) => task.send(PullEvent::Error {
                        error: format!("Downloaded but could not register model: {e}"),
                    }),
                }
            }
            Err(e) => {
                tracing::error!(%e, target = %source.describe(), "pull failed");
                task.send(PullEvent::Error { error: e.to_string() });
            }
        }
        engine.finish_pull(&task.key).await;
    });
    handle
}

pub async fn pull(
    State(state): State<AppState>,
    Json(body): Json<PullBody>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let source = resolve_source(&body.registry, &body.model)
        .await
        .map_err(err_response)?;
    let handle = start_pull(state.engine.clone(), source, body.token, body.quant).await;
    let mut rx = handle.tx.subscribe();
    let snapshot = handle.snapshot();
    let id = handle.id.clone();
    let stream = async_stream::stream! {
        yield event(serde_json::to_value(PullEvent::Started { id }).unwrap_or_default());
        if let Some(s) = snapshot {
            if let Some(file) = s.file {
                yield event(serde_json::to_value(PullEvent::Downloading { file }).unwrap_or_default());
            }
            if s.downloaded > 0 {
                yield event(serde_json::to_value(PullEvent::Progress { downloaded: s.downloaded, total: s.total }).unwrap_or_default());
            }
        }
        loop {
            let ev = match rx.recv().await {
                Ok(ev) => ev,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => {
                    yield event(json!({ "event": "error", "error": "Pull ended unexpectedly; retry the download." }));
                    break;
                }
            };
            let done = matches!(ev, PullEvent::Done { .. } | PullEvent::Error { .. });
            yield event(serde_json::to_value(&ev).unwrap_or_default());
            if done {
                break;
            }
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

pub async fn list_pulls(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({ "pulls": state.engine.list_pulls().await }))
}

#[derive(Debug, Deserialize)]
pub struct CancelBody {
    pub id: String,
}

pub async fn cancel_pull(
    State(state): State<AppState>,
    Json(body): Json<CancelBody>,
) -> Result<impl IntoResponse, ApiError> {
    if state.engine.cancel_pull(&body.id).await {
        Ok(Json(json!({ "cancelled": body.id })))
    } else {
        Err(err_response(Ml5Error::InvalidRequest(format!(
            "no running download with id {}",
            body.id
        ))))
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelBody {
    pub model: String,
}

pub async fn delete_model(
    State(state): State<AppState>,
    Json(body): Json<ModelBody>,
) -> Result<impl IntoResponse, ApiError> {
    state.engine.delete_model(&body.model).await.map_err(err_response)?;
    Ok(Json(json!({ "deleted": body.model })))
}

pub async fn unload_model(
    State(state): State<AppState>,
    Json(body): Json<ModelBody>,
) -> Result<impl IntoResponse, ApiError> {
    state.engine.unload_model(&body.model).await.map_err(err_response)?;
    Ok(Json(json!({ "unloaded": body.model })))
}

#[derive(Debug, Deserialize)]
pub struct LoadBody {
    pub model: String,
    #[serde(default)]
    pub keep_alive: Option<Value>,
    #[serde(flatten)]
    pub overrides: RequestOverrides,
}

pub async fn load_model(
    State(state): State<AppState>,
    Json(body): Json<LoadBody>,
) -> Result<impl IntoResponse, ApiError> {
    let keep_alive_secs = body.keep_alive.as_ref().and_then(parse_keep_alive);
    let name = state
        .engine
        .preload(&body.model, &body.overrides, keep_alive_secs)
        .await
        .map_err(err_response)?;
    Ok(Json(json!({ "loaded": name })))
}

#[derive(Debug, Deserialize)]
pub struct RenameBody {
    pub model: String,
    pub new_name: String,
}

pub async fn rename_model(
    State(state): State<AppState>,
    Json(body): Json<RenameBody>,
) -> Result<impl IntoResponse, ApiError> {
    state
        .engine
        .rename_model(&body.model, &body.new_name)
        .await
        .map_err(err_response)?;
    Ok(Json(json!({ "renamed": body.model, "new_name": body.new_name })))
}

pub async fn rescan(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    state.engine.scan_models().await.map_err(err_response)?;
    Ok(Json(json!({ "models": state.engine.list_models().await.len() })))
}

pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.engine.status().await)
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let s = state.engine.status().await;
    Json(json!({
        "version": s.version,
        "model_count": s.model_count,
        "loaded_models": s.loaded_models,
        "uptime_secs": s.uptime_secs,
        "committed_bytes": s.memory.committed_bytes,
        "active_requests": s.loaded.iter().map(|l| l.active_requests).sum::<usize>(),
        "pulls": s.pulls.len()
    }))
}
