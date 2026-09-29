use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use ml5_core::error::Ml5Error;
use ml5_core::types::*;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

fn err_response(e: Ml5Error) -> (StatusCode, Json<serde_json::Value>) {
    let status = match &e {
        Ml5Error::ModelNotFound(_) => StatusCode::NOT_FOUND,
        Ml5Error::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(serde_json::json!({ "error": e.to_string() })))
}

#[derive(Debug, Deserialize)]
pub struct ChatBody {
    pub model: String,
    pub messages: Vec<Message>,
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
    #[serde(flatten)]
    pub overrides: RequestOverrides,
    #[serde(default = "default_true")]
    pub stream: bool,
}

fn default_true() -> bool {
    true
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

pub async fn chat(
    State(state): State<AppState>,
    Json(body): Json<ChatBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    state
        .engine
        .get_model(&body.model)
        .await
        .map_err(err_response)?;

    let req = ChatRequest {
        model: body.model.clone(),
        messages: body.messages,
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            min_p: body.min_p,
            max_tokens: body.max_tokens,
            stop: body.stop,
            seed: body.seed,
            repeat_penalty: body.repeat_penalty,
            frequency_penalty: body.frequency_penalty,
            presence_penalty: body.presence_penalty,
            penalty_last_n: None,
        },
        overrides: body.overrides,
        stream: body.stream,
    };

    native_response(state, body.model, body.stream, Some(req), None).await
}

type ApiError = (StatusCode, Json<serde_json::Value>);

fn event(value: serde_json::Value) -> Result<Event, Infallible> {
    Ok(Event::default()
        .json_data(value)
        .expect("JSON value is serializable"))
}

fn chunk_value(model: &str, chat: bool, c: StreamChunk) -> serde_json::Value {
    if let Some(progress) = c.progress {
        return serde_json::json!({ "status": progress.stage, "completed": progress.completed, "total": progress.total });
    }
    if chat {
        serde_json::to_value(ChatChunk {
            model: model.into(),
            message: Message {
                role: "assistant".into(),
                content: c.text,
                images: vec![],
            },
            done: c.done,
            usage: c.usage,
            finish_reason: c.finish_reason,
        })
        .unwrap()
    } else {
        serde_json::json!({ "model": model, "response": c.text, "done": c.done, "usage": c.usage, "finish_reason": c.finish_reason })
    }
}

async fn native_response(
    state: AppState,
    model: String,
    streaming: bool,
    chat: Option<ChatRequest>,
    generate: Option<GenerateRequest>,
) -> Result<axum::response::Response, ApiError> {
    let is_chat = chat.is_some();
    let engine = state.engine;
    let was_loaded = engine.is_loaded(&model).await;
    let defaults = engine.config.model.clone();
    let load = async move {
        if let Some(req) = chat {
            engine.chat(req).await
        } else {
            engine.generate(generate.unwrap()).await
        }
    };
    if !streaming {
        let mut stream = load.await.map_err(err_response)?;
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            let mut chunk = chunk.map_err(err_response)?;
            text.push_str(&chunk.text);
            if chunk.done {
                chunk.text = text;
                return Ok(Json(chunk_value(&model, is_chat, chunk)).into_response());
            }
        }
        return Err(err_response(Ml5Error::Backend(
            "Inference ended without a completion event".into(),
        )));
    }
    let stream = async_stream::stream! {
        yield event(serde_json::json!({ "status": if was_loaded { "queued" } else { "loading_model" }, "model": model, "defaults": defaults }));
        let mut stream = match load.await {
            Ok(stream) => stream,
            Err(e) => { yield event(serde_json::json!({ "error": e.to_string() })); return; }
        };
        yield event(serde_json::json!({ "status": "queued", "message": "Model ready; waiting for prompt processing" }));
        let mut done = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(c) => {
                    done = c.done;
                    yield event(chunk_value(&model, is_chat, c));
                    if done { break; }
                }
                Err(e) => { yield event(serde_json::json!({ "error": e.to_string() })); return; }
            }
        }
        if !done { yield event(serde_json::json!({ "error": "Inference ended unexpectedly; retry the request." })); }
    };
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[derive(Debug, Deserialize)]
pub struct GenerateBody {
    pub model: String,
    pub prompt: String,
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
    #[serde(flatten)]
    pub overrides: RequestOverrides,
    #[serde(default = "default_true")]
    pub stream: bool,
}

pub async fn generate(
    State(state): State<AppState>,
    Json(body): Json<GenerateBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    state
        .engine
        .get_model(&body.model)
        .await
        .map_err(err_response)?;
    let req = GenerateRequest {
        model: body.model.clone(),
        prompt: body.prompt,
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            min_p: body.min_p,
            max_tokens: body.max_tokens,
            stop: body.stop,
            seed: body.seed,
            repeat_penalty: body.repeat_penalty,
            frequency_penalty: body.frequency_penalty,
            presence_penalty: body.presence_penalty,
            penalty_last_n: None,
        },
        overrides: body.overrides,
        stream: body.stream,
    };

    native_response(state, body.model, body.stream, None, Some(req)).await
}

#[derive(Debug, Deserialize)]
pub struct EmbedBody {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbedBodyOut {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
}

pub async fn embed(
    State(state): State<AppState>,
    Json(body): Json<EmbedBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let model = body.model.clone();
    let req = EmbedRequest {
        model: body.model,
        input: body.input,
    };
    let resp = state.engine.embed(req).await.map_err(err_response)?;
    Ok(Json(EmbedBodyOut {
        model,
        embeddings: resp.embeddings,
    }))
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub name: String,
    pub size_bytes: u64,
    pub capabilities: Vec<Capability>,
}

#[derive(Debug, Serialize)]
pub struct ModelsOut {
    pub models: Vec<ModelEntry>,
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let models = state
        .engine
        .list_models()
        .await
        .into_iter()
        .map(|m| ModelEntry {
            name: m.name,
            size_bytes: m.size_bytes,
            capabilities: m.capabilities,
        })
        .collect();
    Json(ModelsOut { models })
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

pub async fn pull(
    State(state): State<AppState>,
    Json(body): Json<PullBody>,
) -> Result<
    Sse<impl futures::Stream<Item = std::result::Result<Event, Infallible>>>,
    (StatusCode, Json<serde_json::Value>),
> {
    let (source, auto_resolved) = match body.registry.as_str() {
        "routes" => {
            match ml5_core::pull::resolve_via_routes(&body.model).await {
                Ok(Some(source)) => (source, true),
                Ok(None) => {
                    (
                        ml5_core::pull::parse_pull_target("core", &body.model)
                            .map_err(err_response)?,
                        false,
                    )
                }
                Err(e) => {
                    tracing::warn!("routes resolution failed, falling back to core: {e}");
                    (
                        ml5_core::pull::parse_pull_target("core", &body.model)
                            .map_err(err_response)?,
                        false,
                    )
                }
            }
        }
        _ => (
            ml5_core::pull::parse_pull_target(&body.registry, &body.model)
                .map_err(err_response)?,
            false,
        ),
    };

    let engine = state.engine.clone();
    let models_dir = engine.config.models_dir.clone();
    let pull_key = format!("{source:?}");
    let tx = engine
        .register_pull(&pull_key)
        .await
        .map_err(err_response)?;
    let mut rx = tx.subscribe();

    tokio::spawn(async move {
        use ml5_core::pull::PullEvent;

        let result = ml5_core::pull::pull(
            &source,
            &models_dir,
            body.token.as_deref(),
            body.quant.as_deref(),
            |ev| {
            match &ev {
                PullEvent::Resolving { target } => {
                    if auto_resolved {
                        tracing::info!(%target, "auto-resolved via routes server");
                    } else {
                        tracing::info!(%target, "resolving model");
                    }
                }
                PullEvent::Downloading { file } => tracing::info!(%file, "downloading"),
                PullEvent::Progress { .. } => {}
                PullEvent::Verifying { file } => tracing::info!(%file, "verifying"),

                PullEvent::Done { .. } => return,
                PullEvent::Error { .. } => {}
            }
            let _ = tx.send(ev);
        })
        .await;

        match result {
            Ok((name, path)) => {
                tracing::info!(%name, path = %path.display(), "model pulled");
                match engine.scan_models().await {
                    Ok(()) => {
                        let _ = tx.send(PullEvent::Done { model: name });
                    }
                    Err(e) => {
                        let _ = tx.send(PullEvent::Error {
                            error: format!("Downloaded but could not register model: {e}"),
                        });
                    }
                }
            }
            Err(e) => {
                tracing::error!(%e, model = %pull_key, "pull failed");
                let _ = tx.send(PullEvent::Error {
                    error: e.to_string(),
                });
            }
        }
        engine.finish_pull(&pull_key).await;
    });

    let stream = async_stream::stream! {
        loop {
            let ev = match rx.recv().await {
                Ok(ev) => ev,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => { yield event(serde_json::json!({ "event": "error", "error": "Pull ended unexpectedly; retry the download." })); break; }
            };
            let done = matches!(ev, ml5_core::pull::PullEvent::Done { .. } | ml5_core::pull::PullEvent::Error { .. });
            let json = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
            yield Ok(Event::default().data(json));
            if done {
                break;
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[derive(Debug, Deserialize)]
pub struct DeleteBody {
    pub model: String,
}

pub async fn delete_model(
    State(state): State<AppState>,
    Json(body): Json<DeleteBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    state
        .engine
        .delete_model(&body.model)
        .await
        .map_err(err_response)?;
    Ok(Json(serde_json::json!({ "deleted": body.model })))
}

#[derive(Debug, Deserialize)]
pub struct UnloadBody {
    pub model: String,
}

pub async fn unload_model(
    State(state): State<AppState>,
    Json(body): Json<UnloadBody>,
) -> Result<impl IntoResponse, ApiError> {
    state
        .engine
        .unload_model(&body.model)
        .await
        .map_err(err_response)?;
    Ok(Json(serde_json::json!({ "unloaded": body.model })))
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
    Ok(Json(serde_json::json!({
        "renamed": body.model,
        "new_name": body.new_name
    })))
}

pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    Json(state.engine.status().await)
}

pub async fn metrics(State(state): State<AppState>) -> impl IntoResponse {
    let s = state.engine.status().await;
    Json(serde_json::json!({
        "version": s.version,
        "model_count": s.model_count,
        "loaded_models": s.loaded_models,
        "uptime_secs": s.uptime_secs
    }))
}
