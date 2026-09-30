use crate::state::AppState;
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use ml5_core::error::Ml5Error;
use ml5_core::types::*;
use serde::Deserialize;
use std::convert::Infallible;

fn ndjson_response<S>(stream: S) -> axum::response::Response
where
    S: futures::Stream<Item = std::result::Result<String, Infallible>> + Send + 'static,
{
    let body = Body::from_stream(stream.map(|r| r.map(|s| format!("{s}\n"))));
    let mut resp = axum::response::Response::new(body);
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    resp
}

fn err_response(e: Ml5Error) -> (StatusCode, Json<serde_json::Value>) {
    let status = match &e {
        Ml5Error::ModelNotFound(_) => StatusCode::NOT_FOUND,
        Ml5Error::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(serde_json::json!({ "error": e.to_string() })))
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
}

fn ollama_usage(u: &Usage) -> serde_json::Value {
    serde_json::json!({
        "prompt_eval_count": u.prompt_tokens,
        "eval_count": u.completion_tokens,
        "prompt_eval_duration": u.prompt_ms * 1_000_000,
        "eval_duration": u.generation_ms * 1_000_000,
        "total_duration": (u.prompt_ms + u.generation_ms) * 1_000_000,
    })
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateBody {
    pub model: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatBody {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Default)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub num_predict: Option<u32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub num_ctx: Option<u32>,
    pub num_gpu: Option<i32>,
    pub num_thread: Option<i32>,
    #[serde(default, alias = "keep_in_cache")]
    pub cache: bool,
}

impl OllamaOptions {
    fn to_sampling(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            max_tokens: self.num_predict,
            stop: self.stop.clone().unwrap_or_default(),
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            penalty_last_n: None,
        }
    }

    fn to_overrides(&self) -> RequestOverrides {
        RequestOverrides {
            n_ctx: self.num_ctx,
            n_gpu_layers: self.num_gpu,
            n_threads: self.num_thread,
            ..Default::default()
        }
    }
}

fn wants_unload(keep_alive: &Option<serde_json::Value>) -> bool {
    match keep_alive {
        Some(serde_json::Value::Number(n)) => n.as_i64() == Some(0),
        Some(serde_json::Value::String(s)) => s == "0" || s == "0s",
        _ => false,
    }
}

pub async fn generate(
    State(state): State<AppState>,
    Json(body): Json<OllamaGenerateBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if wants_unload(&body.keep_alive) && body.prompt.as_deref().unwrap_or("").is_empty() {
        state.engine.unload_model(&body.model).await.map_err(err_response)?;
        return Ok(Json(serde_json::json!({
            "model": body.model,
            "created_at": now_rfc3339(),
            "response": "",
            "done": true,
        }))
        .into_response());
    }

    let streaming = body.stream.unwrap_or(true);
    let opts = body.options.unwrap_or_default();
    let mut prompt = body.prompt.clone().unwrap_or_default();
    if let Some(sys) = &body.system {
        prompt = format!("{sys}\n\n{prompt}");
    }

    let req = GenerateRequest {
        model: body.model.clone(),
        prompt,
        params: opts.to_sampling(),
        overrides: opts.to_overrides(),
        stream: streaming,
        cache: opts.cache,
    };

    let mut stream = state.engine.generate(req).await.map_err(err_response)?;
    let model = body.model.clone();

    if !streaming {
        let mut text = String::new();
        let mut usage = Usage::default();
        while let Some(chunk) = stream.next().await {
            let c = chunk.map_err(err_response)?;
            text.push_str(&c.text);
            if c.done {
                usage = c.usage.unwrap_or_default();
                break;
            }
        }
        let mut v = serde_json::json!({
            "model": model,
            "created_at": now_rfc3339(),
            "response": text,
            "done": true,
        });
        v.as_object_mut().unwrap().extend(match ollama_usage(&usage) {
            serde_json::Value::Object(m) => m,
            _ => Default::default(),
        });
        return Ok(Json(v).into_response());
    }

    let ndjson = stream.filter_map(move |chunk| {
        let line = match chunk {
            Ok(c) => {
                let mut v = serde_json::json!({
                    "model": model,
                    "created_at": now_rfc3339(),
                    "response": c.text,
                    "done": c.done,
                });
                if c.done {
                    let u = c.usage.unwrap_or_default();
                    v.as_object_mut().unwrap().extend(match ollama_usage(&u) {
                        serde_json::Value::Object(m) => m,
                        _ => Default::default(),
                    });
                }
                serde_json::to_string(&v).ok()
            }
            Err(e) => serde_json::to_string(&serde_json::json!({ "error": e.to_string() })).ok(),
        };
        futures::future::ready(line.map(Ok::<_, Infallible>))
    });

    Ok(ndjson_response(ndjson))
}

pub async fn chat(
    State(state): State<AppState>,
    Json(body): Json<OllamaChatBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    if wants_unload(&body.keep_alive) && body.messages.is_empty() {
        state.engine.unload_model(&body.model).await.map_err(err_response)?;
        return Ok(Json(serde_json::json!({
            "model": body.model,
            "created_at": now_rfc3339(),
            "message": { "role": "assistant", "content": "" },
            "done": true,
        }))
        .into_response());
    }

    let streaming = body.stream.unwrap_or(true);
    let opts = body.options.unwrap_or_default();
    let req = ChatRequest {
        model: body.model.clone(),
        messages: body
            .messages
            .into_iter()
            .map(|m| Message {
                role: m.role,
                content: m.content,
                images: m.images.unwrap_or_default(),
            })
            .collect(),
        params: opts.to_sampling(),
        overrides: opts.to_overrides(),
        stream: streaming,
        cache: opts.cache,
    };

    let mut stream = state.engine.chat(req).await.map_err(err_response)?;
    let model = body.model.clone();

    if !streaming {
        let mut text = String::new();
        let mut usage = Usage::default();
        while let Some(chunk) = stream.next().await {
            let c = chunk.map_err(err_response)?;
            text.push_str(&c.text);
            if c.done {
                usage = c.usage.unwrap_or_default();
                break;
            }
        }
        let mut v = serde_json::json!({
            "model": model,
            "created_at": now_rfc3339(),
            "message": { "role": "assistant", "content": text },
            "done": true,
        });
        v.as_object_mut().unwrap().extend(match ollama_usage(&usage) {
            serde_json::Value::Object(m) => m,
            _ => Default::default(),
        });
        return Ok(Json(v).into_response());
    }

    let ndjson = stream.filter_map(move |chunk| {
        let line = match chunk {
            Ok(c) => {
                let mut v = serde_json::json!({
                    "model": model,
                    "created_at": now_rfc3339(),
                    "message": { "role": "assistant", "content": c.text },
                    "done": c.done,
                });
                if c.done {
                    let u = c.usage.unwrap_or_default();
                    v.as_object_mut().unwrap().extend(match ollama_usage(&u) {
                        serde_json::Value::Object(m) => m,
                        _ => Default::default(),
                    });
                }
                serde_json::to_string(&v).ok()
            }
            Err(e) => serde_json::to_string(&serde_json::json!({ "error": e.to_string() })).ok(),
        };
        futures::future::ready(line.map(Ok::<_, Infallible>))
    });

    Ok(ndjson_response(ndjson))
}

pub async fn tags(State(state): State<AppState>) -> impl IntoResponse {
    let models: Vec<serde_json::Value> = state
        .engine
        .list_models()
        .await
        .into_iter()
        .map(|m| {
            let digest = m
                .digest
                .clone()
                .unwrap_or_else(|| format!("sha256:{:016x}", m.size_bytes));
            serde_json::json!({
                "name": m.name,
                "model": m.name,
                "modified_at": m.modified_at.map(|t| t.to_rfc3339()).unwrap_or_default(),
                "size": m.size_bytes,
                "digest": digest,
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "",
                    "families": null,
                    "parameter_size": "",
                    "quantization_level": ""
                }
            })
        })
        .collect();
    Json(serde_json::json!({ "models": models }))
}

#[derive(Debug, Deserialize)]
pub struct ShowBody {
    pub model: Option<String>,
    pub name: Option<String>,
}

pub async fn show(
    State(state): State<AppState>,
    Json(body): Json<ShowBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let name = body
        .model
        .or(body.name)
        .ok_or_else(|| err_response(Ml5Error::InvalidRequest("missing model".into())))?;
    let info = state.engine.get_model(&name).await.map_err(err_response)?;

    let mut caps = vec!["completion"];
    if info.capabilities.contains(&Capability::Embed) {
        caps.push("embedding");
    }
    if model_supports_thinking(&info.path) {
        caps.push("thinking");
    }

    Ok(Json(serde_json::json!({
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "modified_at": info.modified_at.map(|t| t.to_rfc3339()).unwrap_or_default(),
        "details": { "format": "gguf", "family": "", "parameter_size": "", "quantization_level": "" },
        "capabilities": caps,
        "model_info": {},
    })))
}

fn model_supports_thinking(path: &std::path::Path) -> bool {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let needles: &[&[u8]] = &[b"<|channel|>", b"<think>"];
    needles.iter().any(|n| data.windows(n.len()).any(|w| w == *n))
}

pub async fn version() -> impl IntoResponse {
    Json(serde_json::json!({ "version": env!("CARGO_PKG_VERSION") }))
}

#[derive(Debug, Deserialize)]
pub struct OllamaEmbedBody {
    pub model: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
}

pub async fn embeddings(
    State(state): State<AppState>,
    Json(body): Json<OllamaEmbedBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let input: Vec<String> = if let Some(inp) = body.input {
        match inp {
            serde_json::Value::String(s) => vec![s],
            serde_json::Value::Array(a) => a
                .into_iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect(),
            _ => vec![],
        }
    } else if let Some(p) = body.prompt {
        vec![p]
    } else {
        vec![]
    };

    let req = EmbedRequest {
        model: body.model.clone(),
        input,
    };
    let resp = state.engine.embed(req).await.map_err(err_response)?;

    if resp.embeddings.len() == 1 {
        Ok(Json(serde_json::json!({ "embedding": resp.embeddings[0] })).into_response())
    } else {
        Ok(Json(serde_json::json!({ "embeddings": resp.embeddings })).into_response())
    }
}

pub async fn ps(State(state): State<AppState>) -> impl IntoResponse {
    let status = state.engine.status().await;
    let models: Vec<serde_json::Value> = status
        .loaded_models
        .iter()
        .map(|name| {
            serde_json::json!({
                "name": name,
                "model": name,
                "size": 0,
                "digest": "",
                "details": {},
                "expires_at": now_rfc3339(),
                "size_vram": 0,
            })
        })
        .collect();
    Json(serde_json::json!({ "models": models }))
}


#[derive(Debug, Deserialize)]
pub struct OllamaPullBody {
    pub model: Option<String>,
    pub name: Option<String>,
    #[serde(default)]
    pub insecure: bool,
    #[serde(default)]
    pub stream: Option<bool>,
}

pub async fn pull(
    State(state): State<AppState>,
    Json(body): Json<OllamaPullBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let name = body
        .model
        .clone()
        .or(body.name.clone())
        .ok_or_else(|| err_response(Ml5Error::InvalidRequest("missing model name".into())))?;

    let (registry, model) = if let Some(rest) = name
        .strip_prefix("hf.co/")
        .or_else(|| name.strip_prefix("huggingface.co/"))
    {
        ("hf", rest.to_string())
    } else if name.contains('/') {
        ("hf", name.clone())
    } else {
        ("core", name.clone())
    };

    let source = ml5_core::pull::parse_pull_target(registry, &model).map_err(err_response)?;
    let engine = state.engine.clone();
    let models_dir = engine.config.models_dir.clone();
    let pull_key = format!("{source:?}");
    let tx = engine.register_pull(&pull_key).await.map_err(err_response)?;
    let mut rx = tx.subscribe();
    let streaming = body.stream.unwrap_or(true);

    tokio::spawn(async move {
        use ml5_core::pull::PullEvent;
        let result = ml5_core::pull::pull(&source, &models_dir, None, None, |ev| {
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
                tracing::error!(%e, "pull failed");
                let _ = tx.send(PullEvent::Error {
                    error: e.to_string(),
                });
            }
        }
        engine.finish_pull(&pull_key).await;
    });

    if !streaming {
        let mut last = serde_json::json!({ "status": "pulling" });
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let done = matches!(
                        ev,
                        ml5_core::pull::PullEvent::Done { .. } | ml5_core::pull::PullEvent::Error { .. }
                    );
                    last = ollama_pull_status(&ev);
                    if done {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        return Ok(Json(last).into_response());
    }

    let stream = async_stream::stream! {
        loop {
            let ev = match rx.recv().await {
                Ok(ev) => ev,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => {
                    yield Ok::<Event, Infallible>(Event::default().json_data(serde_json::json!({ "status": "error" })).unwrap());
                    break;
                }
            };
            let done = matches!(ev, ml5_core::pull::PullEvent::Done { .. } | ml5_core::pull::PullEvent::Error { .. });
            yield Ok::<Event, Infallible>(Event::default().json_data(ollama_pull_status(&ev)).unwrap());
            if done {
                break;
            }
        }
    };
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

fn ollama_pull_status(ev: &ml5_core::pull::PullEvent) -> serde_json::Value {
    use ml5_core::pull::PullEvent;
    match ev {
        PullEvent::Resolving { target } => serde_json::json!({ "status": format!("pulling {target}") }),
        PullEvent::Downloading { file } => serde_json::json!({ "status": format!("pulling {file}") }),
        PullEvent::Progress { downloaded, total } => serde_json::json!({
            "status": "pulling",
            "completed": downloaded,
            "total": total,
        }),
        PullEvent::Verifying { file } => serde_json::json!({ "status": format!("verifying {file}") }),
        PullEvent::Done { model } => serde_json::json!({ "status": "success", "model": model }),
        PullEvent::Error { error } => serde_json::json!({ "status": "error", "error": error }),
    }
}

#[derive(Debug, Deserialize)]
pub struct OllamaDeleteBody {
    pub model: Option<String>,
    pub name: Option<String>,
}

pub async fn delete(
    State(state): State<AppState>,
    Json(body): Json<OllamaDeleteBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
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
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let info = state
        .engine
        .get_model(&body.source)
        .await
        .map_err(err_response)?;
    if info.model_type != ml5_core::model::ModelType::Gguf {
        return Err(err_response(Ml5Error::InvalidRequest(
            "copy is only supported for single-file GGUF models".into(),
        )));
    }
    let dest = info
        .path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(format!("{}.gguf", sanitize_name(&body.destination)));
    if dest.exists() {
        return Err(err_response(Ml5Error::InvalidRequest(format!(
            "'{}' already exists",
            body.destination
        ))));
    }
    tokio::fs::copy(&info.path, &dest).await.map_err(|e| {
        err_response(Ml5Error::Backend(format!("copy failed: {e}")))
    })?;
    state.engine.scan_models().await.map_err(err_response)?;
    Ok(StatusCode::OK.into_response())
}

fn sanitize_name(n: &str) -> String {
    n.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '-' })
        .collect()
}
