use crate::state::AppState;
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
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatBody {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    #[serde(default)]
    pub stream: Option<bool>,
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

pub async fn generate(
    State(state): State<AppState>,
    Json(body): Json<OllamaGenerateBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
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

    let sse = stream.filter_map(move |chunk| {
        let event = match chunk {
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
                Event::default().json_data(v)
            }
            Err(e) => Event::default()
                .json_data(serde_json::json!({ "error": e.to_string() })),
        };
        futures::future::ready(Some(Ok::<_, Infallible>(
            event.unwrap_or_else(|_| Event::default().data("serialization error")),
        )))
    });

    Ok(Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response())
}

pub async fn chat(
    State(state): State<AppState>,
    Json(body): Json<OllamaChatBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
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

    let sse = stream.filter_map(move |chunk| {
        let event = match chunk {
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
                Event::default().json_data(v)
            }
            Err(e) => Event::default()
                .json_data(serde_json::json!({ "error": e.to_string() })),
        };
        futures::future::ready(Some(Ok::<_, Infallible>(
            event.unwrap_or_else(|_| Event::default().data("serialization error")),
        )))
    });

    Ok(Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response())
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
    Ok(Json(serde_json::json!({
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "modified_at": info.modified_at.map(|t| t.to_rfc3339()).unwrap_or_default(),
        "details": { "format": "gguf", "family": "", "parameter_size": "", "quantization_level": "" },
        "model_info": {},
    })))
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
