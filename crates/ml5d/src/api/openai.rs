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
    (
        status,
        Json(serde_json::json!({
            "error": { "message": e.to_string(), "type": "ml5_error" }
        })),
    )
}

#[derive(Debug, Deserialize)]
pub struct OaiMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsBody {
    pub model: String,
    pub messages: Vec<OaiMessage>,
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
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<StopInput>,
    #[serde(default)]
    pub seed: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    fn strings(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct OaiDelta {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OaiChoice {
    pub index: u32,
    pub delta: OaiDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct OaiChunk {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<OaiChoice>,
    pub created: i64,
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<ChatCompletionsBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let req = ChatRequest {
        model: body.model.clone(),
        messages: body
            .messages
            .into_iter()
            .map(|m| Message {
                role: m.role,
                content: m.content,
                images: vec![],
            })
            .collect(),
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            min_p: body.min_p,
            max_tokens: body.max_tokens,
            repeat_penalty: body.repeat_penalty,
            frequency_penalty: body.frequency_penalty,
            presence_penalty: body.presence_penalty,
            stop: body.stop.map(StopInput::strings).unwrap_or_default(),
            seed: body.seed,
            ..Default::default()
        },
        overrides: RequestOverrides::default(),
        stream: body.stream,
        cache: false,
    };

    let mut stream = state.engine.chat(req).await.map_err(err_response)?;
    let model = body.model.clone();
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    if !body.stream {
        let (text, usage, reason) = collect_response(&mut stream).await.map_err(err_response)?;
        return Ok(Json(serde_json::json!({
            "id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": reason }],
            "usage": usage_json(usage)
        })).into_response());
    }

    let sse = stream.filter_map(move |chunk| {
        if chunk.as_ref().is_ok_and(|c| c.progress.is_some()) {
            return futures::future::ready(None);
        }
        let event = match chunk {
            Ok(c) => Event::default().json_data(OaiChunk {
                id: id.clone(),
                object: "chat.completion.chunk".into(),
                model: model.clone(),
                created,
                choices: vec![OaiChoice {
                    index: 0,
                    delta: OaiDelta {
                        role: "assistant".into(),
                        content: c.text,
                    },
                    finish_reason: if c.done {
                        Some(c.finish_reason.unwrap_or_else(|| "stop".into()))
                    } else {
                        None
                    },
                }],
            }),
            Err(e) => Event::default().json_data(serde_json::json!({
                "error": { "message": e.to_string() }
            })),
        };
        futures::future::ready(Some(Ok::<_, Infallible>(
            event.unwrap_or_else(|_| Event::default().data("serialization error")),
        )))
    });

    let done =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    Ok(Sse::new(sse.chain(done))
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn usage_json(usage: Usage) -> serde_json::Value {
    serde_json::json!({ "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.prompt_tokens + usage.completion_tokens })
}

async fn collect_response(
    stream: &mut ml5_core::backend::TokenStream,
) -> ml5_core::Result<(String, Usage, String)> {
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        let c = chunk?;
        text.push_str(&c.text);
        if c.done {
            return Ok((
                text,
                c.usage.unwrap_or_default(),
                c.finish_reason.unwrap_or_else(|| "stop".into()),
            ));
        }
    }
    Err(Ml5Error::Backend(
        "Inference ended before completion".into(),
    ))
}

#[derive(Debug, Deserialize)]
pub struct CompletionsBody {
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
    pub stop: Option<StopInput>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub stream: bool,
}

pub async fn completions(
    State(state): State<AppState>,
    Json(body): Json<CompletionsBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let req = GenerateRequest {
        model: body.model.clone(),
        prompt: body.prompt,
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            min_p: body.min_p,
            max_tokens: body.max_tokens,
            repeat_penalty: body.repeat_penalty,
            frequency_penalty: body.frequency_penalty,
            presence_penalty: body.presence_penalty,
            stop: body.stop.map(StopInput::strings).unwrap_or_default(),
            seed: body.seed,
            ..Default::default()
        },
        overrides: RequestOverrides::default(),
        stream: body.stream,
        cache: false,
    };

    let mut stream = state.engine.generate(req).await.map_err(err_response)?;
    let model = body.model.clone();
    let id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    if !body.stream {
        let (text, usage, reason) = collect_response(&mut stream).await.map_err(err_response)?;
        return Ok(Json(serde_json::json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{ "index": 0, "text": text, "finish_reason": reason }], "usage": usage_json(usage)
        })).into_response());
    }

    let sse = stream.filter_map(move |chunk| {
        if chunk.as_ref().is_ok_and(|c| c.progress.is_some()) { return futures::future::ready(None); }
        let event = match chunk {
            Ok(c) => Event::default().json_data(serde_json::json!({
                "id": id,
                "object": "text_completion",
                "model": model,
                "created": created,
                "choices": [{
                    "index": 0,
                    "text": c.text,
                    "finish_reason": if c.done { Some(c.finish_reason.unwrap_or_else(|| "stop".into())) } else { None }
                }]
            })),
            Err(e) => Event::default()
                .json_data(serde_json::json!({ "error": { "message": e.to_string() } })),
        };
        futures::future::ready(Some(Ok::<_, Infallible>(event.unwrap_or_else(|_| Event::default().data("serialization error")))))
    });

    let done =
        futures::stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
    Ok(Sse::new(sse.chain(done))
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OaiEmbedInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsBody {
    pub model: String,
    pub input: OaiEmbedInput,
}

pub async fn embeddings(
    State(state): State<AppState>,
    Json(body): Json<EmbeddingsBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let input = match body.input {
        OaiEmbedInput::One(s) => vec![s],
        OaiEmbedInput::Many(v) => v,
    };
    let req = EmbedRequest {
        model: body.model.clone(),
        input,
    };
    let resp = state.engine.embed(req).await.map_err(err_response)?;

    let data: Vec<serde_json::Value> = resp
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(i, e)| serde_json::json!({ "object": "embedding", "index": i, "embedding": e }))
        .collect();

    Ok(Json(serde_json::json!({
        "object": "list",
        "model": body.model,
        "data": data,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "total_tokens": resp.usage.prompt_tokens
        }
    })))
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let data: Vec<serde_json::Value> = state
        .engine
        .list_models()
        .await
        .into_iter()
        .map(|m| {
            serde_json::json!({
                "id": m.name,
                "object": "model",
                "created": m.modified_at.map(|t| t.timestamp()).unwrap_or(0),
                "owned_by": "ml5"
            })
        })
        .collect();
    Json(serde_json::json!({ "object": "list", "data": data }))
}
