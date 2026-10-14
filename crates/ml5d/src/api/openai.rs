use super::common::*;
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
use serde_json::{json, Value};
use std::convert::Infallible;

type ApiError = (StatusCode, Json<Value>);

fn err_response(e: Ml5Error) -> ApiError {
    let kind = match status_of(&e) {
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::BAD_REQUEST => "invalid_request_error",
        _ => "server_error",
    };
    (
        status_of(&e),
        Json(json!({ "error": { "message": e.to_string(), "type": kind } })),
    )
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.prompt_tokens + u.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": u.cached_tokens }
    })
}

fn event(v: Value) -> Result<Event, Infallible> {
    Ok(Event::default().data(v.to_string()))
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionsBody {
    pub model: String,
    pub messages: Vec<Value>,
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
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<Value>,
    #[serde(default)]
    pub stop: Option<StopInput>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub keep_alive: Option<Value>,
    #[serde(default)]
    pub cache_prompt: Option<bool>,
    #[serde(flatten)]
    pub overrides: RequestOverrides,
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<ChatCompletionsBody>,
) -> Result<axum::response::Response, ApiError> {
    let messages = openai_messages(&body.messages).map_err(err_response)?;
    let include_usage = body
        .stream_options
        .as_ref()
        .and_then(|o| o["include_usage"].as_bool())
        .unwrap_or(false);
    let req = ChatRequest {
        model: body.model.clone(),
        messages,
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            min_p: body.min_p,
            max_tokens: body.max_completion_tokens.or(body.max_tokens),
            repeat_penalty: body.repeat_penalty,
            frequency_penalty: body.frequency_penalty,
            presence_penalty: body.presence_penalty,
            stop: body.stop.map(StopInput::strings).unwrap_or_default(),
            seed: body.seed,
            penalty_last_n: None,
        },
        overrides: body.overrides,
        tools: non_empty_tools(body.tools),
        tool_choice: body.tool_choice,
        parallel_tool_calls: body.parallel_tool_calls,
        think: None,
        reasoning_effort: body.reasoning_effort,
        response_format: body.response_format.filter(|f| f["type"].as_str() != Some("text")),
        keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
        cache_prompt: body.cache_prompt,
    };

    let mut stream = state.engine.chat(req).await.map_err(err_response)?;
    let model = body.model;
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    if !body.stream {
        let c = collect(&mut stream).await.map_err(err_response)?;
        let mut message = serde_json::Map::new();
        message.insert("role".into(), "assistant".into());
        message.insert(
            "content".into(),
            if c.text.is_empty() && !c.tool_calls.is_empty() {
                Value::Null
            } else {
                c.text.into()
            },
        );
        if !c.thinking.is_empty() {
            message.insert("reasoning_content".into(), c.thinking.into());
        }
        if !c.tool_calls.is_empty() {
            message.insert(
                "tool_calls".into(),
                c.tool_calls
                    .iter()
                    .map(|t| json!({ "id": t.id, "type": "function", "function": { "name": t.name, "arguments": t.arguments } }))
                    .collect(),
            );
        }
        return Ok(Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "system_fingerprint": format!("ml5-{}", env!("CARGO_PKG_VERSION")),
            "choices": [{ "index": 0, "message": message, "finish_reason": c.finish_reason }],
            "usage": usage_json(&c.usage)
        }))
        .into_response());
    }

    let chunk = move |delta: Value, finish: Option<String>| -> Value {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
        })
    };
    let sse = async_stream::stream! {
        yield event(chunk(json!({ "role": "assistant", "content": "" }), None));
        while let Some(item) = stream.next().await {
            match item {
                Ok(c) if c.progress.is_some() => continue,
                Ok(c) if c.done => {
                    if !c.tool_calls.is_empty() {
                        let calls: Vec<Value> = c.tool_calls.iter().enumerate().map(|(i, t)| t.to_openai(i)).collect();
                        yield event(chunk(json!({ "tool_calls": calls }), None));
                    }
                    let usage = c.usage.unwrap_or_default();
                    let mut last = chunk(json!({}), Some(c.finish_reason.unwrap_or_else(|| "stop".into())));
                    if include_usage {
                        yield event(last);
                        let mut u = chunk(json!({}), None);
                        u["choices"] = json!([]);
                        u["usage"] = usage_json(&usage);
                        yield event(u);
                    } else {
                        last["usage"] = usage_json(&usage);
                        yield event(last);
                    }
                    break;
                }
                Ok(c) => {
                    if !c.thinking.is_empty() {
                        yield event(chunk(json!({ "reasoning_content": c.thinking }), None));
                    }
                    if !c.text.is_empty() {
                        yield event(chunk(json!({ "content": c.text }), None));
                    }
                }
                Err(e) => {
                    yield event(json!({ "error": { "message": e.to_string(), "type": "server_error" } }));
                    break;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Ok(Sse::new(sse).keep_alive(KeepAlive::default()).into_response())
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
pub struct CompletionsBody {
    pub model: String,
    pub prompt: PromptInput,
    #[serde(default)]
    pub suffix: Option<String>,
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
    #[serde(default)]
    pub stream_options: Option<Value>,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub keep_alive: Option<Value>,
    #[serde(default)]
    pub cache_prompt: Option<bool>,
    #[serde(flatten)]
    pub overrides: RequestOverrides,
}

pub async fn completions(
    State(state): State<AppState>,
    Json(body): Json<CompletionsBody>,
) -> Result<axum::response::Response, ApiError> {
    let prompt = match body.prompt {
        PromptInput::One(s) => s,
        PromptInput::Many(v) if v.len() == 1 => v.into_iter().next().unwrap_or_default(),
        PromptInput::Many(_) => {
            return Err(err_response(Ml5Error::InvalidRequest(
                "send one prompt per request".into(),
            )))
        }
    };
    if body.suffix.as_deref().is_some_and(|s| !s.is_empty()) {
        return Err(err_response(Ml5Error::InvalidRequest(
            "suffix (fill-in-the-middle) is not supported on /v1/completions".into(),
        )));
    }
    let echo = body.echo.then(|| prompt.clone());
    let include_usage = body
        .stream_options
        .as_ref()
        .and_then(|o| o["include_usage"].as_bool())
        .unwrap_or(false);
    let req = GenerateRequest {
        model: body.model.clone(),
        prompt,
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
            penalty_last_n: None,
        },
        overrides: body.overrides,
        keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
        cache_prompt: body.cache_prompt,
    };

    let mut stream = state.engine.generate(req).await.map_err(err_response)?;
    let model = body.model;
    let id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();
    if !body.stream {
        let c = collect(&mut stream).await.map_err(err_response)?;
        let text = format!("{}{}", echo.unwrap_or_default(), c.text);
        return Ok(Json(json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{ "index": 0, "text": text, "logprobs": null, "finish_reason": c.finish_reason }],
            "usage": usage_json(&c.usage)
        }))
        .into_response());
    }

    let chunk = move |text: String, finish: Option<String>| -> Value {
        json!({
            "id": id, "object": "text_completion", "created": created, "model": model,
            "choices": [{ "index": 0, "text": text, "logprobs": null, "finish_reason": finish }]
        })
    };
    let sse = async_stream::stream! {
        if let Some(prefix) = echo {
            yield event(chunk(prefix, None));
        }
        while let Some(item) = stream.next().await {
            match item {
                Ok(c) if c.progress.is_some() => continue,
                Ok(c) if c.done => {
                    let usage = c.usage.unwrap_or_default();
                    let mut last = chunk(c.text, Some(c.finish_reason.unwrap_or_else(|| "stop".into())));
                    if include_usage {
                        last["usage"] = usage_json(&usage);
                    }
                    yield event(last);
                    break;
                }
                Ok(c) => {
                    if !c.text.is_empty() {
                        yield event(chunk(c.text, None));
                    }
                }
                Err(e) => {
                    yield event(json!({ "error": { "message": e.to_string(), "type": "server_error" } }));
                    break;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Ok(Sse::new(sse).keep_alive(KeepAlive::default()).into_response())
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
    #[serde(default)]
    pub keep_alive: Option<Value>,
}

pub async fn embeddings(
    State(state): State<AppState>,
    Json(body): Json<EmbeddingsBody>,
) -> Result<impl IntoResponse, ApiError> {
    let input = match body.input {
        OaiEmbedInput::One(s) => vec![s],
        OaiEmbedInput::Many(v) => v,
    };
    let req = EmbedRequest {
        model: body.model.clone(),
        input,
        keep_alive_secs: body.keep_alive.as_ref().and_then(parse_keep_alive),
    };
    let resp = state.engine.embed(req).await.map_err(err_response)?;
    let data: Vec<Value> = resp
        .embeddings
        .into_iter()
        .enumerate()
        .map(|(i, e)| json!({ "object": "embedding", "index": i, "embedding": e }))
        .collect();
    Ok(Json(json!({
        "object": "list",
        "model": body.model,
        "data": data,
        "usage": { "prompt_tokens": resp.usage.prompt_tokens, "total_tokens": resp.usage.prompt_tokens }
    })))
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let data: Vec<Value> = state
        .engine
        .list_models()
        .await
        .into_iter()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "created": m.modified_at.map(|t| t.timestamp()).unwrap_or(0),
                "owned_by": "ml5",
                "name": m.display_name(),
                "capabilities": m.capabilities,
                "context_length": m.details.as_ref().and_then(|d| d.context_length),
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

pub async fn get_model(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let m = state.engine.get_model(&id).await.map_err(err_response)?;
    Ok(Json(json!({
        "id": m.name,
        "object": "model",
        "created": m.modified_at.map(|t| t.timestamp()).unwrap_or(0),
        "owned_by": "ml5",
        "name": m.display_name(),
        "capabilities": m.capabilities,
        "context_length": m.details.as_ref().and_then(|d| d.context_length),
    })))
}
