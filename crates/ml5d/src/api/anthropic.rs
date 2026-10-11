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
        _ => "api_error",
    };
    (
        status_of(&e),
        Json(json!({ "type": "error", "error": { "type": kind, "message": e.to_string() } })),
    )
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Deserialize)]
pub struct MessagesBody {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub thinking: Option<Value>,
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|p| p["type"].as_str() == Some("text"))
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn convert_messages(body: &MessagesBody) -> Vec<Message> {
    let mut out = Vec::new();
    if let Some(sys) = &body.system {
        let text = text_of(sys);
        if !text.is_empty() {
            out.push(Message::text("system", text));
        }
    }
    for m in &body.messages {
        let blocks = match &m.content {
            Value::String(s) => {
                out.push(Message::text(&m.role, s.clone()));
                continue;
            }
            Value::Array(b) => b,
            _ => continue,
        };
        let mut msg = Message {
            role: m.role.clone(),
            ..Default::default()
        };
        let mut thinking = String::new();
        for b in blocks {
            match b["type"].as_str().unwrap_or("") {
                "text" => {
                    if !msg.content.is_empty() {
                        msg.content.push('\n');
                    }
                    msg.content.push_str(b["text"].as_str().unwrap_or(""));
                }
                "image" => msg.images.extend(content_parts(&json!([b])).1),
                "thinking" => thinking.push_str(b["thinking"].as_str().unwrap_or("")),
                "tool_use" => {
                    if let Some(call) = ToolCall::from_value(b) {
                        msg.tool_calls.push(call);
                    }
                }
                "tool_result" => {
                    let mut content = text_of(&b["content"]);
                    if b["is_error"].as_bool() == Some(true) {
                        content = format!("Error: {content}");
                    }
                    out.push(Message {
                        role: "tool".into(),
                        content,
                        tool_call_id: b["tool_use_id"].as_str().map(str::to_string),
                        ..Default::default()
                    });
                }
                _ => {}
            }
        }
        if !thinking.is_empty() {
            msg.thinking = Some(thinking);
        }
        if !msg.content.is_empty() || !msg.images.is_empty() || !msg.tool_calls.is_empty() {
            out.push(msg);
        }
    }
    out
}

fn convert_tools(tools: &Option<Vec<Value>>) -> Option<Value> {
    let list: Vec<Value> = tools
        .as_ref()?
        .iter()
        .filter(|t| t["name"].is_string())
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"].as_str().unwrap_or(""),
                    "parameters": if t["input_schema"].is_object() { t["input_schema"].clone() } else { json!({ "type": "object", "properties": {} }) }
                }
            })
        })
        .collect();
    (!list.is_empty()).then_some(Value::Array(list))
}

fn convert_tool_choice(choice: &Option<Value>) -> (Option<Value>, Option<bool>) {
    let Some(c) = choice else { return (None, None) };
    let parallel = c["disable_parallel_tool_use"].as_bool().map(|d| !d);
    let mapped = match c["type"].as_str() {
        Some("auto") => Some(json!("auto")),
        Some("any") => Some(json!("required")),
        Some("none") => Some(json!("none")),
        Some("tool") => c["name"]
            .as_str()
            .map(|n| json!({ "type": "function", "function": { "name": n } })),
        _ => None,
    };
    (mapped, parallel)
}

fn stop_reason(finish: &str) -> &'static str {
    match finish {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

fn to_chat_request(body: &MessagesBody) -> ChatRequest {
    let (tool_choice, parallel) = convert_tool_choice(&body.tool_choice);
    let think = body.thinking.as_ref().and_then(|t| match t["type"].as_str() {
        Some("enabled") => Some(true),
        Some("disabled") => Some(false),
        _ => None,
    });
    ChatRequest {
        model: body.model.clone(),
        messages: convert_messages(body),
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            max_tokens: Some(body.max_tokens),
            stop: body.stop_sequences.iter().filter(|s| !s.is_empty()).cloned().collect(),
            ..Default::default()
        },
        tools: convert_tools(&body.tools),
        tool_choice,
        parallel_tool_calls: parallel,
        think,
        ..Default::default()
    }
}

fn sse(name: &str, data: Value) -> Result<Event, Infallible> {
    Ok(Event::default().event(name).data(data.to_string()))
}

pub async fn messages(
    State(state): State<AppState>,
    Json(body): Json<MessagesBody>,
) -> Result<axum::response::Response, ApiError> {
    let req = to_chat_request(&body);
    let model = body.model.clone();
    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let mut stream = state.engine.chat(req).await.map_err(err_response)?;

    if !body.stream {
        let c = collect(&mut stream).await.map_err(err_response)?;
        let mut content = Vec::new();
        if !c.thinking.is_empty() {
            content.push(json!({ "type": "thinking", "thinking": c.thinking, "signature": "" }));
        }
        if !c.text.is_empty() || c.tool_calls.is_empty() {
            content.push(json!({ "type": "text", "text": c.text }));
        }
        for t in &c.tool_calls {
            content.push(json!({ "type": "tool_use", "id": t.id, "name": t.name, "input": t.arguments_value() }));
        }
        return Ok(Json(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason(&c.finish_reason),
            "stop_sequence": null,
            "usage": {
                "input_tokens": c.usage.prompt_tokens,
                "output_tokens": c.usage.completion_tokens,
                "cache_read_input_tokens": c.usage.cached_tokens
            }
        }))
        .into_response());
    }

    let events = async_stream::stream! {
        yield sse("message_start", json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": { "input_tokens": 0, "output_tokens": 0 }
            }
        }));
        let mut index: i64 = -1;
        let mut open: Option<&'static str> = None;
        while let Some(item) = stream.next().await {
            let c = match item {
                Ok(c) => c,
                Err(e) => {
                    yield sse("error", json!({ "type": "error", "error": { "type": "api_error", "message": e.to_string() } }));
                    return;
                }
            };
            if c.progress.is_some() {
                yield sse("ping", json!({ "type": "ping" }));
                continue;
            }
            for (kind, piece) in [("thinking", &c.thinking), ("text", &c.text)] {
                if piece.is_empty() {
                    continue;
                }
                if open != Some(kind) {
                    if open.is_some() {
                        yield sse("content_block_stop", json!({ "type": "content_block_stop", "index": index }));
                    }
                    index += 1;
                    open = Some(kind);
                    let block = if kind == "thinking" {
                        json!({ "type": "thinking", "thinking": "", "signature": "" })
                    } else {
                        json!({ "type": "text", "text": "" })
                    };
                    yield sse("content_block_start", json!({ "type": "content_block_start", "index": index, "content_block": block }));
                }
                let delta = if kind == "thinking" {
                    json!({ "type": "thinking_delta", "thinking": piece })
                } else {
                    json!({ "type": "text_delta", "text": piece })
                };
                yield sse("content_block_delta", json!({ "type": "content_block_delta", "index": index, "delta": delta }));
            }
            if c.done {
                if open.is_some() {
                    yield sse("content_block_stop", json!({ "type": "content_block_stop", "index": index }));
                }
                for t in &c.tool_calls {
                    index += 1;
                    yield sse("content_block_start", json!({
                        "type": "content_block_start", "index": index,
                        "content_block": { "type": "tool_use", "id": t.id, "name": t.name, "input": {} }
                    }));
                    yield sse("content_block_delta", json!({
                        "type": "content_block_delta", "index": index,
                        "delta": { "type": "input_json_delta", "partial_json": t.arguments_value().to_string() }
                    }));
                    yield sse("content_block_stop", json!({ "type": "content_block_stop", "index": index }));
                }
                let usage = c.usage.unwrap_or_default();
                let finish = c.finish_reason.unwrap_or_else(|| "stop".into());
                yield sse("message_delta", json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": stop_reason(&finish), "stop_sequence": null },
                    "usage": { "input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens }
                }));
                yield sse("message_stop", json!({ "type": "message_stop" }));
                return;
            }
        }
        yield sse("error", json!({ "type": "error", "error": { "type": "api_error", "message": "Inference ended unexpectedly; retry the request." } }));
    };
    Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response())
}

#[derive(Debug, Deserialize)]
pub struct CountTokensBody {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<Value>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
}

pub async fn count_tokens(
    State(state): State<AppState>,
    Json(body): Json<CountTokensBody>,
) -> Result<impl IntoResponse, ApiError> {
    let as_messages = MessagesBody {
        model: body.model,
        messages: body.messages,
        system: body.system,
        max_tokens: 1,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: Vec::new(),
        stream: false,
        tools: body.tools,
        tool_choice: body.tool_choice,
        thinking: None,
    };
    let tokens = state
        .engine
        .count_tokens(to_chat_request(&as_messages))
        .await
        .map_err(err_response)?;
    Ok(Json(json!({ "input_tokens": tokens })))
}
