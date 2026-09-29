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
    (
        status,
        Json(serde_json::json!({
            "type": "error",
            "error": { "type": "api_error", "message": e.to_string() }
        })),
    )
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: serde_json::Value,
}

fn anthropic_content_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter(|p| p["type"].as_str() == Some("text"))
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[derive(Debug, Deserialize)]
pub struct MessagesBody {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
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
}

fn to_chat_request(body: &MessagesBody) -> ChatRequest {
    let mut messages: Vec<Message> = Vec::new();
    if let Some(sys) = &body.system {
        let text = match sys {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) => parts
                .iter()
                .filter(|p| p["type"].as_str() == Some("text"))
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        if !text.is_empty() {
            messages.push(Message {
                role: "system".into(),
                content: text,
                images: vec![],
            });
        }
    }
    for m in &body.messages {
        messages.push(Message {
            role: m.role.clone(),
            content: anthropic_content_to_string(&m.content),
            images: vec![],
        });
    }
    ChatRequest {
        model: body.model.clone(),
        messages,
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            top_k: body.top_k,
            max_tokens: Some(body.max_tokens),
            stop: body.stop_sequences.clone(),
            ..Default::default()
        },
        overrides: RequestOverrides::default(),
        stream: body.stream,
    }
}

pub async fn messages(
    State(state): State<AppState>,
    Json(body): Json<MessagesBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let req = to_chat_request(&body);
    let model = body.model.clone();
    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let mut stream = state.engine.chat(req).await.map_err(err_response)?;

    if !body.stream {
        let mut text = String::new();
        let mut usage = Usage::default();
        let mut finish = "end_turn".to_string();
        while let Some(chunk) = stream.next().await {
            let c = chunk.map_err(err_response)?;
            text.push_str(&c.text);
            if c.done {
                usage = c.usage.unwrap_or_default();
                finish = match c.finish_reason.as_deref() {
                    Some("stop") => "end_turn".into(),
                    Some("length") => "max_tokens".into(),
                    Some(other) => other.to_string(),
                    None => "end_turn".into(),
                };
                break;
            }
        }
        return Ok(Json(serde_json::json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{ "type": "text", "text": text }],
            "stop_reason": finish,
            "stop_sequence": null,
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens
            }
        }))
        .into_response());
    }

    let sse = stream.filter_map(move |chunk| {
        let event = match chunk {
            Ok(c) => {
                if c.done {
                    let usage = c.usage.unwrap_or_default();
                    Event::default()
                        .event("message_stop")
                        .json_data(serde_json::json!({
                            "type": "message_stop",
                            "usage": { "output_tokens": usage.completion_tokens }
                        }))
                } else {
                    Event::default()
                        .event("content_block_delta")
                        .json_data(serde_json::json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": { "type": "text_delta", "text": c.text }
                        }))
                }
            }
            Err(e) => Event::default().event("error").json_data(serde_json::json!({
                "type": "error",
                "error": { "type": "api_error", "message": e.to_string() }
            })),
        };
        futures::future::ready(Some(Ok::<_, Infallible>(
            event.unwrap_or_else(|_| Event::default().data("serialization error")),
        )))
    });

    Ok(Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response())
}

#[derive(Debug, Deserialize)]
pub struct CountTokensBody {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<serde_json::Value>,
}

pub async fn count_tokens(
    State(state): State<AppState>,
    Json(body): Json<CountTokensBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let _ = state.engine.get_model(&body.model).await.map_err(err_response)?;
    let mut chars = 0usize;
    if let Some(sys) = &body.system {
        chars += anthropic_content_to_string(sys).len();
    }
    for m in &body.messages {
        chars += anthropic_content_to_string(&m.content).len();
    }
    let approx = (chars / 4) as u32;
    Ok(Json(serde_json::json!({ "input_tokens": approx })))
}
