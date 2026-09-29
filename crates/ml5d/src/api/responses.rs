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
            "error": { "message": e.to_string(), "type": "server_error" }
        })),
    )
}

#[derive(Debug, Deserialize)]
pub struct ResponsesBody {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub stream: bool,
}

fn input_to_messages(input: &serde_json::Value, instructions: Option<&str>) -> Vec<Message> {
    let mut messages = Vec::new();
    if let Some(sys) = instructions {
        if !sys.is_empty() {
            messages.push(Message {
                role: "system".into(),
                content: sys.to_string(),
                images: vec![],
            });
        }
    }
    match input {
        serde_json::Value::String(s) => {
            messages.push(Message {
                role: "user".into(),
                content: s.clone(),
                images: vec![],
            });
        }
        serde_json::Value::Array(items) => {
            for item in items {
                let role = item["role"].as_str().unwrap_or("user").to_string();
                let content = match &item["content"] {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(parts) => parts
                        .iter()
                        .filter(|p| {
                            let t = p["type"].as_str().unwrap_or("");
                            t == "input_text" || t == "output_text" || t == "text"
                        })
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join(""),
                    _ => String::new(),
                };
                messages.push(Message {
                    role,
                    content,
                    images: vec![],
                });
            }
        }
        _ => {}
    }
    messages
}

pub async fn responses(
    State(state): State<AppState>,
    Json(body): Json<ResponsesBody>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let messages = input_to_messages(&body.input, body.instructions.as_deref());
    let req = ChatRequest {
        model: body.model.clone(),
        messages,
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            max_tokens: body.max_output_tokens,
            ..Default::default()
        },
        overrides: RequestOverrides::default(),
        stream: body.stream,
    };

    let mut stream = state.engine.chat(req).await.map_err(err_response)?;
    let model = body.model.clone();
    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();

    if !body.stream {
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
        return Ok(Json(serde_json::json!({
            "id": id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": model,
            "output": [{
                "type": "message",
                "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "status": "completed",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": text }]
            }],
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.prompt_tokens + usage.completion_tokens
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
                        .event("response.completed")
                        .json_data(serde_json::json!({
                            "type": "response.completed",
                            "response": {
                                "id": id,
                                "object": "response",
                                "created_at": created,
                                "status": "completed",
                                "model": model,
                                "usage": {
                                    "input_tokens": usage.prompt_tokens,
                                    "output_tokens": usage.completion_tokens,
                                    "total_tokens": usage.prompt_tokens + usage.completion_tokens
                                }
                            }
                        }))
                } else {
                    Event::default()
                        .event("response.output_text.delta")
                        .json_data(serde_json::json!({
                            "type": "response.output_text.delta",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": c.text
                        }))
                }
            }
            Err(e) => Event::default().event("error").json_data(serde_json::json!({
                "type": "error",
                "error": { "message": e.to_string() }
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
