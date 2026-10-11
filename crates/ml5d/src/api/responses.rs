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
    (
        status_of(&e),
        Json(json!({ "error": { "message": e.to_string(), "type": "server_error" } })),
    )
}

#[derive(Debug, Deserialize)]
pub struct ResponsesBody {
    pub model: String,
    pub input: Value,
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
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub text: Option<Value>,
}

fn input_to_messages(input: &Value, instructions: Option<&str>) -> Result<Vec<Message>, Ml5Error> {
    let mut messages = Vec::new();
    if let Some(sys) = instructions.filter(|s| !s.is_empty()) {
        messages.push(Message::text("system", sys));
    }
    let items = match input {
        Value::String(s) => {
            messages.push(Message::text("user", s.clone()));
            return Ok(messages);
        }
        Value::Array(items) => items,
        _ => return Err(Ml5Error::InvalidRequest("input must be a string or an array of items".into())),
    };
    for item in items {
        match item["type"].as_str().unwrap_or("message") {
            "message" => {
                let mut m = openai_message(item)?;
                if m.role == "developer" {
                    m.role = "system".into();
                }
                messages.push(m);
            }
            "function_call" => {
                let Some(call) = ToolCall::from_value(item) else { continue };
                match messages.last_mut() {
                    Some(last) if last.role == "assistant" && last.tool_call_id.is_none() => last.tool_calls.push(call),
                    _ => messages.push(Message {
                        role: "assistant".into(),
                        tool_calls: vec![call],
                        ..Default::default()
                    }),
                }
            }
            "function_call_output" => {
                let output = match &item["output"] {
                    Value::String(s) => s.clone(),
                    other => content_parts(other).0,
                };
                messages.push(Message {
                    role: "tool".into(),
                    content: output,
                    tool_call_id: item["call_id"].as_str().map(str::to_string),
                    ..Default::default()
                });
            }
            "reasoning" => {
                let text: String = item["summary"]
                    .as_array()
                    .map(|s| s.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"))
                    .unwrap_or_default();
                if !text.is_empty() {
                    messages.push(Message {
                        role: "assistant".into(),
                        thinking: Some(text),
                        ..Default::default()
                    });
                }
            }
            _ => {}
        }
    }
    let mut merged: Vec<Message> = Vec::new();
    for m in messages {
        match merged.last_mut() {
            Some(prev)
                if prev.role == "assistant"
                    && m.role == "assistant"
                    && prev.content.is_empty()
                    && prev.tool_calls.is_empty()
                    && prev.thinking.is_some() =>
            {
                let thinking = prev.thinking.take();
                *prev = m;
                if prev.thinking.is_none() {
                    prev.thinking = thinking;
                }
            }
            _ => merged.push(m),
        }
    }
    Ok(merged)
}

fn convert_tools(tools: &Option<Vec<Value>>) -> Option<Value> {
    let list: Vec<Value> = tools
        .as_ref()?
        .iter()
        .filter(|t| t["type"].as_str() == Some("function") && t["name"].is_string())
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"].as_str().unwrap_or(""),
                    "parameters": if t["parameters"].is_object() { t["parameters"].clone() } else { json!({ "type": "object", "properties": {} }) }
                }
            })
        })
        .collect();
    (!list.is_empty()).then_some(Value::Array(list))
}

fn convert_tool_choice(choice: &Option<Value>) -> Option<Value> {
    match choice.as_ref()? {
        Value::String(s) => Some(json!(s)),
        Value::Object(o) if o.get("type").and_then(|t| t.as_str()) == Some("function") => o
            .get("name")
            .and_then(|n| n.as_str())
            .map(|n| json!({ "type": "function", "function": { "name": n } })),
        _ => None,
    }
}

fn response_format(text: &Option<Value>) -> Option<Value> {
    let format = &text.as_ref()?["format"];
    match format["type"].as_str()? {
        "json_object" => Some(json!({ "type": "json_object" })),
        "json_schema" => Some(json!({
            "type": "json_schema",
            "json_schema": { "name": format["name"].as_str().unwrap_or("response"), "schema": format["schema"] }
        })),
        _ => None,
    }
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "input_tokens": u.prompt_tokens,
        "input_tokens_details": { "cached_tokens": u.cached_tokens },
        "output_tokens": u.completion_tokens,
        "output_tokens_details": { "reasoning_tokens": 0 },
        "total_tokens": u.prompt_tokens + u.completion_tokens
    })
}

fn output_items(text: &str, thinking: &str, calls: &[ToolCall], msg_id: &str, reasoning_id: &str) -> Vec<Value> {
    let mut out = Vec::new();
    if !thinking.is_empty() {
        out.push(json!({
            "type": "reasoning", "id": reasoning_id,
            "summary": [{ "type": "summary_text", "text": thinking }]
        }));
    }
    if !text.is_empty() || calls.is_empty() {
        out.push(json!({
            "type": "message", "id": msg_id, "status": "completed", "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }]
        }));
    }
    for c in calls {
        out.push(json!({
            "type": "function_call", "id": format!("fc_{}", c.id), "call_id": c.id,
            "name": c.name, "arguments": c.arguments, "status": "completed"
        }));
    }
    out
}

pub async fn responses(
    State(state): State<AppState>,
    Json(body): Json<ResponsesBody>,
) -> Result<axum::response::Response, ApiError> {
    let messages = input_to_messages(&body.input, body.instructions.as_deref()).map_err(err_response)?;
    let effort = body.reasoning.as_ref().and_then(|r| r["effort"].as_str()).map(str::to_string);
    let req = ChatRequest {
        model: body.model.clone(),
        messages,
        params: SamplingParams {
            temperature: body.temperature,
            top_p: body.top_p,
            max_tokens: body.max_output_tokens,
            ..Default::default()
        },
        tools: convert_tools(&body.tools),
        tool_choice: convert_tool_choice(&body.tool_choice),
        parallel_tool_calls: body.parallel_tool_calls,
        reasoning_effort: effort,
        response_format: response_format(&body.text),
        ..Default::default()
    };

    let mut stream = state.engine.chat(req).await.map_err(err_response)?;
    let model = body.model.clone();
    let id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let reasoning_id = format!("rs_{}", uuid::Uuid::new_v4().simple());
    let created = chrono::Utc::now().timestamp();
    let envelope = {
        let id = id.clone();
        let model = model.clone();
        move |status: &str, output: Vec<Value>, usage: Option<Value>| -> Value {
            json!({
                "id": id, "object": "response", "created_at": created, "status": status,
                "model": model, "output": output, "usage": usage,
                "parallel_tool_calls": true, "tool_choice": "auto", "tools": [], "error": null,
                "incomplete_details": null
            })
        }
    };

    if !body.stream {
        let c = collect(&mut stream).await.map_err(err_response)?;
        let status = if c.finish_reason == "length" { "incomplete" } else { "completed" };
        let mut resp = envelope(
            status,
            output_items(&c.text, &c.thinking, &c.tool_calls, &msg_id, &reasoning_id),
            Some(usage_json(&c.usage)),
        );
        if status == "incomplete" {
            resp["incomplete_details"] = json!({ "reason": "max_output_tokens" });
        }
        resp["output_text"] = c.text.into();
        return Ok(Json(resp).into_response());
    }

    let events = async_stream::stream! {
        let mut seq: u64 = 0;
        let mut ev = |name: &str, mut data: Value| -> Result<Event, Infallible> {
            data["type"] = name.into();
            data["sequence_number"] = seq.into();
            seq += 1;
            Ok(Event::default().event(name).data(data.to_string()))
        };
        yield ev("response.created", json!({ "response": envelope("in_progress", vec![], None) }));
        yield ev("response.in_progress", json!({ "response": envelope("in_progress", vec![], None) }));
        let mut text = String::new();
        let mut thinking = String::new();
        let mut reasoning_open = false;
        let mut message_open = false;
        let mut next_index: u64 = 0;
        let mut reasoning_index = 0;
        let mut message_index = 0;
        while let Some(item) = stream.next().await {
            let c = match item {
                Ok(c) => c,
                Err(e) => {
                    let mut failed = envelope("failed", vec![], None);
                    failed["error"] = json!({ "code": "server_error", "message": e.to_string() });
                    yield ev("response.failed", json!({ "response": failed }));
                    return;
                }
            };
            if c.progress.is_some() {
                continue;
            }
            if !c.thinking.is_empty() {
                if !reasoning_open {
                    reasoning_open = true;
                    reasoning_index = next_index;
                    next_index += 1;
                    yield ev("response.output_item.added", json!({
                        "output_index": reasoning_index,
                        "item": { "type": "reasoning", "id": reasoning_id, "summary": [] }
                    }));
                    yield ev("response.reasoning_summary_part.added", json!({
                        "item_id": reasoning_id, "output_index": reasoning_index, "summary_index": 0,
                        "part": { "type": "summary_text", "text": "" }
                    }));
                }
                thinking.push_str(&c.thinking);
                yield ev("response.reasoning_summary_text.delta", json!({
                    "item_id": reasoning_id, "output_index": reasoning_index, "summary_index": 0, "delta": c.thinking
                }));
            }
            if !c.text.is_empty() {
                if !message_open {
                    message_open = true;
                    message_index = next_index;
                    next_index += 1;
                    yield ev("response.output_item.added", json!({
                        "output_index": message_index,
                        "item": { "type": "message", "id": msg_id, "status": "in_progress", "role": "assistant", "content": [] }
                    }));
                    yield ev("response.content_part.added", json!({
                        "item_id": msg_id, "output_index": message_index, "content_index": 0,
                        "part": { "type": "output_text", "text": "", "annotations": [] }
                    }));
                }
                text.push_str(&c.text);
                yield ev("response.output_text.delta", json!({
                    "item_id": msg_id, "output_index": message_index, "content_index": 0, "delta": c.text
                }));
            }
            if c.done {
                if reasoning_open {
                    yield ev("response.reasoning_summary_text.done", json!({
                        "item_id": reasoning_id, "output_index": reasoning_index, "summary_index": 0, "text": thinking
                    }));
                    yield ev("response.reasoning_summary_part.done", json!({
                        "item_id": reasoning_id, "output_index": reasoning_index, "summary_index": 0,
                        "part": { "type": "summary_text", "text": thinking }
                    }));
                    yield ev("response.output_item.done", json!({
                        "output_index": reasoning_index,
                        "item": { "type": "reasoning", "id": reasoning_id, "summary": [{ "type": "summary_text", "text": thinking }] }
                    }));
                }
                if message_open {
                    yield ev("response.output_text.done", json!({
                        "item_id": msg_id, "output_index": message_index, "content_index": 0, "text": text
                    }));
                    yield ev("response.content_part.done", json!({
                        "item_id": msg_id, "output_index": message_index, "content_index": 0,
                        "part": { "type": "output_text", "text": text, "annotations": [] }
                    }));
                    yield ev("response.output_item.done", json!({
                        "output_index": message_index,
                        "item": {
                            "type": "message", "id": msg_id, "status": "completed", "role": "assistant",
                            "content": [{ "type": "output_text", "text": text, "annotations": [] }]
                        }
                    }));
                }
                for call in &c.tool_calls {
                    let index = next_index;
                    next_index += 1;
                    let item_id = format!("fc_{}", call.id);
                    yield ev("response.output_item.added", json!({
                        "output_index": index,
                        "item": { "type": "function_call", "id": item_id, "call_id": call.id, "name": call.name, "arguments": "", "status": "in_progress" }
                    }));
                    yield ev("response.function_call_arguments.delta", json!({
                        "item_id": item_id, "output_index": index, "delta": call.arguments
                    }));
                    yield ev("response.function_call_arguments.done", json!({
                        "item_id": item_id, "output_index": index, "arguments": call.arguments
                    }));
                    yield ev("response.output_item.done", json!({
                        "output_index": index,
                        "item": { "type": "function_call", "id": item_id, "call_id": call.id, "name": call.name, "arguments": call.arguments, "status": "completed" }
                    }));
                }
                let usage = c.usage.unwrap_or_default();
                let incomplete = c.finish_reason.as_deref() == Some("length");
                let mut done = envelope(
                    if incomplete { "incomplete" } else { "completed" },
                    output_items(&text, &thinking, &c.tool_calls, &msg_id, &reasoning_id),
                    Some(usage_json(&usage)),
                );
                if incomplete {
                    done["incomplete_details"] = json!({ "reason": "max_output_tokens" });
                    yield ev("response.incomplete", json!({ "response": done }));
                } else {
                    yield ev("response.completed", json!({ "response": done }));
                }
                return;
            }
        }
        let mut failed = envelope("failed", vec![], None);
        failed["error"] = json!({ "code": "server_error", "message": "Inference ended unexpectedly; retry the request." });
        yield ev("response.failed", json!({ "response": failed }));
    };
    Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response())
}
