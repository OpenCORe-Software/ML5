use axum::http::StatusCode;
use futures::StreamExt;
use ml5_core::backend::TokenStream;
use ml5_core::error::Ml5Error;
use ml5_core::types::*;
use serde::Deserialize;
use serde_json::Value;

pub fn status_of(e: &Ml5Error) -> StatusCode {
    match e {
        Ml5Error::ModelNotFound(_) => StatusCode::NOT_FOUND,
        Ml5Error::InvalidRequest(_) | Ml5Error::Json(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopInput {
    One(String),
    Many(Vec<String>),
}

impl StopInput {
    pub fn strings(self) -> Vec<String> {
        match self {
            Self::One(s) if s.is_empty() => Vec::new(),
            Self::One(s) => vec![s],
            Self::Many(v) => v.into_iter().filter(|s| !s.is_empty()).collect(),
        }
    }
}

pub fn parse_keep_alive(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_f64().map(|f| if f < 0.0 { -1 } else { f as i64 }),
        Value::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                return None;
            }
            if let Ok(n) = s.parse::<f64>() {
                return Some(if n < 0.0 { -1 } else { n as i64 });
            }
            if s.starts_with('-') {
                return Some(-1);
            }
            let mut total = 0f64;
            let mut num = String::new();
            let mut chars = s.chars().peekable();
            while let Some(c) = chars.next() {
                if c.is_ascii_digit() || c == '.' {
                    num.push(c);
                    continue;
                }
                let mut unit = c.to_string();
                if c == 'm' && chars.peek() == Some(&'s') {
                    unit.push(chars.next().unwrap_or('s'));
                }
                let value: f64 = num.parse().ok()?;
                num.clear();
                total += match unit.as_str() {
                    "h" => value * 3600.0,
                    "m" => value * 60.0,
                    "s" => value,
                    "ms" => value / 1000.0,
                    _ => return None,
                };
            }
            if !num.is_empty() {
                total += num.parse::<f64>().ok()?;
            }
            Some(total as i64)
        }
        _ => None,
    }
}

pub fn content_parts(content: &Value) -> (String, Vec<String>) {
    match content {
        Value::String(s) => (s.clone(), Vec::new()),
        Value::Array(parts) => {
            let mut text = String::new();
            let mut images = Vec::new();
            for p in parts {
                match p["type"].as_str().unwrap_or("text") {
                    "text" | "input_text" | "output_text" => {
                        if let Some(t) = p["text"].as_str() {
                            text.push_str(t);
                        }
                    }
                    "image_url" | "input_image" => {
                        let url = p["image_url"]["url"]
                            .as_str()
                            .or_else(|| p["image_url"].as_str())
                            .or_else(|| p["url"].as_str());
                        if let Some(u) = url.filter(|u| !u.is_empty()) {
                            images.push(u.to_string());
                        }
                    }
                    "image" => {
                        let src = &p["source"];
                        match src["type"].as_str() {
                            Some("base64") => {
                                let mime = src["media_type"].as_str().unwrap_or("image/png");
                                if let Some(data) = src["data"].as_str() {
                                    images.push(format!("data:{mime};base64,{data}"));
                                }
                            }
                            Some("url") => {
                                if let Some(u) = src["url"].as_str() {
                                    images.push(u.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            (text, images)
        }
        _ => (String::new(), Vec::new()),
    }
}

pub fn openai_message(v: &Value) -> Result<Message, Ml5Error> {
    let role = match v["role"].as_str() {
        Some("developer") => "system",
        Some(r) => r,
        None => return Err(Ml5Error::InvalidRequest("every message needs a role".into())),
    };
    let (content, images) = content_parts(&v["content"]);
    let tool_calls = v["tool_calls"]
        .as_array()
        .map(|calls| calls.iter().filter_map(ToolCall::from_value).collect())
        .unwrap_or_default();
    let thinking = v["reasoning_content"]
        .as_str()
        .or_else(|| v["reasoning"].as_str())
        .or_else(|| v["thinking"].as_str())
        .filter(|t| !t.is_empty())
        .map(str::to_string);
    Ok(Message {
        role: role.to_string(),
        content,
        images,
        tool_calls,
        tool_call_id: v["tool_call_id"].as_str().map(str::to_string),
        name: v["name"].as_str().map(str::to_string),
        thinking,
    })
}

pub fn openai_messages(list: &[Value]) -> Result<Vec<Message>, Ml5Error> {
    list.iter().map(openai_message).collect()
}

pub fn non_empty_tools(v: Option<Value>) -> Option<Value> {
    v.filter(|t| t.as_array().is_some_and(|a| !a.is_empty()))
}

#[derive(Debug, Default)]
pub struct Collected {
    pub text: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub finish_reason: String,
}

pub async fn collect(stream: &mut TokenStream) -> Result<Collected, Ml5Error> {
    let mut out = Collected::default();
    while let Some(chunk) = stream.next().await {
        let c = chunk?;
        out.text.push_str(&c.text);
        out.thinking.push_str(&c.thinking);
        if c.done {
            out.tool_calls = c.tool_calls;
            out.usage = c.usage.unwrap_or_default();
            out.finish_reason = c.finish_reason.unwrap_or_else(|| "stop".into());
            return Ok(out);
        }
    }
    Err(Ml5Error::Backend(
        "Inference ended before completion; retry the request.".into(),
    ))
}

pub fn response_format_from_ollama(format: &Value) -> Option<Value> {
    match format {
        Value::String(s) if s == "json" => Some(serde_json::json!({ "type": "json_object" })),
        Value::Object(_) => Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": "response", "schema": format }
        })),
        _ => None,
    }
}
