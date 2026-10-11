use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Chat,
    Generate,
    Embed,
    Vision,
    Tools,
    Thinking,
    AudioTranscribe,
    AudioSpeech,
    ImageGenerate,
    VideoUnderstand,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn from_value(v: &serde_json::Value) -> Option<Self> {
        let f = v.get("function").filter(|f| f.is_object()).unwrap_or(v);
        let name = f["name"].as_str().filter(|n| !n.is_empty())?.to_string();
        let arguments = match f.get("arguments").or_else(|| f.get("input")) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Null) | None => "{}".into(),
            Some(other) => other.to_string(),
        };
        let id = v["id"]
            .as_str()
            .or_else(|| v["call_id"].as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
        Some(Self { id, name, arguments })
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = serde_json::Value::deserialize(d)?;
        Self::from_value(&v).ok_or_else(|| serde::de::Error::custom("tool call needs a function name"))
    }
}

impl ToolCall {
    pub fn to_openai(&self, index: usize) -> serde_json::Value {
        serde_json::json!({
            "index": index,
            "id": self.id,
            "type": "function",
            "function": { "name": self.name, "arguments": self.arguments }
        })
    }

    pub fn arguments_value(&self) -> serde_json::Value {
        serde_json::from_str(&self.arguments).unwrap_or_else(|_| {
            if self.arguments.trim().is_empty() {
                serde_json::json!({})
            } else {
                serde_json::Value::String(self.arguments.clone())
            }
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

impl Message {
    pub fn text(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            ..Default::default()
        }
    }

    pub fn to_openai(&self) -> serde_json::Value {
        let mut out = serde_json::Map::new();
        out.insert("role".into(), self.role.clone().into());
        if self.images.is_empty() {
            out.insert("content".into(), self.content.clone().into());
        } else {
            let mut parts = Vec::new();
            if !self.content.is_empty() {
                parts.push(serde_json::json!({ "type": "text", "text": self.content }));
            }
            for image in &self.images {
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": image_data_url(image) }
                }));
            }
            out.insert("content".into(), serde_json::Value::Array(parts));
        }
        if !self.tool_calls.is_empty() {
            out.insert(
                "tool_calls".into(),
                serde_json::Value::Array(
                    self.tool_calls
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let mut v = c.to_openai(i);
                            v.as_object_mut().map(|o| o.remove("index"));
                            v
                        })
                        .collect(),
                ),
            );
        }
        if let Some(id) = &self.tool_call_id {
            out.insert("tool_call_id".into(), id.clone().into());
        }
        if let Some(name) = &self.name {
            out.insert("name".into(), name.clone().into());
        }
        if let Some(thinking) = self.thinking.as_ref().filter(|t| !t.is_empty()) {
            out.insert("reasoning_content".into(), thinking.clone().into());
        }
        serde_json::Value::Object(out)
    }
}

pub fn image_data_url(image: &str) -> String {
    if image.starts_with("data:") || image.starts_with("http://") || image.starts_with("https://") {
        return image.to_string();
    }
    let mime = match image.get(..8) {
        Some(p) if p.starts_with("iVBOR") => "image/png",
        Some(p) if p.starts_with("/9j/") => "image/jpeg",
        Some(p) if p.starts_with("R0lGOD") => "image/gif",
        Some(p) if p.starts_with("UklGR") => "image/webp",
        Some(p) if p.starts_with("Qk") => "image/bmp",
        _ => "image/png",
    };
    format!("data:{mime};base64,{image}")
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SamplingParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalty_last_n: Option<i32>,
}

impl SamplingParams {
    pub fn validate(&self) -> crate::Result<()> {
        let invalid = self.temperature.is_some_and(|v| !v.is_finite() || v < 0.0)
            || self
                .top_p
                .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            || self
                .min_p
                .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
            || self
                .repeat_penalty
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || self.max_tokens == Some(0);
        if invalid {
            return Err(crate::error::Ml5Error::InvalidRequest("Use temperature >= 0, top_p/min_p between 0 and 1, repeat_penalty > 0, and max_tokens > 0.".into()));
        }
        Ok(())
    }

    pub fn apply_openai(&self, body: &mut serde_json::Map<String, serde_json::Value>) {
        let mut set = |key: &str, value: serde_json::Value| {
            if !value.is_null() {
                body.insert(key.into(), value);
            }
        };
        set("temperature", serde_json::json!(self.temperature));
        set("top_p", serde_json::json!(self.top_p));
        set("top_k", serde_json::json!(self.top_k));
        set("min_p", serde_json::json!(self.min_p));
        set("max_tokens", serde_json::json!(self.max_tokens));
        set("seed", serde_json::json!(self.seed));
        set("repeat_penalty", serde_json::json!(self.repeat_penalty));
        set("frequency_penalty", serde_json::json!(self.frequency_penalty));
        set("presence_penalty", serde_json::json!(self.presence_penalty));
        set("repeat_last_n", serde_json::json!(self.penalty_last_n));
        if !self.stop.is_empty() {
            body.insert("stop".into(), serde_json::json!(self.stop));
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_batch: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<crate::config::GpuLayers>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attn: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_k: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_v: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_threads: Option<i32>,
}

pub const KV_CACHE_TYPES: &[&str] = &[
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];

impl RequestOverrides {
    pub fn validate(&self) -> crate::Result<()> {
        if self.n_ctx.is_some_and(|v| v != 0 && (v < 256 || v > i32::MAX as u32))
            || self.n_batch.is_some_and(|v| v == 0 || v > i32::MAX as u32)
            || self.n_threads.is_some_and(|v| v < 0)
        {
            return Err(crate::error::Ml5Error::InvalidRequest(
                "Use n_ctx >= 256 (or 0 for automatic), n_batch > 0, and n_threads >= 0.".into(),
            ));
        }
        if self
            .flash_attn
            .as_deref()
            .is_some_and(|v| !["auto", "on", "off"].contains(&v))
        {
            return Err(crate::error::Ml5Error::InvalidRequest(
                "flash_attn must be auto, on, or off.".into(),
            ));
        }
        for value in [&self.cache_type_k, &self.cache_type_v]
            .into_iter()
            .flatten()
        {
            if !KV_CACHE_TYPES.contains(&value.as_str()) {
                return Err(crate::error::Ml5Error::InvalidRequest(format!(
                    "Unknown KV cache type '{value}'. Use one of: {}.",
                    KV_CACHE_TYPES.join(", ")
                )));
            }
        }
        Ok(())
    }

    pub fn affects_load(&self) -> bool {
        self.n_ctx.is_some()
            || self.n_batch.is_some()
            || self.n_gpu_layers.is_some()
            || self.flash_attn.is_some()
            || self.cache_type_k.is_some()
            || self.cache_type_v.is_some()
            || self.n_threads.is_some()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub params: SamplingParams,
    pub overrides: RequestOverrides,
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<serde_json::Value>,
    pub parallel_tool_calls: Option<bool>,
    pub think: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub response_format: Option<serde_json::Value>,
    pub keep_alive_secs: Option<i64>,
}

impl ChatRequest {
    pub fn to_openai(&self) -> serde_json::Value {
        let mut body = serde_json::Map::new();
        body.insert("model".into(), self.model.clone().into());
        body.insert(
            "messages".into(),
            serde_json::Value::Array(self.messages.iter().map(Message::to_openai).collect()),
        );
        body.insert("stream".into(), true.into());
        body.insert("stream_options".into(), serde_json::json!({ "include_usage": true }));
        body.insert("return_progress".into(), true.into());
        body.insert("timings_per_token".into(), false.into());
        self.params.apply_openai(&mut body);
        if let Some(tools) = self.tools.as_ref().filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
            body.insert("tools".into(), tools.clone());
            if let Some(choice) = &self.tool_choice {
                body.insert("tool_choice".into(), choice.clone());
            }
        }
        if let Some(parallel) = self.parallel_tool_calls {
            body.insert("parallel_tool_calls".into(), parallel.into());
        }
        let effort = self
            .reasoning_effort
            .as_deref()
            .map(|e| e.trim().to_ascii_lowercase())
            .filter(|e| !e.is_empty());
        let think = match (self.think, effort.as_deref()) {
            (Some(t), _) => Some(t),
            (None, Some("none")) => Some(false),
            (None, Some(_)) => Some(true),
            (None, None) => None,
        };
        if let Some(think) = think {
            body.insert(
                "chat_template_kwargs".into(),
                serde_json::json!({ "enable_thinking": think }),
            );
            if think {
                if let Some(level) = effort.filter(|e| e != "none") {
                    body.insert("reasoning_effort".into(), level.into());
                }
            } else {
                body.insert("reasoning_effort".into(), "none".into());
            }
        }
        if let Some(format) = &self.response_format {
            body.insert("response_format".into(), format.clone());
        }
        serde_json::Value::Object(body)
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub params: SamplingParams,
    pub overrides: RequestOverrides,
    pub keep_alive_secs: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>,
    pub keep_alive_secs: Option<i64>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    pub prompt_ms: u64,
    pub generation_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct StreamChunk {
    pub text: String,
    pub thinking: String,
    pub tool_calls: Vec<ToolCall>,
    pub done: bool,
    pub usage: Option<Usage>,
    pub progress: Option<InferenceProgress>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceProgress {
    pub stage: String,
    pub completed: usize,
    pub total: usize,
}

#[derive(Debug, Clone)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub usage: Usage,
}
