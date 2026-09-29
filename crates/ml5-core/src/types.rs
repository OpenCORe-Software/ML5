use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Chat,
    Generate,
    Embed,
    Vision,
    AudioTranscribe,
    AudioSpeech,
    ImageGenerate,
    VideoUnderstand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
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
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_batch: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_gpu_layers: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attn: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_k: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_type_v: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_threads: Option<i32>,
}

impl RequestOverrides {
    pub fn validate(&self) -> crate::Result<()> {
        if self.n_ctx.is_some_and(|v| v < 2 || v > i32::MAX as u32)
            || self.n_batch.is_some_and(|v| v == 0 || v > i32::MAX as u32)
            || self.n_gpu_layers.is_some_and(|v| v < -1)
            || self.n_threads.is_some_and(|v| v < 0)
        {
            return Err(crate::error::Ml5Error::InvalidRequest(
                "Use n_ctx >= 2, n_batch > 0, n_gpu_layers >= -1, and n_threads >= 0.".into(),
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
            if ![
                "f32", "f16", "bf16", "q8_0", "q8_1", "q4_0", "q4_1", "q5_0", "q5_1", "q4_k",
                "q5_k", "q6_k", "q2_k", "q3_k", "iq4_nl",
            ]
            .contains(&value.as_str())
            {
                return Err(crate::error::Ml5Error::InvalidRequest(format!(
                    "Unknown KV cache type '{value}'. Try f16, q8_0, or q4_0."
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub params: SamplingParams,
    pub overrides: RequestOverrides,
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub params: SamplingParams,
    pub overrides: RequestOverrides,
    pub stream: bool,
}

#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub prompt_ms: u64,
    pub generation_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct StreamChunk {
    pub text: String,
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
