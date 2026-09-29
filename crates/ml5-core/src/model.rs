use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub capabilities: Vec<crate::types::Capability>,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub modified_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub friendly_name: Option<String>,
    #[serde(default)]
    pub model_type: ModelType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    #[default]
    Gguf,
    Safetensors,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelMeta {
    #[serde(default)]
    pub friendly_name: Option<String>,
    #[serde(default)]
    pub source_repo: Option<String>,
    #[serde(default)]
    pub source_file: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Cold,
    Loading,
    Warm,
}
