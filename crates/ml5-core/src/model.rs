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
    #[serde(default)]
    pub mmproj: Option<PathBuf>,
    #[serde(default)]
    pub files: Vec<PathBuf>,
    #[serde(default)]
    pub meta_path: Option<PathBuf>,
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelDetails {
    pub architecture: String,
    pub parameter_size: String,
    pub quantization: String,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub kv_bytes_per_token_f16: u64,
}

impl ModelInfo {
    pub fn display_name(&self) -> &str {
        self.friendly_name.as_deref().unwrap_or(&self.name)
    }

    pub fn has(&self, cap: crate::types::Capability) -> bool {
        self.capabilities.contains(&cap)
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_file: Option<String>,
}

impl ModelMeta {
    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(tmp, path)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelState {
    Cold,
    Loading,
    Warm,
}

pub fn is_mmproj_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf") && lower.contains("mmproj")
}

pub fn split_part(file_name: &str) -> Option<(String, u32, u32)> {
    let stem = file_name.strip_suffix(".gguf")?;
    let (base, tail) = stem.rsplit_once("-of-")?;
    let (base, index) = base.rsplit_once('-')?;
    if index.len() != 5 || tail.len() != 5 {
        return None;
    }
    let index: u32 = index.parse().ok()?;
    let total: u32 = tail.parse().ok()?;
    (index >= 1 && total >= 1 && index <= total).then(|| (base.to_string(), index, total))
}

pub fn split_part_name(base: &str, index: u32, total: u32) -> String {
    format!("{base}-{index:05}-of-{total:05}.gguf")
}

pub fn gguf_model_name(file_name: &str) -> String {
    match split_part(file_name) {
        Some((base, _, _)) => base,
        None => file_name
            .strip_suffix(".gguf")
            .unwrap_or(file_name)
            .to_string(),
    }
}

pub fn gguf_meta_path(first_file: &std::path::Path) -> PathBuf {
    let name = first_file
        .file_name()
        .and_then(|n| n.to_str())
        .map(gguf_model_name)
        .unwrap_or_else(|| "model".into());
    first_file.with_file_name(format!("{name}.meta.json"))
}
