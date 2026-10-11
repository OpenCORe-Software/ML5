use crate::error::{Ml5Error, Result};
use serde::{Deserialize, Serialize};

pub const ROUTES_BASE: &str = "https://routing.opencore.one";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRoute {
    pub hf_repo: String,
    pub hf_file: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveResponse {
    pub found: bool,
    pub route: Option<ModelRoute>,
    pub canonical_name: Option<String>,
    pub message: Option<String>,
}

const BUILTIN_ROUTES: &[(&[&str], &str, &str, &str)] = &[
    (
        &["gemma3:4b", "gemma3"],
        "gemma3:4b",
        "ggml-org/gemma-3-4b-it-GGUF",
        "gemma-3-4b-it-Q4_K_M.gguf",
    ),
    (
        &["gemma3:12b"],
        "gemma3:12b",
        "ggml-org/gemma-3-12b-it-GGUF",
        "gemma-3-12b-it-Q4_K_M.gguf",
    ),
    (
        &["gemma3:27b"],
        "gemma3:27b",
        "ggml-org/gemma-3-27b-it-GGUF",
        "gemma-3-27b-it-Q4_K_M.gguf",
    ),
    (
        &["qwen3:8b", "qwen3"],
        "qwen3:8b",
        "Qwen/Qwen3-8B-GGUF",
        "Qwen3-8B-Q4_K_M.gguf",
    ),
    (
        &["qwen3:4b"],
        "qwen3:4b",
        "Qwen/Qwen3-4B-GGUF",
        "Qwen3-4B-Q4_K_M.gguf",
    ),
    (
        &["qwen2.5-coder:7b", "qwen2.5-coder"],
        "qwen2.5-coder:7b",
        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    ),
    (
        &["llava:7b", "llava"],
        "llava:7b",
        "second-state/Llava-v1.5-7B-GGUF",
        "llava-v1.5-7b-Q4_K_M.gguf",
    ),
];

pub fn builtin_route(name: &str) -> Option<(String, ModelRoute)> {
    let wanted = name.trim().to_ascii_lowercase();
    BUILTIN_ROUTES
        .iter()
        .find(|(aliases, ..)| aliases.contains(&wanted.as_str()))
        .map(|(_, canonical, repo, file)| {
            (
                canonical.to_string(),
                ModelRoute {
                    hf_repo: repo.to_string(),
                    hf_file: Some(file.to_string()),
                    description: None,
                    tags: Vec::new(),
                },
            )
        })
}

pub async fn resolve_friendly_name(name: &str) -> Result<ResolveResponse> {
    let client = reqwest::Client::builder()
        .user_agent("ml5/0.1")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| Ml5Error::Download(format!("routes client build failed: {e}")))?;

    let url = format!("{}/api/resolve?name={}", ROUTES_BASE, urlencoding::encode(name));
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| Ml5Error::Download(format!("routes request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(Ml5Error::Download(format!(
            "routes server returned {}",
            resp.status()
        )));
    }

    resp.json::<ResolveResponse>()
        .await
        .map_err(|e| Ml5Error::Download(format!("bad routes response: {e}")))
}
