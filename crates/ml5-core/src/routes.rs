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
