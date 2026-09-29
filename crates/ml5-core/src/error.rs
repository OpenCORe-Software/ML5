use thiserror::Error;

#[derive(Debug, Error)]
pub enum Ml5Error {
    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("model already exists: {0}")]
    ModelExists(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("no backend available for capability: {0}")]
    NoBackend(String),

    #[error("download failed: {0}")]
    Download(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Ml5Error>;
