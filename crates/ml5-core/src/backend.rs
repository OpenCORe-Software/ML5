use crate::config::ModelParams;
use crate::error::Result;
use crate::types::*;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>;

#[async_trait]
pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    fn capabilities(&self) -> Vec<Capability>;

    async fn load(&self, model_path: &std::path::Path, params: &ModelParams) -> Result<()>;

    async fn unload(&self) -> Result<()>;

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream>;

    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream>;

    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse>;
}
