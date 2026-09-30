use crate::backend::{Backend, TokenStream};
use crate::config::ModelParams;
use crate::error::{Ml5Error, Result};
use crate::runtime;
use crate::types::*;
use async_trait::async_trait;
use futures::StreamExt;
use std::net::TcpListener;
use std::path::Path;
use std::process::Stdio;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

pub struct LlamaServerBackend {
    child: Mutex<Option<Child>>,
    port: Mutex<u16>,
    kind: Mutex<runtime::GpuKind>,
}

impl LlamaServerBackend {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            port: Mutex::new(0),
            kind: Mutex::new(runtime::GpuKind::Cpu),
        }
    }

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr().map(|a| a.port()))
            .unwrap_or(12355)
    }

    async fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", *self.port.lock().await)
    }

    async fn wait_ready(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url().await);
        let client = reqwest::Client::new();
        for _ in 0..120 {
            if let Ok(r) = client.get(&url).send().await {
                if r.status().is_success() {
                    return Ok(());
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        Err(Ml5Error::Backend(
            "llama-server did not become ready".into(),
        ))
    }
}

impl Default for LlamaServerBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Backend for LlamaServerBackend {
    fn name(&self) -> &str {
        "llama-server"
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::Chat, Capability::Generate, Capability::Embed]
    }

    async fn load(&self, model_path: &Path, params: &ModelParams) -> Result<()> {
        let kind = runtime::detect_gpu();
        *self.kind.lock().await = kind;
        let bin = runtime::ensure_runtime(kind, |msg| tracing::info!("{msg}")).await?;

        let port = Self::free_port();
        let mut cmd = Command::new(&bin);
        cmd.arg("--model")
            .arg(model_path)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(params.n_ctx.to_string())
            .arg("--batch-size")
            .arg(params.n_batch.to_string())
            .arg("--ubatch-size")
            .arg(params.n_ubatch.to_string());

        if params.n_gpu_layers != 0 {
            cmd.arg("--n-gpu-layers")
                .arg(params.n_gpu_layers.to_string());
        }
        if let Some(ts) = &params.tensor_split {
            cmd.arg("--tensor-split").arg(ts);
        }
        if let Some(nodes) = &params.rpc_nodes {
            cmd.arg("--rpc").arg(nodes);
        }
        if params.n_threads > 0 {
            cmd.arg("--threads").arg(params.n_threads.to_string());
        }
        match params.flash_attn.to_ascii_lowercase().as_str() {
            "on" | "enabled" | "true" | "1" => {
                cmd.arg("--flash-attn");
            }
            _ => {}
        }
        if params.cache_type_k != "f16" {
            cmd.arg("--cache-type-k").arg(&params.cache_type_k);
        }
        if params.cache_type_v != "f16" {
            cmd.arg("--cache-type-v").arg(&params.cache_type_v);
        }

        cmd.stdout(Stdio::null()).stderr(Stdio::null());
        #[cfg(windows)]
        {
            cmd.creation_flags(0x08000000);
        }

        let child = cmd
            .spawn()
            .map_err(|e| Ml5Error::Backend(format!("failed to spawn llama-server: {e}")))?;

        *self.port.lock().await = port;
        *self.child.lock().await = Some(child);
        tracing::info!(?kind, port, "llama-server spawned");

        self.wait_ready().await?;
        Ok(())
    }

    async fn unload(&self) -> Result<()> {
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
            tracing::info!("llama-server stopped");
        }
        Ok(())
    }

    async fn chat(&self, req: ChatRequest) -> Result<TokenStream> {
        let url = format!("{}/v1/chat/completions", self.base_url().await);
        let body = serde_json::json!({
            "messages": req.messages.iter().map(|m| serde_json::json!({"role": m.role, "content": m.content})).collect::<Vec<_>>(),
            "stream": true,
            "temperature": req.params.temperature,
            "top_p": req.params.top_p,
            "top_k": req.params.top_k,
            "min_p": req.params.min_p,
            "repeat_penalty": req.params.repeat_penalty,
            "frequency_penalty": req.params.frequency_penalty,
            "presence_penalty": req.params.presence_penalty,
            "seed": req.params.seed,
            "max_tokens": req.params.max_tokens,
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Ml5Error::Backend(format!("chat request failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Ml5Error::Backend(format!("llama-server error: {text}")));
        }

        let stream = resp.bytes_stream();
        let out = async_stream::stream! {
            let mut buf = String::new();
            let mut stream = std::pin::pin!(stream);
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk { Ok(c) => c, Err(e) => { yield Err(Ml5Error::Backend(e.to_string())); return; } };
                buf.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(end) = buf.find("\n\n") {
                    let block = buf[..end].to_string();
                    buf.drain(..end + 2);
                    for line in block.lines() {
                        let Some(data) = line.trim_end_matches('\r').strip_prefix("data:") else { continue };
                        let data = data.trim();
                        if data == "[DONE]" { yield Ok(StreamChunk { done: true, ..Default::default() }); return; }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                                yield Ok(StreamChunk { text: content.to_string(), ..Default::default() });
                            }
                            if v["choices"][0]["finish_reason"].as_str().is_some() {
                                yield Ok(StreamChunk { done: true, ..Default::default() });
                            }
                        }
                    }
                }
            }
            yield Ok(StreamChunk { done: true, ..Default::default() });
        };
        Ok(out.boxed())
    }

    async fn generate(&self, req: GenerateRequest) -> Result<TokenStream> {
        let url = format!("{}/v1/completions", self.base_url().await);
        let body = serde_json::json!({
            "prompt": req.prompt,
            "stream": true,
            "temperature": req.params.temperature,
            "top_p": req.params.top_p,
            "top_k": req.params.top_k,
            "min_p": req.params.min_p,
            "repeat_penalty": req.params.repeat_penalty,
            "frequency_penalty": req.params.frequency_penalty,
            "presence_penalty": req.params.presence_penalty,
            "seed": req.params.seed,
            "max_tokens": req.params.max_tokens,
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Ml5Error::Backend(format!("generate request failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Ml5Error::Backend(format!("llama-server error: {text}")));
        }

        let stream = resp.bytes_stream();
        let out = async_stream::stream! {
            let mut buf = String::new();
            let mut stream = std::pin::pin!(stream);
            while let Some(chunk) = stream.next().await {
                let chunk = match chunk { Ok(c) => c, Err(e) => { yield Err(Ml5Error::Backend(e.to_string())); return; } };
                buf.push_str(&String::from_utf8_lossy(&chunk));
                while let Some(end) = buf.find("\n\n") {
                    let block = buf[..end].to_string();
                    buf.drain(..end + 2);
                    for line in block.lines() {
                        let Some(data) = line.trim_end_matches('\r').strip_prefix("data:") else { continue };
                        let data = data.trim();
                        if data == "[DONE]" { yield Ok(StreamChunk { done: true, ..Default::default() }); return; }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(content) = v["choices"][0]["text"].as_str() {
                                yield Ok(StreamChunk { text: content.to_string(), ..Default::default() });
                            }
                        }
                    }
                }
            }
            yield Ok(StreamChunk { done: true, ..Default::default() });
        };
        Ok(out.boxed())
    }

    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse> {
        let url = format!("{}/v1/embeddings", self.base_url().await);
        let body = serde_json::json!({ "input": req.input });

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Ml5Error::Backend(format!("embed request failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Ml5Error::Backend(format!(
                "llama-server embed error: {text}"
            )));
        }

        let v: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| Ml5Error::Backend(format!("bad embed response: {e}")))?;
        let mut embeddings = Vec::new();
        if let Some(data) = v["data"].as_array() {
            for item in data {
                if let Some(emb) = item["embedding"].as_array() {
                    embeddings.push(
                        emb.iter()
                            .filter_map(|x| x.as_f64().map(|f| f as f32))
                            .collect(),
                    );
                }
            }
        }
        Ok(EmbedResponse {
            embeddings,
            usage: Usage::default(),
        })
    }
}
