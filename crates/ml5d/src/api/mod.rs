pub mod anthropic;
pub mod native;
pub mod ollama;
pub mod openai;
pub mod responses;

use crate::state::AppState;
use axum::routing::{delete, get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(root))
        .route("/api/generate", post(native::generate))
        .route("/api/chat", post(native::chat))
        .route("/api/embed", post(native::embed))
        .route("/api/models", get(native::list_models))
        .route("/api/pull", post(native::pull))
        .route("/api/delete", delete(native::delete_model))
        .route("/api/unload", post(native::unload_model))
        .route("/api/rename", post(native::rename_model))
        .route("/api/status", get(native::status))
        .route("/api/metrics", get(native::metrics))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        .route("/v1/embeddings", post(openai::embeddings))
        .route("/v1/models", get(openai::list_models))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/ollama/api/generate", post(ollama::generate))
        .route("/ollama/api/chat", post(ollama::chat))
        .route("/ollama/api/tags", get(ollama::tags))
        .route("/ollama/api/show", post(ollama::show))
        .route("/ollama/api/version", get(ollama::version))
        .route("/ollama/api/embeddings", post(ollama::embeddings))
        .route("/ollama/api/ps", get(ollama::ps))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn root() -> axum::response::Html<String> {
    axum::response::Html(
        r#"<!DOCTYPE html>
<html><head><title>CORe ML5</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#0f1115;color:#e6e6e6}
h1{color:#6cf}
h2{color:#9df;border-bottom:1px solid #333;padding-bottom:.3rem;margin-top:2rem}
code{background:#1b1f27;padding:.15rem .4rem;border-radius:4px}
table{border-collapse:collapse;width:100%}
td,th{padding:.4rem .6rem;border-bottom:1px solid #222;text-align:left}
th{color:#8af}
.m{color:#888}
</style></head><body>
<h1>CORe ML5 is up</h1>

<h2>ML5 Native</h2>
<table>
<tr><th>Method</th><th>Path</th><th>Description</th></tr>
<tr><td>POST</td><td><code>/api/chat</code></td><td class="m">Chat (SSE)</td></tr>
<tr><td>POST</td><td><code>/api/generate</code></td><td class="m">Raw completion (SSE)</td></tr>
<tr><td>POST</td><td><code>/api/embed</code></td><td class="m">Embeddings</td></tr>
<tr><td>GET</td><td><code>/api/models</code></td><td class="m">List installed models</td></tr>
<tr><td>POST</td><td><code>/api/pull</code></td><td class="m">Pull a model (SSE progress)</td></tr>
<tr><td>DELETE</td><td><code>/api/delete</code></td><td class="m">Delete a model</td></tr>
<tr><td>POST</td><td><code>/api/unload</code></td><td class="m">Unload a model from memory</td></tr>
<tr><td>POST</td><td><code>/api/rename</code></td><td class="m">Rename a model's friendly name</td></tr>
<tr><td>GET</td><td><code>/api/status</code></td><td class="m">Daemon status</td></tr>
<tr><td>GET</td><td><code>/api/metrics</code></td><td class="m">Metrics</td></tr>
</table>

<h2>Model Formats</h2>
<table>
<tr><th>Format</th><th>Backend</th><th>Description</th></tr>
<tr><td>GGUF</td><td><code>llama.cpp</code></td><td class="m">Quantized models (default)</td></tr>
<tr><td>Safetensors</td><td><code>candle</code></td><td class="m">Full HF models (Llama-family arch)</td></tr>
</table>

<h2>OpenAI-Compatible</h2>
<table>
<tr><th>Method</th><th>Path</th><th>Description</th></tr>
<tr><td>POST</td><td><code>/v1/chat/completions</code></td><td class="m">Chat completions (SSE)</td></tr>
<tr><td>POST</td><td><code>/v1/completions</code></td><td class="m">Text completions (SSE)</td></tr>
<tr><td>POST</td><td><code>/v1/embeddings</code></td><td class="m">Embeddings</td></tr>
<tr><td>GET</td><td><code>/v1/models</code></td><td class="m">List models</td></tr>
<tr><td>POST</td><td><code>/v1/responses</code></td><td class="m">Responses API (Codex, SSE)</td></tr>
</table>

<h2>Anthropic-Compatible</h2>
<table>
<tr><th>Method</th><th>Path</th><th>Description</th></tr>
<tr><td>POST</td><td><code>/v1/messages</code></td><td class="m">Messages API (SSE)</td></tr>
<tr><td>POST</td><td><code>/v1/messages/count_tokens</code></td><td class="m">Token counting</td></tr>
</table>

<h2>Ollama-Compatible <span class="m">(mounted under /ollama)</span></h2>
<table>
<tr><th>Method</th><th>Path</th><th>Description</th></tr>
<tr><td>POST</td><td><code>/ollama/api/generate</code></td><td class="m">Generate (Ollama shape)</td></tr>
<tr><td>POST</td><td><code>/ollama/api/chat</code></td><td class="m">Chat (Ollama shape)</td></tr>
<tr><td>GET</td><td><code>/ollama/api/tags</code></td><td class="m">List models</td></tr>
<tr><td>POST</td><td><code>/ollama/api/show</code></td><td class="m">Model info</td></tr>
<tr><td>GET</td><td><code>/ollama/api/version</code></td><td class="m">Version</td></tr>
<tr><td>POST</td><td><code>/ollama/api/embeddings</code></td><td class="m">Embeddings</td></tr>
<tr><td>GET</td><td><code>/ollama/api/ps</code></td><td class="m">Running models</td></tr>
</table>

<p class="m">LM Studio clients work out of the box via the OpenAI-compatible endpoints above.</p>
</body></html>"#
            .to_string(),
    )
}
