mod sse;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use std::io::{IsTerminal, Read, Write};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "ml5", version, about = "CORe ML5 CLI")]
struct Cli {
    #[arg(long, global = true, default_value = "http://127.0.0.1:11435", env = "ML5_HOST")]
    host: String,

    #[arg(long, short, global = true, help = "Only print model output and errors")]
    quiet: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone, Default)]
struct ChatOpts {
    #[arg(long, help = "Context size override (0 = automatic)")]
    ctx_size: Option<u32>,

    #[arg(long, allow_hyphen_values = true, help = "GPU layers: auto, all, or a number")]
    gpu_layers: Option<String>,

    #[arg(long, value_parser = ["auto", "on", "off"], help = "Flash attention")]
    flash_attn: Option<String>,

    #[arg(long, help = "KV cache K type (f16/q8_0/q4_0/...)")]
    cache_type_k: Option<String>,

    #[arg(long, help = "KV cache V type")]
    cache_type_v: Option<String>,

    #[arg(long)]
    temperature: Option<f32>,

    #[arg(long)]
    top_p: Option<f32>,

    #[arg(long)]
    top_k: Option<i32>,

    #[arg(long)]
    min_p: Option<f32>,

    #[arg(long)]
    repeat_penalty: Option<f32>,

    #[arg(long)]
    max_tokens: Option<u32>,

    #[arg(long, value_parser = ["on", "off"], help = "Turn the model's reasoning on or off")]
    think: Option<String>,

    #[arg(long, help = "How long the model stays loaded afterwards (e.g. 5m, 1h, -1 = forever)")]
    keep_alive: Option<String>,
}

impl ChatOpts {
    fn to_map(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut o = serde_json::Map::new();
        let mut set = |k: &str, v: serde_json::Value| {
            if !v.is_null() {
                o.insert(k.into(), v);
            }
        };
        set("n_ctx", serde_json::json!(self.ctx_size));
        set("n_gpu_layers", serde_json::json!(self.gpu_layers));
        set("flash_attn", serde_json::json!(self.flash_attn));
        set("cache_type_k", serde_json::json!(self.cache_type_k));
        set("cache_type_v", serde_json::json!(self.cache_type_v));
        set("temperature", serde_json::json!(self.temperature));
        set("top_p", serde_json::json!(self.top_p));
        set("top_k", serde_json::json!(self.top_k));
        set("min_p", serde_json::json!(self.min_p));
        set("repeat_penalty", serde_json::json!(self.repeat_penalty));
        set("max_tokens", serde_json::json!(self.max_tokens));
        set("think", serde_json::json!(self.think.as_deref().map(|t| t == "on")));
        set("keep_alive", serde_json::json!(self.keep_alive));
        o
    }
}

#[derive(Subcommand)]
enum Cmd {
    #[command(about = "List installed models")]
    List,

    #[command(about = "Show daemon status")]
    Status {
        #[arg(long, help = "Print the full machine-readable status")]
        json: bool,
    },

    #[command(about = "Show models currently loaded in memory")]
    Ps,

    #[command(about = "Load a model into memory now")]
    Load {
        model: String,
        #[arg(long, help = "Context size (0 = automatic)")]
        ctx_size: Option<u32>,
        #[arg(long, help = "How long it stays loaded (e.g. 30m, -1 = forever)")]
        keep_alive: Option<String>,
    },

    #[command(about = "Release a loaded model from memory")]
    Unload { model: String },

    #[command(
        visible_alias = "run",
        about = "Chat with a model (omit --message for interactive chat, or pipe a prompt)"
    )]
    Chat {
        model: String,
        #[arg(long)]
        message: Option<String>,

        #[arg(long = "image", help = "Attach an image file (repeatable; needs a vision model)")]
        images: Vec<std::path::PathBuf>,

        #[arg(long, help = "Unload the model after the chat ends")]
        unload_after: bool,

        #[arg(long, help = "Instruction prepended to the conversation")]
        system: Option<String>,

        #[command(flatten)]
        opts: Box<ChatOpts>,
    },

    #[command(
        about = "Pull a model: `ml5 pull hf:owner/repo[:quant]`, `ml5 pull hf:owner/repo/file.gguf`, or `ml5 pull <name>`"
    )]
    Pull {
        target: String,

        #[arg(long, help = "Force registry: hf | core | routes")]
        registry: Option<String>,

        #[arg(long, help = "Hugging Face API token for gated/private repos")]
        token: Option<String>,

        #[arg(long, help = "Quantization preference: q4_k_m, q5_k_m, q8_0, f16, etc.")]
        quant: Option<String>,
    },

    #[command(about = "List downloads in progress")]
    Pulls,

    #[command(about = "Cancel a download (id from `ml5 pulls`)")]
    Cancel { id: String },

    #[command(about = "Check for and install updates")]
    Update {
        #[arg(long, help = "Only check, don't download")]
        check: bool,
    },

    #[command(about = "Delete a model")]
    Delete { model: String },

    #[command(about = "Rename a model's friendly name")]
    Rename { model: String, new_name: String },

    #[command(about = "Hugging Face account: authenticate / deauthenticate / status")]
    Hf {
        #[command(subcommand)]
        action: HfAction,
    },

    #[command(about = "llama.cpp runtime: detect hardware, show or download llama-server")]
    Backend {
        #[command(subcommand)]
        action: BackendAction,
    },
}

#[derive(Subcommand)]
enum BackendAction {
    #[command(about = "Detect GPU hardware and recommend a runtime build")]
    Detect,

    #[command(about = "Show which llama-server ML5 will use and which builds are installed")]
    Status,

    #[command(about = "Download a llama.cpp runtime build into ~/.ml5/runtime")]
    Download {
        #[arg(long, value_parser = ["auto", "cpu", "vulkan", "cuda", "rocm", "metal"], default_value = "auto")]
        kind: String,
    },
}

#[derive(Subcommand)]
enum HfAction {
    #[command(about = "Save a Hugging Face API token (used on pulls of gated/private repos)")]
    Authenticate { token: String },

    #[command(about = "Remove the saved Hugging Face token")]
    Deauthenticate,

    #[command(about = "Show whether a token is saved")]
    Status,
}

fn resolve_pull_target(registry: Option<String>, target: &str) -> (String, String) {
    if let Some(r) = registry {
        return (r, target.strip_prefix("hf:").unwrap_or(target).to_string());
    }
    if let Some(rest) = target.strip_prefix("hf:") {
        return ("hf".into(), rest.to_string());
    }
    if target.contains("huggingface.co/") || target.contains("hf.co/") {
        return ("hf".into(), target.to_string());
    }
    if target.split('/').count() >= 2 && !target.contains("://") {
        return ("hf".into(), target.to_string());
    }
    ("routes".into(), target.to_string())
}

fn hf_token_path() -> std::path::PathBuf {
    ml5_core::config::Config::ml5_home().join("hf_token")
}

fn load_hf_token() -> Option<String> {
    std::fs::read_to_string(hf_token_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn save_private(path: &std::path::Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        f.write_all(content.as_bytes())?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    std::fs::write(path, content)?;
    Ok(())
}

fn humansize(b: u64) -> String {
    ml5_core::config::humansize(b)
}

async fn checked_response(request: reqwest::RequestBuilder) -> Result<reqwest::Response> {
    let response = request.send().await.map_err(|e| {
        if e.is_connect() {
            anyhow!("Cannot reach ML5. Start it with `systemctl --user start ml5d` or `ml5d --background`, or check --host. ({e})")
        } else {
            anyhow!(e)
        }
    })?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| {
                v["error"]
                    .as_str()
                    .or_else(|| v["error"]["message"].as_str())
                    .map(str::to_string)
            })
            .unwrap_or(text);
        bail!("{status}: {message}");
    }
    Ok(response)
}

fn encode_image(path: &std::path::Path) -> Result<String> {
    let data = std::fs::read(path).with_context(|| format!("cannot read image {}", path.display()))?;
    if data.len() > 48 * 1024 * 1024 {
        bail!("{} is larger than 48 MB", path.display());
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(data))
}

struct ProgressCleanup(indicatif::ProgressBar);

impl Drop for ProgressCleanup {
    fn drop(&mut self) {
        self.0.disable_steady_tick();
        self.0.finish_and_clear();
    }
}

struct Reply {
    text: String,
    thinking: String,
}

async fn stream_chat_body(
    client: &reqwest::Client,
    body: serde_json::Value,
    host: &str,
    quiet: bool,
) -> Result<Reply> {
    let started = Instant::now();
    let mut first_token = None;
    let progress = indicatif::ProgressBar::new_spinner();
    if quiet || !std::io::stderr().is_terminal() {
        progress.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }
    progress.set_style(indicatif::ProgressStyle::with_template("{spinner} {msg} [{elapsed_precise}]")?);
    progress.set_message("Connecting to ML5");
    progress.enable_steady_tick(Duration::from_millis(100));
    let _cleanup = ProgressCleanup(progress.clone());
    let tty = std::io::stdout().is_terminal();
    let resp = checked_response(client.post(format!("{host}/api/chat")).json(&body)).await?;
    let mut stream = resp.bytes_stream();
    let mut reply = Reply {
        text: String::new(),
        thinking: String::new(),
    };
    let mut in_thinking = false;
    let mut decoder = sse::Decoder::default();
    let stdout = std::io::stdout();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        for data in decoder.push(&chunk)? {
            let v: serde_json::Value = serde_json::from_str(&data).context("Invalid chat event from daemon")?;
            if let Some(err) = v["error"].as_str() {
                return Err(anyhow!("inference error: {err}"));
            }
            if let Some(status) = v["status"].as_str() {
                let message = match status {
                    "loading_model" => format!("Loading {} into memory", body["model"].as_str().unwrap_or("model")),
                    "queued" => "Model ready; waiting for the runtime".into(),
                    "processing_prompt" => format!("Processing prompt: {}/{} tokens", v["completed"], v["total"]),
                    other => other.replace('_', " "),
                };
                progress.set_message(message.clone());
                if !quiet && !std::io::stderr().is_terminal() {
                    eprintln!("{message}");
                }
                continue;
            }
            let thinking = v["message"]["thinking"].as_str().unwrap_or("");
            let content = v["message"]["content"].as_str().unwrap_or("");
            if (!thinking.is_empty() || !content.is_empty()) && first_token.is_none() {
                first_token = Some(started.elapsed());
                progress.finish_and_clear();
            }
            let mut h = stdout.lock();
            if !thinking.is_empty() {
                reply.thinking.push_str(thinking);
                if !quiet {
                    if !in_thinking && tty {
                        write!(h, "\x1b[2;3m")?;
                    }
                    in_thinking = true;
                    write!(h, "{thinking}")?;
                }
            }
            if !content.is_empty() {
                if in_thinking {
                    if tty {
                        write!(h, "\x1b[0m")?;
                    }
                    writeln!(h)?;
                    in_thinking = false;
                }
                reply.text.push_str(content);
                write!(h, "{content}")?;
            }
            h.flush()?;
            drop(h);
            if v["done"].as_bool() == Some(true) {
                progress.finish_and_clear();
                if in_thinking && tty {
                    print!("\x1b[0m");
                }
                println!();
                if reply.text.trim().is_empty() && reply.thinking.trim().is_empty() {
                    bail!("Model ended without visible text. Check its chat template or increase --max-tokens if the token limit was reached.");
                }
                if !quiet {
                    let usage = &v["usage"];
                    let tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
                    let generation_ms = usage["generation_ms"].as_u64().unwrap_or(0);
                    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
                    let rate = if tokens > 1 && generation_ms >= 100 {
                        format!("{:.1} tok/s", (tokens - 1) as f64 * 1000.0 / generation_ms as f64)
                    } else {
                        "rate n/a (short reply)".into()
                    };
                    eprintln!(
                        "\x1b[90m{prompt_tokens} prompt tokens · {tokens} generated · {rate} · {:.2}s total · {:.2}s to first token\x1b[0m",
                        started.elapsed().as_secs_f64(),
                        first_token.unwrap_or_else(|| started.elapsed()).as_secs_f64()
                    );
                    if v["finish_reason"].as_str() == Some("length") {
                        eprintln!("Stopped at the token/context limit. Increase --max-tokens or --ctx-size for longer replies.");
                    }
                }
                return Ok(reply);
            }
        }
    }
    bail!("Connection ended before the response completed; retry the request.")
}

fn print_chat_help() {
    println!("Commands: /exit /quit (leave)  /clear (reset)  /model <name> (switch)");
    println!("          /set system <text>  /set system show  /set system clear");
    println!("          /think on|off  /image <path> (attach to next message)  /help");
}

async fn unload_model(client: &reqwest::Client, host: &str, model: &str) -> Result<()> {
    checked_response(
        client
            .post(format!("{host}/api/unload"))
            .json(&serde_json::json!({ "model": model })),
    )
    .await?;
    Ok(())
}

async fn interactive_chat(
    client: &reqwest::Client,
    host: &str,
    model: &str,
    unload_after: bool,
    mut opts: serde_json::Map<String, serde_json::Value>,
    system: Option<String>,
    quiet: bool,
) -> Result<()> {
    let mut model = model.to_string();
    let mut system = system;
    let mut history: Vec<serde_json::Value> = Vec::new();
    let mut pending_images: Vec<String> = Vec::new();
    println!("ML5 chat — {model}");
    print_chat_help();
    if let Some(system) = system.as_deref() {
        println!("\x1b[90m(system prompt: {system})\x1b[0m");
    }
    loop {
        {
            let mut h = std::io::stdout().lock();
            print!("\n\x1b[1;36myou\x1b[0m \x1b[90m>\x1b[0m ");
            let _ = h.flush();
        }
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/exit" | "/quit" => break,
            "/help" => {
                print_chat_help();
                continue;
            }
            "/clear" => {
                history.clear();
                pending_images.clear();
                println!("\x1b[90m(conversation cleared; system prompt kept)\x1b[0m");
                continue;
            }
            "/set system show" => {
                match system.as_deref() {
                    Some(value) => println!("\x1b[90m(system prompt: {value})\x1b[0m"),
                    None => println!("\x1b[90m(no system prompt set)\x1b[0m"),
                }
                continue;
            }
            "/set system clear" => {
                system = None;
                println!("\x1b[90m(system prompt cleared)\x1b[0m");
                continue;
            }
            "/think on" | "/think off" => {
                opts.insert("think".into(), (input == "/think on").into());
                println!("\x1b[90m(reasoning {})\x1b[0m", if input == "/think on" { "on" } else { "off" });
                continue;
            }
            _ if input.starts_with("/image ") => {
                let path = std::path::PathBuf::from(input.trim_start_matches("/image ").trim());
                match encode_image(&path) {
                    Ok(img) => {
                        pending_images.push(img);
                        println!("\x1b[90m(attached {}; it goes with your next message)\x1b[0m", path.display());
                    }
                    Err(e) => println!("\x1b[31m{e}\x1b[0m"),
                }
                continue;
            }
            _ if input.starts_with("/set system ") => {
                let value = input.trim_start_matches("/set system ").trim();
                if value.is_empty() {
                    println!("\x1b[90musage: /set system <text> | /set system show | /set system clear\x1b[0m");
                } else {
                    system = Some(value.to_string());
                    println!("\x1b[90m(system prompt set)\x1b[0m");
                }
                continue;
            }
            _ if input.starts_with("/model ") => {
                let m = input.trim_start_matches("/model ").trim();
                if m.is_empty() {
                    println!("\x1b[90musage: /model <name>\x1b[0m");
                } else {
                    model = m.to_string();
                    println!("\x1b[90m(switched to {model}; conversation kept)\x1b[0m");
                }
                continue;
            }
            _ if input.starts_with('/') => {
                println!("\x1b[90munknown command — /help\x1b[0m");
                continue;
            }
            _ => {}
        }
        let mut user = serde_json::json!({ "role": "user", "content": input });
        if !pending_images.is_empty() {
            user["images"] = serde_json::json!(pending_images);
        }
        history.push(user);
        {
            let mut h = std::io::stdout().lock();
            print!("\x1b[1;32mml5\x1b[0m  \x1b[90m>\x1b[0m ");
            let _ = h.flush();
        }
        let mut messages = Vec::new();
        if let Some(s) = system.as_deref().filter(|s| !s.is_empty()) {
            messages.push(serde_json::json!({ "role": "system", "content": s }));
        }
        messages.extend(history.iter().cloned());
        let mut body = serde_json::json!({ "model": model, "messages": messages, "stream": true });
        if let Some(o) = body.as_object_mut() {
            o.extend(opts.clone());
        }
        let result = tokio::select! {
            result = stream_chat_body(client, body, host, quiet) => result,
            _ = tokio::signal::ctrl_c() => Err(anyhow!("Interrupted. You can enter another prompt.")),
        };
        match result {
            Ok(reply) => {
                pending_images.clear();
                let mut msg = serde_json::json!({ "role": "assistant", "content": reply.text });
                if !reply.thinking.is_empty() {
                    msg["thinking"] = reply.thinking.into();
                }
                history.push(msg);
            }
            Err(e) => {
                eprintln!("\nerror: {e}");
                history.pop();
            }
        }
    }
    println!("\x1b[90m(bye)\x1b[0m");
    if unload_after {
        unload_model(client, host, &model).await?;
        println!("\x1b[90m(model unloaded)\x1b[0m");
    }
    Ok(())
}

fn caps_of(m: &serde_json::Value) -> String {
    m["capabilities"]
        .as_array()
        .map(|c| c.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(","))
        .unwrap_or_default()
}

async fn get_json(client: &reqwest::Client, url: String) -> Result<serde_json::Value> {
    Ok(checked_response(client.get(url)).await?.json().await?)
}

fn print_loaded(status: &serde_json::Value) {
    let loaded = status["loaded"].as_array().cloned().unwrap_or_default();
    if loaded.is_empty() {
        println!("No models loaded.");
        return;
    }
    let mut table = comfy_table::Table::new();
    table.set_header(vec!["NAME", "CONTEXT", "SLOTS", "GPU LAYERS", "MEMORY", "ACTIVE", "UNLOADS IN"]);
    for l in loaded {
        let expires = match l["expires_in_secs"].as_i64() {
            Some(s) => format!("{}m{:02}s", s / 60, s % 60),
            None if l["active_requests"].as_u64().unwrap_or(0) > 0 => "busy".into(),
            None => "never".into(),
        };
        table.add_row(vec![
            l["name"].as_str().unwrap_or("?").to_string(),
            l["n_ctx"].to_string(),
            l["n_parallel"].to_string(),
            l["gpu_layers"].as_str().unwrap_or("?").to_string(),
            humansize(l["memory_bytes"].as_u64().unwrap_or(0)),
            l["active_requests"].to_string(),
            expires,
        ]);
    }
    println!("{table}");
}

async fn run_pull(
    client: &reqwest::Client,
    host: &str,
    quiet: bool,
    body: serde_json::Value,
) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    let resp = checked_response(client.post(format!("{host}/api/pull")).json(&body)).await?;
    let bar = ProgressBar::new(0);
    if quiet || !std::io::stderr().is_terminal() {
        bar.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }
    bar.set_style(
        ProgressStyle::with_template("{msg:30!} {bytes}/{total_bytes} [{bar:40.cyan/blue}] {bytes_per_sec} ETA {eta}")
            .unwrap_or_else(|_| ProgressStyle::default_bar()),
    );
    let _cleanup = ProgressCleanup(bar.clone());
    let mut stream = resp.bytes_stream();
    let mut decoder = sse::Decoder::default();
    while let Some(chunk) = stream.next().await {
        for data in decoder.push(&chunk?)? {
            let v: serde_json::Value = serde_json::from_str(&data).context("Invalid pull event from daemon")?;
            if let Some(error) = v["error"].as_str() {
                bar.finish_and_clear();
                bail!("Pull failed: {error}");
            }
            match v["event"].as_str() {
                Some("started") if !quiet => {
                    eprintln!("\x1b[90mdownload id {} (cancel with `ml5 cancel <id>`)\x1b[0m", v["id"].as_str().unwrap_or("?"));
                }
                Some("resolving") if !quiet => {
                    let target = v["target"].as_str().unwrap_or("");
                    if target.contains("->") {
                        bar.println(format!("\x1b[36m(auto-resolved)\x1b[0m {target}"));
                    } else {
                        bar.println(format!("Resolving {target}"));
                    }
                }
                Some("downloading") => {
                    let file = v["file"].as_str().unwrap_or("model");
                    if !quiet {
                        bar.println(format!("Downloading {file}"));
                    }
                    bar.set_message(file.rsplit('/').next().unwrap_or(file).to_string());
                }
                Some("progress") => {
                    if let Some(t) = v["total"].as_u64() {
                        bar.set_length(t);
                    }
                    bar.set_position(v["downloaded"].as_u64().unwrap_or(0));
                }
                Some("verifying") => bar.set_message("verifying"),
                Some("done") => {
                    bar.finish_and_clear();
                    let model = v["model"].as_str().unwrap_or("model");
                    println!("pulled {model}");
                    if !quiet {
                        eprintln!("Ready to chat: ml5 run {model}");
                    }
                    return Ok(());
                }
                _ => {}
            }
        }
    }
    bail!("Download connection ended before completion; run the pull again to resume.")
}

async fn backend_command(action: BackendAction, quiet: bool) -> Result<()> {
    use ml5_core::runtime;
    match action {
        BackendAction::Detect => {
            let gpus = runtime::detect_gpus();
            if gpus.is_empty() {
                println!("No GPU detected — the CPU build will be used.");
            } else {
                println!("Detected GPUs:");
                for g in &gpus {
                    println!("  {} — {}", g.vendor, g.name);
                }
                if !runtime::vulkan_loader_present() && !cfg!(target_os = "macos") {
                    println!("Vulkan loader not found; install your distribution's Vulkan package (e.g. libvulkan1 and mesa-vulkan-drivers) to use the GPU.");
                }
            }
            println!("Recommended runtime: {}", runtime::detect_gpu().label());
        }
        BackendAction::Status => {
            let info = runtime::runtime_info(None);
            println!("llama.cpp release: {}", info.release_tag);
            match &info.server {
                Some(p) => println!("llama-server in use: {}", p.display()),
                None => println!("llama-server: not installed (downloaded automatically on first model load, or run `ml5 backend download`)"),
            }
            println!("Detected hardware: {}", info.detected.label());
            if info.installed.is_empty() {
                println!("Downloaded runtimes: none");
            } else {
                println!("Downloaded runtimes:");
                for p in &info.installed {
                    println!("  {}", p.display());
                }
            }
        }
        BackendAction::Download { kind } => {
            let kind = if kind == "auto" {
                runtime::detect_gpu()
            } else {
                runtime::GpuKind::parse(&kind).ok_or_else(|| anyhow!("unknown runtime kind {kind}"))?
            };
            let mut report = |msg: String| {
                if !quiet {
                    eprintln!("{msg}");
                }
            };
            let server = runtime::download_runtime(kind, &mut report)
                .await
                .context("Runtime download failed")?;
            println!("llama-server ready: {}", server.display());
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let host = cli.host.trim_end_matches('/').to_string();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()?;

    match cli.cmd {
        Cmd::List => {
            let resp = get_json(&client, format!("{host}/api/models")).await?;
            let models = resp["models"].as_array().cloned().unwrap_or_default();
            if models.is_empty() {
                eprintln!("No models installed. Get one with `ml5 pull hf:owner/repo` or `ml5 pull <name>`.");
                return Ok(());
            }
            let mut table = comfy_table::Table::new();
            table.set_header(vec!["NAME", "FRIENDLY NAME", "SIZE", "PARAMS", "QUANT", "CONTEXT", "CAPABILITIES", "LOADED"]);
            for m in models {
                let d = &m["details"];
                table.add_row(vec![
                    m["name"].as_str().unwrap_or("?").to_string(),
                    m["friendly_name"].as_str().unwrap_or("").to_string(),
                    humansize(m["size_bytes"].as_u64().unwrap_or(0)),
                    d["parameter_size"].as_str().unwrap_or("").to_string(),
                    d["quantization"].as_str().unwrap_or("").to_string(),
                    d["context_length"].as_u64().map(|c| c.to_string()).unwrap_or_default(),
                    caps_of(&m),
                    if m["loaded"].as_bool() == Some(true) {
                        format!("yes ({} ctx)", m["n_ctx"])
                    } else {
                        String::new()
                    },
                ]);
            }
            println!("{table}");
        }
        Cmd::Status { json } => {
            let resp = get_json(&client, format!("{host}/api/status")).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp)?);
                return Ok(());
            }
            println!("ML5 {} — {host}", resp["version"].as_str().unwrap_or("?"));
            println!("Uptime: {}s | Installed models: {}", resp["uptime_secs"], resp["model_count"]);
            println!("Model store: {}", resp["models_dir"].as_str().unwrap_or("unknown"));
            println!(
                "llama-server: {} (llama.cpp {})",
                resp["runtime"]["server"].as_str().unwrap_or("not installed yet"),
                resp["runtime"]["release_tag"].as_str().unwrap_or("?")
            );
            let mem = &resp["memory"];
            println!(
                "Memory: {} free of {} RAM{} | held by loaded models: {}",
                humansize(mem["available_bytes"].as_u64().unwrap_or(0)),
                humansize(mem["total_bytes"].as_u64().unwrap_or(0)),
                match mem["gpu_total_bytes"].as_u64() {
                    Some(t) if t > 0 => format!(" | {} free of {} VRAM", humansize(mem["gpu_free_bytes"].as_u64().unwrap_or(0)), humansize(t)),
                    _ => String::new(),
                },
                humansize(mem["committed_bytes"].as_u64().unwrap_or(0))
            );
            let ctx = resp["defaults"]["n_ctx"].as_u64().unwrap_or(0);
            println!(
                "Defaults: context {} | GPU layers {} | flash attention {} | slots {} | up to {} models | keep-alive {}",
                if ctx == 0 { format!("auto (≤{})", resp["auto_ctx_max"]) } else { ctx.to_string() },
                resp["defaults"]["n_gpu_layers"].as_str().map(str::to_string).unwrap_or_else(|| resp["defaults"]["n_gpu_layers"].to_string()),
                resp["defaults"]["flash_attn"].as_str().unwrap_or("auto"),
                resp["n_parallel"],
                resp["max_loaded_models"],
                match resp["keep_alive_secs"].as_i64() {
                    Some(k) if k < 0 => "forever".to_string(),
                    Some(k) => format!("{k}s"),
                    None => "?".into(),
                }
            );
            if let Some(loading) = resp["loading"].as_array().filter(|l| !l.is_empty()) {
                println!("Loading: {}", loading.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "));
            }
            print_loaded(&resp);
        }
        Cmd::Ps => {
            let resp = get_json(&client, format!("{host}/api/status")).await?;
            print_loaded(&resp);
        }
        Cmd::Load { model, ctx_size, keep_alive } => {
            let mut body = serde_json::json!({ "model": model });
            if let Some(c) = ctx_size {
                body["n_ctx"] = c.into();
            }
            if let Some(k) = keep_alive {
                body["keep_alive"] = k.into();
            }
            if !cli.quiet {
                eprintln!("Loading {model}…");
            }
            let resp: serde_json::Value = checked_response(
                reqwest::Client::new().post(format!("{host}/api/load")).json(&body),
            )
            .await?
            .json()
            .await?;
            println!("Loaded {}", resp["loaded"].as_str().unwrap_or(&model));
        }
        Cmd::Unload { model } => {
            unload_model(&client, &host, &model).await?;
            println!("Unloaded {model}");
        }
        Cmd::Chat {
            model,
            message,
            images,
            unload_after,
            system,
            opts,
        } => {
            let opts = opts.to_map();
            let encoded = images.iter().map(|p| encode_image(p)).collect::<Result<Vec<_>>>()?;
            let message = if message.is_none() && !std::io::stdin().is_terminal() {
                let mut input = String::new();
                std::io::stdin().read_to_string(&mut input)?;
                if input.trim().is_empty() {
                    bail!("Piped prompt is empty. Pass --message or enter a prompt interactively.");
                }
                Some(input)
            } else {
                message
            };
            match message {
                Some(m) => {
                    let mut messages = Vec::new();
                    if let Some(s) = system.as_deref().filter(|s| !s.is_empty()) {
                        messages.push(serde_json::json!({ "role": "system", "content": s }));
                    }
                    let mut user = serde_json::json!({ "role": "user", "content": m });
                    if !encoded.is_empty() {
                        user["images"] = serde_json::json!(encoded);
                    }
                    messages.push(user);
                    let mut body = serde_json::json!({ "model": model, "messages": messages, "stream": true });
                    if let Some(o) = body.as_object_mut() {
                        o.extend(opts);
                    }
                    let result = tokio::select! {
                        result = stream_chat_body(&client, body, &host, cli.quiet) => result.map(|_| ()),
                        _ = tokio::signal::ctrl_c() => Err(anyhow!("Interrupted")),
                    };
                    if unload_after {
                        unload_model(&client, &host, &model).await?;
                        if !cli.quiet {
                            eprintln!("Model unloaded");
                        }
                    }
                    result?;
                }
                None => {
                    if !encoded.is_empty() {
                        bail!("--image needs --message (in interactive chat use /image <path>)");
                    }
                    interactive_chat(&client, &host, &model, unload_after, opts, system, cli.quiet).await?;
                }
            }
        }
        Cmd::Pull {
            target,
            registry,
            token,
            quant,
        } => {
            let (registry, reference) = resolve_pull_target(registry, &target);
            let mut body = serde_json::json!({ "model": reference, "registry": registry });
            if let Some(t) = token.or_else(load_hf_token) {
                body["token"] = t.into();
            }
            if let Some(q) = quant {
                body["quant"] = q.into();
            }
            run_pull(&client, &host, cli.quiet, body).await?;
        }
        Cmd::Pulls => {
            let resp = get_json(&client, format!("{host}/api/pulls")).await?;
            let pulls = resp["pulls"].as_array().cloned().unwrap_or_default();
            if pulls.is_empty() {
                println!("No downloads in progress.");
            } else {
                let mut table = comfy_table::Table::new();
                table.set_header(vec!["ID", "TARGET", "STAGE", "PROGRESS"]);
                for p in pulls {
                    let done = p["downloaded"].as_u64().unwrap_or(0);
                    let progress = match p["total"].as_u64() {
                        Some(t) if t > 0 => format!("{} / {} ({:.0}%)", humansize(done), humansize(t), done as f64 * 100.0 / t as f64),
                        _ => humansize(done),
                    };
                    table.add_row(vec![
                        p["id"].as_str().unwrap_or("?").to_string(),
                        p["target"].as_str().unwrap_or("?").to_string(),
                        p["stage"].as_str().unwrap_or("?").to_string(),
                        progress,
                    ]);
                }
                println!("{table}");
            }
        }
        Cmd::Cancel { id } => {
            checked_response(
                client
                    .post(format!("{host}/api/pull/cancel"))
                    .json(&serde_json::json!({ "id": id })),
            )
            .await?;
            println!("Cancelled download {id}");
        }
        Cmd::Delete { model } => {
            checked_response(
                client
                    .delete(format!("{host}/api/delete"))
                    .json(&serde_json::json!({ "model": model })),
            )
            .await?;
            println!("Deleted {model}");
        }
        Cmd::Rename { model, new_name } => {
            let v: serde_json::Value = checked_response(
                client
                    .post(format!("{host}/api/rename"))
                    .json(&serde_json::json!({ "model": model, "new_name": new_name })),
            )
            .await?
            .json()
            .await?;
            println!(
                "Renamed {} -> {}",
                v["renamed"].as_str().unwrap_or("?"),
                v["new_name"].as_str().unwrap_or("?")
            );
        }
        Cmd::Hf { action } => match action {
            HfAction::Authenticate { token } => {
                let path = hf_token_path();
                save_private(&path, token.trim())?;
                println!("Hugging Face token saved to {}", path.display());
            }
            HfAction::Deauthenticate => {
                let path = hf_token_path();
                if path.exists() {
                    std::fs::remove_file(&path)?;
                    println!("Hugging Face token removed");
                } else {
                    println!("No token saved");
                }
            }
            HfAction::Status => match load_hf_token() {
                Some(t) => {
                    let masked = if t.len() > 7 {
                        format!("{}...{}", &t[..3], &t[t.len() - 4..])
                    } else {
                        "***".to_string()
                    };
                    println!("Hugging Face token saved ({masked})");
                }
                None => println!("No Hugging Face token saved — pulls are anonymous"),
            },
        },
        Cmd::Backend { action } => backend_command(action, cli.quiet).await?,
        Cmd::Update { check } => update(&client, check).await?,
    }
    Ok(())
}

async fn update(client: &reqwest::Client, check: bool) -> Result<()> {
    const UPDATE_SERVER: &str = "https://updates.ml5.opencore.one";
    let current_version = env!("CARGO_PKG_VERSION");
    let binary = format!("ml5{}", std::env::consts::EXE_SUFFIX);
    let resp: serde_json::Value = client
        .get(format!("{UPDATE_SERVER}/api/check?version={current_version}&binary={binary}&os={}", std::env::consts::OS))
        .send()
        .await
        .context("update server unreachable")?
        .error_for_status()?
        .json()
        .await?;
    let latest = resp["latest"].as_str().unwrap_or("unknown");
    if !resp["update_available"].as_bool().unwrap_or(false) {
        println!("ML5 {current_version} is up to date");
        return Ok(());
    }
    println!("Update available: {current_version} -> {latest}");
    if resp["mandatory"].as_bool().unwrap_or(false) {
        println!("\x1b[33mThis is a mandatory update.\x1b[0m");
    }
    if check {
        return Ok(());
    }
    let download_url = resp["download_url"]
        .as_str()
        .ok_or_else(|| anyhow!("no download URL for {binary} on {}", std::env::consts::OS))?;
    let expected_sha = resp["sha256"].as_str().unwrap_or("");
    if expected_sha.is_empty() {
        bail!("the update server did not publish a SHA-256 for {binary}; refusing to install an unverified binary");
    }
    let url = if download_url.starts_with("http") {
        download_url.to_string()
    } else {
        format!("{UPDATE_SERVER}{download_url}")
    };
    println!("Downloading {url}...");
    let data = client.get(&url).send().await?.error_for_status()?.bytes().await?;
    use sha2::Digest;
    let actual_sha = hex::encode(sha2::Sha256::digest(&data));
    if actual_sha != expected_sha {
        bail!("SHA256 mismatch: expected {expected_sha}, got {actual_sha}");
    }
    println!("SHA256 verified");
    let current = std::env::current_exe()?;
    let dir = current.parent().ok_or_else(|| anyhow!("no parent dir"))?;
    let tmp = dir.join(format!("{binary}.new"));
    let old = dir.join(format!("{binary}.old"));
    std::fs::write(&tmp, &data).with_context(|| {
        format!("cannot write to {} (installed system-wide? update it through your package manager or with sudo)", dir.display())
    })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))?;
    }
    if old.exists() {
        std::fs::remove_file(&old)?;
    }
    std::fs::rename(&current, &old)?;
    std::fs::rename(&tmp, &current)?;
    println!("Updated to {latest}. Old binary saved as {}", old.display());
    Ok(())
}
