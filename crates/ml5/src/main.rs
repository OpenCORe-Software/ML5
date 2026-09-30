mod sse;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use std::io::{IsTerminal, Read, Write};
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "ml5", version, about = "CORe ML5 CLI")]
struct Cli {
    #[arg(long, global = true, default_value = "http://127.0.0.1:11435")]
    host: String,

    #[arg(
        long,
        short,
        global = true,
        help = "Only print model output and errors"
    )]
    quiet: bool,

    #[command(subcommand)]
    cmd: Cmd,
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

        #[arg(long, help = "Context size override")]
        ctx_size: Option<u32>,

        #[arg(
            long,
            allow_hyphen_values = true,
            help = "GPU layers: N, or -1 to request all layers"
        )]
        gpu_layers: Option<i32>,

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

        #[arg(long, help = "Unload the model after the chat ends")]
        unload_after: bool,

        #[arg(long, help = "Instruction prepended to the conversation")]
        system: Option<String>,
    },

    #[command(
        about = "Pull a model: `ml5 pull hf:owner/repo` (Hugging Face) or `ml5 pull <name>` (CORe registry)"
    )]
    Pull {
        #[arg(help = "'hf:owner/repo', 'hf:owner/repo/file.gguf', or a CORe model name")]
        target: String,

        #[arg(long, help = "Force registry: hf | core")]
        registry: Option<String>,

        #[arg(long, help = "Hugging Face API token for gated/private repos")]
        token: Option<String>,

        #[arg(long, help = "Quantization preference: q4_k_m, q5_k_m, q8_0, f16, etc.")]
        quant: Option<String>,
    },

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

    #[command(about = "GPU backends: detect hardware, list/download llama.cpp backend DLLs")]
    Backend {
        #[command(subcommand)]
        action: BackendAction,
    },
}

#[derive(Subcommand)]
enum BackendAction {
    #[command(about = "Detect GPU hardware and recommend a backend")]
    Detect,

    #[command(about = "List downloaded backends")]
    List,

    #[command(about = "Download the recommended backend from llama.cpp releases")]
    Download,
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
    if target.split('/').count() >= 2 && !target.contains(':') {
        return ("hf".into(), target.to_string());
    }
    ("routes".into(), target.to_string())
}

fn hf_token_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".ml5")
        .join("hf_token")
}

fn backends_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".ml5")
        .join("backends")
}

#[cfg(windows)]
fn detect_gpus() -> Vec<String> {
    let mut gpus = Vec::new();
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        if out.status.success() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                let line = line.trim();
                if !line.is_empty() {
                    gpus.push(format!("NVIDIA {line}"));
                }
            }
        }
    }

    if let Ok(out) = std::process::Command::new("wmic")
        .args(["path", "win32_VideoController", "get", "name"])
        .output()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line == "Name" || line.contains("NVIDIA") {
                continue;
            }
            if line.contains("AMD") || line.contains("Radeon") {
                gpus.push(format!("AMD {line}"));
            } else if line.contains("Intel") {
                gpus.push(format!("Intel {line}"));
            }
        }
    }
    gpus
}

#[cfg(not(windows))]
fn detect_gpus() -> Vec<String> {
    vec![]
}

fn recommend_backend(gpus: &[String]) -> String {
    #[cfg(target_os = "macos")]
    {
        return "metal".to_string();
    }
    #[cfg(not(target_os = "macos"))]
    {
        let has_nvidia = gpus.iter().any(|g| g.starts_with("NVIDIA"));
        let has_amd = gpus.iter().any(|g| g.starts_with("AMD"));
        let has_intel = gpus.iter().any(|g| g.starts_with("Intel"));
        if has_nvidia {
            return "vulkan".to_string();
        }
        if has_amd || has_intel {
            return "vulkan".to_string();
        }
        "cpu".to_string()
    }
}

#[cfg(windows)]
async fn download_backend(backend: &str) -> Result<std::path::PathBuf> {
    let dir = backends_dir();
    std::fs::create_dir_all(&dir)?;
    let url = match backend {
        "vulkan" => "https://github.com/ggml-org/llama.cpp/releases/download/b10687/llama-b10687-bin-win-vulkan-x64.zip".to_string(),
        "cpu" => "https://github.com/ggml-org/llama.cpp/releases/download/b10687/llama-b10687-bin-win-cpu-x64.zip".to_string(),
        other => return Err(anyhow!("no download URL for backend '{other}' yet")),
    };
    let zip_path = dir.join("backend.zip");
    let bytes = reqwest::get(&url).await?.bytes().await?;
    std::fs::write(&zip_path, &bytes)?;

    let file = std::fs::File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut f = archive.by_index(i)?;
        let name = f.name().to_string();
        if (name.starts_with("ggml") && name.ends_with(".dll")) || name == "ggml-rpc-server.exe" {
            let out = dir.join(&name);
            let mut outf = std::fs::File::create(&out)?;
            std::io::copy(&mut f, &mut outf)?;
        }
    }
    let _ = std::fs::remove_file(&zip_path);
    Ok(dir)
}

#[cfg(not(windows))]
async fn download_backend(_backend: &str) -> Result<std::path::PathBuf> {
    Err(anyhow!(
        "backend auto-download not yet implemented for this OS"
    ))
}

fn load_hf_token() -> Option<String> {
    std::fs::read_to_string(hf_token_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

async fn checked_response(request: reqwest::RequestBuilder) -> Result<reqwest::Response> {
    let response = request.send().await.map_err(|e| {
        if e.is_connect() {
            anyhow!("Cannot reach ML5. Start it with `ml5d --background`, or check --host. ({e})")
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

async fn stream_chat_body(
    client: &reqwest::Client,
    body: serde_json::Value,
    host: &str,
    quiet: bool,
) -> Result<String> {
    let started = Instant::now();
    let mut first_token = None;
    let progress = indicatif::ProgressBar::new_spinner();
    if quiet || !std::io::stderr().is_terminal() {
        progress.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }
    progress.set_style(indicatif::ProgressStyle::with_template(
        "{spinner} {msg} [{elapsed_precise}]",
    )?);
    progress.set_message("Connecting to ML5");
    progress.enable_steady_tick(Duration::from_millis(100));
    let _cleanup = ProgressCleanup(progress.clone());
    let result = async {
        let resp = checked_response(client
            .post(format!("{host}/api/chat"))
            .json(&body)).await?;

        let mut stream = resp.bytes_stream();
        let mut acc = String::new();
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
                            "queued" => "Model ready; waiting for inference worker".into(),
                            "processing_prompt" => format!("Processing prompt: {}/{} tokens", v["completed"], v["total"]),
                            other => other.replace('_', " "),
                        };
                        progress.set_message(message.clone());
                        if !quiet && !std::io::stderr().is_terminal() { eprintln!("{message}"); }
                        continue;
                    }
                    if let Some(content) = v["message"]["content"].as_str() {
                        if !content.is_empty() {
                            if first_token.is_none() {
                                first_token = Some(started.elapsed());
                                progress.finish_and_clear();
                            }
                            let mut h = stdout.lock();
                            acc.push_str(content);
                            write!(h, "{content}")?;
                            h.flush()?;
                        }
                    }
                    if v["done"].as_bool() == Some(true) {
                        progress.finish_and_clear();
                        println!();
                        if acc.trim().is_empty() {
                            bail!("Model ended without visible text. Check its chat template or increase --max-tokens if the token limit was reached.");
                        }
                        if !quiet {
                            let usage = &v["usage"];
                            let tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
                            let generation_ms = usage["generation_ms"].as_u64().unwrap_or(0);
                            let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
                            let rate = if tokens > 1 && generation_ms >= 100 {
                                format!("{:.1} tok/s decode", (tokens - 1) as f64 * 1000.0 / generation_ms as f64)
                            } else { "decode rate unavailable (short sample)".into() };
                            if tokens > 0 {
                                eprintln!("{prompt_tokens} prompt tokens · {tokens} generated · {rate} · {:.2}s total · {:.2}s to first text",
                                    started.elapsed().as_secs_f64(), first_token.unwrap_or_else(|| started.elapsed()).as_secs_f64());
                            } else {
                                eprintln!("{prompt_tokens} prompt tokens · no output tokens · {:.2}s total",
                                    started.elapsed().as_secs_f64());
                            }
                            if v["finish_reason"].as_str() == Some("length") {
                                eprintln!("Stopped at the token/context limit. Increase --max-tokens or --ctx-size for longer replies.");
                            }
                        }
                        return Ok(acc);
                    }
            }
        }
        bail!("Connection ended before the response completed; retry the request.")
        }.await;
    progress.finish_and_clear();
    result
}

struct ProgressCleanup(indicatif::ProgressBar);

impl Drop for ProgressCleanup {
    fn drop(&mut self) {
        self.0.disable_steady_tick();
        self.0.finish_and_clear();
    }
}

fn print_chat_help() {
    println!("Commands: /exit /quit (leave)  /clear (reset)  /model <name> (switch)");
    println!("          /set system <text>  /set system show  /set system clear  /help");
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
    opts: serde_json::Map<String, serde_json::Value>,
    system: Option<String>,
    quiet: bool,
) -> Result<()> {
    let mut model = model.to_string();
    let mut system = system;
    let mut history: Vec<serde_json::Value> = Vec::new();

    history.push(
        serde_json::json!({ "role": "system", "content": system.clone().unwrap_or_default() }),
    );

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
                history.retain(|m| m["role"] == "system");
                println!("\x1b[90m(conversation cleared; system prompt kept)\x1b[0m");
                continue;
            }
            "/set system show" => match system.as_deref() {
                Some(value) => println!("\x1b[90m(system prompt: {value})\x1b[0m"),
                None => println!("\x1b[90m(no system prompt set)\x1b[0m"),
            },
            "/set system clear" => {
                system = None;
                history.retain(|m| m["role"] != "system");
                history.insert(0, serde_json::json!({ "role": "system", "content": "" }));
                println!("\x1b[90m(system prompt cleared)\x1b[0m");
                continue;
            }
            _ if input.starts_with("/set system ") => {
                let value = input.trim_start_matches("/set system ").trim();
                if value.is_empty() {
                    println!("\x1b[90musage: /set system <text> | /set system show | /set system clear\x1b[0m");
                } else {
                    system = Some(value.to_string());
                    history.retain(|m| m["role"] != "system");
                    history.insert(0, serde_json::json!({ "role": "system", "content": value }));
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

        history.push(serde_json::json!({ "role": "user", "content": input }));

        {
            let mut h = std::io::stdout().lock();
            print!("\x1b[1;32mml5\x1b[0m  \x1b[90m>\x1b[0m ");
            let _ = h.flush();
        }

        let mut body = serde_json::json!({
            "model": model,
            "messages": history,
            "stream": true
        });
        body.as_object_mut().unwrap().extend(opts.clone());

        let result = tokio::select! {
            result = stream_chat_body(client, body, host, quiet) => result,
            _ = tokio::signal::ctrl_c() => Err(anyhow!("Interrupted. You can enter another prompt.")),
        };
        match result {
            Ok(reply) => {
                history.push(serde_json::json!({ "role": "assistant", "content": reply }));
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()?;

    match cli.cmd {
        Cmd::List => {
            let resp: serde_json::Value =
                checked_response(client.get(format!("{}/api/models", cli.host)))
                    .await?
                    .json()
                    .await?;
            let mut table = comfy_table::Table::new();
            table.set_header(vec!["NAME", "FRIENDLY NAME", "SIZE (MB)", "CAPABILITIES"]);
            if let Some(models) = resp["models"].as_array() {
                if models.is_empty() {
                    eprintln!("No models installed. Get one with `ml5 pull hf:owner/repo` or `ml5 pull <name>`.");
                }
                for m in models {
                    let size_mb = m["size_bytes"].as_u64().unwrap_or(0) / 1_048_576;
                    let caps = m["capabilities"]
                        .as_array()
                        .map(|c| {
                            c.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(",")
                        })
                        .unwrap_or_default();
                    let name = m["name"].as_str().unwrap_or("?");
                    let friendly = m["friendly_name"].as_str().unwrap_or("");
                    table.add_row(vec![
                        name.to_string(),
                        friendly.to_string(),
                        size_mb.to_string(),
                        caps,
                    ]);
                }
            }
            println!("{table}");
        }
        Cmd::Status { json } => {
            let resp: serde_json::Value =
                checked_response(client.get(format!("{}/api/status", cli.host)))
                    .await?
                    .json()
                    .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp)?);
            } else {
                println!(
                    "ML5 {} — {}",
                    resp["version"].as_str().unwrap_or("?"),
                    cli.host
                );
                println!(
                    "Uptime: {}s | Installed models: {}",
                    resp["uptime_secs"], resp["model_count"]
                );
                println!(
                    "Model store: {}",
                    resp["models_dir"].as_str().unwrap_or("unknown")
                );
                let loaded = resp["loaded_models"]
                    .as_array()
                    .map(|models| {
                        models
                            .iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                println!(
                    "Loaded: {}",
                    if loaded.is_empty() { "none" } else { &loaded }
                );
                println!(
                    "Defaults: context {} | GPU layers {} | flash attention {}",
                    resp["defaults"]["n_ctx"],
                    resp["defaults"]["n_gpu_layers"],
                    resp["defaults"]["flash_attn"].as_str().unwrap_or("auto")
                );
            }
        }
        Cmd::Unload { model } => {
            unload_model(&client, &cli.host, &model).await?;
            println!("Unloaded {model}");
        }
        Cmd::Chat {
            model,
            message,
            ctx_size,
            gpu_layers,
            flash_attn,
            cache_type_k,
            cache_type_v,
            temperature,
            top_p,
            top_k,
            min_p,
            repeat_penalty,
            max_tokens,
            unload_after,
            system,
        } => {
            let mut opts = serde_json::Map::new();
            if let Some(v) = ctx_size {
                opts.insert("n_ctx".into(), v.into());
            }
            if let Some(v) = gpu_layers {
                opts.insert("n_gpu_layers".into(), v.into());
            }
            if let Some(v) = flash_attn {
                opts.insert("flash_attn".into(), v.into());
            }
            if let Some(v) = cache_type_k {
                opts.insert("cache_type_k".into(), v.into());
            }
            if let Some(v) = cache_type_v {
                opts.insert("cache_type_v".into(), v.into());
            }
            if let Some(v) = temperature {
                opts.insert("temperature".into(), v.into());
            }
            if let Some(v) = top_p {
                opts.insert("top_p".into(), v.into());
            }
            if let Some(v) = top_k {
                opts.insert("top_k".into(), v.into());
            }
            if let Some(v) = min_p {
                opts.insert("min_p".into(), v.into());
            }
            if let Some(v) = repeat_penalty {
                opts.insert("repeat_penalty".into(), v.into());
            }
            if let Some(v) = max_tokens {
                opts.insert("max_tokens".into(), v.into());
            }

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

                    let sys = system.clone().unwrap_or_default();
                    messages.push(serde_json::json!({ "role": "system", "content": sys }));
                    messages.push(serde_json::json!({ "role": "user", "content": m }));
                    let mut body = serde_json::json!({
                        "model": model,
                        "messages": messages,
                        "stream": true
                    });
                    body.as_object_mut().unwrap().extend(opts.clone());

                    let result = tokio::select! {
                        result = stream_chat_body(&client, body, &cli.host, cli.quiet) => result.map(|_| ()),
                        _ = tokio::signal::ctrl_c() => Err(anyhow!("Interrupted")),
                    };
                    if unload_after {
                        unload_model(&client, &cli.host, &model).await?;
                        if !cli.quiet {
                            eprintln!("Model unloaded");
                        }
                    }
                    result?;
                }
                None => {
                    interactive_chat(
                        &client,
                        &cli.host,
                        &model,
                        unload_after,
                        opts,
                        system,
                        cli.quiet,
                    )
                    .await?;
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
            let token = token.or_else(load_hf_token);
            let mut body = serde_json::json!({ "model": reference, "registry": registry });
            if let Some(t) = token {
                body["token"] = serde_json::Value::String(t);
            }
            if let Some(q) = quant {
                body["quant"] = serde_json::Value::String(q);
            }

            let resp =
                checked_response(client.post(format!("{}/api/pull", cli.host)).json(&body)).await?;

            use indicatif::{ProgressBar, ProgressStyle};
            let mut pb = None::<ProgressBar>;

            let mut stream = resp.bytes_stream();
            let mut decoder = sse::Decoder::default();
            let mut done = false;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                for data in decoder.push(&chunk)? {
                    let v: serde_json::Value =
                        serde_json::from_str(&data).context("Invalid pull event from daemon")?;
                    if let Some(error) = v["error"].as_str() {
                        if let Some(bar) = pb.take() {
                            bar.finish_and_clear();
                        }
                        bail!("Pull failed: {error}");
                    }
                    match v["event"].as_str() {
                        Some("resolving") => {
                            let target = v["target"].as_str().unwrap_or("");
                            if !cli.quiet {
                                if target.contains("->") {
                                    eprintln!("\x1b[36m(auto-resolved)\x1b[0m {target}");
                                } else {
                                    eprintln!("Resolving {target}");
                                }
                            }
                        }
                        Some("downloading") => {
                            let file = v["file"].as_str().unwrap_or("model");
                            if !cli.quiet {
                                eprintln!("Selected file: {file}");
                            }
                            let bar = ProgressBar::new(0);
                            if cli.quiet {
                                bar.set_draw_target(indicatif::ProgressDrawTarget::hidden());
                            }
                            bar.set_style(
                                    ProgressStyle::with_template(
                                        "{msg} {bytes}/{total_bytes} [{bar:40.cyan/blue}] {bytes_per_sec} ETA {eta}",
                                    )
                                    .unwrap_or_else(|_| ProgressStyle::default_bar()),
                                );
                            bar.set_message(file.to_string());
                            pb = Some(bar);
                        }
                        Some("progress") => {
                            let d = v["downloaded"].as_u64().unwrap_or(0);
                            if let Some(bar) = pb.as_ref() {
                                if let Some(t) = v["total"].as_u64() {
                                    bar.set_length(t);
                                }
                                bar.set_position(d);
                            }
                        }
                        Some("done") => {
                            done = true;
                            if let Some(bar) = pb.take() {
                                bar.finish_and_clear();
                            }
                            println!("pulled {}", v["model"].as_str().unwrap_or("model"));
                            if !cli.quiet {
                                eprintln!(
                                    "Ready to chat: ml5 run {}",
                                    v["model"].as_str().unwrap_or("model")
                                );
                            }
                        }
                        Some("verifying")
                            if !cli.quiet => {
                                eprintln!("Checking download length and GGUF header…");
                            }
                        _ => {}
                    }
                }
                if done {
                    break;
                }
            }
            let final_bar = pb.take();
            if let Some(bar) = final_bar {
                bar.finish_and_clear();
            }
            if !done {
                bail!("Download connection ended before completion; retry the pull.");
            }
        }
        Cmd::Delete { model } => {
            let resp = checked_response(
                client
                    .delete(format!("{}/api/delete", cli.host))
                    .json(&serde_json::json!({ "model": model })),
            )
            .await?;
            println!("{}", resp.text().await?);
        }
        Cmd::Rename { model, new_name } => {
            let resp = checked_response(
                client
                    .post(format!("{}/api/rename", cli.host))
                    .json(&serde_json::json!({ "model": model, "new_name": new_name })),
            )
            .await?;
            let v: serde_json::Value = resp.json().await?;
            println!(
                "Renamed {} -> {}",
                v["renamed"].as_str().unwrap_or("?"),
                v["new_name"].as_str().unwrap_or("?")
            );
        }
        Cmd::Hf { action } => match action {
            HfAction::Authenticate { token } => {
                let path = hf_token_path();
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&path, token.trim())?;
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
        Cmd::Backend { action } => match action {
            BackendAction::Detect => {
                let gpus = detect_gpus();
                if gpus.is_empty() {
                    println!("No GPU detected — CPU backend will be used.");
                } else {
                    println!("Detected GPUs:");
                    for g in &gpus {
                        println!("  {g}");
                    }
                    println!("\nRecommended backend: {}", recommend_backend(&gpus));
                }
            }
            BackendAction::List => {
                let dir = backends_dir();
                if dir.exists() {
                    let mut found = false;
                    for entry in std::fs::read_dir(&dir)?.flatten() {
                        let p = entry.path();
                        if p.extension().and_then(|e| e.to_str()) == Some("dll") {
                            println!("{}", p.file_name().unwrap_or_default().to_string_lossy());
                            found = true;
                        }
                    }
                    if !found {
                        println!("No backends downloaded yet. Run `ml5 backend download`.");
                    }
                } else {
                    println!("No backends downloaded yet. Run `ml5 backend download`.");
                }
            }
            BackendAction::Download => {
                let gpus = detect_gpus();
                let backend = recommend_backend(&gpus);
                println!("Downloading backend: {backend}");
                match download_backend(&backend).await {
                    Ok(path) => println!("Downloaded to {}", path.display()),
                    Err(e) => return Err(e.context("Backend download failed")),
                }
            }
        },
        Cmd::Update { check } => {
            const UPDATE_SERVER: &str = "https://updates.ml5.opencore.one";
            let current_version = env!("CARGO_PKG_VERSION");

            let resp: serde_json::Value = client
                .get(format!(
                    "{UPDATE_SERVER}/api/check?version={current_version}&binary=ml5.exe"
                ))
                .send()
                .await?
                .json()
                .await?;

            let update_available = resp["update_available"].as_bool().unwrap_or(false);
            let latest = resp["latest"].as_str().unwrap_or("unknown");
            let mandatory = resp["mandatory"].as_bool().unwrap_or(false);

            if !update_available {
                println!("ML5 {current_version} is up to date");
                return Ok(());
            }

            println!("Update available: {current_version} -> {latest}");
            if mandatory {
                println!("\x1b[33mThis is a mandatory update.\x1b[0m");
            }

            if check {
                return Ok(());
            }

            let download_url = resp["download_url"]
                .as_str()
                .ok_or_else(|| anyhow!("no download URL in update response"))?;
            let expected_sha = resp["sha256"].as_str().unwrap_or("");

            println!("Downloading {download_url}...");
            let data = client
                .get(format!("{UPDATE_SERVER}{download_url}"))
                .send()
                .await?
                .bytes()
                .await?;

            use sha2::Digest;
            let actual_sha = hex::encode(sha2::Sha256::digest(&data));
            if !expected_sha.is_empty() && actual_sha != expected_sha {
                bail!("SHA256 mismatch: expected {expected_sha}, got {actual_sha}");
            }
            println!("SHA256 verified");

            let install_dir = std::env::current_exe()?
                .parent()
                .ok_or_else(|| anyhow!("no parent dir"))?
                .to_path_buf();
            let tmp = install_dir.join("ml5.exe.new");
            std::fs::write(&tmp, &data)?;

            let current = install_dir.join("ml5.exe");
            let old = install_dir.join("ml5.exe.old");
            if old.exists() {
                std::fs::remove_file(&old)?;
            }
            std::fs::rename(&current, &old)?;
            std::fs::rename(&tmp, &current)?;

            println!("Updated to {latest}. Old binary saved as ml5.exe.old");
        }
    }
    Ok(())
}
