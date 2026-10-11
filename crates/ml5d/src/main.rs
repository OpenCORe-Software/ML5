mod api;
mod daemon;
mod state;

use clap::Parser;
use ml5_core::config::{Config, GpuLayers};
use ml5_core::Engine;
use std::future::IntoFuture;
use std::net::SocketAddr;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser)]
#[command(name = "ml5d", version, about = "CORe ML5 daemon")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 11435)]
    port: u16,

    #[arg(long, help = "Model store (default ~/.ml5/models, or $ML5_HOME/models)")]
    models_dir: Option<std::path::PathBuf>,

    #[arg(long, help = "Detach and run in the background (no console needed)")]
    background: bool,

    #[arg(long, help = "Stop a running background daemon")]
    stop: bool,

    #[arg(long, hide = true)]
    detached_child: bool,

    #[arg(long, help = "llama-server binary or directory (default: bundled, ~/.ml5/runtime, then PATH)")]
    llama_server: Option<std::path::PathBuf>,

    #[arg(long, help = "Context size per request slot; 0 = automatic from free memory (default 0)")]
    ctx_size: Option<u32>,

    #[arg(long, help = "Upper bound for the automatic context size (default 32768)")]
    auto_ctx_max: Option<u32>,

    #[arg(long, help = "Logical batch size (default 2048)")]
    batch_size: Option<u32>,

    #[arg(long, help = "Physical (micro) batch size (default 512)")]
    ubatch_size: Option<u32>,

    #[arg(
        long,
        allow_hyphen_values = true,
        value_parser = GpuLayers::parse,
        help = "GPU layers: auto (fit to free VRAM), all, or a number; 0 = CPU only (default auto)"
    )]
    gpu_layers: Option<GpuLayers>,

    #[arg(long, help = "Main GPU index (default 0)")]
    main_gpu: Option<i32>,

    #[arg(long, help = "Tensor split ratios across GPUs/cluster nodes, e.g. 0.5,0.5")]
    tensor_split: Option<String>,

    #[arg(long, help = "RPC cluster workers, e.g. node1:50052,node2:50052")]
    rpc_nodes: Option<String>,

    #[arg(long, help = "CPU threads for generation")]
    threads: Option<i32>,

    #[arg(long, help = "CPU threads for prompt processing")]
    threads_batch: Option<i32>,

    #[arg(long, value_parser = ["auto", "on", "off"], help = "Flash attention (default auto)")]
    flash_attn: Option<String>,

    #[arg(long, help = "KV cache K type: f16 bf16 q8_0 q4_0 ... (default f16)")]
    cache_type_k: Option<String>,

    #[arg(long, help = "KV cache V type (default f16)")]
    cache_type_v: Option<String>,

    #[arg(long, help = "Keep the KV cache in system RAM instead of VRAM")]
    no_kv_offload: bool,

    #[arg(long, help = "Disable mmap")]
    no_mmap: bool,

    #[arg(long, help = "Pin the model in RAM (mlock)")]
    mlock: bool,

    #[arg(
        long,
        visible_alias = "n-parallel",
        num_args = 0..=1,
        default_missing_value = "4",
        help = "Simultaneous requests per model (continuous batching); bare --parallel means 4 (default 1)"
    )]
    parallel: Option<usize>,

    #[arg(long, help = "How many models may stay loaded at once (default 2)")]
    max_loaded_models: Option<usize>,

    #[arg(
        long,
        allow_hyphen_values = true,
        help = "Unload idle models after this long: seconds or 30s/15m/2h; -1 keeps them loaded (default 15m)"
    )]
    keep_alive: Option<String>,

    #[arg(long, help = "Max fraction of RAM+VRAM a load may claim (default 0.85)")]
    max_memory_fraction: Option<f32>,
}

fn parse_duration(raw: &str) -> anyhow::Result<i64> {
    let s = raw.trim();
    if s.starts_with('-') {
        return Ok(-1);
    }
    let (num, mult) = match s.char_indices().find(|(_, c)| c.is_ascii_alphabetic()) {
        Some((i, _)) => {
            let mult = match &s[i..] {
                "s" => 1,
                "m" => 60,
                "h" => 3600,
                other => anyhow::bail!("unknown duration unit '{other}' in --keep-alive (use s, m, or h)"),
            };
            (&s[..i], mult)
        }
        None => (s, 1),
    };
    Ok(num.parse::<i64>().map_err(|_| anyhow::anyhow!("invalid --keep-alive value '{raw}'"))? * mult)
}

fn build_config(args: &Args) -> anyhow::Result<Config> {
    let mut config = Config {
        host: args.host.clone(),
        port: args.port,
        ..Default::default()
    };
    if let Some(dir) = &args.models_dir {
        config.models_dir = dir.clone();
    }
    let m = &mut config.model;
    if let Some(v) = args.ctx_size {
        m.n_ctx = v;
    }
    if let Some(v) = args.batch_size {
        m.n_batch = v;
    }
    if let Some(v) = args.ubatch_size {
        m.n_ubatch = v;
    }
    if let Some(v) = args.gpu_layers {
        m.n_gpu_layers = v;
    }
    if let Some(v) = args.main_gpu {
        m.main_gpu = v;
    }
    if let Some(v) = &args.tensor_split {
        m.tensor_split = Some(v.clone());
    }
    if let Some(v) = &args.rpc_nodes {
        m.rpc_nodes = Some(v.clone());
    }
    if let Some(v) = args.threads {
        m.n_threads = v;
    }
    if let Some(v) = args.threads_batch {
        m.n_threads_batch = v;
    }
    if let Some(v) = &args.flash_attn {
        m.flash_attn = v.clone();
    }
    if let Some(v) = &args.cache_type_k {
        m.cache_type_k = v.clone();
    }
    if let Some(v) = &args.cache_type_v {
        m.cache_type_v = v.clone();
    }
    if args.no_kv_offload {
        m.offload_kqv = false;
    }
    if args.no_mmap {
        m.use_mmap = false;
    }
    if args.mlock {
        m.use_mlock = true;
    }
    m.n_ubatch = m.n_ubatch.min(m.n_batch);
    if let Some(v) = args.parallel {
        config.n_parallel = v;
    }
    if let Some(v) = args.max_loaded_models {
        config.max_loaded_models = v;
    }
    if let Some(v) = &args.keep_alive {
        config.keep_alive_secs = parse_duration(v)?;
    }
    if let Some(v) = args.max_memory_fraction {
        config.max_memory_fraction = v;
    }
    if let Some(v) = args.auto_ctx_max {
        config.auto_ctx_max = v;
    }
    config.llama_server = args.llama_server.clone();
    config.validate()?;
    Ok(config)
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = build_config(&args)?;
    config.ensure_dirs()?;
    let pid_path = daemon::pid_file(&config);

    if args.stop {
        return daemon::stop(&pid_path);
    }
    if args.background && !args.detached_child {
        return daemon::spawn_background(&config);
    }
    if let Some(pid) = daemon::running_pid(&pid_path) {
        if pid != std::process::id() {
            eprintln!("ml5d is already running (pid {pid}). Use `ml5d --stop` to stop it first.");
            std::process::exit(1);
        }
    }

    let file_appender = tracing_appender::rolling::daily(config.log_dir(), "ml5d.log");
    let (file_writer, _file_guard) = tracing_appender::non_blocking(file_appender);
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "ml5d=info,ml5_core=info".into());
    let to_journal = std::env::var_os("JOURNAL_STREAM").is_some();
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_writer(file_writer).with_ansi(false))
        .with(to_journal.then(|| {
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(false)
                .without_time()
        }))
        .init();

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        anyhow::anyhow!("cannot listen on {addr}: {e}. Is another ml5d already running?")
    })?;

    daemon::write_pid(&pid_path)?;
    let _pid_guard = daemon::PidGuard(pid_path.clone());

    let engine = Engine::new(config.clone())?;
    engine.scan_models().await?;
    let app = api::router(state::AppState {
        engine: engine.clone(),
    });

    tracing::info!(%addr, "ml5d listening");
    let runtime = ml5_core::runtime::locate_server(config.llama_server.as_deref());
    eprintln!("ML5 ready at http://{addr} | OpenAI API: http://{addr}/v1");
    eprintln!(
        "Models: {} | Context: {} | GPU layers: {} | Slots per model: {} | Loaded models: up to {} | Keep-alive: {}",
        config.models_dir.display(),
        if config.model.n_ctx == 0 {
            format!("auto (≤{})", config.auto_ctx_max)
        } else {
            config.model.n_ctx.to_string()
        },
        config.model.n_gpu_layers,
        config.n_parallel,
        config.max_loaded_models,
        if config.keep_alive_secs < 0 {
            "forever".to_string()
        } else {
            format!("{}s", config.keep_alive_secs)
        }
    );
    match runtime {
        Some(p) => eprintln!("llama-server: {}", p.display()),
        None => eprintln!(
            "llama-server: not installed yet; llama.cpp {} is downloaded on the first model load (or run `ml5 backend download`)",
            ml5_core::runtime::LLAMA_RELEASE_TAG
        ),
    }
    eprintln!("Try `ml5 list` or `ml5 pull hf:owner/repo`. Ctrl+C stops a foreground server; use `ml5d --stop` for a background server.");

    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = stop_rx.wait_for(|stop| *stop).await;
        })
        .into_future();
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result?,
        _ = shutdown_signal() => {
            let _ = stop_tx.send(true);
            match tokio::time::timeout(std::time::Duration::from_secs(10), &mut server).await {
                Ok(result) => result?,
                Err(_) => tracing::warn!("clients kept connections open for 10s; closing them"),
            }
        }
    }
    tracing::info!("shutting down; unloading models");
    engine.unload_all().await;
    Ok(())
}
