mod api;
mod daemon;
mod state;

use clap::Parser;
use ml5_core::config::Config;
use ml5_core::Engine;
use std::net::SocketAddr;

#[derive(Parser)]
#[command(name = "ml5d", about = "CORe ML5 daemon")]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 11435)]
    port: u16,

    #[arg(long)]
    models_dir: Option<std::path::PathBuf>,

    #[arg(long, help = "Detach and run in the background (no console needed)")]
    background: bool,

    #[arg(long, help = "Stop a running background daemon")]
    stop: bool,

    #[arg(long, hide = true, help = "Internal: marks the detached child process")]
    detached_child: bool,

    #[arg(long, help = "Context size (default 2048)")]
    ctx_size: Option<u32>,

    #[arg(long, help = "Logical batch size (default 2048)")]
    batch_size: Option<u32>,

    #[arg(long, help = "Physical (micro) batch size (default 512)")]
    ubatch_size: Option<u32>,

    #[arg(
        long,
        allow_hyphen_values = true,
        help = "GPU layers: N to offload N, -1 to request all (default 0 = CPU)"
    )]
    gpu_layers: Option<i32>,

    #[arg(long, help = "Main GPU index (default 0)")]
    main_gpu: Option<i32>,

    #[arg(long, help = "CPU threads for generation")]
    threads: Option<i32>,

    #[arg(long, help = "CPU threads for batch/prompt processing")]
    threads_batch: Option<i32>,

    #[arg(long, help = "Flash attention: auto | on | off (default auto)")]
    flash_attn: Option<String>,

    #[arg(
        long,
        help = "KV cache K type: f16 bf16 q8_0 q4_0 q4_k ... (default f16)"
    )]
    cache_type_k: Option<String>,

    #[arg(long, help = "KV cache V type (default f16)")]
    cache_type_v: Option<String>,

    #[arg(long, help = "Disable mmap")]
    no_mmap: bool,

    #[arg(long, help = "Use mlock (pin model in RAM)")]
    mlock: bool,

    #[arg(
        long,
        help = "Max fraction of RAM+VRAM to use before aborting load (default 0.85)"
    )]
    max_memory_fraction: Option<f32>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut config = Config {
        host: args.host.clone(),
        port: args.port,
        ..Default::default()
    };
    if let Some(dir) = args.models_dir {
        config.models_dir = dir;
    }

    if let Some(v) = args.ctx_size {
        config.model.n_ctx = v;
    }
    if let Some(v) = args.batch_size {
        config.model.n_batch = v;
    }
    if let Some(v) = args.ubatch_size {
        config.model.n_ubatch = v;
    }
    if let Some(v) = args.gpu_layers {
        config.model.n_gpu_layers = v;
    }
    if let Some(v) = args.main_gpu {
        config.model.main_gpu = v;
    }
    if let Some(v) = args.threads {
        config.model.n_threads = v;
    }
    if let Some(v) = args.threads_batch {
        config.model.n_threads_batch = v;
    }
    if let Some(v) = args.flash_attn {
        config.model.flash_attn = v;
    }
    if let Some(v) = args.cache_type_k {
        config.model.cache_type_k = v;
    }
    if let Some(v) = args.cache_type_v {
        config.model.cache_type_v = v;
    }
    if args.no_mmap {
        config.model.use_mmap = false;
    }
    if args.mlock {
        config.model.use_mlock = true;
    }
    if let Some(v) = args.max_memory_fraction {
        config.max_memory_fraction = v;
    }

    config.validate()?;
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
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ml5d=info,ml5_core=info".into()),
        )
        .with_writer(non_blocking)
        .with_ansi(false)
        .init();

    daemon::write_pid(&pid_path)?;
    let _pid_guard = daemon::PidGuard(pid_path.clone());

    let engine = Engine::new(config.clone())?;
    engine.scan_models().await?;

    let app = api::router(state::AppState { engine });

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    tracing::info!(%addr, "ml5d listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("ML5 ready at http://{addr} | OpenAI API: http://{addr}/v1");
    eprintln!(
        "Models: {} | Context: {} | Requested GPU layers: {}",
        config.models_dir.display(),
        config.model.n_ctx,
        config.model.n_gpu_layers
    );
    eprintln!("Try `ml5 list` or `ml5 pull hf:owner/repo`. Ctrl+C stops a foreground server; use `ml5d --stop` for a background server.");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
