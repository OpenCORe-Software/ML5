use ml5_core::config::Config;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

pub fn pid_file(config: &Config) -> PathBuf {
    config
        .models_dir
        .parent()
        .map(|p| p.join("ml5d.pid"))
        .unwrap_or_else(|| PathBuf::from("./ml5d.pid"))
}

pub fn write_pid(path: &Path) -> anyhow::Result<()> {
    std::fs::write(path, std::process::id().to_string())?;
    Ok(())
}

pub struct PidGuard(pub PathBuf);

impl Drop for PidGuard {
    fn drop(&mut self) {
        let ours = std::fs::read_to_string(&self.0)
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            == Some(std::process::id());
        if ours {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

pub fn running_pid(path: &Path) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let pid: u32 = raw.trim().parse().ok()?;
    if process_alive(pid) {
        Some(pid)
    } else {
        let _ = std::fs::remove_file(path);
        None
    }
}

#[cfg(target_os = "linux")]
fn is_ml5d(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|c| c.trim() == "ml5d")
        .unwrap_or(true)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn is_ml5d(_pid: u32) -> bool {
    true
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    let Ok(pid_t) = libc::pid_t::try_from(pid) else { return false };
    let rc = unsafe { libc::kill(pid_t, 0) };
    let alive = rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM);
    alive && is_ml5d(pid)
}

#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
        .unwrap_or(false)
}

#[cfg(unix)]
fn signal(pid: u32, sig: libc::c_int) -> bool {
    libc::pid_t::try_from(pid)
        .map(|p| unsafe { libc::kill(p, sig) } == 0)
        .unwrap_or(false)
}

pub fn stop(pid_path: &Path) -> anyhow::Result<()> {
    let Some(pid) = running_pid(pid_path) else {
        println!("ml5d is not running");
        return Ok(());
    };
    #[cfg(unix)]
    {
        if !signal(pid, libc::SIGTERM) {
            anyhow::bail!(
                "could not signal ml5d (pid {pid}): {}",
                std::io::Error::last_os_error()
            );
        }
        let started = Instant::now();
        while process_alive(pid) {
            if started.elapsed() > Duration::from_secs(30) {
                signal(pid, libc::SIGKILL);
                std::thread::sleep(Duration::from_millis(500));
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
    #[cfg(windows)]
    {
        let status = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status()?;
        if !status.success() {
            anyhow::bail!("taskkill could not stop ml5d (pid {pid})");
        }
        let started = Instant::now();
        while process_alive(pid) && started.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(200));
        }
    }
    let _ = std::fs::remove_file(pid_path);
    println!("ml5d stopped (pid {pid})");
    Ok(())
}

fn config_args(config: &Config) -> Vec<OsString> {
    let m = &config.model;
    let mut args: Vec<OsString> = vec![
        "--detached-child".into(),
        "--host".into(),
        config.host.clone().into(),
        "--port".into(),
        config.port.to_string().into(),
        "--models-dir".into(),
        config.models_dir.clone().into(),
        "--ctx-size".into(),
        m.n_ctx.to_string().into(),
        "--auto-ctx-max".into(),
        config.auto_ctx_max.to_string().into(),
        "--batch-size".into(),
        m.n_batch.to_string().into(),
        "--ubatch-size".into(),
        m.n_ubatch.to_string().into(),
        "--gpu-layers".into(),
        m.n_gpu_layers.as_arg().into(),
        "--main-gpu".into(),
        m.main_gpu.to_string().into(),
        "--flash-attn".into(),
        m.flash_attn.clone().into(),
        "--cache-type-k".into(),
        m.cache_type_k.clone().into(),
        "--cache-type-v".into(),
        m.cache_type_v.clone().into(),
        "--parallel".into(),
        config.n_parallel.to_string().into(),
        "--max-loaded-models".into(),
        config.max_loaded_models.to_string().into(),
        format!("--keep-alive={}", config.keep_alive_secs).into(),
        "--max-memory-fraction".into(),
        config.max_memory_fraction.to_string().into(),
    ];
	for dir in &config.extra_model_dirs {
		args.extend(["--models-dir".into(), dir.clone().into()]);
	}
    if m.n_threads > 0 {
        args.extend(["--threads".into(), m.n_threads.to_string().into()]);
    }
    if m.n_threads_batch > 0 {
        args.extend(["--threads-batch".into(), m.n_threads_batch.to_string().into()]);
    }
    if let Some(v) = &m.tensor_split {
        args.extend(["--tensor-split".into(), v.clone().into()]);
    }
    if let Some(v) = &m.rpc_nodes {
        args.extend(["--rpc-nodes".into(), v.clone().into()]);
    }
    if let Some(v) = &config.llama_server {
        args.extend(["--llama-server".into(), v.clone().into()]);
    }
    if !m.offload_kqv {
        args.push("--no-kv-offload".into());
    }
    if !m.cache_prompt {
        args.push("--no-cache-prompt".into());
    }
    if m.opt_cache_prompt {
        args.push("--opt-cache-prompt".into());
    }
    if m.cache_reuse > 0 {
        args.extend(["--cache-reuse".into(), m.cache_reuse.to_string().into()]);
    }
    if !m.use_mmap {
        args.push("--no-mmap".into());
    }
    if m.use_mlock {
        args.push("--mlock".into());
    }
    args
}

pub fn spawn_background(config: &Config) -> anyhow::Result<()> {
    if let Some(pid) = running_pid(&pid_file(config)) {
        println!("ml5d is already running (pid {pid})");
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    let log_dir = config.log_dir();
    std::fs::create_dir_all(&log_dir)?;
    let stdout_log = std::fs::File::create(log_dir.join("ml5d.out.log"))?;
    let stderr_log = stdout_log.try_clone()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.args(config_args(config))
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_log))
        .stderr(Stdio::from(stderr_log));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        use winapi::um::winbase::{CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, DETACHED_PROCESS};
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    let child = cmd.spawn()?;
    println!(
        "ml5d running in background (pid {}), logs: {}",
        child.id(),
        log_dir.display()
    );
    Ok(())
}
