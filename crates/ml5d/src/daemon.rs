use ml5_core::config::Config;
use std::path::PathBuf;
use std::process::Stdio;

pub fn pid_file(config: &Config) -> PathBuf {
    config
        .models_dir
        .parent()
        .map(|p| p.join("ml5d.pid"))
        .unwrap_or_else(|| PathBuf::from("./ml5d.pid"))
}

pub fn write_pid(path: &PathBuf) -> anyhow::Result<()> {
    std::fs::write(path, std::process::id().to_string())?;
    Ok(())
}

pub struct PidGuard(pub PathBuf);

impl Drop for PidGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub fn running_pid(path: &PathBuf) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let pid: u32 = raw.trim().parse().ok()?;
    if process_alive(pid) {
        Some(pid)
    } else {
        let _ = std::fs::remove_file(path);
        None
    }
}

fn process_alive(pid: u32) -> bool {
    let output = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH"])
        .output();
    match output {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            text.contains(&pid.to_string())
        }
        Err(_) => false,
    }
}

pub fn stop(pid_path: &PathBuf) -> anyhow::Result<()> {
    match running_pid(pid_path) {
        Some(pid) => {
            let status = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/F"])
                .status()?;
            let _ = std::fs::remove_file(pid_path);
            if status.success() {
                println!("ml5d stopped (pid {pid})");
            } else {
                println!("sent stop to pid {pid}");
            }
            Ok(())
        }
        None => {
            println!("ml5d is not running");
            Ok(())
        }
    }
}

#[cfg(windows)]
pub fn spawn_background(config: &Config) -> anyhow::Result<()> {
    use std::os::windows::process::CommandExt;
    use winapi::um::winbase::{CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, DETACHED_PROCESS};

    let exe = std::env::current_exe()?;
    let log_dir = config.log_dir();
    std::fs::create_dir_all(&log_dir)?;
    let stdout_log = std::fs::File::create(log_dir.join("ml5d.out.log"))?;
    let stderr_log = stdout_log.try_clone()?;

    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--detached-child")
        .arg("--host")
        .arg(&config.host)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--models-dir")
        .arg(&config.models_dir);

    let m = &config.model;
    cmd.arg("--ctx-size")
        .arg(m.n_ctx.to_string())
        .arg("--batch-size")
        .arg(m.n_batch.to_string())
        .arg("--ubatch-size")
        .arg(m.n_ubatch.to_string())
        .arg("--gpu-layers")
        .arg(m.n_gpu_layers.to_string())
        .arg("--main-gpu")
        .arg(m.main_gpu.to_string())
        .arg("--flash-attn")
        .arg(&m.flash_attn)
        .arg("--cache-type-k")
        .arg(&m.cache_type_k)
        .arg("--cache-type-v")
        .arg(&m.cache_type_v)
        .arg("--max-memory-fraction")
        .arg(config.max_memory_fraction.to_string());
    if m.n_threads > 0 {
        cmd.arg("--threads").arg(m.n_threads.to_string());
    }
    if m.n_threads_batch > 0 {
        cmd.arg("--threads-batch")
            .arg(m.n_threads_batch.to_string());
    }
    if !m.use_mmap {
        cmd.arg("--no-mmap");
    }
    if m.use_mlock {
        cmd.arg("--mlock");
    }

    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(stdout_log))
        .stderr(Stdio::from(stderr_log))
        .creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);

    let child = cmd.spawn()?;
    println!(
        "ml5d running in background (pid {}), logs: {}",
        child.id(),
        log_dir.display()
    );
    Ok(())
}

#[cfg(not(windows))]
pub fn spawn_background(config: &Config) -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let log_dir = config.log_dir();
    std::fs::create_dir_all(&log_dir)?;
    let stdout_log = std::fs::File::create(log_dir.join("ml5d.out.log"))?;
    let stderr_log = stdout_log.try_clone()?;

    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--detached-child")
        .arg("--host")
        .arg(&config.host)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--models-dir")
        .arg(&config.models_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_log))
        .stderr(Stdio::from(stderr_log));

    let child = cmd.spawn()?;
    println!(
        "ml5d running in background (pid {}), logs: {}",
        child.id(),
        log_dir.display()
    );
    Ok(())
}
