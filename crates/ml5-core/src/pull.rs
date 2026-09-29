use crate::error::{Ml5Error, Result};
use crate::routes;
use futures::StreamExt;
use serde::Deserialize;
use std::path::PathBuf;

const HF_API_BASE: &str = "https://huggingface.co/api";
const HF_RESOLVE_BASE: &str = "https://huggingface.co";
const CORE_REGISTRY_BASE: &str = "https://registry.coreml5.dev";
const ROUTES_BASE: &str = "https://routing.opencore.one";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullSource {
    HuggingFace { repo: String, file: Option<String> },
    CoreRegistry { name: String },
    Routed { name: String, repo: String, file: Option<String> },
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum PullEvent {
    Resolving { target: String },
    Downloading { file: String },
    Progress { downloaded: u64, total: Option<u64> },
    Verifying { file: String },
    Done { model: String },
    Error { error: String },
}

pub fn parse_pull_target(registry: &str, reference: &str) -> Result<PullSource> {
    let reference = reference.trim();
    if reference.is_empty() {
        return Err(Ml5Error::InvalidRequest("empty model reference".into()));
    }

    match registry {
        "hf" | "huggingface" => parse_hf(reference),
        "core" | "coreml5" => Ok(PullSource::CoreRegistry {
            name: reference.to_string(),
        }),
        other => Err(Ml5Error::InvalidRequest(format!(
            "unknown registry '{other}' (supported: hf, core)"
        ))),
    }
}

fn parse_hf(reference: &str) -> Result<PullSource> {
    let reference = reference.strip_prefix("hf:").unwrap_or(reference);
    let mut r = reference;
    for prefix in [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
        "hf.co/",
    ] {
        if let Some(stripped) = r.strip_prefix(prefix) {
            r = stripped;
            break;
        }
    }
    let r = r.trim_end_matches('/');

    if let Some((repo, file)) = r.split_once("/resolve/") {
        let file = file.split('/').skip(1).collect::<Vec<_>>().join("/");
        let file = if file.is_empty() { None } else { Some(file) };
        return Ok(PullSource::HuggingFace {
            repo: repo.to_string(),
            file,
        });
    }
    if let Some((repo, file)) = r.split_once("/blob/") {
        let file = file.split('/').skip(1).collect::<Vec<_>>().join("/");
        let file = if file.is_empty() { None } else { Some(file) };
        return Ok(PullSource::HuggingFace {
            repo: repo.to_string(),
            file,
        });
    }

    let parts: Vec<&str> = r.split('/').collect();
    match parts.len() {
        2 => Ok(PullSource::HuggingFace {
            repo: r.to_string(),
            file: None,
        }),
        n if n > 2 => {
            let repo = parts[..2].join("/");
            let file = parts[2..].join("/");
            Ok(PullSource::HuggingFace {
                repo,
                file: Some(file),
            })
        }
        _ => Err(Ml5Error::InvalidRequest(format!(
            "invalid Hugging Face reference '{reference}' (want owner/repo or owner/repo/file.gguf)"
        ))),
    }
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    #[serde(rename = "rfilename")]
    rfilename: String,
}

fn pick_gguf(files: Vec<String>, quant: Option<&str>) -> Result<String> {
    let ggufs: Vec<&String> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    if ggufs.is_empty() {
        return Err(Ml5Error::ModelNotFound(
            "repository contains no .gguf files".into(),
        ));
    }
    if ggufs.len() == 1 {
        return Ok(ggufs[0].clone());
    }

    if let Some(q) = quant {
        let q = q.to_ascii_lowercase();
        if let Some(hit) = ggufs.iter().find(|f| f.to_ascii_lowercase().contains(&q)) {
            return Ok((*hit).clone());
        }
        return Err(Ml5Error::ModelNotFound(format!(
            "no .gguf file matching quant '{q}' in repository (available: {})",
            ggufs.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
        )));
    }

    const PREF: &[&str] = &["q4_k_m", "q4km", "q5_k_m", "q4_0", "q5_0", "q8_0", "f16"];
    for tag in PREF {
        if let Some(hit) = ggufs.iter().find(|f| f.to_ascii_lowercase().contains(tag)) {
            return Ok((*hit).clone());
        }
    }
    Ok(ggufs[0].clone())
}

fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_")
}

pub async fn resolve_via_routes(name: &str) -> Result<Option<PullSource>> {
    let client = reqwest::Client::builder()
        .user_agent("ml5/0.1")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| Ml5Error::Download(format!("routes client build failed: {e}")))?;

    let url = format!(
        "{}/api/resolve?name={}",
        ROUTES_BASE,
        urlencoding::encode(name)
    );

    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| Ml5Error::Download(format!("routes request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(Ml5Error::Download(format!(
            "routes server returned {}",
            resp.status()
        )));
    }

    let resolved: routes::ResolveResponse = resp
        .json()
        .await
        .map_err(|e| Ml5Error::Download(format!("bad routes response: {e}")))?;

    if !resolved.found {
        if let Some(msg) = &resolved.message {
            tracing::info!("routes: {msg}");
        }
        return Ok(None);
    }

    let route = resolved
        .route
        .ok_or_else(|| Ml5Error::Download("routes response missing route".into()))?;

    if let Some(msg) = &resolved.message {
        tracing::info!("routes: {msg}");
    }

    Ok(Some(PullSource::Routed {
        name: resolved.canonical_name.unwrap_or_else(|| name.to_string()),
        repo: route.hf_repo,
        file: route.hf_file,
    }))
}

pub async fn pull<F>(
    source: &PullSource,
    models_dir: &std::path::Path,
    token: Option<&str>,
    quant: Option<&str>,
    mut on_event: F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    tokio::fs::create_dir_all(models_dir).await?;

    match source {
        PullSource::HuggingFace { repo, file } => {
            pull_hf(repo, file.as_deref(), models_dir, token, quant, &mut on_event).await
        }
        PullSource::CoreRegistry { name } => pull_core(name, models_dir, &mut on_event).await,
        PullSource::Routed { name, repo, file } => {
            pull_routed(name, repo, file.as_deref(), models_dir, token, quant, &mut on_event).await
        }
    }
}

async fn pull_hf<F>(
    repo: &str,
    file: Option<&str>,
    models_dir: &std::path::Path,
    token: Option<&str>,
    quant: Option<&str>,
    on_event: &mut F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    on_event(PullEvent::Resolving {
        target: repo.to_string(),
    });

    let client = reqwest::Client::builder()
        .user_agent("ml5/0.1")
        .build()
        .map_err(|e| Ml5Error::Download(e.to_string()))?;

    let chosen_file = match file {
        Some(f) => f.to_string(),
        None => {
            let url = format!("{HF_API_BASE}/models/{repo}");
            let mut req = client.get(&url);
            if let Some(t) = token {
                req = req.bearer_auth(t);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| Ml5Error::Download(format!("HF API request failed: {e}")))?;
            if !resp.status().is_success() {
                return Err(Ml5Error::Download(format!(
                    "HF API returned {} for {repo}",
                    resp.status()
                )));
            }
            #[derive(Deserialize)]
            struct ModelInfoResp {
                siblings: Vec<HfSibling>,
            }
            let info: ModelInfoResp = resp
                .json()
                .await
                .map_err(|e| Ml5Error::Download(format!("bad HF API response: {e}")))?;
            let files: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
            let ggufs: Vec<&String> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
            if ggufs.is_empty() {
                let has_st = files.iter().any(|f| f.ends_with(".safetensors"));
                let has_config = files.iter().any(|f| f == "config.json");
                let has_tok = files.iter().any(|f| f == "tokenizer.json");
                if has_st && has_config && has_tok {
                    return pull_safetensors(repo, &files, models_dir, token, on_event).await;
                }
                return Err(Ml5Error::ModelNotFound(
                    "repository contains no .gguf files and is not a valid safetensors model (missing config.json, tokenizer.json, or *.safetensors)".into(),
                ));
            }
            let picked = pick_gguf(files, quant)?;
            tracing::info!(repo, file = %picked, "auto-selected gguf");
            picked
        }
    };

    let download_url = format!("{HF_RESOLVE_BASE}/{repo}/resolve/main/{chosen_file}");
    on_event(PullEvent::Downloading {
        file: chosen_file.clone(),
    });

    let mut req = client.get(&download_url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| Ml5Error::Download(format!("download failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(Ml5Error::Download(format!(
            "download of {chosen_file} returned {}",
            resp.status()
        )));
    }

    let total = resp.content_length();
    let safe = sanitize_filename(&format!("{}__{}", repo.replace('/', "--"), chosen_file));
    let final_path = models_dir.join(&safe);
    let tmp_path = models_dir.join(format!("{safe}.part"));

    let file = tokio::fs::File::create(&tmp_path).await?;
    let mut file = tokio::io::BufWriter::with_capacity(1024 * 1024, file);
    let mut downloaded: u64 = 0;
    let mut last_progress = std::time::Instant::now();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| Ml5Error::Download(format!("stream error: {e}")))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk).await?;
        downloaded += chunk.len() as u64;
        if last_progress.elapsed() >= std::time::Duration::from_millis(100) {
            on_event(PullEvent::Progress { downloaded, total });
            last_progress = std::time::Instant::now();
        }
    }
    tokio::io::AsyncWriteExt::flush(&mut file).await?;
    drop(file);

    on_event(PullEvent::Progress { downloaded, total });
    on_event(PullEvent::Verifying {
        file: chosen_file.clone(),
    });
    if total.is_some_and(|expected| expected != downloaded) {
        return Err(Ml5Error::Download(
            "Incomplete download; retry the pull.".into(),
        ));
    }
    let mut header = [0u8; 4];
    let mut check = tokio::fs::File::open(&tmp_path).await?;
    tokio::io::AsyncReadExt::read_exact(&mut check, &mut header).await?;
    if &header != b"GGUF" {
        return Err(Ml5Error::Download(
            "Downloaded file is not a GGUF model; choose a .gguf file.".into(),
        ));
    }
    drop(check);

    tokio::fs::rename(&tmp_path, &final_path).await?;

    let meta = crate::model::ModelMeta {
        friendly_name: None,
        source_repo: Some(repo.to_string()),
        source_file: Some(chosen_file.clone()),
    };
    let meta_path = final_path.with_extension("meta.json");
    if let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = tokio::fs::write(&meta_path, json).await;
    }

    let model_name = final_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();

    on_event(PullEvent::Done {
        model: model_name.clone(),
    });
    Ok((model_name, final_path))
}

async fn pull_safetensors<F>(
    repo: &str,
    files: &[String],
    models_dir: &std::path::Path,
    token: Option<&str>,
    on_event: &mut F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    on_event(PullEvent::Resolving {
        target: format!("{repo} (safetensors)"),
    });

    let client = reqwest::Client::builder()
        .user_agent("ml5/0.1")
        .build()
        .map_err(|e| Ml5Error::Download(e.to_string()))?;

    let wanted: Vec<&String> = files
        .iter()
        .filter(|f| {
            let f = f.as_str();
            f == "config.json"
                || f == "tokenizer.json"
                || f == "tokenizer_config.json"
                || f == "special_tokens_map.json"
                || f == "generation_config.json"
                || f.ends_with(".safetensors")
        })
        .collect();

    if wanted.is_empty() {
        return Err(Ml5Error::ModelNotFound(
            "no safetensors model files found".into(),
        ));
    }

    let dir_name = sanitize_filename(&repo.replace('/', "--"));
    let model_dir = models_dir.join(&dir_name);
    tokio::fs::create_dir_all(&model_dir).await?;

    let mut total_downloaded: u64 = 0;
    for file in wanted {
        let url = format!("{HF_RESOLVE_BASE}/{repo}/resolve/main/{file}");
        on_event(PullEvent::Downloading { file: file.clone() });

        let mut req = client.get(&url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Ml5Error::Download(format!("download failed: {e}")))?;
        if !resp.status().is_success() {
            if file.ends_with("tokenizer_config.json")
                || file.ends_with("special_tokens_map.json")
                || file.ends_with("generation_config.json")
            {
                continue;
            }
            return Err(Ml5Error::Download(format!(
                "download of {file} returned {}",
                resp.status()
            )));
        }

        let total = resp.content_length();
        let out_path = model_dir.join(file);
        let tmp_path = model_dir.join(format!("{file}.part"));
        let mut out = tokio::fs::File::create(&tmp_path).await?;
        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let mut last_progress = std::time::Instant::now();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| Ml5Error::Download(format!("stream error: {e}")))?;
            tokio::io::AsyncWriteExt::write_all(&mut out, &chunk).await?;
            downloaded += chunk.len() as u64;
            total_downloaded += chunk.len() as u64;
            if last_progress.elapsed() >= std::time::Duration::from_millis(100) {
                on_event(PullEvent::Progress {
                    downloaded: total_downloaded,
                    total,
                });
                last_progress = std::time::Instant::now();
            }
        }
        tokio::io::AsyncWriteExt::flush(&mut out).await?;
        drop(out);
        tokio::fs::rename(&tmp_path, &out_path).await?;
    }

    let meta = crate::model::ModelMeta {
        friendly_name: None,
        source_repo: Some(repo.to_string()),
        source_file: None,
    };
    let meta_path = model_dir.join("model.meta.json");
    if let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = tokio::fs::write(&meta_path, json).await;
    }

    on_event(PullEvent::Done {
        model: dir_name.clone(),
    });
    Ok((dir_name, model_dir))
}

async fn pull_routed<F>(
    friendly_name: &str,
    repo: &str,
    file: Option<&str>,
    models_dir: &std::path::Path,
    token: Option<&str>,
    quant: Option<&str>,
    on_event: &mut F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    on_event(PullEvent::Resolving {
        target: format!("{friendly_name} -> {repo}"),
    });
    pull_hf(repo, file, models_dir, token, quant, on_event).await
}

async fn pull_core<F>(
    name: &str,
    _models_dir: &std::path::Path,
    on_event: &mut F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    on_event(PullEvent::Resolving {
        target: format!("{CORE_REGISTRY_BASE}/{name}"),
    });
    Err(Ml5Error::Other(anyhow::anyhow!(
        "CORe registry pulls are not live yet — use `ml5 pull hf:owner/repo` or `ml5 pull hf:owner/repo/file.gguf`"
    )))
}
