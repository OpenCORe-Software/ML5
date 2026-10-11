use crate::error::{Ml5Error, Result};
use crate::model::{self, ModelMeta};
use crate::routes;
use futures::StreamExt;
use serde::Deserialize;
use sha2::Digest;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const HF_API_BASE: &str = "https://huggingface.co/api";
const HF_RESOLVE_BASE: &str = "https://huggingface.co";
const CORE_REGISTRY_BASE: &str = "https://registry.coreml5.dev";
const USER_AGENT: &str = concat!("ml5/", env!("CARGO_PKG_VERSION"));
const PROGRESS_INTERVAL: Duration = Duration::from_millis(150);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullSource {
    HuggingFace {
        repo: String,
        file: Option<String>,
        quant: Option<String>,
    },
    CoreRegistry { name: String },
    Routed { name: String, repo: String, file: Option<String> },
}

impl PullSource {
    pub fn describe(&self) -> String {
        match self {
            Self::HuggingFace { repo, file: Some(f), .. } => format!("hf:{repo}/{f}"),
            Self::HuggingFace { repo, quant: Some(q), .. } => format!("hf:{repo}:{q}"),
            Self::HuggingFace { repo, .. } => format!("hf:{repo}"),
            Self::CoreRegistry { name } => format!("core:{name}"),
            Self::Routed { name, repo, .. } => format!("{name} -> hf:{repo}"),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum PullEvent {
    Started { id: String },
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
            "unknown registry '{other}' (supported: hf, core, routes)"
        ))),
    }
}

fn parse_hf(reference: &str) -> Result<PullSource> {
    let mut r = reference.strip_prefix("hf:").unwrap_or(reference);
    for prefix in [
        "https://huggingface.co/",
        "http://huggingface.co/",
        "https://hf.co/",
        "http://hf.co/",
        "huggingface.co/",
        "hf.co/",
    ] {
        if let Some(stripped) = r.strip_prefix(prefix) {
            r = stripped;
            break;
        }
    }
    let r = r.trim_end_matches('/');
    let (r, tag) = match r.rsplit_once(':') {
        Some((repo, tag)) if !tag.contains('/') && repo.contains('/') => (repo, Some(tag)),
        _ => (r, None),
    };
    for marker in ["/resolve/", "/blob/"] {
        if let Some((repo, file)) = r.split_once(marker) {
            let file = file.split('/').skip(1).collect::<Vec<_>>().join("/");
            return Ok(PullSource::HuggingFace {
                repo: repo.to_string(),
                file: (!file.is_empty()).then_some(file),
                quant: None,
            });
        }
    }
    let parts: Vec<&str> = r.split('/').filter(|p| !p.is_empty()).collect();
    match parts.len() {
        2 => Ok(PullSource::HuggingFace {
            repo: parts.join("/"),
            file: None,
            quant: tag.map(str::to_string),
        }),
        n if n > 2 => Ok(PullSource::HuggingFace {
            repo: parts[..2].join("/"),
            file: Some(parts[2..].join("/")),
            quant: None,
        }),
        _ => Err(Ml5Error::InvalidRequest(format!(
            "invalid Hugging Face reference '{reference}' (want owner/repo, owner/repo:quant, or owner/repo/file.gguf)"
        ))),
    }
}

#[derive(Debug, Clone)]
struct Remote {
    path: String,
    size: Option<u64>,
    sha256: Option<String>,
}

impl Remote {
    fn bare(path: &str) -> Self {
        Self {
            path: path.to_string(),
            size: None,
            sha256: None,
        }
    }

    fn base_name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }

    fn dir(&self) -> &str {
        self.path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
    }
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfModel {
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| Ml5Error::Download(e.to_string()))
}

fn authed(req: reqwest::RequestBuilder, token: Option<&str>) -> reqwest::RequestBuilder {
    match token.filter(|t| !t.is_empty()) {
        Some(t) => req.bearer_auth(t),
        None => req,
    }
}

fn access_error(status: reqwest::StatusCode, what: &str) -> Ml5Error {
    match status.as_u16() {
        401 | 403 => Ml5Error::Download(format!(
            "{what} needs Hugging Face access (gated or private repository). Accept the license on huggingface.co, then run `ml5 hf authenticate <token>`."
        )),
        404 => Ml5Error::ModelNotFound(format!("{what} was not found on Hugging Face")),
        _ => Ml5Error::Download(format!("{what} returned HTTP {status}")),
    }
}

async fn hf_listing(client: &reqwest::Client, repo: &str, token: Option<&str>) -> Result<Vec<Remote>> {
    let url = format!("{HF_API_BASE}/models/{repo}?blobs=true");
    let resp = authed(client.get(&url), token)
        .send()
        .await
        .map_err(|e| Ml5Error::Download(format!("Hugging Face API request failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(access_error(resp.status(), &format!("repository {repo}")));
    }
    let info: HfModel = resp
        .json()
        .await
        .map_err(|e| Ml5Error::Download(format!("bad Hugging Face API response: {e}")))?;
    Ok(info
        .siblings
        .into_iter()
        .map(|s| {
            let (sha256, lfs_size) = s
                .lfs
                .map(|l| (l.sha256, l.size))
                .unwrap_or((None, None));
            Remote {
                path: s.rfilename,
                size: lfs_size.or(s.size),
                sha256,
            }
        })
        .collect())
}

fn is_model_gguf(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.to_ascii_lowercase().ends_with(".gguf")
        && !model::is_mmproj_file(name)
        && model::split_part(name).is_none_or(|(_, idx, _)| idx == 1)
}

fn pick_gguf<'a>(files: &'a [Remote], quant: Option<&str>) -> Result<&'a Remote> {
    let ggufs: Vec<&Remote> = files.iter().filter(|f| is_model_gguf(&f.path)).collect();
    if ggufs.is_empty() {
        return Err(Ml5Error::ModelNotFound("repository contains no GGUF model files".into()));
    }
    let squash = |s: &str| s.to_ascii_lowercase().replace(['_', '-', '.'], "");
    if let Some(q) = quant.filter(|q| !q.is_empty()) {
        let wanted = squash(q);
        return ggufs
            .iter()
            .copied()
            .find(|f| squash(f.base_name()).contains(&wanted))
            .ok_or_else(|| {
                Ml5Error::ModelNotFound(format!(
                    "no GGUF file matching quantization '{q}' (available: {})",
                    ggufs.iter().map(|f| f.path.as_str()).collect::<Vec<_>>().join(", ")
                ))
            });
    }
    if ggufs.len() == 1 {
        return Ok(ggufs[0]);
    }
    const PREF: &[&str] = &["q4km", "q4ks", "iq4xs", "q5km", "q4_0", "q40", "q6k", "q8_0", "q80", "bf16", "f16"];
    for tag in PREF {
        let tag = squash(tag);
        if let Some(hit) = ggufs.iter().copied().find(|f| squash(f.base_name()).contains(&tag)) {
            return Ok(hit);
        }
    }
    Ok(ggufs
        .iter()
        .copied()
        .min_by_key(|f| f.size.unwrap_or(u64::MAX))
        .unwrap_or(ggufs[0]))
}

fn pick_mmproj<'a>(files: &'a [Remote], near: &Remote) -> Option<&'a Remote> {
    let mut candidates: Vec<&Remote> = files
        .iter()
        .filter(|f| model::is_mmproj_file(f.base_name()))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let same_dir: Vec<&Remote> = candidates.iter().copied().filter(|f| f.dir() == near.dir()).collect();
    if !same_dir.is_empty() {
        candidates = same_dir;
    }
    for tag in ["f16", "bf16", "q8_0", "f32"] {
        if let Some(hit) = candidates
            .iter()
            .copied()
            .find(|f| f.base_name().to_ascii_lowercase().contains(tag))
        {
            return Some(hit);
        }
    }
    candidates.first().copied()
}

fn sanitize_filename(name: &str) -> String {
    name.replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_")
}

fn local_name(repo: &str, remote_path: &str) -> String {
    sanitize_filename(&format!("{}__{}", repo.replace('/', "--"), remote_path))
}

fn builtin_source(name: &str) -> Option<PullSource> {
    routes::builtin_route(name).map(|(canonical, route)| PullSource::Routed {
        name: canonical,
        repo: route.hf_repo,
        file: route.hf_file,
    })
}

pub async fn resolve_via_routes(name: &str) -> Result<Option<PullSource>> {
    let resolved = match routes::resolve_friendly_name(name).await {
        Ok(resolved) => resolved,
        Err(e) => {
            if let Some(source) = builtin_source(name) {
                tracing::warn!("routes lookup failed ({e}); using the built-in route for {name}");
                return Ok(Some(source));
            }
            return Err(e);
        }
    };
    if let Some(msg) = &resolved.message {
        tracing::info!("routes: {msg}");
    }
    if !resolved.found {
        return Ok(builtin_source(name));
    }
    let route = resolved
        .route
        .ok_or_else(|| Ml5Error::Download("routes response missing route".into()))?;
    Ok(Some(PullSource::Routed {
        name: resolved.canonical_name.unwrap_or_else(|| name.to_string()),
        repo: route.hf_repo,
        file: route.hf_file,
    }))
}

struct Progress<'a, F: FnMut(PullEvent)> {
    done: u64,
    total: Option<u64>,
    last: Instant,
    cancel: &'a AtomicBool,
    on_event: &'a mut F,
}

impl<F: FnMut(PullEvent)> Progress<'_, F> {
    fn emit(&mut self, current: u64, force: bool) {
        if force || self.last.elapsed() >= PROGRESS_INTERVAL {
            (self.on_event)(PullEvent::Progress {
                downloaded: self.done + current,
                total: self.total,
            });
            self.last = Instant::now();
        }
    }

    fn check_cancel(&self) -> Result<()> {
        if self.cancel.load(Ordering::Acquire) {
            Err(Ml5Error::Download("download cancelled".into()))
        } else {
            Ok(())
        }
    }
}

async fn hash_existing(path: &Path, hasher: &mut sha2::Sha256) -> Result<u64> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok(total)
}

async fn download_file<F: FnMut(PullEvent)>(
    client: &reqwest::Client,
    repo: &str,
    remote: &Remote,
    dest: &Path,
    token: Option<&str>,
    progress: &mut Progress<'_, F>,
) -> Result<()> {
    progress.check_cancel()?;
    let label = remote.path.clone();
    if let (Ok(meta), Some(size)) = (std::fs::metadata(dest), remote.size) {
        if meta.len() == size {
            progress.done += size;
            progress.emit(0, true);
            return Ok(());
        }
    }
    (progress.on_event)(PullEvent::Downloading { file: label.clone() });
    let tmp = dest.with_file_name(format!(
        "{}.part",
        dest.file_name().and_then(|n| n.to_str()).unwrap_or("download")
    ));
    let url = format!("{HF_RESOLVE_BASE}/{repo}/resolve/main/{}", remote.path);
    let mut hasher = sha2::Sha256::new();
    let mut have = match std::fs::metadata(&tmp) {
        Ok(m) if remote.size.is_some_and(|s| m.len() < s) => m.len(),
        Ok(_) => {
            let _ = tokio::fs::remove_file(&tmp).await;
            0
        }
        Err(_) => 0,
    };
    if have > 0 && remote.sha256.is_some() {
        have = hash_existing(&tmp, &mut hasher).await?;
    }
    let mut req = authed(client.get(&url), token);
    if have > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| Ml5Error::Download(format!("download of {label} failed: {e}")))?;
    let status = resp.status();
    let append = have > 0 && status == reqwest::StatusCode::PARTIAL_CONTENT;
    if !status.is_success() {
        if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            let _ = tokio::fs::remove_file(&tmp).await;
        }
        return Err(access_error(status, &label));
    }
    if !append {
        have = 0;
        hasher = sha2::Sha256::new();
    }
    let file = if append {
        tokio::fs::OpenOptions::new().append(true).open(&tmp).await?
    } else {
        tokio::fs::File::create(&tmp).await?
    };
    let mut writer = tokio::io::BufWriter::with_capacity(1 << 20, file);
    let mut current = have;
    progress.emit(current, true);
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if let Err(e) = progress.check_cancel() {
            drop(writer);
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(e);
        }
        let chunk = chunk.map_err(|e| {
            Ml5Error::Download(format!(
                "connection lost while downloading {label} ({e}); run the pull again to resume"
            ))
        })?;
        writer.write_all(&chunk).await?;
        hasher.update(&chunk);
        current += chunk.len() as u64;
        progress.emit(current, false);
    }
    writer.flush().await?;
    drop(writer);
    progress.emit(current, true);

    (progress.on_event)(PullEvent::Verifying { file: label.clone() });
    if let Some(size) = remote.size {
        if current != size {
            return Err(Ml5Error::Download(format!(
                "{label} is incomplete ({current} of {size} bytes); run the pull again to resume"
            )));
        }
    }
    if let Some(expected) = &remote.sha256 {
        let actual = hex::encode(hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Ml5Error::Download(format!(
                "{label} failed its SHA-256 check; the corrupt download was removed, retry the pull"
            )));
        }
    }
    if label.to_ascii_lowercase().ends_with(".gguf") {
        let mut header = [0u8; 4];
        let mut check = tokio::fs::File::open(&tmp).await?;
        check.read_exact(&mut header).await?;
        if &header != b"GGUF" {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Ml5Error::Download(format!("{label} is not a GGUF file")));
        }
    }
    tokio::fs::rename(&tmp, dest).await?;
    progress.done += current;
    Ok(())
}

pub async fn pull<F>(
    source: &PullSource,
    models_dir: &Path,
    token: Option<&str>,
    quant: Option<&str>,
    cancel: &AtomicBool,
    mut on_event: F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    tokio::fs::create_dir_all(models_dir).await?;
    match source {
        PullSource::HuggingFace {
            repo,
            file,
            quant: tag,
        } => {
            on_event(PullEvent::Resolving { target: repo.clone() });
            let quant = tag.as_deref().or(quant);
            pull_hf(repo, file.as_deref(), models_dir, token, quant, cancel, &mut on_event).await
        }
        PullSource::Routed { name, repo, file } => {
            on_event(PullEvent::Resolving {
                target: format!("{name} -> {repo}"),
            });
            pull_hf(repo, file.as_deref(), models_dir, token, quant, cancel, &mut on_event).await
        }
        PullSource::CoreRegistry { name } => {
            on_event(PullEvent::Resolving {
                target: format!("{CORE_REGISTRY_BASE}/{name}"),
            });
            Err(Ml5Error::Download(format!(
                "'{name}' is not a known model name and the CORe registry is not live yet. Use `ml5 pull hf:owner/repo` or `ml5 pull hf:owner/repo/file.gguf`."
            )))
        }
    }
}

async fn pull_hf<F>(
    repo: &str,
    file: Option<&str>,
    models_dir: &Path,
    token: Option<&str>,
    quant: Option<&str>,
    cancel: &AtomicBool,
    on_event: &mut F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    let client = client()?;
    let listing = match hf_listing(&client, repo, token).await {
        Ok(listing) => listing,
        Err(e) if file.is_some() => {
            tracing::warn!(%e, "Hugging Face listing failed; downloading the named file without size/hash data");
            Vec::new()
        }
        Err(e) => return Err(e),
    };

    let main = match file {
        Some(f) => listing
            .iter()
            .find(|r| r.path == f)
            .cloned()
            .unwrap_or_else(|| Remote::bare(f)),
        None => {
            if !listing.iter().any(|f| is_model_gguf(&f.path)) {
                let has = |n: &str| listing.iter().any(|f| f.path == n);
                if listing.iter().any(|f| f.path.ends_with(".safetensors"))
                    && has("config.json")
                    && has("tokenizer.json")
                {
                    return pull_safetensors(&client, repo, &listing, models_dir, token, cancel, on_event).await;
                }
                return Err(Ml5Error::ModelNotFound(format!(
                    "{repo} has no GGUF files and is not a complete safetensors model (needs config.json, tokenizer.json and *.safetensors)"
                )));
            }
            let picked = pick_gguf(&listing, quant)?.clone();
            tracing::info!(repo, file = %picked.path, "auto-selected GGUF");
            picked
        }
    };
    if model::is_mmproj_file(main.base_name()) {
        return Err(Ml5Error::InvalidRequest(format!(
            "{} is a vision projector, not a model; pull the model file and its projector is fetched automatically",
            main.path
        )));
    }

    let mut parts = vec![main.clone()];
    if let Some((base, idx, total)) = model::split_part(main.base_name()) {
        if idx != 1 {
            return Err(Ml5Error::InvalidRequest(format!(
                "{} is part {idx} of a split model; pull part 00001 instead",
                main.path
            )));
        }
        let dir = main.dir();
        parts = (1..=total)
            .map(|i| {
                let name = model::split_part_name(&base, i, total);
                let path = if dir.is_empty() { name } else { format!("{dir}/{name}") };
                listing
                    .iter()
                    .find(|r| r.path == path)
                    .cloned()
                    .unwrap_or_else(|| Remote::bare(&path))
            })
            .collect();
    }
    let mmproj = pick_mmproj(&listing, &main).cloned();

    let mut all: Vec<&Remote> = parts.iter().collect();
    if let Some(m) = &mmproj {
        all.push(m);
    }
    let total = all
        .iter()
        .map(|r| r.size)
        .try_fold(0u64, |acc, s| s.map(|s| acc + s));
    let mut progress = Progress {
        done: 0,
        total,
        last: Instant::now(),
        cancel,
        on_event,
    };

    let mut local_first = PathBuf::new();
    for (i, part) in parts.iter().enumerate() {
        let dest = models_dir.join(local_name(repo, &part.path));
        download_file(&client, repo, part, &dest, token, &mut progress).await?;
        if i == 0 {
            local_first = dest;
        }
    }
    let mmproj_local = match &mmproj {
        Some(m) => {
            let dest = models_dir.join(local_name(repo, &m.path));
            download_file(&client, repo, m, &dest, token, &mut progress).await?;
            dest.file_name().and_then(|n| n.to_str()).map(str::to_string)
        }
        None => None,
    };

    let meta_path = model::gguf_meta_path(&local_first);
    let previous = ModelMeta::load(&meta_path);
    let meta = ModelMeta {
        friendly_name: previous.friendly_name,
        source_repo: Some(repo.to_string()),
        source_file: Some(main.path.clone()),
        mmproj_file: mmproj_local,
    };
    if let Err(e) = meta.save(&meta_path) {
        tracing::warn!(%e, "could not write model metadata");
    }
    let file_name = local_first
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf")
        .to_string();
    Ok((model::gguf_model_name(&file_name), local_first))
}

async fn pull_safetensors<F>(
    client: &reqwest::Client,
    repo: &str,
    listing: &[Remote],
    models_dir: &Path,
    token: Option<&str>,
    cancel: &AtomicBool,
    on_event: &mut F,
) -> Result<(String, PathBuf)>
where
    F: FnMut(PullEvent),
{
    on_event(PullEvent::Resolving {
        target: format!("{repo} (safetensors)"),
    });
    let wanted: Vec<&Remote> = listing
        .iter()
        .filter(|f| {
            !f.path.contains('/')
                && (matches!(
                    f.path.as_str(),
                    "config.json"
                        | "tokenizer.json"
                        | "tokenizer_config.json"
                        | "special_tokens_map.json"
                        | "generation_config.json"
                        | "chat_template.jinja"
                        | "model.safetensors.index.json"
                ) || f.path.ends_with(".safetensors"))
        })
        .collect();
    let dir_name = sanitize_filename(&repo.replace('/', "--"));
    let model_dir = models_dir.join(&dir_name);
    tokio::fs::create_dir_all(&model_dir).await?;
    let total = wanted
        .iter()
        .map(|r| r.size)
        .try_fold(0u64, |acc, s| s.map(|s| acc + s));
    let mut progress = Progress {
        done: 0,
        total,
        last: Instant::now(),
        cancel,
        on_event,
    };
    for remote in wanted {
        let dest = model_dir.join(&remote.path);
        download_file(client, repo, remote, &dest, token, &mut progress).await?;
    }
    let meta_path = model_dir.join("model.meta.json");
    let previous = ModelMeta::load(&meta_path);
    let meta = ModelMeta {
        friendly_name: previous.friendly_name,
        source_repo: Some(repo.to_string()),
        source_file: None,
        mmproj_file: None,
    };
    if let Err(e) = meta.save(&meta_path) {
        tracing::warn!(%e, "could not write model metadata");
    }
    Ok((dir_name, model_dir))
}
