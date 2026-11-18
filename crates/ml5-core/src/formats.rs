//! Pluggable model formats.
//!
//! A distribution of ML5 can register extra on-disk model formats at startup (before the engine is
//! created). The engine then lists, loads and serves those files like built-in models: discovery asks
//! each registered format whether it handles a path, and loading asks it for a backend.

use crate::backend::Backend;
use crate::model::ModelInfo;
use std::path::Path;
use std::sync::{Arc, OnceLock, RwLock};

/// An on-disk model format provided outside ml5-core.
pub trait ModelFormat: Send + Sync {
    /// Short name, reported as the model's `format`.
    fn name(&self) -> &'static str;

    /// Does this format handle the file at `path`? Should be cheap (extension or magic check).
    fn matches(&self, path: &Path) -> bool;

    /// Describe a model file for listing. `subdir` is the store subdirectory the file sits in, if
    /// any (used to build the model name). Return `None` to skip an unreadable file.
    /// The returned info must use `ModelType::External`.
    fn describe(&self, path: &Path, subdir: Option<&str>) -> Option<ModelInfo>;

    /// A fresh backend instance to load and serve one model of this format.
    fn backend(&self) -> Arc<dyn Backend>;

    /// Upper bound for the automatic context size, if the format needs one.
    fn context_cap(&self) -> Option<u32> {
        None
    }

    /// If true, at most one model of this format is loaded at a time (loading one unloads the others).
    fn exclusive_gpu(&self) -> bool {
        false
    }
}

fn registry() -> &'static RwLock<Vec<Arc<dyn ModelFormat>>> {
    static R: OnceLock<RwLock<Vec<Arc<dyn ModelFormat>>>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(Vec::new()))
}

/// Register a format. Call before creating the engine.
pub fn register(format: Arc<dyn ModelFormat>) {
    registry().write().unwrap_or_else(|e| e.into_inner()).push(format);
}

/// The registered format that handles `path`, if any.
pub fn for_path(path: &Path) -> Option<Arc<dyn ModelFormat>> {
    registry().read().unwrap_or_else(|e| e.into_inner()).iter().find(|f| f.matches(path)).cloned()
}
