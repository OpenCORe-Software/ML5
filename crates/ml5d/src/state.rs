use ml5_core::Engine;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
}
