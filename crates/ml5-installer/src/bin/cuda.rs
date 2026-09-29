use ml5_installer::{run_installer, Backend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("ML5 Installer — CUDA backend");
    run_installer(Backend::Cuda).await
}
