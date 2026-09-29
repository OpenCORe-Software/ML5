use ml5_installer::{run_installer, Backend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("ML5 Installer — Vulkan backend");
    run_installer(Backend::Vulkan).await
}
