use ml5_installer::{run_installer, Backend};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("ML5 Installer — Legacy-CUDA backend (GTX 10-series)");
    run_installer(Backend::LegacyCuda).await
}
