use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;

use huginn_server::{AppState, build_app};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let address: SocketAddr = env::var("HUGINN_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:3000".to_owned())
        .parse()?;
    let database =
        PathBuf::from(env::var("HUGINN_DATABASE").unwrap_or_else(|_| "huginn.sqlite3".to_owned()));
    let state = AppState::open(database)?;
    let listener = TcpListener::bind(address).await?;
    tracing::info!(%address, "Huginn server listening");
    axum::serve(listener, build_app(state)).await?;
    Ok(())
}
