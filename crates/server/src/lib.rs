mod api;
mod db;
mod web;

use std::path::Path;

use axum::Router;
use db::Database;
use huginn_alphazero::{MuninnBot, SearchConfig};
use huginn_random_bot::RavenBot;

pub use api::AppState;

impl AppState {
    /// Opens the `SQLite` database, applies migrations, and registers built-in bots.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let database = Database::open(path)?;
        let search = SearchConfig {
            simulations: std::env::var("HUGINN_AZ_SIMULATIONS")
                .ok()
                .map(|value| value.parse())
                .transpose()?
                .unwrap_or(48),
            ..SearchConfig::default()
        };
        let muninn = if let Some(path) = std::env::var_os("HUGINN_AZ_MODEL") {
            MuninnBot::load(path, search, 0x4d55_4e49_4e4e)?
        } else {
            let default_path = Path::new("models/training/best.json");
            if default_path.exists() {
                MuninnBot::load(default_path, search, 0x4d55_4e49_4e4e)?
            } else {
                tracing::warn!(
                    "models/training/best.json is absent; Muninn is using an untrained bootstrap network"
                );
                MuninnBot::bootstrap(search, 0x4d55_4e49_4e4e)
            }
        };
        Self::new(
            database,
            vec![Box::new(RavenBot::default()), Box::new(muninn)],
        )
    }
}

pub fn build_app(state: AppState) -> Router {
    api::router(state)
}
