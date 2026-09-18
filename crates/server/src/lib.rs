mod api;
mod db;
mod web;

use std::path::Path;

use axum::Router;
use db::Database;
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
        Self::new(database, vec![Box::new(RavenBot::default())])
    }
}

pub fn build_app(state: AppState) -> Router {
    api::router(state)
}
