//! AlphaZero-style neural search, self-play, and training primitives.
//!
//! The network consumes only game state and legal-action representations. Its
//! targets come from MCTS visit counts and self-play outcomes; no expert games
//! or handcrafted position scores are used.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod bot;
mod encoding;
mod network;
mod search;
mod self_play;

pub use bot::MuninnBot;
pub use encoding::{
    ACTION_FEATURES, EncodedPosition, STATE_FEATURES, encode, encode_action, encode_state,
};
pub use network::{
    ModelError, NetworkConfig, PolicyValueNetwork, Prediction, TrainConfig, TrainMetrics,
    TrainingExample,
};
pub use search::{Mcts, SearchConfig, SearchResult};
pub use self_play::{
    ArenaGameConfig, ArenaGameResult, ArenaProgress, ArenaReport, ReplayBuffer, ReplayError,
    SelfPlayConfig, SelfPlayGame, play_arena_game, play_self_play_game, run_arena,
    run_arena_schedule, run_arena_schedule_with_progress,
};
