//! AlphaZero-style neural search, self-play, and training primitives.
//!
//! The network consumes only game state and legal-action representations. Its
//! targets come from MCTS visit counts and self-play outcomes; no expert games
//! or handcrafted position scores are used.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod bot;
mod search;
mod self_play;

mod encoding {
    pub use huginn_neural::encode;
}

mod network {
    pub use huginn_neural::{
        ModelError, NetworkConfig, PolicyValueEvaluator, PolicyValueNetwork, TrainingExample,
    };
}

pub use bot::{HuginnBot, MuninnBot};
pub use huginn_neural::{
    ACTION_FEATURES, BOARD_METADATA_FEATURES, BOARD_PLANES, EncodedAction, EncodedBoard,
    EncodedPosition, GLOBAL_FEATURES, ModelError, ModelSize, NetworkConfig, PolicyValueEvaluator,
    PolicyValueNetwork, Prediction, TrainConfig, TrainMetrics, TrainingExample, encode,
    encode_action,
};
pub use search::{Mcts, SearchConfig, SearchResult};
pub use self_play::{
    ArenaGameConfig, ArenaGameResult, ArenaProgress, ArenaReport, ReplayBuffer, ReplayError,
    ReplayStep, SelfPlayConfig, SelfPlayGame, play_arena_game, play_self_play_game, run_arena,
    run_arena_schedule, run_arena_schedule_with_progress,
};
