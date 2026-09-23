//! Backend-neutral multiverse policy/value network.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod encoding;
mod model;
mod network;

pub use encoding::{
    ACTION_FEATURES, BOARD_METADATA_FEATURES, BOARD_PLANES, EncodedAction, EncodedBoard,
    EncodedPosition, GLOBAL_FEATURES, encode, encode_action,
};
pub use model::{BatchTensors, ModelOutput, ModelSize, MultiverseNet, NetworkConfig};
pub use network::{
    ModelError, PolicyValueEvaluator, PolicyValueNetwork, Prediction, TrainConfig, TrainMetrics,
    TrainingExample,
};
