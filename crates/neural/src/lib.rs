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
    BatchMemoryEstimate, ModelError, PolicyValueEvaluator, PolicyValueNetwork, Prediction,
    TrainConfig, TrainMetrics, TrainingExample, estimate_batch_memory_from_shape,
    estimate_inference_batch_memory, estimate_training_batch_memory,
};
#[cfg(all(feature = "training", feature = "vulkan"))]
pub use network::{VulkanDeviceInfo, VulkanPolicyValueEvaluator, VulkanTrainingDevice};
