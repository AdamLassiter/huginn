use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use huginn_core::BOARD_SIZE;
use serde::{Deserialize, Serialize};

use crate::{
    ACTION_FEATURES, BOARD_METADATA_FEATURES, BOARD_PLANES, EncodedPosition, GLOBAL_FEATURES,
};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelSize {
    Compact,
    #[default]
    Large,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetworkConfig {
    pub channels: usize,
    pub residual_blocks: usize,
    pub board_embedding: usize,
    pub context_width: usize,
    pub action_width: usize,
    pub value_width: usize,
    pub group_norm_groups: usize,
}

impl NetworkConfig {
    #[must_use]
    pub const fn compact() -> Self {
        Self {
            channels: 24,
            residual_blocks: 2,
            board_embedding: 96,
            context_width: 128,
            action_width: 96,
            value_width: 64,
            group_norm_groups: 8,
        }
    }

    #[must_use]
    pub const fn large() -> Self {
        Self {
            channels: 64,
            residual_blocks: 6,
            board_embedding: 192,
            context_width: 256,
            action_width: 192,
            value_width: 128,
            group_norm_groups: 8,
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            channels: 8,
            residual_blocks: 1,
            board_embedding: 16,
            context_width: 16,
            action_width: 16,
            value_width: 8,
            group_norm_groups: 4,
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self::large()
    }
}

impl From<ModelSize> for NetworkConfig {
    fn from(value: ModelSize) -> Self {
        match value {
            ModelSize::Compact => Self::compact(),
            ModelSize::Large => Self::large(),
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    norm1: GroupNorm<B>,
    conv2: Conv2d<B>,
    norm2: GroupNorm<B>,
}

impl<B: Backend> ResidualBlock<B> {
    fn new(config: NetworkConfig, device: &B::Device) -> Self {
        let convolution = || {
            Conv2dConfig::new([config.channels, config.channels], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        let normalization =
            || GroupNormConfig::new(config.group_norm_groups, config.channels).init(device);
        Self {
            conv1: convolution(),
            norm1: normalization(),
            conv2: convolution(),
            norm2: normalization(),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = input.clone();
        let output = gelu(self.norm1.forward(self.conv1.forward(input)));
        gelu(self.norm2.forward(self.conv2.forward(output)) + residual)
    }
}

#[derive(Module, Debug)]
pub struct MultiverseNet<B: Backend> {
    stem: Conv2d<B>,
    stem_norm: GroupNorm<B>,
    residual: Vec<ResidualBlock<B>>,
    spatial_embedding: Linear<B>,
    metadata_embedding: Linear<B>,
    context: Linear<B>,
    action_hidden: Linear<B>,
    policy: Linear<B>,
    value_hidden: Linear<B>,
    value: Linear<B>,
    config: NetworkConfig,
}

impl<B: Backend> MultiverseNet<B> {
    /// Creates the network on a backend device.
    ///
    /// # Panics
    ///
    /// Panics when the configured channel count is not divisible by the group count.
    #[must_use]
    pub fn new(config: NetworkConfig, device: &B::Device) -> Self {
        assert!(config.channels.is_multiple_of(config.group_norm_groups));
        let stem = Conv2dConfig::new([BOARD_PLANES, config.channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let stem_norm =
            GroupNormConfig::new(config.group_norm_groups, config.channels).init(device);
        let residual = (0..config.residual_blocks)
            .map(|_| ResidualBlock::new(config, device))
            .collect();
        Self {
            stem,
            stem_norm,
            residual,
            spatial_embedding: LinearConfig::new(config.channels, config.board_embedding)
                .init(device),
            metadata_embedding: LinearConfig::new(BOARD_METADATA_FEATURES, config.board_embedding)
                .init(device),
            context: LinearConfig::new(
                config.board_embedding * 2 + GLOBAL_FEATURES,
                config.context_width,
            )
            .init(device),
            action_hidden: LinearConfig::new(
                config.context_width + config.board_embedding * 2 + ACTION_FEATURES,
                config.action_width,
            )
            .init(device),
            policy: LinearConfig::new(config.action_width, 1).init(device),
            value_hidden: LinearConfig::new(config.context_width, config.value_width).init(device),
            value: LinearConfig::new(config.value_width, 1).init(device),
            config,
        }
    }

    #[must_use]
    pub fn config(&self) -> NetworkConfig {
        self.config
    }

    #[must_use]
    pub fn forward(&self, batch: BatchTensors<B>) -> ModelOutput<B> {
        let [batch_size, board_count, _, _, _] = batch.boards.dims();
        let action_count = batch.actions.dims()[1];
        let mut spatial = batch.boards.reshape([
            batch_size * board_count,
            BOARD_PLANES,
            BOARD_SIZE,
            BOARD_SIZE,
        ]);
        spatial = gelu(self.stem_norm.forward(self.stem.forward(spatial)));
        for block in &self.residual {
            spatial = block.forward(spatial);
        }
        let spatial = spatial.mean_dim(3).mean_dim(2).squeeze_dims::<2>(&[2, 3]);
        let spatial = self.spatial_embedding.forward(spatial);
        let metadata = self.metadata_embedding.forward(
            batch
                .metadata
                .reshape([batch_size * board_count, BOARD_METADATA_FEATURES]),
        );
        let embeddings = gelu(spatial + metadata).reshape([
            batch_size,
            board_count,
            self.config.board_embedding,
        ]);

        let board_mask = batch.board_mask.clone().unsqueeze_dim::<3>(2);
        let board_sum = (embeddings.clone() * board_mask.clone())
            .sum_dim(1)
            .squeeze_dim::<2>(1);
        let board_divisor = board_mask
            .clone()
            .sum_dim(1)
            .clamp_min(1.0)
            .squeeze_dim::<2>(1);
        let board_mean = board_sum / board_divisor;
        let board_max = (embeddings.clone() * board_mask.clone()
            + (board_mask.clone().neg() + 1.0) * -1.0e9)
            .max_dim(1)
            .squeeze_dim::<2>(1);
        let context = gelu(
            self.context
                .forward(Tensor::cat(vec![board_mean, board_max, batch.global], 1)),
        );

        let source = embeddings.clone().gather(1, batch.source_indices) * batch.move_mask.clone();
        let destination = embeddings.gather(1, batch.destination_indices) * batch.move_mask;
        let repeated_context = context
            .clone()
            .unsqueeze_dim::<3>(1)
            .repeat_dim(1, action_count);
        let action_input = Tensor::cat(
            vec![repeated_context, source, destination, batch.actions],
            2,
        );
        let logits = self
            .policy
            .forward(gelu(self.action_hidden.forward(action_input)))
            .squeeze_dim::<2>(2);
        let logits =
            logits * batch.action_mask.clone() + (batch.action_mask.clone().neg() + 1.0) * -1.0e9;
        let value = self
            .value
            .forward(gelu(self.value_hidden.forward(context)))
            .tanh()
            .squeeze_dim::<1>(1);
        ModelOutput { logits, value }
    }
}

pub struct ModelOutput<B: Backend> {
    pub logits: Tensor<B, 2>,
    pub value: Tensor<B, 1>,
}

pub struct BatchTensors<B: Backend> {
    pub boards: Tensor<B, 5>,
    pub metadata: Tensor<B, 3>,
    pub board_mask: Tensor<B, 2>,
    pub global: Tensor<B, 2>,
    pub actions: Tensor<B, 3>,
    pub action_mask: Tensor<B, 2>,
    pub source_indices: Tensor<B, 3, Int>,
    pub destination_indices: Tensor<B, 3, Int>,
    pub move_mask: Tensor<B, 3>,
}

impl<B: Backend> BatchTensors<B> {
    /// Pads and tensors a non-empty collection of encoded positions.
    ///
    /// # Panics
    ///
    /// Panics for an empty batch or if a board index cannot fit the backend index type.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn from_positions(
        positions: &[EncodedPosition],
        embedding_width: usize,
        device: &B::Device,
    ) -> Self {
        assert!(!positions.is_empty(), "a neural batch cannot be empty");
        let batch_size = positions.len();
        let board_count = positions
            .iter()
            .map(|position| position.boards.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let action_count = positions
            .iter()
            .map(|position| position.actions.len())
            .max()
            .unwrap_or(1)
            .max(1);

        let mut boards = vec![0.0; batch_size * board_count * BOARD_PLANES * 121];
        let mut metadata = vec![0.0; batch_size * board_count * BOARD_METADATA_FEATURES];
        let mut board_mask = vec![0.0; batch_size * board_count];
        let mut global = vec![0.0; batch_size * GLOBAL_FEATURES];
        let mut actions = vec![0.0; batch_size * action_count * ACTION_FEATURES];
        let mut action_mask = vec![0.0; batch_size * action_count];
        let mut source_indices = vec![0_i64; batch_size * action_count * embedding_width];
        let mut destination_indices = vec![0_i64; batch_size * action_count * embedding_width];
        let mut move_mask = vec![0.0; batch_size * action_count * embedding_width];
        for (batch, position) in positions.iter().enumerate() {
            global[batch * GLOBAL_FEATURES..(batch + 1) * GLOBAL_FEATURES]
                .copy_from_slice(&position.global);
            for (board, encoded) in position.boards.iter().enumerate() {
                let plane_start = (batch * board_count + board) * BOARD_PLANES * 121;
                boards[plane_start..plane_start + BOARD_PLANES * 121]
                    .copy_from_slice(&encoded.planes);
                let metadata_start = (batch * board_count + board) * BOARD_METADATA_FEATURES;
                metadata[metadata_start..metadata_start + BOARD_METADATA_FEATURES]
                    .copy_from_slice(&encoded.metadata);
                board_mask[batch * board_count + board] = 1.0;
            }
            for (action, encoded) in position.actions.iter().enumerate() {
                let action_start = (batch * action_count + action) * ACTION_FEATURES;
                actions[action_start..action_start + ACTION_FEATURES]
                    .copy_from_slice(&encoded.features);
                action_mask[batch * action_count + action] = 1.0;
                if let (Some(source), Some(destination)) =
                    (encoded.source_board, encoded.destination_board)
                {
                    let gather_start = (batch * action_count + action) * embedding_width;
                    let source = i64::try_from(source).expect("board index fits i64");
                    let destination = i64::try_from(destination).expect("board index fits i64");
                    source_indices[gather_start..gather_start + embedding_width].fill(source);
                    destination_indices[gather_start..gather_start + embedding_width]
                        .fill(destination);
                    move_mask[gather_start..gather_start + embedding_width].fill(1.0);
                }
            }
        }
        Self {
            boards: Tensor::from_data(
                TensorData::new(
                    boards,
                    [
                        batch_size,
                        board_count,
                        BOARD_PLANES,
                        BOARD_SIZE,
                        BOARD_SIZE,
                    ],
                ),
                device,
            ),
            metadata: Tensor::from_data(
                TensorData::new(metadata, [batch_size, board_count, BOARD_METADATA_FEATURES]),
                device,
            ),
            board_mask: Tensor::from_data(
                TensorData::new(board_mask, [batch_size, board_count]),
                device,
            ),
            global: Tensor::from_data(
                TensorData::new(global, [batch_size, GLOBAL_FEATURES]),
                device,
            ),
            actions: Tensor::from_data(
                TensorData::new(actions, [batch_size, action_count, ACTION_FEATURES]),
                device,
            ),
            action_mask: Tensor::from_data(
                TensorData::new(action_mask, [batch_size, action_count]),
                device,
            ),
            source_indices: Tensor::from_data(
                TensorData::new(source_indices, [batch_size, action_count, embedding_width]),
                device,
            ),
            destination_indices: Tensor::from_data(
                TensorData::new(
                    destination_indices,
                    [batch_size, action_count, embedding_width],
                ),
                device,
            ),
            move_mask: Tensor::from_data(
                TensorData::new(move_mask, [batch_size, action_count, embedding_width]),
                device,
            ),
        }
    }
}
