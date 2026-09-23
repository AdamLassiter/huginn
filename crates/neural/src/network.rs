use std::fs;
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::Backend;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "training")]
use burn::optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig};
#[cfg(feature = "training")]
use burn::tensor::{Tensor, TensorData, backend::AutodiffBackend};

use crate::{BatchTensors, EncodedPosition, MultiverseNet, NetworkConfig};

const FORMAT_VERSION: u32 = 2;
type CpuBackend = NdArray<f32>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingExample {
    pub position: EncodedPosition,
    pub policy: Vec<f32>,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub policy: Vec<f32>,
    pub value: f32,
}

/// Backend-independent policy/value evaluation boundary used by search code.
pub trait PolicyValueEvaluator: Send + Sync {
    fn predict(&self, position: &EncodedPosition) -> Prediction;

    fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction>;
}

#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub batch_token_budget: usize,
    pub learning_rate: f32,
    pub l2: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 4,
            batch_size: 64,
            batch_token_budget: 262_144,
            learning_rate: 0.001,
            l2: 0.0001,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub examples: usize,
}

#[derive(Clone, Debug)]
pub struct PolicyValueNetwork {
    model: MultiverseNet<CpuBackend>,
    config: NetworkConfig,
    training_steps: u64,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("model JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("model record is invalid: {0}")]
    Record(String),
    #[error(
        "unsupported model format version {0}; version-1 checkpoints are not compatible with the multiverse network"
    )]
    Version(u32),
    #[error("checkpoint architecture does not match its declared configuration")]
    Shape,
}

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    format_version: u32,
    config: NetworkConfig,
    training_steps: u64,
    encoding: String,
    weights: String,
}

impl PolicyValueNetwork {
    #[must_use]
    pub fn random(config: NetworkConfig, rng: &mut impl Rng) -> Self {
        let device = NdArrayDevice::default();
        let seed: u64 = rng.random();
        <CpuBackend as Backend>::seed(&device, seed);
        Self {
            model: MultiverseNet::new(config, &device),
            config,
            training_steps: 0,
        }
    }

    #[must_use]
    pub const fn config(&self) -> NetworkConfig {
        self.config
    }

    #[must_use]
    pub const fn training_steps(&self) -> u64 {
        self.training_steps
    }

    #[must_use]
    /// Evaluates one encoded position on the CPU.
    ///
    /// # Panics
    ///
    /// Panics if the encoded position is empty or contains malformed tensor data.
    pub fn predict(&self, position: &EncodedPosition) -> Prediction {
        <Self as PolicyValueEvaluator>::predict(self, position)
    }

    /// Evaluates multiple variably-sized positions in one padded CPU batch.
    ///
    /// # Panics
    ///
    /// Panics if any encoded position is empty or contains malformed tensor data.
    #[must_use]
    pub fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
        <Self as PolicyValueEvaluator>::predict_batch(self, positions)
    }

    fn predict_batch_cpu(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
        if positions.is_empty() {
            return Vec::new();
        }
        let device = NdArrayDevice::default();
        let batch = BatchTensors::from_positions(positions, self.config.board_embedding, &device);
        let output = self.model.forward(batch);
        let action_count = output.logits.dims()[1];
        let policies = softmax(output.logits, 1)
            .to_data()
            .to_vec::<f32>()
            .expect("CPU policy tensor uses f32");
        let values = output
            .value
            .to_data()
            .to_vec::<f32>()
            .expect("CPU value tensor uses f32");
        positions
            .iter()
            .enumerate()
            .map(|(row, position)| Prediction {
                policy: policies[row * action_count..row * action_count + position.actions.len()]
                    .to_vec(),
                value: values[row],
            })
            .collect()
    }

    #[cfg(feature = "training")]
    /// Fits examples with the CPU autodiff backend.
    ///
    /// # Panics
    ///
    /// Panics only if Burn cannot round-trip its own in-memory model record.
    pub fn train(
        &mut self,
        examples: &[TrainingExample],
        config: TrainConfig,
        rng: &mut impl Rng,
    ) -> TrainMetrics {
        let seed: u64 = rng.random();
        let bytes = record_bytes(self.model.clone()).expect("in-memory CPU record");
        let (model, metrics, updates) = train_backend::<Autodiff<CpuBackend>>(
            &bytes,
            self.config,
            examples,
            config,
            seed,
            &NdArrayDevice::default(),
        )
        .expect("CPU training record is compatible");
        let bytes = record_bytes(model).expect("in-memory trained CPU record");
        self.model = model_from_bytes(self.config, bytes, &NdArrayDevice::default())
            .expect("trained CPU record is compatible");
        self.training_steps += updates;
        metrics
    }

    /// Saves a versioned JSON manifest containing a backend-neutral full-precision record.
    ///
    /// # Errors
    ///
    /// Returns an error when the checkpoint cannot be encoded or written atomically.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let checkpoint = Checkpoint {
            format_version: FORMAT_VERSION,
            config: self.config,
            training_steps: self.training_steps,
            encoding: "base64-named-messagepack-f32".to_owned(),
            weights: BASE64.encode(record_bytes(self.model.clone())?),
        };
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec(&checkpoint)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Loads a version-2 checkpoint for CPU inference.
    ///
    /// # Errors
    ///
    /// Returns an error for I/O, malformed JSON or records, or incompatible versions.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let checkpoint: Checkpoint = serde_json::from_slice(&fs::read(path)?)?;
        if checkpoint.format_version != FORMAT_VERSION {
            return Err(ModelError::Version(checkpoint.format_version));
        }
        if checkpoint.encoding != "base64-named-messagepack-f32" {
            return Err(ModelError::Record(format!(
                "unsupported record encoding {}",
                checkpoint.encoding
            )));
        }
        let bytes = BASE64
            .decode(checkpoint.weights)
            .map_err(|error| ModelError::Record(error.to_string()))?;
        let model = model_from_bytes(checkpoint.config, bytes, &NdArrayDevice::default())?;
        Ok(Self {
            model,
            config: checkpoint.config,
            training_steps: checkpoint.training_steps,
        })
    }

    #[cfg(all(feature = "training", feature = "vulkan"))]
    /// Fits examples on an explicitly initialized Vulkan device.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend-neutral model record cannot be transferred.
    /// Vulkan adapter and driver initialization failures are surfaced by Burn.
    pub fn train_vulkan(
        &mut self,
        examples: &[TrainingExample],
        config: TrainConfig,
        rng: &mut impl Rng,
    ) -> Result<TrainMetrics, ModelError> {
        use burn::backend::Vulkan;
        use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics, init_setup};

        let device = WgpuDevice::default();
        init_setup::<graphics::Vulkan>(&device, RuntimeOptions::default());
        let seed: u64 = rng.random();
        let bytes = record_bytes(self.model.clone())?;
        let (model, metrics, updates) = train_backend::<Autodiff<Vulkan>>(
            &bytes,
            self.config,
            examples,
            config,
            seed,
            &device,
        )?;
        let bytes = record_bytes(model)?;
        self.model = model_from_bytes(self.config, bytes, &NdArrayDevice::default())?;
        self.training_steps += updates;
        Ok(metrics)
    }
}

impl PolicyValueEvaluator for PolicyValueNetwork {
    fn predict(&self, position: &EncodedPosition) -> Prediction {
        self.predict_batch_cpu(std::slice::from_ref(position))
            .pop()
            .expect("single-position batch returns one prediction")
    }

    fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
        self.predict_batch_cpu(positions)
    }
}

fn record_bytes<B: Backend>(model: MultiverseNet<B>) -> Result<Vec<u8>, ModelError> {
    NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .record(model.into_record(), ())
        .map_err(|error| ModelError::Record(error.to_string()))
}

fn model_from_bytes<B: Backend>(
    config: NetworkConfig,
    bytes: Vec<u8>,
    device: &B::Device,
) -> Result<MultiverseNet<B>, ModelError> {
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load(bytes, device)
        .map_err(|error| ModelError::Record(error.to_string()))?;
    Ok(MultiverseNet::new(config, device).load_record(record))
}

#[cfg(feature = "training")]
fn train_backend<AB: AutodiffBackend>(
    record: &[u8],
    network_config: NetworkConfig,
    examples: &[TrainingExample],
    config: TrainConfig,
    seed: u64,
    device: &AB::Device,
) -> Result<(MultiverseNet<AB::InnerBackend>, TrainMetrics, u64), ModelError> {
    use rand::seq::SliceRandom;

    if examples.is_empty() || config.epochs == 0 {
        let model = model_from_bytes::<AB>(network_config, record.to_vec(), device)?;
        return Ok((model.valid(), TrainMetrics::default(), 0));
    }
    AB::seed(device, seed);
    let mut model = model_from_bytes::<AB>(network_config, record.to_vec(), device)?;
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.l2)))
        .init();
    let mut order = (0..examples.len()).collect::<Vec<_>>();
    let mut random = rand::rngs::StdRng::seed_from_u64(seed);
    let mut metrics = TrainMetrics::default();
    let mut updates = 0_u64;
    for _ in 0..config.epochs {
        order.shuffle(&mut random);
        let mut cursor = 0;
        while cursor < order.len() {
            let end = choose_batch_end(examples, &order, cursor, config);
            let selected = order[cursor..end]
                .iter()
                .map(|&index| &examples[index])
                .collect::<Vec<_>>();
            let positions = selected
                .iter()
                .map(|example| example.position.clone())
                .collect::<Vec<_>>();
            let batch =
                BatchTensors::from_positions(&positions, network_config.board_embedding, device);
            let action_count = positions
                .iter()
                .map(|position| position.actions.len())
                .max()
                .unwrap_or(1)
                .max(1);
            let mut targets = vec![0.0; selected.len() * action_count];
            let mut values = Vec::with_capacity(selected.len());
            for (row, example) in selected.iter().enumerate() {
                targets[row * action_count..row * action_count + example.policy.len()]
                    .copy_from_slice(&example.policy);
                values.push(example.value);
            }
            let targets = Tensor::<AB, 2>::from_data(
                TensorData::new(targets, [selected.len(), action_count]),
                device,
            );
            let values =
                Tensor::<AB, 1>::from_data(TensorData::new(values, [selected.len()]), device);
            let output = model.forward(batch);
            let policy_loss =
                (targets * log_softmax(output.logits, 1)).sum().neg() / selected.len() as f32;
            let value_loss = (output.value - values).powf_scalar(2.0).mean();
            let loss = policy_loss.clone() + value_loss.clone();
            metrics.policy_loss += scalar(&policy_loss) * selected.len() as f32;
            metrics.value_loss += scalar(&value_loss) * selected.len() as f32;
            metrics.examples += selected.len();
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(f64::from(config.learning_rate), model, gradients);
            updates += 1;
            cursor = end;
        }
    }
    if metrics.examples > 0 {
        metrics.policy_loss /= metrics.examples as f32;
        metrics.value_loss /= metrics.examples as f32;
    }
    Ok((model.valid(), metrics, updates))
}

#[cfg(feature = "training")]
fn choose_batch_end(
    examples: &[TrainingExample],
    order: &[usize],
    start: usize,
    config: TrainConfig,
) -> usize {
    let maximum = (start + config.batch_size.max(1)).min(order.len());
    let mut end = start;
    let mut boards = 0_usize;
    let mut actions = 0_usize;
    while end < maximum {
        let position = &examples[order[end]].position;
        let cost = position.boards.len() * 121 + position.actions.len();
        if end > start && boards + actions + cost > config.batch_token_budget.max(1) {
            break;
        }
        boards += position.boards.len() * 121;
        actions += position.actions.len();
        end += 1;
    }
    end.max(start + 1)
}

#[cfg(feature = "training")]
fn scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f32 {
    tensor
        .clone()
        .to_data()
        .to_vec::<f32>()
        .expect("loss tensor uses f32")[0]
}

#[cfg(test)]
mod tests {
    use huginn_core::{Game, Ruleset};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::encode;

    #[test]
    fn prediction_masks_padding_and_checkpoint_round_trips() {
        let game = Game::new(Ruleset::Classic);
        let position = encode(&game, &game.legal_actions());
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let network = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let before = network.predict(&position);
        assert_eq!(before.policy.len(), position.actions.len());
        assert!((before.policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-4);
        let mut shorter = position.clone();
        shorter.actions.truncate(3);
        let batched = network.predict_batch(&[position.clone(), shorter]);
        assert_eq!(batched[0].policy.len(), position.actions.len());
        assert_eq!(batched[1].policy.len(), 3);
        assert!((batched[1].policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-4);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model-v2.json");
        network.save(&path).unwrap();
        let after = PolicyValueNetwork::load(path).unwrap().predict(&position);
        assert_eq!(before.policy, after.policy);
        assert!((before.value - after.value).abs() < f32::EPSILON);
    }

    #[test]
    fn cpu_training_updates_steps_and_reports_finite_losses() {
        let game = Game::new(Ruleset::Classic);
        let position = encode(&game, &game.legal_actions());
        let policy = vec![1.0 / position.actions.len() as f32; position.actions.len()];
        let example = TrainingExample {
            position,
            policy,
            value: 0.25,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut network = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let metrics = network.train(
            &[example],
            TrainConfig {
                epochs: 1,
                batch_size: 1,
                ..TrainConfig::default()
            },
            &mut rng,
        );
        assert_eq!(metrics.examples, 1);
        assert!(metrics.policy_loss.is_finite());
        assert!(metrics.value_loss.is_finite());
        assert_eq!(network.training_steps(), 1);
    }
}
