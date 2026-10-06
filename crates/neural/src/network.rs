use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
#[cfg(all(feature = "training", feature = "vulkan"))]
use huginn_core::{Game, Ruleset};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "training")]
use burn::optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig};
#[cfg(feature = "training")]
use burn::tensor::backend::AutodiffBackend;

use crate::{
    ACTION_FEATURES, BOARD_METADATA_FEATURES, BOARD_PLANES, BatchTensors, EncodedBoard,
    EncodedPosition, MultiverseNet, NetworkConfig,
};

const FORMAT_VERSION: u32 = 2;
const SPATIAL_CACHE_LIMIT: usize = 16_384;
const SPATIAL_KEY_WORDS: usize = (BOARD_PLANES * 121).div_ceil(64);
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
    /// Conservative upper bound for the dynamic tensors in one padded batch.
    pub batch_memory_budget_bytes: usize,
    pub learning_rate: f32,
    pub l2: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 4,
            batch_size: 64,
            batch_token_budget: 262_144,
            batch_memory_budget_bytes: 4096 * 1024 * 1024,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchMemoryEstimate {
    pub batch_size: usize,
    pub max_boards: usize,
    pub max_actions: usize,
    pub bytes: usize,
}

#[derive(Debug)]
pub struct PolicyValueNetwork {
    model: MultiverseNet<CpuBackend>,
    config: NetworkConfig,
    training_steps: u64,
    spatial_cache: Arc<Mutex<SpatialCache>>,
}

impl Clone for PolicyValueNetwork {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            config: self.config,
            training_steps: self.training_steps,
            spatial_cache: Arc::new(Mutex::new(SpatialCache::default())),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SpatialKey([u64; SPATIAL_KEY_WORDS]);

impl SpatialKey {
    fn from_board(board: &EncodedBoard) -> Self {
        let mut words = [0_u64; SPATIAL_KEY_WORDS];
        for (index, value) in board.planes.iter().enumerate() {
            if *value != 0.0 {
                words[index / 64] |= 1_u64 << (index % 64);
            }
        }
        Self(words)
    }
}

#[derive(Debug, Default)]
struct SpatialCache {
    embeddings: HashMap<SpatialKey, Vec<f32>>,
    hits: u64,
    misses: u64,
}

/// Estimates the padded inference working set for a collection of positions.
#[must_use]
pub fn estimate_inference_batch_memory(
    positions: &[EncodedPosition],
    config: NetworkConfig,
) -> BatchMemoryEstimate {
    let (max_boards, max_actions) = batch_dimensions(positions);
    estimate_batch_memory_from_shape(positions.len(), max_boards, max_actions, config, false)
}

/// Estimates the padded training working set for a collection of positions.
#[must_use]
pub fn estimate_training_batch_memory(
    positions: &[EncodedPosition],
    config: NetworkConfig,
) -> BatchMemoryEstimate {
    let (max_boards, max_actions) = batch_dimensions(positions);
    estimate_batch_memory_from_shape(positions.len(), max_boards, max_actions, config, true)
}

/// Estimates a padded network batch from dimensions without materializing it.
#[must_use]
pub fn estimate_batch_memory_from_shape(
    batch_size: usize,
    max_boards: usize,
    max_actions: usize,
    config: NetworkConfig,
    training: bool,
) -> BatchMemoryEstimate {
    let batch_size = batch_size.max(1);
    let max_boards = max_boards.max(1);
    let max_actions = max_actions.max(1);
    let padded_boards = batch_size.saturating_mul(max_boards);
    let padded_actions = batch_size.saturating_mul(max_actions);

    // Includes input tensors, convolution/residual intermediates, gathered board
    // embeddings, action-head intermediates, and a conservative autodiff margin.
    // The two i64 gather-index tensors are accounted for explicitly.
    let board_channel_copies = if training {
        12_usize.saturating_add(config.residual_blocks.saturating_mul(12))
    } else {
        4_usize.saturating_add(config.residual_blocks.saturating_mul(3))
    };
    let board_values = BOARD_PLANES
        .saturating_mul(121)
        .saturating_add(BOARD_METADATA_FEATURES)
        .saturating_add(config.board_embedding)
        .saturating_add(
            121_usize
                .saturating_mul(config.channels)
                .saturating_mul(board_channel_copies),
        );
    let action_values = ACTION_FEATURES
        .saturating_add(config.board_embedding.saturating_mul(5))
        .saturating_add(config.context_width.saturating_mul(3))
        .saturating_add(
            config
                .action_width
                .saturating_mul(if training { 8 } else { 3 }),
        );
    let board_bytes = padded_boards.saturating_mul(board_values).saturating_mul(4);
    let action_float_bytes = padded_actions
        .saturating_mul(action_values)
        .saturating_mul(4);
    let gather_index_bytes = padded_actions
        .saturating_mul(config.board_embedding)
        .saturating_mul(2)
        .saturating_mul(8);
    let masks_and_targets = padded_actions.saturating_mul(if training { 16 } else { 8 });
    BatchMemoryEstimate {
        batch_size,
        max_boards,
        max_actions,
        bytes: board_bytes
            .saturating_add(action_float_bytes)
            .saturating_add(gather_index_bytes)
            .saturating_add(masks_and_targets),
    }
}

fn batch_dimensions(positions: &[EncodedPosition]) -> (usize, usize) {
    positions
        .iter()
        .fold((1, 1), |(boards, actions), position| {
            (
                boards.max(position.boards.len()),
                actions.max(position.actions.len()),
            )
        })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct SpatialCacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
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
    #[error(
        "one training position exceeds the batch memory budget: boards={boards}, actions={actions}, estimated={estimated_mib} MiB, budget={budget_mib} MiB"
    )]
    BatchMemoryLimit {
        boards: usize,
        actions: usize,
        estimated_mib: usize,
        budget_mib: usize,
    },
    #[cfg(all(feature = "training", feature = "vulkan"))]
    #[error("Vulkan training device initialization failed: {0}")]
    VulkanInitialization(String),
    #[cfg(all(feature = "training", feature = "vulkan"))]
    #[error("Vulkan training failed: {0}")]
    VulkanExecution(String),
}

#[cfg(all(feature = "training", feature = "vulkan"))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDeviceInfo {
    pub discrete_gpu_index: usize,
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
}

#[cfg(all(feature = "training", feature = "vulkan"))]
#[derive(Debug)]
pub struct VulkanTrainingDevice {
    device: burn::backend::wgpu::WgpuDevice,
    info: VulkanDeviceInfo,
}

/// Policy/value evaluator whose forward passes execute on one verified Vulkan GPU.
#[cfg(all(feature = "training", feature = "vulkan"))]
#[derive(Debug)]
pub struct VulkanPolicyValueEvaluator {
    model: MultiverseNet<burn::backend::Vulkan>,
    config: NetworkConfig,
    device: burn::backend::wgpu::WgpuDevice,
    spatial_cache: Mutex<SpatialCache>,
}

#[cfg(all(feature = "training", feature = "vulkan"))]
impl VulkanTrainingDevice {
    /// Initializes one discrete GPU through Vulkan and executes a synchronized
    /// compute probe. This never selects a CPU or software fallback adapter.
    ///
    /// # Errors
    ///
    /// Returns an error if the requested adapter is absent, is not a discrete
    /// Vulkan GPU, or cannot execute and read back the probe operation.
    pub fn initialize(discrete_gpu_index: usize) -> Result<Self, ModelError> {
        use burn::backend::Vulkan;
        use burn::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics, init_setup};
        use burn::tensor::Tensor;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters =
            cubecl_common::future::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let available = adapters
            .iter()
            .map(wgpu::Adapter::get_info)
            .collect::<Vec<_>>();
        let expected = available
            .iter()
            .filter(|info| info.device_type == wgpu::DeviceType::DiscreteGpu)
            .nth(discrete_gpu_index)
            .ok_or_else(|| {
                let description = if available.is_empty() {
                    "no Vulkan adapters were reported".to_owned()
                } else {
                    available
                        .iter()
                        .map(|info| format!("'{}' ({:?})", info.name, info.device_type))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                ModelError::VulkanInitialization(format!(
                    "discrete GPU index {discrete_gpu_index} is unavailable; detected: {description}"
                ))
            })?
            .clone();

        let initialized = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let device = WgpuDevice::DiscreteGpu(discrete_gpu_index);
            let setup = init_setup::<graphics::Vulkan>(&device, RuntimeOptions::default());
            let adapter = setup.adapter.get_info();
            if adapter.backend != wgpu::Backend::Vulkan {
                return Err(ModelError::VulkanInitialization(format!(
                    "adapter '{}' uses {:?}, not Vulkan",
                    adapter.name, adapter.backend
                )));
            }
            if adapter.device_type != wgpu::DeviceType::DiscreteGpu {
                return Err(ModelError::VulkanInitialization(format!(
                    "adapter '{}' is {:?}, not a discrete GPU; software and CPU fallbacks are disabled",
                    adapter.name, adapter.device_type
                )));
            }
            if adapter.vendor != expected.vendor
                || adapter.device != expected.device
                || adapter.name != expected.name
            {
                return Err(ModelError::VulkanInitialization(format!(
                    "Burn selected '{}', but Vulkan discovery selected '{}'; refusing an ambiguous device selection",
                    adapter.name, expected.name
                )));
            }

            let result = Tensor::<Vulkan, 1>::from_floats([1.25, -0.5], &device)
                .mul_scalar(2.0)
                .into_data()
                .to_vec::<f32>()
                .map_err(|error| ModelError::VulkanInitialization(error.to_string()))?;
            if result != [2.5, -1.0] {
                return Err(ModelError::VulkanInitialization(format!(
                    "compute probe returned unexpected values {result:?}"
                )));
            }

            Ok(Self {
                device,
                info: VulkanDeviceInfo {
                    discrete_gpu_index,
                    name: adapter.name,
                    vendor: adapter.vendor,
                    device: adapter.device,
                    device_type: format!("{:?}", adapter.device_type),
                    backend: format!("{:?}", adapter.backend),
                    driver: adapter.driver,
                    driver_info: adapter.driver_info,
                },
            })
        }));

        match initialized {
            Ok(result) => result,
            Err(payload) => Err(ModelError::VulkanInitialization(panic_message(
                payload.as_ref(),
            ))),
        }
    }

    #[must_use]
    pub const fn info(&self) -> &VulkanDeviceInfo {
        &self.info
    }

    /// Copies a CPU-resident checkpoint to this device for batched inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend-neutral model record cannot be transferred.
    pub fn evaluator(
        &self,
        network: &PolicyValueNetwork,
    ) -> Result<VulkanPolicyValueEvaluator, ModelError> {
        let bytes = record_bytes(network.model.clone())?;
        let evaluator = VulkanPolicyValueEvaluator {
            model: model_from_bytes(network.config, bytes, &self.device)?,
            config: network.config,
            device: self.device.clone(),
            spatial_cache: Mutex::new(SpatialCache::default()),
        };
        let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let game = Game::new(Ruleset::Classic);
            let actions = game.legal_actions();
            evaluator.predict(&crate::encode(&game, &actions))
        }))
        .map_err(|payload| ModelError::VulkanInitialization(panic_message(payload.as_ref())))?;
        if probe.policy.is_empty()
            || !probe.value.is_finite()
            || probe.policy.iter().any(|value| !value.is_finite())
        {
            return Err(ModelError::VulkanInitialization(
                "full policy/value inference probe returned invalid values".to_owned(),
            ));
        }
        Ok(evaluator)
    }
}

#[cfg(all(feature = "training", feature = "vulkan"))]
impl PolicyValueEvaluator for VulkanPolicyValueEvaluator {
    fn predict(&self, position: &EncodedPosition) -> Prediction {
        self.predict_batch(std::slice::from_ref(position))
            .pop()
            .expect("single-position Vulkan batch returns one prediction")
    }

    fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
        predict_batch_cached(
            &self.model,
            self.config,
            &self.spatial_cache,
            positions,
            &self.device,
        )
    }
}

#[cfg(all(feature = "training", feature = "vulkan"))]
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload.downcast_ref::<String>().map_or_else(
        || {
            payload.downcast_ref::<&str>().map_or_else(
                || "unknown Vulkan runtime failure".to_owned(),
                |value| (*value).to_owned(),
            )
        },
        Clone::clone,
    )
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
            spatial_cache: Arc::new(Mutex::new(SpatialCache::default())),
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

    #[doc(hidden)]
    #[must_use]
    pub fn spatial_cache_stats(&self) -> SpatialCacheStats {
        let cache = self.spatial_cache.lock().expect("spatial cache lock");
        SpatialCacheStats {
            entries: cache.embeddings.len(),
            hits: cache.hits,
            misses: cache.misses,
        }
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
        let device = NdArrayDevice::default();
        predict_batch_cached(
            &self.model,
            self.config,
            &self.spatial_cache,
            positions,
            &device,
        )
    }

    #[cfg(feature = "training")]
    /// Fits examples with the CPU autodiff backend.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be transferred or a position exceeds
    /// the configured batch memory budget.
    ///
    /// # Panics
    ///
    /// Panics only if another thread poisons the internal spatial-cache lock.
    pub fn train(
        &mut self,
        examples: &[TrainingExample],
        config: TrainConfig,
        rng: &mut impl Rng,
    ) -> Result<TrainMetrics, ModelError> {
        let seed: u64 = rng.random();
        let bytes = record_bytes(self.model.clone())?;
        let (model, metrics, updates) = train_backend::<Autodiff<CpuBackend>>(
            &bytes,
            self.config,
            examples,
            config,
            seed,
            &NdArrayDevice::default(),
        )?;
        let bytes = record_bytes(model)?;
        self.model = model_from_bytes(self.config, bytes, &NdArrayDevice::default())?;
        self.training_steps += updates;
        self.spatial_cache
            .lock()
            .expect("spatial cache lock")
            .embeddings
            .clear();
        Ok(metrics)
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
            spatial_cache: Arc::new(Mutex::new(SpatialCache::default())),
        })
    }

    #[cfg(all(feature = "training", feature = "vulkan"))]
    /// Fits examples on an explicitly initialized Vulkan device.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend-neutral model record cannot be transferred.
    /// Vulkan adapter and driver initialization failures are surfaced by Burn.
    ///
    /// # Panics
    ///
    /// Panics only if another thread poisons the internal spatial-cache lock.
    pub fn train_vulkan(
        &mut self,
        device: &VulkanTrainingDevice,
        examples: &[TrainingExample],
        config: TrainConfig,
        rng: &mut impl Rng,
    ) -> Result<TrainMetrics, ModelError> {
        use burn::backend::Vulkan;

        let seed: u64 = rng.random();
        let bytes = record_bytes(self.model.clone())?;
        let trained = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            train_backend::<Autodiff<Vulkan>>(
                &bytes,
                self.config,
                examples,
                config,
                seed,
                &device.device,
            )
        }))
        .map_err(|payload| ModelError::VulkanExecution(panic_message(payload.as_ref())))?;
        let (model, metrics, updates) = trained?;
        let bytes = record_bytes(model)?;
        self.model = model_from_bytes(self.config, bytes, &NdArrayDevice::default())?;
        self.training_steps += updates;
        self.spatial_cache
            .lock()
            .expect("spatial cache lock")
            .embeddings
            .clear();
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

#[allow(clippy::too_many_lines)]
fn predict_batch_cached<B: Backend>(
    model: &MultiverseNet<B>,
    config: NetworkConfig,
    spatial_cache: &Mutex<SpatialCache>,
    positions: &[EncodedPosition],
    device: &B::Device,
) -> Vec<Prediction> {
    if positions.is_empty() {
        return Vec::new();
    }
    let batch = BatchTensors::from_positions(positions, config.board_embedding, device);
    let board_count = positions
        .iter()
        .map(|position| position.boards.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let keys = positions
        .iter()
        .flat_map(|position| position.boards.iter().map(SpatialKey::from_board))
        .collect::<Vec<_>>();
    let mut resolved = HashMap::with_capacity(keys.len());
    let mut missing_keys = Vec::new();
    let mut missing_boards = Vec::new();
    {
        let mut cache = spatial_cache.lock().expect("spatial cache lock");
        if cache.embeddings.len() >= SPATIAL_CACHE_LIMIT {
            cache.embeddings.clear();
        }
        let mut seen_missing = HashSet::new();
        for board in positions.iter().flat_map(|position| &position.boards) {
            let key = SpatialKey::from_board(board);
            if let Some(embedding) = cache.embeddings.get(&key).cloned() {
                resolved.insert(key, embedding);
                cache.hits += 1;
            } else if seen_missing.insert(key) {
                missing_keys.push(key);
                missing_boards.push(board);
                cache.misses += 1;
            }
        }
    }
    if !missing_boards.is_empty() {
        let planes = missing_boards
            .iter()
            .flat_map(|board| board.planes.iter().copied())
            .collect::<Vec<_>>();
        let embeddings = model
            .encode_spatial(Tensor::from_data(
                TensorData::new(planes, [missing_boards.len(), BOARD_PLANES, 11, 11]),
                device,
            ))
            .to_data()
            .to_vec::<f32>()
            .expect("spatial tensor uses f32");
        for (index, key) in missing_keys.into_iter().enumerate() {
            let start = index * config.board_embedding;
            resolved.insert(
                key,
                embeddings[start..start + config.board_embedding].to_vec(),
            );
        }
        let mut cache = spatial_cache.lock().expect("spatial cache lock");
        if cache.embeddings.len() + resolved.len() > SPATIAL_CACHE_LIMIT {
            cache.embeddings.clear();
        }
        let remaining = SPATIAL_CACHE_LIMIT.saturating_sub(cache.embeddings.len());
        cache.embeddings.extend(
            resolved
                .iter()
                .take(remaining)
                .map(|(key, value)| (*key, value.clone())),
        );
    }
    let mut spatial = vec![0.0; positions.len() * board_count * config.board_embedding];
    let mut key_index = 0;
    for (batch_index, position) in positions.iter().enumerate() {
        for board_index in 0..position.boards.len() {
            let embedding = resolved
                .get(&keys[key_index])
                .expect("every board embedding is resolved");
            let start = (batch_index * board_count + board_index) * config.board_embedding;
            spatial[start..start + config.board_embedding].copy_from_slice(embedding);
            key_index += 1;
        }
    }
    let spatial = Tensor::from_data(
        TensorData::new(
            spatial,
            [positions.len(), board_count, config.board_embedding],
        ),
        device,
    );
    let output = model.forward_with_spatial(batch, spatial);
    let action_count = output.logits.dims()[1];
    let policies = softmax(output.logits, 1)
        .to_data()
        .to_vec::<f32>()
        .expect("policy tensor uses f32");
    let values = output
        .value
        .to_data()
        .to_vec::<f32>()
        .expect("value tensor uses f32");
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
        order.sort_by_key(|&index| shape_bucket(&examples[index].position));
        let mut cursor = 0;
        while cursor < order.len() {
            let end = choose_batch_end(examples, &order, cursor, config, network_config)?;
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
    network_config: NetworkConfig,
) -> Result<usize, ModelError> {
    let maximum = (start + config.batch_size.max(1)).min(order.len());
    let mut end = start;
    let mut max_boards = 1_usize;
    let mut max_actions = 1_usize;
    while end < maximum {
        let position = &examples[order[end]].position;
        let candidate_size = end - start + 1;
        max_boards = max_boards.max(position.boards.len());
        max_actions = max_actions.max(position.actions.len());
        let padded_tokens = candidate_size
            .saturating_mul(max_boards.saturating_mul(121).saturating_add(max_actions));
        let memory = estimate_batch_memory_from_shape(
            candidate_size,
            max_boards,
            max_actions,
            network_config,
            true,
        );
        let exceeds_tokens = padded_tokens > config.batch_token_budget.max(1);
        let exceeds_memory = memory.bytes > config.batch_memory_budget_bytes.max(1);
        if end == start && exceeds_memory {
            return Err(ModelError::BatchMemoryLimit {
                boards: max_boards,
                actions: max_actions,
                estimated_mib: bytes_to_mib_ceil(memory.bytes),
                budget_mib: bytes_to_mib_ceil(config.batch_memory_budget_bytes.max(1)),
            });
        }
        if end > start && (exceeds_tokens || exceeds_memory) {
            break;
        }
        end += 1;
    }
    Ok(end.max(start + 1))
}

#[cfg(feature = "training")]
fn shape_bucket(position: &EncodedPosition) -> (usize, usize) {
    (
        position.boards.len().max(1).next_power_of_two(),
        position.actions.len().max(1).next_power_of_two(),
    )
}

const fn bytes_to_mib_ceil(bytes: usize) -> usize {
    bytes.saturating_add(1024 * 1024 - 1) / (1024 * 1024)
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
    fn repeated_positions_reuse_cached_spatial_embeddings() {
        let game = Game::new(Ruleset::Multiverse);
        let position = encode(&game, &game.legal_actions());
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let network = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);

        let _ = network.predict(&position);
        let first = network.spatial_cache_stats();
        let _ = network.predict_batch(&[position.clone(), position]);
        let second = network.spatial_cache_stats();

        assert_eq!(first.entries, 1);
        assert_eq!(first.misses, 1);
        assert_eq!(second.entries, 1);
        assert_eq!(second.misses, first.misses);
        assert!(second.hits >= first.hits + 2);
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
        let metrics = network
            .train(
                &[example],
                TrainConfig {
                    epochs: 1,
                    batch_size: 1,
                    ..TrainConfig::default()
                },
                &mut rng,
            )
            .unwrap();
        assert_eq!(metrics.examples, 1);
        assert!(metrics.policy_loss.is_finite());
        assert!(metrics.value_loss.is_finite());
        assert_eq!(network.training_steps(), 1);
    }

    #[test]
    fn padded_memory_budget_splits_training_batches_before_allocation() {
        let game = Game::new(Ruleset::Classic);
        let position = encode(&game, &game.legal_actions());
        let policy = vec![1.0; position.actions.len()];
        let examples = vec![
            TrainingExample {
                position: position.clone(),
                policy: policy.clone(),
                value: 0.0,
            },
            TrainingExample {
                position,
                policy,
                value: 0.0,
            },
        ];
        let network_config = NetworkConfig::tiny();
        let single = estimate_training_batch_memory(
            std::slice::from_ref(&examples[0].position),
            network_config,
        );
        let pair = estimate_training_batch_memory(
            &[examples[0].position.clone(), examples[1].position.clone()],
            network_config,
        );
        assert!(pair.bytes > single.bytes);
        let end = choose_batch_end(
            &examples,
            &[0, 1],
            0,
            TrainConfig {
                batch_size: 2,
                batch_token_budget: usize::MAX,
                batch_memory_budget_bytes: pair.bytes - 1,
                ..TrainConfig::default()
            },
            network_config,
        )
        .unwrap();
        assert_eq!(end, 1);

        let error = choose_batch_end(
            &examples,
            &[0, 1],
            0,
            TrainConfig {
                batch_memory_budget_bytes: single.bytes - 1,
                ..TrainConfig::default()
            },
            network_config,
        )
        .unwrap_err();
        assert!(matches!(error, ModelError::BatchMemoryLimit { .. }));
    }
}
