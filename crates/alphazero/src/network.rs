use std::fs;
use std::path::Path;

use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::{ACTION_FEATURES, EncodedPosition, STATE_FEATURES};

const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct NetworkConfig {
    pub hidden: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self { hidden: 48 }
    }
}

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

#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub l2: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 4,
            batch_size: 64,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyValueNetwork {
    format_version: u32,
    config: NetworkConfig,
    weights: Vec<f32>,
    training_steps: u64,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("model JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported model format version {0}")]
    Version(u32),
    #[error("model has {actual} weights; expected {expected}")]
    Shape { expected: usize, actual: usize },
}

struct Layout {
    w1: usize,
    b1: usize,
    w2: usize,
    b2: usize,
    wv: usize,
    bv: usize,
    wa: usize,
    ba: usize,
    len: usize,
}

struct Forward {
    h0: Vec<f32>,
    residual: Vec<f32>,
    hidden: Vec<f32>,
    action_embeddings: Vec<Vec<f32>>,
    prediction: Prediction,
}

impl PolicyValueNetwork {
    #[must_use]
    pub fn random(config: NetworkConfig, rng: &mut impl Rng) -> Self {
        let layout = Layout::new(config.hidden);
        let mut weights = vec![0.0; layout.len];
        fill_xavier(
            &mut weights[layout.w1..layout.b1],
            STATE_FEATURES,
            config.hidden,
            rng,
        );
        fill_xavier(
            &mut weights[layout.w2..layout.b2],
            config.hidden,
            config.hidden,
            rng,
        );
        fill_xavier(&mut weights[layout.wv..layout.bv], config.hidden, 1, rng);
        fill_xavier(
            &mut weights[layout.wa..layout.ba],
            ACTION_FEATURES,
            config.hidden,
            rng,
        );
        Self {
            format_version: FORMAT_VERSION,
            config,
            weights,
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
    pub fn predict(&self, position: &EncodedPosition) -> Prediction {
        self.forward(position).prediction
    }

    /// Trains the policy and value heads jointly with mini-batch Adam.
    pub fn train(
        &mut self,
        examples: &[TrainingExample],
        config: TrainConfig,
        rng: &mut impl Rng,
    ) -> TrainMetrics {
        if examples.is_empty() || config.epochs == 0 {
            return TrainMetrics::default();
        }
        let mut first_moment = vec![0.0; self.weights.len()];
        let mut second_moment = vec![0.0; self.weights.len()];
        let mut order = (0..examples.len()).collect::<Vec<_>>();
        let mut metrics = TrainMetrics::default();
        let mut updates = 0_i32;
        for _ in 0..config.epochs {
            order.shuffle(rng);
            for batch in order.chunks(config.batch_size.max(1)) {
                let mut gradient = vec![0.0; self.weights.len()];
                let mut policy_loss = 0.0;
                let mut value_loss = 0.0;
                for &index in batch {
                    let losses = self.accumulate_gradient(&examples[index], &mut gradient);
                    policy_loss += losses.0;
                    value_loss += losses.1;
                }
                let inverse_batch = 1.0 / batch.len() as f32;
                for (index, value) in gradient.iter_mut().enumerate() {
                    *value = *value * inverse_batch + config.l2 * self.weights[index];
                }
                updates += 1;
                adam_step(
                    &mut self.weights,
                    &gradient,
                    &mut first_moment,
                    &mut second_moment,
                    config.learning_rate,
                    updates,
                );
                self.training_steps += 1;
                metrics.policy_loss += policy_loss;
                metrics.value_loss += value_loss;
                metrics.examples += batch.len();
            }
        }
        if metrics.examples > 0 {
            let denominator = metrics.examples as f32;
            metrics.policy_loss /= denominator;
            metrics.value_loss /= denominator;
        }
        metrics
    }

    /// Saves an inspectable, versioned JSON checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination cannot be created or serialized.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec(self)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Loads and shape-checks a checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error for inaccessible, malformed, unsupported, or
    /// incorrectly shaped checkpoints.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let model: Self = serde_json::from_slice(&fs::read(path)?)?;
        if model.format_version != FORMAT_VERSION {
            return Err(ModelError::Version(model.format_version));
        }
        let expected = Layout::new(model.config.hidden).len;
        if model.weights.len() != expected {
            return Err(ModelError::Shape {
                expected,
                actual: model.weights.len(),
            });
        }
        Ok(model)
    }

    fn forward(&self, position: &EncodedPosition) -> Forward {
        assert_eq!(position.state.len(), STATE_FEATURES);
        let hidden_size = self.config.hidden;
        let layout = Layout::new(hidden_size);
        let mut h0 = vec![0.0; hidden_size];
        for (row, output) in h0.iter_mut().enumerate() {
            let weights = &self.weights
                [layout.w1 + row * STATE_FEATURES..layout.w1 + (row + 1) * STATE_FEATURES];
            *output = (dot(weights, &position.state) + self.weights[layout.b1 + row]).tanh();
        }
        let mut residual = vec![0.0; hidden_size];
        for (row, output) in residual.iter_mut().enumerate() {
            let weights =
                &self.weights[layout.w2 + row * hidden_size..layout.w2 + (row + 1) * hidden_size];
            *output = (dot(weights, &h0) + self.weights[layout.b2 + row]).tanh();
        }
        let hidden = h0
            .iter()
            .zip(&residual)
            .map(|(base, skip)| (base + skip).tanh())
            .collect::<Vec<_>>();
        let value =
            (dot(&self.weights[layout.wv..layout.bv], &hidden) + self.weights[layout.bv]).tanh();
        let scale = (hidden_size as f32).sqrt().recip();
        let mut action_embeddings = Vec::with_capacity(position.actions.len());
        let mut logits = Vec::with_capacity(position.actions.len());
        for action in &position.actions {
            assert_eq!(action.len(), ACTION_FEATURES);
            let mut embedding = vec![0.0; hidden_size];
            for (row, output) in embedding.iter_mut().enumerate() {
                let weights = &self.weights
                    [layout.wa + row * ACTION_FEATURES..layout.wa + (row + 1) * ACTION_FEATURES];
                *output = dot(weights, action) + self.weights[layout.ba + row];
            }
            logits.push(dot(&hidden, &embedding) * scale);
            action_embeddings.push(embedding);
        }
        Forward {
            h0,
            residual,
            hidden,
            action_embeddings,
            prediction: Prediction {
                policy: softmax(&logits),
                value,
            },
        }
    }

    fn accumulate_gradient(&self, example: &TrainingExample, gradient: &mut [f32]) -> (f32, f32) {
        let forward = self.forward(&example.position);
        assert_eq!(forward.prediction.policy.len(), example.policy.len());
        let hidden_size = self.config.hidden;
        let layout = Layout::new(hidden_size);
        let mut hidden_gradient = vec![0.0; hidden_size];
        let value_error = forward.prediction.value - example.value;
        let value_delta = 2.0 * value_error * (1.0 - forward.prediction.value.powi(2));
        for (index, hidden) in forward.hidden.iter().copied().enumerate() {
            gradient[layout.wv + index] += value_delta * hidden;
            hidden_gradient[index] += value_delta * self.weights[layout.wv + index];
        }
        gradient[layout.bv] += value_delta;

        let scale = (hidden_size as f32).sqrt().recip();
        let mut policy_loss = 0.0;
        for (action_index, (&predicted, &target)) in forward
            .prediction
            .policy
            .iter()
            .zip(&example.policy)
            .enumerate()
        {
            if target > 0.0 {
                policy_loss -= target * predicted.max(1.0e-12).ln();
            }
            let delta = predicted - target;
            let action = &example.position.actions[action_index];
            for row in 0..hidden_size {
                let embedding = forward.action_embeddings[action_index][row];
                hidden_gradient[row] += delta * embedding * scale;
                let embedding_gradient = delta * forward.hidden[row] * scale;
                gradient[layout.ba + row] += embedding_gradient;
                for (column, action_feature) in action.iter().copied().enumerate() {
                    gradient[layout.wa + row * ACTION_FEATURES + column] +=
                        embedding_gradient * action_feature;
                }
            }
        }

        let mut base_gradient = vec![0.0; hidden_size];
        let mut residual_delta = vec![0.0; hidden_size];
        for row in 0..hidden_size {
            let sum_delta = hidden_gradient[row] * (1.0 - forward.hidden[row].powi(2));
            base_gradient[row] += sum_delta;
            residual_delta[row] = sum_delta * (1.0 - forward.residual[row].powi(2));
            gradient[layout.b2 + row] += residual_delta[row];
            for column in 0..hidden_size {
                gradient[layout.w2 + row * hidden_size + column] +=
                    residual_delta[row] * forward.h0[column];
                base_gradient[column] +=
                    residual_delta[row] * self.weights[layout.w2 + row * hidden_size + column];
            }
        }
        for row in 0..hidden_size {
            let delta = base_gradient[row] * (1.0 - forward.h0[row].powi(2));
            gradient[layout.b1 + row] += delta;
            for (column, state_feature) in example.position.state.iter().copied().enumerate() {
                gradient[layout.w1 + row * STATE_FEATURES + column] += delta * state_feature;
            }
        }
        (policy_loss, value_error * value_error)
    }
}

impl Layout {
    const fn new(hidden: usize) -> Self {
        let w1 = 0;
        let b1 = w1 + hidden * STATE_FEATURES;
        let w2 = b1 + hidden;
        let b2 = w2 + hidden * hidden;
        let wv = b2 + hidden;
        let bv = wv + hidden;
        let wa = bv + 1;
        let ba = wa + hidden * ACTION_FEATURES;
        let len = ba + hidden;
        Self {
            w1,
            b1,
            w2,
            b2,
            wv,
            bv,
            wa,
            ba,
            len,
        }
    }
}

fn fill_xavier(values: &mut [f32], inputs: usize, outputs: usize, rng: &mut impl Rng) {
    let limit = (6.0 / (inputs + outputs) as f32).sqrt();
    for value in values {
        *value = rng.random_range(-limit..=limit);
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|logit| (logit - maximum).exp())
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    for value in &mut values {
        *value /= total;
    }
    values
}

fn adam_step(
    weights: &mut [f32],
    gradient: &[f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    learning_rate: f32,
    step: i32,
) {
    let beta_one = 0.9_f32;
    let beta_two = 0.999_f32;
    let first_correction = 1.0 - beta_one.powi(step);
    let second_correction = 1.0 - beta_two.powi(step);
    for index in 0..weights.len() {
        first_moment[index] = beta_one * first_moment[index] + (1.0 - beta_one) * gradient[index];
        second_moment[index] =
            beta_two * second_moment[index] + (1.0 - beta_two) * gradient[index].powi(2);
        let first = first_moment[index] / first_correction;
        let second = second_moment[index] / second_correction;
        weights[index] -= learning_rate * first / (second.sqrt() + 1.0e-8);
    }
}

#[cfg(test)]
mod tests {
    use huginn_core::{Game, Ruleset};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::encoding::encode;

    use super::*;

    #[test]
    fn policy_is_normalized_and_value_is_bounded() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let model = PolicyValueNetwork::random(NetworkConfig { hidden: 8 }, &mut rng);
        let game = Game::new(Ruleset::Classic);
        let actions = game.legal_actions();
        let prediction = model.predict(&encode(&game, &actions));
        assert!((prediction.policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
        assert!((-1.0..=1.0).contains(&prediction.value));
        assert_eq!(prediction.policy.len(), actions.len());
    }

    #[test]
    fn training_reduces_a_single_example_loss_and_checkpoint_round_trips() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut model = PolicyValueNetwork::random(NetworkConfig { hidden: 8 }, &mut rng);
        let game = Game::new(Ruleset::Classic);
        let actions = game.legal_actions();
        let position = encode(&game, &actions);
        let mut policy = vec![0.0; actions.len()];
        policy[0] = 1.0;
        let example = TrainingExample {
            position: position.clone(),
            policy,
            value: 1.0,
        };
        let before = model.predict(&position);
        model.train(
            &[example],
            TrainConfig {
                epochs: 20,
                batch_size: 1,
                learning_rate: 0.01,
                l2: 0.0,
            },
            &mut rng,
        );
        let after = model.predict(&position);
        assert!(after.policy[0] > before.policy[0]);
        assert!(after.value > before.value);

        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("model.json");
        model.save(&path).expect("save");
        let loaded = PolicyValueNetwork::load(path).expect("load");
        assert_eq!(loaded.predict(&position), after);
    }
}
