use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use huginn_core::{Game, GameOutcome, PlayerAction, Ruleset, Side};
use rand::Rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::encode;
use crate::network::{PolicyValueEvaluator, PolicyValueNetwork, TrainingExample};
use crate::search::{Mcts, SearchConfig};

#[derive(Clone, Copy, Debug)]
pub struct SelfPlayConfig {
    pub search: SearchConfig,
    pub max_actions: usize,
    pub exploration_actions: usize,
    pub temperature: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            search: SearchConfig {
                simulations: 64,
                ..SearchConfig::default()
            },
            max_actions: 600,
            exploration_actions: 30,
            temperature: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SelfPlayGame {
    pub ruleset: Ruleset,
    pub steps: Vec<ReplayStep>,
    pub outcome: Option<GameOutcome>,
    pub truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayStep {
    pub action: PlayerAction,
    pub policy: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ReplayGame {
    ruleset: Ruleset,
    steps: Vec<ReplayStep>,
    outcome: Option<GameOutcome>,
    truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayBuffer {
    format_version: u32,
    games: Vec<ReplayGame>,
}

#[derive(Debug)]
pub struct SampledReplay {
    pub examples: Vec<TrainingExample>,
    pub estimated_bytes: usize,
    pub skipped_for_memory: usize,
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self {
            format_version: 2,
            games: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaReport {
    pub candidate_wins: usize,
    pub incumbent_wins: usize,
    pub draws: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaProgress {
    pub game: usize,
    pub games: usize,
    pub ruleset: Ruleset,
    pub candidate_side: Side,
    pub actions: usize,
    pub completed: bool,
    pub outcome: Option<GameOutcome>,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaGameConfig {
    pub ruleset: Ruleset,
    pub candidate_side: Side,
    pub search: SearchConfig,
    pub max_actions: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaGameResult {
    pub outcome: Option<GameOutcome>,
    pub actions: usize,
}

impl ArenaReport {
    #[must_use]
    pub fn candidate_score(self) -> f32 {
        let games = self.candidate_wins + self.incumbent_wins + self.draws;
        if games == 0 {
            0.0
        } else {
            (self.candidate_wins as f32 + self.draws as f32 * 0.5) / games as f32
        }
    }
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("replay I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("replay data is invalid: {0}")]
    Codec(String),
    #[error("unsupported replay format version {0}")]
    Version(u32),
    #[error("replay trajectory is invalid: {0}")]
    Trajectory(String),
    #[error(
        "a replay position cannot fit the reconstruction memory budget: estimated={estimated_mib} MiB, budget={budget_mib} MiB"
    )]
    MemoryBudget {
        estimated_mib: usize,
        budget_mib: usize,
    },
}

#[must_use]
pub fn play_self_play_game<E: PolicyValueEvaluator + ?Sized>(
    model: &E,
    ruleset: Ruleset,
    config: SelfPlayConfig,
    rng: &mut impl Rng,
) -> SelfPlayGame {
    let mut game = Game::new(ruleset);
    let mut steps = Vec::new();
    for action_index in 0..config.max_actions {
        if game.outcome().is_some() {
            break;
        }
        let search = Mcts::new(model, config.search).search(&game, true, rng);
        if search.actions.is_empty() {
            break;
        }
        let temperature = if action_index < config.exploration_actions {
            config.temperature
        } else {
            0.0
        };
        let Some(action) = search.sample_action(temperature, rng) else {
            break;
        };
        steps.push(ReplayStep {
            action,
            policy: search.policy,
        });
        if game.apply_action(action).is_err() {
            break;
        }
    }
    let outcome = game.outcome();
    SelfPlayGame {
        ruleset,
        steps,
        outcome,
        truncated: outcome.is_none(),
    }
}

#[must_use]
pub fn run_arena(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
    ruleset: Ruleset,
    games: usize,
    search: SearchConfig,
    max_actions: usize,
    rng: &mut impl Rng,
) -> ArenaReport {
    run_arena_schedule(
        candidate,
        incumbent,
        &vec![ruleset; games],
        search,
        max_actions,
        rng,
    )
}

#[must_use]
pub fn run_arena_schedule(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
    rulesets: &[Ruleset],
    search: SearchConfig,
    max_actions: usize,
    rng: &mut impl Rng,
) -> ArenaReport {
    run_arena_schedule_with_progress(
        candidate,
        incumbent,
        rulesets,
        search,
        max_actions,
        rng,
        |_| {},
    )
}

#[must_use]
pub fn run_arena_schedule_with_progress(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
    rulesets: &[Ruleset],
    search: SearchConfig,
    max_actions: usize,
    rng: &mut impl Rng,
    mut progress: impl FnMut(ArenaProgress),
) -> ArenaReport {
    let mut report = ArenaReport::default();
    for (game_index, &ruleset) in rulesets.iter().enumerate() {
        let candidate_side = if game_index % 2 == 0 {
            Side::Attacker
        } else {
            Side::Defender
        };
        let mut actions = 0;
        progress(ArenaProgress {
            game: game_index + 1,
            games: rulesets.len(),
            ruleset,
            candidate_side,
            actions,
            completed: false,
            outcome: None,
        });
        let result = play_arena_game(
            candidate,
            incumbent,
            ArenaGameConfig {
                ruleset,
                candidate_side,
                search,
                max_actions,
            },
            rng,
            |current_actions, outcome| {
                actions = current_actions;
                progress(ArenaProgress {
                    game: game_index + 1,
                    games: rulesets.len(),
                    ruleset,
                    candidate_side,
                    actions,
                    completed: false,
                    outcome,
                });
            },
        );
        match result.outcome {
            Some(outcome) if outcome.winner == candidate_side => report.candidate_wins += 1,
            Some(_) => report.incumbent_wins += 1,
            None => report.draws += 1,
        }
        progress(ArenaProgress {
            game: game_index + 1,
            games: rulesets.len(),
            ruleset,
            candidate_side,
            actions: result.actions,
            completed: true,
            outcome: result.outcome,
        });
    }
    report
}

#[must_use]
pub fn play_arena_game<E: PolicyValueEvaluator + ?Sized>(
    candidate: &E,
    incumbent: &E,
    config: ArenaGameConfig,
    rng: &mut impl Rng,
    mut progress: impl FnMut(usize, Option<GameOutcome>),
) -> ArenaGameResult {
    let mut game = Game::new(config.ruleset);
    let mut actions = 0;
    for _ in 0..config.max_actions {
        if game.outcome().is_some() {
            break;
        }
        let model = if game.turn() == config.candidate_side {
            candidate
        } else {
            incumbent
        };
        let result = Mcts::new(model, config.search).search(&game, false, rng);
        let Some(action) = result.best_action() else {
            break;
        };
        if game.apply_action(action).is_err() {
            break;
        }
        actions += 1;
        progress(actions, game.outcome());
    }
    ArenaGameResult {
        outcome: game.outcome(),
        actions,
    }
}

impl ReplayBuffer {
    #[must_use]
    pub fn len(&self) -> usize {
        self.games.iter().map(|game| game.steps.len()).sum()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    /// Reconstructs encoded training positions from compact authoritative trajectories.
    ///
    /// # Errors
    ///
    /// Returns an error if a stored policy/action no longer matches authoritative rules.
    pub fn examples(&self) -> Result<Vec<TrainingExample>, ReplayError> {
        let mut rng = rand::rng();
        Ok(self
            .sample_examples(self.len(), usize::MAX, &mut rng)?
            .examples)
    }

    /// Reconstructs a random, host-memory-bounded sample of replay positions.
    /// Every trajectory is still validated in full.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid trajectories or when one selected position
    /// cannot fit the memory budget by itself.
    pub fn sample_examples(
        &self,
        max_examples: usize,
        max_bytes: usize,
        rng: &mut impl Rng,
    ) -> Result<SampledReplay, ReplayError> {
        let mut selected = (0..self.len()).collect::<Vec<_>>();
        selected.shuffle(rng);
        selected.truncate(max_examples.min(selected.len()));
        let mut selected = selected.into_iter().collect::<HashSet<_>>();
        let mut game_order = (0..self.games.len()).collect::<Vec<_>>();
        if max_examples < self.len() || max_bytes != usize::MAX {
            game_order.shuffle(rng);
        }
        let mut game_offsets = Vec::with_capacity(self.games.len());
        let mut offset = 0_usize;
        for replay in &self.games {
            game_offsets.push(offset);
            offset = offset.saturating_add(replay.steps.len());
        }
        let mut examples = Vec::with_capacity(max_examples.min(self.len()));
        let mut estimated_bytes = 0_usize;
        let mut skipped_for_memory = 0_usize;
        let mut smallest_rejected = usize::MAX;
        for game_index in game_order {
            let replay = &self.games[game_index];
            let mut game = Game::new(replay.ruleset);
            for (index, step) in replay.steps.iter().enumerate() {
                let global_index = game_offsets[game_index].saturating_add(index);
                let actions = game.legal_actions();
                if actions.len() != step.policy.len() {
                    return Err(ReplayError::Trajectory(format!(
                        "step {index} has {} policy entries for {} legal actions",
                        step.policy.len(),
                        actions.len()
                    )));
                }
                if !actions.contains(&step.action) {
                    return Err(ReplayError::Trajectory(format!(
                        "step {index} selects an action that is no longer legal"
                    )));
                }
                if selected.remove(&global_index) {
                    let estimate = estimate_example_bytes(&game, actions.len());
                    if estimated_bytes.saturating_add(estimate) <= max_bytes {
                        let side = game.turn();
                        examples.push(TrainingExample {
                            position: encode(&game, &actions),
                            policy: step.policy.clone(),
                            value: replay.outcome.map_or(0.0, |outcome| {
                                if outcome.winner == side { 1.0 } else { -1.0 }
                            }),
                        });
                        estimated_bytes = estimated_bytes.saturating_add(estimate);
                    } else {
                        skipped_for_memory += 1;
                        smallest_rejected = smallest_rejected.min(estimate);
                    }
                }
                game.apply_action(step.action).map_err(|error| {
                    ReplayError::Trajectory(format!("step {index} cannot be applied: {error}"))
                })?;
            }
        }
        if examples.is_empty() && skipped_for_memory > 0 {
            return Err(ReplayError::MemoryBudget {
                estimated_mib: bytes_to_mib_ceil(smallest_rejected),
                budget_mib: bytes_to_mib_ceil(max_bytes),
            });
        }
        Ok(SampledReplay {
            examples,
            estimated_bytes,
            skipped_for_memory,
        })
    }

    pub fn extend_game(&mut self, game: SelfPlayGame, capacity: usize) {
        self.format_version = 2;
        self.games.push(ReplayGame {
            ruleset: game.ruleset,
            steps: game.steps,
            outcome: game.outcome,
            truncated: game.truncated,
        });
        while self.games.len() > 1 && self.len() > capacity {
            self.games.remove(0);
        }
    }

    /// Saves the replay window atomically as `MessagePack` data compressed with zstd.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination cannot be created or serialized.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ReplayError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("zst.tmp");
        let writer = BufWriter::new(File::create(&temporary)?);
        let mut encoder = zstd::stream::write::Encoder::new(writer, 3)?;
        rmp_serde::encode::write_named(&mut encoder, self)
            .map_err(|error| ReplayError::Codec(error.to_string()))?;
        let mut writer = encoder.finish()?;
        writer.flush()?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Loads a version-2 compressed replay window.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be read or decoded.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let reader = BufReader::new(File::open(path)?);
        let decoder = zstd::stream::read::Decoder::new(reader)?;
        let replay: Self =
            rmp_serde::from_read(decoder).map_err(|error| ReplayError::Codec(error.to_string()))?;
        if replay.format_version != 2 {
            return Err(ReplayError::Version(replay.format_version));
        }
        Ok(replay)
    }
}

fn estimate_example_bytes(game: &Game, actions: usize) -> usize {
    let boards = if game.ruleset() == Ruleset::Classic {
        1
    } else {
        game.timelines()
            .map(|timeline| timeline.boards.len())
            .sum::<usize>()
            .max(1)
    };
    // Deliberately includes allocator/Vec overhead above the serialized f32 data.
    boards
        .saturating_mul(4 * 1024)
        .saturating_add(actions.saturating_mul(192))
        .saturating_add(1024)
}

const fn bytes_to_mib_ceil(bytes: usize) -> usize {
    bytes.saturating_add(1024 * 1024 - 1) / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::network::NetworkConfig;

    use super::*;

    #[test]
    fn short_self_play_game_produces_aligned_examples_and_draw_targets() {
        let mut rng = ChaCha8Rng::seed_from_u64(23);
        let model = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let game = play_self_play_game(
            &model,
            Ruleset::Classic,
            SelfPlayConfig {
                search: SearchConfig {
                    simulations: 2,
                    ..SearchConfig::default()
                },
                max_actions: 2,
                exploration_actions: 2,
                temperature: 1.0,
            },
            &mut rng,
        );
        assert_eq!(game.steps.len(), 2);
        assert!(game.truncated);
        let mut replay = ReplayBuffer::default();
        replay.extend_game(game, 10);
        for example in replay.examples().unwrap() {
            assert_eq!(example.position.actions.len(), example.policy.len());
            assert!((example.policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
            assert!(example.value.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn replay_capacity_discards_oldest_complete_games_and_round_trips() {
        let make_game = || {
            let game = Game::new(Ruleset::Classic);
            let actions = game.legal_actions();
            SelfPlayGame {
                ruleset: Ruleset::Classic,
                steps: vec![ReplayStep {
                    action: actions[0],
                    policy: vec![1.0 / actions.len() as f32; actions.len()],
                }],
                outcome: None,
                truncated: true,
            }
        };
        let mut replay = ReplayBuffer::default();
        replay.extend_game(make_game(), 2);
        replay.extend_game(make_game(), 2);
        replay.extend_game(make_game(), 2);
        assert_eq!(replay.len(), 2);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("replay-v2.bin.zst");
        replay.save(&path).unwrap();
        let loaded = ReplayBuffer::load(path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.examples().unwrap().len(), 2);
    }

    #[test]
    fn replay_sampling_bounds_expanded_positions_and_host_memory() {
        let game = Game::new(Ruleset::Classic);
        let actions = game.legal_actions();
        let replay_game = SelfPlayGame {
            ruleset: Ruleset::Classic,
            steps: vec![ReplayStep {
                action: actions[0],
                policy: vec![1.0 / actions.len() as f32; actions.len()],
            }],
            outcome: None,
            truncated: true,
        };
        let mut replay = ReplayBuffer::default();
        replay.extend_game(replay_game.clone(), 10);
        replay.extend_game(replay_game, 10);
        let mut rng = ChaCha8Rng::seed_from_u64(41);

        let sample = replay.sample_examples(1, 1024 * 1024, &mut rng).unwrap();
        assert_eq!(sample.examples.len(), 1);
        assert!(sample.estimated_bytes <= 1024 * 1024);

        let error = replay.sample_examples(1, 1, &mut rng).unwrap_err();
        assert!(matches!(error, ReplayError::MemoryBudget { .. }));
    }

    #[test]
    fn arena_progress_reports_actions_and_alternating_candidate_sides() {
        let mut rng = ChaCha8Rng::seed_from_u64(37);
        let candidate = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let incumbent = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let mut events = Vec::new();
        let report = run_arena_schedule_with_progress(
            &candidate,
            &incumbent,
            &[Ruleset::Classic, Ruleset::Multiverse],
            SearchConfig {
                simulations: 1,
                ..SearchConfig::default()
            },
            1,
            &mut rng,
            |event| events.push(event),
        );
        assert_eq!(report.draws, 2);
        let completed = events
            .iter()
            .filter(|event| event.completed)
            .collect::<Vec<_>>();
        assert_eq!(completed.len(), 2);
        assert_eq!(completed[0].candidate_side, Side::Attacker);
        assert_eq!(completed[1].candidate_side, Side::Defender);
        assert_eq!(completed[0].actions, 1);
        assert_eq!(completed[1].actions, 1);
    }
}
