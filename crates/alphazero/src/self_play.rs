use std::fs;
use std::path::Path;

use huginn_core::{Game, GameOutcome, PlayerAction, Ruleset, Side};
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::encoding::encode;
use crate::network::{PolicyValueNetwork, TrainingExample};
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
}

#[must_use]
pub fn play_self_play_game(
    model: &PolicyValueNetwork,
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
pub fn play_arena_game(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
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
        let mut examples = Vec::with_capacity(self.len());
        for replay in &self.games {
            let mut game = Game::new(replay.ruleset);
            for (index, step) in replay.steps.iter().enumerate() {
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
                let side = game.turn();
                examples.push(TrainingExample {
                    position: encode(&game, &actions),
                    policy: step.policy.clone(),
                    value: replay.outcome.map_or(0.0, |outcome| {
                        if outcome.winner == side { 1.0 } else { -1.0 }
                    }),
                });
                game.apply_action(step.action).map_err(|error| {
                    ReplayError::Trajectory(format!("step {index} cannot be applied: {error}"))
                })?;
            }
        }
        Ok(examples)
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
        let bytes =
            rmp_serde::to_vec_named(self).map_err(|error| ReplayError::Codec(error.to_string()))?;
        let compressed = zstd::stream::encode_all(bytes.as_slice(), 3)?;
        fs::write(&temporary, compressed)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Loads a version-2 compressed replay window.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be read or decoded.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let compressed = fs::read(path)?;
        let bytes = zstd::stream::decode_all(compressed.as_slice())?;
        let replay: Self =
            rmp_serde::from_slice(&bytes).map_err(|error| ReplayError::Codec(error.to_string()))?;
        if replay.format_version != 2 {
            return Err(ReplayError::Version(replay.format_version));
        }
        Ok(replay)
    }
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
