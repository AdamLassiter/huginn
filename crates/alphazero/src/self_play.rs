use std::fs;
use std::path::Path;

use huginn_core::{Game, GameOutcome, Ruleset, Side};
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
    pub examples: Vec<TrainingExample>,
    pub outcome: Option<GameOutcome>,
    pub truncated: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReplayBuffer {
    examples: Vec<TrainingExample>,
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
    #[error("replay JSON is invalid: {0}")]
    Json(#[from] serde_json::Error),
}

#[must_use]
pub fn play_self_play_game(
    model: &PolicyValueNetwork,
    ruleset: Ruleset,
    config: SelfPlayConfig,
    rng: &mut impl Rng,
) -> SelfPlayGame {
    let mut game = Game::new(ruleset);
    let mut pending = Vec::new();
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
        pending.push((encode(&game, &search.actions), search.policy, game.turn()));
        if game.apply_action(action).is_err() {
            break;
        }
    }
    let outcome = game.outcome();
    let examples = pending
        .into_iter()
        .map(|(position, policy, side)| TrainingExample {
            position,
            policy,
            value: outcome.map_or(0.0, |result| if result.winner == side { 1.0 } else { -1.0 }),
        })
        .collect();
    SelfPlayGame {
        examples,
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
        self.examples.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    #[must_use]
    pub fn examples(&self) -> &[TrainingExample] {
        &self.examples
    }

    pub fn extend(&mut self, examples: impl IntoIterator<Item = TrainingExample>, capacity: usize) {
        self.examples.extend(examples);
        let overflow = self.examples.len().saturating_sub(capacity);
        if overflow > 0 {
            self.examples.drain(..overflow);
        }
    }

    /// Saves the replay window atomically as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the destination cannot be created or serialized.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ReplayError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec(self)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Loads a replay window from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot be read or decoded.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
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
        let model = PolicyValueNetwork::random(NetworkConfig { hidden: 8 }, &mut rng);
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
        assert_eq!(game.examples.len(), 2);
        assert!(game.truncated);
        for example in game.examples {
            assert_eq!(example.position.actions.len(), example.policy.len());
            assert!((example.policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
            assert!(example.value.abs() < f32::EPSILON);
        }
    }

    #[test]
    fn replay_capacity_discards_oldest_examples() {
        let game = Game::new(Ruleset::Classic);
        let actions = game.legal_actions();
        let example = TrainingExample {
            position: encode(&game, &actions),
            policy: vec![1.0 / actions.len() as f32; actions.len()],
            value: 0.0,
        };
        let mut replay = ReplayBuffer::default();
        replay.extend([example.clone(), example.clone(), example], 2);
        assert_eq!(replay.len(), 2);
    }

    #[test]
    fn arena_progress_reports_actions_and_alternating_candidate_sides() {
        let mut rng = ChaCha8Rng::seed_from_u64(37);
        let candidate = PolicyValueNetwork::random(NetworkConfig { hidden: 4 }, &mut rng);
        let incumbent = PolicyValueNetwork::random(NetworkConfig { hidden: 4 }, &mut rng);
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
