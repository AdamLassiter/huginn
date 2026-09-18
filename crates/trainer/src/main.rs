use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use clap::{Parser, ValueEnum};
use huginn_alphazero::{
    ArenaGameConfig, ArenaReport, NetworkConfig, PolicyValueNetwork, ReplayBuffer, SearchConfig,
    SelfPlayConfig, SelfPlayGame, TrainConfig, play_arena_game, play_self_play_game,
};
use huginn_core::{Ruleset, Side};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

const SELF_PLAY_SEED: u64 = 0x5345_4c46_504c_4159;
const ARENA_SEED: u64 = 0x4152_454e_415f_415a;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RuleSelection {
    Classic,
    Multiverse,
    Both,
}

#[derive(Debug, Parser)]
#[command(about = "Train Huginn's AlphaZero opponent entirely through self-play")]
struct Arguments {
    #[arg(long, default_value = "models/training")]
    work_dir: PathBuf,
    /// Worker threads used for independent self-play and arena games.
    #[arg(long, default_value_t = default_threads())]
    threads: usize,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long, default_value_t = 8)]
    games: usize,
    #[arg(long, default_value_t = 48)]
    simulations: usize,
    #[arg(long, default_value_t = 4)]
    arena_games: usize,
    #[arg(long, default_value_t = 25)]
    arena_progress_actions: usize,
    /// Evaluate the saved candidate against best without generating or training.
    #[arg(long)]
    arena_only: bool,
    #[arg(long, default_value_t = 400)]
    max_actions: usize,
    #[arg(long, default_value_t = 100_000)]
    replay_capacity: usize,
    #[arg(long, default_value_t = 4)]
    epochs: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.55)]
    promotion_score: f32,
    #[arg(long, default_value_t = 48)]
    hidden: usize,
    #[arg(long, default_value_t = 0x4855_4749_4e4e)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = RuleSelection::Both)]
    ruleset: RuleSelection,
}

#[derive(Clone, Copy)]
struct ArenaRunConfig {
    selection: RuleSelection,
    games: usize,
    progress_actions: usize,
    search: SearchConfig,
    max_actions: usize,
    seed: u64,
}

struct GeneratedGame {
    game: SelfPlayGame,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse();
    if arguments.threads == 0 {
        return Err("--threads must be greater than zero".into());
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(arguments.threads)
        .thread_name(|index| format!("huginn-trainer-{index}"))
        .build()?;
    train(&arguments, &pool)
}

fn train(arguments: &Arguments, pool: &ThreadPool) -> Result<(), Box<dyn Error>> {
    if arguments.hidden == 0 {
        return Err("--hidden must be greater than zero".into());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(arguments.seed);
    let best_path = arguments.work_dir.join("best.json");
    let candidate_path = arguments.work_dir.join("candidate.json");
    let replay_path = arguments.work_dir.join("replay.json");
    let mut best = load_or_initialize(&best_path, arguments.hidden, &mut rng)?;
    let search = SearchConfig {
        simulations: arguments.simulations,
        ..SearchConfig::default()
    };
    if arguments.arena_only {
        if !candidate_path.exists() {
            return Err(format!(
                "{} does not exist; complete training before using --arena-only",
                candidate_path.display()
            )
            .into());
        }
        let candidate = PolicyValueNetwork::load(&candidate_path)?;
        if candidate.config() != best.config() {
            return Err("candidate and incumbent network shapes differ".into());
        }
        evaluate_and_maybe_promote(&candidate, &mut best, &best_path, arguments, search, pool)?;
        return Ok(());
    }
    let mut replay = if replay_path.exists() {
        ReplayBuffer::load(&replay_path)?
    } else {
        ReplayBuffer::default()
    };

    for iteration in 0..arguments.iterations {
        let mut decisive = 0;
        let mut truncated = 0;
        let generated = generate_self_play(arguments, &best, search, iteration, replay.len(), pool);
        for generated in generated {
            decisive += usize::from(generated.game.outcome.is_some());
            truncated += usize::from(generated.game.truncated);
            replay.extend(generated.game.examples, arguments.replay_capacity);
        }
        println!(
            "iteration {} self-play batch complete: replay={}, decisive={}, truncated={}",
            iteration + 1,
            replay.len(),
            decisive,
            truncated
        );
        replay.save(&replay_path)?;

        let mut candidate = best.clone();
        let metrics = candidate.train(
            replay.examples(),
            TrainConfig {
                epochs: arguments.epochs,
                batch_size: arguments.batch_size,
                learning_rate: arguments.learning_rate,
                ..TrainConfig::default()
            },
            &mut rng,
        );
        candidate.save(&candidate_path)?;
        println!(
            "trained {} examples: policy_loss={:.5}, value_loss={:.5}, steps={}",
            metrics.examples,
            metrics.policy_loss,
            metrics.value_loss,
            candidate.training_steps()
        );

        evaluate_and_maybe_promote(&candidate, &mut best, &best_path, arguments, search, pool)?;
    }
    Ok(())
}

fn evaluate_and_maybe_promote(
    candidate: &PolicyValueNetwork,
    best: &mut PolicyValueNetwork,
    best_path: &Path,
    arguments: &Arguments,
    search: SearchConfig,
    pool: &ThreadPool,
) -> Result<(), Box<dyn Error>> {
    let report = evaluate(
        candidate,
        best,
        ArenaRunConfig {
            selection: arguments.ruleset,
            games: arguments.arena_games,
            progress_actions: arguments.arena_progress_actions,
            search,
            max_actions: arguments.max_actions,
            seed: arguments.seed,
        },
        pool,
    );
    let score = report.candidate_score();
    let promoted = arguments.arena_games == 0 || score >= arguments.promotion_score;
    println!(
        "arena: candidate={} incumbent={} draws={} score={score:.3} promoted={promoted}",
        report.candidate_wins, report.incumbent_wins, report.draws
    );
    if promoted {
        candidate.save(best_path)?;
        *best = candidate.clone();
    }
    Ok(())
}

fn load_or_initialize(
    path: &Path,
    hidden: usize,
    rng: &mut ChaCha8Rng,
) -> Result<PolicyValueNetwork, Box<dyn Error>> {
    if path.exists() {
        let model = PolicyValueNetwork::load(path)?;
        if model.config().hidden != hidden {
            return Err(format!(
                "checkpoint hidden size is {}, but --hidden is {hidden}",
                model.config().hidden
            )
            .into());
        }
        Ok(model)
    } else {
        let model = PolicyValueNetwork::random(NetworkConfig { hidden }, rng);
        model.save(path)?;
        Ok(model)
    }
}

fn evaluate(
    candidate: &PolicyValueNetwork,
    incumbent: &PolicyValueNetwork,
    config: ArenaRunConfig,
    pool: &ThreadPool,
) -> ArenaReport {
    let schedule = (0..config.games)
        .map(|game_index| {
            let side = candidate_side(config.selection, game_index);
            (selected_ruleset(config.selection, game_index), side)
        })
        .collect::<Vec<_>>();
    println!(
        "starting arena: {} games on {} threads, {} simulations/action, {} actions/game",
        config.games,
        pool.current_num_threads(),
        config.search.simulations,
        config.max_actions
    );
    let completed = Mutex::new(0_usize);
    let results = pool.install(|| {
        schedule
            .par_iter()
            .enumerate()
            .map(|(game_index, &(ruleset, candidate_side))| {
                println!(
                    "arena game {}/{} started ({ruleset}; candidate {candidate_side})",
                    game_index + 1,
                    config.games
                );
                let mut rng = ChaCha8Rng::seed_from_u64(derived_seed(
                    config.seed,
                    ARENA_SEED,
                    candidate.training_steps(),
                    usize_to_u64(game_index),
                ));
                let result = play_arena_game(
                    candidate,
                    incumbent,
                    ArenaGameConfig {
                        ruleset,
                        candidate_side,
                        search: config.search,
                        max_actions: config.max_actions,
                    },
                    &mut rng,
                    |actions, _| {
                        if config.progress_actions > 0
                            && actions.is_multiple_of(config.progress_actions)
                        {
                            println!(
                                "arena game {}/{}: {actions} actions searched",
                                game_index + 1,
                                config.games
                            );
                        }
                    },
                );
                let description = result.outcome.map_or_else(
                    || "draw (action limit)".to_owned(),
                    |outcome| format!("{} won ({:?})", outcome.winner, outcome.reason),
                );
                {
                    let mut finished = completed
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *finished += 1;
                    println!(
                        "arena game {}/{} complete ({ruleset}; candidate {candidate_side}): actions={}, {description} ({finished}/{} finished)",
                        game_index + 1,
                        config.games,
                        result.actions,
                        config.games
                    );
                }
                (candidate_side, result.outcome)
            })
            .collect::<Vec<_>>()
    });
    let mut report = ArenaReport::default();
    for (candidate_side, outcome) in results {
        match outcome {
            Some(outcome) if outcome.winner == candidate_side => report.candidate_wins += 1,
            Some(_) => report.incumbent_wins += 1,
            None => report.draws += 1,
        }
    }
    report
}

fn generate_self_play(
    arguments: &Arguments,
    best: &PolicyValueNetwork,
    search: SearchConfig,
    iteration: usize,
    replay_len: usize,
    pool: &ThreadPool,
) -> Vec<GeneratedGame> {
    println!(
        "iteration {} starting {} self-play games on {} threads",
        iteration + 1,
        arguments.games,
        pool.current_num_threads()
    );
    let completed = Mutex::new(0_usize);
    pool.install(|| {
        (0..arguments.games)
            .into_par_iter()
            .map(|game_index| {
                let ruleset = selected_ruleset(arguments.ruleset, iteration + game_index);
                let mut rng = ChaCha8Rng::seed_from_u64(derived_seed(
                    arguments.seed,
                    SELF_PLAY_SEED,
                    usize_to_u64(iteration) ^ usize_to_u64(replay_len).rotate_left(32),
                    usize_to_u64(game_index),
                ));
                let game = play_self_play_game(
                    best,
                    ruleset,
                    SelfPlayConfig {
                        search,
                        max_actions: arguments.max_actions,
                        ..SelfPlayConfig::default()
                    },
                    &mut rng,
                );
                {
                    let mut finished = completed
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *finished += 1;
                    println!(
                        "iteration {} self-play game {}/{} complete ({ruleset}): examples={}, decisive={}, truncated={} ({finished}/{} finished)",
                        iteration + 1,
                        game_index + 1,
                        arguments.games,
                        game.examples.len(),
                        usize::from(game.outcome.is_some()),
                        usize::from(game.truncated),
                        arguments.games
                    );
                }
                GeneratedGame { game }
            })
            .collect()
    })
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

const fn derived_seed(base: u64, phase: u64, sequence: u64, index: u64) -> u64 {
    splitmix64(base ^ phase ^ sequence.rotate_left(17) ^ index.rotate_left(41))
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

const fn selected_ruleset(selection: RuleSelection, index: usize) -> Ruleset {
    match selection {
        RuleSelection::Classic => Ruleset::Classic,
        RuleSelection::Multiverse => Ruleset::Multiverse,
        RuleSelection::Both => {
            if index.is_multiple_of(2) {
                Ruleset::Classic
            } else {
                Ruleset::Multiverse
            }
        }
    }
}

const fn candidate_side(selection: RuleSelection, game_index: usize) -> Side {
    let ruleset_game_index = match selection {
        RuleSelection::Classic | RuleSelection::Multiverse => game_index,
        RuleSelection::Both => game_index / 2,
    };
    if ruleset_game_index.is_multiple_of(2) {
        Side::Attacker
    } else {
        Side::Defender
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn command_line_defaults_to_both_rulesets() {
        let arguments = Arguments::try_parse_from(["trainer"]).expect("arguments");
        assert!(matches!(arguments.ruleset, RuleSelection::Both));
        assert_eq!(arguments.threads, default_threads());
        assert_eq!(arguments.iterations, 1);
    }

    #[test]
    fn mixed_arena_alternates_colours_within_each_ruleset() {
        let sides = (0..8)
            .map(|index| candidate_side(RuleSelection::Both, index))
            .collect::<Vec<_>>();
        assert_eq!(
            sides,
            vec![
                Side::Attacker,
                Side::Attacker,
                Side::Defender,
                Side::Defender,
                Side::Attacker,
                Side::Attacker,
                Side::Defender,
                Side::Defender,
            ]
        );
    }
}
