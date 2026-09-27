use std::error::Error;
use std::time::Instant;

use huginn_alphazero::{
    NetworkConfig, PolicyValueNetwork, SearchConfig, SelfPlayConfig, encode, play_self_play_game,
};
use huginn_core::{Game, Ruleset};
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;

const SEED: u64 = 0x4855_4749_4e4e_4245;
const MEASURED_ACTIONS: usize = 8;
const SIMULATIONS: usize = 4;
// Roughly 16x below the 16.8 actions/s reference run on a 16-thread Linux host.
// This is deliberately broad enough for shared CI runners while still catching
// the multi-hour throughput class that motivated the benchmark.
const MIN_ACTIONS_PER_SECOND: f64 = 1.0;

fn main() {
    if let Err(error) = run() {
        eprintln!("throughput regression check failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let check = std::env::args()
        .skip(1)
        .any(|argument| argument == "--check");
    if check && cfg!(debug_assertions) {
        return Err("--check must run with cargo run --release".into());
    }

    verify_classic_encoding_is_history_independent()?;
    verify_multiverse_spatial_cache()?;

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let network = PolicyValueNetwork::random(NetworkConfig::large(), &mut rng);
    let opening = Game::new(Ruleset::Classic);
    let opening = encode(&opening, &opening.legal_actions());
    let _ = std::hint::black_box(network.predict(&opening));

    let started = Instant::now();
    let game = play_self_play_game(
        &network,
        Ruleset::Classic,
        SelfPlayConfig {
            search: SearchConfig {
                simulations: SIMULATIONS,
                inference_batch_size: 4,
                ..SearchConfig::default()
            },
            max_actions: MEASURED_ACTIONS,
            exploration_actions: 0,
            temperature: 0.0,
        },
        &mut rng,
    );
    let elapsed = started.elapsed();
    let measured_actions = u32::try_from(game.steps.len()).unwrap_or(u32::MAX);
    let throughput = f64::from(measured_actions) / elapsed.as_secs_f64();
    println!(
        "classic_large_self_play actions={} simulations_per_action={} elapsed={elapsed:.3?} actions_per_second={throughput:.3}",
        game.steps.len(),
        SIMULATIONS,
    );

    if game.steps.len() != MEASURED_ACTIONS {
        return Err(format!(
            "deterministic benchmark completed {} of {MEASURED_ACTIONS} actions",
            game.steps.len()
        )
        .into());
    }
    if check && throughput < MIN_ACTIONS_PER_SECOND {
        return Err(format!(
            "{throughput:.3} actions/s is below the {MIN_ACTIONS_PER_SECOND:.3} actions/s floor"
        )
        .into());
    }
    Ok(())
}

fn verify_classic_encoding_is_history_independent() -> Result<(), Box<dyn Error>> {
    let mut game = Game::new(Ruleset::Classic);
    for _ in 0..24 {
        if game.outcome().is_some() {
            break;
        }
        let Some(action) = game.legal_actions().into_iter().next() else {
            break;
        };
        game.apply_action(action)?;
    }
    let history = game
        .timelines()
        .next()
        .map_or(0, |timeline| timeline.boards.len());
    let encoded = encode(&game, &game.legal_actions());
    println!(
        "classic_encoding history_boards={history} encoded_boards={}",
        encoded.boards.len()
    );
    if encoded.boards.len() != 1 {
        return Err("classic encoding grew with immutable board history".into());
    }
    Ok(())
}

fn verify_multiverse_spatial_cache() -> Result<(), Box<dyn Error>> {
    let mut game = Game::new(Ruleset::Multiverse);
    for _ in 0..6 {
        if game.outcome().is_some() {
            break;
        }
        let Some(action) = game.legal_actions().into_iter().next() else {
            break;
        };
        game.apply_action(action)?;
    }
    let position = encode(&game, &game.legal_actions());
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 1);
    let network = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
    let _ = network.predict(&position);
    let first = network.spatial_cache_stats();
    let _ = network.predict(&position);
    let second = network.spatial_cache_stats();
    println!(
        "multiverse_spatial_cache boards={} entries={} first_misses={} second_misses={} hits={}",
        position.boards.len(),
        second.entries,
        first.misses,
        second.misses,
        second.hits
    );
    if first.misses == 0 || second.misses != first.misses || second.hits == first.hits {
        return Err("immutable multiverse board embeddings were not reused".into());
    }
    Ok(())
}
