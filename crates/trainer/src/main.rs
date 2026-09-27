use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use huginn_alphazero::{
    ArenaGameConfig, ArenaReport, EncodedPosition, ModelSize, NetworkConfig, PolicyValueEvaluator,
    PolicyValueNetwork, Prediction, ReplayBuffer, SearchConfig, SelfPlayConfig, SelfPlayGame,
    TrainConfig, play_arena_game, play_self_play_game,
};
use huginn_core::{Ruleset, Side};
use huginn_neural::VulkanTrainingDevice;
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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum TrainingDevice {
    Cpu,
    Vulkan,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum InferenceDevice {
    /// Match the fitting device (Vulkan fitting also enables Vulkan self-play).
    Auto,
    Cpu,
    Vulkan,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelSizeArgument {
    Compact,
    Large,
}

impl From<ModelSizeArgument> for ModelSize {
    fn from(value: ModelSizeArgument) -> Self {
        match value {
            ModelSizeArgument::Compact => Self::Compact,
            ModelSizeArgument::Large => Self::Large,
        }
    }
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
    /// Leaves selected by each MCTS search before one inference request.
    #[arg(long, default_value_t = 8)]
    mcts_batch_size: usize,
    /// Positions merged from concurrent games into one network inference batch.
    #[arg(long, default_value_t = 64)]
    inference_batch_size: usize,
    #[arg(long, default_value_t = 4)]
    arena_games: usize,
    #[arg(long, default_value_t = 25)]
    arena_progress_actions: usize,
    /// Evaluate the saved candidate against best without generating or training.
    #[arg(long)]
    arena_only: bool,
    /// Fit the saved replay only, without self-play or arena evaluation.
    #[arg(long)]
    fit_only: bool,
    /// Initialize and verify the selected fitting device, then exit.
    #[arg(long)]
    check_training_device: bool,
    #[arg(long, default_value_t = 400)]
    max_actions: usize,
    #[arg(long, default_value_t = 100_000)]
    replay_capacity: usize,
    #[arg(long, default_value_t = 4)]
    epochs: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Approximate padded board/action elements allowed in one fitting batch.
    #[arg(long, default_value_t = 262_144)]
    batch_token_budget: usize,
    #[arg(long, default_value_t = 0.001)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.55)]
    promotion_score: f32,
    #[arg(long, value_enum, default_value_t = ModelSizeArgument::Large)]
    model_size: ModelSizeArgument,
    /// Backend used for network fitting.
    #[arg(long, value_enum, default_value_t = TrainingDevice::Cpu)]
    training_device: TrainingDevice,
    /// Backend used for self-play and arena network inference.
    #[arg(long, value_enum, default_value_t = InferenceDevice::Auto)]
    inference_device: InferenceDevice,
    /// Zero-based index among discrete Vulkan GPUs.
    #[arg(long, default_value_t = 0)]
    gpu_index: usize,
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
    inference_batch_size: usize,
    max_actions: usize,
    seed: u64,
    training_steps: u64,
}

struct GeneratedGame {
    game: SelfPlayGame,
}

struct InferenceRequest {
    positions: Vec<EncodedPosition>,
    response: SyncSender<Vec<Prediction>>,
}

#[derive(Clone)]
struct BatchingEvaluator {
    requests: SyncSender<InferenceRequest>,
}

enum FittingDevice {
    Cpu,
    Vulkan(VulkanTrainingDevice),
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse();
    if arguments.threads == 0 {
        return Err("--threads must be greater than zero".into());
    }
    if arguments.mcts_batch_size == 0 || arguments.inference_batch_size == 0 {
        return Err(
            "--mcts-batch-size and --inference-batch-size must be greater than zero".into(),
        );
    }
    let pool = ThreadPoolBuilder::new()
        .num_threads(arguments.threads)
        .thread_name(|index| format!("huginn-trainer-{index}"))
        .build()?;
    train(&arguments, &pool)
}

#[allow(clippy::too_many_lines)]
fn train(arguments: &Arguments, pool: &ThreadPool) -> Result<(), Box<dyn Error>> {
    let fitting_device = prepare_fitting_device(arguments)?;
    if arguments.check_training_device {
        println!("training device check complete; no self-play, fitting, or arena was run");
        return Ok(());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(arguments.seed);
    let best_path = arguments.work_dir.join("best-v2.json");
    let candidate_path = arguments.work_dir.join("candidate-v2.json");
    let replay_path = arguments.work_dir.join("replay-v2.bin.zst");
    warn_about_v1_files(&arguments.work_dir, &best_path, &replay_path);
    let network_config = NetworkConfig::from(ModelSize::from(arguments.model_size));
    let mut best = load_or_initialize(&best_path, network_config, &mut rng)?;
    let search = SearchConfig {
        simulations: arguments.simulations,
        inference_batch_size: arguments.mcts_batch_size,
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
        evaluate_and_maybe_promote(
            &candidate,
            &mut best,
            &best_path,
            arguments,
            search,
            pool,
            fitting_device.as_ref(),
        )?;
        return Ok(());
    }
    let mut replay = if replay_path.exists() {
        ReplayBuffer::load(&replay_path)?
    } else {
        ReplayBuffer::default()
    };

    if arguments.fit_only {
        if replay.is_empty() {
            return Err(format!("{} contains no training positions", replay_path.display()).into());
        }
        let mut candidate = best.clone();
        fit_candidate(
            arguments,
            fitting_device
                .as_ref()
                .expect("fit-only initializes a fitting device"),
            &replay,
            &mut candidate,
            &mut rng,
        )?;
        candidate.save(&candidate_path)?;
        println!(
            "fit-only complete; candidate saved to {} (arena and promotion skipped)",
            candidate_path.display()
        );
        return Ok(());
    }

    for iteration in 0..arguments.iterations {
        let self_play_started = Instant::now();
        let mut decisive = 0;
        let mut truncated = 0;
        let generated = match resolved_inference_device(arguments) {
            InferenceDevice::Cpu | InferenceDevice::Auto => {
                generate_self_play(arguments, &best, search, iteration, replay.len(), pool)
            }
            InferenceDevice::Vulkan => {
                let device = vulkan_device(fitting_device.as_ref())?;
                let evaluator = device.evaluator(&best)?;
                println!(
                    "self-play inference confirmed: full policy/value probe passed on Vulkan adapter '{}'",
                    device.info().name
                );
                generate_self_play(arguments, &evaluator, search, iteration, replay.len(), pool)
            }
        };
        for generated in generated {
            decisive += usize::from(generated.game.outcome.is_some());
            truncated += usize::from(generated.game.truncated);
            replay.extend_game(generated.game, arguments.replay_capacity);
        }
        println!(
            "iteration {} self-play batch complete: replay={}, decisive={}, truncated={}",
            iteration + 1,
            replay.len(),
            decisive,
            truncated,
        );
        println!("self-play time: {:.2?}", self_play_started.elapsed());
        replay.save(&replay_path)?;

        let mut candidate = best.clone();
        fit_candidate(
            arguments,
            fitting_device
                .as_ref()
                .expect("training initializes a fitting device"),
            &replay,
            &mut candidate,
            &mut rng,
        )?;
        candidate.save(&candidate_path)?;

        let arena_started = Instant::now();
        evaluate_and_maybe_promote(
            &candidate,
            &mut best,
            &best_path,
            arguments,
            search,
            pool,
            fitting_device.as_ref(),
        )?;
        println!("arena time: {:.2?}", arena_started.elapsed());
    }
    Ok(())
}

fn prepare_fitting_device(arguments: &Arguments) -> Result<Option<FittingDevice>, Box<dyn Error>> {
    if arguments.arena_only && arguments.fit_only {
        return Err("--arena-only and --fit-only cannot be used together".into());
    }
    if arguments.check_training_device && (arguments.arena_only || arguments.fit_only) {
        return Err(
            "--check-training-device cannot be combined with --arena-only or --fit-only".into(),
        );
    }
    let inference_device = resolved_inference_device(arguments);
    if inference_device == InferenceDevice::Vulkan
        && !matches!(arguments.training_device, TrainingDevice::Vulkan)
        && !arguments.fit_only
        && !arguments.check_training_device
    {
        return Err(
            "Vulkan inference currently requires --training-device vulkan so both phases share one verified adapter"
                .into(),
        );
    }
    println!(
        "trainer startup: work_dir={}, model={:?}, fitting_device={:?}, inference_device={inference_device:?}, threads={}, mcts_batch={}, inference_batch={}",
        arguments.work_dir.display(),
        arguments.model_size,
        arguments.training_device,
        arguments.threads,
        arguments.mcts_batch_size,
        arguments.inference_batch_size
    );
    if !arguments.fit_only && arguments.threads > arguments.games.max(1) {
        println!(
            "parallelism note: --threads={} is worker capacity, but {} self-play games provide at most {} simultaneous game tasks; batched inference is handled by a dedicated coordinator",
            arguments.threads, arguments.games, arguments.games
        );
    }
    let fitting_device = if arguments.arena_only && inference_device == InferenceDevice::Cpu {
        println!("arena-only mode: no fitting device is initialized");
        None
    } else {
        Some(initialize_fitting_device(arguments)?)
    };
    if !arguments.fit_only && !arguments.check_training_device {
        match inference_device {
            InferenceDevice::Cpu | InferenceDevice::Auto => {
                println!("inference backend confirmed: CPU (Burn ndarray)");
            }
            InferenceDevice::Vulkan => println!(
                "inference backend confirmed: Vulkan (self-play and arena will use the verified GPU)"
            ),
        }
    }
    Ok(fitting_device)
}

const fn resolved_inference_device(arguments: &Arguments) -> InferenceDevice {
    match arguments.inference_device {
        InferenceDevice::Auto => match arguments.training_device {
            TrainingDevice::Cpu => InferenceDevice::Cpu,
            TrainingDevice::Vulkan => InferenceDevice::Vulkan,
        },
        selected => selected,
    }
}

fn vulkan_device(
    fitting_device: Option<&FittingDevice>,
) -> Result<&VulkanTrainingDevice, Box<dyn Error>> {
    match fitting_device {
        Some(FittingDevice::Vulkan(device)) => Ok(device),
        Some(FittingDevice::Cpu) | None => {
            Err("Vulkan inference was selected without an initialized Vulkan device".into())
        }
    }
}

fn initialize_fitting_device(arguments: &Arguments) -> Result<FittingDevice, Box<dyn Error>> {
    match arguments.training_device {
        TrainingDevice::Cpu => {
            println!("fitting backend confirmed: CPU (Burn ndarray)");
            Ok(FittingDevice::Cpu)
        }
        TrainingDevice::Vulkan => {
            println!(
                "initializing Vulkan discrete GPU index {} (CPU/software fallback disabled)...",
                arguments.gpu_index
            );
            let device = VulkanTrainingDevice::initialize(arguments.gpu_index)?;
            let info = device.info();
            println!(
                "Vulkan GPU confirmed: adapter='{}', type={}, backend={}, index={}, vendor=0x{:04x}, device=0x{:04x}",
                info.name,
                info.device_type,
                info.backend,
                info.discrete_gpu_index,
                info.vendor,
                info.device
            );
            println!(
                "Vulkan driver: '{}' ({})",
                info.driver,
                if info.driver_info.is_empty() {
                    "no version reported"
                } else {
                    &info.driver_info
                }
            );
            println!(
                "Vulkan compute probe passed on '{}'; fitting will use this GPU",
                info.name
            );
            Ok(FittingDevice::Vulkan(device))
        }
    }
}

fn evaluate_and_maybe_promote(
    candidate: &PolicyValueNetwork,
    best: &mut PolicyValueNetwork,
    best_path: &Path,
    arguments: &Arguments,
    search: SearchConfig,
    pool: &ThreadPool,
    fitting_device: Option<&FittingDevice>,
) -> Result<(), Box<dyn Error>> {
    let config = ArenaRunConfig {
        selection: arguments.ruleset,
        games: arguments.arena_games,
        progress_actions: arguments.arena_progress_actions,
        search,
        inference_batch_size: arguments.inference_batch_size,
        max_actions: arguments.max_actions,
        seed: arguments.seed,
        training_steps: candidate.training_steps(),
    };
    let report = match resolved_inference_device(arguments) {
        InferenceDevice::Cpu | InferenceDevice::Auto => evaluate(candidate, best, config, pool),
        InferenceDevice::Vulkan => {
            let device = vulkan_device(fitting_device)?;
            let candidate_evaluator = device.evaluator(candidate)?;
            let incumbent_evaluator = device.evaluator(best)?;
            println!(
                "arena inference confirmed: full policy/value probes passed on Vulkan adapter '{}'",
                device.info().name
            );
            evaluate(&candidate_evaluator, &incumbent_evaluator, config, pool)
        }
    };
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
    config: NetworkConfig,
    rng: &mut ChaCha8Rng,
) -> Result<PolicyValueNetwork, Box<dyn Error>> {
    if path.exists() {
        let model = PolicyValueNetwork::load(path)?;
        if model.config() != config {
            return Err(format!(
                "checkpoint architecture is {:?}, but --model-size requests {:?}",
                model.config(),
                config
            )
            .into());
        }
        Ok(model)
    } else {
        let model = PolicyValueNetwork::random(config, rng);
        model.save(path)?;
        Ok(model)
    }
}

fn fit_candidate(
    arguments: &Arguments,
    fitting_device: &FittingDevice,
    replay: &ReplayBuffer,
    candidate: &mut PolicyValueNetwork,
    rng: &mut ChaCha8Rng,
) -> Result<(), Box<dyn Error>> {
    let reconstruction_started = Instant::now();
    let examples = replay.examples()?;
    println!(
        "reconstructed {} training positions in {:.2?}",
        examples.len(),
        reconstruction_started.elapsed()
    );
    let fit_started = Instant::now();
    let train_config = TrainConfig {
        epochs: arguments.epochs,
        batch_size: arguments.batch_size,
        batch_token_budget: arguments.batch_token_budget,
        learning_rate: arguments.learning_rate,
        ..TrainConfig::default()
    };
    match fitting_device {
        FittingDevice::Cpu => println!("starting fitting on CPU (Burn ndarray)"),
        FittingDevice::Vulkan(device) => println!(
            "starting GPU fitting on Vulkan adapter '{}'",
            device.info().name
        ),
    }
    let metrics = match fitting_device {
        FittingDevice::Cpu => candidate.train(&examples, train_config, rng),
        FittingDevice::Vulkan(device) => {
            candidate.train_vulkan(device, &examples, train_config, rng)?
        }
    };
    println!(
        "trained {} examples on {:?}: policy_loss={:.5}, value_loss={:.5}, steps={}, fit_time={:.2?}",
        metrics.examples,
        arguments.training_device,
        metrics.policy_loss,
        metrics.value_loss,
        candidate.training_steps(),
        fit_started.elapsed()
    );
    Ok(())
}

fn warn_about_v1_files(work_dir: &Path, best_v2: &Path, replay_v2: &Path) {
    let old_best = work_dir.join("best.json");
    if !best_v2.exists() && old_best.exists() {
        eprintln!(
            "warning: {} is a version-1 checkpoint and cannot initialize the new multiverse network; creating {}",
            old_best.display(),
            best_v2.display()
        );
    }
    let old_replay = work_dir.join("replay.json");
    if !replay_v2.exists() && old_replay.exists() {
        eprintln!(
            "warning: {} is a version-1 replay and is intentionally left untouched; starting {}",
            old_replay.display(),
            replay_v2.display()
        );
    }
}

fn evaluate<E: PolicyValueEvaluator + ?Sized>(
    candidate: &E,
    incumbent: &E,
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
    let results = with_batched_evaluator(
        candidate,
        config.inference_batch_size,
        |candidate_evaluator| {
            with_batched_evaluator(
                incumbent,
                config.inference_batch_size,
                |incumbent_evaluator| {
                    pool.install(|| {
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
                                    config.training_steps,
                                    usize_to_u64(game_index),
                                ));
                                let result = play_arena_game(
                                    candidate_evaluator,
                                    incumbent_evaluator,
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
                                    |outcome| {
                                        format!("{} won ({:?})", outcome.winner, outcome.reason)
                                    },
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
                    })
                },
            )
        },
    );
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

fn generate_self_play<E: PolicyValueEvaluator + ?Sized>(
    arguments: &Arguments,
    best: &E,
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
    with_batched_evaluator(best, arguments.inference_batch_size, |evaluator| {
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
                    evaluator,
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
                            game.steps.len(),
                            usize::from(game.outcome.is_some()),
                            usize::from(game.truncated),
                            arguments.games
                        );
                    }
                    GeneratedGame { game }
                })
                .collect()
        })
    })
}

impl PolicyValueEvaluator for BatchingEvaluator {
    fn predict(&self, position: &EncodedPosition) -> Prediction {
        self.predict_batch(std::slice::from_ref(position))
            .pop()
            .expect("single-position inference returns one result")
    }

    fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
        if positions.is_empty() {
            return Vec::new();
        }
        let (response, results) = mpsc::sync_channel(1);
        self.requests
            .send(InferenceRequest {
                positions: positions.to_vec(),
                response,
            })
            .expect("inference coordinator remains available");
        results
            .recv()
            .expect("inference coordinator returns a result")
    }
}

fn with_batched_evaluator<E: PolicyValueEvaluator + ?Sized, R>(
    model: &E,
    target_batch_size: usize,
    operation: impl FnOnce(&BatchingEvaluator) -> R,
) -> R {
    std::thread::scope(|scope| {
        let (requests, receiver) = mpsc::sync_channel(256);
        let evaluator = BatchingEvaluator { requests };
        let coordinator = scope.spawn(move || {
            run_inference_coordinator(model, &receiver, target_batch_size.max(1));
        });
        let result = operation(&evaluator);
        drop(evaluator);
        coordinator
            .join()
            .expect("inference coordinator must not panic");
        result
    })
}

fn run_inference_coordinator<E: PolicyValueEvaluator + ?Sized>(
    model: &E,
    receiver: &Receiver<InferenceRequest>,
    target_batch_size: usize,
) {
    while let Ok(first) = receiver.recv() {
        let mut count = first.positions.len();
        let mut requests = vec![first];
        while count < target_batch_size {
            match receiver.recv_timeout(Duration::from_micros(200)) {
                Ok(request) => {
                    count += request.positions.len();
                    requests.push(request);
                }
                Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
                    break;
                }
            }
        }
        let positions = requests
            .iter()
            .flat_map(|request| request.positions.iter().cloned())
            .collect::<Vec<_>>();
        let mut predictions = model.predict_batch(&positions).into_iter();
        for request in requests {
            let result = predictions
                .by_ref()
                .take(request.positions.len())
                .collect::<Vec<_>>();
            request
                .response
                .send(result)
                .expect("inference requester remains available");
        }
        assert!(predictions.next().is_none(), "inference result count");
    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use clap::Parser;
    use huginn_alphazero::encode;
    use huginn_core::Game;

    use super::*;

    #[derive(Default)]
    struct CountingEvaluator(AtomicUsize);

    impl PolicyValueEvaluator for CountingEvaluator {
        fn predict(&self, position: &EncodedPosition) -> Prediction {
            self.predict_batch(std::slice::from_ref(position))
                .pop()
                .unwrap()
        }

        fn predict_batch(&self, positions: &[EncodedPosition]) -> Vec<Prediction> {
            self.0.fetch_max(positions.len(), Ordering::Relaxed);
            positions
                .iter()
                .map(|position| Prediction {
                    policy: vec![1.0; position.actions.len()],
                    value: 0.0,
                })
                .collect()
        }
    }

    #[test]
    fn command_line_defaults_to_both_rulesets() {
        let arguments = Arguments::try_parse_from(["trainer"]).expect("arguments");
        assert!(matches!(arguments.ruleset, RuleSelection::Both));
        assert_eq!(arguments.threads, default_threads());
        assert_eq!(arguments.iterations, 1);
        assert!(matches!(arguments.training_device, TrainingDevice::Cpu));
        assert_eq!(arguments.inference_device, InferenceDevice::Auto);
        assert_eq!(resolved_inference_device(&arguments), InferenceDevice::Cpu);
        assert_eq!(arguments.gpu_index, 0);
        assert!(!arguments.check_training_device);
        assert!(matches!(arguments.model_size, ModelSizeArgument::Large));
        assert_eq!(arguments.batch_token_budget, 262_144);
        assert_eq!(arguments.mcts_batch_size, 8);
        assert_eq!(arguments.inference_batch_size, 64);
    }

    #[test]
    fn vulkan_device_and_index_are_parsed_without_initializing_hardware() {
        let arguments = Arguments::try_parse_from([
            "trainer",
            "--training-device",
            "vulkan",
            "--gpu-index",
            "2",
            "--check-training-device",
        ])
        .expect("arguments");
        assert!(matches!(arguments.training_device, TrainingDevice::Vulkan));
        assert_eq!(
            resolved_inference_device(&arguments),
            InferenceDevice::Vulkan
        );
        assert_eq!(arguments.gpu_index, 2);
        assert!(arguments.check_training_device);
    }

    #[test]
    fn obsolete_hidden_option_is_rejected() {
        let error = Arguments::try_parse_from(["trainer", "--hidden", "48"])
            .expect_err("old flat-network option must not be accepted");
        assert!(error.to_string().contains("unexpected argument '--hidden'"));
    }

    #[test]
    fn inference_coordinator_merges_queued_game_requests() {
        let game = Game::new(Ruleset::Classic);
        let position = encode(&game, &game.legal_actions());
        let (requests, receiver) = mpsc::sync_channel(8);
        let mut results = Vec::new();
        for _ in 0..3 {
            let (response, result) = mpsc::sync_channel(1);
            requests
                .send(InferenceRequest {
                    positions: vec![position.clone()],
                    response,
                })
                .unwrap();
            results.push(result);
        }
        drop(requests);
        let evaluator = CountingEvaluator::default();

        run_inference_coordinator(&evaluator, &receiver, 64);

        assert_eq!(evaluator.0.load(Ordering::Relaxed), 3);
        assert!(
            results
                .into_iter()
                .all(|result| result.recv().unwrap().len() == 1)
        );
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
