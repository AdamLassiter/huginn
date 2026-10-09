<p align="center">
  <img src="assets/huginn-logo.svg" alt="Huginn — five-dimensional hnefatafl" width="760">
</p>

# Huginn

Huginn is a Rust rules engine with terminal and multiplayer web interfaces for
11x11 Copenhagen hnefatafl described in [RULES.md](RULES.md). It supports both
conventional play and the optional multiverse time-travel extension described
in [5D-RULES.md](5D-RULES.md).

## Run

```sh
cargo run -p huginn-tui
```

The opening menu selects the ruleset and opponent. Both can also be chosen
directly:

```sh
cargo run -p huginn-tui -- --mode classic
cargo run -p huginn-tui -- --mode multiverse
cargo run -p huginn-tui -- --mode classic --opponent muninn
cargo run -p huginn-tui -- --mode classic --opponent huginn
```

The new-game menu selects the ruleset and a human, Muninn, or Huginn opponent.
Muninn uses the CPU-trained `models/training/best-v2.json` checkpoint (overridden
by `HUGINN_AZ_MODEL`); Huginn uses the GPU-trained
`models/training-gpu/best-v2.json` checkpoint (overridden by
`HUGINN_GPU_MODEL`). In AI games the human plays Attackers and the bot plays
Defenders. `--ai-simulations` controls TUI search strength and latency.

Controls:

- Arrow keys or `h/j/k/l`: move the board cursor
- `Enter` or `Space`: select a piece or highlighted destination
- `[` / `]`: move backward or forward through time
- `J` / `K`: move between timelines
- `u`: undo the latest staged multiverse move
- `s`: submit a complete multiverse turn
- `?`: show help
- `n`: return to the new-game menu
- `q`: quit

In multiverse mode, select a piece on a playable Present board, navigate to an
older board of the same parity, then select the same square to make a temporal
move. Moves remain staged until all Present obligations have advanced and the
turn is submitted.

## Multiplayer web server

Start the server and open <http://127.0.0.1:3000>:

```sh
cargo run -p huginn-server
```

The default database is `huginn.sqlite3`. Both settings can be overridden:

```sh
HUGINN_ADDR=0.0.0.0:8080 HUGINN_DATABASE=/var/lib/huginn/huginn.sqlite3 \
  cargo run -p huginn-server
```

The web interface supports account creation, sign-in, classic and multiverse
games, human opponents, the Raven baseline, Muninn CPU-model AlphaZero bot, and
Huginn GPU-model AlphaZero bot, game history, and Elo standings.
The browser polls for opponent moves while a game is open. The server is
authoritative: every submitted move is validated by `huginn-core` before its
new state and history entry are committed to SQLite.

The multiverse view renders every immutable board in one coordinate grid. L
increases from left to right and T increases from top to bottom, so negative
times appear above T0 and positive times below it. The view initially scrolls
T0L0 to the middle-left of the viewport.

## Containers

Build and start the web server with persistent SQLite storage:

```sh
docker compose up --build -d
```

The service is available at <http://127.0.0.1:3000>. Change the host port or
search strength with `HUGINN_PORT` and `HUGINN_AZ_SIMULATIONS`. To use trained
Muninn and Huginn checkpoints, uncomment the model environment entries and bind
mounts in [`docker-compose.yaml`](docker-compose.yaml).

The GitHub workflow tests every change, builds the image on pull requests, and
publishes `ghcr.io/<owner>/<repository>` from `main` and `v*` tags. The matching
Gitea workflow publishes to `<gitea-host>/<owner>/<repository>` using the
built-in `GITEA_TOKEN`. Its runner must expose a Docker daemon, and repository
Actions must allow the token `packages: write` permission.

Both workflows also publish checked, release-mode trainer archives for
`x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc`, including SHA-256
checksums and the rules documentation. GitHub attaches the same archives to
`v*` releases. A self-hosted Gitea installation must register a native Windows
runner with the `windows-latest` label (and Visual Studio C++ build tools) in
addition to its `ubuntu-latest` runner; the Windows binary cannot be produced
by a Linux-only runner.

## Architecture

This is a Cargo workspace:

- `crates/core`: serializable game state, rules, legal moves, and the `AiPlayer`
  integration contract.
- `crates/neural`: Burn-based, backend-neutral multiverse encoder and residual
  policy/value network.
- `crates/alphazero`: PUCT search, compact trajectory replay, self-play, and
  arena evaluation.
- `crates/random-bot`: a separately packaged baseline AI implementation.
- `crates/server`: Axum server, authoritative game orchestration, embedded web
  client, Argon2 passwords, SQLite history, and Elo updates.
- `crates/trainer`: resumable self-play/training/evaluation command-line loop.
- `crates/tui`: the original terminal human-player interface.
- `web`: dependency-free browser client embedded into the server binary.

The `huginn-core` library is independent of either UI. `Game` owns immutable board
history, validates moves, resolves captures and outcomes, and exposes legal
moves and read-only timeline snapshots. AI crates implement `AiPlayer` using an
immutable `Game`; the server still validates every returned `PlayerAction`.
Each loaded AI is provisioned as a virtual user, so it participates in the same
history and Elo system as human accounts.

## AlphaZero training

The overall split follows [TaflZero](https://github.com/sovelin/taflzero): a
Rust search engine generates self-play data for a policy/value learner, then an
arena gate decides whether a candidate replaces the incumbent. Huginn keeps the
trainer in Rust and uses a variable-action head so temporal moves do not require
a finite, pre-enumerated move vocabulary.

Muninn and Huginn learn from MCTS visit distributions and final self-play
outcomes only. Classic positions encode the current board plus compact
repetition state; they do not repeatedly convolve the complete move history.
Multiverse positions encode every immutable board as attacker, defender, king,
throne, and corner planes plus timeline metadata. A shared convolutional
residual trunk embeds each board and caches immutable spatial embeddings;
masked mean/max pooling supplies multiverse context, and the policy head scores
each legal action from that context and its source and destination board
embeddings. Dynamically padded batches handle multiverse histories without a
fixed maximum timeline or time coordinate. Version-2 replay files store compact
authoritative trajectories in `replay-v2.bin.zst` and reconstruct positions for
fitting.

Start or resume the default mixed-rules training loop with:

```sh
cargo run --release -p huginn-trainer -- \
  --work-dir models/training --ruleset both
```

`--model-size auto` is the default. It preserves the original large Muninn v1
network for CPU training, so existing `models/training/best-v2.json`
checkpoints remain compatible. (`v2` in that filename is the serialization
format, not the Muninn model generation.) Vulkan/Huginn training resolves
`auto` to the new balanced
preset: 40 trunk channels, four residual blocks, and a 128-wide board
embedding, compared with 64 channels, six blocks, and a 192-wide embedding in
the large preset. Its residual trunk requires about 26% of the large preset's
convolution work. `--model-size large` and `--model-size compact` remain
available as explicit overrides.

Independent self-play and arena games run concurrently. MCTS evaluates eight
selected leaves per request by default, and the trainer merges requests from
concurrent games into batches of up to 64 positions. Inference batches are also
limited by a conservative padded-tensor estimate (1 GiB by default). Use
`--mcts-batch-size`, `--inference-batch-size`, and
`--inference-memory-budget-mib` to tune those limits. The
default worker count is the logical CPU availability reported by the operating
system through Rust's `available_parallelism()`; pass `--threads N` to override
it. `--threads` is capacity rather than a promise of occupancy: with the
default eight games, at most eight game tasks are runnable at once. Increase
`--games` if you want a larger CPU to keep more game workers busy; startup logs
this distinction explicitly.

Each iteration generates noisy self-play games, appends them to a bounded replay
window, trains a candidate, and evaluates it against the incumbent with colours
alternated. The compact replay can retain 100,000 positions, but fitting
randomly samples at most 20,000 per iteration and caps their expanded host-side
representation at an estimated 4 GiB. Replay compression and decompression are
streamed to avoid duplicate whole-file buffers. A candidate is promoted to
`best-v2.json` only when it reaches the
configured arena score. This JSON checkpoint embeds a backend-neutral,
full-precision Burn record. The server loads the conventional checkpoint paths
automatically; use `HUGINN_AZ_MODEL` to override Muninn's path,
`HUGINN_GPU_MODEL` to override Huginn's path, and `HUGINN_AZ_SIMULATIONS` to
trade response time for search strength. Until a checkpoint exists, the server
explicitly logs that the corresponding bot is using an untrained bootstrap
network. CPU-trained checkpoints conventionally live under `models/training`;
Vulkan-trained checkpoints live under `models/training-gpu`, allowing both bots
to coexist.

The default trainer output is grouped into configuration, device, self-play,
fitting, and arena sections. It reports completed games but suppresses the
per-game action counter to keep long arena runs readable. Add `--verbose` to
show game starts, low-level Vulkan identifiers, and arena progress every 25
actions; change that interval with `--arena-progress-actions` or set it to zero
to disable the action updates. If training is interrupted after
`candidate-v2.json` is written, resume only its promotion match without
repeating self-play:

```sh
cargo run --release -p huginn-trainer -- \
  --work-dir models/training --ruleset both --arena-only
```

### AMD GPU training and inference on Windows

The trainer can fit the network through Burn's Vulkan backend on native
Windows. This does not require ROCm: install a current AMD Adrenalin driver
(which supplies the RX 7900 XTX Vulkan driver), then download and extract the
`huginn-trainer-windows-x86_64` CI artifact or release archive. It can be run
directly from PowerShell without installing Rust:

```powershell
.\huginn-trainer.exe --work-dir models/training-gpu --ruleset both `
  --training-device vulkan
```

The command reports `model=Auto (resolved=Balanced)` at startup. A checkpoint
created with `--model-size large` cannot be resized in place. When switching an
existing GPU work directory, remove its `best-v2.json` and `candidate-v2.json`
and start again; `replay-v2.bin.zst` is architecture-independent and may be
kept. CPU/Muninn work directories continue to resolve to the original large
model.

The trainer initializes and verifies Vulkan before beginning self-play. With
the default `--inference-device auto`, selecting `--training-device vulkan`
also runs self-play and arena inference through Vulkan. A
successful startup names the physical adapter and driver, followed by
`Vulkan compute probe passed`; it rejects CPU, software-rendering, and
non-Vulkan fallback adapters. On a multi-GPU system, select a different
discrete adapter with `--gpu-index N`. Use `--inference-device cpu` to keep
self-play and arena on the CPU while fitting on Vulkan. Check the packaged
executable without starting a training iteration with:

```powershell
.\huginn-trainer.exe --training-device vulkan --gpu-index 0 `
  --check-training-device
```

To build it locally instead, install Rust 1.92 or newer and run:

```powershell
cargo run --release -p huginn-trainer -- `
  --work-dir models/training-gpu --ruleset both `
  --training-device vulkan --model-size large
```

The TUI and server continue to use CPU inference. The trainer reports fitting
and inference backends separately and never silently changes Vulkan to CPU.
During training, messages beginning `self-play inference confirmed` and
`arena inference confirmed` name the adapter after a full policy/value probe
and synchronized readback have succeeded on it.

To fit an existing replay immediately without generating games or running an
arena, add `--fit-only`. Training positions are grouped into similarly shaped
buckets before batching. Both `--batch-token-budget 262144` and the default
`--training-memory-budget-mib 4096` apply to the actual padded batch dimensions,
not the sum of the unpadded positions. An individual position that exceeds the
budget is rejected before Burn allocates it, with its board count, action count,
estimate, and configured budget in the error.

The memory limits control dynamic tensor and expanded-replay estimates; Vulkan
driver, shader, model, and allocator overhead sit outside them. For the 24 GB
RX 7900 XTX and roughly 32 GB of free system memory, the defaults intentionally
leave substantial headroom. If another GPU workload is also active, reduce the
limits without changing the replay or checkpoint:

```powershell
.\huginn-trainer.exe --work-dir models/training-gpu --ruleset both `
  --training-device vulkan `
  --training-memory-budget-mib 512 --inference-memory-budget-mib 256 `
  --reconstruction-memory-budget-mib 1024 --training-examples 10000
```

Startup logs the memory, sample, and batch limits. Before fitting, the trainer
reports the sampled host-memory estimate and the largest position's board/action
dimensions. A Vulkan panic or out-of-memory message is surfaced as
`Vulkan training failed` instead of silently falling back to CPU.

Version-1 `best.json` and `replay.json` files are left untouched because their
hashed inputs and flat network are not compatible with this model. The trainer
warns about them and creates version-2 files alongside them.

For a quick pipeline smoke test rather than useful training:

```sh
cargo run -p huginn-trainer -- \
  --work-dir /tmp/huginn-smoke --games 1 --simulations 2 \
  --max-actions 4 --epochs 1 --arena-games 0 --ruleset classic
```

## Development

```sh
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Run the deterministic release throughput gate locally with:

```sh
cargo run --release -p huginn-benchmarks -- --check
```

It verifies that classic encoding remains independent of history length, that
immutable multiverse embeddings are reused, and that a fixed-seed large-model
self-play workload stays above a conservative throughput floor. Both CI
pipelines run the same command.

The test suite covers the initial setup, Copenhagen movement and capture rules,
special wins, multiverse staging and activity, temporal and timeline captures,
causal board successors, repetition, transactional undo, engine serialization,
the AI boundary, SQLite persistence and rating updates, and an end-to-end
authenticated human-versus-bot HTTP flow.
