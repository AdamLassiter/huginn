# Huginn

Huginn is a Rust rules engine with terminal and multiplayer web interfaces for
11x11 Copenhagen hnefatafl. It supports both conventional play and the optional
multiverse time-travel extension described in [5D-RULES.md](5D-RULES.md).

## Run

```sh
cargo run -p huginn-tui
```

The opening menu selects classic or multiverse play. A mode can also be chosen
directly:

```sh
cargo run -p huginn-tui -- --mode classic
cargo run -p huginn-tui -- --mode multiverse
```

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
games, human opponents, the built-in Raven bot, game history, and Elo standings.
The browser polls for opponent moves while a game is open. The server is
authoritative: every submitted move is validated by `huginn-core` before its
new state and history entry are committed to SQLite.

The multiverse view renders every immutable board in one coordinate grid. L
increases from left to right and T increases from top to bottom, so negative
times appear above T0 and positive times below it. The view initially scrolls
T0L0 to the middle-left of the viewport.

## Architecture

This is a Cargo workspace:

- `crates/core`: serializable game state, rules, legal moves, and the `AiPlayer`
  integration contract.
- `crates/random-bot`: a separately packaged baseline AI implementation.
- `crates/server`: Axum server, authoritative game orchestration, embedded web
  client, Argon2 passwords, SQLite history, and Elo updates.
- `crates/tui`: the original terminal human-player interface.
- `web`: dependency-free browser client embedded into the server binary.

The `huginn-core` library is independent of either UI. `Game` owns immutable board
history, validates moves, resolves captures and outcomes, and exposes legal
moves and read-only timeline snapshots. AI crates implement `AiPlayer` using a
read-only `GameView`; the server still validates every returned `PlayerAction`.
Each loaded AI is provisioned as a virtual user, so it participates in the same
history and Elo system as human accounts.

## Development

```sh
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

The test suite covers the initial setup, Copenhagen movement and capture rules,
special wins, multiverse staging and activity, temporal and timeline captures,
causal board successors, repetition, transactional undo, engine serialization,
the AI boundary, SQLite persistence and rating updates, and an end-to-end
authenticated human-versus-bot HTTP flow.
