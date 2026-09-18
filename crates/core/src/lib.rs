//! Rules engine for Copenhagen hnefatafl and its multiverse variant.
//!
//! The engine is UI-independent. Consumers inspect immutable board snapshots,
//! enumerate legal moves, stage moves, and submit complete multiverse turns.

mod game;
mod model;
mod player;

pub use game::{Game, MoveError};
pub use model::{
    BOARD_EDGE, BOARD_SIZE, BOARD_SIZE_U8, Board, BoardCoordinate, BoardSnapshot, GameOutcome,
    Move, OutcomeReason, Piece, Position, Ruleset, Side, Square, Timeline,
};
pub use player::{AiPlayer, AiProfile, GameView, PlayerAction};
