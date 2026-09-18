use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const BOARD_SIZE: usize = 11;
pub(crate) const BOARD_SIZE_I32: i32 = 11;
pub const BOARD_SIZE_U8: u8 = 11;
pub const BOARD_EDGE: u8 = 10;
pub(crate) const CENTER: u8 = 5;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub enum Ruleset {
    #[default]
    Classic,
    Multiverse,
}

impl fmt::Display for Ruleset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Classic => f.write_str("Classic Copenhagen"),
            Self::Multiverse => f.write_str("5D Copenhagen"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Side {
    Attacker,
    Defender,
}

impl Side {
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::Attacker => Self::Defender,
            Self::Defender => Self::Attacker,
        }
    }

    #[must_use]
    pub const fn timeline_direction(self) -> i32 {
        match self {
            Self::Attacker => 1,
            Self::Defender => -1,
        }
    }

    #[must_use]
    pub const fn for_time(time: i32) -> Self {
        if time.rem_euclid(2) == 0 {
            Self::Attacker
        } else {
            Self::Defender
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attacker => f.write_str("Attackers"),
            Self::Defender => f.write_str("Defenders"),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub enum Piece {
    Attacker,
    Defender,
    King,
}

impl Piece {
    #[must_use]
    pub const fn side(self) -> Side {
        match self {
            Self::Attacker => Side::Attacker,
            Self::Defender | Self::King => Side::Defender,
        }
    }

    #[must_use]
    pub const fn glyph(self) -> char {
        match self {
            Self::Attacker => 'A',
            Self::Defender => 'D',
            Self::King => 'K',
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Square {
    pub x: u8,
    pub y: u8,
}

impl Square {
    #[must_use]
    pub const fn new(x: u8, y: u8) -> Option<Self> {
        if x < BOARD_SIZE_U8 && y < BOARD_SIZE_U8 {
            Some(Self { x, y })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn is_corner(self) -> bool {
        (self.x == 0 || self.x == BOARD_EDGE) && (self.y == 0 || self.y == BOARD_EDGE)
    }

    #[must_use]
    pub const fn is_throne(self) -> bool {
        self.x == CENTER && self.y == CENTER
    }

    #[must_use]
    pub const fn is_restricted(self) -> bool {
        self.is_corner() || self.is_throne()
    }

    #[must_use]
    pub const fn is_edge(self) -> bool {
        self.x == 0 || self.y == 0 || self.x == BOARD_EDGE || self.y == BOARD_EDGE
    }

    #[must_use]
    pub fn notation(self) -> String {
        let file = char::from(b'A' + self.x);
        format!("{file}{}", self.y + 1)
    }

    pub(crate) fn offset(self, dx: i32, dy: i32) -> Option<Self> {
        let x = i32::from(self.x) + dx;
        let y = i32::from(self.y) + dy;
        if (0..BOARD_SIZE_I32).contains(&x) && (0..BOARD_SIZE_I32).contains(&y) {
            Some(Self {
                x: u8::try_from(x).ok()?,
                y: u8::try_from(y).ok()?,
            })
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.notation())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BoardCoordinate {
    pub time: i32,
    pub timeline: i32,
}

impl BoardCoordinate {
    #[must_use]
    pub const fn new(time: i32, timeline: i32) -> Self {
        Self { time, timeline }
    }
}

impl fmt::Display for BoardCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}L{}", self.time, self.timeline)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Position {
    pub board: BoardCoordinate,
    pub square: Square,
}

impl Position {
    #[must_use]
    pub const fn new(board: BoardCoordinate, square: Square) -> Self {
        Self { board, square }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.board, self.square)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub from: Position,
    pub to: Position,
}

impl Move {
    #[must_use]
    pub const fn new(from: Position, to: Position) -> Self {
        Self { from, to }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct Board {
    cells: [[Option<Piece>; BOARD_SIZE]; BOARD_SIZE],
}

impl Default for Board {
    fn default() -> Self {
        Self::empty()
    }
}

impl Board {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            cells: [[None; BOARD_SIZE]; BOARD_SIZE],
        }
    }

    #[must_use]
    pub fn copenhagen() -> Self {
        let mut board = Self::empty();
        let attackers = [
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (5, 1),
            (0, 3),
            (10, 3),
            (0, 4),
            (10, 4),
            (0, 5),
            (1, 5),
            (9, 5),
            (10, 5),
            (0, 6),
            (10, 6),
            (0, 7),
            (10, 7),
            (5, 9),
            (3, 10),
            (4, 10),
            (5, 10),
            (6, 10),
            (7, 10),
        ];
        let defenders = [
            (5, 3),
            (4, 4),
            (5, 4),
            (6, 4),
            (3, 5),
            (4, 5),
            (6, 5),
            (7, 5),
            (4, 6),
            (5, 6),
            (6, 6),
            (5, 7),
        ];
        for (x, y) in attackers {
            board.cells[y][x] = Some(Piece::Attacker);
        }
        for (x, y) in defenders {
            board.cells[y][x] = Some(Piece::Defender);
        }
        board.cells[usize::from(CENTER)][usize::from(CENTER)] = Some(Piece::King);
        board
    }

    #[must_use]
    pub fn get(&self, square: Square) -> Option<Piece> {
        self.cells[usize::from(square.y)][usize::from(square.x)]
    }

    pub fn set(&mut self, square: Square, piece: Option<Piece>) {
        self.cells[usize::from(square.y)][usize::from(square.x)] = piece;
    }

    pub fn pieces(&self) -> impl Iterator<Item = (Square, Piece)> + '_ {
        (0..BOARD_SIZE_U8).flat_map(move |y| {
            (0..BOARD_SIZE_U8).filter_map(move |x| {
                let square = Square { x, y };
                self.get(square).map(|piece| (square, piece))
            })
        })
    }

    #[must_use]
    pub fn count(&self, piece: Piece) -> usize {
        self.pieces()
            .filter(|(_, candidate)| *candidate == piece)
            .count()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BoardSnapshot {
    pub coordinate: BoardCoordinate,
    pub board: Board,
    pub parent: Option<BoardCoordinate>,
    pub created_by: Option<Side>,
}

impl BoardSnapshot {
    #[must_use]
    pub const fn side_to_move(&self) -> Side {
        Side::for_time(self.coordinate.time)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Timeline {
    pub row: i32,
    pub owner: Option<Side>,
    pub boards: BTreeMap<i32, BoardSnapshot>,
}

impl Timeline {
    #[must_use]
    pub fn latest(&self) -> Option<&BoardSnapshot> {
        self.boards.last_key_value().map(|(_, board)| board)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum OutcomeReason {
    KingEscaped,
    ExitFort,
    KingCaptured,
    Encirclement,
    NoEscape,
    NoLegalTurn,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct GameOutcome {
    pub winner: Side,
    pub reason: OutcomeReason,
}
