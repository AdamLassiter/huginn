use std::collections::BTreeMap;

use huginn_core::{BOARD_SIZE, BoardCoordinate, Game, Piece, PlayerAction, Side, Square, Timeline};
use serde::{Deserialize, Serialize};

pub const BOARD_PLANES: usize = 5;
pub const BOARD_METADATA_FEATURES: usize = 15;
pub const GLOBAL_FEATURES: usize = 8;
pub const ACTION_FEATURES: usize = 24;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedBoard {
    pub coordinate: BoardCoordinate,
    pub planes: Vec<f32>,
    pub metadata: [f32; BOARD_METADATA_FEATURES],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedAction {
    pub features: [f32; ACTION_FEATURES],
    pub source_board: Option<usize>,
    pub destination_board: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EncodedPosition {
    pub global: [f32; GLOBAL_FEATURES],
    pub boards: Vec<EncodedBoard>,
    pub actions: Vec<EncodedAction>,
}

#[must_use]
pub fn encode(game: &Game, actions: &[PlayerAction]) -> EncodedPosition {
    let present = game.present_time().unwrap_or_default();
    let timelines = game.timelines().collect::<Vec<_>>();
    let board_count = timelines
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum::<usize>();
    let active_count = timelines
        .iter()
        .filter(|timeline| game.is_active_timeline(timeline.row))
        .count();
    let global = [
        side_sign(game.turn()),
        f32::from(game.ruleset() == huginn_core::Ruleset::Multiverse),
        f32::from(game.can_submit()),
        f32::from(game.has_staged_moves()),
        bounded(present),
        (board_count as f32).ln_1p() / 6.0,
        (timelines.len() as f32).ln_1p() / 4.0,
        (active_count as f32).ln_1p() / 4.0,
    ];

    let mut boards = Vec::with_capacity(board_count);
    for timeline in timelines {
        encode_timeline(game, timeline, present, &mut boards);
    }
    let indices = boards
        .iter()
        .enumerate()
        .map(|(index, board)| (board.coordinate, index))
        .collect::<BTreeMap<_, _>>();
    let actions = actions
        .iter()
        .copied()
        .map(|action| encode_action(game, action, &indices))
        .collect();
    EncodedPosition {
        global,
        boards,
        actions,
    }
}

fn encode_timeline(game: &Game, timeline: &Timeline, present: i32, output: &mut Vec<EncodedBoard>) {
    let latest = timeline.latest().map(|board| board.coordinate);
    let active = game.is_active_timeline(timeline.row);
    for snapshot in timeline.boards.values() {
        let mut planes = vec![0.0; BOARD_PLANES * BOARD_SIZE * BOARD_SIZE];
        for y in 0..BOARD_SIZE {
            for x in 0..BOARD_SIZE {
                let square = Square::new(x as u8, y as u8).expect("board coordinate");
                let cell = y * BOARD_SIZE + x;
                if let Some(piece) = snapshot.board.get(square) {
                    let plane = match piece {
                        Piece::Attacker => 0,
                        Piece::Defender => 1,
                        Piece::King => 2,
                    };
                    planes[plane * BOARD_SIZE * BOARD_SIZE + cell] = 1.0;
                }
                planes[3 * BOARD_SIZE * BOARD_SIZE + cell] = f32::from(square.is_throne());
                planes[4 * BOARD_SIZE * BOARD_SIZE + cell] = f32::from(square.is_corner());
            }
        }
        let owner = timeline.owner;
        let created_by = snapshot.created_by;
        let parent = snapshot.parent;
        let metadata = [
            bounded(snapshot.coordinate.time - present),
            bounded(snapshot.coordinate.timeline),
            side_sign(snapshot.side_to_move()),
            f32::from(Some(snapshot.coordinate) == latest),
            f32::from(active),
            f32::from(game.is_playable_board(snapshot.coordinate)),
            f32::from(owner == Some(Side::Attacker)),
            f32::from(owner == Some(Side::Defender)),
            f32::from(owner.is_none()),
            f32::from(created_by == Some(Side::Attacker)),
            f32::from(created_by == Some(Side::Defender)),
            f32::from(created_by.is_none()),
            f32::from(parent.is_some()),
            parent.map_or(0.0, |value| bounded(value.time - snapshot.coordinate.time)),
            parent.map_or(0.0, |value| {
                bounded(value.timeline - snapshot.coordinate.timeline)
            }),
        ];
        output.push(EncodedBoard {
            coordinate: snapshot.coordinate,
            planes,
            metadata,
        });
    }
}

#[must_use]
pub fn encode_action(
    game: &Game,
    action: PlayerAction,
    board_indices: &BTreeMap<BoardCoordinate, usize>,
) -> EncodedAction {
    let mut features = [0.0; ACTION_FEATURES];
    features[0] = 1.0;
    features[1] = side_sign(game.turn());
    let (source_board, destination_board) = match action {
        PlayerAction::SubmitTurn => {
            features[2] = 1.0;
            (None, None)
        }
        PlayerAction::Move { movement } => {
            features[3] = 1.0;
            let from = movement.from;
            let to = movement.to;
            if let Some(piece) = game
                .board(from.board)
                .and_then(|board| board.board.get(from.square))
            {
                let piece_index = match piece {
                    Piece::Attacker => 4,
                    Piece::Defender => 5,
                    Piece::King => 6,
                };
                features[piece_index] = 1.0;
            }
            features[7] = unit_square(from.square.x);
            features[8] = unit_square(from.square.y);
            features[9] = unit_square(to.square.x);
            features[10] = unit_square(to.square.y);
            features[11] = bounded(from.board.time - game.present_time().unwrap_or_default());
            features[12] = bounded(from.board.timeline);
            features[13] = bounded(to.board.time - game.present_time().unwrap_or_default());
            features[14] = bounded(to.board.timeline);
            features[15] = f32::from(to.square.x) - f32::from(from.square.x);
            features[15] /= 10.0;
            features[16] = f32::from(to.square.y) - f32::from(from.square.y);
            features[16] /= 10.0;
            features[17] = bounded(to.board.time - from.board.time);
            features[18] = bounded(to.board.timeline - from.board.timeline);
            features[19] = f32::from(from.board != to.board);
            features[20] = f32::from(from.square == to.square);
            features[21] = f32::from(to.square.is_edge());
            features[22] = f32::from(to.square.is_corner());
            features[23] = f32::from(to.square.is_throne());
            (
                board_indices.get(&from.board).copied(),
                board_indices.get(&to.board).copied(),
            )
        }
    };
    EncodedAction {
        features,
        source_board,
        destination_board,
    }
}

const fn side_sign(side: Side) -> f32 {
    match side {
        Side::Attacker => 1.0,
        Side::Defender => -1.0,
    }
}

fn unit_square(value: u8) -> f32 {
    f32::from(value) / 5.0 - 1.0
}

fn bounded(value: i32) -> f32 {
    (value as f32 / 8.0).tanh()
}

#[cfg(test)]
mod tests {
    use huginn_core::{Game, Ruleset};

    use super::*;

    #[test]
    fn classic_encoding_contains_spatial_board_and_action_links() {
        let game = Game::new(Ruleset::Classic);
        let actions = game.legal_actions();
        let encoded = encode(&game, &actions);
        assert_eq!(encoded.boards.len(), 1);
        assert_eq!(encoded.boards[0].planes.len(), BOARD_PLANES * 121);
        assert_eq!(encoded.actions.len(), actions.len());
        assert!(encoded.actions.iter().all(|action| {
            action.source_board == Some(0) && action.destination_board == Some(0)
        }));
    }

    #[test]
    fn multiverse_board_order_is_stable_and_submit_uses_sentinels() {
        let mut game = Game::new(Ruleset::Multiverse);
        let first = game.legal_actions()[0];
        game.apply_action(first).expect("legal move");
        let actions = game.legal_actions();
        let one = encode(&game, &actions);
        let two = encode(&game, &actions);
        assert_eq!(one, two);
        if let Some(index) = actions
            .iter()
            .position(|action| *action == PlayerAction::SubmitTurn)
        {
            assert_eq!(one.actions[index].source_board, None);
            assert_eq!(one.actions[index].destination_board, None);
        }
    }
}
