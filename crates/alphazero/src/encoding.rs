use huginn_core::{Game, Piece, PlayerAction, Ruleset, Side};
use serde::{Deserialize, Serialize};

pub const STATE_FEATURES: usize = 384;
pub const ACTION_FEATURES: usize = 24;
const HASH_START: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodedPosition {
    pub state: Vec<f32>,
    pub actions: Vec<Vec<f32>>,
}

#[must_use]
pub fn encode(game: &Game, actions: &[PlayerAction]) -> EncodedPosition {
    EncodedPosition {
        state: encode_state(game),
        actions: actions
            .iter()
            .map(|action| encode_action(game, *action))
            .collect(),
    }
}

#[must_use]
pub fn encode_state(game: &Game) -> Vec<f32> {
    let mut features = vec![0.0; STATE_FEATURES];
    features[0] = side_sign(game.turn());
    features[1] = if game.ruleset() == Ruleset::Multiverse {
        1.0
    } else {
        0.0
    };
    features[2] = f32::from(game.can_submit());
    features[3] = f32::from(game.has_staged_moves());
    features[4] = bounded(game.present_time().unwrap_or_default());

    let timelines = game.timelines().collect::<Vec<_>>();
    let board_count = timelines
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum::<usize>();
    features[5] = (board_count as f32).ln_1p() / 6.0;
    features[6] = (timelines.len() as f32).ln_1p() / 4.0;
    let scale = 1.0 / (board_count.max(1) as f32).sqrt();

    for timeline in timelines {
        let active = game.is_active_timeline(timeline.row);
        let latest = timeline.latest().map(|board| board.coordinate);
        hash_add(&mut features, &[1, timeline.row, i32::from(active)], scale);
        if let Some(owner) = timeline.owner {
            hash_add(&mut features, &[2, timeline.row, side_code(owner)], scale);
        }
        for snapshot in timeline.boards.values() {
            let coordinate = snapshot.coordinate;
            let relative_time = coordinate.time - game.present_time().unwrap_or_default();
            let board_flags = i32::from(Some(coordinate) == latest)
                | (i32::from(game.is_playable_board(coordinate)) << 1)
                | (i32::from(active) << 2);
            hash_add(
                &mut features,
                &[3, relative_time, coordinate.timeline, board_flags],
                scale,
            );
            for (square, piece) in snapshot.board.pieces() {
                hash_add(
                    &mut features,
                    &[
                        4,
                        relative_time,
                        coordinate.timeline,
                        i32::from(square.x),
                        i32::from(square.y),
                        piece_code(piece),
                        board_flags,
                    ],
                    scale,
                );
            }
        }
    }
    features
}

#[must_use]
pub fn encode_action(game: &Game, action: PlayerAction) -> Vec<f32> {
    let mut features = vec![0.0; ACTION_FEATURES];
    features[0] = 1.0;
    features[1] = side_sign(game.turn());
    match action {
        PlayerAction::SubmitTurn => features[2] = 1.0,
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
        }
    }
    features
}

const fn side_code(side: Side) -> i32 {
    match side {
        Side::Attacker => 1,
        Side::Defender => 2,
    }
}

const fn piece_code(piece: Piece) -> i32 {
    match piece {
        Piece::Attacker => 1,
        Piece::Defender => 2,
        Piece::King => 3,
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

fn hash_add(features: &mut [f32], values: &[i32], magnitude: f32) {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let width = features.len() - HASH_START;
    let index = HASH_START + usize::try_from(hash % width as u64).expect("bucket fits");
    let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
    features[index] += sign * magnitude;
}

#[cfg(test)]
mod tests {
    use huginn_core::{Game, Ruleset};

    use super::*;

    #[test]
    fn encoding_is_fixed_size_and_changes_after_a_move() {
        let game = Game::new(Ruleset::Classic);
        let before = encode_state(&game);
        let mut after_game = game.clone();
        after_game
            .apply_action(game.legal_actions()[0])
            .expect("opening action");
        let after = encode_state(&after_game);
        assert_eq!(before.len(), STATE_FEATURES);
        assert_ne!(before, after);
    }

    #[test]
    fn action_encoding_supports_moves_and_submission() {
        let game = Game::new(Ruleset::Classic);
        let move_features = encode_action(&game, game.legal_actions()[0]);
        let submit_features = encode_action(&game, PlayerAction::SubmitTurn);
        assert_eq!(move_features.len(), ACTION_FEATURES);
        assert!((move_features[3] - 1.0).abs() < f32::EPSILON);
        assert!((submit_features[2] - 1.0).abs() < f32::EPSILON);
    }
}
