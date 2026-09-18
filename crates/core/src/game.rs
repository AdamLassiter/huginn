use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::model::{
    BOARD_EDGE, BOARD_SIZE, BOARD_SIZE_I32, Board, BoardCoordinate, BoardSnapshot, GameOutcome,
    Move, OutcomeReason, Piece, Position, Ruleset, Side, Square, Timeline,
};

const ORTHOGONAL: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

#[derive(Clone, Debug, Deserialize, Serialize)]
struct State {
    ruleset: Ruleset,
    turn: Side,
    timelines: BTreeMap<i32, Timeline>,
    outcome: Option<GameOutcome>,
    message: String,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    fn sq(x: u8, y: u8) -> Square {
        Square::new(x, y).expect("test square is valid")
    }

    fn pos(time: i32, timeline: i32, x: u8, y: u8) -> Position {
        Position::new(BoardCoordinate::new(time, timeline), sq(x, y))
    }

    fn snapshot(
        time: i32,
        timeline: i32,
        board: Board,
        parent: Option<BoardCoordinate>,
    ) -> BoardSnapshot {
        BoardSnapshot {
            coordinate: BoardCoordinate::new(time, timeline),
            board,
            parent,
            created_by: None,
        }
    }

    fn timeline(row: i32, owner: Option<Side>, snapshots: Vec<BoardSnapshot>) -> Timeline {
        Timeline {
            row,
            owner,
            boards: snapshots
                .into_iter()
                .map(|snapshot| (snapshot.coordinate.time, snapshot))
                .collect(),
        }
    }

    fn game_with(ruleset: Ruleset, turn: Side, timelines: Vec<Timeline>) -> Game {
        let mut game = Game::new(ruleset);
        game.replace_state(
            timelines
                .into_iter()
                .map(|timeline| (timeline.row, timeline))
                .collect(),
            turn,
        );
        game
    }

    fn board_from_rows(rows: [&str; BOARD_SIZE]) -> Board {
        let mut board = Board::empty();
        for (row_index, row) in rows.into_iter().enumerate() {
            let symbols: Vec<char> = row.split_whitespace().flat_map(str::chars).collect();
            assert_eq!(symbols.len(), BOARD_SIZE);
            for (x, symbol) in symbols.into_iter().enumerate() {
                let piece = match symbol {
                    'A' => Some(Piece::Attacker),
                    'D' => Some(Piece::Defender),
                    'K' => Some(Piece::King),
                    '.' | '#' => None,
                    _ => panic!("unknown diagram symbol: {symbol}"),
                };
                if piece.is_some() {
                    board.set(
                        sq(
                            u8::try_from(x).expect("diagram x fits"),
                            BOARD_EDGE - u8::try_from(row_index).expect("diagram y fits"),
                        ),
                        piece,
                    );
                }
            }
        }
        board
    }

    #[test]
    fn copenhagen_setup_has_expected_armies() {
        let game = Game::new(Ruleset::Classic);
        let board = &game
            .board(BoardCoordinate::new(0, 0))
            .expect("origin")
            .board;
        assert_eq!(board.count(Piece::Attacker), 24);
        assert_eq!(board.count(Piece::Defender), 12);
        assert_eq!(board.count(Piece::King), 1);
        assert_eq!(game.turn(), Side::Attacker);
    }

    #[test]
    fn classic_move_creates_successor_and_changes_turn() {
        let mut game = Game::new(Ruleset::Classic);
        game.apply_move(Move::new(pos(0, 0, 3, 0), pos(0, 0, 2, 0)))
            .expect("D1-C1 is legal");
        assert_eq!(game.turn(), Side::Defender);
        assert_eq!(game.latest_coordinate(0), Some(BoardCoordinate::new(1, 0)));
        let board = &game
            .board(BoardCoordinate::new(1, 0))
            .expect("successor")
            .board;
        assert_eq!(board.get(sq(2, 0)), Some(Piece::Attacker));
        assert_eq!(board.get(sq(3, 0)), None);
    }

    #[test]
    fn soldiers_cannot_stop_on_restricted_squares_but_can_cross_empty_throne() {
        let mut board = Board::empty();
        board.set(sq(5, 3), Some(Piece::Attacker));
        board.set(sq(9, 9), Some(Piece::King));
        let game = Game::from_board(Ruleset::Classic, board, Side::Attacker);
        assert_eq!(
            game.validate_move(Move::new(pos(0, 0, 5, 3), pos(0, 0, 5, 5))),
            Err(MoveError::RestrictedDestination)
        );
        game.validate_move(Move::new(pos(0, 0, 5, 3), pos(0, 0, 5, 7)))
            .expect("an empty throne may be crossed");
    }

    #[test]
    fn aggressor_move_closes_a_spatial_capture() {
        let mut board = Board::empty();
        board.set(sq(2, 3), Some(Piece::Attacker));
        board.set(sq(3, 3), Some(Piece::Defender));
        board.set(sq(4, 0), Some(Piece::Attacker));
        board.set(sq(8, 8), Some(Piece::King));
        let mut game = Game::from_board(Ruleset::Classic, board, Side::Attacker);
        game.apply_move(Move::new(pos(0, 0, 4, 0), pos(0, 0, 4, 3)))
            .expect("capture move");
        let board = &game
            .board(BoardCoordinate::new(1, 0))
            .expect("result")
            .board;
        assert_eq!(board.get(sq(3, 3)), None);
    }

    #[test]
    fn occupied_throne_is_not_hostile_to_its_defender_neighbor() {
        let mut board = Board::empty();
        board.set(sq(5, 5), Some(Piece::King));
        board.set(sq(5, 4), Some(Piece::Defender));
        board.set(sq(0, 3), Some(Piece::Attacker));
        let mut game = Game::from_board(Ruleset::Classic, board, Side::Attacker);

        game.apply_move(Move::new(pos(0, 0, 0, 3), pos(0, 0, 5, 3)))
            .expect("move beside the defender");

        assert_eq!(
            game.board(BoardCoordinate::new(1, 0))
                .expect("result")
                .board
                .get(sq(5, 4)),
            Some(Piece::Defender)
        );
    }

    #[test]
    fn empty_throne_is_hostile_to_a_defender() {
        let mut board = Board::empty();
        board.set(sq(5, 4), Some(Piece::Defender));
        board.set(sq(0, 3), Some(Piece::Attacker));
        board.set(sq(8, 8), Some(Piece::King));
        let mut game = Game::from_board(Ruleset::Classic, board, Side::Attacker);

        game.apply_move(Move::new(pos(0, 0, 0, 3), pos(0, 0, 5, 3)))
            .expect("capture against the empty throne");

        assert_eq!(
            game.board(BoardCoordinate::new(1, 0))
                .expect("result")
                .board
                .get(sq(5, 4)),
            None
        );
    }

    #[test]
    fn moving_between_enemies_does_not_self_capture() {
        let mut board = Board::empty();
        board.set(sq(2, 3), Some(Piece::Attacker));
        board.set(sq(4, 3), Some(Piece::Attacker));
        board.set(sq(3, 0), Some(Piece::Defender));
        board.set(sq(8, 8), Some(Piece::King));
        let coordinate = BoardCoordinate::new(1, 0);
        let mut game = game_with(
            Ruleset::Classic,
            Side::Defender,
            vec![timeline(0, None, vec![snapshot(1, 0, board, None)])],
        );
        game.apply_move(Move::new(
            Position::new(coordinate, sq(3, 0)),
            Position::new(coordinate, sq(3, 3)),
        ))
        .expect("defender may enter the sandwich");
        assert_eq!(
            game.board(BoardCoordinate::new(2, 0))
                .expect("result")
                .board
                .get(sq(3, 3)),
            Some(Piece::Defender)
        );
    }

    #[test]
    fn king_requires_two_complete_spatial_axes_in_classic_play() {
        let mut board = Board::empty();
        board.set(sq(5, 5), Some(Piece::King));
        board.set(sq(5, 4), Some(Piece::Attacker));
        board.set(sq(5, 6), Some(Piece::Attacker));
        board.set(sq(4, 5), Some(Piece::Attacker));
        board.set(sq(10, 5), Some(Piece::Attacker));
        let mut game = Game::from_board(Ruleset::Classic, board, Side::Attacker);
        game.apply_move(Move::new(pos(0, 0, 10, 5), pos(0, 0, 6, 5)))
            .expect("closing move");
        assert_eq!(
            game.outcome(),
            Some(GameOutcome {
                winner: Side::Attacker,
                reason: OutcomeReason::KingCaptured,
            })
        );
    }

    #[test]
    fn king_is_not_captured_spatially_on_board_edge() {
        let mut board = Board::empty();
        board.set(sq(4, 0), Some(Piece::King));
        board.set(sq(3, 0), Some(Piece::Attacker));
        board.set(sq(5, 0), Some(Piece::Attacker));
        board.set(sq(4, 2), Some(Piece::Attacker));
        let mut game = Game::from_board(Ruleset::Classic, board, Side::Attacker);
        game.apply_move(Move::new(pos(0, 0, 4, 2), pos(0, 0, 4, 1)))
            .expect("closing three-sided surround");
        assert_ne!(
            game.outcome().map(|outcome| outcome.reason),
            Some(OutcomeReason::KingCaptured)
        );
    }

    #[test]
    fn shield_wall_can_be_closed_by_front_pressure() {
        let mut board = Board::empty();
        for x in 3..=5 {
            board.set(sq(x, 0), Some(Piece::Attacker));
            board.set(sq(x, 1), Some(Piece::Defender));
        }
        board.set(sq(3, 1), None);
        board.set(sq(3, 2), Some(Piece::Defender));
        board.set(sq(2, 0), Some(Piece::Defender));
        board.set(sq(6, 0), Some(Piece::Defender));
        board.set(sq(8, 8), Some(Piece::King));
        let mut game = game_with(
            Ruleset::Classic,
            Side::Defender,
            vec![timeline(0, None, vec![snapshot(1, 0, board, None)])],
        );
        game.apply_move(Move::new(pos(1, 0, 3, 2), pos(1, 0, 3, 1)))
            .expect("shield wall move");
        let board = &game
            .board(BoardCoordinate::new(2, 0))
            .expect("result")
            .board;
        for x in 3..=5 {
            assert_eq!(board.get(sq(x, 0)), None);
        }
    }

    #[test]
    fn king_escape_to_corner_wins() {
        let mut board = Board::empty();
        board.set(sq(0, 2), Some(Piece::King));
        board.set(sq(5, 5), Some(Piece::Defender));
        let mut game = game_with(
            Ruleset::Classic,
            Side::Defender,
            vec![timeline(0, None, vec![snapshot(1, 0, board, None)])],
        );
        game.apply_move(Move::new(pos(1, 0, 0, 2), pos(1, 0, 0, 0)))
            .expect("corner escape");
        assert_eq!(
            game.outcome().map(|outcome| outcome.reason),
            Some(OutcomeReason::KingEscaped)
        );
    }

    #[test]
    fn documented_edge_fort_shape_is_recognized() {
        let mut board = Board::empty();
        board.set(sq(4, 0), Some(Piece::King));
        for square in [sq(3, 0), sq(6, 0), sq(4, 1), sq(5, 1)] {
            board.set(square, Some(Piece::Defender));
        }
        assert!(State::is_exit_fort(&board));
        board.set(sq(4, 1), None);
        assert!(!State::is_exit_fort(&board));
    }

    #[test]
    fn documented_two_space_edge_block_is_a_no_escape_loss() {
        let board = board_from_rows([
            "# . . . . A D D A . #",
            ". . . . A . . . . A .",
            ". . . A . . . . . . A",
            ". . A . . . . . D D D",
            ". . A . . . . . D . K",
            ". . A . . # . . D D A",
            ". . A . . . . . . . A",
            ". . A . . . . . . . A",
            ". . . A . . . . . . A",
            ". . . . A . . . . A .",
            "# . . . . A . . A . #",
        ]);

        assert!(State::defenders_cannot_escape(&board));
    }

    #[test]
    fn documented_low_material_block_is_a_no_escape_loss() {
        let board = board_from_rows([
            "# . A . . . . . A . #",
            ". A . . . . . . . A .",
            "A . . . . . . . . . A",
            ". . . . . . . . . . .",
            ". . . . . . . . . . D",
            "A . . . . # . . . . K",
            ". . . . . . . . . D .",
            ". . . . . . . . . . D",
            "A . . . . . . . . . A",
            ". A . . . . . . . A .",
            "# . A . . . . . A . #",
        ]);

        assert!(State::defenders_cannot_escape(&board));
    }

    #[test]
    fn disconnected_edge_block_is_not_a_no_escape_loss() {
        let board = board_from_rows([
            "# . A A A A D D A . #",
            ". A . . . . . . . A .",
            "A . . . . . . . . . A",
            ". . . . . . . . D D D",
            "A . . . . . . . D . K",
            "A . . . . # . . D D A",
            "A . . . . . . . . . A",
            ". . . . . . . . . . A",
            "A . . . . . . . . . A",
            ". A . . . . . . . A .",
            "# . A A A A . . A . #",
        ]);

        assert!(!State::defenders_cannot_escape(&board));
    }

    #[test]
    fn multiverse_turn_is_staged_until_present_changes_side() {
        let mut game = Game::new(Ruleset::Multiverse);
        game.apply_move(Move::new(pos(0, 0, 3, 0), pos(0, 0, 2, 0)))
            .expect("opening move");
        assert_eq!(game.turn(), Side::Attacker);
        assert!(game.has_staged_moves());
        assert!(game.can_submit());
        game.submit_turn().expect("turn is complete");
        assert_eq!(game.turn(), Side::Defender);
        assert!(!game.has_staged_moves());
    }

    #[test]
    fn temporal_move_advances_source_and_branches_destination() {
        let empty = Board::empty();
        let mut latest = Board::empty();
        latest.set(sq(2, 2), Some(Piece::Attacker));
        latest.set(sq(8, 8), Some(Piece::King));
        let boards = vec![
            snapshot(0, 0, empty.clone(), None),
            snapshot(1, 0, empty.clone(), Some(BoardCoordinate::new(0, 0))),
            snapshot(2, 0, latest, Some(BoardCoordinate::new(1, 0))),
        ];
        let mut game = game_with(
            Ruleset::Multiverse,
            Side::Attacker,
            vec![timeline(0, None, boards)],
        );
        game.apply_move(Move::new(pos(2, 0, 2, 2), pos(0, 0, 2, 2)))
            .expect("temporal move");
        assert_eq!(
            game.board(BoardCoordinate::new(3, 0))
                .expect("source successor")
                .board
                .get(sq(2, 2)),
            None
        );
        assert_eq!(
            game.board(BoardCoordinate::new(1, 1))
                .expect("branch successor")
                .board
                .get(sq(2, 2)),
            Some(Piece::Attacker)
        );
        assert_eq!(
            game.board(BoardCoordinate::new(0, 0))
                .expect("history remains")
                .board
                .get(sq(2, 2)),
            None
        );
    }

    #[test]
    fn temporal_path_checks_intervening_same_parity_boards() {
        let empty = Board::empty();
        let mut middle = Board::empty();
        middle.set(sq(2, 2), Some(Piece::Defender));
        let mut latest = Board::empty();
        latest.set(sq(2, 2), Some(Piece::Attacker));
        latest.set(sq(8, 8), Some(Piece::King));
        let boards = vec![
            snapshot(0, 0, empty.clone(), None),
            snapshot(1, 0, empty.clone(), Some(BoardCoordinate::new(0, 0))),
            snapshot(2, 0, middle, Some(BoardCoordinate::new(1, 0))),
            snapshot(3, 0, empty, Some(BoardCoordinate::new(2, 0))),
            snapshot(4, 0, latest, Some(BoardCoordinate::new(3, 0))),
        ];
        let game = game_with(
            Ruleset::Multiverse,
            Side::Attacker,
            vec![timeline(0, None, boards)],
        );
        assert_eq!(
            game.validate_move(Move::new(pos(4, 0, 2, 2), pos(0, 0, 2, 2))),
            Err(MoveError::BlockedPath)
        );
    }

    #[test]
    fn next_free_outward_timeline_is_used_and_excess_is_inactive() {
        let empty = Board::empty();
        let mut source = Board::empty();
        source.set(sq(2, 2), Some(Piece::Attacker));
        source.set(sq(8, 8), Some(Piece::King));
        let origin = timeline(
            0,
            None,
            vec![
                snapshot(0, 0, empty.clone(), None),
                snapshot(1, 0, empty.clone(), Some(BoardCoordinate::new(0, 0))),
                snapshot(2, 0, empty.clone(), Some(BoardCoordinate::new(1, 0))),
                snapshot(3, 0, empty.clone(), Some(BoardCoordinate::new(2, 0))),
                snapshot(4, 0, source, Some(BoardCoordinate::new(3, 0))),
            ],
        );
        let existing = timeline(
            1,
            Some(Side::Attacker),
            vec![snapshot(5, 1, empty, Some(BoardCoordinate::new(0, 0)))],
        );
        let mut game = game_with(Ruleset::Multiverse, Side::Attacker, vec![origin, existing]);
        game.apply_move(Move::new(pos(4, 0, 2, 2), pos(0, 0, 2, 2)))
            .expect("branch skips occupied L1");
        assert!(game.timeline(2).is_some());
        assert!(!game.is_active_timeline(2));
    }

    #[test]
    fn timeline_balance_activates_only_nearest_owned_rows() {
        let board = Board::empty();
        let game = game_with(
            Ruleset::Multiverse,
            Side::Attacker,
            vec![
                timeline(0, None, vec![snapshot(4, 0, board.clone(), None)]),
                timeline(
                    1,
                    Some(Side::Attacker),
                    vec![snapshot(4, 1, board.clone(), None)],
                ),
                timeline(
                    2,
                    Some(Side::Attacker),
                    vec![snapshot(4, 2, board.clone(), None)],
                ),
                timeline(-1, Some(Side::Defender), vec![snapshot(4, -1, board, None)]),
            ],
        );
        assert!(game.is_active_timeline(0));
        assert!(game.is_active_timeline(1));
        assert!(game.is_active_timeline(2));
        assert!(game.is_active_timeline(-1));
    }

    #[test]
    fn temporal_capture_creates_causal_successor_without_rewriting_history() {
        let mut boards = Vec::new();
        for time in 0..=4 {
            let mut board = Board::empty();
            if time == 1 {
                board.set(sq(6, 5), Some(Piece::Attacker));
            }
            if time == 3 {
                board.set(sq(6, 5), Some(Piece::Defender));
            }
            if time == 4 {
                board.set(sq(6, 1), Some(Piece::Attacker));
                board.set(sq(8, 8), Some(Piece::King));
            }
            boards.push(snapshot(
                time,
                0,
                board,
                (time > 0).then_some(BoardCoordinate::new(time - 1, 0)),
            ));
        }
        let mut game = game_with(
            Ruleset::Multiverse,
            Side::Attacker,
            vec![timeline(0, None, boards)],
        );
        game.apply_move(Move::new(pos(4, 0, 6, 1), pos(4, 0, 6, 5)))
            .expect("move closes temporal sandwich");
        assert_eq!(
            game.board(BoardCoordinate::new(3, 0))
                .expect("history")
                .board
                .get(sq(6, 5)),
            Some(Piece::Defender)
        );
        assert_eq!(
            game.board(BoardCoordinate::new(4, 1))
                .expect("capture branch")
                .board
                .get(sq(6, 5)),
            None
        );
    }

    #[test]
    fn timeline_axis_capture_advances_victim_board() {
        let target = sq(6, 5);
        let mut source_board = Board::empty();
        source_board.set(sq(6, 1), Some(Piece::Attacker));
        source_board.set(sq(8, 8), Some(Piece::King));
        let mut victim_board = Board::empty();
        victim_board.set(target, Some(Piece::Defender));
        let mut support_board = Board::empty();
        support_board.set(target, Some(Piece::Attacker));
        let game_board = Board::empty();
        let mut game = game_with(
            Ruleset::Multiverse,
            Side::Attacker,
            vec![
                timeline(
                    1,
                    Some(Side::Attacker),
                    vec![snapshot(4, 1, source_board, None)],
                ),
                timeline(0, None, vec![snapshot(5, 0, victim_board, None)]),
                timeline(
                    -1,
                    Some(Side::Defender),
                    vec![snapshot(5, -1, support_board, None)],
                ),
                timeline(
                    -2,
                    Some(Side::Defender),
                    vec![snapshot(5, -2, game_board, None)],
                ),
            ],
        );
        game.apply_move(Move::new(pos(4, 1, 6, 1), pos(4, 1, 6, 5)))
            .expect("move closes timeline sandwich");
        assert_eq!(
            game.board(BoardCoordinate::new(6, 0))
                .expect("victim successor")
                .board
                .get(target),
            None
        );
        assert_eq!(
            game.board(BoardCoordinate::new(5, 0))
                .expect("victim history")
                .board
                .get(target),
            Some(Piece::Defender)
        );
    }

    #[test]
    fn two_temporal_axes_can_capture_edge_king() {
        let target = sq(5, 0);
        let mut origin_boards = Vec::new();
        for time in 0..=4 {
            let mut board = Board::empty();
            if time == 1 {
                board.set(target, Some(Piece::Attacker));
            }
            if time == 3 {
                board.set(target, Some(Piece::King));
            }
            if time == 4 {
                board.set(sq(5, 4), Some(Piece::Attacker));
            }
            origin_boards.push(snapshot(
                time,
                0,
                board,
                (time > 0).then_some(BoardCoordinate::new(time - 1, 0)),
            ));
        }
        let mut plus_t3 = Board::empty();
        plus_t3.set(target, Some(Piece::Attacker));
        let mut minus_t3 = Board::empty();
        minus_t3.set(target, Some(Piece::Attacker));
        let plus = timeline(
            1,
            Some(Side::Attacker),
            vec![
                snapshot(3, 1, plus_t3, None),
                snapshot(4, 1, Board::empty(), Some(BoardCoordinate::new(3, 1))),
            ],
        );
        let minus = timeline(
            -1,
            Some(Side::Defender),
            vec![
                snapshot(3, -1, minus_t3, None),
                snapshot(4, -1, Board::empty(), Some(BoardCoordinate::new(3, -1))),
            ],
        );
        let mut game = game_with(
            Ruleset::Multiverse,
            Side::Attacker,
            vec![timeline(0, None, origin_boards), plus, minus],
        );
        game.apply_move(Move::new(pos(4, 0, 5, 4), pos(4, 0, 5, 0)))
            .expect("move completes T and L axes");
        assert_eq!(
            game.outcome().map(|outcome| outcome.reason),
            Some(OutcomeReason::KingCaptured)
        );
    }

    #[test]
    fn defender_cannot_repeat_a_causal_ancestor() {
        let mut at_a = Board::empty();
        at_a.set(sq(1, 1), Some(Piece::Defender));
        at_a.set(sq(8, 8), Some(Piece::King));
        let mut at_b = Board::empty();
        at_b.set(sq(1, 2), Some(Piece::Defender));
        at_b.set(sq(8, 8), Some(Piece::King));
        let boards = vec![
            snapshot(1, 0, at_a, None),
            snapshot(2, 0, at_b.clone(), Some(BoardCoordinate::new(1, 0))),
            snapshot(3, 0, at_b, Some(BoardCoordinate::new(2, 0))),
        ];
        let mut game = game_with(
            Ruleset::Multiverse,
            Side::Defender,
            vec![timeline(0, None, boards)],
        );
        let repetition = Move::new(pos(3, 0, 1, 2), pos(3, 0, 1, 1));
        assert_eq!(
            game.validate_move(repetition),
            Err(MoveError::DefenderRepetition)
        );
        assert!(!game.legal_moves().contains(&repetition));
        assert_eq!(
            game.apply_move(repetition),
            Err(MoveError::DefenderRepetition)
        );
        assert_eq!(game.latest_coordinate(0), Some(BoardCoordinate::new(3, 0)));
    }

    #[test]
    fn undo_restores_entire_staged_multiverse_transaction() {
        let mut game = Game::new(Ruleset::Multiverse);
        let before = game
            .board(BoardCoordinate::new(0, 0))
            .expect("origin")
            .board
            .clone();
        game.apply_move(Move::new(pos(0, 0, 3, 0), pos(0, 0, 2, 0)))
            .expect("opening move");
        game.undo_staged_move().expect("undo");
        assert_eq!(game.latest_coordinate(0), Some(BoardCoordinate::new(0, 0)));
        assert_eq!(
            &game
                .board(BoardCoordinate::new(0, 0))
                .expect("origin")
                .board,
            &before
        );
    }

    #[test]
    fn multiverse_player_loses_when_no_sequence_clears_every_present_board() {
        let mut trapped = Board::empty();
        trapped.set(sq(5, 5), Some(Piece::Attacker));
        trapped.set(sq(5, 4), Some(Piece::Defender));
        trapped.set(sq(5, 6), Some(Piece::Defender));
        trapped.set(sq(4, 5), Some(Piece::Defender));
        trapped.set(sq(6, 5), Some(Piece::Defender));

        let mut movable = Board::empty();
        movable.set(sq(1, 1), Some(Piece::Attacker));

        let mut game = game_with(
            Ruleset::Multiverse,
            Side::Attacker,
            vec![
                timeline(0, None, vec![snapshot(0, 0, trapped, None)]),
                timeline(1, Some(Side::Attacker), vec![snapshot(0, 1, movable, None)]),
            ],
        );
        assert!(game.state.has_any_legal_move());

        game.state.finish_turn_status();

        assert_eq!(
            game.outcome(),
            Some(GameOutcome {
                winner: Side::Defender,
                reason: OutcomeReason::NoLegalTurn,
            })
        );
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Game {
    state: State,
    staged: Vec<State>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoveKind {
    Spatial,
    Temporal,
}

#[derive(Clone, Copy, Debug)]
struct ValidatedMove {
    piece: Piece,
    kind: MoveKind,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MoveError {
    #[error("the game is already over")]
    GameOver,
    #[error("the source or destination board does not exist")]
    MissingBoard,
    #[error("only an active latest board at the Present may be played")]
    SourceNotPlayable,
    #[error("there is no piece at the source square")]
    EmptySource,
    #[error("that piece belongs to the other side")]
    WrongSide,
    #[error("the destination is occupied")]
    OccupiedDestination,
    #[error("soldiers may not stop on the throne or a corner")]
    RestrictedDestination,
    #[error("pieces move orthogonally along exactly one axis")]
    InvalidGeometry,
    #[error("another piece blocks the path")]
    BlockedPath,
    #[error("time travel must move backward an even number of turns on the same timeline")]
    InvalidTimeTravel,
    #[error("the defender may not repeat an ancestral board position")]
    DefenderRepetition,
    #[error("make at least one move before submitting")]
    NothingToSubmit,
    #[error("more Present boards must be played before submitting")]
    PendingPresentBoards,
    #[error("there is no staged move to undo")]
    NothingToUndo,
}

impl Game {
    #[must_use]
    pub fn new(ruleset: Ruleset) -> Self {
        Self::from_board(ruleset, Board::copenhagen(), Side::Attacker)
    }

    #[must_use]
    pub fn from_board(ruleset: Ruleset, board: Board, side: Side) -> Self {
        let coordinate = BoardCoordinate::new(i32::from(side == Side::Defender), 0);
        let snapshot = BoardSnapshot {
            coordinate,
            board,
            parent: None,
            created_by: None,
        };
        let timeline = Timeline {
            row: 0,
            owner: None,
            boards: BTreeMap::from([(coordinate.time, snapshot)]),
        };
        Self {
            state: State {
                ruleset,
                turn: side,
                timelines: BTreeMap::from([(0, timeline)]),
                outcome: None,
                message: format!("{side} to move."),
            },
            staged: Vec::new(),
        }
    }

    #[must_use]
    pub const fn ruleset(&self) -> Ruleset {
        self.state.ruleset
    }

    #[must_use]
    pub const fn turn(&self) -> Side {
        self.state.turn
    }

    #[must_use]
    pub const fn outcome(&self) -> Option<GameOutcome> {
        self.state.outcome
    }

    #[must_use]
    pub fn message(&self) -> &str {
        &self.state.message
    }

    pub fn timelines(&self) -> impl Iterator<Item = &Timeline> {
        self.state.timelines.values()
    }

    #[must_use]
    pub fn timeline(&self, row: i32) -> Option<&Timeline> {
        self.state.timelines.get(&row)
    }

    #[must_use]
    pub fn board(&self, coordinate: BoardCoordinate) -> Option<&BoardSnapshot> {
        self.state
            .timelines
            .get(&coordinate.timeline)?
            .boards
            .get(&coordinate.time)
    }

    #[must_use]
    pub fn latest_coordinate(&self, timeline: i32) -> Option<BoardCoordinate> {
        self.state
            .timelines
            .get(&timeline)?
            .latest()
            .map(|board| board.coordinate)
    }

    #[must_use]
    pub fn present_time(&self) -> Option<i32> {
        self.state.present_time()
    }

    #[must_use]
    pub fn is_active_timeline(&self, timeline: i32) -> bool {
        self.state.is_active_timeline(timeline)
    }

    #[must_use]
    pub fn is_playable_board(&self, coordinate: BoardCoordinate) -> bool {
        self.state.is_playable_board(coordinate)
    }

    #[must_use]
    pub fn playable_boards(&self) -> Vec<BoardCoordinate> {
        self.state.playable_coordinates()
    }

    #[must_use]
    pub fn has_staged_moves(&self) -> bool {
        !self.staged.is_empty()
    }

    #[must_use]
    pub fn can_submit(&self) -> bool {
        self.state.ruleset == Ruleset::Multiverse
            && !self.staged.is_empty()
            && !self.state.has_pending_present_board(self.state.turn)
    }

    #[must_use]
    pub fn legal_moves_from(&self, from: Position) -> Vec<Move> {
        if self.state.outcome.is_some() || !self.state.is_playable_board(from.board) {
            return Vec::new();
        }
        let Some(piece) = self
            .board(from.board)
            .and_then(|board| board.board.get(from.square))
        else {
            return Vec::new();
        };
        if piece.side() != self.state.turn {
            return Vec::new();
        }

        let mut moves = Vec::new();
        for (dx, dy) in ORTHOGONAL {
            for distance in 1..BOARD_SIZE_I32 {
                let Some(square) = from.square.offset(dx * distance, dy * distance) else {
                    break;
                };
                let candidate = Move::new(from, Position::new(from.board, square));
                match self.validate_move(candidate) {
                    Ok(()) => moves.push(candidate),
                    Err(MoveError::RestrictedDestination | MoveError::DefenderRepetition) => {}
                    Err(_) => break,
                }
            }
        }

        if self.state.ruleset == Ruleset::Multiverse {
            let Some(timeline) = self.state.timelines.get(&from.board.timeline) else {
                return moves;
            };
            for time in timeline.boards.keys().copied().rev() {
                if time >= from.board.time || (from.board.time - time).rem_euclid(2) != 0 {
                    continue;
                }
                let candidate = Move::new(
                    from,
                    Position::new(BoardCoordinate::new(time, from.board.timeline), from.square),
                );
                if self.validate_move(candidate).is_ok() {
                    moves.push(candidate);
                }
            }
        }
        moves
    }

    #[must_use]
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        for coordinate in self.state.playable_coordinates() {
            let Some(board) = self.board(coordinate) else {
                continue;
            };
            for (square, piece) in board.board.pieces() {
                if piece.side() == self.state.turn {
                    moves.extend(self.legal_moves_from(Position::new(coordinate, square)));
                }
            }
        }
        moves
    }

    /// Checks a move without changing the game.
    ///
    /// # Errors
    ///
    /// Returns a [`MoveError`] describing the first failed legality rule.
    pub fn validate_move(&self, movement: Move) -> Result<(), MoveError> {
        self.state.validate_move_fully(movement).map(|_| ())
    }

    /// Applies one legal move transactionally.
    ///
    /// # Errors
    ///
    /// Returns a [`MoveError`] when the move is illegal or repeats a defender
    /// position. The game is unchanged on error.
    pub fn apply_move(&mut self, movement: Move) -> Result<(), MoveError> {
        let validated = self.state.validate_move(movement)?;
        let before = self.state.clone();
        let mut next = self.state.clone();
        let created = next.apply_validated_move(movement, validated);

        if validated.piece.side() == Side::Defender && next.repeats_ancestor(&created) {
            return Err(MoveError::DefenderRepetition);
        }

        next.evaluate_outcome();
        next.message = format!(
            "{} moved {} to {}.",
            validated.piece.side(),
            movement.from,
            movement.to
        );

        if next.ruleset == Ruleset::Classic {
            next.turn = next.turn.opponent();
            next.finish_turn_status();
            self.staged.clear();
        } else {
            self.staged.push(before);
        }
        self.state = next;
        Ok(())
    }

    /// Restores the state before the latest staged multiverse move.
    ///
    /// # Errors
    ///
    /// Returns [`MoveError::NothingToUndo`] when no move is staged.
    pub fn undo_staged_move(&mut self) -> Result<(), MoveError> {
        let Some(previous) = self.staged.pop() else {
            return Err(MoveError::NothingToUndo);
        };
        self.state = previous;
        self.state.message = if self.staged.is_empty() {
            format!("{} to move.", self.state.turn)
        } else {
            "Undid staged move.".to_owned()
        };
        Ok(())
    }

    /// Commits a completed multiverse turn and hands play to the new Present.
    ///
    /// # Errors
    ///
    /// Returns an error when there are no staged moves or when the current
    /// side still has an active Present-board obligation.
    pub fn submit_turn(&mut self) -> Result<(), MoveError> {
        if self.state.ruleset == Ruleset::Classic || self.staged.is_empty() {
            return Err(MoveError::NothingToSubmit);
        }
        if self.state.has_pending_present_board(self.state.turn) {
            return Err(MoveError::PendingPresentBoards);
        }
        if self.state.outcome.is_none() {
            let Some(present) = self.state.present_time() else {
                return Err(MoveError::MissingBoard);
            };
            self.state.turn = Side::for_time(present);
            self.state.finish_turn_status();
        }
        self.staged.clear();
        Ok(())
    }

    #[cfg(test)]
    fn replace_state(&mut self, timelines: BTreeMap<i32, Timeline>, turn: Side) {
        self.state.timelines = timelines;
        self.state.turn = turn;
        self.state.outcome = None;
        self.staged.clear();
    }
}

impl State {
    fn validate_move_fully(&self, movement: Move) -> Result<ValidatedMove, MoveError> {
        let validated = self.validate_move(movement)?;
        if validated.piece.side() == Side::Defender {
            let mut next = self.clone();
            let created = next.apply_validated_move(movement, validated);
            if next.repeats_ancestor(&created) {
                return Err(MoveError::DefenderRepetition);
            }
        }
        Ok(validated)
    }

    fn board(&self, coordinate: BoardCoordinate) -> Option<&BoardSnapshot> {
        self.timelines
            .get(&coordinate.timeline)?
            .boards
            .get(&coordinate.time)
    }

    fn board_mut(&mut self, coordinate: BoardCoordinate) -> Option<&mut BoardSnapshot> {
        self.timelines
            .get_mut(&coordinate.timeline)?
            .boards
            .get_mut(&coordinate.time)
    }

    fn piece_at(&self, position: Position) -> Option<Piece> {
        self.board(position.board)?.board.get(position.square)
    }

    fn latest_coordinate(&self, row: i32) -> Option<BoardCoordinate> {
        self.timelines
            .get(&row)?
            .latest()
            .map(|board| board.coordinate)
    }

    fn present_time(&self) -> Option<i32> {
        self.timelines
            .values()
            .filter(|timeline| self.is_active_timeline(timeline.row))
            .filter_map(Timeline::latest)
            .map(|board| board.coordinate.time)
            .min()
    }

    fn active_quota(&self) -> usize {
        let attackers = self
            .timelines
            .values()
            .filter(|timeline| timeline.owner == Some(Side::Attacker))
            .count();
        let defenders = self
            .timelines
            .values()
            .filter(|timeline| timeline.owner == Some(Side::Defender))
            .count();
        attackers.min(defenders) + 1
    }

    fn is_active_timeline(&self, row: i32) -> bool {
        let Some(timeline) = self.timelines.get(&row) else {
            return false;
        };
        let Some(owner) = timeline.owner else {
            return true;
        };
        let mut owned_rows: Vec<i32> = self
            .timelines
            .values()
            .filter(|candidate| candidate.owner == Some(owner))
            .map(|candidate| candidate.row)
            .collect();
        owned_rows.sort_by_key(|candidate| candidate.abs());
        owned_rows
            .into_iter()
            .take(self.active_quota())
            .any(|candidate| candidate == row)
    }

    fn is_playable_board(&self, coordinate: BoardCoordinate) -> bool {
        if self.outcome.is_some()
            || !self.is_active_timeline(coordinate.timeline)
            || self.latest_coordinate(coordinate.timeline) != Some(coordinate)
        {
            return false;
        }
        if self.ruleset == Ruleset::Classic {
            return coordinate.timeline == 0 && Side::for_time(coordinate.time) == self.turn;
        }
        self.present_time() == Some(coordinate.time) && Side::for_time(coordinate.time) == self.turn
    }

    fn playable_coordinates(&self) -> Vec<BoardCoordinate> {
        self.timelines
            .values()
            .filter_map(Timeline::latest)
            .map(|board| board.coordinate)
            .filter(|coordinate| self.is_playable_board(*coordinate))
            .collect()
    }

    fn has_pending_present_board(&self, side: Side) -> bool {
        let Some(present) = self.present_time() else {
            return false;
        };
        self.timelines.values().any(|timeline| {
            self.is_active_timeline(timeline.row)
                && timeline.latest().is_some_and(|board| {
                    board.coordinate.time == present && board.side_to_move() == side
                })
        })
    }

    fn validate_move(&self, movement: Move) -> Result<ValidatedMove, MoveError> {
        if self.outcome.is_some() {
            return Err(MoveError::GameOver);
        }
        if self.board(movement.from.board).is_none() || self.board(movement.to.board).is_none() {
            return Err(MoveError::MissingBoard);
        }
        if !self.is_playable_board(movement.from.board) {
            return Err(MoveError::SourceNotPlayable);
        }
        let piece = self.piece_at(movement.from).ok_or(MoveError::EmptySource)?;
        if piece.side() != self.turn {
            return Err(MoveError::WrongSide);
        }
        if self.piece_at(movement.to).is_some() {
            return Err(MoveError::OccupiedDestination);
        }
        if piece != Piece::King && movement.to.square.is_restricted() {
            return Err(MoveError::RestrictedDestination);
        }

        let same_board = movement.from.board == movement.to.board;
        if same_board {
            let dx = i32::from(movement.to.square.x) - i32::from(movement.from.square.x);
            let dy = i32::from(movement.to.square.y) - i32::from(movement.from.square.y);
            if (dx == 0) == (dy == 0) {
                return Err(MoveError::InvalidGeometry);
            }
            let distance = dx.abs().max(dy.abs());
            let step_x = dx.signum();
            let step_y = dy.signum();
            let board = &self
                .board(movement.from.board)
                .ok_or(MoveError::MissingBoard)?
                .board;
            for step in 1..distance {
                let square = movement
                    .from
                    .square
                    .offset(step_x * step, step_y * step)
                    .ok_or(MoveError::BlockedPath)?;
                if board.get(square).is_some() {
                    return Err(MoveError::BlockedPath);
                }
            }
            return Ok(ValidatedMove {
                piece,
                kind: MoveKind::Spatial,
            });
        }

        if self.ruleset != Ruleset::Multiverse
            || movement.from.square != movement.to.square
            || movement.from.board.timeline != movement.to.board.timeline
        {
            return Err(MoveError::InvalidGeometry);
        }
        let distance = movement.from.board.time - movement.to.board.time;
        if distance <= 0
            || distance.rem_euclid(2) != 0
            || Side::for_time(movement.to.board.time) != piece.side()
        {
            return Err(MoveError::InvalidTimeTravel);
        }
        let timeline = self
            .timelines
            .get(&movement.from.board.timeline)
            .ok_or(MoveError::MissingBoard)?;
        for time in ((movement.to.board.time + 2)..movement.from.board.time).step_by(2) {
            let Some(board) = timeline.boards.get(&time) else {
                return Err(MoveError::BlockedPath);
            };
            if board.board.get(movement.from.square).is_some() {
                return Err(MoveError::BlockedPath);
            }
        }
        Ok(ValidatedMove {
            piece,
            kind: MoveKind::Temporal,
        })
    }

    fn apply_validated_move(
        &mut self,
        movement: Move,
        validated: ValidatedMove,
    ) -> Vec<BoardCoordinate> {
        let mut created = Vec::new();
        let moved_to = match validated.kind {
            MoveKind::Spatial => {
                let source = self
                    .board(movement.from.board)
                    .expect("validated source")
                    .clone();
                let mut board = source.board;
                board.set(movement.from.square, None);
                board.set(movement.to.square, Some(validated.piece));
                let coordinate = BoardCoordinate::new(
                    movement.from.board.time + 1,
                    movement.from.board.timeline,
                );
                self.insert_snapshot(BoardSnapshot {
                    coordinate,
                    board,
                    parent: Some(movement.from.board),
                    created_by: Some(validated.piece.side()),
                });
                created.push(coordinate);
                Position::new(coordinate, movement.to.square)
            }
            MoveKind::Temporal => {
                let source = self
                    .board(movement.from.board)
                    .expect("validated source")
                    .clone();
                let target = self
                    .board(movement.to.board)
                    .expect("validated target")
                    .clone();

                let mut source_board = source.board;
                source_board.set(movement.from.square, None);
                let source_coordinate = BoardCoordinate::new(
                    movement.from.board.time + 1,
                    movement.from.board.timeline,
                );
                self.insert_snapshot(BoardSnapshot {
                    coordinate: source_coordinate,
                    board: source_board,
                    parent: Some(movement.from.board),
                    created_by: Some(validated.piece.side()),
                });
                created.push(source_coordinate);

                let mut target_board = target.board;
                target_board.set(movement.to.square, Some(validated.piece));
                let destination_coordinate = self.insert_successor_or_branch(
                    movement.to.board,
                    target_board,
                    validated.piece.side(),
                );
                created.push(destination_coordinate);
                Position::new(destination_coordinate, movement.to.square)
            }
        };

        let captures = self.captures_closed_by(moved_to, validated.piece.side());
        let (capture_created, king_capture_results) =
            self.apply_captures(captures, &created, validated.piece.side());
        created.extend(capture_created);
        if king_capture_results.iter().any(|coordinate| {
            self.is_active_timeline(coordinate.timeline)
                && self.latest_coordinate(coordinate.timeline) == Some(*coordinate)
        }) {
            self.outcome = Some(GameOutcome {
                winner: Side::Attacker,
                reason: OutcomeReason::KingCaptured,
            });
        }
        created
    }

    fn insert_snapshot(&mut self, snapshot: BoardSnapshot) {
        self.timelines
            .get_mut(&snapshot.coordinate.timeline)
            .expect("timeline exists")
            .boards
            .insert(snapshot.coordinate.time, snapshot);
    }

    fn insert_successor_or_branch(
        &mut self,
        parent: BoardCoordinate,
        board: Board,
        creator: Side,
    ) -> BoardCoordinate {
        let successor = BoardCoordinate::new(parent.time + 1, parent.timeline);
        let occupied = self.board(successor).is_some()
            || self
                .timelines
                .get(&parent.timeline)
                .is_some_and(|timeline| {
                    timeline.boards.range((parent.time + 1)..).next().is_some()
                });
        if !occupied {
            self.insert_snapshot(BoardSnapshot {
                coordinate: successor,
                board,
                parent: Some(parent),
                created_by: Some(creator),
            });
            return successor;
        }

        let direction = creator.timeline_direction();
        let mut row = parent.timeline + direction;
        while self.timelines.contains_key(&row) {
            row += direction;
        }
        let coordinate = BoardCoordinate::new(parent.time + 1, row);
        let snapshot = BoardSnapshot {
            coordinate,
            board,
            parent: Some(parent),
            created_by: Some(creator),
        };
        self.timelines.insert(
            row,
            Timeline {
                row,
                owner: Some(creator),
                boards: BTreeMap::from([(coordinate.time, snapshot)]),
            },
        );
        coordinate
    }

    fn captures_closed_by(&self, moved: Position, side: Side) -> BTreeSet<Position> {
        let mut captures = BTreeSet::new();

        for (dx, dy) in ORTHOGONAL {
            let Some(victim_square) = moved.square.offset(dx, dy) else {
                continue;
            };
            let victim = Position::new(moved.board, victim_square);
            let Some(piece) = self.piece_at(victim) else {
                continue;
            };
            if piece != Piece::King
                && piece.side() != side
                && self.spatial_flank_supports(moved.board, moved.square, dx, dy, side)
            {
                captures.insert(victim);
            }
        }

        if self.ruleset == Ruleset::Multiverse && self.is_active_timeline(moved.board.timeline) {
            for direction in [-1, 1] {
                let victim = Position::new(
                    BoardCoordinate::new(moved.board.time + direction * 2, moved.board.timeline),
                    moved.square,
                );
                let support = Position::new(
                    BoardCoordinate::new(moved.board.time + direction * 4, moved.board.timeline),
                    moved.square,
                );
                if self.is_capturable_with_support(victim, support, side) {
                    captures.insert(victim);
                }

                let victim = Position::new(
                    BoardCoordinate::new(moved.board.time, moved.board.timeline + direction),
                    moved.square,
                );
                let support = Position::new(
                    BoardCoordinate::new(moved.board.time, moved.board.timeline + direction * 2),
                    moved.square,
                );
                if self.is_active_timeline(victim.board.timeline)
                    && self.is_active_timeline(support.board.timeline)
                    && self.is_capturable_with_support(victim, support, side)
                {
                    captures.insert(victim);
                }
            }
        }

        captures.extend(self.shield_wall_captures(moved, side));

        if side == Side::Attacker {
            captures.extend(self.captured_kings_closed_by(moved));
        }
        captures
    }

    fn spatial_flank_supports(
        &self,
        board_coordinate: BoardCoordinate,
        moved: Square,
        dx: i32,
        dy: i32,
        side: Side,
    ) -> bool {
        let Some(flank) = moved.offset(dx * 2, dy * 2) else {
            return false;
        };
        let board = &self
            .board(board_coordinate)
            .expect("moved board exists")
            .board;
        if board.get(flank).is_some_and(|piece| piece.side() == side) {
            return true;
        }
        if flank.is_corner() {
            return true;
        }
        if flank.is_throne() {
            return side == Side::Defender || board.get(flank).is_none();
        }
        false
    }

    fn is_capturable_with_support(&self, victim: Position, support: Position, side: Side) -> bool {
        self.piece_at(victim)
            .is_some_and(|piece| piece != Piece::King && piece.side() != side)
            && self
                .piece_at(support)
                .is_some_and(|piece| piece.side() == side)
    }

    fn captured_kings_closed_by(&self, moved: Position) -> BTreeSet<Position> {
        let mut candidates = BTreeSet::new();
        for (dx, dy) in ORTHOGONAL {
            if let Some(square) = moved.square.offset(dx, dy) {
                candidates.insert(Position::new(moved.board, square));
            }
        }
        if self.ruleset == Ruleset::Multiverse {
            for direction in [-1, 1] {
                candidates.insert(Position::new(
                    BoardCoordinate::new(moved.board.time + direction * 2, moved.board.timeline),
                    moved.square,
                ));
                candidates.insert(Position::new(
                    BoardCoordinate::new(moved.board.time, moved.board.timeline + direction),
                    moved.square,
                ));
            }
        }
        candidates
            .into_iter()
            .filter(|position| self.piece_at(*position) == Some(Piece::King))
            .filter(|position| self.complete_attacker_axes(*position) >= 2)
            .collect()
    }

    fn complete_attacker_axes(&self, king: Position) -> usize {
        let spatial = [((1, 0), (-1, 0)), ((0, 1), (0, -1))];
        let mut axes = spatial
            .into_iter()
            .filter(|(a, b)| {
                self.king_flank_is_hostile(king, a.0, a.1)
                    && self.king_flank_is_hostile(king, b.0, b.1)
            })
            .count();
        if self.ruleset == Ruleset::Multiverse {
            let before = Position::new(
                BoardCoordinate::new(king.board.time - 2, king.board.timeline),
                king.square,
            );
            let after = Position::new(
                BoardCoordinate::new(king.board.time + 2, king.board.timeline),
                king.square,
            );
            axes += usize::from(
                self.piece_at(before) == Some(Piece::Attacker)
                    && self.piece_at(after) == Some(Piece::Attacker),
            );
            let below = Position::new(
                BoardCoordinate::new(king.board.time, king.board.timeline - 1),
                king.square,
            );
            let above = Position::new(
                BoardCoordinate::new(king.board.time, king.board.timeline + 1),
                king.square,
            );
            axes += usize::from(
                self.is_active_timeline(below.board.timeline)
                    && self.is_active_timeline(above.board.timeline)
                    && self.piece_at(below) == Some(Piece::Attacker)
                    && self.piece_at(above) == Some(Piece::Attacker),
            );
        }
        axes
    }

    fn king_flank_is_hostile(&self, king: Position, dx: i32, dy: i32) -> bool {
        let Some(square) = king.square.offset(dx, dy) else {
            return false;
        };
        self.board(king.board).is_some_and(|snapshot| {
            snapshot.board.get(square) == Some(Piece::Attacker)
                || square.is_corner()
                || (square.is_throne() && snapshot.board.get(square).is_none())
        })
    }

    fn shield_wall_captures(&self, moved: Position, side: Side) -> BTreeSet<Position> {
        let mut captures = BTreeSet::new();
        let Some(snapshot) = self.board(moved.board) else {
            return captures;
        };
        let board = &snapshot.board;
        let last = BOARD_EDGE;
        let edges = [
            (true, 0_u8, 1_i32),
            (true, last, -1),
            (false, 0_u8, 1_i32),
            (false, last, -1),
        ];
        for (horizontal, fixed, inward) in edges {
            let edge_square = |axis: u8| {
                if horizontal {
                    Square { x: axis, y: fixed }
                } else {
                    Square { x: fixed, y: axis }
                }
            };
            let inward_square = |axis: u8| {
                let edge = edge_square(axis);
                if horizontal {
                    edge.offset(0, inward)
                } else {
                    edge.offset(inward, 0)
                }
            };
            let bracket = |axis: i32| {
                if !(0..BOARD_SIZE_I32).contains(&axis) {
                    return false;
                }
                let square = edge_square(u8::try_from(axis).expect("edge index fits"));
                square.is_corner() || board.get(square).is_some_and(|piece| piece.side() == side)
            };

            let mut cursor = 0_i32;
            while cursor < BOARD_SIZE_I32 {
                let axis = u8::try_from(cursor).expect("edge index fits");
                let square = edge_square(axis);
                if board.get(square).is_none_or(|piece| piece.side() == side) {
                    cursor += 1;
                    continue;
                }
                let start = cursor;
                let mut run = Vec::new();
                let mut pressure = Vec::new();
                while cursor < BOARD_SIZE_I32 {
                    let axis = u8::try_from(cursor).expect("edge index fits");
                    let square = edge_square(axis);
                    if board.get(square).is_none_or(|piece| piece.side() == side) {
                        break;
                    }
                    let Some(inward) = inward_square(axis) else {
                        break;
                    };
                    if board.get(inward).is_none_or(|piece| piece.side() != side) {
                        break;
                    }
                    run.push(square);
                    pressure.push(inward);
                    cursor += 1;
                }
                let end = cursor;
                let moved_participates = pressure.contains(&moved.square)
                    || (start > 0
                        && edge_square(u8::try_from(start - 1).expect("index fits"))
                            == moved.square)
                    || (end < BOARD_SIZE_I32
                        && edge_square(u8::try_from(end).expect("index fits")) == moved.square);
                if run.len() >= 2 && bracket(start - 1) && bracket(end) && moved_participates {
                    captures.extend(
                        run.into_iter()
                            .filter(|square| board.get(*square) != Some(Piece::King))
                            .map(|square| Position::new(moved.board, square)),
                    );
                }
                if cursor == start {
                    cursor += 1;
                }
            }
        }
        captures
    }

    fn apply_captures(
        &mut self,
        capture_positions: BTreeSet<Position>,
        created_by_move: &[BoardCoordinate],
        capturing_side: Side,
    ) -> (Vec<BoardCoordinate>, Vec<BoardCoordinate>) {
        let mut grouped: BTreeMap<BoardCoordinate, Vec<(Square, Piece)>> = BTreeMap::new();
        for position in capture_positions {
            if let Some(piece) = self.piece_at(position) {
                grouped
                    .entry(position.board)
                    .or_default()
                    .push((position.square, piece));
            }
        }
        let mut created = Vec::new();
        let mut king_results = Vec::new();
        for (coordinate, victims) in grouped {
            let contains_king = victims.iter().any(|(_, piece)| *piece == Piece::King);
            if created_by_move.contains(&coordinate) {
                if let Some(snapshot) = self.board_mut(coordinate) {
                    for (square, _) in victims {
                        snapshot.board.set(square, None);
                    }
                }
                if contains_king {
                    king_results.push(coordinate);
                }
                continue;
            }
            let Some(snapshot) = self.board(coordinate).cloned() else {
                continue;
            };
            let mut board = snapshot.board;
            for (square, _) in victims {
                board.set(square, None);
            }
            let result = self.insert_successor_or_branch(coordinate, board, capturing_side);
            created.push(result);
            if contains_king {
                king_results.push(result);
            }
        }
        (created, king_results)
    }

    fn repeats_ancestor(&self, created: &[BoardCoordinate]) -> bool {
        created.iter().any(|coordinate| {
            let Some(snapshot) = self.board(*coordinate) else {
                return false;
            };
            let board = &snapshot.board;
            let mut parent = snapshot.parent;
            while let Some(parent_coordinate) = parent {
                let Some(ancestor) = self.board(parent_coordinate) else {
                    break;
                };
                if &ancestor.board == board {
                    return true;
                }
                parent = ancestor.parent;
            }
            false
        })
    }

    fn evaluate_outcome(&mut self) {
        if self.outcome.is_some() {
            return;
        }
        let latest: Vec<BoardCoordinate> = self
            .timelines
            .values()
            .filter(|timeline| self.is_active_timeline(timeline.row))
            .filter_map(Timeline::latest)
            .map(|board| board.coordinate)
            .collect();
        for coordinate in latest {
            let Some(board) = self
                .board(coordinate)
                .map(|snapshot| snapshot.board.clone())
            else {
                continue;
            };
            if board
                .pieces()
                .any(|(square, piece)| piece == Piece::King && square.is_corner())
            {
                self.outcome = Some(GameOutcome {
                    winner: Side::Defender,
                    reason: OutcomeReason::KingEscaped,
                });
                return;
            }
            if Self::is_exit_fort(&board) {
                self.outcome = Some(GameOutcome {
                    winner: Side::Defender,
                    reason: OutcomeReason::ExitFort,
                });
                return;
            }
            if Self::attackers_encircle_all_defenders(&board) {
                self.outcome = Some(GameOutcome {
                    winner: Side::Attacker,
                    reason: OutcomeReason::Encirclement,
                });
                return;
            }
            if Self::defenders_cannot_escape(&board) {
                self.outcome = Some(GameOutcome {
                    winner: Side::Attacker,
                    reason: OutcomeReason::NoEscape,
                });
                return;
            }
        }
    }

    fn finish_turn_status(&mut self) {
        if let Some(outcome) = self.outcome {
            self.message = format!("{} win: {:?}.", outcome.winner, outcome.reason);
            return;
        }
        let turn_can_be_completed = if self.ruleset == Ruleset::Classic {
            self.has_any_legal_move()
        } else {
            self.has_legal_turn_completion()
        };
        if turn_can_be_completed {
            self.message = format!("{} to move.", self.turn);
        } else {
            self.outcome = Some(GameOutcome {
                winner: self.turn.opponent(),
                reason: OutcomeReason::NoLegalTurn,
            });
            self.message = format!("{} have no legal move.", self.turn);
        }
    }

    fn has_any_legal_move(&self) -> bool {
        self.playable_coordinates().into_iter().any(|coordinate| {
            let Some(snapshot) = self.board(coordinate) else {
                return false;
            };
            snapshot.board.pieces().any(|(square, piece)| {
                piece.side() == self.turn
                    && self.has_legal_destination(Position::new(coordinate, square))
            })
        })
    }

    fn has_legal_turn_completion(&self) -> bool {
        if self.outcome.is_some() || !self.has_pending_present_board(self.turn) {
            return true;
        }

        let game = Game {
            state: self.clone(),
            staged: Vec::new(),
        };
        game.legal_moves().into_iter().any(|movement| {
            let mut continuation = game.clone();
            continuation.apply_move(movement).is_ok()
                && continuation.state.has_legal_turn_completion()
        })
    }

    fn has_legal_destination(&self, from: Position) -> bool {
        for (dx, dy) in ORTHOGONAL {
            if let Some(square) = from.square.offset(dx, dy) {
                let movement = Move::new(from, Position::new(from.board, square));
                if self.validate_move_fully(movement).is_ok() {
                    return true;
                }
            }
        }
        if self.ruleset == Ruleset::Multiverse
            && let Some(timeline) = self.timelines.get(&from.board.timeline)
        {
            return timeline.boards.keys().copied().any(|time| {
                time < from.board.time
                    && (from.board.time - time).rem_euclid(2) == 0
                    && self
                        .validate_move_fully(Move::new(
                            from,
                            Position::new(
                                BoardCoordinate::new(time, from.board.timeline),
                                from.square,
                            ),
                        ))
                        .is_ok()
            });
        }
        false
    }

    fn attackers_encircle_all_defenders(board: &Board) -> bool {
        let Some((king, _)) = board.pieces().find(|(_, piece)| *piece == Piece::King) else {
            return false;
        };
        let reachable = Self::flood_non_attackers(board, king, None);
        let all_defenders_inside = board
            .pieces()
            .filter(|(_, piece)| piece.side() == Side::Defender)
            .all(|(square, _)| reachable.contains(&square));
        let reaches_corner = reachable.iter().any(|square| square.is_corner());
        all_defenders_inside && !reaches_corner
    }

    fn defenders_cannot_escape(board: &Board) -> bool {
        if Self::is_exit_fort(board) {
            return false;
        }
        let Some((king, _)) = board.pieces().find(|(_, piece)| *piece == Piece::King) else {
            return false;
        };
        let defender_region = Self::flood_non_attackers(board, king, None);
        if defender_region.iter().any(|square| square.is_corner()) {
            return false;
        }
        if board.count(Piece::Defender) <= 5 {
            return true;
        }

        let side_openings = [
            defender_region
                .iter()
                .filter(|square| square.y == 0)
                .map(|square| square.x)
                .collect::<Vec<_>>(),
            defender_region
                .iter()
                .filter(|square| square.y == BOARD_EDGE)
                .map(|square| square.x)
                .collect(),
            defender_region
                .iter()
                .filter(|square| square.x == 0)
                .map(|square| square.y)
                .collect(),
            defender_region
                .iter()
                .filter(|square| square.x == BOARD_EDGE)
                .map(|square| square.y)
                .collect(),
        ];
        let mut touches_edge = false;
        for opening in side_openings {
            let (Some(first), Some(last)) = (opening.first(), opening.last()) else {
                continue;
            };
            touches_edge = true;
            let connected = usize::from(*last - *first) + 1 == opening.len();
            if !connected || opening.len() > 2 {
                return false;
            }
        }
        touches_edge
    }

    fn is_exit_fort(board: &Board) -> bool {
        let Some((king, _)) = board.pieces().find(|(_, piece)| *piece == Piece::King) else {
            return false;
        };
        if !king.is_edge() || king.is_corner() {
            return false;
        }
        if !ORTHOGONAL.iter().any(|(dx, dy)| {
            king.offset(*dx, *dy)
                .is_some_and(|square| board.get(square).is_none() && !square.is_corner())
        }) {
            return false;
        }

        let horizontal = king.y == 0 || usize::from(king.y) == BOARD_SIZE - 1;
        let edge_defenders: Vec<Square> = board
            .pieces()
            .filter(|(square, piece)| {
                *piece == Piece::Defender
                    && if horizontal {
                        square.y == king.y
                    } else {
                        square.x == king.x
                    }
            })
            .map(|(square, _)| square)
            .collect();
        if edge_defenders.len() < 2 {
            return false;
        }
        let before = edge_defenders.iter().any(|square| {
            if horizontal {
                square.x < king.x
            } else {
                square.y < king.y
            }
        });
        let after = edge_defenders.iter().any(|square| {
            if horizontal {
                square.x > king.x
            } else {
                square.y > king.y
            }
        });
        if !before || !after {
            return false;
        }

        let Some(start) = edge_defenders
            .iter()
            .find(|square| {
                if horizontal {
                    square.x < king.x
                } else {
                    square.y < king.y
                }
            })
            .copied()
        else {
            return false;
        };
        let connected = Self::connected_defenders(board, start);
        if !edge_defenders.iter().any(|square| {
            connected.contains(square)
                && if horizontal {
                    square.x > king.x
                } else {
                    square.y > king.y
                }
        }) {
            return false;
        }
        let interior = Self::flood_avoiding_wall(king, &connected);
        !interior
            .iter()
            .any(|square| board.get(*square) == Some(Piece::Attacker))
            && interior.len() < BOARD_SIZE * BOARD_SIZE - connected.len()
    }

    fn connected_defenders(board: &Board, start: Square) -> BTreeSet<Square> {
        Self::connected_piece(board, start, Piece::Defender)
    }

    fn connected_piece(board: &Board, start: Square, piece: Piece) -> BTreeSet<Square> {
        let mut connected = BTreeSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(square) = queue.pop_front() {
            if !connected.insert(square) {
                continue;
            }
            for dx in -1..=1 {
                for dy in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    if let Some(next) = square.offset(dx, dy)
                        && board.get(next) == Some(piece)
                        && !connected.contains(&next)
                    {
                        queue.push_back(next);
                    }
                }
            }
        }
        connected
    }

    fn flood_non_attackers(
        board: &Board,
        start: Square,
        wall: Option<&BTreeSet<Square>>,
    ) -> BTreeSet<Square> {
        let mut reached = BTreeSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(square) = queue.pop_front() {
            if reached.contains(&square) || wall.is_some_and(|wall| wall.contains(&square)) {
                continue;
            }
            if board.get(square) == Some(Piece::Attacker) {
                continue;
            }
            reached.insert(square);
            for (dx, dy) in ORTHOGONAL {
                if let Some(next) = square.offset(dx, dy) {
                    queue.push_back(next);
                }
            }
        }
        reached
    }

    fn flood_avoiding_wall(start: Square, wall: &BTreeSet<Square>) -> BTreeSet<Square> {
        let mut reached = BTreeSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(square) = queue.pop_front() {
            if reached.contains(&square) || wall.contains(&square) {
                continue;
            }
            reached.insert(square);
            for (dx, dy) in ORTHOGONAL {
                if let Some(next) = square.offset(dx, dy) {
                    queue.push_back(next);
                }
            }
        }
        reached
    }
}
