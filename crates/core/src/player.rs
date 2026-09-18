use serde::{Deserialize, Serialize};

use crate::{BoardCoordinate, Game, GameOutcome, Move, Ruleset, Side, Timeline};

/// Read-only state supplied to human frontends and pluggable AI players.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GameView {
    pub ruleset: Ruleset,
    pub turn: Side,
    pub outcome: Option<GameOutcome>,
    pub message: String,
    pub timelines: Vec<Timeline>,
    pub legal_moves: Vec<Move>,
    pub can_submit: bool,
    pub has_staged_moves: bool,
    pub present_time: Option<i32>,
    pub active_timelines: Vec<i32>,
    pub playable_boards: Vec<BoardCoordinate>,
}

impl GameView {
    #[must_use]
    pub fn from_game(game: &Game) -> Self {
        let timelines: Vec<Timeline> = game.timelines().cloned().collect();
        let active_timelines = timelines
            .iter()
            .filter(|timeline| game.is_active_timeline(timeline.row))
            .map(|timeline| timeline.row)
            .collect();
        Self {
            ruleset: game.ruleset(),
            turn: game.turn(),
            outcome: game.outcome(),
            message: game.message().to_owned(),
            timelines,
            legal_moves: game.legal_moves(),
            can_submit: game.can_submit(),
            has_staged_moves: game.has_staged_moves(),
            present_time: game.present_time(),
            active_timelines,
            playable_boards: game.playable_boards(),
        }
    }
}

/// Stable account metadata for an AI implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiProfile {
    pub username: &'static str,
    pub display_name: &'static str,
}

/// An action an AI can request from the authoritative engine.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum PlayerAction {
    Move { movement: Move },
    SubmitTurn,
}

/// Interface implemented by separately packaged AI players.
///
/// The server remains authoritative: it validates the returned action through
/// [`Game`] before committing it.
pub trait AiPlayer: Send {
    fn profile(&self) -> AiProfile;

    /// Selects one move or turn submission from a read-only position.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic if the implementation cannot select an action.
    fn choose_action(&mut self, game: &GameView) -> Result<PlayerAction, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_view_round_trips_as_json() {
        let view = GameView::from_game(&Game::new(Ruleset::Multiverse));
        let encoded = serde_json::to_string(&view).expect("serialize view");
        let decoded: GameView = serde_json::from_str(&encoded).expect("deserialize view");
        assert_eq!(decoded.turn, Side::Attacker);
        assert_eq!(decoded.timelines.len(), 1);
        assert!(!decoded.legal_moves.is_empty());
    }
}
