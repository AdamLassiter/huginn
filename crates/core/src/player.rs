use serde::{Deserialize, Serialize};

use crate::{BoardCoordinate, Game, GameOutcome, Move, Ruleset, Side, Timeline};

/// Read-only state supplied to human frontends.
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

impl Game {
    /// Returns every action currently accepted by the authoritative engine.
    #[must_use]
    pub fn legal_actions(&self) -> Vec<PlayerAction> {
        let mut actions = self
            .legal_moves()
            .into_iter()
            .map(|movement| PlayerAction::Move { movement })
            .collect::<Vec<_>>();
        if self.can_submit() {
            actions.push(PlayerAction::SubmitTurn);
        }
        actions
    }

    /// Applies one player action through the normal rules validation path.
    ///
    /// # Errors
    ///
    /// Returns the same [`crate::MoveError`] produced by the underlying move
    /// or multiverse turn-submission operation.
    pub fn apply_action(&mut self, action: PlayerAction) -> Result<(), crate::MoveError> {
        match action {
            PlayerAction::Move { movement } => self.apply_move(movement),
            PlayerAction::SubmitTurn => self.submit_turn(),
        }
    }
}

/// Interface implemented by separately packaged AI players.
///
/// The server remains authoritative: it validates the returned action through
/// [`Game`] before committing it.
pub trait AiPlayer: Send {
    fn profile(&self) -> AiProfile;

    /// Selects one move or turn submission from an immutable position.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic if the implementation cannot select an action.
    fn choose_action(&mut self, game: &Game) -> Result<PlayerAction, String>;
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

    #[test]
    fn legal_actions_can_be_applied_to_a_clone() {
        let game = Game::new(Ruleset::Classic);
        let action = game.legal_actions()[0];
        let mut next = game.clone();
        next.apply_action(action).expect("legal action");
        assert_eq!(next.turn(), Side::Defender);
    }
}
