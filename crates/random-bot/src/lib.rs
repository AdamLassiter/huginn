use huginn_core::{AiPlayer, AiProfile, GameView, PlayerAction};

/// A tiny deterministic baseline used to prove the pluggable-AI boundary.
pub struct RavenBot {
    state: u64,
}

impl Default for RavenBot {
    fn default() -> Self {
        Self {
            state: 0x6875_6769_6e6e,
        }
    }
}

impl RavenBot {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_index(&mut self, length: usize) -> usize {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        usize::try_from(self.state % u64::try_from(length).expect("length fits in u64"))
            .expect("modulo result fits in usize")
    }
}

impl AiPlayer for RavenBot {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-raven",
            display_name: "Raven (baseline)",
        }
    }

    fn choose_action(&mut self, game: &GameView) -> Result<PlayerAction, String> {
        if game.can_submit {
            return Ok(PlayerAction::SubmitTurn);
        }
        let Some(movement) = game
            .legal_moves
            .get(self.next_index(game.legal_moves.len().max(1)))
            .copied()
        else {
            return Err("no legal action is available".to_owned());
        };
        Ok(PlayerAction::Move { movement })
    }
}

#[cfg(test)]
mod tests {
    use huginn_core::{Game, Ruleset};

    use super::*;

    #[test]
    fn bot_returns_an_authoritatively_legal_opening() {
        let game = Game::new(Ruleset::Classic);
        let mut bot = RavenBot::default();
        let PlayerAction::Move { movement } = bot
            .choose_action(&GameView::from_game(&game))
            .expect("bot action")
        else {
            panic!("classic opening cannot be a submission");
        };
        game.validate_move(movement).expect("legal bot move");
    }
}
