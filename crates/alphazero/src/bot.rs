use std::path::Path;

use huginn_core::{AiPlayer, AiProfile, Game, PlayerAction};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::network::{ModelError, NetworkConfig, PolicyValueNetwork};
use crate::search::{Mcts, SearchConfig};

pub struct MuninnBot {
    model: PolicyValueNetwork,
    search: SearchConfig,
    rng: ChaCha8Rng,
}

impl MuninnBot {
    #[must_use]
    pub fn new(model: PolicyValueNetwork, search: SearchConfig, seed: u64) -> Self {
        Self {
            model,
            search,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Loads a trained checkpoint for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint cannot be read, decoded, or does not
    /// match the supported network format.
    pub fn load(
        path: impl AsRef<Path>,
        search: SearchConfig,
        seed: u64,
    ) -> Result<Self, ModelError> {
        Ok(Self::new(PolicyValueNetwork::load(path)?, search, seed))
    }

    /// Creates an untrained network, useful only as the start of self-play.
    #[must_use]
    pub fn bootstrap(search: SearchConfig, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let model = PolicyValueNetwork::random(NetworkConfig::default(), &mut rng);
        Self::new(model, search, seed ^ 0xa17e_20a0)
    }
}

impl AiPlayer for MuninnBot {
    fn profile(&self) -> AiProfile {
        AiProfile {
            username: "bot-muninn",
            display_name: "Muninn (AlphaZero)",
        }
    }

    fn choose_action(&mut self, game: &Game) -> Result<PlayerAction, String> {
        Mcts::new(&self.model, self.search)
            .search(game, false, &mut self.rng)
            .best_action()
            .ok_or_else(|| "no legal action is available".to_owned())
    }
}

#[cfg(test)]
mod tests {
    use huginn_core::{Game, Ruleset};

    use super::*;

    #[test]
    fn muninn_returns_an_authoritatively_legal_action() {
        let mut bot = MuninnBot::bootstrap(
            SearchConfig {
                simulations: 3,
                ..SearchConfig::default()
            },
            31,
        );
        let game = Game::new(Ruleset::Classic);
        let action = bot.choose_action(&game).expect("bot action");
        let mut next = game.clone();
        next.apply_action(action).expect("legal bot action");
    }
}
