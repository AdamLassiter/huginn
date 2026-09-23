use huginn_core::{Game, PlayerAction, Side};
use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::encoding::encode;
use crate::network::{PolicyValueEvaluator, PolicyValueNetwork};

#[derive(Clone, Copy, Debug)]
pub struct SearchConfig {
    pub simulations: usize,
    pub exploration: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_fraction: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 96,
            exploration: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_fraction: 0.25,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub actions: Vec<PlayerAction>,
    pub visits: Vec<u32>,
    pub policy: Vec<f32>,
    pub root_value: f32,
}

impl SearchResult {
    #[must_use]
    pub fn best_action(&self) -> Option<PlayerAction> {
        self.visits
            .iter()
            .enumerate()
            .max_by_key(|(_, visits)| *visits)
            .map(|(index, _)| self.actions[index])
    }

    #[must_use]
    pub fn sample_action(&self, temperature: f32, rng: &mut impl Rng) -> Option<PlayerAction> {
        if self.actions.is_empty() {
            return None;
        }
        if temperature <= 0.01 {
            return self.best_action();
        }
        let inverse_temperature = temperature.recip();
        let weights = self
            .visits
            .iter()
            .map(|visits| (*visits as f32).max(1.0).powf(inverse_temperature))
            .collect::<Vec<_>>();
        let choice = sample_weighted(&weights, rng);
        Some(self.actions[choice])
    }
}

pub struct Mcts<'a, E: PolicyValueEvaluator + ?Sized = PolicyValueNetwork> {
    network: &'a E,
    config: SearchConfig,
}

struct Node {
    side: Side,
    expanded: bool,
    edges: Vec<Edge>,
}

struct Edge {
    action: PlayerAction,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<Box<Node>>,
}

impl<'a, E: PolicyValueEvaluator + ?Sized> Mcts<'a, E> {
    #[must_use]
    pub const fn new(network: &'a E, config: SearchConfig) -> Self {
        Self { network, config }
    }

    #[must_use]
    pub fn search(&self, game: &Game, add_noise: bool, rng: &mut impl Rng) -> SearchResult {
        if let Some(outcome) = game.outcome() {
            return SearchResult {
                actions: Vec::new(),
                visits: Vec::new(),
                policy: Vec::new(),
                root_value: outcome_value(outcome.winner, game.turn()),
            };
        }
        let mut root = Node::new(game.turn());
        let root_value = self.expand(&mut root, game);
        if add_noise {
            add_dirichlet_noise(&mut root.edges, self.config, rng);
        }
        for _ in 0..self.config.simulations {
            self.simulate(&mut root, game);
        }
        let actions = root
            .edges
            .iter()
            .map(|edge| edge.action)
            .collect::<Vec<_>>();
        let visits = root
            .edges
            .iter()
            .map(|edge| edge.visits)
            .collect::<Vec<_>>();
        let total = visits.iter().sum::<u32>();
        let policy = if total == 0 {
            vec![1.0 / visits.len().max(1) as f32; visits.len()]
        } else {
            visits
                .iter()
                .map(|visits| *visits as f32 / total as f32)
                .collect()
        };
        SearchResult {
            actions,
            visits,
            policy,
            root_value,
        }
    }

    fn simulate(&self, node: &mut Node, game: &Game) -> f32 {
        if let Some(outcome) = game.outcome() {
            return outcome_value(outcome.winner, node.side);
        }
        if !node.expanded {
            return self.expand(node, game);
        }
        if node.edges.is_empty() {
            return -1.0;
        }
        let total_visits = node.edges.iter().map(|edge| edge.visits).sum::<u32>();
        let edge_index = node
            .edges
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                puct(left, total_visits, self.config.exploration).total_cmp(&puct(
                    right,
                    total_visits,
                    self.config.exploration,
                ))
            })
            .map_or(0, |(index, _)| index);
        let edge = &mut node.edges[edge_index];
        let mut next = game.clone();
        if next.apply_action(edge.action).is_err() {
            edge.prior = 0.0;
            edge.visits += 1;
            edge.value_sum -= 1.0;
            return -1.0;
        }
        let child = edge
            .child
            .get_or_insert_with(|| Box::new(Node::new(next.turn())));
        let child_value = self.simulate(child, &next);
        let value = if child.side == node.side {
            child_value
        } else {
            -child_value
        };
        edge.visits += 1;
        edge.value_sum += value;
        value
    }

    fn expand(&self, node: &mut Node, game: &Game) -> f32 {
        let actions = game.legal_actions();
        let prediction = self.network.predict(&encode(game, &actions));
        node.expanded = true;
        node.edges = actions
            .into_iter()
            .zip(prediction.policy)
            .map(|(action, prior)| Edge {
                action,
                prior,
                visits: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();
        prediction.value
    }
}

impl Node {
    const fn new(side: Side) -> Self {
        Self {
            side,
            expanded: false,
            edges: Vec::new(),
        }
    }
}

fn puct(edge: &Edge, total_visits: u32, exploration: f32) -> f32 {
    let mean = if edge.visits == 0 {
        0.0
    } else {
        edge.value_sum / edge.visits as f32
    };
    mean + exploration * edge.prior * ((total_visits + 1) as f32).sqrt() / (edge.visits + 1) as f32
}

fn outcome_value(winner: Side, perspective: Side) -> f32 {
    if winner == perspective { 1.0 } else { -1.0 }
}

fn add_dirichlet_noise(edges: &mut [Edge], config: SearchConfig, rng: &mut impl Rng) {
    if edges.is_empty() || config.dirichlet_fraction <= 0.0 {
        return;
    }
    let Ok(gamma) = Gamma::new(f64::from(config.dirichlet_alpha.max(0.001)), 1.0) else {
        return;
    };
    let mut noise = (0..edges.len())
        .map(|_| gamma.sample(rng) as f32)
        .collect::<Vec<_>>();
    let total = noise.iter().sum::<f32>();
    for value in &mut noise {
        *value /= total;
    }
    for (edge, value) in edges.iter_mut().zip(noise) {
        edge.prior =
            (1.0 - config.dirichlet_fraction) * edge.prior + config.dirichlet_fraction * value;
    }
}

fn sample_weighted(weights: &[f32], rng: &mut impl Rng) -> usize {
    let total = weights.iter().sum::<f32>();
    let mut threshold = rng.random_range(0.0..total);
    for (index, weight) in weights.iter().copied().enumerate() {
        threshold -= weight;
        if threshold <= 0.0 {
            return index;
        }
    }
    weights.len().saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use huginn_core::{Game, Ruleset};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::network::NetworkConfig;

    use super::*;

    #[test]
    fn search_returns_legal_action_and_accounts_for_every_simulation() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let network = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let game = Game::new(Ruleset::Classic);
        let search = Mcts::new(
            &network,
            SearchConfig {
                simulations: 12,
                ..SearchConfig::default()
            },
        );
        let result = search.search(&game, false, &mut rng);
        assert_eq!(result.visits.iter().sum::<u32>(), 12);
        let action = result.best_action().expect("best action");
        let mut next = game.clone();
        next.apply_action(action).expect("legal search action");
    }

    #[test]
    fn multiverse_search_handles_an_action_without_a_side_change() {
        let mut rng = ChaCha8Rng::seed_from_u64(19);
        let network = PolicyValueNetwork::random(NetworkConfig::tiny(), &mut rng);
        let game = Game::new(Ruleset::Multiverse);
        let result = Mcts::new(
            &network,
            SearchConfig {
                simulations: 4,
                ..SearchConfig::default()
            },
        )
        .search(&game, false, &mut rng);
        assert!(result.best_action().is_some());
    }
}
