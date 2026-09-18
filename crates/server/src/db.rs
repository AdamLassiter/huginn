use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use huginn_core::{Game, GameOutcome, Ruleset, Side};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::Serialize;

#[derive(Clone)]
pub(crate) struct Database {
    connection: Arc<Mutex<Connection>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UserRecord {
    pub id: i64,
    pub username: String,
    pub display_name: String,
    pub is_bot: bool,
    pub rating: i32,
    pub games_played: i32,
}

#[derive(Clone, Debug)]
pub(crate) struct AuthRecord {
    pub user: UserRecord,
    pub password_hash: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StoredGame {
    pub id: i64,
    pub ruleset: Ruleset,
    pub status: String,
    pub attacker: UserRecord,
    pub defender: UserRecord,
    pub winner_id: Option<i64>,
    pub outcome_reason: Option<String>,
    pub game: Game,
    pub version: i64,
    pub created_at: i64,
    pub updated_at: i64,
    pub finished_at: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GameSummary {
    pub id: i64,
    pub ruleset: String,
    pub status: String,
    pub attacker: UserRecord,
    pub defender: UserRecord,
    pub winner_id: Option<i64>,
    pub outcome_reason: Option<String>,
    pub version: i64,
    pub created_at: i64,
    pub updated_at: i64,
    pub finished_at: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MoveRecord {
    pub sequence: i64,
    pub actor: UserRecord,
    pub action: serde_json::Value,
    pub created_at: i64,
}

impl Database {
    pub fn open(path: impl AsRef<Path>) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        let database = Self {
            connection: Arc::new(Mutex::new(connection)),
        };
        database.migrate()?;
        Ok(database)
    }

    #[cfg(test)]
    pub fn in_memory() -> rusqlite::Result<Self> {
        Self::open(":memory:")
    }

    fn migrate(&self) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("database mutex poisoned")
            .execute_batch(
                "
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT,
                is_bot INTEGER NOT NULL DEFAULT 0,
                rating INTEGER NOT NULL DEFAULT 1500,
                games_played INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY,
                ruleset TEXT NOT NULL,
                status TEXT NOT NULL,
                attacker_id INTEGER NOT NULL REFERENCES users(id),
                defender_id INTEGER NOT NULL REFERENCES users(id),
                winner_id INTEGER REFERENCES users(id),
                outcome_reason TEXT,
                state_json TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                rated INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                finished_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY,
                game_id INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                actor_id INTEGER NOT NULL REFERENCES users(id),
                action_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(game_id, sequence)
            );
            CREATE INDEX IF NOT EXISTS games_players ON games(attacker_id, defender_id, updated_at);
            CREATE INDEX IF NOT EXISTS moves_game ON moves(game_id, sequence);
            ",
            )
    }

    pub fn create_human(
        &self,
        username: &str,
        display_name: &str,
        password_hash: &str,
    ) -> rusqlite::Result<UserRecord> {
        let connection = self.connection.lock().expect("database mutex poisoned");
        connection.execute(
            "INSERT INTO users (username, display_name, password_hash, is_bot, created_at)
             VALUES (?1, ?2, ?3, 0, ?4)",
            params![username, display_name, password_hash, now()],
        )?;
        Self::user_by_id_on(&connection, connection.last_insert_rowid())?
            .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows)
    }

    pub fn ensure_bot(&self, username: &str, display_name: &str) -> rusqlite::Result<UserRecord> {
        let connection = self.connection.lock().expect("database mutex poisoned");
        connection.execute(
            "INSERT INTO users (username, display_name, is_bot, created_at)
             VALUES (?1, ?2, 1, ?3)
             ON CONFLICT(username) DO UPDATE SET display_name = excluded.display_name, is_bot = 1",
            params![username, display_name, now()],
        )?;
        Self::auth_by_username_on(&connection, username)?
            .map(|record| record.user)
            .ok_or(rusqlite::Error::QueryReturnedNoRows)
    }

    pub fn auth_by_username(&self, username: &str) -> rusqlite::Result<Option<AuthRecord>> {
        Self::auth_by_username_on(
            &self.connection.lock().expect("database mutex poisoned"),
            username,
        )
    }

    fn auth_by_username_on(
        connection: &Connection,
        username: &str,
    ) -> rusqlite::Result<Option<AuthRecord>> {
        connection
            .query_row(
                "SELECT id, username, display_name, is_bot, rating, games_played, password_hash
                 FROM users WHERE username = ?1",
                [username],
                |row| {
                    Ok(AuthRecord {
                        user: user_from_row(row)?,
                        password_hash: row.get(6)?,
                    })
                },
            )
            .optional()
    }

    fn user_by_id_on(connection: &Connection, id: i64) -> rusqlite::Result<Option<UserRecord>> {
        connection
            .query_row(
                "SELECT id, username, display_name, is_bot, rating, games_played
                 FROM users WHERE id = ?1",
                [id],
                user_from_row,
            )
            .optional()
    }

    pub fn create_session(&self, user_id: i64, token: &str) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("database mutex poisoned")
            .execute(
                "INSERT INTO sessions (token, user_id, created_at) VALUES (?1, ?2, ?3)",
                params![token, user_id, now()],
            )?;
        Ok(())
    }

    pub fn delete_session(&self, token: &str) -> rusqlite::Result<()> {
        self.connection
            .lock()
            .expect("database mutex poisoned")
            .execute("DELETE FROM sessions WHERE token = ?1", [token])?;
        Ok(())
    }

    pub fn user_for_session(&self, token: &str) -> rusqlite::Result<Option<UserRecord>> {
        self.connection
            .lock()
            .expect("database mutex poisoned")
            .query_row(
                "SELECT u.id, u.username, u.display_name, u.is_bot, u.rating, u.games_played
                 FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?1",
                [token],
                user_from_row,
            )
            .optional()
    }

    pub fn leaderboard(&self) -> rusqlite::Result<Vec<UserRecord>> {
        let connection = self.connection.lock().expect("database mutex poisoned");
        let mut statement = connection.prepare(
            "SELECT id, username, display_name, is_bot, rating, games_played
             FROM users ORDER BY rating DESC, games_played DESC, username LIMIT 100",
        )?;
        statement.query_map([], user_from_row)?.collect()
    }

    pub fn create_game(
        &self,
        ruleset: Ruleset,
        attacker_id: i64,
        defender_id: i64,
    ) -> Result<StoredGame, Box<dyn std::error::Error>> {
        let game = Game::new(ruleset);
        let state_json = serde_json::to_string(&game)?;
        let timestamp = now();
        let connection = self.connection.lock().expect("database mutex poisoned");
        connection.execute(
            "INSERT INTO games
             (ruleset, status, attacker_id, defender_id, state_json, created_at, updated_at)
             VALUES (?1, 'active', ?2, ?3, ?4, ?5, ?5)",
            params![
                ruleset_name(ruleset),
                attacker_id,
                defender_id,
                state_json,
                timestamp
            ],
        )?;
        Self::game_by_id_on(&connection, connection.last_insert_rowid())?
            .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows.into())
    }

    pub fn game_by_id(&self, id: i64) -> Result<Option<StoredGame>, Box<dyn std::error::Error>> {
        Self::game_by_id_on(
            &self.connection.lock().expect("database mutex poisoned"),
            id,
        )
    }

    fn game_by_id_on(
        connection: &Connection,
        id: i64,
    ) -> Result<Option<StoredGame>, Box<dyn std::error::Error>> {
        let raw = connection
            .query_row(
                "SELECT id, ruleset, status, attacker_id, defender_id, winner_id,
                        outcome_reason, state_json, version, created_at, updated_at, finished_at
                 FROM games WHERE id = ?1",
                [id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, Option<i64>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, i64>(9)?,
                        row.get::<_, i64>(10)?,
                        row.get::<_, Option<i64>>(11)?,
                    ))
                },
            )
            .optional()?;
        let Some(raw) = raw else { return Ok(None) };
        let ruleset = parse_ruleset(&raw.1)?;
        let game = serde_json::from_str(&raw.7)?;
        let attacker =
            Self::user_by_id_on(connection, raw.3)?.ok_or(rusqlite::Error::QueryReturnedNoRows)?;
        let defender =
            Self::user_by_id_on(connection, raw.4)?.ok_or(rusqlite::Error::QueryReturnedNoRows)?;
        Ok(Some(StoredGame {
            id: raw.0,
            ruleset,
            status: raw.2,
            attacker,
            defender,
            winner_id: raw.5,
            outcome_reason: raw.6,
            game,
            version: raw.8,
            created_at: raw.9,
            updated_at: raw.10,
            finished_at: raw.11,
        }))
    }

    pub fn games_for_user(
        &self,
        user_id: i64,
    ) -> Result<Vec<GameSummary>, Box<dyn std::error::Error>> {
        let connection = self.connection.lock().expect("database mutex poisoned");
        let mut statement = connection.prepare(
            "SELECT id FROM games WHERE attacker_id = ?1 OR defender_id = ?1
             ORDER BY updated_at DESC LIMIT 100",
        )?;
        let ids: Vec<i64> = statement
            .query_map([user_id], |row| row.get(0))?
            .collect::<Result<_, _>>()?;
        ids.into_iter()
            .map(|id| {
                Self::game_by_id_on(&connection, id)?.map_or_else(
                    || Err(rusqlite::Error::QueryReturnedNoRows.into()),
                    |game| Ok(game.summary()),
                )
            })
            .collect()
    }

    pub fn moves_for_game(
        &self,
        game_id: i64,
    ) -> Result<Vec<MoveRecord>, Box<dyn std::error::Error>> {
        let connection = self.connection.lock().expect("database mutex poisoned");
        let mut statement = connection.prepare(
            "SELECT sequence, actor_id, action_json, created_at FROM moves
             WHERE game_id = ?1 ORDER BY sequence",
        )?;
        let raw: Vec<(i64, i64, String, i64)> = statement
            .query_map([game_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<Result<_, _>>()?;
        raw.into_iter()
            .map(|(sequence, actor_id, action_json, created_at)| {
                Ok(MoveRecord {
                    sequence,
                    actor: Self::user_by_id_on(&connection, actor_id)?
                        .ok_or(rusqlite::Error::QueryReturnedNoRows)?,
                    action: serde_json::from_str(&action_json)?,
                    created_at,
                })
            })
            .collect()
    }

    pub fn save_action(
        &self,
        game_id: i64,
        expected_version: i64,
        actor_id: i64,
        action_json: &str,
        game: &Game,
    ) -> Result<i64, Box<dyn std::error::Error>> {
        let state_json = serde_json::to_string(game)?;
        let timestamp = now();
        let mut connection = self.connection.lock().expect("database mutex poisoned");
        let transaction = connection.transaction()?;
        let changed = transaction.execute(
            "UPDATE games SET state_json = ?1, version = version + 1, updated_at = ?2
             WHERE id = ?3 AND version = ?4 AND status = 'active'",
            params![state_json, timestamp, game_id, expected_version],
        )?;
        if changed == 0 {
            return Err("game changed in another client; reload and retry".into());
        }
        let next_version = expected_version + 1;
        transaction.execute(
            "INSERT INTO moves (game_id, sequence, actor_id, action_json, state_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                game_id,
                next_version,
                actor_id,
                action_json,
                state_json,
                timestamp
            ],
        )?;
        if let Some(outcome) = game.outcome() {
            finish_and_rate(&transaction, game_id, outcome, timestamp)?;
        }
        transaction.commit()?;
        Ok(next_version)
    }
}

impl StoredGame {
    pub fn side_user(&self, side: Side) -> &UserRecord {
        match side {
            Side::Attacker => &self.attacker,
            Side::Defender => &self.defender,
        }
    }

    pub fn summary(&self) -> GameSummary {
        GameSummary {
            id: self.id,
            ruleset: ruleset_name(self.ruleset).to_owned(),
            status: self.status.clone(),
            attacker: self.attacker.clone(),
            defender: self.defender.clone(),
            winner_id: self.winner_id,
            outcome_reason: self.outcome_reason.clone(),
            version: self.version,
            created_at: self.created_at,
            updated_at: self.updated_at,
            finished_at: self.finished_at,
        }
    }
}

#[allow(clippy::cast_possible_truncation)] // Elo delta is mathematically bounded to -32..=32.
fn finish_and_rate(
    transaction: &Transaction<'_>,
    game_id: i64,
    outcome: GameOutcome,
    timestamp: i64,
) -> rusqlite::Result<()> {
    let (attacker_id, defender_id, rated): (i64, i64, bool) = transaction.query_row(
        "SELECT attacker_id, defender_id, rated FROM games WHERE id = ?1",
        [game_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    if rated {
        return Ok(());
    }
    let attacker_rating: i32 = transaction.query_row(
        "SELECT rating FROM users WHERE id = ?1",
        [attacker_id],
        |row| row.get(0),
    )?;
    let defender_rating: i32 = transaction.query_row(
        "SELECT rating FROM users WHERE id = ?1",
        [defender_id],
        |row| row.get(0),
    )?;
    let expected_attacker =
        1.0 / (1.0 + 10_f64.powf(f64::from(defender_rating - attacker_rating) / 400.0));
    let attacker_score = if outcome.winner == Side::Attacker {
        1.0
    } else {
        0.0
    };
    let delta = (32.0 * (attacker_score - expected_attacker)).round() as i32;
    transaction.execute(
        "UPDATE users SET rating = rating + ?1, games_played = games_played + 1 WHERE id = ?2",
        params![delta, attacker_id],
    )?;
    transaction.execute(
        "UPDATE users SET rating = rating - ?1, games_played = games_played + 1 WHERE id = ?2",
        params![delta, defender_id],
    )?;
    let winner_id = if outcome.winner == Side::Attacker {
        attacker_id
    } else {
        defender_id
    };
    transaction.execute(
        "UPDATE games SET status = 'finished', winner_id = ?1, outcome_reason = ?2,
                          rated = 1, finished_at = ?3, updated_at = ?3 WHERE id = ?4",
        params![
            winner_id,
            format!("{:?}", outcome.reason),
            timestamp,
            game_id
        ],
    )?;
    Ok(())
}

fn user_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<UserRecord> {
    Ok(UserRecord {
        id: row.get(0)?,
        username: row.get(1)?,
        display_name: row.get(2)?,
        is_bot: row.get(3)?,
        rating: row.get(4)?,
        games_played: row.get(5)?,
    })
}

fn ruleset_name(ruleset: Ruleset) -> &'static str {
    match ruleset {
        Ruleset::Classic => "classic",
        Ruleset::Multiverse => "multiverse",
    }
}

fn parse_ruleset(value: &str) -> Result<Ruleset, Box<dyn std::error::Error>> {
    match value {
        "classic" => Ok(Ruleset::Classic),
        "multiverse" => Ok(Ruleset::Multiverse),
        _ => Err(format!("unknown ruleset: {value}").into()),
    }
}

fn now() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after Unix epoch")
            .as_secs(),
    )
    .expect("Unix timestamp fits in i64")
}

#[cfg(test)]
mod tests {
    use huginn_core::{Board, BoardCoordinate, Move, Piece, Position, Square};

    use super::*;

    fn square(x: u8, y: u8) -> Square {
        Square::new(x, y).expect("valid test square")
    }

    #[test]
    fn finished_game_is_historicized_and_rates_both_accounts_once() {
        let database = Database::in_memory().expect("database");
        let attacker = database
            .create_human("attacker", "Attacker", "hash")
            .expect("attacker");
        let defender = database
            .create_human("defender", "Defender", "hash")
            .expect("defender");
        let stored = database
            .create_game(Ruleset::Classic, attacker.id, defender.id)
            .expect("game");

        let mut board = Board::empty();
        board.set(square(0, 1), Some(Piece::King));
        let mut game = Game::from_board(Ruleset::Classic, board, Side::Defender);
        game.apply_move(Move::new(
            Position::new(BoardCoordinate::new(1, 0), square(0, 1)),
            Position::new(BoardCoordinate::new(1, 0), square(0, 0)),
        ))
        .expect("escape move");
        database
            .save_action(stored.id, 0, defender.id, r#"{"type":"move"}"#, &game)
            .expect("save outcome");

        let finished = database
            .game_by_id(stored.id)
            .expect("query")
            .expect("game exists");
        assert_eq!(finished.status, "finished");
        assert_eq!(finished.winner_id, Some(defender.id));
        assert_eq!(
            database
                .auth_by_username("attacker")
                .expect("query")
                .expect("user")
                .user
                .rating,
            1484
        );
        assert_eq!(
            database
                .auth_by_username("defender")
                .expect("query")
                .expect("user")
                .user
                .rating,
            1516
        );
        assert_eq!(
            database.moves_for_game(stored.id).expect("history").len(),
            1
        );
    }

    #[test]
    fn bot_accounts_are_normal_rated_users_without_passwords() {
        let database = Database::in_memory().expect("database");
        let bot = database
            .ensure_bot("bot-test", "Test bot")
            .expect("bot account");
        let human = database
            .create_human("human", "Human", "hash")
            .expect("human account");
        let auth = database
            .auth_by_username("bot-test")
            .expect("query")
            .expect("bot");
        assert!(bot.is_bot);
        assert_eq!(bot.rating, 1500);
        assert!(auth.password_hash.is_none());

        let stored = database
            .create_game(Ruleset::Classic, human.id, bot.id)
            .expect("game");
        let mut board = Board::empty();
        board.set(square(0, 1), Some(Piece::King));
        let mut won = Game::from_board(Ruleset::Classic, board, Side::Defender);
        won.apply_move(Move::new(
            Position::new(BoardCoordinate::new(1, 0), square(0, 1)),
            Position::new(BoardCoordinate::new(1, 0), square(0, 0)),
        ))
        .expect("bot escape");
        database
            .save_action(stored.id, 0, bot.id, r#"{"type":"move"}"#, &won)
            .expect("save bot win");

        let rated_bot = database
            .auth_by_username("bot-test")
            .expect("query")
            .expect("bot")
            .user;
        assert_eq!(rated_bot.rating, 1516);
        assert_eq!(rated_bot.games_played, 1);
    }
}
