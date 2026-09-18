use std::collections::HashMap;
use std::fmt::Display;
use std::sync::{Arc, Mutex};

use argon2::Argon2;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use huginn_core::{AiPlayer, GameView, Move, PlayerAction, Ruleset, Side};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::db::{Database, GameSummary, MoveRecord, StoredGame, UserRecord};
use crate::web;

type BotMap = HashMap<i64, Box<dyn AiPlayer>>;

#[derive(Clone)]
pub struct AppState {
    database: Database,
    bots: Arc<Mutex<BotMap>>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RegisterBody {
    username: String,
    display_name: Option<String>,
    password: String,
}

#[derive(Deserialize)]
struct LoginBody {
    username: String,
    password: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionResponse {
    token: String,
    user: UserRecord,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateGameBody {
    opponent: String,
    ruleset: String,
    play_as: String,
}

#[derive(Deserialize)]
struct MoveBody {
    version: i64,
    movement: Move,
}

#[derive(Deserialize)]
struct VersionBody {
    version: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GameDetail {
    summary: GameSummary,
    game: GameView,
    viewer_side: Option<Side>,
    history: Vec<MoveRecord>,
}

impl AppState {
    pub(crate) fn new(
        database: Database,
        players: Vec<Box<dyn AiPlayer>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut bots = HashMap::new();
        for player in players {
            let profile = player.profile();
            let user = database.ensure_bot(profile.username, profile.display_name)?;
            bots.insert(user.id, player);
        }
        Ok(Self {
            database,
            bots: Arc::new(Mutex::new(bots)),
        })
    }
}

pub(crate) fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(web::index))
        .route("/app.js", get(web::app))
        .route("/styles.css", get(web::styles))
        .route("/favicon.svg", get(web::favicon))
        .route("/api/health", get(health))
        .route("/api/auth/register", post(register))
        .route("/api/auth/login", post(login))
        .route("/api/auth/logout", post(logout))
        .route("/api/me", get(me))
        .route("/api/users", get(users))
        .route("/api/leaderboard", get(leaderboard))
        .route("/api/games", get(games).post(create_game))
        .route("/api/games/{id}", get(get_game))
        .route("/api/games/{id}/moves", post(play_move))
        .route("/api/games/{id}/submit", post(submit_turn))
        .fallback(web::not_found)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({ "ok": true, "version": env!("CARGO_PKG_VERSION") }))
}

async fn register(
    State(state): State<AppState>,
    Json(body): Json<RegisterBody>,
) -> Result<Json<SessionResponse>, ApiError> {
    let username = normalize_username(&body.username)?;
    validate_password(&body.password)?;
    let display_name = body
        .display_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or(&username);
    if display_name.chars().count() > 40 {
        return Err(ApiError::bad_request("display name is too long"));
    }
    let salt = SaltString::generate(&mut OsRng);
    let password_hash = Argon2::default()
        .hash_password(body.password.as_bytes(), &salt)
        .map_err(ApiError::internal)?
        .to_string();
    let user = state
        .database
        .create_human(&username, display_name, &password_hash)
        .map_err(|error| {
            if error.to_string().contains("UNIQUE constraint failed") {
                ApiError::conflict("that username is already registered")
            } else {
                ApiError::internal(error)
            }
        })?;
    create_session(&state, user)
}

async fn login(
    State(state): State<AppState>,
    Json(body): Json<LoginBody>,
) -> Result<Json<SessionResponse>, ApiError> {
    let username = normalize_username(&body.username)?;
    let record = state
        .database
        .auth_by_username(&username)
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::unauthorized("invalid username or password"))?;
    let encoded = record
        .password_hash
        .as_deref()
        .ok_or_else(|| ApiError::unauthorized("bot accounts cannot sign in"))?;
    let hash = PasswordHash::new(encoded).map_err(ApiError::internal)?;
    Argon2::default()
        .verify_password(body.password.as_bytes(), &hash)
        .map_err(|_| ApiError::unauthorized("invalid username or password"))?;
    create_session(&state, record.user)
}

fn create_session(state: &AppState, user: UserRecord) -> Result<Json<SessionResponse>, ApiError> {
    let token = Uuid::new_v4().to_string();
    state
        .database
        .create_session(user.id, &token)
        .map_err(ApiError::internal)?;
    Ok(Json(SessionResponse { token, user }))
}

async fn logout(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let token = bearer_token(&headers)?;
    state
        .database
        .delete_session(token)
        .map_err(ApiError::internal)?;
    Ok(Json(json!({ "ok": true })))
}

async fn me(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<UserRecord>, ApiError> {
    Ok(Json(authenticated_user(&state, &headers)?))
}

async fn users(State(state): State<AppState>) -> Result<Json<Vec<UserRecord>>, ApiError> {
    Ok(Json(
        state.database.leaderboard().map_err(ApiError::internal)?,
    ))
}

async fn leaderboard(State(state): State<AppState>) -> Result<Json<Vec<UserRecord>>, ApiError> {
    users(State(state)).await
}

async fn games(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<GameSummary>>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    Ok(Json(
        state
            .database
            .games_for_user(user.id)
            .map_err(ApiError::internal)?,
    ))
}

async fn create_game(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<CreateGameBody>,
) -> Result<Json<GameDetail>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    let opponent_name = normalize_username(&body.opponent)?;
    let opponent = state
        .database
        .auth_by_username(&opponent_name)
        .map_err(ApiError::internal)?
        .map(|record| record.user)
        .ok_or_else(|| ApiError::not_found("opponent not found"))?;
    if opponent.id == user.id {
        return Err(ApiError::bad_request("choose a different opponent"));
    }
    let ruleset = parse_ruleset(&body.ruleset)?;
    let (attacker, defender) = match body.play_as.as_str() {
        "attacker" => (user.id, opponent.id),
        "defender" => (opponent.id, user.id),
        _ => return Err(ApiError::bad_request("playAs must be attacker or defender")),
    };
    let game = state
        .database
        .create_game(ruleset, attacker, defender)
        .map_err(ApiError::internal)?;
    let game = advance_bots(&state, game)?;
    Ok(Json(game_detail(&state, &game, user.id)?))
}

async fn get_game(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
) -> Result<Json<GameDetail>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    let game = load_participating_game(&state, id, user.id)?;
    Ok(Json(game_detail(&state, &game, user.id)?))
}

async fn play_move(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
    Json(body): Json<MoveBody>,
) -> Result<Json<GameDetail>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    let mut stored = load_participating_game(&state, id, user.id)?;
    ensure_actor(&stored, &user, body.version)?;
    stored
        .game
        .apply_move(body.movement)
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let action = PlayerAction::Move {
        movement: body.movement,
    };
    stored.version = state
        .database
        .save_action(
            stored.id,
            stored.version,
            user.id,
            &serde_json::to_string(&action).map_err(ApiError::internal)?,
            &stored.game,
        )
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    let stored = reload_and_advance_bots(&state, stored.id)?;
    Ok(Json(game_detail(&state, &stored, user.id)?))
}

async fn submit_turn(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<i64>,
    Json(body): Json<VersionBody>,
) -> Result<Json<GameDetail>, ApiError> {
    let user = authenticated_user(&state, &headers)?;
    let mut stored = load_participating_game(&state, id, user.id)?;
    ensure_actor(&stored, &user, body.version)?;
    stored
        .game
        .submit_turn()
        .map_err(|error| ApiError::bad_request(error.to_string()))?;
    let action = PlayerAction::SubmitTurn;
    stored.version = state
        .database
        .save_action(
            stored.id,
            stored.version,
            user.id,
            &serde_json::to_string(&action).map_err(ApiError::internal)?,
            &stored.game,
        )
        .map_err(|error| ApiError::conflict(error.to_string()))?;
    let stored = reload_and_advance_bots(&state, stored.id)?;
    Ok(Json(game_detail(&state, &stored, user.id)?))
}

fn ensure_actor(stored: &StoredGame, user: &UserRecord, version: i64) -> Result<(), ApiError> {
    if stored.version != version {
        return Err(ApiError::conflict("game changed; reload before moving"));
    }
    if stored.status != "active" || stored.game.outcome().is_some() {
        return Err(ApiError::conflict("the game is already finished"));
    }
    if stored.side_user(stored.game.turn()).id != user.id {
        return Err(ApiError::forbidden("it is not your turn"));
    }
    Ok(())
}

fn advance_bots(state: &AppState, mut stored: StoredGame) -> Result<StoredGame, ApiError> {
    for _ in 0..512 {
        if stored.game.outcome().is_some() {
            break;
        }
        let actor = stored.side_user(stored.game.turn()).clone();
        if !actor.is_bot {
            break;
        }
        let action = {
            let mut bots = state.bots.lock().expect("bot mutex poisoned");
            let bot = bots
                .get_mut(&actor.id)
                .ok_or_else(|| ApiError::internal("bot account has no loaded implementation"))?;
            bot.choose_action(&GameView::from_game(&stored.game))
                .map_err(ApiError::internal)?
        };
        match action {
            PlayerAction::Move { movement } => {
                stored.game.apply_move(movement).map_err(|error| {
                    ApiError::internal(format!("bot returned illegal move: {error}"))
                })?;
            }
            PlayerAction::SubmitTurn => stored.game.submit_turn().map_err(|error| {
                ApiError::internal(format!("bot returned illegal submission: {error}"))
            })?,
        }
        stored.version = state
            .database
            .save_action(
                stored.id,
                stored.version,
                actor.id,
                &serde_json::to_string(&action).map_err(ApiError::internal)?,
                &stored.game,
            )
            .map_err(ApiError::internal)?;
    }
    state
        .database
        .game_by_id(stored.id)
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::not_found("game disappeared"))
}

fn reload_and_advance_bots(state: &AppState, id: i64) -> Result<StoredGame, ApiError> {
    let stored = state
        .database
        .game_by_id(id)
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::not_found("game not found"))?;
    advance_bots(state, stored)
}

fn game_detail(
    state: &AppState,
    game: &StoredGame,
    viewer_id: i64,
) -> Result<GameDetail, ApiError> {
    let viewer_side = if game.attacker.id == viewer_id {
        Some(Side::Attacker)
    } else if game.defender.id == viewer_id {
        Some(Side::Defender)
    } else {
        None
    };
    let history = state
        .database
        .moves_for_game(game.id)
        .map_err(ApiError::internal)?;
    Ok(GameDetail {
        summary: game.summary(),
        game: GameView::from_game(&game.game),
        viewer_side,
        history,
    })
}

fn load_participating_game(
    state: &AppState,
    id: i64,
    user_id: i64,
) -> Result<StoredGame, ApiError> {
    let game = state
        .database
        .game_by_id(id)
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::not_found("game not found"))?;
    if game.attacker.id != user_id && game.defender.id != user_id {
        return Err(ApiError::forbidden("you are not a player in this game"));
    }
    Ok(game)
}

fn authenticated_user(state: &AppState, headers: &HeaderMap) -> Result<UserRecord, ApiError> {
    state
        .database
        .user_for_session(bearer_token(headers)?)
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::unauthorized("session is invalid"))
}

fn bearer_token(headers: &HeaderMap) -> Result<&str, ApiError> {
    headers
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .filter(|token| !token.is_empty())
        .ok_or_else(|| ApiError::unauthorized("sign in first"))
}

fn normalize_username(value: &str) -> Result<String, ApiError> {
    let username = value.trim().to_ascii_lowercase();
    if !(3..=24).contains(&username.len())
        || !username
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(ApiError::bad_request(
            "username must be 3-24 letters, numbers, hyphens, or underscores",
        ));
    }
    Ok(username)
}

fn validate_password(password: &str) -> Result<(), ApiError> {
    if password.len() < 8 || password.len() > 128 {
        return Err(ApiError::bad_request("password must be 8-128 characters"));
    }
    Ok(())
}

fn parse_ruleset(value: &str) -> Result<Ruleset, ApiError> {
    match value {
        "classic" => Ok(Ruleset::Classic),
        "multiverse" => Ok(Ruleset::Multiverse),
        _ => Err(ApiError::bad_request("unknown ruleset")),
    }
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, message)
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, message)
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self::new(StatusCode::CONFLICT, message)
    }

    fn internal(error: impl Display) -> Self {
        tracing::error!(%error, "request failed");
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal server error")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "error": self.message }))).into_response()
    }
}

#[cfg(test)]
mod tests {
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use super::*;
    use huginn_random_bot::RavenBot;

    async fn request(
        app: &Router,
        method: &str,
        uri: &str,
        token: Option<&str>,
        body: Option<Value>,
    ) -> (StatusCode, Value) {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(token) = token {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        let request = if let Some(body) = body {
            builder
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request")
        } else {
            builder.body(Body::empty()).expect("request")
        };
        let response = app.clone().oneshot(request).await.expect("response");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    #[tokio::test]
    async fn account_game_and_authoritative_bot_reply_work_end_to_end() {
        let state = AppState::new(
            Database::in_memory().expect("database"),
            vec![Box::new(RavenBot::default())],
        )
        .expect("state");
        let app = router(state);

        let (status, session) = request(
            &app,
            "POST",
            "/api/auth/register",
            None,
            Some(json!({
                "username": "alice",
                "displayName": "Alice",
                "password": "correct horse battery staple"
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let token = session["token"].as_str().expect("token");

        let (status, created) = request(
            &app,
            "POST",
            "/api/games",
            Some(token),
            Some(json!({
                "opponent": "bot-raven",
                "ruleset": "classic",
                "playAs": "attacker"
            })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(created["summary"]["version"], 0);
        let game_id = created["summary"]["id"].as_i64().expect("game id");
        let movement = created["game"]["legalMoves"][0].clone();

        let (status, updated) = request(
            &app,
            "POST",
            &format!("/api/games/{game_id}/moves"),
            Some(token),
            Some(json!({ "version": 0, "movement": movement })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(updated["summary"]["version"], 2);
        assert_eq!(updated["history"].as_array().expect("history").len(), 2);
        assert_eq!(updated["game"]["turn"], "Attacker");

        let (status, stale_response) = request(
            &app,
            "POST",
            &format!("/api/games/{game_id}/moves"),
            Some(token),
            Some(json!({ "version": 0, "movement": movement })),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(stale_response["error"].as_str().is_some());
    }

    #[tokio::test]
    async fn two_human_accounts_share_one_authoritative_game() {
        let state =
            AppState::new(Database::in_memory().expect("database"), Vec::new()).expect("state");
        let app = router(state);

        let (_, alice) = request(
            &app,
            "POST",
            "/api/auth/register",
            None,
            Some(json!({ "username": "alice2", "password": "alice-password" })),
        )
        .await;
        let (_, bob) = request(
            &app,
            "POST",
            "/api/auth/register",
            None,
            Some(json!({ "username": "bob2", "password": "bob-password" })),
        )
        .await;
        let alice_token = alice["token"].as_str().expect("alice token");
        let bob_token = bob["token"].as_str().expect("bob token");
        let (_, created) = request(
            &app,
            "POST",
            "/api/games",
            Some(alice_token),
            Some(json!({
                "opponent": "bob2",
                "ruleset": "classic",
                "playAs": "attacker"
            })),
        )
        .await;
        let game_id = created["summary"]["id"].as_i64().expect("game id");
        let attacker_move = created["game"]["legalMoves"][0].clone();

        let (status, bob_view) = request(
            &app,
            "GET",
            &format!("/api/games/{game_id}"),
            Some(bob_token),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(bob_view["viewerSide"], "Defender");

        let (status, _) = request(
            &app,
            "POST",
            &format!("/api/games/{game_id}/moves"),
            Some(bob_token),
            Some(json!({ "version": 0, "movement": attacker_move })),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);

        let (status, after_alice) = request(
            &app,
            "POST",
            &format!("/api/games/{game_id}/moves"),
            Some(alice_token),
            Some(json!({ "version": 0, "movement": attacker_move })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(after_alice["game"]["turn"], "Defender");
        let defender_move = after_alice["game"]["legalMoves"][0].clone();

        let (status, after_bob) = request(
            &app,
            "POST",
            &format!("/api/games/{game_id}/moves"),
            Some(bob_token),
            Some(json!({ "version": 1, "movement": defender_move })),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(after_bob["summary"]["version"], 2);
        assert_eq!(after_bob["game"]["turn"], "Attacker");
        assert_eq!(after_bob["history"].as_array().expect("history").len(), 2);
    }
}
