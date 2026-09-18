const state = {
  token: localStorage.getItem("huginn-token"),
  user: null,
  users: [],
  current: null,
  selected: null,
  poll: null,
};

const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body && typeof options.body !== "string") {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function show(view) {
  for (const id of ["auth-view", "lobby-view", "game-view"]) {
    $(`#${id}`).hidden = id !== view;
  }
}

function setIdentity() {
  const el = $("#identity");
  if (!state.user) {
    el.replaceChildren();
    return;
  }
  el.innerHTML = `<span><strong>${escapeHtml(state.user.displayName)}</strong><br><small>${state.user.rating} Elo</small></span><button id="logout">Sign out</button>`;
  $("#logout").onclick = logout;
}

async function logout() {
  try { await api("/api/auth/logout", { method: "POST" }); } catch (_) { /* expire locally */ }
  localStorage.removeItem("huginn-token");
  state.token = null;
  state.user = null;
  state.current = null;
  stopPolling();
  setIdentity();
  show("auth-view");
}

async function loadLobby() {
  stopPolling();
  const [games, users, leaders] = await Promise.all([
    api("/api/games"), api("/api/users"), api("/api/leaderboard"),
  ]);
  state.users = users;
  const opponent = $("#new-game-form select[name=opponent]");
  opponent.replaceChildren(...users.filter((user) => user.id !== state.user.id).map((user) => {
    const option = document.createElement("option");
    option.value = user.username;
    option.textContent = `${user.displayName}${user.isBot ? " · bot" : ""} (${user.rating})`;
    return option;
  }));
  const list = $("#game-list");
  if (!games.length) list.innerHTML = "<p>No games yet. Challenge Raven or another account.</p>";
  else list.replaceChildren(...games.map(gameCard));
  $("#leaderboard").replaceChildren(...leaders.map((user, index) => {
    const item = document.createElement("li");
    item.innerHTML = `<span>${index + 1}</span><span>${escapeHtml(user.displayName)} ${user.isBot ? '<b class="bot-badge">bot</b>' : ""}</span><strong>${user.rating}</strong>`;
    return item;
  }));
  show("lobby-view");
}

function gameIdFromHash() {
  const match = location.hash.match(/^#game\/(\d+)$/);
  return match ? Number(match[1]) : null;
}

function gameCard(game) {
  const button = document.createElement("button");
  button.className = "game-card";
  const opponent = game.attacker.id === state.user.id ? game.defender : game.attacker;
  const result = game.status === "active" ? "Active" : game.winnerId === state.user.id ? "Won" : "Lost";
  button.innerHTML = `<strong>vs ${escapeHtml(opponent.displayName)}</strong><small>${game.ruleset} · ${new Date(game.updatedAt * 1000).toLocaleString()}</small><span class="result">${result}</span>`;
  button.onclick = () => openGame(game.id);
  return button;
}

async function openGame(id) {
  location.hash = `game/${id}`;
  state.selected = null;
  delete $("#multiverse").dataset.positioned;
  state.current = await api(`/api/games/${id}`);
  renderGame();
  show("game-view");
  startPolling(id);
}

function startPolling(id) {
  stopPolling();
  state.poll = setInterval(async () => {
    try {
      const latest = await api(`/api/games/${id}`);
      if (!state.current || latest.summary.version !== state.current.summary.version) {
        state.current = latest;
        state.selected = null;
        renderGame();
      }
    } catch (error) { notice(error.message, true); }
  }, 2000);
}

function stopPolling() {
  if (state.poll) clearInterval(state.poll);
  state.poll = null;
}

function renderGame() {
  const detail = state.current;
  const summary = detail.summary;
  const game = detail.game;
  $("#game-title").textContent = `${summary.attacker.displayName} (A) vs ${summary.defender.displayName} (D)`;
  $("#game-status").textContent = game.outcome
    ? `${game.outcome.winner} win · ${game.outcome.reason}`
    : `${game.turn} to move · ${game.message}`;
  const mine = detail.viewerSide === game.turn;
  $("#submit-turn").disabled = !mine || !game.canSubmit;

  const snapshots = game.timelines.flatMap((timeline) =>
    Object.values(timeline.boards).map((snapshot) => ({ timeline, snapshot })));
  const times = snapshots.map(({ snapshot }) => snapshot.coordinate.time).concat(0);
  const rows = game.timelines.map((timeline) => timeline.row).concat(0);
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const minRow = Math.min(...rows);
  const maxRow = Math.max(...rows);
  const stage = $("#multiverse");
  stage.style.setProperty("--timeline-count", String(maxRow - minRow + 1));
  stage.style.setProperty("--time-count", String(maxTime - minTime + 1));
  const children = [];
  for (let row = minRow; row <= maxRow; row += 1) {
    const label = document.createElement("div");
    label.className = "axis-label timeline";
    label.style.gridColumn = String(row - minRow + 2);
    label.style.gridRow = "1";
    label.textContent = `L${row}`;
    children.push(label);
  }
  for (let time = minTime; time <= maxTime; time += 1) {
    const label = document.createElement("div");
    label.className = "axis-label time";
    label.style.gridColumn = "1";
    label.style.gridRow = String(time - minTime + 2);
    label.textContent = `T${time}`;
    children.push(label);
  }
  for (const item of snapshots) children.push(renderBoard(item, minRow, minTime, mine));
  stage.replaceChildren(...children);
  $("#move-history").replaceChildren(...detail.history.map((entry) => {
    const item = document.createElement("li");
    const action = entry.action.type === "submitTurn" ? "submitted turn" : formatMove(entry.action.movement);
    item.textContent = `${entry.sequence}. ${entry.actor.displayName}: ${action}`;
    return item;
  }));

  requestAnimationFrame(() => {
    const origin = stage.querySelector('[data-board="0:0"]');
    if (origin && !stage.dataset.positioned) {
      origin.scrollIntoView({ block: "center", inline: "start" });
      stage.dataset.positioned = "true";
    }
  });
}

function renderBoard({ timeline, snapshot }, minRow, minTime, mine) {
  const coordinate = snapshot.coordinate;
  const card = document.createElement("article");
  const latestTime = Math.max(...Object.keys(timeline.boards).map(Number));
  const active = state.current.game.activeTimelines.includes(timeline.row);
  const sourceMoves = state.current.game.legalMoves.filter((move) => sameBoard(move.from.board, coordinate));
  const playable = state.current.game.playableBoards.some((board) => sameBoard(board, coordinate));
  card.className = `board-card${coordinate.time === latestTime ? " latest" : ""}${playable && mine ? " playable" : ""}${active ? "" : " inactive"}`;
  card.style.gridColumn = String(timeline.row - minRow + 2);
  card.style.gridRow = String(coordinate.time - minTime + 2);
  card.dataset.board = `${coordinate.time}:${coordinate.timeline}`;
  const caption = document.createElement("div");
  caption.className = "board-caption";
  caption.innerHTML = `<strong>T${coordinate.time}L${timeline.row}</strong><span>${coordinate.time === latestTime ? "latest" : "history"}${active ? "" : " · inactive"}</span>`;
  const board = document.createElement("div");
  board.className = "board";
  for (let y = 10; y >= 0; y -= 1) {
    for (let x = 0; x < 11; x += 1) {
      const position = { board: coordinate, square: { x, y } };
      const square = document.createElement("button");
      square.className = `square ${(x + y) % 2 ? "dark" : ""}${isRestricted(x, y) ? " restricted" : ""}`;
      const piece = snapshot.board.cells[y][x];
      if (piece) square.innerHTML = `<span class="piece ${piece.toLowerCase()}">${piece === "King" ? "K" : piece[0]}</span>`;
      if (sourceMoves.some((move) => samePosition(move.from, position))) square.classList.add("source");
      if (state.selected && samePosition(state.selected, position)) square.classList.add("selected");
      if (state.selected && legalTargets(state.selected).some((move) => samePosition(move.to, position))) square.classList.add("target");
      square.onclick = () => chooseSquare(position);
      board.append(square);
    }
  }
  card.append(caption, board);
  return card;
}

function chooseSquare(position) {
  if (state.current.viewerSide !== state.current.game.turn) return;
  if (state.selected) {
    const movement = legalTargets(state.selected).find((move) => samePosition(move.to, position));
    if (movement) {
      playMove(movement);
      return;
    }
  }
  const hasMoves = state.current.game.legalMoves.some((move) => samePosition(move.from, position));
  state.selected = hasMoves ? position : null;
  renderGame();
}

function legalTargets(position) {
  return state.current.game.legalMoves.filter((move) => samePosition(move.from, position));
}

async function playMove(movement) {
  try {
    state.current = await api(`/api/games/${state.current.summary.id}/moves`, {
      method: "POST", body: { version: state.current.summary.version, movement },
    });
    state.selected = null;
    notice("Move accepted.");
    renderGame();
  } catch (error) { notice(error.message, true); }
}

function sameBoard(a, b) { return a.time === b.time && a.timeline === b.timeline; }
function samePosition(a, b) { return sameBoard(a.board, b.board) && a.square.x === b.square.x && a.square.y === b.square.y; }
function isRestricted(x, y) { return (x === 5 && y === 5) || ((x === 0 || x === 10) && (y === 0 || y === 10)); }
function formatMove(move) { return `${formatPosition(move.from)} → ${formatPosition(move.to)}`; }
function formatPosition(position) { return `T${position.board.time}L${position.board.timeline}:${String.fromCharCode(65 + position.square.x)}${position.square.y + 1}`; }
function escapeHtml(value) { const span = document.createElement("span"); span.textContent = value; return span.innerHTML; }
function notice(message, isError = false) { const el = $("#game-notice"); el.textContent = message; el.classList.toggle("error", isError); }

document.querySelectorAll("[data-auth-mode]").forEach((button) => {
  button.onclick = () => {
    document.querySelectorAll("[data-auth-mode]").forEach((other) => other.classList.toggle("active", other === button));
    $("#auth-form").dataset.mode = button.dataset.authMode;
    $("#display-name-field").hidden = button.dataset.authMode !== "register";
  };
});
$("#auth-form").dataset.mode = "login";
$("#auth-form").onsubmit = async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const mode = event.currentTarget.dataset.mode;
  try {
    const session = await api(`/api/auth/${mode}`, {
      method: "POST",
      body: { username: form.get("username"), password: form.get("password"), displayName: form.get("displayName") || undefined },
    });
    state.token = session.token;
    state.user = session.user;
    localStorage.setItem("huginn-token", state.token);
    setIdentity();
    await loadLobby();
  } catch (error) { $("#auth-error").textContent = error.message; }
};
$("#new-game-form").onsubmit = async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    state.current = await api("/api/games", { method: "POST", body: Object.fromEntries(form) });
    delete $("#multiverse").dataset.positioned;
    renderGame();
    show("game-view");
    startPolling(state.current.summary.id);
  } catch (error) { alert(error.message); }
};
$("#back-to-lobby").onclick = () => {
  history.replaceState(null, "", location.pathname);
  loadLobby();
};
$("#refresh-game").onclick = () => openGame(state.current.summary.id);
$("#submit-turn").onclick = async () => {
  try {
    state.current = await api(`/api/games/${state.current.summary.id}/submit`, { method: "POST", body: { version: state.current.summary.version } });
    state.selected = null;
    renderGame();
  } catch (error) { notice(error.message, true); }
};

(async () => {
  if (!state.token) return show("auth-view");
  try {
    state.user = await api("/api/me");
    setIdentity();
    await loadLobby();
    const gameId = gameIdFromHash();
    if (gameId) await openGame(gameId);
  } catch (_) {
    await logout();
  }
})();
