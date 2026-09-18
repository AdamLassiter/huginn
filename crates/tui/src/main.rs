use std::fmt::Write as _;
use std::io::{self, Stdout};
use std::time::Duration;

use clap::{Parser, ValueEnum};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use huginn_core::{
    BOARD_EDGE, BOARD_SIZE_U8, BoardCoordinate, Game, GameOutcome, Move, Piece, Position, Ruleset,
    Square,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Classic,
    Multiverse,
}

impl From<Mode> for Ruleset {
    fn from(value: Mode) -> Self {
        match value {
            Mode::Classic => Self::Classic,
            Mode::Multiverse => Self::Multiverse,
        }
    }
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Start directly in the selected rules mode.
    #[arg(long, value_enum)]
    mode: Option<Mode>,
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Menu,
    Playing,
}

struct App {
    phase: Phase,
    menu_choice: Ruleset,
    game: Game,
    focus: BoardCoordinate,
    cursor: Square,
    selected: Option<Position>,
    legal_moves: Vec<Move>,
    notice: String,
    help: bool,
    quit: bool,
}

impl App {
    fn new(mode: Option<Mode>) -> Self {
        let ruleset = mode.map_or(Ruleset::Classic, Into::into);
        Self {
            phase: if mode.is_some() {
                Phase::Playing
            } else {
                Phase::Menu
            },
            menu_choice: ruleset,
            game: Game::new(ruleset),
            focus: BoardCoordinate::new(0, 0),
            cursor: Square::new(5, 5).expect("center is on board"),
            selected: None,
            legal_moves: Vec::new(),
            notice: String::new(),
            help: false,
            quit: false,
        }
    }

    fn reset(&mut self, ruleset: Ruleset) {
        self.game = Game::new(ruleset);
        self.focus = BoardCoordinate::new(0, 0);
        self.cursor = Square::new(5, 5).expect("center is on board");
        self.selected = None;
        self.legal_moves.clear();
        self.notice.clear();
        self.help = false;
        self.phase = Phase::Playing;
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('q') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return;
        }
        if self.phase == Phase::Menu {
            self.handle_menu_key(key);
            return;
        }
        if self.help {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('?' | 'q')) {
                self.help = false;
            }
            return;
        }
        match key.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('?') => self.help = true,
            KeyCode::Char('n') => {
                self.menu_choice = self.game.ruleset();
                self.phase = Phase::Menu;
                self.selected = None;
                self.legal_moves.clear();
            }
            KeyCode::Left | KeyCode::Char('h') => self.move_cursor(-1, 0),
            KeyCode::Right | KeyCode::Char('l') => self.move_cursor(1, 0),
            KeyCode::Up | KeyCode::Char('k') => self.move_cursor(0, 1),
            KeyCode::Down | KeyCode::Char('j') => self.move_cursor(0, -1),
            KeyCode::Char('[') => self.change_time(-1),
            KeyCode::Char(']') => self.change_time(1),
            KeyCode::Char('J') => self.change_timeline(-1),
            KeyCode::Char('K') => self.change_timeline(1),
            KeyCode::Enter | KeyCode::Char(' ') => self.activate_square(),
            KeyCode::Esc => {
                self.selected = None;
                self.legal_moves.clear();
                "Selection cleared.".clone_into(&mut self.notice);
            }
            KeyCode::Char('u') => match self.game.undo_staged_move() {
                Ok(()) => {
                    "Undid staged move.".clone_into(&mut self.notice);
                    self.clear_selection_and_sync();
                }
                Err(error) => self.notice = error.to_string(),
            },
            KeyCode::Char('s') => match self.game.submit_turn() {
                Ok(()) => {
                    "Turn submitted.".clone_into(&mut self.notice);
                    self.clear_selection_and_sync();
                }
                Err(error) => self.notice = error.to_string(),
            },
            _ => {}
        }
    }

    fn handle_menu_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Down | KeyCode::Char('j' | 'k') => {
                self.menu_choice = match self.menu_choice {
                    Ruleset::Classic => Ruleset::Multiverse,
                    Ruleset::Multiverse => Ruleset::Classic,
                };
            }
            KeyCode::Enter => self.reset(self.menu_choice),
            KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
            _ => {}
        }
    }

    fn move_cursor(&mut self, dx: i32, dy: i32) {
        let x = (i32::from(self.cursor.x) + dx).clamp(0, i32::from(BOARD_EDGE));
        let y = (i32::from(self.cursor.y) + dy).clamp(0, i32::from(BOARD_EDGE));
        self.cursor = Square::new(
            u8::try_from(x).expect("clamped x fits"),
            u8::try_from(y).expect("clamped y fits"),
        )
        .expect("clamped cursor is valid");
    }

    fn change_time(&mut self, direction: i32) {
        let Some(timeline) = self.game.timeline(self.focus.timeline) else {
            return;
        };
        let times: Vec<i32> = timeline.boards.keys().copied().collect();
        let Some(index) = times.iter().position(|time| *time == self.focus.time) else {
            return;
        };
        let next = (i32::try_from(index).expect("index fits") + direction).clamp(
            0,
            i32::try_from(times.len().saturating_sub(1)).expect("length fits"),
        );
        self.focus.time = times[usize::try_from(next).expect("nonnegative index")];
    }

    fn change_timeline(&mut self, direction: i32) {
        let rows: Vec<i32> = self.game.timelines().map(|timeline| timeline.row).collect();
        let Some(index) = rows.iter().position(|row| *row == self.focus.timeline) else {
            return;
        };
        let next = (i32::try_from(index).expect("index fits") + direction).clamp(
            0,
            i32::try_from(rows.len().saturating_sub(1)).expect("length fits"),
        );
        let row = rows[usize::try_from(next).expect("nonnegative index")];
        if let Some(coordinate) = self.game.latest_coordinate(row) {
            self.focus = coordinate;
        }
    }

    fn activate_square(&mut self) {
        if self.game.outcome().is_some() {
            "The game is over; press n for a new game.".clone_into(&mut self.notice);
            return;
        }
        let here = Position::new(self.focus, self.cursor);
        if let Some(movement) = self
            .legal_moves
            .iter()
            .copied()
            .find(|movement| movement.to == here)
        {
            match self.game.apply_move(movement) {
                Ok(()) => {
                    self.notice = format!("Moved {} to {}.", movement.from, movement.to);
                    self.clear_selection_and_sync();
                }
                Err(error) => self.notice = error.to_string(),
            }
            return;
        }

        let moves = self.game.legal_moves_from(here);
        if moves.is_empty() {
            self.selected = None;
            self.legal_moves.clear();
            "That square has no legal move from this board.".clone_into(&mut self.notice);
        } else {
            self.selected = Some(here);
            self.legal_moves = moves;
            self.notice = format!("Selected {here}; {} destinations.", self.legal_moves.len());
        }
    }

    fn clear_selection_and_sync(&mut self) {
        self.selected = None;
        self.legal_moves.clear();
        if self.game.board(self.focus).is_none() {
            self.focus = self
                .game
                .latest_coordinate(self.focus.timeline)
                .or_else(|| self.game.latest_coordinate(0))
                .unwrap_or(BoardCoordinate::new(0, 0));
        } else if let Some(latest) = self.game.latest_coordinate(self.focus.timeline) {
            self.focus = latest;
        }
    }

    fn draw(&self, frame: &mut ratatui::Frame<'_>) {
        if self.phase == Phase::Menu {
            self.draw_menu(frame);
            return;
        }
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(20),
                Constraint::Length(3),
            ])
            .split(frame.area());
        self.draw_header(frame, outer[0]);
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(17),
                Constraint::Min(38),
                Constraint::Length(25),
            ])
            .split(outer[1]);
        self.draw_timelines(frame, body[0]);
        self.draw_board(frame, body[1]);
        self.draw_info(frame, body[2]);
        Self::draw_footer(frame, outer[2]);
        if self.help {
            Self::draw_help(frame);
        }
    }

    fn draw_menu(&self, frame: &mut ratatui::Frame<'_>) {
        let area = centered_rect(58, 15, frame.area());
        frame.render_widget(Clear, area);
        let classic = if self.menu_choice == Ruleset::Classic {
            ">"
        } else {
            " "
        };
        let multiverse = if self.menu_choice == Ruleset::Multiverse {
            ">"
        } else {
            " "
        };
        let text = vec![
            Line::from(Span::styled(
                "HUGINN",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from("Copenhagen hnefatafl"),
            Line::from(""),
            Line::from(format!("{classic} Classic Copenhagen")),
            Line::from(format!("{multiverse} 5D Copenhagen")),
            Line::from(""),
            Line::from("↑/↓ choose   Enter start   q quit"),
        ];
        frame.render_widget(
            Paragraph::new(text)
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(" New game ")),
            area,
        );
    }

    fn draw_header(&self, frame: &mut ratatui::Frame<'_>, area: Rect) {
        let outcome = self
            .game
            .outcome()
            .map_or_else(|| format!("{} to move", self.game.turn()), outcome_text);
        let text = format!(
            " HUGINN — {}    {}    Present T{} ",
            self.game.ruleset(),
            outcome,
            self.game.present_time().unwrap_or(0)
        );
        frame.render_widget(
            Paragraph::new(text)
                .style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
                .block(Block::default().borders(Borders::ALL)),
            area,
        );
    }

    fn draw_timelines(&self, frame: &mut ratatui::Frame<'_>, area: Rect) {
        let mut items = Vec::new();
        for timeline in self.game.timelines() {
            let active = self.game.is_active_timeline(timeline.row);
            let marker = if timeline.row == self.focus.timeline {
                ">"
            } else {
                " "
            };
            let status = if active { "active" } else { "inactive" };
            items.push(ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{marker} L{:<3}", timeline.row),
                    if active {
                        Style::default().fg(Color::Green)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
                Span::raw(status),
            ])));
            let boards = timeline
                .boards
                .keys()
                .map(|time| {
                    let focused = *time == self.focus.time && timeline.row == self.focus.timeline;
                    let playable = self
                        .game
                        .is_playable_board(BoardCoordinate::new(*time, timeline.row));
                    let tag = if playable {
                        "*"
                    } else if focused {
                        ">"
                    } else {
                        " "
                    };
                    format!("   {tag} T{time}")
                })
                .collect::<Vec<_>>()
                .join(" ");
            items.push(ListItem::new(boards));
        }
        frame.render_widget(
            List::new(items).block(Block::default().borders(Borders::ALL).title(" Timelines ")),
            area,
        );
    }

    fn draw_board(&self, frame: &mut ratatui::Frame<'_>, area: Rect) {
        let Some(snapshot) = self.game.board(self.focus) else {
            frame.render_widget(
                Paragraph::new("Board unavailable")
                    .block(Block::default().borders(Borders::ALL).title(" Board ")),
                area,
            );
            return;
        };
        let mut lines = Vec::new();
        for y in (0..BOARD_SIZE_U8).rev() {
            let mut spans = vec![Span::styled(
                format!("{:>2} ", y + 1),
                Style::default().fg(Color::DarkGray),
            )];
            for x in 0..BOARD_SIZE_U8 {
                let square = Square::new(x, y).expect("board coordinate valid");
                let position = Position::new(self.focus, square);
                let piece = snapshot.board.get(square);
                let symbol = piece.map_or_else(
                    || {
                        if square.is_restricted() { "#" } else { "·" }
                    },
                    |piece| match piece {
                        Piece::Attacker => "A",
                        Piece::Defender => "D",
                        Piece::King => "K",
                    },
                );
                let mut style = match piece {
                    Some(Piece::Attacker) => Style::default().fg(Color::Red),
                    Some(Piece::Defender) => Style::default().fg(Color::White),
                    Some(Piece::King) => Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                    None if square.is_restricted() => Style::default().fg(Color::Magenta),
                    None => Style::default().fg(Color::DarkGray),
                };
                if self
                    .legal_moves
                    .iter()
                    .any(|movement| movement.to == position)
                {
                    style = style.bg(Color::Blue).add_modifier(Modifier::BOLD);
                }
                if self.selected == Some(position) {
                    style = style.bg(Color::Magenta).add_modifier(Modifier::BOLD);
                }
                if self.cursor == square {
                    style = style
                        .bg(Color::Gray)
                        .fg(Color::Black)
                        .add_modifier(Modifier::BOLD);
                }
                spans.push(Span::styled(format!(" {symbol} "), style));
            }
            lines.push(Line::from(spans));
        }
        let mut files = String::from("   ");
        for x in 0..BOARD_SIZE_U8 {
            write!(&mut files, " {} ", char::from(b'A' + x))
                .expect("writing to a String cannot fail");
        }
        lines.push(Line::from(Span::styled(
            files,
            Style::default().fg(Color::DarkGray),
        )));
        let title = format!(
            " {} {}{} ",
            self.focus,
            if self.game.is_active_timeline(self.focus.timeline) {
                "active"
            } else {
                "inactive"
            },
            if self.game.is_playable_board(self.focus) {
                ", playable"
            } else {
                ""
            }
        );
        frame.render_widget(
            Paragraph::new(lines)
                .alignment(Alignment::Center)
                .block(Block::default().borders(Borders::ALL).title(title)),
            area,
        );
    }

    fn draw_info(&self, frame: &mut ratatui::Frame<'_>, area: Rect) {
        let selected = self.selected.map_or_else(
            || "None".to_owned(),
            |position| format!("{position} ({} moves)", self.legal_moves.len()),
        );
        let staged = if self.game.has_staged_moves() {
            "yes"
        } else {
            "no"
        };
        let submit = if self.game.can_submit() {
            "ready"
        } else {
            "not ready"
        };
        let text = vec![
            Line::from(format!("Cursor: {}:{}", self.focus, self.cursor)),
            Line::from(format!("Selected: {selected}")),
            Line::from(format!("Staged: {staged}")),
            Line::from(format!("Submit: {submit}")),
            Line::from(""),
            Line::from(Span::styled(
                "Engine",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(self.game.message().to_owned()),
            Line::from(""),
            Line::from(Span::styled(
                "Notice",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::from(if self.notice.is_empty() {
                "Select a piece with Enter.".to_owned()
            } else {
                self.notice.clone()
            }),
        ];
        frame.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::ALL).title(" Status ")),
            area,
        );
    }

    fn draw_footer(frame: &mut ratatui::Frame<'_>, area: Rect) {
        frame.render_widget(
            Paragraph::new(
                " arrows/hjkl move  Enter select  [/] time  J/K timeline  u undo  s submit  ? help  n new  q quit ",
            )
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::ALL)),
            area,
        );
    }

    fn draw_help(frame: &mut ratatui::Frame<'_>) {
        let area = centered_rect(76, 24, frame.area());
        frame.render_widget(Clear, area);
        let text = vec![
            Line::from("Move the cursor with arrow keys or h/j/k/l."),
            Line::from("Press Enter to select a piece, then Enter on a highlighted target."),
            Line::from("Use [ and ] to inspect time on the current timeline."),
            Line::from("Use uppercase J and K to change timeline."),
            Line::from(""),
            Line::from("For time travel, select a piece on a playable Present board,"),
            Line::from("navigate to an older same-parity board, and choose the same square."),
            Line::from(""),
            Line::from("Multiverse moves are staged. Press s when every Present obligation"),
            Line::from("has advanced; press u to undo the latest staged move."),
            Line::from(""),
            Line::from("A/D/K are attackers, defenders, and king. # marks restricted squares."),
            Line::from("Press ?, Esc, or q to close this help."),
        ];
        frame.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .block(Block::default().borders(Borders::ALL).title(" Help ")),
            area,
        );
    }
}

fn outcome_text(outcome: GameOutcome) -> String {
    format!("{} win ({:?})", outcome.winner, outcome.reason)
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width.saturating_sub(2));
    let height = height.min(area.height.saturating_sub(2));
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn run(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> io::Result<()> {
    while !app.quit {
        terminal.draw(|frame| app.draw(frame))?;
        if event::poll(Duration::from_millis(200))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            app.handle_key(key);
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();
    let mut guard = TerminalGuard::enter()?;
    let mut app = App::new(cli.mode);
    run(&mut guard.terminal, &mut app)
}
