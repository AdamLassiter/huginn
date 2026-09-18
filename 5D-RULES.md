# 5D Copenhagen Hnefatafl

This document extends the Copenhagen rules in [`RULES.md`](RULES.md) with
multiverse time travel. The rules in `RULES.md` continue to apply unless this
document explicitly changes them.

The multiverse rules are optional. A game configured without them is an
ordinary game of Copenhagen Hnefatafl.

## 1. Coordinates

A position in the multiverse has four coordinates:

- **X**: the file within an 11x11 board.
- **Y**: the rank within an 11x11 board.
- **T**: time, represented by successive board snapshots.
- **L**: the timeline containing the board.

The game begins on board **T0L0**. Attackers move first.

- An even-T board is an attacker board: the attacker is to play there.
- An odd-T board is a defender board: the defender is to play there.
- A move made on a board at T creates its successor at T+1.

The origin timeline is L0. Timelines created by an attacker extend in the
positive-L direction. Timelines created by a defender extend in the negative-L
direction.

Every board snapshot is immutable. Creating a successor never changes the board
from which it was derived.

## 2. Latest Boards and the Present

The **latest board** on a timeline is the board on that timeline with no
successor on the same timeline.

The **Present** is the lowest T occupied by a latest board on any active
timeline. A latest board at the Present is a **Present board**.

Only active Present boards belonging to the player whose turn it is may be used
as source boards. A player may therefore have to make more than one move during
one turn.

Moves are staged until every active Present board that requires the current
player has advanced. The player then submits the complete turn, and play passes
to the opponent indicated by the new Present. A staged move may be undone before
the turn is submitted.

If a staged move creates another active board at the Present for the same
player, that board must also be played before the turn can be submitted.

## 3. Active and Inactive Timelines

L0 is always active.

Let `A` be the number of timelines created by the attackers and `D` the number
created by the defenders. On each side of L0, the active timelines are the
nearest `min(A, D) + 1` timelines created by that side. Consequently, one player
may have at most one more active created timeline than the other. Any additional
timelines still exist, but are inactive.

An inactive timeline:

- does not affect the Present;
- does not create a move obligation;
- cannot be used as the source of a move;
- cannot supply or receive a cross-board capture; and
- cannot cause a win or loss.

Its immutable boards remain visible as part of the game's history. A timeline
may become active later if the other player creates enough timelines to restore
the balance.

## 4. Movement

All pieces, including the king, retain their Copenhagen castle-like movement.
A move changes exactly one of X, Y, or T. Direct movement along L is not allowed.

### 4.1 Spatial Movement

A spatial move changes X or Y on one board and follows the movement and
restricted-square rules in `RULES.md`. It may not jump over another piece.

A spatial move from TnLm creates T(n+1)Lm. The source board remains unchanged;
the successor contains the result of the move and any captures.

Example: moving on T0L0 creates T1L0. Moving later on T3L-1 creates T4L-1.

### 4.2 Time Travel

A temporal move travels backward along T and keeps X, Y, and L unchanged.

A temporal move is legal only when:

- its destination has a lower T than its source;
- the raw difference in T is a positive even number;
- source and destination are on the same timeline;
- the destination board is a board on which the moving side was to play;
- the destination square is empty and may be occupied by that piece; and
- the same square is empty on every intervening board belonging to the moving
  side.

Opponent-turn boards between two waypoints do not count as temporal movement
steps. Moving forward in time is not allowed.

A temporal move advances both ends of the move:

1. The source timeline receives a successor at source T+1 with the moving piece
   removed.
2. The destination board produces a successor at destination T+1 with the
   moving piece added.

Because a backward destination is historical, its normal successor coordinate
is already occupied. The destination successor therefore starts a new timeline.

An attacker first attempts to place that branch at destination L+1; a defender
first attempts destination L-1. If that row is occupied, continue outward in
the same direction until the next free L row is found. Existing timeline
coordinates are never renumbered.

Examples:

- An attacker moving from T4L0 to T0L0 creates a source board at T5L0 and a
  destination board at T1L1, if L1 is free.
- A defender moving from T5L0 to T1L0 creates T6L0 and T2L-1, if L-1 is free.
- If L1 is already occupied, an attacker branching from L0 uses L2, then L3,
  and so on until a free row is found.

## 5. Capture in Four Dimensions

Copenhagen capture remains custodial: a move captures by closing a trap. A
piece that moves voluntarily between enemies is not captured merely for being
there.

In multiverse play, X, Y, T, and L can each form a capture axis:

- adjacent X and Y coordinates differ by one square;
- adjacent T capture coordinates differ by two raw T positions, so all three
  boards in a temporal sandwich have the same board-to-move parity; and
- adjacent L capture coordinates occupy consecutive active timeline rows at the
  same X, Y, and T.

If a required board coordinate does not exist, or its timeline is inactive, it
cannot complete a sandwich.

### 5.1 Ordinary Pieces

An attacker or defender is captured when an enemy move brackets it on the two
opposite sides of any one axis. The piece just moved must be one of the two
bracketing pieces.

A move may close several traps at once, including traps on different axes. The
king may serve as a friendly bracketing piece for the defenders.

Restricted squares may replace a bracketing piece only on X or Y. They have no
special meaning along T or L.

### 5.2 Cross-Board Capture

History is never rewritten to remove a piece from an existing board. Instead, a
T- or L-axis capture advances the board containing the victim:

- If the victim is on a board already being created by the move, remove it from
  that new board.
- Otherwise, copy the victim's board to a successor at T+1 and remove the victim
  from the copy.
- If that successor coordinate is already occupied, create a new branch in the
  capturing player's L direction, using the next free row as described for time
  travel.

All captures directly closed by the move are determined from the same tentative
post-move multiverse and are then applied together. Multiple victims on one
board are removed in one successor. Successors created only to record captures
do not trigger further captures, so captures do not cascade.

Capture successors are ordinary boards. If active, they affect the Present and
may create further move obligations during the current turn.

Example of a temporal sandwich: an attacker newly placed at `(x, y, T5, L0)`
may capture a defender at `(x, y, T3, L0)` if another attacker occupies
`(x, y, T1, L0)`. The captured defender is removed through a successor of its
T3 board; T3 itself remains unchanged.

Example of a timeline sandwich: an attacker newly placed at `(x, y, T5, L1)`
may capture a defender at `(x, y, T5, L0)` if an attacker also occupies
`(x, y, T5, L-1)` and all three timelines are active.

### 5.3 Shield Walls

Shield-wall captures remain spatial. They apply only to rows of pieces along an
X/Y board edge and are resolved independently on each board. There is no
temporal or cross-timeline shield wall.

## 6. Capturing the King

Away from the throne and spatial board edges, the king is captured when an
attacker's move surrounds him on two complete axes. A complete axis has an
attacker on each of the two opposite adjacent coordinates.

The two axes may be any pair of X, Y, T, and L. The moved attacker must complete
the surround. As with ordinary capture, T uses a stride of two raw T positions
and L uses consecutive active timeline rows.

The throne may substitute for an attacker only as an X- or Y-axis flank. Thus,
the ordinary Copenhagen capture beside the throne still requires attackers on
the other three spatial sides and must be completed by an attacker move.

The spatial edge is not hostile. A king on the edge cannot be captured by an
incomplete spatial surround. He can, however, be captured there if two complete
axes exist without relying on a coordinate beyond that board edge. For example,
complete T and L surrounds can capture a king on a spatial edge.

The king is never captured as part of a shield wall. Other defenders in the
same shield wall are captured normally.

## 7. Winning the Game

A winning move must leave its deciding result on an active latest successor.
That result ends the entire game. An unchanged position on a historical or
inactive board cannot decide the game.

### 7.1 Defender Wins

The defenders win if a king on an active latest board:

- reaches a corner square; or
- forms a valid Copenhagen exit fort.

Corner escape and exit forts remain spatial, board-local rules. Other timelines
do not help form or break a fort.

### 7.2 Attacker Wins

The attackers win if a move creates an active latest successor in which:

- the king is captured under section 6;
- the king and all remaining defenders are spatially encircled by an unbroken
  ring; or
- the defenders cannot reach a corner or form an exit fort under the automatic
  Copenhagen rules in `RULES.md`.

Encirclement, exit-fort analysis, blocked corners, and the determination that
the defenders cannot escape are evaluated separately on each board using X and
Y only. They are not extended into T or L.

When branches contain several current kings, capturing any king or escaping
with any king on an active latest board decides the game. It is not necessary
to capture every active king. Kings on inactive timelines do not decide the
game.

## 8. Repetition

The defender may not create a board whose piece placement repeats a placement
among that board's causal ancestors. Such a move is illegal.

For a board created by an ordinary move, its causal ancestors are its earlier
boards on that lineage. For a branch, ancestry continues through the historical
board from which the branch was created. A move that creates several successors
is illegal if any resulting successor violates this rule.

The restriction applies only to the defender, as stated in `RULES.md`.

## 9. No Legal Turn

A player loses if there is no legal sequence of moves that completes all of
that player's active Present obligations.

It is not sufficient to inspect one board in isolation. A move on one Present
board may create, activate, capture on, or advance another board before the turn
is complete.

## 10. Rules Not Inherited from 5D Chess

Hnefatafl has no concept of check or checkmate. A king may enter a threatened
position; the attackers win only when a legal move actually completes a king
capture or another Copenhagen attacker-win condition.

The following are not part of this ruleset:

- direct movement along L;
- forward time travel;
- diagonal movement through multiple axes;
- chess displacement captures;
- chess castling, promotion, or en passant;
- special 13x13 starting positions; and
- rules that depend on whether a player is human or controlled by an AI.
