---
id: game
title: Game
label: Game
keywords: [game, games, puzzle, arcade, player, players, board game, card game, quiz, tic-tac-toe, chess, snake, tetris, score, levels, multiplayer]
---

Internal guidance for the team. Adapt it to the actual game; it is not a feature list.

## Considerations

- Rules: state them precisely enough to be tested without playing.
- State transitions: what the game state is, and which actions change it.
- Player actions: how many players, whose turn it is, and what each may do.
- Invalid moves: what is rejected, and how the player finds out.
- Win and loss conditions, including who wins and when it is detected.
- Draw conditions, or a clear statement that the game cannot draw.
- Restart and reset: what clears, and what, such as a score, survives.
- User input: keyboard, mouse, touch or terminal, and how bad input is handled.
- Interface and responsiveness: board size, screen sizes, feedback on each action.
- Keep the rules separate from the interface so they can be tested without one.

## Typical Development Phases

1. Core rules and state model.
2. Turn or game loop.
3. Win, lose and draw detection.
4. Playable interface.
5. Restart, scoring and polish.
6. Final review.

## Quality Requirements

- Rule logic can be tested without any graphics or terminal interface.
- The game never reaches a state the rules do not allow.
- Each action gives immediate, visible feedback.
- A finished game cannot be changed by further moves.

## Common Edge Cases

- A move on an occupied or off-board position.
- A move made after the game has ended.
- A move made out of turn.
- A win on the very last available move, which must not be reported as a draw.
- Restarting in the middle of a game.
- Rapid repeated input.
- Empty or malformed input.

## Suggested Acceptance Checks

- Every winning pattern is detected for each player.
- A full board or finished game with no winner is reported as a draw, where draws exist.
- Invalid moves leave the state unchanged and report why.
- Turn order alternates correctly and survives an invalid move.
- Restart returns the game to its documented initial state.
