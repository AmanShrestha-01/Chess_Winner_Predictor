---
title: Chess Winner Predictor
emoji: ♟
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
---

# Chess Winner Predictor

Predicts whether a chess game ends in a **white win, black win, or draw** — using
only information known **before the first move**: the two players' ratings and the
time control. It never sees a single move.

**62.6% accuracy** on held-out games, against a 49.9% baseline (always guess white).

## How it was built

Trained on the [Lichess games dataset](https://www.kaggle.com/datasets/datasnaek/chess)
(20,058 games). Random forest, 300 trees, `max_depth=8`.

**Features (all pre-game):** both ratings, their difference, their average, whether the
game was rated, and the time control split into base minutes and increment seconds.

**Deliberately excluded:** `victory_status`, `turns`, `moves`, `last_move_at`. Those
describe how the game *ended*, so using them is data leakage — it lifts accuracy to
71.8%, which looks better and is worthless, because none of those values exist for a
game that hasn't been played yet.

## Honest limits

- The model rarely predicts a draw. Draws are only 4.7% of games, so ignoring them
  costs almost nothing in accuracy — which is exactly why accuracy is the wrong
  metric here on its own.
- Trained on online games from 2017. Nothing here transfers to over-the-board or
  elite play.

Full write-up, notebook and notes:
[github.com/AmanShrestha-01/Chess_Winner_Predictor](https://github.com/AmanShrestha-01/Chess_Winner_Predictor)
