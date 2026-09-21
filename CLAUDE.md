# Project: Chess winner predictor

A beginner ML learning project. I am new to machine learning.
Goal: predict who wins a chess game (white, black, or draw) using the
Lichess dataset in `data/games.csv`.

## Teaching rules (most important)

- I'm here to LEARN, not just get working code.
- Before writing code, explain what we're about to do and why, simply. Explain like you would to a 15 year old kid . but not missing the technical terms and keys thats required.
- Keep each step small. One concept at a time.
- Add a short comment on every important line.
- After each step, ask me one question to check I understood.
- When there's a key line (`fit`, `predict`, `train_test_split`), let me
  type it myself first, then check my version.

## Stack

Python, Jupyter notebook, pandas, matplotlib, scikit-learn.
Use a virtual environment called `.venv`.

## Don't let me cheat by accident (leakage)

The target is the `winner` column. Some columns describe how the game *ended*,
so using them as features is cheating — the model would be reading the answer.
Never use these to predict:

- `victory_status` — literally says `mate`, `resign`, `draw`
- `turns` — only known once the game is over
- `moves` — the whole game, including the checkmate
- `last_move_at` — a post-game timestamp

Fair to use, because they're known before the first move: `white_rating`,
`black_rating`, `rated`, `increment_code`.

The opening columns (`opening_eco`, `opening_name`, `opening_ply`) are a grey
zone — the opening happens during play. Leave them out to start. If we add them
later, say so out loud and show the score without them too.

If my accuracy on three classes ever looks better than about 70%, stop and
suspect leakage before celebrating.

## Rules for Claude

- Don't commit anything in `data/`.
- Don't invent numbers about the dataset. Run the code and show me the real
  output. If `data/games.csv` is missing, say so instead of guessing.
- Before using a column as a feature, check it against the leakage list above
  and tell me the verdict.
- Set `random_state=42` everywhere, so my results don't change between runs.
