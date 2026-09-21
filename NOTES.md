# Project notes — Chess Winner Predictor

Reference for when something feels confusing. Jump to whatever you need.

- [1. What this project is](#1-what-this-project-is)
- [2. The vocabulary](#2-the-vocabulary)
- [3. Reading the data](#3-reading-the-data)
- [4. The baseline](#4-the-baseline)
- [5. Data leakage](#5-data-leakage)
- [6. Features](#6-features)
- [7. Splitting](#7-splitting)
- [8. The models](#8-the-models)
- [9. Overfitting](#9-overfitting)
- [10. Why draws are the hard part](#10-why-draws-are-the-hard-part)
- [11. Every number, in one place](#11-every-number-in-one-place)
- [12. Commands](#12-commands)

---

## 1. What this project is

Predict whether a chess game ends in **white**, **black**, or **draw**, using only
what is known **before the first move**.

In proper terms: **supervised learning**, **multiclass classification**, on
**tabular data**, using **classical ML**.

- **Supervised** — we have 20,058 games where we already know the answer. The model
  learns from examples with answers attached.
- **Multiclass** — three possible answers, not two.
- **Tabular** — the data is a table. Not images, not text.
- **Classical ML** — decision trees and forests. No neural networks.

---

## 2. The vocabulary

| Word | Plain meaning |
|---|---|
| **label** | The thing you're predicting. Ours is `winner` |
| **feature** | One number the model gets to look at |
| **feature matrix** (`X`) | The table of all features |
| **training set** | The 80% the model learns from |
| **test set** | The 20% it never sees, used to judge it |
| **baseline** | The score you get with no effort. The bar to beat |
| **leakage** | Accidentally giving the model the answer |
| **overfitting** | Memorising the training data instead of learning |
| **class imbalance** | One answer is much rarer than the others |
| **DataFrame** | A table, in pandas |

---

## 3. Reading the data

`games.csv` is plain text with commas. The first line names the columns; every line
after is one game.

```python
import pandas as pd
games = pd.read_csv("data/games.csv")    # -> (20058, 16)
```

`read_csv` splits every line at the commas and builds a **DataFrame** — a table with
named columns you can ask for by name.

The chain used constantly:

```python
games["winner"].value_counts()
#     ^^^^^^^^  picks one column
#               ^^^^^^^^^^^^^^  counts each value
```

**file → read_csv → DataFrame → ["column"] → .value_counts() → answer**

Each step narrows what came before. That pattern is most of pandas.

---

## 4. The baseline

**49.86%**

That's what you score by always guessing "white" — no model, no thinking. White wins
49.86% of games, so guessing white is right that often.

**Why it matters:** accuracy means nothing on its own. "62% accurate" sounds fine
until you know that doing nothing scores 49.86%. The real gain is **12 points**, not 62.

Always ask: *better than what?*

---

## 5. Data leakage

**The most important idea in the project.**

The test for any column: **would I know this value before the first move?**

| Column | Known before? | Verdict |
|---|---|---|
| `white_rating` | Yes — players have ratings already | ✅ use it |
| `victory_status` | No — it says how the game ended | ❌ banned |
| `turns` | No — you learn it when the game ends | ❌ banned |
| `moves` | No — it's the whole game | ❌ banned |
| `last_move_at` | No — a post-game timestamp | ❌ banned |

**The proof we ran:**

| Features | Accuracy |
|---|---|
| Baseline | 49.86% |
| Honest (pre-game only) | 60.11% |
| **Cheating (+ turns, victory_status)** | **71.79%** |

**Leakage makes your score go UP.** That's what makes it dangerous — it looks like
success. Then the model meets a real game that hasn't been played, the columns it
depends on don't exist, and it's useless.

The clearest example: when `victory_status` says `draw`, `winner` is `draw` in all
906 cases. The model wasn't predicting. It was copying the next column.

---

## 6. Features

Seven, all pre-game:

| Feature | Where it comes from |
|---|---|
| `white_rating` | straight from the file |
| `black_rating` | straight from the file |
| `rating_diff` | white − black **(we made this one)** |
| `rating_mean` | their average |
| `rated` | True/False → 1/0 |
| `base_min` | from `"15+2"` → 15 |
| `increment_sec` | from `"15+2"` → 2 |

**`rating_diff` is a derived feature** — it wasn't in the file, we created it by
subtracting two columns. It ended up being **50.9%** of the model's decisions.

The information was already there in the two rating columns. Doing the subtraction
ourselves made it far easier for the model to use. That's **feature engineering**.

`increment_code` needed splitting because `"15+2"` is *text*. Models need numbers.

---

## 7. Splitting

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Why:** testing a model on data it studied proves nothing. It's revising from a book
with the answers printed underneath, then being tested on those same questions. You
score 100% and learn nothing about whether you understand.

So we hide 20% (4,012 games), train on 80% (16,046), and test on the hidden part.

| Argument | What it does |
|---|---|
| `test_size=0.2` | hold back 20% |
| `stratify=y` | keep the same white/black/draw mix in both halves |
| `random_state=42` | same split every run, so results are repeatable |

**Why `stratify` matters here:** draws are only 4.7%. A random split could put far more
draws in one half than the other by bad luck. Stratifying prevents that.

---

## 8. The models

**Decision tree** — asks yes/no questions about the features. *Is rating_diff above 50?
Is base_min under 3?* Each question splits the games into two groups. `max_depth` is
how many questions deep it may go.

**Random forest** — 300 decision trees, each trained on a random slice of the data and
a random subset of the features. They vote. Individual trees overfit, but they overfit
*differently*, so the errors cancel when they vote.

| Model | Train | Test |
|---|---|---|
| Tree, depth 4 | 62.2% | 61.9% |
| Tree, unlimited | 99.0% | 57.4% |
| **Forest, depth 8** | 65.8% | **62.6%** ← what we shipped |

**What the forest used:**

| Feature | Share |
|---|---|
| `rating_diff` | 50.9% |
| `black_rating` | 15.9% |
| `white_rating` | 15.4% |
| `rating_mean` | 10.3% |
| `base_min` | 3.4% |
| `increment_sec` | 3.0% |
| `rated` | 1.1% |

Rating is **92.5%** of the model. Everything else is noise around the edges.

---

## 9. Overfitting

The clearest result in the project:

| max_depth | train | test | gap | leaves |
|---|---|---|---|---|
| 2 | 61.4% | 61.1% | 0.3% | 4 |
| **4** | **62.2%** | **61.9%** | **0.3%** | 16 |
| 8 | 64.3% | 61.2% | 3.1% | 196 |
| 12 | 68.2% | 59.9% | 8.4% | 790 |
| None | **99.0%** | **57.4%** | **41.6%** | 5,364 |

The unlimited tree got **99% on training data**. It looks like the best model. It's the
worst one — 57.4% on games it hadn't seen, below every other row.

**What happened:** with no depth limit it kept splitting until each group held one
single game. 5,364 leaves. It didn't learn "stronger players win" — it memorised
*"white 1487 vs black 1502, 10-minute clock → black won."* One specific game.

**Why the 99% is worthless:** it was measured on games the model had already studied.
A score on data you've seen can't tell memorising and learning apart — both produce a
high number.

**The gap is your instrument.** Small gap → it generalises. Big gap → it memorised. You
need both numbers; neither one alone tells you.

**The simplest real model won.** Depth 4, sixteen leaves.

---

## 10. Why draws are the hard part

Draws are **4.7%** of games — 950 out of 20,058.

A model that *never* predicts a draw is still right about 95% of the time on that
question, just by never saying it. So there's almost no pressure to learn draws, and
our model largely doesn't.

This is **class imbalance**, and it's why overall accuracy lies. You need **per-class
metrics** — how did it do on each of the three answers separately?

Same problem shape as fraud detection, rare diseases, and manufacturing defects. The
thing you care most about is the rare one.

**This is still unfixed in our model.** It's the honest next step: `class_weight` tells
the model to care more about the rare class.

---

## 11. Every number, in one place

| | |
|---|---|
| Games | 20,058 |
| Columns | 16 |
| Missing values | 0 |
| Duplicate rows | 429 |
| White wins | 49.9% |
| Black wins | 45.4% |
| Draws | 4.7% |
| Baseline accuracy | 49.86% |
| Honest model | 60.11% |
| Leaky model (don't) | 71.79% |
| **Final model** | **62.6%** |
| Train / test | 16,046 / 4,012 |

**Patterns found:**
- Black 400+ stronger → white wins **15.6%**. White 400+ stronger → white wins **84.2%**
- Draws: **3.1%** in bullet, **7.8%** in classical. They need time *and* even players
- Opening win rates are **confounded** — weak players pick weak openings, so the
  opening looks like the cause when it's really the rating

---

## 12. Commands

```bash
# activate the environment
source .venv/bin/activate

# the notebook
.venv/bin/jupyter lab chess_ml.ipynb

# the web app
.venv/bin/python app.py          # http://127.0.0.1:7860
lsof -ti:7860 | xargs kill       # stop it

# save the environment after installing something new
.venv/bin/pip freeze > requirements.txt
```

---

## The four habits worth keeping

1. **Ask what the baseline is.** A score means nothing without it.
2. **Ask if each feature is knowable at prediction time.** That's the leakage test.
3. **Check train *and* test.** The gap between them is the truth.
4. **Look at each class separately.** Overall accuracy hides the rare ones.

These apply to every ML project you will ever do, in any domain. The models change.
These don't.
