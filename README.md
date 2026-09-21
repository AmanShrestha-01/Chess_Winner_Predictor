# Chess Winner Predictor

Predicting whether **white wins, black wins, or the game is drawn** — using only
information known *before the first move*.

A beginner machine-learning project, built step by step. **Final model: 62.6%
accuracy** against a 49.9% baseline.

---

## 1. Tools

Claude Code (the assistant used to build this), Python 3.12, and a virtual
environment. The whole project runs locally.

## 2. The data

[**Lichess games dataset**](https://www.kaggle.com/datasets/datasnaek/chess)
(`datasnaek/chess`) — 20,058 real online games, 16 columns, no missing values.

Download `games.csv` from the **Data** tab and put it in `data/`.

> A mistake worth recording: the first download came from the **Code** tab, which
> gives somebody else's *notebook*, not the dataset. The file was 882 KB of JSON
> that read `/kaggle/input/chess/games.csv` — a path that only exists on Kaggle's
> own servers.

`scripts/get_data.sh` automates this if you have a Kaggle API token.

## 3. Layout

```
data/            games.csv (never committed - 7.3 MB)
src/             chess_model.py - features, training, prediction
scripts/         get_data.sh - fetch the dataset
chess_ml.ipynb   the learning notebook, built step by step
app.py           Gradio web app
CLAUDE.md        the project rulebook
```

## 4. The rulebook: `CLAUDE.md`

`CLAUDE.md` is read automatically at the start of every session, so the rules
persist without being restated. It holds three kinds of rule:

- **Teaching rules** — explain before coding, one concept at a time, ask a check
  question after each step, and let the learner type the key lines
  (`fit`, `predict`, `train_test_split`) first.
- **Leakage rules** — which columns are banned, and why.
- **Working rules** — never commit `data/`, never invent dataset numbers, always
  `random_state=42`.

Writing the rules down once changed every later session.

## 5. The virtual environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

A **virtual environment** is a private copy of Python and its packages belonging
to one project.

Without one, `pip install pandas` puts pandas in a single shared location used by
everything on the machine. One project needing pandas 1.5 and another needing 3.0
then cannot coexist. With `.venv`, each project keeps its own copy — nothing is
shared, nothing conflicts, and deleting the folder undoes it all.

`requirements.txt` pins exact versions so anyone can rebuild the same setup.
`.venv/` itself is never committed: 697 MB of platform-specific binaries.

## 6. Loading the data

```python
import pandas as pd
games = pd.read_csv("data/games.csv")   # -> (20058, 16)
```

A CSV is plain text with commas separating values; the first line names the
columns. `read_csv` parses that into a **DataFrame** — a table with named columns
and numbered rows.

The chain used throughout:

| Step | Code | Term |
|---|---|---|
| 1 | the file on disk | CSV |
| 2 | `pd.read_csv(...)` | DataFrame |
| 3 | `games["winner"]` | column selection |
| 4 | `.value_counts()` | aggregation |

## 7. Patterns and the baseline

**Outcomes:** white 49.9%, black 45.4%, **draw 4.7%**.

The **baseline** is what you score by always guessing the most common answer:
**49.86%**. No model is worth anything until it beats that number. It is the zero
point on the ruler.

Three findings from `chess_ml.ipynb`:

1. **Rating difference dominates.** When black is 400+ points stronger, white wins
   15.6%. When white is 400+ stronger, white wins 84.2%.
2. **Draws need time and equality.** 3.1% in bullet games, 7.8% in classical, and
   they peak when ratings are level.
3. **The opening chart is confounded.** Van't Kruijs Opening shows white winning
   34.2% — but that is a beginner's opening. Weak players choose it and then lose
   for the reasons they chose it. The rating effect wearing a disguise.

## 8. Data leakage

**The most important idea in the project.**

The test for any column: *would I know this value before the first move?*

Banned by that test:

| Column | Why |
|---|---|
| `victory_status` | Says `mate` / `resign` / `draw` — it is the answer |
| `turns` | Only known once the game is over |
| `moves` | The whole game, checkmate included |
| `last_move_at` | A post-game timestamp |

Proof, measured on this dataset:

| Features | Accuracy |
|---|---|
| Baseline (always white) | 49.86% |
| Honest — pre-game only | 60.11% |
| **Leaky — plus `turns`, `victory_status`** | **71.79%** |

Leakage *raises* your score. That is what makes it dangerous: it looks like
success. A leaky model collapses the moment it meets a game that has not been
played yet — the columns it depends on do not exist.

When `victory_status` is `draw`, `winner` is `draw` in all 906 cases. The model
was not predicting. It was copying.

## 9. Features and the split

Seven features, all pre-game:

`white_rating`, `black_rating`, `rating_diff`, `rating_mean`, `rated`,
`base_min`, `increment_sec`

`rating_diff` is a **derived feature** — made by subtracting two columns. It ended
up being over half the model's decisions.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

Train on 80%, hide 20%. Testing on data the model studied proves nothing — like
revising from a book of answers and being tested on the same questions.
`stratify=y` keeps the 4.7% draw rate in both halves.

## 10. Models, and overfitting

Decision tree across `max_depth`:

| max_depth | train | test | gap | leaves |
|---|---|---|---|---|
| 2 | 61.4% | 61.1% | 0.3% | 4 |
| **4** | **62.2%** | **61.9%** | **0.3%** | 16 |
| 8 | 64.3% | 61.2% | 3.1% | 196 |
| 12 | 68.2% | 59.9% | 8.4% | 790 |
| None | **99.0%** | **57.4%** | **41.6%** | 5,364 |

The unlimited tree scored **99% on training data and 57.4% on test** — worse than
a 4-leaf tree. With 5,364 leaves it memorised individual games rather than
learning a pattern. That is **overfitting**, and the train/test *gap* is how you
detect it.

**The simplest real model won.** Depth 4, sixteen leaves.

A **random forest** — 300 trees, each on a random slice of data and features,
voting — reached **62.6%**. Its unlimited version also hit 99% on train but still
scored 62.5% on test: each tree memorises *different* noise, so voting cancels it.

**Feature importance:** `rating_diff` 50.9%, `black_rating` 15.9%,
`white_rating` 15.4%, `rating_mean` 10.3%. Rating is 92.5% of the model.

## 11. The Gradio app

```bash
.venv/bin/python app.py     # http://127.0.0.1:7860
```

**Gradio** turns a Python function into a web page without HTML, CSS or
JavaScript. You describe the inputs and outputs as *components*; it builds the
interface and runs a small local web server.

```python
white = gr.Slider(600, 2800, value=1500, label="White rating")
go.click(predict, inputs, outputs)   # on click: call predict, show the result
```

The model trains once at startup, not on every click. `predict_proba` is used
instead of `predict`, so the app returns a probability per outcome rather than a
single label.

Example — 1500 vs 1800: **white 28.8%, black 66.4%, draw 4.8%.**

## 12. Running it

```bash
git clone <this repo>
cd Chess_Winner_Predictor
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# download games.csv into data/
.venv/bin/jupyter lab chess_ml.ipynb    # the notebook
.venv/bin/python app.py                 # the web app
```

## Honest limits

- **62.6% accuracy**, 12.7 points above baseline. Real, and modest.
- The model **almost never predicts a draw** — at 4.7% of games, ignoring draws
  is nearly free in accuracy terms. Accuracy hides this; per-class metrics do not.
- Opening columns were deliberately excluded as confounded with player strength.
- These are online games from one site in 2017. Nothing here generalises to
  over-the-board or elite play.
