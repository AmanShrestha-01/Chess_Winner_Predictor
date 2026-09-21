"""Gradio web app for the chess winner predictor.

Run it with:  .venv/bin/python app.py
Then open http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no GUI backend on a web server
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

import joblib
import pandas as pd

# Fixed colour per outcome, used everywhere in the app
C = {"white": "#2a78d6", "black": "#eb6834", "draw": "#1baf7a"}
BASELINE = 49.86


def predict_winner(model, white_rating: int, black_rating: int,
                   time_control: str = "10+0", rated: bool = True) -> dict[str, float]:
    """Probability of each outcome, as percentages that sum to 100.

    >>> predict_winner(model, 1500, 1800, "10+0")
    {'white': 26.4, 'black': 68.1, 'draw': 5.5}
    """
    base, _, inc = time_control.partition("+")

    # One row, same columns in the same order the model was trained on
    row = pd.DataFrame([{
        "white_rating": white_rating,
        "black_rating": black_rating,
        "rating_diff": white_rating - black_rating,
        "rating_mean": (white_rating + black_rating) / 2,
        "rated": int(rated),
        "base_min": pd.to_numeric(base, errors="coerce"),
        "increment_sec": pd.to_numeric(inc or 0, errors="coerce"),
    }])

    # predict_proba gives a probability per class, not just the winning label
    probs = model.predict_proba(row)[0]
    return {cls: round(float(p) * 100, 1) for cls, p in zip(model.classes_, probs)}


_bundle = joblib.load(Path(__file__).parent / "model.joblib")
MODEL, TEST_ACC = _bundle["model"], _bundle["test_acc"]
print(f"loaded model — test accuracy {TEST_ACC:.1%}")


def _bars(probs: dict) -> str:
    """The result card: a headline, then one labelled bar per outcome."""
    top = max(probs, key=probs.get)
    headline = {"white": "White is favoured", "black": "Black is favoured",
                "draw": "A draw is most likely"}[top]

    rows = ""
    for outcome in ("white", "black", "draw"):
        pct = probs[outcome]
        rows += f"""
        <div class="row">
          <div class="name">{outcome}</div>
          <div class="track"><div class="fill" style="width:{max(pct,0.8)}%;
               background:{C[outcome]}"></div></div>
          <div class="pct">{pct:.1f}%</div>
        </div>"""

    return f"""
    <div class="card">
      <div class="headline" style="color:{C[top]}">{headline}</div>
      {rows}
    </div>"""


def _curve(white_rating, black_rating, tc, rated):
    """How the odds shift as the opponent's rating changes, holding yours fixed."""
    lo, hi = 600, 2800
    opponents = np.arange(lo, hi + 1, 50)

    series = {k: [] for k in ("white", "black", "draw")}
    for opp in opponents:
        p = predict_winner(MODEL, int(white_rating), int(opp), tc, rated)
        for k in series:
            series[k].append(p[k])

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for outcome in ("white", "black", "draw"):
        ax.plot(opponents, series[outcome], lw=2.4, color=C[outcome], label=outcome)

    # Where the sliders currently sit
    ax.axvline(black_rating, color="#8a8a86", lw=1.2, ls="--")
    ax.text(black_rating, 101, " current", fontsize=8.5, color="#8a8a86", va="bottom")

    ax.set_xlabel("black's rating", fontsize=9.5, color="#8a8a86")
    ax.set_ylabel("probability (%)", fontsize=9.5, color="#8a8a86")
    ax.set_title(f"With white fixed at {int(white_rating)}",
                 fontsize=11, fontweight="bold", color="#8a8a86", loc="left")
    ax.set_ylim(0, 100)
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="upper right",
              labelcolor="#8a8a86")
    ax.grid(color="#8a8a86", alpha=0.18, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8a86")
        ax.spines[s].set_alpha(0.4)
    ax.tick_params(colors="#8a8a86", labelsize=8.5)
    plt.tight_layout()
    return fig


def predict(white_rating, black_rating, tc, rated):
    probs = predict_winner(MODEL, int(white_rating), int(black_rating), tc, rated)
    diff = int(white_rating) - int(black_rating)

    if diff == 0:
        note = "Evenly matched."
    else:
        side = "White" if diff > 0 else "Black"
        note = f"{side} is stronger by **{abs(diff)}** rating points."

    caption = (
        f"{note}  \n"
        f"<sub>Random forest · {TEST_ACC:.1%} accuracy on held-out games · "
        f"baseline (always guess white) {BASELINE}%</sub>"
    )
    return _bars(probs), caption, _curve(white_rating, black_rating, tc, rated)


CSS = """
.gradio-container { max-width: 1120px !important; }
#title h1 { margin-bottom: 2px; font-size: 1.9rem; }
#title p  { color: var(--body-text-color-subdued); margin-top: 0; }

.card { padding: 4px 2px 2px; }
.card .headline { font-size: 1.35rem; font-weight: 700; margin-bottom: 14px; }
.card .row { display: flex; align-items: center; gap: 12px; margin-bottom: 9px; }
.card .name { width: 52px; font-size: 0.87rem; text-transform: capitalize;
              color: var(--body-text-color-subdued); }
.card .track { flex: 1; height: 15px; border-radius: 4px;
               background: var(--background-fill-secondary); overflow: hidden; }
.card .fill { height: 100%; border-radius: 4px; transition: width .28s ease; }
.card .pct { width: 52px; text-align: right; font-variant-numeric: tabular-nums;
             font-weight: 700; font-size: 0.93rem; }
"""

theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="Chess Winner Predictor") as demo:
    gr.Markdown(
        "# ♟ Chess Winner Predictor\n"
        "Predicts the result from **pre-game information only** — ratings and the "
        "clock. It never sees a single move.",
        elem_id="title",
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            white = gr.Slider(600, 2800, value=1500, step=10, label="⚪ White rating")
            black = gr.Slider(600, 2800, value=1500, step=10, label="⚫ Black rating")
            tc = gr.Dropdown(
                ["1+0", "3+0", "3+2", "5+0", "5+3", "10+0", "15+2", "30+0", "60+0"],
                value="10+0", label="⏱ Time control (minutes + increment)",
            )
            rated = gr.Checkbox(value=True, label="Rated game")

            gr.Markdown("<sub>**Try one:**</sub>")
            with gr.Row():
                b_even = gr.Button("Even match", size="sm")
                b_under = gr.Button("You vs +300", size="sm")
                b_gm = gr.Button("Club vs GM", size="sm")
                b_bullet = gr.Button("Bullet chaos", size="sm")

        with gr.Column(scale=5):
            out_bars = gr.HTML()
            out_note = gr.Markdown()
            out_plot = gr.Plot(label=None, container=False)

    with gr.Accordion("How it works, and what it can't do", open=False):
        gr.Markdown(
            f"""
**Features used** — white rating, black rating, their difference, their average,
whether the game is rated, and the time control split into base minutes and
increment seconds. Seven numbers, all knowable before the first move.

**Deliberately excluded** — `victory_status`, `turns`, `moves` and `last_move_at`.
Those describe how the game *ended*. Including them lifts accuracy to 71.8%, which
looks better and is worthless: they don't exist for a game that hasn't been played.

**Honest limits** — {TEST_ACC:.1%} accuracy against a {BASELINE}% baseline. The model
rarely predicts a draw, because draws are only 4.7% of games. Trained on ~20k
Lichess games from 2017; nothing here transfers to over-the-board or elite play.
"""
        )

    inputs, outputs = [white, black, tc, rated], [out_bars, out_note, out_plot]

    for component in inputs:
        component.change(predict, inputs, outputs)

    b_even.click(lambda: (1500, 1500, "10+0"), None, [white, black, tc])
    b_under.click(lambda: (1500, 1800, "10+0"), None, [white, black, tc])
    b_gm.click(lambda: (1600, 2600, "15+2"), None, [white, black, tc])
    b_bullet.click(lambda: (1500, 1500, "1+0"), None, [white, black, tc])

    demo.load(predict, inputs, outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme, css=CSS)
