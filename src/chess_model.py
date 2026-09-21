"""Train the chess winner model and predict from pre-game information only.

Every feature here is knowable BEFORE the first move. See CLAUDE.md for the
leakage rules this module is built to respect.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data" / "games.csv"
SEED = 42

# Columns that describe how the game ENDED. Never features.
BANNED = ("victory_status", "turns", "moves", "last_move_at")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw games table in, numeric feature matrix out. Pre-game columns only."""
    X = pd.DataFrame(index=df.index)
    X["white_rating"] = df["white_rating"]
    X["black_rating"] = df["black_rating"]
    X["rating_diff"] = df["white_rating"] - df["black_rating"]
    X["rating_mean"] = (df["white_rating"] + df["black_rating"]) / 2
    X["rated"] = df["rated"].astype(int)

    # "15+2" -> 15 minutes base, 2 seconds added per move
    tc = df["increment_code"].astype(str).str.split("+", n=1, expand=True)
    X["base_min"] = pd.to_numeric(tc[0], errors="coerce")
    X["increment_sec"] = pd.to_numeric(tc[1], errors="coerce")
    return X


def train_model(data_path: Path = DATA):
    """Train the random forest. Returns (model, test_accuracy)."""
    games = pd.read_csv(data_path)
    X, y = build_features(games), games["winner"]

    assert not [c for c in X.columns if c in BANNED], "a banned column reached the features"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=SEED, n_jobs=-1
    ).fit(X_train, y_train)

    return model, model.score(X_test, y_test)


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
