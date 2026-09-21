#!/usr/bin/env bash
# Fetch the Lichess games dataset (datasnaek/chess) into data/games.csv
#
# Prerequisites (one time):
#   1. pip install kaggle
#   2. Kaggle -> Settings -> API -> "Create New Token"  (downloads kaggle.json)
#   3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
set -euo pipefail

DATASET="datasnaek/chess"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it with: pip install kaggle" >&2
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_KEY:-}" ]; then
  echo "No Kaggle credentials found at ~/.kaggle/kaggle.json (see header of this script)." >&2
  exit 1
fi

mkdir -p "$DEST"
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip
echo "Done: $DEST/games.csv"
wc -l "$DEST/games.csv"
