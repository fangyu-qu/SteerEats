#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${STEEREATS_PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  printf '%s\n' 'Create .venv and install requirements.txt before running.' >&2
  exit 1
fi
if [ ! -f data/processed/sample_interactions.parquet ]; then
  "$PYTHON" scripts/download_data.py --from-directory "$HOME/Downloads"
  "$PYTHON" src/01_build_nyc_restaurants.py
  "$PYTHON" src/02_extract_nyc_reviews.py
  "$PYTHON" src/04_sample_and_split.py
fi
mkdir -p outputs/logistic_ranker
"$PYTHON" -u src/07_logistic_ranker.py \
  --interactions data/processed/sample_interactions.parquet \
  --out outputs/logistic_ranker
if [ -f report/main.tex ]; then
  "$PYTHON" scripts/update_report.py
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=report report/main.tex
else
  printf '%s\n' 'Model and evaluation saved. Report source is managed separately in Overleaf.'
fi
