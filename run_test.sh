#!/usr/bin/env bash
set -e
INPUT="${1:-input.mp3}"
mkdir -p results
bash preprocess.sh "$INPUT"
source .venv/bin/activate
python python_tagger.py
node js_tagger.js
python compare.py
cat results/comparison_report.txt
