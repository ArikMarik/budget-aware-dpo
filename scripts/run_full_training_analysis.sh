#!/bin/bash
# Full pipeline: enrich training data with levels, then run complete analysis (sections 1-8) on 1M examples.
# Prerequisites: real_openmathinstruct.jsonl with 1M+ examples.
#   If missing or truncated: python scripts/load_real_data.py --split train_1M

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
mkdir -p logs

echo "=== Step 1: Enrich training data with MATH levels ==="
python scripts/enrich_training_data_with_levels.py

echo ""
echo "=== Step 2: Run full analysis (sections 1-8) on 1M training examples ==="
nohup env PYTHONUNBUFFERED=1 python scripts/analyze_complexity_heuristics.py \
  --training-only \
  --training data/real_openmathinstruct.jsonl \
  --training-limit 1000000 \
  -o docs/feature_reports/report_complexity_heuristics_analysis.md \
  > logs/analyze_complexity_heuristics.log 2>&1 &

echo "PID: $!"
echo "Log: logs/analyze_complexity_heuristics.log"
echo "Report: docs/feature_reports/report_complexity_heuristics_analysis.md"
echo ""
echo "Monitor: tail -f logs/analyze_complexity_heuristics.log"
