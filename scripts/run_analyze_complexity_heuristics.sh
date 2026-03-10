#!/bin/bash
# Run complexity heuristics analysis (full run, background)
# Output: docs/feature_reports/report_complexity_heuristics_analysis.md

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

mkdir -p logs

nohup python scripts/analyze_complexity_heuristics.py --no-training \
  -o docs/feature_reports/report_complexity_heuristics_analysis.md \
  > logs/analyze_complexity_heuristics.log 2>&1 &

echo "PID: $!"
echo "Log: logs/analyze_complexity_heuristics.log"
echo "Report: docs/feature_reports/report_complexity_heuristics_analysis.md"
