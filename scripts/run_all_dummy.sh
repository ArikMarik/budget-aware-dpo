#!/bin/bash
# Run entire pipeline end-to-end on dummy data in under 2 minutes.
# Usage: USE_DUMMY_DATA=1 ./scripts/run_all_dummy.sh

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

export USE_DUMMY_DATA=1

echo "=== Phase 1: Generate dummy data ==="
python scripts/generate_dummy_data.py

echo "=== Phase 1: Model load check ==="
python scripts/check_model_load.py

echo "=== Phase 2: Data preprocessing ==="
python scripts/preprocess_dpo_data.py

echo "=== Phase 3: Sanity check (overfitting) ==="
python scripts/train_sanity_check.py

echo "=== Dummy pipeline complete ==="
