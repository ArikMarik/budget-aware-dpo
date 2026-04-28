#!/bin/bash
# Phase 4 Pipeline — runs sequentially on GPU 0
# Steps:
#   1. Wait for hard eval (MATH L3-5) if still running
#   2. Run 8-shot base model eval (GSM8K + MATH L1-2)
#   3. Run SFT training (3 epochs)
#   4. Run 0-shot SFT eval (GSM8K + MATH L1-2)
#   5. Run 8-shot SFT eval (GSM8K + MATH L1-2)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_phase4_pipeline.sh > logs/phase4_pipeline.log 2>&1 &

set -e
export PYTHONPATH=/storage/arik/nlp_final_project
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== PHASE 4 PIPELINE START ==="

# Step 1: Wait for hard eval if still running
HARD_PID=$(pgrep -f "eval_base_model.*hard_256" || true)
if [ -n "$HARD_PID" ]; then
    log "Step 1: Waiting for hard eval (PID $HARD_PID) to finish..."
    while kill -0 "$HARD_PID" 2>/dev/null; do sleep 60; done
    log "Step 1: Hard eval finished."
else
    log "Step 1: Hard eval not running, skipping wait."
fi

# Step 2: 8-shot base model eval
log "Step 2: Running 8-shot base model eval (GSM8K + MATH L1-2)..."
.venv/bin/python scripts/eval_base_model.py \
    --output eval_results/base_qwen_0.5b_easy_8shot.json \
    --use-real --math-levels 1,2 --max-new-tokens 256 --few-shot 8
log "Step 2: 8-shot base eval complete."

# Step 3: SFT training
log "Step 3: Running SFT training (3 epochs, lr=2e-5, batch=4)..."
DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
.venv/bin/python -m scripts.training.train_sft \
    --output-dir checkpoints/sft_v1 \
    --max-epochs 3 \
    --batch-size 4 \
    --lr 2e-5 \
    --gradient-accumulation-steps 4 \
    --max-length 512 \
    --run-name sft_v1_phase4 \
    --wandb
log "Step 3: SFT training complete."

# Step 4: 0-shot SFT eval
log "Step 4: Running 0-shot SFT eval (GSM8K + MATH L1-2)..."
.venv/bin/python scripts/eval_base_model.py \
    --output eval_results/sft_v1_easy_0shot.json \
    --use-real --math-levels 1,2 --max-new-tokens 256
# Note: This evals the base model again. For SFT checkpoint, use eval_checkpoint.py
.venv/bin/python scripts/eval_checkpoint.py \
    --checkpoint checkpoints/sft_v1/best-model \
    --output eval_results/sft_v1_easy_0shot.json \
    --use-real --limit 2650
log "Step 4: 0-shot SFT eval complete."

# Step 5: 8-shot SFT eval
log "Step 5: Running 8-shot SFT eval (GSM8K + MATH L1-2)..."
.venv/bin/python scripts/eval_checkpoint.py \
    --checkpoint checkpoints/sft_v1/best-model \
    --output eval_results/sft_v1_easy_8shot.json \
    --use-real --limit 2650 --few-shot 8
log "Step 5: 8-shot SFT eval complete."

log "=== PHASE 4 PIPELINE COMPLETE ==="
log "Results:"
log "  Base 0-shot: eval_results/base_qwen_0.5b_easy_256.json"
log "  Base 8-shot: eval_results/base_qwen_0.5b_easy_8shot.json"
log "  Base hard:   eval_results/base_qwen_0.5b_hard_256.json"
log "  SFT 0-shot:  eval_results/sft_v1_easy_0shot.json"
log "  SFT 8-shot:  eval_results/sft_v1_easy_8shot.json"
