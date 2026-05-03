#!/bin/bash
# Run 4 evaluation configurations for baseline Qwen model
# Configurations: (zero-shot / few-shot) × (no-LoRA / with-LoRA)

set -e

export PYTHONPATH=/storage/arik/nlp_final_project
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SCRIPT=".venv/bin/python scripts/eval_base_model.py"
MAX_NEW_TOKENS=1024
OUTPUT_DIR="eval_results"

echo "=============================================="
echo "Running 4 baseline evaluation configurations"
echo "=============================================="

# 1. Zero-shot, no LoRA
echo ""
echo ">>> [1/4] Baseline + Zero-shot + no LoRA"
$SCRIPT \
    --output $OUTPUT_DIR/baseline_zeroShot_noLora.json \
    --max-new-tokens $MAX_NEW_TOKENS \
    --zero-shot

# 2. Zero-shot, with LoRA
echo ""
echo ">>> [2/4] Baseline + Zero-shot + with LoRA"
$SCRIPT \
    --output $OUTPUT_DIR/baseline_zeroShot_withLora.json \
    --max-new-tokens $MAX_NEW_TOKENS \
    --zero-shot \
    --with-lora-init

# 3. Few-shot, no LoRA
echo ""
echo ">>> [3/4] Baseline + Few-shot + no LoRA"
$SCRIPT \
    --output $OUTPUT_DIR/baseline_fewShot_noLora.json \
    --max-new-tokens $MAX_NEW_TOKENS

# 4. Few-shot, with LoRA
echo ""
echo ">>> [4/4] Baseline + Few-shot + with LoRA"
$SCRIPT \
    --output $OUTPUT_DIR/baseline_fewShot_withLora.json \
    --max-new-tokens $MAX_NEW_TOKENS \
    --with-lora-init

echo ""
echo "=============================================="
echo "All 4 baseline evaluations completed!"
echo "=============================================="