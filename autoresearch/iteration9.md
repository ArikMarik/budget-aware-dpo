# Iteration 9 — 1.5B Baseline Without KL + Eval Round

**Date**: 2026-03-28
**Branch**: `autoresearch/mar26`

## 1. Hypothesis

The 1.5B baseline with KL=0.01 only reached 5-6% accuracy (iter 7a). This is puzzlingly bad — the budget-aware model with the same KL hit 43% at E2. Two possible explanations:

1. **KL=0.01 is too strong for baseline**: Without the length penalty providing additional gradient signal, the KL may over-constrain the baseline.
2. **The length penalty helps learning**: Lambda provides a useful learning signal that accelerates convergence on 1.5B.

This iteration tests hypothesis 1 by running baseline with KL=0.0.

## 2. Hyperparameters

```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_1.5b_noKL_iter9 \
  --model Qwen/Qwen2.5-1.5B \
  --max-epochs 3 --batch-size 2 --lr 1e-6 --kl-penalty 0.0 \
  --run-name baseline_1.5b_noKL_iter9 --wandb
```

## 3. Changes Made

No code changes.

## 4. Results

*In progress — launched at 17:58 UTC*

## 5. Evaluation Plan

After training completes, run post-training eval on best checkpoints:
- 1.5B Budget best epoch (E1 or E2 — whichever has best val_loss)
- 1.5B Baseline no-KL best epoch
- Compare: accuracy, TPCA, avg_tokens_easy, avg_tokens_hard
