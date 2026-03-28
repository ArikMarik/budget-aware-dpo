# Iteration 7 — Phase 2: SimPO + 1.5B Model (3 Parallel Experiments)

**Date**: 2026-03-28
**Branch**: `autoresearch/mar26`

## 1. Hypothesis

Three parallel experiments testing the top Phase 2 hypotheses:

### Experiment 7a: 1.5B Baseline DPO (GPU 0)
**Hypothesis**: A larger model (Qwen2.5-1.5B, 3x params) with KL=0.01 establishes a stronger baseline. Higher baseline accuracy (expected 25-35% vs 22%) gives more room to demonstrate budget-aware improvements.

### Experiment 7b: 1.5B Budget-Aware DPO (GPU 1)
**Hypothesis**: The budget-aware length penalty (lambda_easy=5.0) that showed marginal effect on 0.5B may show stronger effect on 1.5B, which has more capacity to learn nuanced behavior.

### Experiment 7c: 0.5B SimPO Budget-Aware (GPU 2)
**Hypothesis**: SimPO (Simple Preference Optimization) uses length-normalized log-probs as implicit reward: `R(x,y) = (beta/|y|) * sum(log pi) - gamma`. This built-in length normalization naturally penalizes verbosity. Combined with budget-aware lambda, it could produce stronger token reduction on easy problems than standard DPO.

## 2. Hyperparameters

### 7a: 1.5B Baseline DPO
```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_1.5b_iter7a \
  --model Qwen/Qwen2.5-1.5B \
  --max-epochs 3 --batch-size 2 --lr 1e-6 \
  --kl-penalty 0.01 \
  --early-stopping-patience 3 \
  --run-name baseline_1.5b_iter7a --wandb \
  > logs/baseline_1.5b_iter7a.log 2>&1 &
```

### 7b: 1.5B Budget-Aware DPO
```bash
CUDA_VISIBLE_DEVICES=1 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_1.5b_iter7b \
  --model Qwen/Qwen2.5-1.5B \
  --max-epochs 3 --batch-size 2 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --early-stopping-patience 3 \
  --run-name budget_1.5b_iter7b --wandb \
  > logs/budget_1.5b_iter7b.log 2>&1 &
```

### 7c: 0.5B SimPO Budget-Aware
```bash
CUDA_VISIBLE_DEVICES=2 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/simpo_budget_0.5b_iter7c \
  --loss-type simpo --dpo-beta 2.0 \
  --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 \
  --early-stopping-patience 3 \
  --run-name simpo_budget_0.5b_iter7c --wandb \
  > logs/simpo_budget_0.5b_iter7c.log 2>&1 &
```

**Key differences from Phase 1**:
- 1.5B model uses batch_size=2 (larger model, more memory)
- SimPO uses beta=2.0 (standard SimPO default, vs DPO beta=0.1)
- SimPO has no KL penalty (no reference model in loss)
- All use lr=1e-6 (proven stable in Phase 1)

## 3. Changes Made

- `src/models/simpo_loss.py` — NEW: SimPO loss function with optional budget-aware penalty
- `src/training/dpo_trainer.py` — Added `model_name` and `loss_type` parameters to `train_dpo()` and `_build_loss_fn()`
- `scripts/training/train_budget_aware_dpo.py` — Added `--model` and `--loss-type` CLI args
- `scripts/training/train_baseline_dpo.py` — Added `--model` and `--kl-penalty` CLI args
- `scripts/eval_checkpoint.py` — Added `--base-model` arg
- `src/evaluation/run_evaluation.py` — Added `base_model` parameter to `evaluate_checkpoint()`

## 4. Results

*To be filled after training completes.*

## 5. Comparison to Baseline

*To be filled after training completes.*

## 6. Analysis

*To be filled after training completes.*

## 7. Open Questions

1. Does 1.5B have enough headroom on MATH to show meaningful budget-aware divergence?
2. Does SimPO's built-in length normalization translate to shorter generation?
3. Is batch_size=2 sufficient for 1.5B, or do we need gradient accumulation?

## 8. Next Iteration Plan

*To be decided based on results.*
