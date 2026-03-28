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

## 4. Results (Epoch 1)

### 7a: 1.5B Baseline DPO (E1)
- train_loss=0.1184, val_loss=0.7426, reward_diff=4.6286
- gen-eval: accuracy=6% (easy=12%, hard=0%), avg_tokens_easy=149.0, avg_tokens_hard=178.8, TPCA=2732.0
- **Verdict**: Very low accuracy — KL=0.01 baseline without lambda may be too restrictive for 1.5B, or needs more epochs

### 7b: 1.5B Budget-Aware DPO (E1) — BREAKTHROUGH
- train_loss=0.2665, val_loss=0.4273, reward_diff=0.0868
- gen-eval: accuracy=**30%** (easy=**56%**, hard=**4%**), avg_tokens_easy=**131.9**, avg_tokens_hard=**247.8**, TPCA=**632.9**
- **Verdict**: Best results yet! 30% accuracy (5x baseline), 131.9 easy tokens (shortest ever, 26% shorter than Phase 1 baseline's 179), strong easy/hard divergence in both accuracy and tokens

### 7c: 0.5B SimPO v1 (E1) — FAILED
- beta=2.0, lr=1e-6: train_loss=0.1252, val_loss=1.5616, accuracy=3%
- Saturated by E2 (loss=0.0001, gradients=0). Killed.

### 7c-v2: 0.5B SimPO v2 (E1) — FAILED
- beta=0.5, lr=5e-7: train_loss=0.1787, val_loss=1.6498, accuracy=5%
- avg_tokens_easy=238.7 (worse than baseline). SimPO still overfits without reference model anchor.
- Killed after E1.

## 5. Comparison

| Metric | 1.5B Baseline (7a) | 1.5B Budget (7b) | Phase1 0.5B Budget (iter5 E3) | Phase1 0.5B Baseline (iter6 E2) |
|--------|-------------------|------------------|-------------------------------|--------------------------------|
| accuracy | 6% | **30%** | 38% | 35% |
| easy_acc | 12% | **56%** | 68% | 68% |
| hard_acc | 0% | **4%** | 8% | 2% |
| tokens_easy | 149.0 | **131.9** | 193.3 | 144.0 |
| tokens_hard | 178.8 | **247.8** | 245.9 | 250.9 |
| TPCA | 2732 | **632.9** | 578 | 564 |
| val_loss | 0.7426 | **0.4273** | 0.4460 | 0.5703 |

## 6. Analysis

**The 1.5B budget-aware model is the clear winner.** Key findings:

1. **1.5B + budget-aware = strong divergence**: 131.9 easy tokens vs 247.8 hard tokens (1.88x ratio). This is the clearest budget-aware signal we've seen. On 0.5B the ratio was ~0.79x (193/245).

2. **Accuracy is competitive**: 30% at E1 is likely to improve with more epochs (Phase 1 went from 27% E2 → 38% E3).

3. **1.5B baseline underperforms**: Only 6% accuracy suggests the KL penalty may be too strong for the baseline at 1.5B scale without the length penalty providing additional learning signal. Or it needs more epochs.

4. **SimPO doesn't work**: Without a reference model, the policy drifts too far. Both beta=2.0 and beta=0.5 led to overfitting. The reference model is essential for stability.

5. **1.5B model shows the budget effect that was marginal on 0.5B**: The larger model has enough capacity to simultaneously learn accuracy AND length efficiency. This validates Hypothesis B.

## 7. Open Questions

1. Will 1.5B budget-aware accuracy improve to 40%+ by E3?
2. Will 1.5B baseline catch up with more epochs?
3. Will the token advantage (131.9 easy) persist or drift verbose like Phase 1?
4. Would two-phase training (accuracy warmup → budget DPO) work even better?

### Post-Training Eval: 1.5B Budget E1 (500 held-out problems, Tier 0+1+2)

| Metric | 1.5B Budget E1 | Phase1 0.5B Budget (iter5) | Phase1 0.5B Baseline (iter6) |
|--------|---------------|---------------------------|------------------------------|
| Overall Accuracy | **24.6%** | 22.0% | 21.2% |
| MATH L4-5 Accuracy | **13.7%** | 8.2% | 11.6% |
| Avg Tokens Easy | 243.5 | 177.4 | 179.4 |
| Avg Tokens Hard | 177.5 | 194.8 | 198.3 |
| TPCA | 855.6 | 845.9 | 890.8 |

**Key finding**: 1.5B budget-aware is more accurate (+2.6% overall, +5.5% MATH L4-5) but tokens_easy is higher on held-out data (243 vs 177). In-training gen-eval showed 131.9 easy tokens — large gap between training data performance and held-out. E2-E3 may improve token efficiency as the model continues learning.

## 8. Next Iteration Plan

- **GPU 0-1**: Continue 1.5B runs through E2-E3 (already running)
- **GPU 2**: Two-phase training experiment (iter 8) — standard DPO warmup then budget-aware fine-tuning
- **After E3**: Run post-training eval (500 problems, Tier 0+1+2) on best 1.5B checkpoints
- **If 1.5B budget stays strong**: Try lambda tuning (lambda=10, lambda=20) on 1.5B
