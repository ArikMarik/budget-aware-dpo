# Iteration 8 — Two-Phase Training (0.5B)

**Date**: 2026-03-28
**Branch**: `autoresearch/mar26`

## 1. Hypothesis

Two-phase training separates accuracy learning from efficiency learning:
- **Phase 1 (warmup)**: Standard DPO for 1 epoch — learn to solve problems correctly
- **Phase 2 (budget)**: Budget-aware DPO with lambda_easy=5.0 for 2 epochs — learn to be concise on easy problems

This tests Hypothesis E from PHASE1_SUMMARY: the model first learns accuracy, then learns length efficiency, avoiding the tension between the two objectives.

## 2. Hyperparameters

### Phase 1: Standard DPO Warmup (1 epoch)
```bash
CUDA_VISIBLE_DEVICES=2 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/twophase_warmup_iter8 \
  --max-epochs 1 --batch-size 4 --lr 1e-6 --kl-penalty 0.01 \
  --run-name twophase_warmup_iter8 --wandb
```

### Phase 2: Budget-Aware DPO (2 epochs, resumed from warmup)
```bash
CUDA_VISIBLE_DEVICES=2 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/twophase_budget_iter8 \
  --resume-from checkpoints/twophase_warmup_iter8/best-model \
  --max-epochs 2 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --run-name twophase_budget_iter8 --wandb
```

## 3. Changes Made

No code changes — uses existing `--resume-from` parameter.

## 4. Results

### Phase 1: Warmup (1 epoch standard DPO)
- train_loss=0.1203, val_loss=0.6322, reward_diff=5.3295
- gen-eval: accuracy=3% (easy=2%, hard=4%), avg_tokens_easy=90.9, avg_tokens_hard=142.0, TPCA=3880.3
- Very short tokens but low accuracy — expected for standard DPO warmup

### Phase 2: Budget DPO E1 (resumed from warmup)
- train_loss=0.2747, val_loss=0.4177, reward_diff=0.0454
- gen-eval: accuracy=**25%** (easy=**46%**, hard=4%), avg_tokens_easy=**160.4**, avg_tokens_hard=228.2, TPCA=777.1
- Significant accuracy jump from 3% → 25% with budget-aware objective

### Phase 2: Budget DPO E2
*Pending — in progress*

## 5. Comparison

| Metric | Two-Phase E1 | 1.5B Budget E1 | Phase1 0.5B Budget (iter5 E3) |
|--------|-------------|----------------|-------------------------------|
| accuracy | 25% | 30% | 38% |
| easy_acc | 46% | 56% | 68% |
| tokens_easy | 160.4 | 131.9 | 193.3 |
| tokens_hard | 228.2 | 247.8 | 245.9 |
| TPCA | 777.1 | 632.9 | 578 |
| val_loss | 0.4177 | 0.4273 | 0.4460 |

## 6. Analysis

The two-phase approach works but is **inferior to 1.5B budget-aware DPO**:
- Lower accuracy (25% vs 30%)
- Longer easy tokens (160 vs 132)
- Comparable val_loss

The warmup phase produced very short but inaccurate responses (3% accuracy, 91 easy tokens). When budget-aware DPO was applied, accuracy jumped to 25% but tokens increased to 160. This suggests the warmup didn't provide a strong enough accuracy foundation — it may need more warmup epochs.

However, two-phase on 0.5B (25% accuracy) is close to the 1.5B budget result (30%), which suggests the approach has merit — the model size is the bigger factor.

## 7. Open Questions

1. Would 2 warmup epochs give a stronger accuracy foundation?
2. Would two-phase training on the 1.5B model be even better?
3. Is the warmup→budget transition smooth or does it undo warmup learning?

## 8. Next Iteration Plan

Focus on 1.5B model which shows the strongest results. Two-phase on 1.5B could be interesting but the single-phase 1.5B budget-aware already shows strong signal.
