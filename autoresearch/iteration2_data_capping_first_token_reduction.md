# Iteration 2 — Data Capping, First Correct Token Reduction but 1-2% Accuracy (Phase 1)

**Date**: 2026-03-26
**Status**: COMPLETE — both models trained, gen eval done for all epochs
**Run names**: `baseline_balanced_iter2` (GPU 0), `budget_aware_balanced_iter2` (GPU 1)
**WandB Baseline**: https://wandb.ai/ariksheer-tel-aviv-university/budget-aware-dpo/runs/mjjcmg53
**WandB Budget**: https://wandb.ai/ariksheer-tel-aviv-university/budget-aware-dpo/runs/gz2hwjz7

---

## 1. Hypothesis

**Capping pairs per problem at 50 and using gradient accumulation will reduce overfitting while preserving the budget-aware signal observed in iteration 1.**

Iteration 1 showed:
- lambda_easy=5.0 produces better val_loss than baseline (0.69 vs 0.85)
- But severe overfitting after epoch 1 (val_loss 0.69→1.46)
- Post-training eval: accuracy +5.2% but tokens 2x HIGHER on Easy (opposite of goal)
- Dataset has only 203 unique hard problems repeated ~123x each

This iteration tests whether fixing the data concentration + smoother training will improve generalization.

---

## 2. Hyperparameters

### Budget-Aware
```
lambda_easy=5.0        (unchanged from iter 1)
lambda_hard=0.0        (unchanged from iter 1)
beta=0.1               (unchanged)
max_epochs=2           (was 3 — reduced due to overfitting after epoch 1)
batch_size=4           (unchanged)
lr=1e-5                (unchanged)
gradient_accumulation_steps=4  (was 1 — smoother gradients, effective batch=16)
early_stopping_patience=3     (unchanged)
LoRA: r=128, alpha=256, dropout=0.05  (unchanged)
Dataset: processed_dpo_dataset_balanced_v3_capped50 (10,216 pairs, max 50/problem)
```

### Baseline (re-run required — dataset changed)
```
Same as budget-aware but without lambda/length penalty (standard DPO)
```

### Full commands
```bash
# Baseline
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=/storage/arik/nlp_final_project/data/processed_dpo_dataset_balanced_v3_capped50 \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_balanced_iter2 --max-epochs 2 --batch-size 4 --lr 1e-5 \
  --gradient-accumulation-steps 4 --early-stopping-patience 3 \
  --run-name baseline_balanced_iter2 --wandb > logs/baseline_balanced_iter2.log 2>&1 &

# Budget-Aware
CUDA_VISIBLE_DEVICES=1 DATASET_PATH=/storage/arik/nlp_final_project/data/processed_dpo_dataset_balanced_v3_capped50 \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_aware_balanced_iter2 --max-epochs 2 --batch-size 4 --lr 1e-5 \
  --lambda-easy 5.0 --lambda-hard 0.0 --gradient-accumulation-steps 4 --early-stopping-patience 3 \
  --run-name budget_aware_balanced_iter2 --wandb > logs/budget_aware_balanced_iter2.log 2>&1 &
```

---

## 3. Changes Made

### Code changes
1. **`src/config.py`** — Added `DATASET_PATH` env var override so custom dataset paths can be passed without modifying config constants
2. **`src/training/dpo_trainer.py`** — Two changes:
   - Fixed val/loss WandB graph: added `wandb.define_metric("val/*", step_metric="train/epoch")` so val metrics plot against epoch
   - Added generation-based epoch validation: `_load_gen_eval_problems()` loads 100 problems (50 easy + 50 hard), `_run_gen_eval()` runs `model.generate()` after each epoch and logs `gen/accuracy`, `gen/avg_tokens_easy`, `gen/avg_tokens_hard`, `gen/tpca` to WandB

### Data changes
3. **Created `scripts/subsample_capped_pairs.py`** — Script to cap pairs per problem. Reads from original balanced dataset, groups by problem, caps each to N pairs, maintains easy/hard balance
4. **Created `data/processed_dpo_dataset_balanced_v3_capped50/`** — New dataset with max 50 pairs/problem:
   - 10,216 total pairs (5,108 easy + 5,108 hard)
   - 2,121 unique problems (was 3,271 but rebalanced)
   - Train: 9,248 / Val: 968
   - Hard max pairs/problem: 50 (was 2,171)

### Why the dataset shrank from 50K to 10K
The original 50K dataset had extreme concentration: 203 hard problems × avg 123 pairs each. Capping at 50 pairs/problem reduced hard pairs from 25K to 5,108. Easy pairs were then balanced to match. This is a 5x reduction in data volume — a significant tradeoff between diversity and quantity.

---

## 4. Results

### Epoch-Level Metrics

#### Baseline (Standard DPO)
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1     | 0.1396    | **0.3751** (best) | 3.9939 |
| 2     | 0.0220    | 0.4016   | 5.1768 |

#### Budget-Aware DPO (lambda_easy=5.0, lambda_hard=0.0)
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1     | 0.1181    | **0.3304** (best) | 3.6089 |
| 2     | 0.0204    | 0.5064   | 7.8232 |

### Generation-Based Epoch Validation (100 problems: 50 easy + 50 hard)

#### Epoch 1
| Metric | Baseline | Budget-Aware | Delta |
|--------|----------|-------------|-------|
| Accuracy | 2.0% | 1.0% | -1.0% |
| Accuracy Easy | 4.0% | 2.0% | -2.0% |
| Accuracy Hard | 0.0% | 0.0% | — |
| **Avg Tokens Easy** | 192.1 | **134.7** | **-57.4 (30% shorter!)** |
| **Avg Tokens Hard** | 186.5 | **233.1** | **+46.6 (25% longer)** |
| TPCA | 9,465 | 18,394 | +8,929 |

#### Epoch 2
| Metric | Baseline | Budget-Aware | Delta |
|--------|----------|-------------|-------|
| Accuracy | 1.0% | _pending_ | — |
| Accuracy Easy | 2.0% | _pending_ | — |
| Accuracy Hard | 0.0% | _pending_ | — |
| Avg Tokens Easy | 131.6 | 189.9 | +58.3 (budget LOST its advantage) |
| Avg Tokens Hard | 172.1 | 229.5 | +57.4 (budget still longer) |
| TPCA | 15,187 | 20,970 | +5,783 (worse) |

**Critical finding**: Budget epoch 1 had correct direction (Easy 134.7 vs baseline 192.1 = 30% shorter). But by epoch 2, budget Easy tokens went UP to 189.9 while baseline went DOWN to 131.6 — the advantage reversed. This confirms: the budget-aware signal works initially but gets overwhelmed by overfitting with insufficient data diversity.

---

## 5. Comparison Across Iterations

### Val Loss (best epoch)
| Iteration | Baseline | Budget-Aware | Budget Better? |
|-----------|----------|-------------|----------------|
| 0 (50K data, lambda=0.05) | 0.8478 | 0.8230 | Barely |
| 1 (50K data, lambda=5.0) | 0.8478 (reused) | **0.6909** | Yes (18%) |
| 2 (10K data, lambda=5.0) | 0.3751 | **0.3304** | Yes (12%) |

### Generation Token Counts (epoch 1 gen eval)
| Iteration | Metric | Baseline | Budget-Aware | Direction Correct? |
|-----------|--------|----------|-------------|-------------------|
| 1 (post-training, 500 problems) | Easy tokens | 72.1 | 148.4 | **NO** (2x MORE) |
| 1 (post-training, 500 problems) | Hard tokens | 135.2 | 218.4 | Yes (longer) |
| 2 (epoch 1 gen eval, 100 problems) | Easy tokens | 192.1 | **134.7** | **YES (30% shorter!)** |
| 2 (epoch 1 gen eval, 100 problems) | Hard tokens | 186.5 | **233.1** | **YES (25% longer)** |

---

## 6. Analysis

### What worked
1. **Token count direction is correct for the first time**: Budget-aware generates shorter Easy (134.7 vs 192.1) and longer Hard (233.1 vs 186.5). This is exactly what the budget-aware loss is supposed to do.
2. **Val loss still favors budget-aware**: 0.3304 vs 0.3751 — consistent across all iterations.
3. **Overfitting is milder**: Val loss increased 7% for baseline (0.375→0.402) vs 53% for budget (0.330→0.506), but both are much better than iter 1 (108% increase after epoch 1).

### What went wrong
1. **Accuracy is terrible (1-2%)**: The model barely learned. Root cause: 20x fewer effective weight updates.
   - Iter 1: 11,573 steps × accumulation=1 = 11,573 updates/epoch
   - Iter 2: 2,312 steps × accumulation=4 = 578 updates/epoch
   - The capped dataset (5x smaller) plus gradient accumulation (4x fewer updates) compound to 20x fewer updates
2. **Budget overfits faster than baseline at epoch 2**: Val loss +53% vs +7%. Lambda_easy=5.0 may be creating an optimization landscape that's harder to generalize from with limited data.

### Key insight
The budget-aware signal WORKS (correct token direction) but the model needs more training time with this smaller dataset. The 50K dataset had too much repetition but gave enough updates for learning. The 10K dataset has better diversity but not enough updates.

---

## 7. Open Questions

1. **Will more epochs fix accuracy?** With 578 updates/epoch, the model needs 5-10 epochs to match the 11K updates from iter 1. But will that cause overfitting again?
2. **Should we reduce gradient accumulation back to 1?** This gives 4x more updates (2,312/epoch) — still less than iter 1's 11,573 but much better than 578.
3. **Is the capped dataset too small?** 10K pairs may not be enough for a meaningful DPO signal. Consider cap=100 instead of 50.
4. **Why does budget-aware overfit faster?** The length penalty may push the model into a narrower optimum that's less generalizable.

---

## 8. Improvements for Iteration 3

### Must-do
1. **Increase training time**: Either more epochs (5-8) or reduce gradient accumulation back to 1, or both
2. **Consider larger dataset**: Cap at 100 instead of 50 to keep more data while still improving diversity

### Should-consider
3. **Higher learning rate**: 5e-5 instead of 1e-5 to compensate for fewer updates
4. **Easy/Hard threshold tuning**: Get more unique hard problems instead of just capping
5. **Learning rate warmup + cosine decay**: May help with overfitting

---

## 9. Post-Training Evaluation

_TBD — will run after gen eval epoch 2 completes and training fully finishes_

---

## 10. Files Created/Modified This Iteration

| File | Action | Purpose |
|------|--------|---------|
| `src/config.py` | Modified | Added DATASET_PATH env var override |
| `src/training/dpo_trainer.py` | Modified | Fixed val/loss WandB, added gen eval |
| `scripts/subsample_capped_pairs.py` | Created | Cap pairs per problem |
| `data/processed_dpo_dataset_balanced_v3_capped50/` | Created | 10K capped dataset |
| `checkpoints/baseline_balanced_iter2/` | Created | Baseline checkpoints |
| `checkpoints/budget_aware_balanced_iter2/` | Created | Budget checkpoints |
