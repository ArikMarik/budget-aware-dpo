# Iteration 3 — Large Diverse Dataset (Model Collapse)

**Date**: 2026-03-26
**Status**: COMPLETE — model collapsed (0% accuracy, gibberish output on all epochs)
**Baseline reference**: Will be re-run (dataset changed)

---

## 1. Hypothesis

**Using the full `processed_dpo_dataset_real` source (3.9M pairs, 20K unique problems) with cap=100 pairs/problem will provide enough data volume AND diversity to achieve both high accuracy AND correct token count direction (shorter Easy, preserved Hard).**

Iterations 1 and 2 showed:
- **Iter 1** (50K data, no cap): Good accuracy (32-37%) but tokens went UP on Easy — dataset concentration caused overfitting
- **Iter 2** (10K data, cap=50): Correct token direction (Easy 30% shorter!) but terrible accuracy (1-2%) — too few weight updates

The root cause of iter 2's failure was identified: the 50K balanced subset was created from only 3,271 unique problems. Capping it to 50/problem left only 10K pairs. But the FULL `processed_dpo_dataset_real` has **3,949,240 pairs from 20,016 unique problems**. Extracting from this source with cap=100 gives us **51,672 pairs from 11,647 unique problems** — same volume as original 50K but 3.5x more diversity.

---

## 2. What Changed and Why

### Change 1: New dataset from full source
- **What**: Created `data/processed_dpo_dataset_real_capped100/` by capping pairs from `processed_dpo_dataset_real` (3.9M pairs) instead of `processed_dpo_dataset_balanced` (50K pairs)
- **Why**: The previous 50K balanced subset only had 3,271 unique problems (203 hard). The full source has 20,016 unique problems (276 hard). Capping at 100 pairs/problem from the full source gives 51,672 pairs with 11,647 unique problems — 3.5x more problem diversity at the same data volume.
- **Dataset stats**:
  - 51,672 pairs (25,836 easy + 25,836 hard)
  - 11,647 unique problems (11,381 easy + 276 hard)
  - Max 100 pairs/problem (was unlimited in iter 1, cap=50 in iter 2)
  - Train: 46,431 / Val: 5,241
- **Risk**: Different source distribution than previous iterations. Easy problems now come from 11K unique problems (vs 3K), which means more diverse but potentially harder to learn patterns across.

### Change 2: Gradient accumulation back to 1
- **What**: `gradient_accumulation_steps=1` (was 4 in iter 2)
- **Why**: Iter 2 had only 578 effective weight updates per epoch (2,312 steps / 4 accumulation). This was 20x fewer than iter 1's 11,573 updates, causing terrible accuracy. With 46K training pairs and batch_size=4, we get 11,608 steps/epoch — similar to iter 1's 11,573.
- **Risk**: Less smooth gradients, but iter 1 showed this works fine.

### Change 3: max_epochs=3
- **What**: `max_epochs=3` (was 2 in iter 2, 3 in iter 1)
- **Why**: With enough data and diversity, 3 epochs should give the model time to learn without catastrophic overfitting. Iter 1 best was epoch 1 but with concentrated data. With better diversity, the model may benefit from more training.
- **Risk**: Overfitting at epoch 3 like iter 0/1, but early stopping (patience=3) protects us, and we keep the best checkpoint.

### No changes to lambda values
- `lambda_easy=5.0`, `lambda_hard=0.0` — keeping these from iter 1/2 since the token direction signal was correct in iter 2.

---

## 3. Hyperparameters

### Budget-Aware
```
lambda_easy=5.0, lambda_hard=0.0, beta=0.1
max_epochs=3, batch_size=4, lr=1e-5
gradient_accumulation_steps=1
early_stopping_patience=3
LoRA: r=128, alpha=256, dropout=0.05
Dataset: data/processed_dpo_dataset_real_capped100 (51,672 pairs, cap=100/problem)
```

### Baseline (re-run — dataset changed)
```
Same as budget-aware but standard DPO (no lambda/length penalty)
```

---

## 4. Expected Outcomes

### Best case
- Accuracy comparable to iter 1 (30%+) WITH correct token direction from iter 2 (Easy shorter, Hard longer)
- Val loss shows budget-aware < baseline (consistent with iter 1 and 2)
- Less overfitting than iter 1 due to better data diversity

### Worst case
- Accuracy still low due to completely different data distribution
- Overfitting returns despite capping (different failure mode)

### What would tell us to change direction
- If accuracy < 10% after epoch 1: learning rate may need increasing
- If overfitting worse than iter 1: the problem is not just data concentration
- If token direction reverses: the budget-aware signal doesn't generalize across data distributions

---

## 5. Results

### Epoch-Level Metrics

#### Baseline
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1     | 0.1648    | **1.3360** (best) | 5.6477 |
| 2     | 0.0955    | 2.3139   | 6.0158 |
| 3     | 0.0923    | 2.5028   | 8.5416 |

#### Budget-Aware
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1     | 0.1619    | **1.2013** (best) | 5.9524 |
| 2     | 0.0737    | 1.7256   | 7.4379 |
| 3     | 0.0563    | 1.6512   | 8.1898 |

### Generation-Based Epoch Validation (100 problems)

ALL epochs, BOTH models: **0% accuracy, 256 tokens (max), TPCA=inf**

The model generates incoherent gibberish: "sides sides sides sides ℉℉℉℉℉℉". This is model collapse — DPO training destroyed the model's ability to generate coherent text.

---

## 6. Analysis

### Root cause: Model collapse from DPO training

The `processed_dpo_dataset_real` has a fundamentally different distribution than the balanced subset. The problems are harder and more diverse (11,647 unique problems vs 3,271). Combined with lr=1e-5, the DPO loss pushed the policy model too far from the reference model, causing catastrophic forgetting of the base model's language ability.

Evidence:
- Val loss starts at 1.34/1.20 (much higher than iter 1's 0.69) — model struggles from the start
- Gen eval shows gibberish, not wrong math — the model forgot how to write English
- This did NOT happen with the balanced subset (iter 1/2) which had fewer unique problems and more repetition

### Lessons
1. **Learning rate 1e-5 is too high** for diverse datasets — need 1e-6 or lower
2. **KL penalty is needed** to prevent policy from diverging too far from reference
3. **The gen eval answer comparison was too strict** (string match instead of math equivalence) — need tiered verification
4. **The original balanced dataset worked better** because repeated problems helped the model learn without collapsing

### Decision for iteration 4
See iteration4.md — lower LR to 1e-6, add KL penalty, use new balanced dataset with capping, fix gen eval verification.

---

## 7. Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `data/processed_dpo_dataset_real_capped100/` | Created | 51K capped dataset from full source |
| `checkpoints/baseline_balanced_iter3/` | Will create | Baseline checkpoints |
| `checkpoints/budget_aware_balanced_iter3/` | Will create | Budget checkpoints |
