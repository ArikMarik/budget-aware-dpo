# Budget-Aware DPO — Auto-Research Iteration 4

**Date**: 2026-03-27
**Status**: Preparing to launch
**Run names**: `baseline_balanced_iter4` (GPU 0), `budget_aware_balanced_iter4` (GPU 1)

---

## 1. Hypothesis

**Combining a properly balanced 50K dataset (with per-problem capping), lower learning rate (1e-6), KL divergence penalty, and tiered answer verification will achieve both accuracy (>30%) AND correct token direction (shorter Easy, preserved Hard).**

Iteration 3 failed catastrophically — model collapse producing gibberish. Root causes identified:
- lr=1e-5 too aggressive for diverse data → policy diverges from reference → model forgets language
- No KL penalty → nothing prevents unbounded divergence
- Gen eval used string comparison → many correct answers marked wrong

---

## 2. What Changed and Why

### Change 1: New balanced dataset with capping
- **What**: Created `data/processed_dpo_dataset_balanced_v4_capped/` using `scripts/subsample_capped_balanced.py`
- **Why**: Original balanced dataset (iter 0-1) had 203 unique hard problems × 123 avg pairs = overfitting. Iter 3's real_capped100 was too different from training distribution. New dataset uses the SAME full source but with per-problem caps (Easy=50, Hard=100) to balance diversity and volume.
- **Stats**: 50,000 pairs (25K easy + 25K hard), 6,347 unique problems (6,076 easy + 276 hard), train=45,184 / val=4,816
- **Compared to original balanced**: Same size (50K) but 2x more unique problems (6,347 vs 3,271) and max 50-100 pairs/problem (vs 2,171)

### Change 2: Lower learning rate (1e-6)
- **What**: `lr=1e-6` (was 1e-5 — 10x reduction)
- **Why**: Iter 3 showed model collapse. Lower LR means slower, more stable training. The model moves closer to the reference at each step, reducing risk of catastrophic forgetting.
- **Risk**: May need more epochs to converge. Starting with 3 epochs, can extend if needed.
- **Note**: Baseline also re-runs at 1e-6 (shared hyperparameter change).

### Change 3: KL divergence penalty (weight=0.1)
- **What**: Added `--kl-penalty 0.1` to budget-aware training. Adds `kl_penalty_weight * |KL(policy || ref)|` to the DPO loss.
- **Why**: Prevents the policy from diverging too far from the reference model. The KL term acts as a leash — even if the DPO reward pushes the model in one direction, the KL penalty pulls it back toward coherent generation.
- **Implementation**: New parameter in `src/models/budget_aware_dpo_loss.py`, logged to WandB as `kl_penalty`.
- **Note**: Only applied to budget-aware (baseline has no budget-aware loss function). This is a deliberate choice — if the KL penalty helps, it confirms the divergence problem is real.

### Change 4: Fix gen eval answer verification
- **What**: Replaced `normalize_answer(pred) == normalize_answer(expected)` with `verify_correctness()` which uses tiered checking: Tier 0 (trivial equality) → Tier 1 (math-verify symbolic) → Tier 2 (LLM judge).
- **Why**: `\frac{1}{3}` vs `\dfrac{1}{3}`, `48` vs `48.0`, decimal vs fraction — all were marked wrong. Tiered verification handles mathematical equivalence.
- **Note**: In-training gen eval uses Tier 0+1 only (no LLM judge, would conflict with GPU). Post-training eval uses full Tier 0+1+2.

### Why KL penalty is only on budget-aware, not baseline
Standard DPO already has an implicit KL constraint (the `beta * log(pi/pi_ref)` term). The baseline's only issue in iter 3 was lr too high — fixed with 1e-6. The budget-aware model has an additional divergence pressure from the length penalty term, which pushes it further from the reference than standard DPO. Evidence: in iter 1, budget-aware overfitted faster than baseline (val_loss collapsed epoch 2 vs epoch 3). The KL penalty compensates for this extra pressure. If baseline still collapses at 1e-6, we'd add KL to it too — but iter 1 showed baseline can learn fine (32.4% accuracy) without it. Keeping KL off baseline also isolates the effect of budget-aware changes for cleaner comparison.

### No change to lambda values
- `lambda_easy=5.0`, `lambda_hard=0.0` — keeping from iter 1-3. The token direction was correct in iter 2 with these values.

---

## 3. Hyperparameters

### Budget-Aware
```
lambda_easy=5.0, lambda_hard=0.0, beta=0.1
kl_penalty_weight=0.1 (NEW)
max_epochs=3, batch_size=4, lr=1e-6 (was 1e-5)
gradient_accumulation_steps=1
early_stopping_patience=3
LoRA: r=128, alpha=256, dropout=0.05
Dataset: data/processed_dpo_dataset_balanced_v4_capped (50K pairs, cap=50/100)
```

### Baseline (re-run — lr changed + dataset changed)
```
Same as budget-aware but standard DPO (no lambda/length penalty, no KL penalty)
lr=1e-6 (same as budget-aware)
```

### Full commands
```bash
# Baseline (GPU 0)
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_balanced_iter4 --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --early-stopping-patience 3 --run-name baseline_balanced_iter4 --wandb \
  > logs/baseline_balanced_iter4.log 2>&1 &

# Budget-Aware (GPU 1)
CUDA_VISIBLE_DEVICES=1 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_aware_balanced_iter4 --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.1 \
  --early-stopping-patience 3 --run-name budget_aware_balanced_iter4 --wandb \
  > logs/budget_aware_balanced_iter4.log 2>&1 &
```

---

## 4. Expected Outcomes

### Best case
- Accuracy > 30% (matching iter 1) with correct token direction (Easy shorter, Hard longer)
- No model collapse thanks to lower LR + KL penalty
- Better gen eval scores due to tiered answer verification

### Worst case
- lr=1e-6 too slow — model barely learns in 3 epochs
- KL penalty too strong — prevents the budget-aware signal from taking effect

### What would tell us to change direction
- If accuracy < 5% after epoch 1: LR still too low, or KL penalty too high
- If model collapses again: fundamental issue with this dataset
- If no token count difference: lambda_easy=5.0 not effective with this data distribution

---

## 5. Results

### Early Detection: Budget-aware learning stalled

**Detected at step ~9,000 (78% epoch 1)**: Budget-aware loss stuck at ~0.40 while baseline dropped to ~0.15. The KL penalty at 0.1 is too strong — effectively freezing the budget-aware model. Every gradient update gets pulled back to the reference, preventing any learning.

**Decision**: Kill budget-aware run, keep baseline running (it's learning fine). Reduce KL penalty from 0.1 to 0.01 for iteration 5.

### Partial Epoch-Level Metrics (before kill)

### Epoch-Level Metrics

#### Baseline
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1     | _TBD_     | _TBD_    | _TBD_       |
| 2     | _TBD_     | _TBD_    | _TBD_       |
| 3     | _TBD_     | _TBD_    | _TBD_       |

#### Budget-Aware
| Epoch | Train Loss | Val Loss | Reward Diff | KL Penalty |
|-------|-----------|----------|-------------|------------|
| 1     | _TBD_     | _TBD_    | _TBD_       | _TBD_      |
| 2     | _TBD_     | _TBD_    | _TBD_       | _TBD_      |
| 3     | _TBD_     | _TBD_    | _TBD_       | _TBD_      |

### Generation-Based Epoch Validation (100 problems)
_TBD_

---

## 6. Analysis

_TBD_

---

## 7. Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `src/evaluation/run_evaluation.py` | Modified | Use verify_correctness() instead of string comparison |
| `src/evaluation/answer_extraction.py` | Modified | Added use_llm_judge parameter to verify_correctness() |
| `src/evaluation/math_grader.py` | Modified | Added use_llm_judge parameter to verify_answer() |
| `src/models/budget_aware_dpo_loss.py` | Modified | Added kl_penalty_weight parameter |
| `src/training/dpo_trainer.py` | Modified | Pass kl_penalty_weight, skip LLM judge in gen eval |
| `scripts/training/train_budget_aware_dpo.py` | Modified | Added --kl-penalty CLI argument |
| `scripts/subsample_capped_balanced.py` | Created | Create balanced dataset with per-problem caps |
| `data/processed_dpo_dataset_balanced_v4_capped/` | Created | 50K balanced dataset with caps (50 easy, 100 hard) |
