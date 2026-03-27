# Budget-Aware DPO — Auto-Research Iteration 1

**Date**: 2026-03-26
**Status**: COMPLETE — killed after epoch 2 (overfitting), post-training eval done
**Run name**: `budget_aware_balanced_4`
**WandB**: https://wandb.ai/ariksheer-tel-aviv-university/budget-aware-dpo/runs/lly7fryu
**Baseline reference**: `baseline_balanced_3` (iteration 0, epoch 2 checkpoint)
**Best checkpoint**: `checkpoints/budget_aware_balanced/checkpoint-epoch-1` (val_loss=0.6909)

---

## 1. Hypothesis

**Lambda_easy=5.0 will make the length penalty ~33% of the DPO reward signal, creating visible behavioral divergence from baseline on Easy problems.**

Iteration 0 showed that lambda_easy=0.05 produced a penalty of ~0.04 against a reward_diff of ~13 — only 0.3% of the signal. By increasing lambda_easy 100x to 5.0, the penalty should reach ~4.3, making it a significant component of the optimization target.

Additionally, lambda_hard is zeroed out (0.0) because correct Hard answers are longer (799 vs 543 tokens), and any positive penalty opposes correctness.

---

## 2. Hyperparameters

```
lambda_easy=5.0        (was 0.05 — 100x increase)
lambda_hard=0.0        (was 0.001 — zeroed out)
beta=0.1               (unchanged)
max_epochs=3           (was 5 — reduced based on iter 0 overfitting at epoch 3+)
batch_size=4           (unchanged)
lr=1e-5                (unchanged)
gradient_accumulation_steps=1  (unchanged)
early_stopping_patience=3     (unchanged)
LoRA: r=128, alpha=256, dropout=0.05  (unchanged)
```

### Full command
```bash
CUDA_VISIBLE_DEVICES=1 DATASET_VARIANT=balanced PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_aware_balanced \
  --max-epochs 3 --batch-size 4 --lr 1e-5 \
  --lambda-easy 5.0 --lambda-hard 0.0 \
  --early-stopping-patience 3 \
  --run-name budget_aware_balanced_4 --wandb \
  > logs/budget_aware_balanced.log 2>&1 &
```

---

## 3. Changes Made

**No code changes.** Lambda values are CLI args passed to the existing loss function. The only operational change was adding `PYTHONUNBUFFERED=1` to the launch command to fix log buffering issues.

---

## 4. Results

### Epoch-Level Metrics

#### Iteration 1: Budget-Aware DPO (lambda_easy=5.0, lambda_hard=0.0)
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1     | 0.0576    | 0.6909   | 9.5302      |
| 2     | 0.0187    | 1.4587   | 11.5545     |
| 3     | (killed — overfitting, epoch 1 was best) | — | — |

#### Baseline Reference (from iteration 0)
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1     | 0.0549    | 1.2979   | 9.91        |
| 2     | 0.0216    | **0.8478** (best) | 13.21 |
| 3     | 0.0203    | 1.3874   | 17.48       |

### Key Observations from Metrics
- val_complexity_0_loss (Easy): 0.0159 at epoch 1 → 0.2434 at epoch 2 (15x increase — severe Easy overfitting)
- val_complexity_1_loss (Hard): 0.6774 at epoch 1 → 1.2822 at epoch 2 (2x increase — Hard also overfitting)
- Best model at epoch 1 (not epoch 2 like baseline) — model overfits faster with lambda_easy=5.0
- Training killed after epoch 2 due to clear overfitting trajectory

---

## 5. Comparison to Baseline

### Epoch 1 Side-by-Side
| Metric | Baseline (iter 0) | Budget-Aware (iter 1) | Delta |
|--------|-------------------|----------------------|-------|
| Train Loss | 0.0549 | 0.0576 | +0.0027 (slightly higher) |
| Val Loss | 1.2979 | **0.6909** | **-0.6070** (much better) |
| Reward Diff | 9.91 | 9.53 | -0.38 (slightly lower) |

**Notable**: Val loss is dramatically better (0.69 vs 1.30). This could indicate that the length penalty acts as a regularizer, preventing the model from becoming overconfident on easy preferences. The lower reward_diff is consistent — the penalty reduces the effective reward margin on easy problems, as intended.

### Full Epoch Comparison
| Epoch | Baseline Val Loss | Budget Val Loss | Baseline Reward Diff | Budget Reward Diff |
|-------|------------------|----------------|---------------------|-------------------|
| 1     | 1.2979           | **0.6909**     | 9.91                | 9.53              |
| 2     | **0.8478**       | 1.4587         | 13.21               | 11.55             |
| 3     | 1.3874           | (killed)       | 17.48               | —                 |

Budget-aware peaks at epoch 1 with much better val_loss (0.69 vs baseline best 0.85), but collapses by epoch 2. Baseline is more stable, peaking at epoch 2.

---

## 6. Analysis

### What we know so far (after epoch 1)

1. **Val loss improvement is significant**: 0.69 vs 1.30 is a 47% reduction. This is the first strong signal that lambda_easy=5.0 has a real effect on model behavior.

2. **The penalty may act as regularization**: By reducing the reward margin on easy problems, the model can't simply memorize "always pick chosen" — it must actually learn the length preference pattern. This could explain the better generalization.

3. **Train loss is similar**: 0.0576 vs 0.0549 — the model is learning at roughly the same rate. The length penalty doesn't prevent learning, it redirects it.

### Critical limitation: validation is DPO preference, not generation

The current validation (`evaluate()` in `dpo_trainer.py`) computes DPO preference metrics on held-out pairs — "does the model assign higher reward to chosen vs rejected?" This tells us about preference learning but **not** about actual generation behavior.

To verify that the model actually generates shorter Easy answers and preserves Hard reasoning, we need the post-training evaluation (`run_evaluation.py`) which calls `model.generate()` and measures real token counts and accuracy.

**Action item for iteration 2**: Add lightweight generation-based validation (50-100 problems) at each epoch boundary, logging `gen/accuracy_easy`, `gen/accuracy_hard`, `gen/avg_tokens_easy`, `gen/avg_tokens_hard` to WandB. This is the only way to see if the model is actually changing behavior during training.

---

## 7. Critical Dataset Issue Discovered

### Hard subset concentration

Analysis during this iteration revealed a severe data concentration problem:

| Subset | Pairs | Unique Problems | Avg Pairs/Problem |
|--------|-------|----------------|-------------------|
| Easy   | 25,000 | 3,069 | 8.1 |
| **Hard** | **25,000** | **203** | **123.2** |

- Top 10 hard problems account for **50% of all hard pairs**
- One single problem has 2,171 pairs (8.7% of hard data)
- Hard is 99.8% from `math` source
- Easy is more diverse across `augmented_gsm8k`, `augmented_math`, `math`, `gsm8k`

### Impact

The model sees ~200 hard problems repeated ~123 times each with different solution pairs. This likely causes:
- Overfitting to specific problem structures rather than learning general hard-problem preferences
- Poor generalization to unseen hard problems
- The "accuracy" metric being inflated (memorization, not learning)

### Source distribution in training data

| Source | Unique Problems | DPO Pairs | % of Pairs |
|--------|----------------|-----------|------------|
| math | 372 | 34,817 | 69.6% |
| augmented_gsm8k | 1,733 | 9,343 | 18.7% |
| augmented_math | 844 | 3,116 | 6.2% |
| gsm8k | 322 | 2,724 | 5.4% |

The `math` source dominates pairs (69.6%) despite having few unique problems — each math problem generates many solution variants.

### Proposed fixes for iteration 2+

1. **Cap pairs per problem** (e.g., max 50) to force diversity
2. **Lower the Easy/Hard threshold** to get more unique hard problems into the hard bucket
3. **Consider MATH-only** for cleaner signal (but insufficient easy diversity — 372 problems, only 9,864 easy pairs)
4. **User concern**: AugmentedMath creates too-similar questions; may want to filter to original MATH only

---

## 8. Open Questions

1. **Does the val_loss improvement translate to better generation?** The 47% val_loss reduction is promising but DPO preference metrics may not predict generation behavior. Post-training eval is required.

2. **Is the model actually generating shorter Easy answers?** We have no generation-based metrics during training. WandB token counts (chosen/rejected lengths) are dataset properties, not model outputs.

3. **Will the model overfit in epochs 2-3 like baseline did?** Baseline val_loss spiked from 0.85 to 1.39 between epochs 2 and 3. If the regularization effect of lambda_easy persists, we might see a flatter val_loss curve.

4. **Is the Hard data too concentrated for meaningful learning?** 203 unique problems × 123 repetitions may not teach generalizable hard-problem preferences regardless of lambda values.

5. **Should we fix the data before more lambda experiments?** If the model can't generalize due to data issues, no amount of lambda tuning will produce meaningful results on unseen problems.

---

## 9. Improvements Needed Before Iteration 2

### Must-do
1. **Fix val/loss WandB graph**: val_loss is not rendering as a graph in WandB. Need to investigate logging in `dpo_trainer.py`.
2. **Add generation-based epoch validation**: Log `gen/accuracy_easy`, `gen/accuracy_hard`, `gen/avg_tokens_easy`, `gen/avg_tokens_hard` to WandB at each epoch end. Use a small sample (50-100 problems) to keep it fast.

### Should-do
3. **Fix Hard data concentration**: Cap pairs per problem or lower the hard threshold to get more unique problems.
4. **Consider Easy/Hard threshold tuning**: Current threshold may be too aggressive, pushing too many problems into Easy and leaving too few unique Hard problems.

### Nice-to-have
5. **Explore source filtering**: Try MATH-only (no AugmentedMath) to reduce question similarity.
6. **Learning rate schedule**: Add warmup + cosine decay to reduce early overfitting.

---

## 10. Decision for Iteration 2

**Chose Option C: Fix data + add generation validation + re-run with lambda_easy=5.0**

### Rationale
Post-training eval showed accuracy improved (+5.2%) but tokens went UP (2x on Easy). This means:
- The lambda signal IS affecting the model (accuracy improvement = regularization effect)
- But DPO preference learning doesn't directly translate to shorter generation
- The dataset concentration (203 unique hard problems × 123 repetitions) likely causes overfitting-driven verbosity
- We need generation-based metrics during training to see what's happening epoch-by-epoch

### What was implemented for iteration 2
1. **Fixed val/loss WandB graph** — added `wandb.define_metric("val/*", step_metric="train/epoch")` in `dpo_trainer.py`
2. **Added generation-based epoch validation** — 100 problems (50 easy + 50 hard) evaluated with `model.generate()` after each epoch, logged as `gen/accuracy`, `gen/avg_tokens_easy`, `gen/avg_tokens_hard`, `gen/tpca` to WandB
3. **Created capped dataset** — `scripts/subsample_capped_pairs.py` caps pairs per problem. Created `data/processed_dpo_dataset_balanced_v3_capped50/` (10,216 pairs, max 50/problem, 2,121 unique problems)
4. **Added DATASET_PATH env var override** in `src/config.py` to support custom dataset paths
5. **Reduced max_epochs to 2** (overfitting after epoch 1 in iter 1)
6. **Added gradient_accumulation_steps=4** (effective batch=16 for smoother gradients)
7. **Re-running baseline** because dataset changed (shared parameter)

### Concern identified during iteration 2
The combination of 5x smaller dataset + 4x gradient accumulation = 20x fewer weight updates per epoch. This caused very low accuracy (1-2%) in gen eval after epoch 1. The token count direction was correct (Easy shorter, Hard longer) but the model barely learned. This needs to be addressed in iteration 3.

---

## 11. Post-Training Evaluation Results

Evaluated on 500 unique problems from the training dataset (generation-based: model.generate(), not preference selection).

- Baseline: `checkpoints/baseline_balanced/checkpoint-epoch-2`
- Budget-aware: `checkpoints/budget_aware_balanced/checkpoint-epoch-1` (best epoch)

| Metric | Baseline | Budget-Aware (iter 1) | Delta |
|--------|----------|----------------------|-------|
| **Accuracy** | 32.4% | **37.6%** | **+5.2%** |
| **TPCA** | **244.8** | 415.9 | +171.1 (70% worse) |
| **Avg Tokens Easy** | **72.1** | 148.4 | +76.3 (**2x MORE**, opposite of goal) |
| **Avg Tokens Hard** | **135.2** | 218.4 | +83.2 (1.6x more) |
| Num Correct | 162/500 | 188/500 | +26 |
| Num Easy | 443 | 443 | — |
| Num Hard | 57 | 57 | — |

### Critical Findings

1. **Accuracy improved** (+5.2%) — the budget-aware model is more accurate despite only training for 1 epoch vs baseline's 2. This suggests the length penalty may act as beneficial regularization.

2. **Token counts went UP, not down** — the budget-aware model generates 2x MORE tokens on Easy problems (148 vs 72). This is the **opposite** of the intended behavior. The DPO length penalty teaches the model to *prefer* shorter responses in a choice scenario, but this doesn't translate to shorter *generated* responses.

3. **TPCA is worse** — more tokens per correct answer despite higher accuracy. The model is more verbose across the board.

4. **Repetitive generation loops** — the budget-aware model frequently gets stuck generating the same LaTeX line repeatedly until hitting max_new_tokens (256). This inflates token counts and suggests model instability.

### Why DPO preference ≠ generation behavior

The core issue: DPO trains the model to assign higher log-probability to shorter responses when presented with a (short, long) pair. But during generation, the model doesn't see both options — it generates autoregressively. The preference learned through DPO may not transfer to generation behavior, especially with:
- Only 1 epoch of training (limited preference learning)
- A 500M parameter model (limited capacity to learn nuanced behaviors)
- The length penalty operating on preference pairs, not on generation
