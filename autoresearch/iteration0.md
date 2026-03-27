# Budget-Aware DPO — Auto-Research Iteration 0

**Date**: 2026-03-26
**Status**: Baseline established, no meaningful divergence observed. Ready for hyperparameter exploration.

---

## 1. Project Objective

Train **Qwen2.5-0.5B** (500M params) with a **Budget-Aware DPO** loss that teaches the model to:
- **Easy math problems (C=0)**: Produce short, direct answers (save compute tokens)
- **Hard math problems (C=1)**: Produce full chain-of-thought reasoning (preserve accuracy)

The key innovation is a modified DPO reward with a length penalty term:

```
R_budget(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x)) - lambda(C) * length_penalty(y)
```

Where `lambda(C)` is high for Easy (penalize verbosity) and near-zero for Hard (prioritize correctness).

### Success Criteria

The budget-aware model must show **measurable differences** from the standard DPO baseline:

| Metric | What "success" looks like |
|--------|--------------------------|
| **TPCA (Tokens Per Correct Answer)** | Lower for budget-aware (fewer tokens for same accuracy) |
| **Accuracy (overall)** | Budget-aware >= baseline (no accuracy sacrifice) |
| **Avg tokens on Easy** | Budget-aware << baseline (shorter Easy responses) |
| **Avg tokens on Hard** | Budget-aware ~= baseline (preserved reasoning depth) |
| **MATH Level 4-5 accuracy** | Budget-aware ~= baseline (no degradation on hardest problems) |
| **reward_diff_easy vs reward_diff_hard** | Budget-aware shows more divergence between Easy/Hard than baseline |
| **accuracy_easy vs accuracy_hard** | Budget-aware shows stronger Easy preference learning |

---

## 2. Model & Training Architecture

### Model
- **Base**: Qwen/Qwen2.5-0.5B (500M parameters)
- **Adaptation**: LoRA with r=128, alpha=256 (scaling factor = alpha/r = 2.0)
- **Target modules**: q_proj, v_proj, k_proj, o_proj (all attention projections)
- **LoRA dropout**: 0.05
- **Effective LR**: lr * alpha/r = 1e-5 * 2.0 = 2e-5

### Optimizer
- **AdamW**: lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01
- **Gradient clipping**: max_norm=1.0 (clip_grad_norm_ reports pre-clip norm)
- **Mixed precision**: float16 (autocast)

### Loss Functions

**Standard DPO (baseline)**:
```
reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)
loss = -logsigmoid(reward_diff).mean()
```

**Budget-Aware DPO**:
```
lambdas = lambda_easy if C==0 else lambda_hard    # per sample
length_diff = (chosen_len - rejected_len) / avg_len    # normalized
length_penalty = lambdas * length_diff
reward_diff = beta * (log_ratio_chosen - log_ratio_rejected) - length_penalty
loss = -logsigmoid(reward_diff).mean()
```

Note: `log_prob` computes **per-token averaged** log-probabilities: `sum(log_probs) / num_tokens`. This means log-ratios are O(1), not O(T).

---

## 3. Dataset: `processed_dpo_dataset_balanced`

**Total**: 50,000 preference pairs
- **Train**: 46,290 pairs | **Val**: 3,710 pairs

### Composition

| Complexity | Count | Rejection Reason | What pairs teach |
|------------|-------|-----------------|------------------|
| Easy (C=0) | 25,000 | 100% length | Prefer short correct over verbose correct |
| Hard (C=1) | 25,000 | 100% incorrectness | Prefer correct (long) over incorrect (short) |

### Token Length Statistics

#### Easy Problems
| | Chosen (short, correct) | Rejected (verbose, correct) |
|---|---|---|
| Mean | 60 tokens | 151 tokens |
| Std | 8 | 53 |
| P50 | 62 | 130 |
| P95 | 70 | 224 |
| Range | 38–70 | 99–1,121 |

**Ratio**: rejected/chosen = **2.5x** mean. Tight chosen distribution, wider rejected tail.

#### Hard Problems
| | Chosen (correct) | Rejected (incorrect) |
|---|---|---|
| Mean | 799 tokens | 543 tokens |
| Std | 138 | 210 |
| P50 | 804 | 510 |
| P95 | 1,016 | 902 |
| Range | 378–1,363 | 131–1,176 |

**Ratio**: rejected/chosen = **0.7x** mean. **Correct answers are LONGER** (more reasoning needed).

### Critical Data Insight

For Hard problems, 25% of pairs (6,246) have the **same boxed answer** — the rejection is based on reasoning quality, not just final answer correctness. The remaining 75% (18,750) have clearly different final answers.

### Length Penalty Magnitude Problem

For Easy: normalized_length_diff = (60 - 151) / 105.5 ≈ **-0.86**
For Hard: normalized_length_diff = (799 - 543) / 671 ≈ **+0.38**

With current lambdas:
- Easy penalty: 0.05 * (-0.86) = **-0.043** (helps chosen, but tiny)
- Hard penalty: 0.001 * (+0.38) = **+0.0004** (negligible, good)

But reward_diff reaches **13–18** by epoch 2-4. The penalty of 0.043 is **0.3% of the signal** — completely invisible to the optimizer.

---

## 4. Current Run Results (Iteration 0)

### Hyperparameters Used
```
batch_size=4, lr=1e-5, max_epochs=5, dpo_beta=0.1
lambda_easy=0.05, lambda_hard=0.001
gradient_accumulation_steps=1, early_stopping_patience=3
LoRA: r=128, alpha=256, dropout=0.05
```

### Epoch-Level Metrics

#### Baseline (Standard DPO)
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1 | 0.0549 | 1.2979 | 9.91 |
| 2 | 0.0216 | **0.8478** (best) | 13.21 |
| 3 | 0.0203 | 1.3874 | 17.48 |
| 4 | 0.0294 | 1.5103 | 18.29 |
| 5 | (running) | — | — |

#### Budget-Aware DPO
| Epoch | Train Loss | Val Loss | Reward Diff |
|-------|-----------|----------|-------------|
| 1 | 0.0533 | 0.8959 | 10.84 |
| 2 | 0.0190 | **0.8230** (best) | 12.99 |
| 3 | 0.0229 | 1.2640 | 13.60 |
| 4 | 0.0153 | 1.3610 | 16.87 |
| 5 | (running) | — | — |

### Key Observations

1. **No meaningful divergence**: Both runs follow nearly identical trajectories. Budget-aware has slightly better val_loss (0.8230 vs 0.8478) but this is within noise.

2. **Severe overfitting**: Train loss drops to 0.015–0.03 while val loss rises after epoch 2. Both models memorize the training set. Notice that if the problem is not enough data - there are preprocessing pipelines that can generate new splits with configurable data amount.

3. **Reward_diff explodes**: Growing from ~10 to ~18 across epochs. The model becomes overconfident in its preferences.

4. **Length penalty is invisible**: At lambda_easy=0.05 with normalized lengths, the penalty term (~0.04) is dwarfed by the DPO reward signal (~13+). The optimizer can't "see" the length penalty.

5. **Best epoch for both = 2**: Validation loss is best at epoch 2, deteriorates after. Early stopping (patience=3) will trigger at epoch 5.

---

## 5. Diagnosis: Why No Divergence?

### Root Cause: Lambda values are too small relative to DPO reward magnitude

The budget-aware loss adds a term of magnitude ~0.04 to a reward_diff of ~13. This is a **0.3% perturbation** — far below the noise floor of stochastic gradient descent.

### Contributing Factors

1. **Normalization dampens the signal**: Dividing `(chosen_len - rejected_len)` by `avg_len` makes the penalty O(1), which is correct for numerical stability, but then lambda=0.05 makes it tiny.

2. **Beta=0.1 amplifies the DPO term**: `beta * (log_ratio_c - log_ratio_r)` grows large because the model quickly learns to differentiate chosen/rejected.

3. **Overfitting**: Both models achieve near-zero train loss by epoch 2-3, meaning the DPO signal dominates completely. There's no "room" for the length penalty to steer behavior.

4. **No learning rate schedule**: Constant lr=1e-5 throughout training. The model may benefit from warmup + decay.

5. **Hard problems: penalty opposes correctness**: For Hard pairs, correct answers are 255 tokens longer on average. Even lambda_hard=0.001 creates a tiny force pushing toward shorter (wrong) answers.

---

## 6. Suggestions for Next Iterations

### Priority 1: Increase lambda_easy dramatically

The length penalty must be **comparable to the DPO reward term** to create visible divergence.

| Approach | lambda_easy | Expected penalty magnitude | % of reward_diff |
|----------|------------|---------------------------|-------------------|
| Current | 0.05 | 0.04 | 0.3% |
| Moderate | 1.0 | 0.86 | 6.6% |
| **Strong** | **5.0** | **4.3** | **33%** |
| Aggressive | 10.0 | 8.6 | 66% |

**Recommendation**: Start with `lambda_easy=5.0`. This makes the penalty ~33% of the reward signal — enough to steer behavior without dominating.

### Priority 2: Set lambda_hard=0.0

The dataset analysis proves that correct Hard answers are longer (799 vs 543 tokens). Any positive lambda_hard creates a gradient that opposes correctness. Zero it out completely.

### Priority 3: Address overfitting

Both models overfit by epoch 2-3. Options:
- **Reduce epochs to 2-3** (use best checkpoint from epoch 2)
- **Increase gradient accumulation** to 4-8 (effective batch 16-32, smoother gradients)
- **Add learning rate warmup + cosine decay** (avoid sharp early updates)
- **Increase LoRA dropout** from 0.05 to 0.1-0.15

### Priority 4: Alternative normalization

Instead of normalizing by avg_len (which compresses the signal), consider:
- **Raw token difference**: `lambda * (chosen_len - rejected_len)` — but needs much smaller lambda
- **Log-length penalty**: `lambda * (log(chosen_len) - log(rejected_len))` — naturally bounded
- **Percentile-based**: Normalize by dataset-wide P50 length instead of per-sample avg

### Priority 5: Separate beta for Easy vs Hard

Instead of one beta=0.1 for all, use:
- `beta_easy=0.05` (weaker DPO signal, let length penalty dominate)
- `beta_hard=0.1` (full DPO signal for accuracy)

This would require modifying the loss function but gives finer control.

---

## 7. Open Research Question: Should Hard Pairs Prioritize Longer Answers?

### The situation

In our Hard dataset:
- **Chosen (correct)**: 799 tokens average
- **Rejected (incorrect)**: 543 tokens average
- 25% of pairs have the **same final answer** but different reasoning quality

### The question

When both answers are correct (same boxed answer), should the model prefer the longer one? Currently, by labeling the longer response as "chosen" in those pairs, we're implicitly teaching: **longer = better for hard problems**.

### Arguments FOR prioritizing longer (current approach)

1. **More thorough reasoning is safer**: A longer correct solution likely covers more edge cases, shows work, and is more verifiable. In math, showing all steps is a feature, not a bug.

2. **Consistency with the budget-aware objective**: The whole point is "short for easy, thorough for hard." Teaching the model that hard problems warrant detailed responses aligns with this.

3. **Grading robustness**: In educational and competition settings, showing work matters. A short correct answer with no justification may be penalized.

### Arguments AGAINST prioritizing longer

1. **Efficiency**: If both answers are correct, the shorter one is strictly more efficient. Budget-aware should optimize tokens — even for hard problems, unnecessary verbosity wastes compute.

2. **Signal pollution**: Teaching the model "longer = better" for hard problems may cause it to pad responses with filler to seem thorough, rather than actually improving reasoning.

3. **Contradiction with TPCA metric**: TPCA (Tokens Per Correct Answer) is a primary success metric. Encouraging longer Hard responses directly increases TPCA.

4. **Real-world deployment**: In production, shorter correct answers save inference cost. The goal is correct + concise, not correct + verbose.

### Proposed resolution

This is an empirical question. Consider creating two dataset variants:

- **Variant A (current)**: Hard same-answer pairs → longer is chosen (reward thoroughness)
- **Variant B (alternative)**: Hard same-answer pairs → shorter is chosen (reward efficiency)
- **Variant C (remove)**: Drop the 6,246 same-answer Hard pairs entirely (cleaner signal)

Run budget-aware training on each and compare TPCA + accuracy. The answer likely depends on whether the "longer correct" solutions actually contain valuable reasoning steps or just filler text.

### For now

Given that same-answer pairs are only 25% of Hard data (6,246 out of 25,000), and the primary Hard signal (75%) is correctness-based (different answers), this is a secondary concern. **Focus first on making the length penalty visible (Priority 1-2) before optimizing the dataset composition.**

---

## 8. Experiment Plan for Iteration 1

### Run 1: Strong lambda, zero hard penalty
```
lambda_easy=5.0, lambda_hard=0.0, beta=0.1
max_epochs=3, batch_size=4, lr=1e-5
gradient_accumulation_steps=1
```

### Run 2: Very strong lambda with gradient accumulation
```
lambda_easy=10.0, lambda_hard=0.0, beta=0.1
max_epochs=3, batch_size=4, lr=1e-5
gradient_accumulation_steps=4 (effective batch=16)
```

### Run 3: Moderate lambda with reduced beta for Easy
(Requires code change to support per-complexity beta)
```
lambda_easy=2.0, lambda_hard=0.0, beta=0.1
max_epochs=3, lr=5e-6 (lower to reduce overfitting)
```

### Baseline (unchanged, for comparison)
```
lambda_easy=0, lambda_hard=0 (standard DPO), beta=0.1
max_epochs=3, batch_size=4, lr=1e-5
```

### Evaluation criteria for each run
After training, compare on wandb:
1. `val/accuracy_easy` and `val/accuracy_hard` vs baseline
2. `train/reward_diff_easy` vs `train/reward_diff_hard` divergence
3. `val/loss` (overfitting trajectory)
4. Then run `evaluate_checkpoint()` on best epoch to get TPCA + accuracy on test set

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `src/models/budget_aware_dpo_loss.py` | Budget-aware loss function |
| `src/models/standard_dpo_loss.py` | Standard DPO loss (baseline) |
| `src/training/dpo_trainer.py` | Training loop, metrics, evaluation |
| `scripts/training/train_budget_aware_dpo.py` | Budget-aware training CLI |
| `scripts/training/train_baseline_dpo.py` | Baseline training CLI |
| `src/config.py` | Paths, model name, hyperparameter defaults |
| `src/evaluation/run_evaluation.py` | Post-training evaluation (accuracy, TPCA) |
| `src/evaluation/answer_extraction.py` | Answer parsing from model outputs |
| `src/evaluation/math_grader.py` | Tiered answer verification |
| `data/processed_dpo_dataset_balanced/` | Training data (50k balanced pairs) |
| `docs/DATASET_ANALYSIS.md` | Dataset statistics and analysis |
| `docs/TRAINING_GUIDE.md` | Training guide with metric explanations |
| `docs/dpo_loss_explainer.md` | DPO loss theory and bug history |
| `implementation_plan.md` | Full project implementation plan (Phases 0-11) |

---

## 10. WandB Metrics Available

### Per-step (train/)
| Metric | Description |
|--------|-------------|
| `train/loss` | DPO training loss |
| `train/reward_diff` | Overall implicit reward margin |
| `train/reward_diff_easy` | Reward margin for Easy (C=0) |
| `train/reward_diff_hard` | Reward margin for Hard (C=1) |
| `train/accuracy` | % samples where model prefers chosen |
| `train/accuracy_easy` | Accuracy on Easy problems |
| `train/accuracy_hard` | Accuracy on Hard problems |
| `train/gradient_norm` | Pre-clipping gradient norm |
| `train/learning_rate` | Current LR |
| `train/avg_chosen_tokens` | Mean chosen response length |
| `train/avg_rejected_tokens` | Mean rejected response length |
| `train/avg_chosen_tokens_easy` | Mean chosen length (Easy) |
| `train/avg_chosen_tokens_hard` | Mean chosen length (Hard) |
| `train/avg_rejected_tokens_easy` | Mean rejected length (Easy) |
| `train/avg_rejected_tokens_hard` | Mean rejected length (Hard) |
| `train/token_diff` | avg_chosen - avg_rejected |
| `train/length_penalty` | Budget-aware only: mean penalty term |
| `train/complexity_0_loss` | DPO loss for Easy samples |
| `train/complexity_1_loss` | DPO loss for Hard samples |

### Per-epoch (val/)
All the same metrics with `val/` prefix, computed on the validation set (3,710 pairs).

---

## 11. Running Training & GPU Management

### Training Commands

Training runs use two GPUs (GPU 0 and GPU 1). Launch both in parallel with `nohup`:

**Budget-Aware DPO** (the experiment — always run this):
```bash
CUDA_VISIBLE_DEVICES=1 DATASET_VARIANT=balanced nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_aware_balanced \
  --max-epochs 3 \
  --batch-size 4 \
  --lr 1e-5 \
  --lambda-easy 5.0 \
  --lambda-hard 0.0 \
  --early-stopping-patience 3 \
  --run-name budget_aware_balanced_N \
  --wandb \
  > logs/budget_aware_balanced.log 2>&1 &
```

**Baseline DPO** (only re-run if changing shared hyperparameters like lr, batch_size, epochs, beta):
```bash
CUDA_VISIBLE_DEVICES=0 DATASET_VARIANT=balanced nohup .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_balanced \
  --max-epochs 3 \
  --batch-size 4 \
  --lr 1e-5 \
  --early-stopping-patience 3 \
  --run-name baseline_balanced_N \
  --wandb \
  > logs/baseline_balanced.log 2>&1 &
```

Replace `N` with the iteration number in the run name (e.g., `budget_aware_balanced_4`, `baseline_balanced_4`).

### When to re-run baseline

| Changed parameter | Re-run baseline? |
|-------------------|-----------------|
| `lambda_easy`, `lambda_hard` | **No** — baseline has no lambdas |
| `lr`, `batch_size`, `max_epochs` | **Yes** — these affect both models |
| `dpo_beta` | **Yes** — shared DPO parameter |
| `gradient_accumulation_steps` | **Yes** — changes effective batch size |
| Dataset (new split, more data) | **Yes** — must compare on same data |
| LoRA config (rank, alpha, dropout) | **Yes** — changes model capacity |

The existing baseline from iteration 0 (best val_loss=0.8478 at epoch 2) can be reused as long as shared hyperparameters haven't changed.

### GPU Keep-Alive

When no training is running, start the keep-alive script to prevent the GPU allocation from being reclaimed:

```bash
# Start keep-alive (runs a 10-min GPU workload every hour)
nohup python keep_alive.py > logs/keep_alive.log 2>&1 &

# Check if running
ps aux | grep keep_alive | grep -v grep

# Kill before starting training
pkill -f keep_alive.py
```

**Rule**: Always kill `keep_alive.py` before launching training. Always start it after training completes or if GPUs will be idle.

### Kill all training processes
```bash
pkill -f 'train_baseline_dpo|train_budget_aware_dpo'
```

---

## 12. Iteration Protocol

Each research iteration must produce a document `autoresearch/iterationN.md` containing:

### Required sections

1. **Hypothesis** — What are we testing and why? What specific change from the previous iteration?
2. **Hyperparameters** — Exact parameters used (full command-line invocation)
3. **Changes made** — Any code changes (file, line, what changed)
4. **Results** — Epoch-level metrics table (train_loss, val_loss, reward_diff for all epochs), plus key wandb observations (accuracy_easy/hard, reward_diff_easy/hard, length_penalty)
5. **Comparison to baseline** — Side-by-side with baseline and previous best iteration
6. **Analysis** — Why did we see these results? Was the hypothesis confirmed?
7. **Open questions** — What remains unclear?
8. **Next iteration plan** — What to try next based on findings

### Naming convention
- `iteration0.md` — Initial baseline + diagnosis (this document)
- `iteration1.md` — First hyperparameter change (e.g., lambda_easy=5.0)
- `iteration2.md` — Refinement based on iteration 1 results
- etc.

### Evaluation checkpoint
After every 2-3 iterations, run the full evaluation pipeline (`evaluate_checkpoint()`) on the best model to get TPCA + accuracy on the test set. This is more expensive than wandb metrics but gives the ground-truth success measurement.
