# The Researcher's Training Guide — Budget-Aware DPO

A ground-up guide to training, evaluating, and iterating on your Budget-Aware DPO model.
Written specifically for this project and this codebase.

---

## Table of Contents

1. [The Big Picture — What Are We Doing and Why](#1-the-big-picture)
2. [Mental Model — How Training Actually Works](#2-mental-model)
3. [Your Hyperparameters — What Each Knob Does](#3-your-hyperparameters)
4. [Reading WandB — What Good and Bad Looks Like](#4-reading-wandb)
5. [The Training Loop Walkthrough](#5-the-training-loop-walkthrough)
6. [Experiment Playbook — What to Try and In What Order](#6-experiment-playbook)
7. [Evaluation — How to Know if You Succeeded](#7-evaluation)
8. [Common Failure Modes and Fixes](#8-common-failure-modes)
9. [Practical Commands Reference](#9-practical-commands)
10. [Glossary](#10-glossary)

---

## 1. The Big Picture

### What are we training?

We are fine-tuning **Qwen-2.5-0.5B** (a 500M parameter language model) to solve math problems with **adaptive verbosity**:
- **Easy problems** (GSM8K-style arithmetic) → short, direct answers
- **Hard problems** (MATH competition-level) → full chain-of-thought reasoning

### Why DPO and not regular fine-tuning?

Regular supervised fine-tuning (SFT) says: "here's a good answer, learn to produce it."

DPO says: "here are two answers — this one is *better* than that one. Learn to prefer the better one."

This is more powerful because:
- It teaches the model to *discriminate* between good and bad, not just memorize
- It doesn't require training a separate reward model (like RLHF does)
- The preference signal can encode subtleties (like "correct but too verbose")

### What makes ours "budget-aware"?

Standard DPO doesn't care about answer length. Our variant adds a **length penalty** that is conditional on problem difficulty:

```
R_budget(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) − λ(C) · normalized_length
```

- For **easy** problems: λ = 0.05 (strong penalty → push toward shorter answers)
- For **hard** problems: λ = 0.001 (weak penalty → allow long chain-of-thought)

The hypothesis: this produces a model that is both **accurate** and **token-efficient**.

### What does success look like?

Your trained model should:
1. **Maintain accuracy** on both easy and hard problems (compared to baseline DPO)
2. **Use fewer tokens on easy problems** (the efficiency gain)
3. **Not sacrifice reasoning quality on hard problems** (no regression)

The key metric: **TPCA (Tokens Per Correct Answer)** — lower is better, because it means the model is getting correct answers with fewer tokens.

---

## 2. Mental Model — How Training Actually Works

### The Two Models

During training, two copies of Qwen-2.5-0.5B are loaded into GPU memory:

1. **Policy model (π_θ)** — the model being trained. Has LoRA adapters attached. Gradients flow through this one.
2. **Reference model (π_ref)** — a frozen copy of the original weights. Never updated. Serves as the "anchor" so the policy doesn't drift too far from the base model.

### What Happens Each Step

```
For each batch of DPO pairs:
  1. Feed the CHOSEN response through both models → get log-probabilities
  2. Feed the REJECTED response through both models → get log-probabilities
  3. Compute log-ratio: how much more does the policy like this text vs. the reference?
  4. Compute reward_diff = β × (log_ratio_chosen − log_ratio_rejected) − length_penalty
  5. Loss = −log(sigmoid(reward_diff))
  6. Backpropagate, clip gradients, update LoRA weights
```

### Why LoRA Instead of Full Fine-Tuning?

LoRA (Low-Rank Adaptation) freezes 99%+ of the model weights and only trains small adapter matrices inserted into the attention layers. Benefits:
- **Memory**: ~2GB for LoRA vs ~4GB+ for full fine-tuning (for a 0.5B model)
- **Speed**: Fewer parameters to update = faster steps
- **Regularization**: The frozen base model acts as a strong prior, reducing overfitting
- **Portability**: The adapter is a small file (~500MB) that sits on top of the base model

Our config: `r=128, alpha=256, targets=[q_proj, v_proj, k_proj, o_proj], dropout=0.05`
- **r=128** is relatively high rank — gives the model a lot of expressive capacity
- **alpha=256** (= 2×r) is the scaling factor; the effective learning rate for LoRA layers is proportional to alpha/r = 2.0
- **target_modules**: we adapt all four attention projections (query, key, value, output)

---

## 3. Your Hyperparameters — What Each Knob Does

### The Most Important Ones (Tune These First)

| Parameter | Default | What It Controls | How to Think About It |
|-----------|---------|------------------|----------------------|
| `--lr` | 1e-5 | Learning rate | How big each weight update is. Too high → instability/spikes. Too low → training is painfully slow and may not converge. |
| `--batch-size` | 4 | Samples per forward pass | Larger = more stable gradients but more GPU memory. If you get OOM, reduce this first. |
| `--gradient-accumulation-steps` | 1 | Virtual batch size multiplier | Effective batch = batch_size × grad_accum. Use this to simulate larger batches without more memory. E.g., `--batch-size 4 --gradient-accumulation-steps 8` = effective batch of 32. |
| `--max-epochs` | 10 | Maximum training epochs | One epoch = one full pass through the dataset. With 46k training pairs, each epoch is 46000/batch_size steps. |
| `--dpo-beta` | 0.1 | DPO temperature (β) | Controls how far the policy can drift from the reference. Lower β = policy can change more aggressively. Higher β = more conservative updates. 0.1 is the standard value from the DPO paper. |

### Budget-Aware Specific

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `--lambda-easy` | 0.05 | Length penalty strength for Easy problems (C=0). Higher = more pressure to be concise. |
| `--lambda-hard` | 0.001 | Length penalty strength for Hard problems (C=1). Keep this low — you don't want to penalize reasoning. |

**Key insight**: The length penalty is *normalized* by average sequence length in our implementation, so these values produce penalties in the range of ~(-0.05, +0.05). This keeps them commensurate with the DPO log-ratio term. If the length penalty dominates the reward signal, the model learns "be short" instead of "be correct and concise."

### Training Stability

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `--early-stopping-patience` | 5 | How many epochs of no improvement before stopping. Set to 3 for faster experiments, 7-10 for final runs. |
| `--early-stopping-threshold` | 0.0 | Minimum relative improvement to count as "better." 0.0 means any improvement counts. Set to 0.01 to require at least 1% improvement. |
| `--no-mixed-precision` | false | Disables FP16 mixed precision. Only use if you see NaN losses — mixed precision is faster and uses less memory. |

### Practical Ones

| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `--checkpoint-every` | 1 | Save checkpoint every N epochs. Each checkpoint is ~500MB. For long runs, set to 2-3 to save disk space. |
| `--data-limit` | None | Use only first N pairs. Great for quick smoke tests: `--data-limit 500` runs a mini-experiment in minutes. |
| `--seed` | 42 | Random seed. Change to run multiple trials: `--seed 42`, `--seed 123`, `--seed 7`. Report mean ± std across seeds. |
| `--resume-from` | None | Path to a checkpoint directory to resume training from. |
| `--num-workers` | 4 | DataLoader workers. Keep at 4 unless you have CPU bottlenecks. |

---

## 4. Reading WandB — What Good and Bad Looks Like

### The Metrics You'll See — Deep Dive

Each WandB metric tells a specific story about your training. Here's what they all mean and why they matter.

#### `train/loss` and `val/loss` — The DPO Preference Loss

| Metric | Healthy Range |
|--------|---------------|
| `train/loss` | Starts ~0.693, decreases to 0.1–0.4 |
| `val/loss` | Tracks train_loss but slightly higher |

**The intuition**: DPO loss = `-log(sigmoid(reward_diff))`. Think of it as: "how confident is the model that the chosen response is better than the rejected one?"

- At `loss = 0.693` (`log(2)`): the model is at 50/50 — coin flip between chosen and rejected. This is where every DPO run starts because the policy is identical to the reference, so `reward_diff = 0` and `-log(sigmoid(0)) = -log(0.5) = log(2)`.
- At `loss = 0.3`: the model gets it right ~74% of the time.
- At `loss = 0.1`: the model gets it right ~90% of the time.
- At `loss → 0`: the model is perfectly confident on every pair (possibly overfitting!).

**The gap between train and val loss** is your generalization gap. If val loss starts climbing while train loss keeps dropping, the model is memorizing training pairs instead of learning general preferences.

#### `train/reward_diff` — The Most Important Metric

**What it is**: `β × ((log π_θ(chosen) - log π_ref(chosen)) − (log π_θ(rejected) - log π_ref(rejected)))`

**In plain English**: How much more does the policy model prefer the chosen response over the rejected response, compared to how the reference model felt? This is the *implicit reward margin*.

- **Starts near 0**: Policy = reference, no learned preference yet.
- **Goes positive and growing**: The model is learning to prefer chosen responses. This is what you want.
- **Goes very negative**: The model is learning to prefer rejected responses. Something is wrong.
- **Plateaus**: The model has learned what it can from the data. Check val/reward_diff — if it plateaued too, that's your model's ceiling.

**Why this matters more than loss**: Loss can decrease just by the model becoming more "confident" about existing preferences. Reward_diff directly measures whether the model is developing *new* preferences for the right answers. In your paper, plot this curve — it's the clearest evidence of learning.

#### `train/gradient_norm` — Pre-Clipping Gradient Magnitude

**What it is**: The L2 norm of all gradients *before* clipping is applied (max_norm=1.0).

**Why it can be very high (50–300+)**: This is normal for LoRA DPO training. Here's why:
1. LoRA with `alpha/r = 256/128 = 2.0` amplifies gradients by 2x through the scaling factor.
2. With batch_size=4, each gradient is computed from only 4 samples — high variance.
3. Mixed precision (FP16) can produce noisier gradients than FP32.
4. The DPO loss landscape has sharp curvatures — small weight changes can flip preferences.

**What clipping does**: `clip_grad_norm_(model.parameters(), max_norm=1.0)` rescales the entire gradient vector so its L2 norm is at most 1.0. If the raw norm is 300, every gradient component gets divided by 300. The model still updates in the correct *direction*, but takes a much smaller step.

**The metric shows pre-clip values** (the return value of `clip_grad_norm_`). This is useful information: it tells you how much the model "wants" to change. If raw norms are consistently 100-300x the clip threshold, the model is effectively doing **gradient sign descent** — every step is the same magnitude, only the direction varies. This isn't ideal but works in practice (Adam's momentum helps smooth things out).

**When to worry**:
- Raw norms suddenly jump from 50 to 10,000+ → possible NaN or corrupted batch
- Raw norms drop to near 0 → sigmoid saturation, model stopped learning
- If you want smoother training, try `--gradient-accumulation-steps 4` (averages gradients over more samples before clipping)

#### `train/length_penalty` — Budget-Aware Only

**What it is**: `λ(complexity) × (chosen_len − rejected_len) / avg_len`, averaged across the batch.

**Only appears for budget-aware runs** (baseline has no length penalty, so this chart won't exist).

This tells you how much the length penalty is pushing the model:
- **Positive**: Chosen responses are longer than rejected → penalty pushes model to prefer shorter
- **Negative**: Chosen responses are shorter than rejected → penalty rewards the model for the shorter choice
- **Near zero**: Either lengths are similar, or λ is very small (hard problems)

**Healthy range**: ±0.01 to ±0.05. If >0.5, the length penalty dominates the DPO signal and the model learns "be short" instead of "be correct and concise."

**Note**: This metric was missing from early runs due to a bug where the loss function returned `length_penalty` but the training loop didn't propagate it to WandB. Fixed in March 2026.

#### `train/token_diff` — Dataset Property, Not Model Behavior

**What it is**: `avg_chosen_tokens − avg_rejected_tokens` for the current batch.

**Why it's constant**: This measures your *data*, not your *model*. The chosen and rejected responses in your dataset are fixed text — they don't change as training progresses. This metric is a sanity check that the data loader is working correctly.

**What the value tells you**: If token_diff is negative, chosen responses tend to be shorter than rejected ones (true for Easy problems: chosen ~60 tokens, rejected ~151 tokens). If positive, chosen responses are longer (true for Hard problems: chosen ~799 tokens, rejected ~543 tokens). The overall average depends on your Easy/Hard ratio.

#### `train/avg_chosen_tokens` and `train/avg_rejected_tokens`

Same story — these are **dataset statistics**, not model outputs. They should be roughly constant throughout training. If they change dramatically, something is wrong with your data loading.

#### `train/complexity_0_loss` and `train/complexity_1_loss`

**What they are**: The DPO loss computed separately for Easy (complexity=0) and Hard (complexity=1) problems.

**Why they matter**: This is where you see if the budget-aware penalty is working. In the budget-aware run:
- `complexity_0_loss` (Easy) should decrease differently than in baseline — the length penalty is steering the model to prefer shorter answers
- `complexity_1_loss` (Hard) should track the baseline closely — the tiny λ_hard means almost no length pressure

**In your paper**: Plot these side-by-side (baseline vs budget-aware) to show that the length penalty selectively affects Easy problems.

### Patterns to Watch For

#### Good Training Run
```
Epoch 1: train_loss=0.68, val_loss=0.65, reward_diff=0.12
Epoch 2: train_loss=0.52, val_loss=0.50, reward_diff=0.45
Epoch 3: train_loss=0.41, val_loss=0.42, reward_diff=0.78
Epoch 4: train_loss=0.35, val_loss=0.38, reward_diff=1.05
Epoch 5: train_loss=0.30, val_loss=0.36, reward_diff=1.20  ← gap starting to form
Early stopping at epoch 8, best model at epoch 5
```

What to look for:
- Loss decreases smoothly
- Reward diff increases (model increasingly prefers chosen over rejected)
- Train and val losses are close (small generalization gap)
- Val loss eventually plateaus or starts increasing → early stopping triggers correctly

#### Bad: Loss Collapse (loss → 0 too fast)
```
Epoch 1: train_loss=0.65
Epoch 2: train_loss=0.001  ← collapse!
Epoch 3: train_loss=0.000
```
**Cause**: Learning rate too high, or log-probabilities not properly normalized.
**Fix**: Reduce `--lr` by 3-5x. Try `3e-6` or `5e-6` instead of `1e-5`.

#### Bad: Loss Spikes
```
Epoch 3: train_loss=0.35
Epoch 4: train_loss=4.50  ← spike!
Epoch 5: train_loss=0.40
```
**Cause**: A batch where the model strongly prefers the wrong answer causes a huge gradient.
**Fix**: Already mitigated by gradient clipping (max_norm=1.0). If still happening, increase `--gradient-accumulation-steps` to average over more samples before updating.

#### Bad: Overfitting
```
Epoch 3: train_loss=0.25, val_loss=0.30
Epoch 5: train_loss=0.12, val_loss=0.35  ← gap widening
Epoch 7: train_loss=0.05, val_loss=0.50  ← val getting worse
```
**Cause**: Model memorizes training data instead of learning generalizable preferences.
**Fix**:
- Reduce `--max-epochs` or rely on early stopping
- Increase LoRA dropout (would require code change, currently 0.05)
- Use more data (our balanced 50k dataset should help)

#### Bad: No Learning
```
Epoch 1: train_loss=0.69, val_loss=0.69
Epoch 5: train_loss=0.68, val_loss=0.69
Epoch 10: train_loss=0.67, val_loss=0.69
```
**Cause**: Learning rate too low, or the data doesn't have a learnable preference signal.
**Fix**: Increase `--lr` by 3-5x. Verify your data has meaningful chosen/rejected differences.

---

## 5. The Training Loop Walkthrough

Here's exactly what happens when you run the training command, in order:

### Phase 1: Setup (~30 seconds)
1. Load Qwen-2.5-0.5B base model (~1GB download on first run, cached after)
2. Attach LoRA adapters to attention layers
3. Load a second copy as the frozen reference model
4. Load pre-tokenized training data from `train_tokens.pt` and `val_tokens.pt`
5. Create DataLoader with shuffling for train, no shuffling for val
6. Initialize AdamW optimizer with weight decay 0.01
7. Initialize WandB run (if `--wandb`)

### Phase 2: Training Loop (minutes to hours, depending on data size)
```
For each epoch (1 to max_epochs):
    Shuffle training data
    For each batch:
        Forward pass: policy model on chosen + rejected (4 forward passes total including ref)
        Compute DPO loss + length penalty
        Backward pass (compute gradients)
        If gradient_accumulation_steps reached:
            Clip gradients to max_norm=1.0
            Optimizer step (update LoRA weights)
            Log metrics to WandB

    Evaluate on validation set (no gradients)
    Log epoch-level metrics

    If val_loss improved:
        Save as best model
    Else:
        Increment early stopping counter

    If checkpoint_every epochs:
        Save checkpoint

    If early stopping triggered:
        Break
```

### Phase 3: Cleanup (~10 seconds)
1. Load best model state (from the epoch with lowest val_loss)
2. Save final model + tokenizer to output directory
3. Save metrics.json with full training history
4. Close WandB run

### Memory Footprint

With the balanced dataset (50k pairs) and batch_size=4:
- **Policy model**: ~2GB (0.5B params in FP32 + LoRA adapters)
- **Reference model**: ~2GB (frozen, no gradients)
- **Optimizer states**: ~1GB (AdamW keeps 2 momentum buffers per parameter)
- **Activations + gradients**: ~1-3GB (depends on sequence length)
- **Data**: ~1.5GB for tokenized tensors
- **Total**: ~8-10GB VRAM

If you get `CUDA OutOfMemoryError`:
1. Reduce `--batch-size` to 2 or even 1
2. Increase `--gradient-accumulation-steps` proportionally
3. Remove `--compile-model` if set

---

## 6. Experiment Playbook — What to Try and In What Order

### Rule #1: Change One Thing at a Time

Never change multiple hyperparameters simultaneously. You won't know which one caused the effect.

### Rule #2: Start Small, Scale Up

Use `--data-limit 1000` for quick iterations (~5-10 min per run). Only scale to the full 50k when you've found a promising configuration.

### Suggested Experiment Sequence

#### Experiment 1: Baseline DPO (The Control)
Train standard DPO without any length penalty. This is your comparison point.

```bash
DATASET_VARIANT=balanced USE_DUMMY_DATA=0 python -m scripts.training.train_baseline_dpo \
    --output-dir checkpoints/baseline_balanced \
    --max-epochs 5 --batch-size 4 --lr 1e-5 --wandb
```

Record: final val_loss, reward_diff, accuracy on eval.

#### Experiment 2: Budget-Aware DPO (Default Lambdas)
Train with the default lambda values (0.05 easy, 0.001 hard).

```bash
DATASET_VARIANT=balanced USE_DUMMY_DATA=0 python -m scripts.training.train_budget_aware_dpo \
    --output-dir checkpoints/budget_aware_balanced \
    --max-epochs 5 --batch-size 4 --lr 1e-5 --wandb
```

Compare to Experiment 1: Does the length penalty change token counts without hurting accuracy?

#### Experiment 3: Lambda Sweep (The Core Experiment)
Test different lambda-easy values to find the sweet spot:

| Run | lambda-easy | lambda-hard | Hypothesis |
|-----|------------|-------------|------------|
| 3a  | 0.01       | 0.001       | Gentle length pressure |
| 3b  | 0.05       | 0.001       | Default (moderate) |
| 3c  | 0.10       | 0.001       | Aggressive shortening |
| 3d  | 0.20       | 0.001       | Very aggressive — may hurt accuracy |

Start with `--data-limit 2000` for fast iteration, then run the winner on the full dataset.

#### Experiment 4: Learning Rate Sensitivity
DPO is sensitive to learning rate. Test:

| Run | lr    | Expected Behavior |
|-----|-------|-------------------|
| 4a  | 3e-6  | More stable, slower convergence |
| 4b  | 1e-5  | Default |
| 4c  | 3e-5  | Faster but potentially unstable |
| 4d  | 5e-5  | Likely too high — expect spikes |

#### Experiment 5: Beta Sensitivity
Test how DPO temperature affects learning:

| Run | beta | Effect |
|-----|------|--------|
| 5a  | 0.05 | Policy can drift further from reference |
| 5b  | 0.1  | Default |
| 5c  | 0.2  | More conservative, closer to reference |
| 5d  | 0.5  | Very conservative — may underfit |

#### Experiment 6: Multiple Seeds (For the Paper)
Once you have your best config, run it 3 times with different seeds and report mean ± std:

```bash
for SEED in 42 123 7; do
    DATASET_VARIANT=balanced USE_DUMMY_DATA=0 python -m scripts.training.train_budget_aware_dpo \
        --output-dir checkpoints/budget_aware_seed_${SEED} \
        --seed ${SEED} --max-epochs 5 --batch-size 4 --lr 1e-5 --wandb
done
```

### How to Compare Runs

In WandB:
1. Go to your project dashboard
2. Select runs to compare
3. Create a comparison table or overlay plots
4. Key comparisons:
   - val/loss curves overlaid
   - val/reward_diff overlaid
   - train/length_penalty (budget-aware runs only)
   - complexity_0_loss vs complexity_1_loss

---

## 7. Evaluation — How to Know if You Succeeded

### Step 1: Training Metrics (During Training)

These tell you if the model *learned the preference signal*:
- **val/loss < 0.5**: The model distinguishes chosen from rejected
- **val/reward_diff > 0.5**: The model reliably prefers chosen responses
- **No overfitting**: val_loss not diverging from train_loss

### Step 2: Generation Quality (After Training)

Run the evaluation script to generate answers and measure accuracy:

```bash
USE_DUMMY_DATA=0 python -m scripts.run_evaluation --limit 200
```

Key metrics:
- **Accuracy**: % of problems where the model gets the correct answer
  - Report separately for Easy (GSM8K) and Hard (MATH)
  - Budget-aware should match or slightly trail baseline on accuracy
- **TPCA (Tokens Per Correct Answer)**: Total tokens generated / number of correct answers
  - Budget-aware should have *lower* TPCA on easy problems (more efficient)
  - Similar TPCA on hard problems (no reasoning cut short)
- **Average Response Length**: By complexity level
  - Easy: budget-aware should generate significantly shorter responses
  - Hard: similar length to baseline

### Step 3: The Results Table (For Your Paper)

The ideal results table looks like:

| Model | GSM8K Acc | MATH Acc | TPCA (Easy) | TPCA (Hard) | Avg Tokens (Easy) | Avg Tokens (Hard) |
|-------|-----------|----------|-------------|-------------|--------------------|--------------------|
| Base Qwen-0.5B | X% | Y% | A | B | C | D |
| + Standard DPO | X+2% | Y+1% | A-10 | B | C-5 | D |
| + Budget-Aware DPO | X+1% | Y+1% | **A-50** | B | **C-40** | D |

The story: Budget-aware DPO maintains accuracy while dramatically reducing token usage on easy problems.

### What "Success" Looks Like for Your Paper

1. **Primary claim**: Budget-aware DPO reduces TPCA on easy problems by 20%+ without accuracy loss
2. **Secondary claim**: Hard problem performance is preserved (no regression)
3. **Ablation**: Different lambda values show a smooth accuracy-efficiency trade-off

### What "Interesting Negative Result" Looks Like

If budget-aware DPO *doesn't* work as expected, that's still publishable if you can explain *why*:
- "The length penalty interfered with the DPO preference signal" → analysis of loss components
- "Easy/hard classification was too noisy" → error analysis of complexity labels
- "0.5B model is too small to develop adaptive behavior" → compare to larger model results in literature

---

## 8. Common Failure Modes and Fixes

### Loss Keeps Growing (Running Sum Bug) — FIXED March 2026

If you see loss climbing linearly like this:
```
step=1,  loss=0.6931
step=2,  loss=1.3834
step=3,  loss=2.0750
step=10, loss=6.8986
```

**What's happening**: The displayed loss is a *cumulative sum* (0.69 + 0.69 + 0.69 + ...) instead of a *running average*. The real per-step loss is ~0.69 at the start — exactly `log(2)`, which is the theoretically expected initial DPO loss when the model has no preference yet.

**Why log(2)?** DPO loss = `-log(sigmoid(reward_diff))`. Before training, the policy and reference models are identical, so reward_diff ≈ 0. And `-log(sigmoid(0)) = -log(0.5) = log(2) ≈ 0.693`. This is the "random chance" baseline — the model is equally likely to prefer chosen or rejected.

**How this was fixed**: In `dpo_trainer.py`, the progress bar and per-step WandB logging were dividing `accum_loss` by `gradient_accumulation_steps` (which is 1) instead of dividing by `num_steps_so_far` (the running count of batches). The other metrics (reward_diff, complexity losses) were already correctly averaged — only the loss had this bug.

**How to tell it's fixed**: After the fix, loss should hover around 0.69 at the start and gradually decrease. If it goes up linearly, the bug is back.

**Lesson**: When you see a metric that "keeps going up," always ask: **is the actual value increasing, or is the display doing a cumulative sum?** Check the first few values — if `value_at_step_N ≈ value_at_step_1 × N`, it's a sum.

### Baseline and Budget-Aware Look Identical

If early in training (epoch 1) both runs show nearly the same loss:

**This is expected.** Both models start from the exact same Qwen-2.5-0.5B weights. The only difference is the length penalty term, which adds ±0.01 to the loss. At the start, the DPO loss itself is ~0.69, so a ±0.01 difference is noise. You need to wait multiple epochs before the length penalty steers the models apart. The divergence shows up in:
- Token counts at *evaluation* time (after training is done)
- The `train/length_penalty` metric in WandB (budget-aware only)
- Response lengths when you actually generate text

**What about constant metrics?** `avg_chosen_tokens` and `avg_rejected_tokens` *should* be roughly constant — they measure the data you're feeding in, not the model's output. The chosen/rejected responses in your dataset are fixed. These are sanity-check metrics.

### NaN Loss
```
train/loss: nan
```
**Cause**: Numerical instability, usually from very large log-probabilities.
**Fixes**:
1. Add `--no-mixed-precision` (FP16 has limited range)
2. Reduce learning rate
3. Check if your data has empty or corrupted sequences

### CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Fixes** (in order of preference):
1. `--batch-size 2` (or 1)
2. `--gradient-accumulation-steps 8` (to compensate for smaller batch)
3. Kill other GPU processes: `nvidia-smi` to check, `kill -9 <PID>` to free

### Training Takes Forever
**Diagnosis**: Check steps per epoch. With 46k training pairs and batch_size=4, that's ~11,500 steps per epoch. At ~0.5s per step, one epoch ≈ 1.5 hours.

**Fixes**:
1. Increase `--batch-size` (if GPU memory allows)
2. Use `--gradient-accumulation-steps` > 1 with larger batch for same effective batch but fewer optimizer steps
3. Use `--data-limit` for faster iteration during hyperparameter search
4. Reduce `--max-epochs` and rely on early stopping

### WandB Issues
```
wandb: ERROR Run initialization has timed out
```
**Fix**: `export WANDB_MODE=offline` — logs locally, sync later with `wandb sync`

### Model Outputs Garbage
After training, if the model generates nonsensical text:
- The LoRA adapter may have diverged. Load an earlier checkpoint.
- Check that evaluation uses the correct tokenizer (must match training).
- Verify the prompt format matches what was used in training: `"Problem: {problem}\nSolution: "`

---

## 9. Practical Commands Reference

### Smoke Test (5-10 minutes)
```bash
DATASET_VARIANT=balanced USE_DUMMY_DATA=0 python -m scripts.training.train_budget_aware_dpo \
    --output-dir checkpoints/smoke_test \
    --max-epochs 2 --batch-size 4 --data-limit 500 --wandb
```

### Full Budget-Aware Training on Balanced Dataset
```bash
DATASET_VARIANT=balanced USE_DUMMY_DATA=0 nohup python -m scripts.training.train_budget_aware_dpo \
    --output-dir checkpoints/budget_aware_balanced \
    --max-epochs 5 --batch-size 4 --lr 1e-5 \
    --early-stopping-patience 3 --wandb \
    > logs/budget_aware_balanced.log 2>&1 &
```

### Full Baseline Training on Balanced Dataset
```bash
DATASET_VARIANT=balanced USE_DUMMY_DATA=0 nohup python -m scripts.training.train_baseline_dpo \
    --output-dir checkpoints/baseline_balanced \
    --max-epochs 5 --batch-size 4 --lr 1e-5 \
    --early-stopping-patience 3 --wandb \
    > logs/baseline_balanced.log 2>&1 &
```

### Monitor Running Training
```bash
# Check GPU usage
nvidia-smi

# Watch training output
tail -f logs/budget_aware_balanced.log

# Check WandB dashboard
# → https://wandb.ai/<your-username>/budget-aware-dpo
```

### Evaluation After Training
```bash
# Quick eval (200 problems)
USE_DUMMY_DATA=0 python -m scripts.run_evaluation --limit 200

# Full eval
USE_DUMMY_DATA=0 python -m scripts.run_evaluation
```

### Time Estimates (approximate, GPU-dependent)

| Dataset Size | Batch Size | Steps/Epoch | Time/Epoch | 5 Epochs |
|-------------|------------|-------------|------------|----------|
| 500 pairs   | 4          | 125         | ~1 min     | ~5 min   |
| 5,000 pairs | 4          | 1,250       | ~10 min    | ~50 min  |
| 46,000 pairs| 4          | 11,500      | ~1.5 hr    | ~7.5 hr  |
| 46,000 pairs| 8          | 5,750       | ~1.5 hr    | ~7.5 hr  |

Note: Larger batch sizes don't halve epoch time because the forward pass does more work per step. They *do* make training more stable by averaging gradients over more samples.

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **DPO** | Direct Preference Optimization — trains a model to prefer chosen over rejected responses without a reward model |
| **LoRA** | Low-Rank Adaptation — fine-tunes a small set of adapter weights instead of the full model |
| **Epoch** | One complete pass through the entire training dataset |
| **Step** | One forward + backward pass on a single batch |
| **Batch size** | Number of DPO pairs processed in one forward pass |
| **Gradient accumulation** | Accumulate gradients over N batches before updating weights. Simulates a larger batch size. |
| **Mixed precision (FP16)** | Use 16-bit floats for forward pass (faster, less memory) and 32-bit for weight updates (precision). Handled automatically by `torch.amp`. |
| **Gradient clipping** | Cap the gradient L2 norm to prevent huge weight updates. We clip at max_norm=1.0. The logged `gradient_norm` is the *pre-clip* value. With LoRA DPO, raw norms of 50-300 are normal — clipping rescales them to 1.0 each step. |
| **Early stopping** | Stop training when validation loss hasn't improved for N epochs. Prevents overfitting. |
| **Reference model (π_ref)** | Frozen copy of the base model. DPO measures policy drift relative to this anchor. |
| **Policy model (π_θ)** | The model being trained. In our case, base Qwen + LoRA adapters. |
| **Log-ratio** | `log(π_θ(y|x)) − log(π_ref(y|x))` — how much more/less likely the policy makes response y compared to the reference. |
| **Reward diff** | Difference in implicit reward between chosen and rejected. Positive = model prefers chosen (good). |
| **β (beta)** | DPO temperature. Controls how aggressively the policy can deviate from the reference. Default: 0.1. |
| **λ (lambda)** | Length penalty coefficient. Per-complexity: λ_easy=0.05, λ_hard=0.001. |
| **TPCA** | Tokens Per Correct Answer — total generated tokens / number correct answers. Lower = more efficient. |
| **AdamW** | Optimizer. Adam with decoupled weight decay. The standard choice for transformer training. |
| **Sigmoid saturation** | When the input to sigmoid is very large (positive or negative), the output is ~1 or ~0, and the gradient is ~0. This stops learning. |
| **Weight decay** | Regularization that slightly shrinks weights each step (0.01 in our config). Prevents weights from growing unbounded. |
| **Validation split** | 10% of data held out for measuring generalization. Never trained on. |
| **Checkpoint** | Saved model weights at a specific epoch. Allows resuming training or picking the best model. |

---

## Appendix A: Your Dataset at a Glance

The **balanced** dataset you'll be training on:

| Stat | Value |
|------|-------|
| Total pairs | 50,000 |
| Easy (C=0) | 25,000 |
| Hard (C=1) | 25,000 |
| Train split | 46,290 (90%) |
| Val split | 3,710 (10%) |
| Avg chosen tokens (Easy) | 60.2 |
| Avg rejected tokens (Easy) | 151.1 |
| Avg length delta (Easy) | 90.9 tokens |
| Avg chosen tokens (Hard) | 798.6 |
| Avg rejected tokens (Hard) | 543.1 |
| Max token length | 512 (truncated) |

**Key observation**: For Easy pairs, chosen is ~60 tokens and rejected is ~151 tokens — a clear signal that "short is preferred." For Hard pairs, chosen is actually *longer* than rejected (799 vs 543) — meaning "more reasoning is preferred." This is exactly the signal the model should learn.

---

## Appendix B: Reading the Code

If you want to understand what's happening under the hood:

| File | What to Read | Why |
|------|-------------|-----|
| `src/models/budget_aware_dpo_loss.py` | The entire file (65 lines) | This IS your research contribution. Understand every line. |
| `src/models/standard_dpo_loss.py` | The entire file (24 lines) | The baseline you're comparing against |
| `src/training/dpo_trainer.py:103-109` | `log_prob()` function | How log-probabilities are computed. Note the normalization by sequence length. |
| `src/training/dpo_trainer.py:539-695` | `train_dpo()` function | The main training orchestrator |
| `src/training/dpo_trainer.py:707-883` | `_run_epoch()` function | The inner training loop with gradient accumulation |
| `docs/dpo_loss_explainer.md` | Full document | Deep dive into bugs encountered and their mathematical causes |

---

## Appendix C: What to Put in Your Paper

### Method Section Checklist
- [ ] DPO loss formulation (standard)
- [ ] Budget-aware modification (the length penalty term)
- [ ] How λ is routed based on complexity
- [ ] Length normalization (why raw token counts would break training)
- [ ] LoRA configuration and why full fine-tuning wasn't used
- [ ] Data: 4-way augmentation strategy
- [ ] Balanced subsampling strategy (25k Easy + 25k Hard from 3.9M source pairs)

### Results Section Checklist
- [ ] Baseline DPO vs Budget-Aware DPO comparison table
- [ ] TPCA by complexity level
- [ ] Accuracy by complexity level
- [ ] Lambda ablation (how different λ_easy values affect the accuracy-efficiency trade-off)
- [ ] Training curves (loss, reward_diff) showing convergence
- [ ] Response length histograms by model and complexity

### Discussion Points
- [ ] Trade-off between token efficiency and accuracy
- [ ] When does the length penalty help vs. hurt?
- [ ] Limitations: 0.5B model, specific math domains, binary complexity classification
- [ ] Future work: continuous complexity scores, larger models, other domains

---

## Appendix D: Copy-Paste Commands — Parallel Training on 2 GPUs

Run both Baseline DPO and Budget-Aware DPO simultaneously, each pinned to its own GPU.
`CUDA_VISIBLE_DEVICES` ensures each process sees only one GPU (as device `cuda:0`), so they don't compete for memory.

### Prerequisites

```bash
# Make sure you're in the project root with the venv active
cd /storage/arik/nlp_final_project
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Create logs directory (if not exists)
mkdir -p logs

# Verify both GPUs are available and free
nvidia-smi

# Verify balanced dataset exists
ls data/processed_dpo_dataset_balanced/train_tokens.pt
```

### Launch Both Runs (copy-paste this entire block)

```bash
# --- GPU 0: Baseline DPO ---
CUDA_VISIBLE_DEVICES=0 DATASET_VARIANT=balanced USE_DUMMY_DATA=0 \
nohup python -m scripts.training.train_baseline_dpo \
    --output-dir checkpoints/baseline_balanced \
    --max-epochs 5 \
    --batch-size 4 \
    --lr 1e-5 \
    --early-stopping-patience 3 \
    --run-name baseline_balanced_1 \
    --wandb \
    > logs/baseline_balanced.log 2>&1 &
echo "Baseline PID: $!"

# --- GPU 1: Budget-Aware DPO ---
CUDA_VISIBLE_DEVICES=1 DATASET_VARIANT=balanced USE_DUMMY_DATA=0 \
nohup python -m scripts.training.train_budget_aware_dpo \
    --output-dir checkpoints/budget_aware_balanced \
    --max-epochs 5 \
    --batch-size 4 \
    --lr 1e-5 \
    --lambda-easy 0.05 \
    --lambda-hard 0.001 \
    --early-stopping-patience 3 \
    --run-name budget_aware_balanced_1 \
    --wandb \
    > logs/budget_aware_balanced.log 2>&1 &
echo "Budget-Aware PID: $!"
```

### Monitor Both Runs

```bash
# Watch both logs side-by-side (in separate terminals, or use tmux)
tail -f logs/baseline_balanced.log         # Terminal 1
tail -f logs/budget_aware_balanced.log     # Terminal 2

# Or interleave both in one terminal
tail -f logs/baseline_balanced.log logs/budget_aware_balanced.log

# Check GPU utilization (both GPUs should show ~80-100% usage)
watch -n 5 nvidia-smi

# Check if both processes are still running
jobs -l
```

### After Both Finish — Evaluate

```bash
# Evaluate baseline
USE_DUMMY_DATA=0 python -m scripts.run_evaluation --limit 200

# Evaluate budget-aware
USE_DUMMY_DATA=0 python -m scripts.run_evaluation --limit 200

# Generate comparison figures
python -m scripts.run_visualization
```

### If You Need to Kill a Run

```bash
# Find the PIDs
ps aux | grep train_

# Kill a specific run
kill <PID>

# Or kill both
kill $(jobs -p)
```

### Troubleshooting Parallel Runs

| Issue | Cause | Fix |
|-------|-------|-----|
| Both runs on same GPU | `CUDA_VISIBLE_DEVICES` not set | Make sure it's on the same line as `nohup`, not exported globally |
| OOM on one GPU | Other process using that GPU | Check `nvidia-smi`, kill stale processes |
| WandB conflict | Two runs initializing simultaneously | Should be fine — WandB handles concurrent runs. If not, add `WANDB_RUN_NAME=baseline` / `WANDB_RUN_NAME=budget_aware` |
| One run finishes early | Early stopping triggered | Normal — check if the run that stopped had good val_loss. Different loss functions converge at different rates |
| `nohup: failed to run command` | venv not activated or PYTHONPATH not set | Run the prerequisites block first |
