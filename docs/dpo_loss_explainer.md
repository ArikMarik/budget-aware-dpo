# DPO Loss: Theory, Implementation, and What Went Wrong

This document is a ground-up explanation of how DPO loss works, how our Budget-Aware variant
extends it, and how the bugs we encountered connect directly to the math.

---

## Part 1: What Problem Does DPO Solve?

Imagine you want to teach a language model to be more helpful. You could use supervised
fine-tuning (SFT) — just show it good answers and maximize the probability of those tokens.
But the thing you actually care about is harder to state: you want the model to prefer good
answers *over* bad ones.

The naive solution is Reinforcement Learning from Human Feedback (RLHF). You:

1. Train a reward model on human preference labels.
2. Run PPO to optimize the policy toward high reward.

This works but is notoriously unstable and requires running three separate models at once
(policy, reference, reward model) plus a PPO update loop with clipping, value baselines,
and careful hyperparameter tuning.

**DPO (Direct Preference Optimization, Rafailov et al. 2023)** is the insight that you can
skip the reward model entirely. It turns out that, given a preference dataset, the optimal
policy has a closed form — and that closed form can be turned directly into a loss function
that you optimize with standard gradient descent.

---

## Part 2: The Standard DPO Loss

### 2.1 The Setup

You have:
- A **policy model** π_θ — the model you are training.
- A **reference model** π_ref — a frozen copy of the starting checkpoint.
- A dataset of **(prompt x, chosen response y_w, rejected response y_l)** triples.

The implicit reward that DPO defines for any response y given prompt x is:

```
r(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ]
```

Read this as: "how much more likely is the policy to generate y, relative to where it
started?" The `β` (beta) hyperparameter controls how far the policy is allowed to drift
from the reference. Large β → policy stays close to reference. Small β → policy can move
further.

### 2.2 The Loss

DPO trains the policy to make the *chosen* response have higher implicit reward than the
*rejected* response:

```
reward_diff = r(x, y_w) - r(x, y_l)
            = β · [log π_θ(y_w|x) - log π_ref(y_w|x)]
            - β · [log π_θ(y_l|x) - log π_ref(y_l|x)]
```

We want `reward_diff > 0`, and we use a sigmoid to convert this to a probability:

```
L_DPO = -log σ(reward_diff)
```

`σ` is the logistic sigmoid function. When `reward_diff` is large and positive, `σ → 1`
and the loss → 0 (the model has clearly learned to prefer chosen over rejected). When
`reward_diff ≈ 0`, `σ ≈ 0.5` and the loss ≈ `log(2) ≈ 0.693` (random chance — neither
is preferred). When `reward_diff` is very negative, the loss is large (the model is
actively preferring the wrong answer).

**In code** (`src/models/standard_dpo_loss.py`):

```python
log_ratio_chosen   = policy_chosen_logps   - reference_chosen_logps
log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)
loss = -F.logsigmoid(reward_diff).mean()
```

This is a direct transcription of the math above.

### 2.3 What "log_prob" Actually Computes

Before we go further, it is essential to understand what `log_prob(logits, input_ids)`
returns (`dpo_trainer.py:85`):

```python
def log_prob(logits, input_ids):
    shift_logits = logits[..., :-1, :]      # predictions at positions 0..T-2
    shift_labels = input_ids[..., 1:]       # targets at positions 1..T-1
    log_probs = log_softmax(shift_logits)
    return gather(log_probs, shift_labels).sum(-1)   # ← SUM over all tokens
```

For a sequence of T tokens, this returns the **sum** of log-probabilities:

```
log π(y|x) = Σ_{t=1}^{T} log π(token_t | token_1, ..., token_{t-1})
```

This is the log of the joint probability of the entire sequence. It grows (in magnitude)
linearly with sequence length. A sequence of 500 tokens might have a log-prob of -1200.
A sequence of 100 tokens might have -200. We will return to why this matters in Part 4.

---

## Part 3: Budget-Aware DPO

### 3.1 The Motivation

Standard DPO has no opinion about answer length. For math problems, this is a problem:

- **Easy problems**: A short, direct solution is better. We want to penalize verbose
  "thinking out loud" that wastes tokens without adding accuracy.
- **Hard problems**: A longer chain-of-thought reasoning is actually necessary. We do not
  want to penalize length here.

Our dataset labels each problem with a **complexity** label:
- `C=0` (Easy) — solvable with a short, direct solution
- `C=1` (Hard) — benefits from extended reasoning

### 3.2 The Modified Reward

Budget-Aware DPO introduces a length penalty term per sample:

```
R_budget(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ] - λ(C) · |y|
```

Where `|y|` is the length of the response in tokens, and:
- `λ(C=0) = lambda_easy ≈ 0.05` — penalty is active for easy problems
- `λ(C=1) = lambda_hard ≈ 0.001` — penalty is near-zero for hard problems

The loss becomes:

```
reward_diff_budget = β·(log_ratio_chosen - log_ratio_rejected) - λ(C)·(|y_w| - |y_l|)
L_budget = -log σ(reward_diff_budget)
```

The length term subtracts a penalty for the *difference* in length between chosen and
rejected. If chosen is shorter (|y_w| < |y_l|), then `(|y_w| - |y_l|) < 0`, meaning
the penalty *adds* to reward_diff — it rewards the shorter solution.

**In code** (`src/models/budget_aware_dpo_loss.py`):

```python
lambdas = torch.where(complexities == 0, lambda_easy, lambda_hard)
length_diff    = chosen_lengths - rejected_lengths
length_penalty = lambdas * length_diff
reward_diff    = beta * (log_ratio_chosen - log_ratio_rejected) - length_penalty
loss = -F.logsigmoid(reward_diff).mean()
```

---

## Part 4: The Bugs and What They Mean

Here are the two training runs we observed:

**Run 1 (run-20260319_142016):**
```
Step 100: train_loss=0.8517, val_loss=1.6900
Step 200: train_loss=0.0000, val_loss=1.5057   ← collapse
Step 300: train_loss=0.1937, val_loss=1.5036
Step 400: train_loss=2.6875, val_loss=1.9524   ← spike
Step 500: train_loss=0.0000, val_loss=0.8609   ← crash (Tensor not JSON serializable)
```

**Run 2 (run-20260319_140244, latest run):**
```
Step 100: train_loss=1.6444, val_loss=1.6135
Step 200: train_loss=0.0000, val_loss=1.4366   ← collapse
Step 300: train_loss=0.0011, val_loss=1.5257
Step 400: train_loss=7.5000, val_loss=2.7660   ← large spike
Step 500: (truncated)
```

Three things are happening. Each has a direct mathematical cause.

---

### Bug 1: The Tensor Serialization Crash

**What happened:** Training crashed at step 500 with:
```
TypeError: Object of type Tensor is not JSON serializable
```

**The cause:** In `budget_aware_dpo_loss.py`, the `extra` dictionary that gets logged
returned a PyTorch Tensor, not a Python float:

```python
# BEFORE (broken):
length_penalty_mean = length_penalty.detach().mean()        # ← still a Tensor
return loss, {"length_penalty": length_penalty_mean}

# AFTER (fixed):
length_penalty_mean = length_penalty.detach().mean().item() # ← Python float
return loss, {"length_penalty": length_penalty_mean}
```

**The connection to the math:** `.mean()` on a PyTorch tensor returns a 0-dimensional
tensor, not a Python scalar. The distinction between "a tensor holding the value 3.14"
and "the Python float 3.14" matters the moment you try to serialize to JSON — Python's
`json` module knows how to handle floats, dicts, lists, and strings, but not PyTorch
memory objects. The fix is `.item()`, which extracts the scalar value from the tensor
into Python's native type system.

---

### Bug 2: `loss=0.0000` — The Sigmoid Saturation / Reward Collapse

**What happened:** After only 100–200 steps, the training loss dropped to exactly 0.
Yet the validation loss remained high (1.4–1.5). This is a collapse — not healthy
convergence.

**The math:** Recall the loss is:

```
L = -log σ(reward_diff)
```

The sigmoid function σ maps any real number to (0, 1):
- σ(-∞) = 0  →  -log(0) = +∞  (very bad — model strongly prefers wrong answer)
- σ(0)  = 0.5 →  -log(0.5) ≈ 0.693  (model is random)
- σ(+∞) = 1  →  -log(1) = 0  (model is certain — loss is zero)

**Loss = 0 means reward_diff has become enormous.** The model has pushed `log π_θ(y_w|x)`
much higher than `log π_θ(y_l|x)` — to the point where the sigmoid saturates to 1 and
the loss signal essentially vanishes.

This sounds like success (it's preferring chosen!) but it is not:
1. The gradient of the loss through the sigmoid becomes vanishingly small. The model
   has entered a flat region and stops learning.
2. Validation loss stays high, which means the "preference" the model learned is brittle —
   it only works on the exact training sequences it memorized, not on new problems.

**Why does this happen so fast?** The root cause is the **unnormalized log-prob** (see
Part 2.3). Because `log_prob` *sums* over all T tokens, a sequence of 500 math tokens
has log-prob ≈ -1000 or lower. The log-ratio `log π_θ - log π_ref` can therefore swing
by hundreds of units when the model is updated. Multiplied by beta=0.1, `reward_diff`
can reach ±10, ±50, or larger — values where σ is completely saturated. A single gradient
step with an unchecked learning rate can push `reward_diff` from 0.5 to 50.

The standard fix is to **normalize by sequence length**:

```python
# Instead of:
return gather(log_probs, labels).sum(-1)   # grows with T

# Use:
token_count = attention_mask.sum(-1).float()
return gather(log_probs, labels).sum(-1) / token_count  # O(1) regardless of T
```

This keeps log-ratios in a numerically stable range regardless of how long the sequences
are.

---

### Bug 3: `loss=2.6875` and `loss=7.5000` — The Loss Spikes

**What happened:** After collapsing to near-zero, the training loss then *spiked* to 2.69
and 7.5 — far above the initial value of ~0.85.

**The math:** After the model's parameters have been pushed to a region where `reward_diff`
is huge, the gradient of the sigmoid is near zero... until the DataLoader shuffles and
presents a batch where `reward_diff` is *negative* for that specific data. Now the loss
is large, and the gradient is:

```
∂L/∂reward_diff = σ(reward_diff) - 1
```

When `reward_diff` is very negative, `σ → 0`, and the gradient magnitude → 1 — the
*maximum possible value*. This full-magnitude gradient propagates back through the entire
computation graph, through LoRA layers, back to the log-prob computation.

**The critical missing line:** The code computes the gradient norm and logs it, but
**never clips it**:

```python
# dpo_trainer.py:569-575 — gradient norm computed but unused
grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        grad_norm += p.grad.data.norm(2).item() ** 2
grad_norm = grad_norm ** 0.5

# grad_norm is logged to wandb but then...
optimizer.step()  # ← fires with raw unclipped gradients
```

**What gradient clipping does:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
rescales all gradients so their global L2 norm is at most 1.0. This is a soft ceiling: if
the gradient is already small, nothing happens. If it is 50× larger than normal, it gets
rescaled down. This prevents a single bad batch from causing a catastrophic weight update.

The fix should go between `.backward()` and `.step()`:

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### Bug 4: Length Penalty Scale Mismatch (Budget-Aware Only)

**What happened:** This is a subtler instability, not visible as a crash but present as
noise in the Budget-Aware runs.

**The math:** The DPO reward_diff has two terms:

```
reward_diff = β · (log_ratio_chosen - log_ratio_rejected)   [term A]
            - λ · (|y_chosen| - |y_rejected|)               [term B]
```

**Term A** is O(0.1 – 2.0) in a healthy run. Beta is 0.1 and log-ratios are small.

**Term B** with raw token counts: `lambda_easy = 0.05`, length difference of 200 tokens →
`0.05 × 200 = 10.0`. This is **5–100× larger** than term A.

The length penalty completely dominates the reward signal for easy problems. The model
is no longer learning "prefer the correct answer" — it is learning "prefer whatever is
shorter, regardless of correctness."

The fix is to normalize the length difference by the average sequence length:

```python
avg_len = (chosen_lengths + rejected_lengths) / 2.0
length_diff_normalized = (chosen_lengths - rejected_lengths) / avg_len
length_penalty = lambdas * length_diff_normalized
```

Now term B is always in the range (-1, +1), commensurate with term A.

---

## Part 5: Summary Table

| Symptom in logs | Mathematical cause | Root code cause | Fix |
|---|---|---|---|
| `TypeError: Tensor not JSON serializable` | Tensor ≠ Python float | `.mean()` without `.item()` in `budget_aware_dpo_loss.py:62` | Add `.item()` |
| `loss=0.0000` after ~200 steps | Sigmoid saturation: reward_diff → ∞ | `log_prob` sums over all tokens, creating huge log-ratios | Normalize log-prob by sequence length |
| `loss=7.5` spike after collapse | Unchecked gradient magnitude on bad batch after saturation | No `clip_grad_norm_` before `optimizer.step()` | Add gradient clipping with `max_norm=1.0` |
| Budget-aware noise / length penalty domination | λ · raw_token_diff >> β · log_ratio | `length_diff` in raw token counts, not normalized | Normalize `length_diff` by average sequence length |

---

## Part 6: The Pattern

These bugs are not random. They all follow the same structure: **a quantity that was
designed to be O(1) silently became O(T)**, where T is sequence length.

- `log_prob` is designed to measure "how well does the model fit this text" — but because
  it sums over T tokens, longer sequences dominate.
- `length_diff` is designed to measure "how much longer is chosen vs. rejected" — but raw
  token counts can be 10× or 100× larger than the DPO term they are paired with.
- The gradient norm is designed to stay bounded during healthy training — but without
  clipping, a single outlier batch can send it off to any magnitude.

The general lesson: **whenever you combine quantities with different natural scales, make
sure you normalize them to a comparable range before adding or comparing them.** The DPO
formula only behaves as described in the paper when all its components are in roughly the
same numerical range.

..... after fix ....
  What's healthy now:                                                                                                                                
  
  - No more total collapse. Before: loss=0.0000 locked in by step 200 and stayed there. Now: near-zero batches (0.0004, 0.0008) appear but then      
  bounce back naturally. This is the gradient clipping working — one saturated batch can no longer drag the weights off a cliff.
  - Val loss genuinely decreases. 0.3778 → 0.1177. That's real learning happening. Before, val loss barely moved (1.50 → 1.50) while train loss      
  collapsed.                                                                                                                                         
  - The spike (1.0395) is 7× smaller than the previous worst (7.5). More importantly: val loss at step 700 is 0.1543 — still healthy. The bad batch
  didn't corrupt the model, it just momentarily pushed the loss up and the run recovered by step 800.                                                
  - Clean finish. Checkpoint saved, best model saved, no crash.
  - Early stopping triggered correctly at step 900. Best val was at step 400 (0.1177), and nothing improved on it within patience — that's the system
   working as designed.                                                                                                                              
                                                                                                                                                     
  What's still worth watching on the real run:                                                                                                       
                  
  - The near-zero train losses (0.0004, 0.0008) indicate individual batches are still saturating the sigmoid. The normalization fix reduced the scale
   of log-ratios, but with a small dummy dataset there's not enough variety to prevent it entirely. On the full dataset this should smooth out
  significantly.                                                                                                                                     
  - Val loss has some bounce (0.1177 → 0.2289 → 0.1177) without a clear LR schedule. A cosine or linear decay would help stabilize the tail of
  training.                                                                                                                                          
  - Early stopping at 900/1000 with best at 400 means the model plateaued halfway through. This is fine for a dummy run but on the full dataset you
  may want to increase patience or add LR warmup/decay to keep improving past that plateau.                                                          
                  
  Short answer: yes, this looks healthy for a dummy run. The pathological patterns (immediate collapse, unrecoverable spikes, crash) are gone.  