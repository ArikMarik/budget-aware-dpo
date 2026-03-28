# Budget-Aware DPO — Phase 1 Summary & Phase 2 Handoff

**Date**: 2026-03-28
**Status**: Phase 1 complete. Ready for Phase 2.
**Branch**: `autoresearch/mar26`

---

## 1. What We Set Out To Do

Build a **Budget-Aware DPO** training method that teaches an LLM to allocate inference tokens based on problem difficulty:
- **Easy problems** (GSM8K, MATH Level 1-2): Generate shorter, more efficient responses
- **Hard problems** (MATH Level 4-5): Preserve or extend reasoning for accuracy

The mechanism: modify the DPO reward signal with a length penalty term:
```
R_budget = beta * log(pi/pi_ref) - lambda(C) * length_penalty - kl_penalty * |KL(policy||ref)|
```
Where `lambda(C)` varies by complexity class — high for Easy (penalize verbosity), zero for Hard.

**Model**: Qwen2.5-0.5B (500M params) with LoRA (r=128, alpha=256)
**Dataset**: OpenMathInstruct DPO pairs, balanced 50K (25K easy + 25K hard)
**Constraint**: DPO must be part of the approach.

---

## 2. Iteration Timeline

| Iter | Key Change | lr | KL | λ_easy | Dataset | In-Training Acc (best epoch) | Easy Tokens | Hard Tokens | TPCA | Outcome |
|------|-----------|-----|-----|--------|---------|------|-------------|-------------|------|---------|
| 0 | λ=0.05 (too small) | 1e-5 | 0 | 0.05 | balanced 50K (3,271 uniq) | N/A | N/A | N/A | N/A | No divergence from baseline |
| 1 | λ=5.0 | 1e-5 | 0 | 5.0 | balanced 50K (3,271 uniq) | 37.6% | 148 (**↑ wrong**) | 218 | 416 | Accuracy up but tokens 2x LONGER on easy |
| 2 | Capped 10K, grad_accum=4 | 1e-5 | 0 | 5.0 | capped50 10K (2,000 uniq) | 1-2% | 135 (**↓ correct!**) | 233 | 18,394 | First correct token direction, but 1% accuracy |
| 3 | Full 51K diverse source | 1e-5 | 0 | 5.0 | real_capped100 51K (11,647 uniq) | 0% | 256 (gibberish) | 256 | ∞ | **Model collapse** — lr too aggressive |
| 4 | lr=1e-6 + KL=0.1 | 1e-6 | 0.1 | 5.0 | balanced_v4 50K (6,347 uniq) | 3% (baseline) | 177 | 187 | 6,067 | KL too strong → budget model frozen |
| 5 | **KL=0.01** | 1e-6 | 0.01 | 5.0 | balanced_v4 50K (6,347 uniq) | **38%** (E3) | 148→193 | 242→246 | 578 | **Breakthrough** — accuracy + token direction |
| 6 | Baseline+KL (fair comparison) | 1e-6 | 0.01 | **0.0** | balanced_v4 50K (6,347 uniq) | **39%** (E3) | 120→198 | 228→256 | 564→583 | KL-only baseline matches budget accuracy |

---

## 3. The Definitive Comparison: Iteration 5 vs 6

This is the cleanest comparison we have. **The ONLY difference is lambda_easy (5.0 vs 0.0)**. Everything else identical: lr=1e-6, KL=0.01, same dataset, same LoRA config.

### In-Training Gen Eval (Tier 0+1, 100 problems from training set)

| Metric | Baseline iter6 E2 | Baseline iter6 E3 | Budget iter5 E2 | Budget iter5 E3 |
|--------|-------------------|-------------------|-----------------|-----------------|
| Overall Accuracy | 35% | **39%** | 27% | 38% |
| Easy Accuracy | 68% | **76%** | 52% | 68% |
| Hard Accuracy | 2% | 2% | 2% | **8%** |
| Tokens Easy | **144.0** | 198.4 | 148.3 | 193.3 |
| Tokens Hard | 250.9 | 256.0 | 242.1 | 245.9 |
| TPCA | **564** | 583 | 723 | 578 |
| Val Loss | 0.5703 | 0.6022 | 0.4561 | 0.4460 |

### Post-Training Eval (Tier 0+1+2, 500 held-out problems)

| Metric | Budget iter5 (λ=5.0) | Baseline iter6 (λ=0.0) | Δ |
|--------|---------------------|----------------------|---|
| Overall Accuracy | **22.0%** (110/500) | 21.2% (106/500) | +0.8% |
| Avg Tokens Easy | **177.4** | 179.4 | **-2.0 tokens** |
| Avg Tokens Hard | **194.8** | 198.3 | -3.5 tokens |
| TPCA | **845.9** | 890.8 | **-44.9 (5% better)** |
| MATH L4-5 Accuracy | 8.2% | **11.6%** | -3.4% |

**Verdict**: The length penalty produces a real but **marginal** effect. ~2 fewer tokens on easy, ~5% TPCA improvement. Not enough for a convincing paper contribution. Phase 2 needs a stronger mechanism.

---

## 4. Key Findings

### 4.1 The Length Penalty (lambda) Has Marginal Effect

The most important finding: **KL regularization, not the length penalty, is the dominant driver of training stability and accuracy.** When we added KL=0.01 to the baseline (iter 6), it matched budget-aware accuracy (39% vs 38%) without any length penalty.

The length penalty (lambda_easy=5.0) does NOT clearly shorten easy responses. Both models show similar token counts at comparable epochs. The token direction signal is weak and inconsistent.

**However**: Budget iter5 showed 8% hard accuracy vs baseline's persistent 2%. This is the most interesting signal — the length penalty may help the model reason better on hard problems, even though lambda_hard=0.0. This could be because the penalty on easy problems forces the model to develop more efficient reasoning that transfers.

### 4.2 In-Training vs Post-Training Gap

Budget iter5 in-training accuracy: 38% (E3). Post-training on held-out data: 22%. This ~16 point drop comes from:
- Training data overlap in gen eval (100 problems sampled from training set)
- Harder distribution in test set (MATH L3-5 problems)
- Stricter grading (Tier 2 LLM judge catches false positives)

**Implication**: In-training metrics are useful for relative comparison and early detection, but publishable numbers must come from post-training eval.

### 4.3 KL Penalty is Critical but Sensitive

| KL Weight | Effect |
|-----------|--------|
| 0.0 (iter 0-3) | No stability — model collapsed on diverse data at lr=1e-5 |
| 0.1 (iter 4) | Too strong — budget model completely frozen |
| **0.01** (iter 5-6) | Sweet spot — stable training + learning |

### 4.4 Learning Rate × Data Diversity Interaction

| lr | Small/concentrated data | Large/diverse data |
|----|------------------------|-------------------|
| 1e-5 | Works (iter 0-1: 3,271 unique) | Collapses (iter 3: 11,647 unique) |
| 1e-6 | Too slow without KL (iter 4: 3%) | Stable with KL (iter 5-6: 27-39%) |

### 4.5 Hard Problem Accuracy is a Model Limitation

Qwen2.5-0.5B consistently gets 0-8% on hard problems (in-training) and ~15% on MATH overall (post-training, including easier MATH levels). This is a model capability ceiling — 500M parameters cannot reliably solve multi-step abstract reasoning.

### 4.6 DPO Preference ≠ Generation Behavior

Iteration 1 showed the model can learn to **prefer** shorter responses (reward_diff positive) while **generating** longer ones. DPO trains preferences, not generation behavior directly.

### 4.7 Dataset Design Matters Enormously

| Dataset | Unique Problems | Pairs/Problem | Result |
|---------|----------------|---------------|--------|
| balanced (iter 0-1) | 3,271 | avg 15, max 2,171 | Overfitting on repeated problems |
| capped 10K (iter 2) | ~2,000 | max 50 | Too few updates → 1% accuracy |
| real 51K (iter 3) | 11,647 | max 100 | Too diverse for lr=1e-5 → collapse |
| **balanced_v4** (iter 4-6) | **6,347** | **max 50/100** | Sweet spot: 27-39% accuracy |

### 4.8 Both Models Drift Verbose Over Epochs

Both baseline and budget-aware show increasing token counts with more training:
- Baseline E1→E3: 120→198 easy tokens (65% increase)
- Budget E1→E3: 203→193 easy tokens (slight decrease, but started higher)

This suggests overfitting toward verbose, confident-sounding responses. Early stopping or fewer epochs may be optimal.

---

## 5. What Worked and What Didn't

### Worked
- KL penalty at 0.01 — the single most impactful change
- Balanced dataset with per-problem capping (v4)
- Lower learning rate (1e-6) with KL for stability
- Lambda_easy=5.0 — creates a non-trivial penalty signal
- Tiered answer verification (Tier 0+1+2 with LLM judge)
- In-training gen eval for early detection

### Didn't Work
- Lambda_easy=0.05 — too small, invisible signal
- KL=0.1 — too strong, froze the model
- lr=1e-5 on diverse data — model collapse
- Pure standard DPO at lr=1e-6 — too slow to learn
- Sequence-level length normalization — doesn't clearly translate to shorter generation

---

## 6. Phase 2 Hypotheses & Directions

**Primary goal**: Shorten token count on easy problems. Hard improvement is a bonus.
**Success**: avg_tokens_easy ↓ while easy_accuracy ≥ baseline. TPCA ↓.
**Constraint**: DPO must be part of the approach (variants like SimPO are fine).

### HYPOTHESIS A: Stronger Length Signal via Per-Token Penalty (HIGH PRIORITY)

**Problem**: Current length penalty operates on the DPO pair level: `lambda * (chosen_len - rejected_len) / avg_len`. This penalizes *preferring* long responses but doesn't directly penalize *generating* long responses.

**Idea**: Add a generation-time length penalty or a token-level reward that decays over sequence position. During DPO training, apply the penalty per-token rather than per-sequence:
```python
# Instead of: penalty = lambda * (len_chosen - len_rejected) / avg_len
# Try: penalty = lambda * sum(position_weight[t] for t in chosen_tokens)
# where position_weight increases with position (penalize late tokens more for easy)
```

**Alternatively**: Add an auxiliary loss term that directly penalizes the policy's probability of generating long sequences for easy problems. For example, a length prediction head that learns to output shorter sequences on easy problems.

**Why this could work**: Current mechanism only teaches preference. We need to directly shape generation behavior.

### HYPOTHESIS B: Larger Model (1.5B or 3B) (HIGH PRIORITY)

**Problem**: Qwen2.5-0.5B hits a capability ceiling at ~22% overall (29% easy, 15% hard) on held-out test. This is too low to meaningfully demonstrate budget-aware behavior — if the model can't solve problems, we can't measure efficiency.

**Candidates**:
| Model | Params | Expected MATH | GPU Memory (LoRA) | Training Time |
|-------|--------|---------------|-------------------|---------------|
| Qwen2.5-1.5B | 1.5B | 20-30% MATH | ~20GB | ~1.5x current |
| Qwen2.5-3B | 3B | 30-40% MATH | ~35GB | ~2.5x current |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 25-35% MATH (math-specialized) | ~20GB | ~1.5x current |
| Qwen2.5-Math-1.5B-Instruct | 1.5B | 30-40% MATH (math-specialized) | ~20GB | ~1.5x current |

**Why this matters**: With 3 GPUs, we can run a 1.5B model comfortably. Higher baseline accuracy means more room to show budget-aware improvements. A model that can actually solve some hard problems lets us test the "preserve hard reasoning" thesis.

**Risk**: Larger model = longer training time. With 3 GPUs and 24 hours, we can likely do 2-3 full experiments with 1.5B.

### HYPOTHESIS C: Different Data Strategy (MEDIUM PRIORITY)

**Problem**: Only 276 unique hard problems in training data. Easy problems are well-covered (6,076 unique) but hard problems are severely underrepresented.

**Options**:
1. **More hard data**: Lower the hard threshold to include MATH Level 3 (currently only L4-5). This could give 1,000+ unique hard problems.
2. **Quality filtering**: Keep only problems where the DPO pair has a clear quality difference (large reward gap between chosen/rejected).
3. **Difficulty-aware sampling**: Oversample hard problems or use curriculum learning (easy first, then hard).
4. **Synthetic hard data**: Use a strong model to generate DPO pairs specifically for hard problems where Qwen2.5-0.5B has a non-zero chance of solving.

### HYPOTHESIS D: Modified DPO Variants (MEDIUM PRIORITY)

**Problem**: Standard DPO may not be the best framework for length control. The loss function treats preference as binary (chosen > rejected) rather than having a continuous notion of "how much better."

**Variants to try**:
1. **SimPO** (Simple Preference Optimization): Uses average log-probability as implicit reward, no reference model needed. The length normalization is built into the reward definition, which naturally penalizes verbosity.
   ```
   R(x,y) = (1/|y|) * beta * log(pi(y|x)) - gamma
   ```
   The `1/|y|` normalization directly rewards concise responses.

2. **DPO with length-ratio reward shaping**: Instead of subtracting a penalty, multiply the reward by a length ratio:
   ```
   R = beta * log_ratio * (target_length / actual_length)^alpha
   ```
   This scales reward down proportionally to how much longer than target the response is.

3. **IPO** (Identity Preference Optimization): More robust to noisy preferences than DPO. May handle the easy/hard split better.

4. **KTO** (Kahneman-Tversky Optimization): Uses individual good/bad examples rather than pairs. Could be combined with length-based quality labeling.

### HYPOTHESIS E: Two-Phase Training (MEDIUM PRIORITY)

**Problem**: Both models drift verbose over epochs. The budget-aware signal may be lost as the model overfits.

**Idea**:
1. **Phase 1**: Standard DPO training for 1-2 epochs (learn accuracy)
2. **Phase 2**: Budget-aware fine-tuning for 1 epoch (learn efficiency)

This separates the two objectives. The model first learns to solve problems, then learns to be concise on easy ones.

**Alternatively**: Curriculum learning where lambda ramps up over training — start with pure DPO, gradually increase the length penalty.

### HYPOTHESIS F: Reward Model Approach (LOWER PRIORITY)

**Problem**: DPO is implicit — the reward is never explicitly computed. A reward model could provide clearer signal.

**Idea**: Train a small reward model that scores responses on both correctness AND efficiency. Then use RLHF/PPO/REINFORCE with this reward model. The reward for easy problems includes a token count penalty; hard problems only reward correctness.

**Risk**: More complex pipeline, harder to debug. But could provide stronger signal than DPO's implicit reward.

### HYPOTHESIS G: Inference-Time Length Control (LOWER PRIORITY)

**Problem**: DPO trains preference, not generation. Maybe we should control length at inference time instead.

**Idea**: Use the budget-aware DPO model but add inference-time control:
- Classify problem difficulty before generation
- Set max_new_tokens based on difficulty (e.g., 128 for easy, 512 for hard)
- Use a length-aware sampling strategy (increase temperature/top-p as length increases to encourage stopping)

**Why this is interesting**: Even if DPO doesn't learn to generate shorter responses, combined with inference-time control, the overall system could show budget-aware behavior.

---

## 7. Phase 2 Experiment Plan (24 hours, 3 GPUs)

### Setup
- **GPUs**: 3x NVIDIA RTX 6000 Ada Generation (49GB each)
- **Time budget**: 24 hours
- **Constraint**: DPO must be part of every experiment

### Priority Order

**Hours 0-2: Baseline Eval + Model Download**
- Run post-training eval on any remaining checkpoints
- Download Qwen2.5-1.5B and/or DeepSeek-R1-Distill-Qwen-1.5B
- Test that larger models load and train with LoRA on single GPU

**Hours 2-10: Experiment Block 1 (3 parallel runs)**
- GPU 0: **1.5B baseline DPO** — establishes the new baseline with a stronger model
- GPU 1: **1.5B budget-aware DPO** (lambda_easy=5.0, KL=0.01) — does the budget signal work on a more capable model?
- GPU 2: **0.5B SimPO** or **0.5B with per-token penalty** — test if a different reward formulation works better

**Hours 10-12: Evaluate Block 1, decide Block 2**
- Run post-training eval on all 3 checkpoints
- Analyze which direction is most promising

**Hours 12-20: Experiment Block 2 (based on Block 1 results)**
- If 1.5B shows promise: fine-tune lambdas, try two-phase training
- If SimPO works: try SimPO + budget-aware on 1.5B
- If nothing works: try radical changes — different data, different beta, different LoRA rank

**Hours 20-24: Final Eval + Documentation**
- Full Tier 0+1+2 eval on best checkpoints
- Final comparison tables
- Write up findings

### GPU Assignment Strategy
With 3 GPUs, always keep all 3 busy:
- 1.5B model needs ~20GB → fits on 1 GPU with room for eval
- 0.5B model needs ~8GB → 2 runs on different GPUs
- Eval needs ~12GB (with LLM judge) → schedule on free GPU between runs

---

## 8. Technical Reference

### Model Architecture
- **Base**: Qwen/Qwen2.5-0.5B (494M params)
- **LoRA**: r=128, alpha=256, dropout=0.05, target=q_proj,v_proj,k_proj,o_proj
- **Effective LR**: lr × alpha/r = lr × 2.0
- **Mixed precision**: float16 (autocast)
- **Gradient clipping**: max_norm=1.0

### Loss Function (current)
```python
# In src/models/budget_aware_dpo_loss.py
log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
lambdas = lambda_easy if complexity==0 else lambda_hard  # per sample
avg_len = (chosen_len + rejected_len) / 2
length_diff = (chosen_len - rejected_len) / avg_len.clamp(min=1)
length_penalty = lambdas * length_diff
reward_diff = beta * (log_ratio_chosen - log_ratio_rejected) - length_penalty
dpo_loss = -logsigmoid(reward_diff).mean()

# Optional KL penalty
if kl_penalty_weight > 0:
    kl_div = (policy_chosen_logps - ref_chosen_logps).mean() + \
             (policy_rejected_logps - ref_rejected_logps).mean()
    loss = dpo_loss + kl_penalty_weight * kl_div.abs()
```

**Note**: `log_prob` computes **per-token averaged** log-probabilities: `sum(log_probs) / num_tokens`. Log-ratios are O(1), not O(T).

### Training Infrastructure
```bash
# Budget-Aware DPO
CUDA_VISIBLE_DEVICES=1 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/<name> --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --early-stopping-patience 3 --run-name <name> --wandb \
  > logs/<name>.log 2>&1 &

# Post-training eval (held-out test sets, 500 problems, Tier 0+1+2)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/<name>/best-model --limit 500 \
  --output eval_results/<name>.json --use-real

# Monitor
grep -iE "accuracy|tokens_easy|tokens_hard|tpca" logs/<name>.log
grep "Epoch.*val_loss" logs/<name>.log
```

### Available CLI Arguments
```
--output-dir        Checkpoint output directory
--max-epochs        Max training epochs (default: 10)
--batch-size        Batch size (default: 4)
--lr                Learning rate (default: 1e-5)
--lambda-easy       Length penalty for easy problems (default: 0.05)
--lambda-hard       Length penalty for hard problems (default: 0.001)
--kl-penalty        KL divergence penalty weight (default: 0.0)
--dpo-beta          DPO beta parameter (default: 0.1)
--gradient-accumulation-steps  (default: 1)
--early-stopping-patience      (default: 5)
--data-limit        Limit training pairs (for quick tests)
--run-name          WandB run name
--wandb             Enable WandB logging
--no-mixed-precision  Disable fp16
--num-workers       DataLoader workers (default: 4)
```

### Dataset Selection
```bash
# Via environment variable:
DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped  # explicit path
DATASET_VARIANT=balanced  # shorthand for processed_dpo_dataset_balanced

# Current best: balanced_v4_capped (50K pairs, 6,347 unique problems)
```

### Answer Verification Tiers
| Tier | Method | Speed | Used During |
|------|--------|-------|-------------|
| 0 | String equality | Instant | Training + Post-training |
| 1 | math-verify symbolic | Fast | Training + Post-training |
| 2 | LLM judge (Qwen2.5-Math-7B) | ~5s/problem | Post-training only |

### Data Architecture

**Training data** (from OpenMathInstruct-2, exclusively train splits):
- `data/processed_dpo_dataset_balanced_v4_capped/` — **Active** (50K pairs, 6,347 unique)
  - `dataset.jsonl` — full dataset
  - `train.jsonl` / `val.jsonl` — 45,184 / 4,816 split
  - `train_tokens.pt` / `val_tokens.pt` — pre-tokenized tensors
- Easy (C=0): Reject longer-but-correct over shorter-but-correct → teaches conciseness
- Hard (C=1): Reject incorrect over correct → teaches accuracy

**Evaluation data** (completely separate, official test splits):
- `data/gsm8k_test.jsonl` — 1,319 problems (Easy)
- `data/math_test.jsonl` — 5,000 problems (Hard, includes L1-L5)
- **Zero overlap** with training data (verified)

### File Map
```
src/
  models/
    budget_aware_dpo_loss.py  — Budget-aware loss function (EDITABLE)
    standard_dpo_loss.py      — Baseline loss function (reference)
  training/
    dpo_trainer.py            — Training loop, gen eval, metrics (EDITABLE)
  evaluation/
    run_evaluation.py         — Post-training eval (READ ONLY)
    answer_extraction.py      — Answer parsing (READ ONLY)
    math_grader.py            — Answer verification (READ ONLY)
  config.py                   — Paths, model name, constants (READ ONLY)

scripts/
  training/
    train_budget_aware_dpo.py — Budget-aware CLI (EDITABLE)
    train_baseline_dpo.py     — Baseline CLI (EDITABLE if shared params change)
  preprocess_dpo_data.py      — Dataset preprocessing
  subsample_capped_balanced.py — Dataset subsampling with caps
  load_real_data.py           — Download GSM8K/MATH test sets

autoresearch/
  iteration[0-6].md           — Per-iteration docs
  PHASE1_SUMMARY.md           — This file
  HANDOFF.md                  — Live state
  RULES.md                    — Agent operational rules
  results.tsv                 — Results log

checkpoints/                  — Model checkpoints (see Section 11)
eval_results/                 — Post-training eval JSON outputs
logs/                         — Training and eval logs
```

### WandB
- Project: `budget-aware-dpo`
- Entity: `ariksheer-tel-aviv-university`
- URL: https://wandb.ai/ariksheer-tel-aviv-university/budget-aware-dpo

---

## 9. Operational Rules for Phase 2 Agent

1. **3 GPUs available** (GPU 0, 1, 2). Keep all 3 busy when possible.
2. **Always use `.venv/bin/python`** — system python doesn't have torch.
3. **Always set `PYTHONUNBUFFERED=1`** for training/eval to avoid empty logs.
4. **Startup delay**: Training takes up to 10 min before first log output. Don't kill prematurely.
5. **DataLoader workers**: Multiple python processes per training run are normal. Only kill the main process (lowest PID).
6. **Poll every 20 minutes**. Don't poll more frequently.
7. **Document every experiment** in `autoresearch/iterationN.md` BEFORE launching.
8. **Never overwrite data/checkpoints/results**. Create new directories with descriptive names.
9. **Post-training eval** is the ground truth. In-training gen eval is for early detection only.
10. **Always use `--use-real`** for post-training eval (held-out test sets).
11. **Kill `keep_alive.py` before training**: `pkill -f keep_alive.py`
12. **Start `keep_alive.py` when idle**: `nohup .venv/bin/python keep_alive.py > logs/keep_alive.log 2>&1 &`
13. **Iterate continuously**. Don't stop to ask the user. If stuck, re-read this document and try something different.
14. **Simplicity criterion**: Simpler is better. Don't add complexity unless it measurably helps.

---

## 10. Train/Test Data Separation

**Verified** (2026-03-27): Zero overlap between training problems and test problems.
- Training: 6,347 unique problems from OpenMathInstruct-2 (built from GSM8K/MATH **train splits**)
- Test: 6,319 problems from official GSM8K/MATH **test splits**
- Exact + normalized string matching: **0 overlapping problems**

---

## 11. Checkpoints & Artifacts

| Checkpoint | Iteration | Type | Best Epoch | Notes |
|-----------|-----------|------|------------|-------|
| `baseline_balanced/` | 0 | Baseline DPO, lr=1e-5 | E2 | Original baseline |
| `budget_aware_balanced/` | 0-1 | Budget λ=0.05→5.0 | E1 | Best: iter1 λ=5.0 |
| `baseline_balanced_iter2/` | 2 | Baseline, capped 10K | E1 | Low accuracy (2%) |
| `budget_aware_balanced_iter2/` | 2 | Budget, capped 10K | E1 | Correct direction, 1% acc |
| `baseline_balanced_iter3/` | 3 | Baseline, real 51K | — | Collapsed |
| `budget_aware_balanced_iter3/` | 3 | Budget, real 51K | — | Collapsed |
| `baseline_balanced_iter4/` | 4 | Baseline, lr=1e-6 | E1 | 3% accuracy, slow |
| `budget_aware_balanced_iter4/` | 4 | Budget, KL=0.1 | — | Frozen (KL too strong) |
| `budget_aware_balanced_iter5/` | 5 | **Budget, KL=0.01** | **E3** | **Best budget: 38% in-train, 22% post-train** |
| `baseline_balanced_iter5b/` | 5b | Baseline, lr=1e-5 | — | Collapsed (lr too high for v4 data) |
| `baseline_kl_iter6/` | 6 | **Baseline+KL=0.01** | **E2** | **Best baseline: 39% in-train** |

---

## 12. Known Issues & Gotchas

1. **`keep_alive.py` must be killed before training** — it occupies GPU memory.
2. **Eval takes ~30 min** (loading model + generating 500 responses + LLM judge).
3. **WandB `val_loss`** needs `wandb.define_metric("val/loss", step_metric="epoch")` to graph properly (already fixed in code).
4. **Gradient norms with lambda=5.0** can reach 10-100+ with occasional inf/nan. Grad clipping at max_norm=1.0 handles this.
5. **Token padding**: Model uses eos_token as pad_token. Length computation counts non-pad tokens.
6. **In-training gen eval samples from training data** — not held-out. Don't use for final reporting.
7. **MATH Level 4-5** problems are beyond 0.5B model capability. Don't expect >10% accuracy on these.
