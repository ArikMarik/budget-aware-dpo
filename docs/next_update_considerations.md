# Considerations for Next Code Update

**Purpose:** Notes and research for future implementation. Do not change code until explicitly requested.

---

## 1. MATH Dataset Level Column

### What It Means

The MATH dataset (Hendrycks et al., NeurIPS 2021) uses a **5-level difficulty scale** aligned with the Art of Problem Solving (AoPS) competition format:

| Level | Meaning |
|-------|---------|
| **Level 1** | Easiest problems within a subject |
| **Level 2** | Easy–medium |
| **Level 3** | Medium |
| **Level 4** | Hard |
| **Level 5** | Hardest problems within a subject |

Levels are **relative to subject area**: Level 1 Algebra = easiest algebra; Level 5 Algebra = hardest algebra. The dataset has 12,500 problems from AMC 10, AMC 12, AIME, etc.

### Using It for Easy–Hard

**Current:** Complexity is inferred from `problem_source` (gsm8k vs math) and `teacher_token_count` (50/80 thresholds).

**Option:** Use MATH `level` when available:
- **Level 1–2** → Easy (C=0)
- **Level 3** → Could be medium or treated as Hard
- **Level 4–5** → Hard (C=1)

**Caveat:** `level` exists only for MATH problems. OpenMathInstruct-2 training data may not have `level` (it comes from GSM8K/MATH training sets or augmented problems). For MATH test set, `level` is present and is already used for `math_level_4_5_accuracy` in evaluation.

**Action:** Add `level` to `load_math_test()` output (already done). For preprocessing, if `level` is present in the raw data, use it for complexity; otherwise fall back to source + token thresholds.

---

## 2. Better Complexity Measurement (Token vs. Length)

### Current Approach

- `approx_tokens(text)` = `len(text.split())` (word count)
- Thresholds: EASY_TOKEN_THRESHOLD=50, HARD_TOKEN_THRESHOLD=80
- Goal: detect “long reasoning” vs. short answers

### Limitations

1. **Word count ≠ token count:** Different tokenizers give different counts; math symbols often use more tokens.
2. **Length ≠ reasoning quality:** Longer outputs can be redundant or “overthinking”; short outputs can be concise.
3. **Fixed thresholds:** 50/80 may not fit all models or datasets.

### Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Real tokenizer** (tiktoken, GPT-2 tokenizer, or model tokenizer) | Accurate token count; matches model behavior | Requires tokenizer load; model-specific |
| **Percentile-based thresholds** | Per-dataset adaptivity | Needs full dataset; may be unstable per batch |
| **CoT indicators** | Captures reasoning structure | Heuristic; regex/patterns for “First”, “Step”, “Therefore” |
| **Sentence count** | Simple proxy for structure | Ignores sentence length |
| **Deep-Thinking Ratio (DTR)** | Research-backed: “thinking” vs “rambling” | Needs access to internal model states; not applicable at preprocessing time |

### Recommendation

1. **Short term:** Replace `approx_tokens` with a real tokenizer (e.g. `tiktoken` for `cl100k_base` or the model’s tokenizer). Use `len(tokenizer.encode(text))` for `teacher_token_count` and thresholds.
2. **Threshold tuning:** Make EASY_TOKEN_THRESHOLD and HARD_TOKEN_THRESHOLD configurable (e.g. via config or env). Optionally compute per-dataset percentiles.
3. **Optional heuristic:** CoT indicators (e.g. “step”, “first”, “therefore”, “thus”) as a secondary signal for “has reasoning”; not a replacement for length.

---

## 3. Weights & Biases (W&B) Logging for Training

### Goal

Track training runs (loss, metrics, hyperparameters) in W&B for experiment management and comparison.

### Integration Points

- **Training scripts:** `train_baseline_dpo.py`, `train_budget_aware_dpo.py` → call `wandb.init()` and `wandb.log()`.
- **DPO trainer:** `src/training/dpo_trainer.py` → log loss, length_penalty, learning rate, etc. at each step or every N steps.

### Typical Setup

```python
import wandb
wandb.init(project="budget-aware-dpo", config=hyperparams)
wandb.log({"loss": loss, "length_penalty": lp}, step=global_step)
```

### Considerations

- **Config:** Pass `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_MODE` (offline/disabled) via env or config.
- **Offline mode:** Run with `WANDB_MODE=offline` when no network; sync later.
- **Dependencies:** Add `wandb` to requirements.

### Action

Add W&B logging to the training loop; log loss, length_penalty (for budget-aware), and any validation metrics. Keep it optional (e.g. via `--wandb` flag or `WANDB_PROJECT` env).
