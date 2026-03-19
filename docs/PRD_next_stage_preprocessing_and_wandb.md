# PRD: Next Stage — Preprocessing Heuristics & W&B Logging

**Version:** 1.0  
**Date:** 2026-03-11  
**Status:** Draft  
**Scope:** `preprocessing.py`, `load_real_data.py`, training scripts, DPO trainer, W&B integration

---

## 1. Overview

This PRD defines the next implementation stage: **(1)** implement the new complexity heuristics from the conclusion report, with GSM8K explicitly protected from level-based logic, and **(2)** add Weights & Biases (W&B) logging to track training metrics, overfitting, accuracy, and token counts.

---

## 2. Goals

| Goal | Description |
|------|--------------|
| **G1** | GSM8K examples are always classified as Easy (C=0); never affected by level or token heuristics |
| **G2** | MATH examples use level when available; otherwise token-based fallback |
| **G3** | Replace word count with tiktoken `cl100k_base` for accurate token measurement |
| **G4** | Add fallback heuristics for edge cases (unknown source, missing level, ambiguous tokens) |
| **G5** | W&B logging for training: loss, overfitting indicators, accuracy (when eval runs), token counts |

---

## 3. Complexity Heuristics — Detailed Specification

### 3.1 Decision Flow (Canonical)

```
classify_complexity(example):
  1. SOURCE CHECK (GSM8K)
     If problem_source contains "gsm" or "gsm8k":
       → return 0 (Easy)  [IMMEDIATE; no further heuristics]

  2. SOURCE CHECK (MATH)
     If problem_source contains "math":
       a. LEVEL HEURISTIC (primary when available)
          If "level" in example and level in {"1","2","3","4","5"}:
            - Level 1 or 2 → return 0 (Easy)
            - Level 4 or 5 → return 1 (Hard)
            - Level 3 → go to step 2b (token fallback)
       b. TOKEN FALLBACK (level missing or Level 3)
          teacher_token_count = tiktoken_cl100k_base.encode(teacher_answer)
          If teacher_token_count < EASY_TOKEN_THRESHOLD (70): → return 0
          If teacher_token_count > HARD_TOKEN_THRESHOLD (130): → return 1
          Else: → return 0 (default Easy for ambiguous medium)

  3. UNKNOWN SOURCE (fallback)
     Use token heuristic only (step 2b).
     If still ambiguous: default to Easy (C=0).
```

### 3.2 GSM8K Invariant

**Constraint:** GSM8K examples must **never** be reclassified by level or token thresholds. They are always Easy. This preserves the existing behavior for GSM8K and avoids any regression from the new heuristics.

### 3.3 Token Count Implementation

| Component | Current | New |
|-----------|---------|-----|
| `approx_tokens(text)` | `len(text.split())` | `len(tiktoken.get_encoding("cl100k_base").encode(text))` |
| EASY_TOKEN_THRESHOLD | 50 (word) | 70 (tiktoken) |
| HARD_TOKEN_THRESHOLD | 80 (word) | 130 (tiktoken) |
| Configurability | Hardcoded | Config/env (EASY_TOKEN_THRESHOLD, HARD_TOKEN_THRESHOLD) |

**Scope:** Both `load_real_data.py` (for `teacher_token_count`) and `preprocessing.py` (for `classify_complexity`, `label_preference`, `compute_statistics`) must use tiktoken.

### 3.4 Additional Fallbacks (Edge Cases)

| Case | Fallback | Notes |
|------|----------|-------|
| MATH, level present but invalid (e.g. "?", empty) | Token heuristic | Treat as missing level |
| MATH, level = 3 | Token heuristic | Medium → use P25/P75-style thresholds |
| Unknown source (not gsm, not math) | Token heuristic only | Default Easy if ambiguous |
| Missing `teacher_token_count` | Compute on-the-fly from `generated_solution` | Use tiktoken |
| CoT indicators | Optional secondary (future) | Not required for this stage; document for later |

### 3.5 `label_preference` Updates

`label_preference` must use the same token thresholds (70/130) and tiktoken for consistency. Logic remains:

- Easy: preferred if tokens ≤ EASY_TOKEN_THRESHOLD; rejected if correct but too long
- Hard: preferred if tokens ≥ HARD_TOKEN_THRESHOLD; rejected if correct but too short

---

## 4. File Changes

### 4.1 `src/utils.py` (or new `src/tokenizer_utils.py`)

- Add `count_tokens_tiktoken(text: str) -> int` using tiktoken `cl100k_base`
- Keep `approx_tokens` for backward compatibility in non-preprocessing code, or deprecate and migrate all usages

### 4.2 `scripts/load_real_data.py`

- Replace `approx_tokens(solution)` with `count_tokens_tiktoken(solution)` in `convert_openmathinstruct`
- Ensure `teacher_token_count` is computed with tiktoken before writing to `real_openmathinstruct.jsonl`

### 4.3 `src/data/preprocessing.py`

| Change | Description |
|--------|--------------|
| Import tiktoken | Use `count_tokens_tiktoken` (or equivalent) for all token counts |
| Configurable thresholds | `EASY_TOKEN_THRESHOLD`, `HARD_TOKEN_THRESHOLD` from config/env; defaults 70, 130 |
| `classify_complexity` | Implement full decision flow (§3.1); GSM8K → 0 immediately |
| `label_preference` | Use tiktoken for token comparison; same thresholds |
| `compute_statistics` | Use tiktoken for `avg_preferred_tokens`, `avg_rejected_tokens` |
| On-the-fly token count | If `teacher_token_count` missing, compute from `generated_solution` |

### 4.4 `requirements.txt`

- Add `tiktoken` if not present

---

## 5. W&B Logging Specification

### 5.1 Integration Points

| Location | Responsibility |
|----------|-----------------|
| `scripts/training/train_baseline_dpo.py` | Add `--wandb` flag; pass to `train_dpo` |
| `scripts/training/train_budget_aware_dpo.py` | Add `--wandb` flag; pass to `train_dpo` |
| `src/training/dpo_trainer.py` | `wandb.init()` when enabled; `wandb.log()` at each eval step |

### 5.2 Configuration

| Env / Arg | Description | Default |
|-----------|-------------|---------|
| `WANDB_PROJECT` | W&B project name | `budget-aware-dpo` |
| `WANDB_RUN_NAME` | Run name (optional) | Auto or from config |
| `WANDB_MODE` | `online`, `offline`, `disabled` | `disabled` if `--wandb` not passed |
| `--wandb` | Enable W&B logging | Off by default |

### 5.3 Metrics to Log

| Metric | When | Description |
|--------|------|-------------|
| `train/loss` | Every `eval_every` steps | DPO loss |
| `train/length_penalty` | Every `eval_every` (budget-aware only) | Mean length penalty term |
| `train/learning_rate` | Every `eval_every` | Current LR (if tracked) |
| `train/avg_chosen_tokens` | Every `eval_every` | Mean token count of chosen responses in batch |
| `train/avg_rejected_tokens` | Every `eval_every` | Mean token count of rejected responses |
| `train/token_diff` | Every `eval_every` | Mean (chosen_tokens - rejected_tokens) |
| `config/*` | At init | Hyperparameters (max_steps, batch_size, lr, use_budget_aware, etc.) |

### 5.4 Overfitting & Validation (Optional Enhancement)

| Metric | When | Description |
|--------|------|-------------|
| `eval/accuracy` | Periodic (e.g. every N steps) | Accuracy on held-out eval set (if eval run implemented) |
| `eval/gsm8k_accuracy` | Same | GSM8K subset accuracy |
| `eval/math_accuracy` | Same | MATH subset accuracy |
| `eval/math_level_4_5_accuracy` | Same | MATH Level 4–5 accuracy |
| `eval/avg_tokens_easy` | Same | Avg tokens on Easy problems |
| `eval/avg_tokens_hard` | Same | Avg tokens on Hard problems |

**Note:** Current trainer does not run evaluation during training. This PRD defines the logging schema; implementing periodic eval runs is a separate task. When added, these metrics should be logged to W&B.

### 5.5 Implementation Sketch

```python
# In train_dpo()
use_wandb = os.environ.get("WANDB_PROJECT") or args.get("wandb", False)
if use_wandb:
    import wandb
    wandb.init(project=os.environ.get("WANDB_PROJECT", "budget-aware-dpo"), config=config)
    if os.environ.get("WANDB_MODE") == "offline":
        wandb.init(..., mode="offline")

# In training loop, at eval_every:
if use_wandb and step % eval_every == 0:
    log_dict = {"train/loss": loss.item(), "train/step": step}
    if extra: log_dict.update(extra)
    wandb.log(log_dict, step=step)
```

---

## 6. Acceptance Criteria

### 6.1 Preprocessing

- [ ] GSM8K examples always get C=0 regardless of token count or level
- [ ] MATH with level 1–2 → C=0; level 4–5 → C=1; level 3 → token fallback
- [ ] MATH without level → token fallback (70/130 thresholds)
- [ ] All token counts use tiktoken `cl100k_base`
- [ ] Thresholds configurable via env or config
- [ ] `load_real_data.py` produces `teacher_token_count` with tiktoken
- [ ] Reprocessing produces different complexity distribution for MATH (more granular); GSM8K distribution unchanged

### 6.2 W&B

- [ ] `--wandb` enables W&B; without it, no W&B calls
- [ ] `train/loss`, `train/avg_chosen_tokens`, `train/avg_rejected_tokens` logged every `eval_every`
- [ ] Budget-aware runs log `train/length_penalty`
- [ ] Config logged at init
- [ ] Offline mode supported via `WANDB_MODE=offline`

### 6.3 Regression

- [ ] Existing tests pass (if any)
- [ ] Dummy data pipeline still works
- [ ] Real data pipeline produces valid `dataset.jsonl` for training

---

## 7. Implementation Order

1. Add `tiktoken` to requirements; implement `count_tokens_tiktoken` in utils
2. Update `load_real_data.py` to use tiktoken for `teacher_token_count`
3. Refactor `preprocessing.py`: `classify_complexity`, `label_preference`, `compute_statistics`
4. Add `wandb` to requirements; integrate W&B in trainer and training scripts
5. Run preprocessing on real data; verify GSM8K all Easy, MATH distribution as expected
6. Run short training with `--wandb`; verify metrics in W&B dashboard
7. Generate feature report and commit

---

## 8. Dependencies

- `tiktoken` (add to requirements.txt)
- `wandb` (add to requirements.txt)

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|-------------|
| tiktoken adds latency to preprocessing | One-time encode per example; acceptable for batch preprocessing |
| W&B network issues on cluster | Use `WANDB_MODE=offline`; sync later |
| MATH level missing for some OpenMathInstruct items | Token fallback handles it |
| Threshold 70/130 may need tuning | Make configurable; document in metadata.json |

---

## 10. Appendix: Reference Tables

### Token Thresholds (from report_complexity_heuristics_conclusion.md)

| Heuristic | Value | Scope |
|-----------|-------|-------|
| EASY_TOKEN_THRESHOLD | 70 | MATH only (when level unavailable or Level 3) |
| HARD_TOKEN_THRESHOLD | 130 | MATH only (when level unavailable or Level 3) |
| MATH Level 1–2 | C=0 | MATH only |
| MATH Level 3 | Token fallback | MATH only |
| MATH Level 4–5 | C=1 | MATH only |
| GSM8K | C=0 always | Never use level or token |

### CoT Indicators (Future Optional)

Patterns: "First", "Step", "Therefore", "Thus", "Hence", "So we", "Let me think", "In summary", "Consequently". Use as secondary signal only; not in scope for this PRD.
