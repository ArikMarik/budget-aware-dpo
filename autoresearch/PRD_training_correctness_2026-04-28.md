# PRD — DPO Training Correctness Fixes

**Date**: 2026-04-28  
**Status**: Ready for implementation  
**Priority**: P0 — affects every training run

---

## Overview

Two confirmed correctness bugs in the DPO training pipeline, plus one architectural concern. The first two are P0 — they corrupt the gradient signal of every training run. The third is a structural issue to be specified separately.

---

## Issue 1 — Prompt Token Contamination in `log_prob`

### Problem

The `log_prob` function in `src/training/dpo_trainer.py` (line 271) computes the **mean log-probability over all non-padding tokens**, including prompt tokens. The training pairs stored in `tokens.pt` have the structure:

```
chosen_input_ids  = [prompt tokens] + [chosen response tokens]  + [pad tokens]
rejected_input_ids = [prompt tokens] + [rejected response tokens] + [pad tokens]
```

The `attention_mask` passed to `log_prob` is the full-sequence tokenizer mask (1 = real token, 0 = pad). No prompt-position zeroing is applied. So `log_prob` averages over prompt + response, not response only.

The prompt format is `"Question: {problem}\nAnswer: "` — typically 10–80 tokens depending on problem length, which is a significant fraction of the total sequence.

### Why this is wrong

Both chosen and rejected share the **same** prompt for a given problem. If the computation used **sum** log-probs, the identical prompt sum would cancel exactly in `log_ratio_chosen - log_ratio_rejected`. But the current code uses **mean** (sum divided by total non-pad token count). The chosen and rejected sequences have different total lengths (prompt length P is the same, but response lengths R_c and R_r differ), so the denominator differs and the cancellation is **imperfect**.

The contamination term in the DPO quality signal is:

```
Contamination = P · Δlp_prompt · (R_r - R_c) / [(P + R_c)(P + R_r)]

where:
  P          = prompt token count (same for both)
  R_c        = chosen response token count
  R_r        = rejected response token count
  Δlp_prompt = lp_prompt(θ) - lp_prompt(ref)  — how much policy has drifted from ref on prompt tokens
```

**Effect on easy pairs** (chosen is shorter, R_c < R_r → R_r - R_c > 0): as the policy improves on prompt tokens during training (`Δlp_prompt > 0`), the contamination term is positive, artificially inflating the chosen log-ratio. The model receives false credit from prompt fluency rather than response quality. The direction and magnitude of this bias changes throughout training as `Δlp_prompt` drifts, injecting noise into every gradient update.

### Evidence from literature

This is a documented production bug. TRL (HuggingFace) had regression issue #1746 where accidentally removing prompt masking prevented DPO from improving over the baseline at all. Every major DPO implementation — TRL, eric-mitchell/direct-preference-optimization (the reference repo), OpenRLHF — masks prompt tokens to zero before computing log-probs. The original Rafailov et al. 2023 paper defines `log π(y|x)` over response tokens `y` only.

### What counts as "prompt"

The full prompt prefix is everything the model conditions on before generating the answer:

```
"Question: {problem}\nAnswer: "
```

This includes the "Answer: " label — it is part of the prompt, not the response. The response tokens begin immediately after "Answer: ". The sequence structure in `tokens.pt` is:

```
[BOS | Question: {problem}\nAnswer:  | <actual answer tokens> | PAD PAD ...]
 ^0    ^1 ......................^P-1    ^P ..................^P+R-1
       |<-------- prompt -------->|    |<----- response ------->|
```

A single scalar `P` is sufficient because the prompt is a **contiguous prefix** from token 0. There is no interleaving of prompt and response tokens anywhere in the sequence.

### Fix — two-part change

#### Part A: Store `prompt_length` during preprocessing

**File**: `scripts/preprocess_dpo_data.py`, inside `tokenize_and_save`

Tokenization already happens here — this is the only place it ever occurs. No tokenization is added at training time. The change is: during the same preprocessing pass, also tokenize the prompt text alone to compute its token count, and store it as an additional tensor in `tokens.pt`.

```python
# Inside the per-batch loop, after building prompt_text:
prompt_text = build_zero_shot_prompt(pair["problem"])  # "Question: ...\nAnswer: "
prompt_tok = tokenizer(prompt_text, add_special_tokens=False)
prompt_length = len(prompt_tok["input_ids"])
```

**Why `add_special_tokens=False`**: The full sequence is tokenized with the default (`add_special_tokens=True`), which prepends a BOS token for Qwen. This shifts the first response token from position P to position P+1 in the full sequence. By computing `prompt_length` *without* BOS (using `add_special_tokens=False`), the count P coincidentally compensates for this shift in the masking step (see Part B). The net effect: `shift_mask[..., :P]` zeros exactly the right positions.

Accumulate per-pair into a tensor and add to `torch.save`:

```python
torch.save({
    "chosen_input_ids": ...,
    "chosen_attention_mask": ...,
    "rejected_input_ids": ...,
    "rejected_attention_mask": ...,
    "complexities": ...,
    "rejection_reason": ...,
    "chosen_length": ...,
    "rejected_length": ...,
    "problem_ids": ...,
    "prompt_lengths": prompt_lengths_tensor,   # NEW — shape (N,), dtype int64
}, tokens_path)
```

**`TokenizedDPODataset.__getitem__`** (in `src/training/dpo_trainer.py`): add `"prompt_length": self.data["prompt_lengths"][real_idx]` to the returned dict.

**`collate_fn_tokenized`** (same file): add `"prompt_length"` to the keys list so it stacks into a batched tensor.

#### Part B: Mask prompt positions in `log_prob`

**File**: `src/training/dpo_trainer.py`, line 271

Replace the current `log_prob` with a vectorized version that accepts a `prompt_lengths` batch tensor and returns **mean over response tokens only** (the denominator is now `R`, not `P + R`):

```python
def log_prob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: Optional[torch.Tensor] = None,  # shape (batch,), int64
) -> torch.Tensor:
    """Mean log-prob over response tokens only. Prompt positions are masked out."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().float()
    if prompt_lengths is not None:
        for i, pl in enumerate(prompt_lengths):
            # prompt_length computed with add_special_tokens=False; full sequence has BOS.
            # The BOS shifts the first response token by 1, so :pl zeros exactly the prompt.
            shift_mask[i, :pl] = 0.0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)
    # denominator is now R (response token count), not P+R
```

**In `_move_batch_to_device`**: add `prompt_lengths` to the extracted tensors (extend return to 6-tuple).

**In `_compute_batch_forward`**: accept `prompt_lengths`, pass to all 4 `log_prob` calls:

```python
policy_chosen_lp  = log_prob(policy_chosen,  chosen_ids,  chosen_mask,  prompt_lengths)
policy_rejected_lp = log_prob(policy_rejected, rejected_ids, rejected_mask, prompt_lengths)
ref_chosen_lp     = log_prob(ref_chosen,     chosen_ids,  chosen_mask,  prompt_lengths)
ref_rejected_lp   = log_prob(ref_rejected,   rejected_ids, rejected_mask, prompt_lengths)
```

**In `compute_batch_loss_train` and `compute_batch_loss_eval`**: unpack the 6-tuple and forward `prompt_lengths` to `_compute_batch_forward`.

#### Dataset rebuild required

After implementing Part A, the existing `tokens.pt` files must be regenerated. The new preprocessing run is a one-time cost. Old `.pt` files are incompatible with the updated `TokenizedDPODataset`.

---

## Issue 2 — Mean vs. Sum Log-Prob Aggregation (DEFERRED)

### Status: Deferred — mean is correct for Budget-Aware DPO; Issue 1 fix is sufficient

### Original framing (and why it was wrong)

The original framing was: after prompt masking, switch from mean to sum so that `λ_easy` is the sole length control and is not double-counted by mean's implicit length bias.

That argument looked at **absolute log-prob** values, where shorter sequences score higher under mean because they have fewer uncertain later tokens. This is true in isolation. But it is the wrong quantity to reason about. DPO does not use absolute log-probs — it uses **log-ratios** (policy vs. reference). At the log-ratio level the conclusion is the opposite.

### Why mean is length-neutral in DPO and sum is not

DPO computes:

```
reward_diff = (log π(y_c|x) − log π_ref(y_c|x)) − (log π(y_r|x) − log π_ref(y_r|x))
              |<------- chosen log-ratio ------->|   |<------ rejected log-ratio ----->|
```

For an easy pair: `y_c` is a short correct answer (`R_c` tokens), `y_r` is a longer answer (`R_r` tokens, `R_r > R_c`).

Suppose the policy has improved uniformly — every response token is predicted 0.5 log-prob units better than the reference. This is a good policy on both responses.

**With mean** (divide each log-ratio by its own response length):

```
reward_chosen   = 0.5  (avg over R_c tokens)
reward_rejected = 0.5  (avg over R_r tokens)
reward_diff     = 0.5 − 0.5 = 0.0
```

The quality term is **zero** — length-neutral. `λ_easy` alone drives the preference for the shorter answer. This is exactly what Budget-Aware DPO needs.

**With sum** (no normalisation):

```
reward_chosen   = 0.5 × R_c          (e.g.  1.0 for R_c=2)
reward_rejected = 0.5 × R_r          (e.g.  5.0 for R_r=10)
reward_diff     = R_c×0.5 − R_r×0.5 = 0.5×(R_c − R_r) < 0   (because R_c < R_r)
```

The quality term is **negative** — it actively penalises the short chosen answer just because it accumulated fewer tokens of log-ratio. `λ_easy` must now overcome this deficit before it can express any length preference at all. The larger `R_r − R_c`, the harder calibration becomes.

**Conclusion:** mean is structurally the right aggregation for Budget-Aware DPO. It makes the quality term length-neutral so that `λ_easy` is the clean, interpretable, sole length-control signal. Sum would fight the budget-aware objective.

### Why the denominator still mattered (Issue 1 connection)

The current broken `log_prob` divides by `(P + R)` — prompt + response length. After Issue 1's prompt masking, the denominator becomes `R` only. This matters because `P` is different from `P` — wait, `P` is the same for both chosen and rejected. But `R_c ≠ R_r`, so dividing by `P + R_c` vs `P + R_r` introduces a spurious difference even in the denominator. After Issue 1, both denominators are purely `R_c` and `R_r` respectively, which is the correct clean mean.

```
Before Issue 1:  mean_chosen   = Σlp / (P + R_c)   ← P inflates denominator differently for each
                 mean_rejected = Σlp / (P + R_r)
After Issue 1:   mean_chosen   = Σlp / R_c          ← clean per-response-token mean
                 mean_rejected = Σlp / R_r
```

### Reason to not switch: gradient instability history

During earlier training runs, sum log-probs produced very large gradient magnitudes that destabilised training. Mean was deliberately chosen as a stabilisation mechanism. Sum log-probs scale as `O(R)` while the length penalty term scales as `O(1)`, so switching would require re-tuning `β` (current `β=0.1` was calibrated against mean). Given that sum is also structurally wrong for the budget-aware objective (see above), there is no motivation to revisit this.

### Decision

**Keep mean permanently for Budget-Aware DPO.** The `log_prob` function returns:

```python
(token_log_probs * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)
```

After Issue 1's prompt masking, `shift_mask` is zero for prompt positions, so the denominator equals `R` (response token count). This is the correct, length-neutral quality signal for Budget-Aware DPO.

---

## Issue 3 — Training Pipeline Decoupling (HPO Static Context)

### Problem

In `scripts/optuna_hpo.py`, `train_dpo()` is called once per Optuna trial. Inside `train_dpo`, every call repeats work that is **identical across all trials**:

| Work | Cost | Varies per trial? |
|------|------|-------------------|
| `torch.load(tokens_path)` — deserialize full `tokens.pt` | ~10–30 s (disk + CPU) | **No** — same file every time |
| `create_tokenizer(model_name)` | ~2–5 s | **No** — same model_name |
| Load `problem_index_dict.json` | ~1 s | **No** — same file |
| `create_ref_model(model_name, device)` — load frozen base weights onto GPU | ~30–60 s | **No** — ref model is never modified |
| `_filter_by_length_ratio(data, length_ratio)` | <1 s (numpy) | **Yes** — `length_ratio` is a trial HP |
| `_cap_pairs_per_problem(...)` | ~1–5 s | **Yes** — `max_pairs_per_problem` is a trial HP |
| `split_pairs_by_problem(...)` | ~1–3 s | **Yes** (inputs change) |
| `_build_dataloaders(...)` | <1 s | **Yes** — `batch_size` is a trial HP |
| `build_val_problems(val_loader, problem_index)` | ~5–15 s | **Yes** — val split changes with above |
| `create_model(...)` — fresh policy model from disk | ~30–60 s | **Yes** — must start from base each trial |
| Training loop | dominant cost | Yes |

With N=20–50 trials, the top four fixed items alone waste **~1–2 minutes per trial** = **20–100 minutes** of pure overhead on a 20-50 trial sweep.

### Why `length_ratio` / `max_pairs_per_problem` being trial HPs does NOT require re-loading raw data

`torch.load(tokens_path)` returns a raw dict of tensors. The filtering and splitting that follow (`_filter_by_length_ratio`, `_cap_pairs_per_problem`, `split_pairs_by_problem`) operate entirely in numpy/Python on top of those tensors — they do not mutate the raw dict, they only produce index lists. So the raw tensor dict can be loaded once and reused. Each trial just re-runs the fast index-selection logic.

### Why `ref_model` is safe to share across trials

The reference model is loaded with `model.eval()` and all `requires_grad=False`. Training only updates the **policy** model — the ref model is read-only in every forward pass. Its weights never change between trials. Keeping it on GPU across trials does not increase peak VRAM (the training loop already requires both models simultaneously), and `_cleanup_gpu()` between trials should skip it.

### Fix — Introduce `StaticTrainingContext`

#### Step 1: Define `StaticTrainingContext` dataclass

**File**: `src/training/dpo_trainer.py`

```python
@dataclass
class StaticTrainingContext:
    raw_data: dict                   # torch.load(tokens_path) output — never mutated
    tokenizer: PreTrainedTokenizer
    problem_index: dict              # {problem_id (int) -> {problem, expected_answer, complexity, ...}}
    ref_model: nn.Module             # frozen base model, shared across all trials
```

#### Step 2: Extract `build_static_context()`

**File**: `src/training/dpo_trainer.py`

New public function:

```python
def build_static_context(
    tokens_path: Path,
    model_name: str,
    device: str,
    problem_index_path: Path,
) -> StaticTrainingContext:
    """Load all trial-invariant state once. Pass the result to every train_dpo call."""
    raw_data = torch.load(tokens_path)
    tokenizer = create_tokenizer(model_name)
    if problem_index_path.exists():
        with open(problem_index_path) as f:
            problem_index = json.load(
                f, object_hook=lambda obj: {int(k) if k.isdigit() else k: v for k, v in obj.items()}
            )
    else:
        problem_index = {}
    ref_model = create_ref_model(model_name, device)
    return StaticTrainingContext(
        raw_data=raw_data,
        tokenizer=tokenizer,
        problem_index=problem_index,
        ref_model=ref_model,
    )
```

#### Step 3: Refactor `load_tokenized_datasets` to accept raw data directly

**File**: `src/training/dpo_trainer.py`

Add a `raw_data` parameter (takes priority over `tokens_path` when provided):

```python
def load_tokenized_datasets(
    tokens_path: Path,
    *,
    raw_data: Optional[dict] = None,   # pass pre-loaded dict to skip torch.load
    length_ratio: float = 1.0,
    ...
) -> tuple[TokenizedDPODataset, TokenizedDPODataset]:
    data = raw_data if raw_data is not None else torch.load(tokens_path)
    ...  # rest unchanged
```

#### Step 4: Add `ctx` parameter to `train_dpo`

**File**: `src/training/dpo_trainer.py`

```python
def train_dpo(
    *,
    ...
    ctx: Optional[StaticTrainingContext] = None,   # NEW — pass to skip static setup
) -> dict:
    effective_model_name = model_name or MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ctx is None:
        # Standalone usage: build context inline (backward-compatible)
        ctx = build_static_context(
            get_tokens_path(), effective_model_name, device, problem_index_path
        )

    tokenizer = ctx.tokenizer
    problem_index = ctx.problem_index
    ref_model = ctx.ref_model

    train_dataset, val_dataset = load_tokenized_datasets(
        get_tokens_path(),
        raw_data=ctx.raw_data,   # skip torch.load
        length_ratio=length_ratio,
        ...
    )
    ...
    # remove: tokenizer = create_tokenizer(...)
    # remove: problem_index = json.load(...)
    # remove: ref_model = create_ref_model(...)
```

#### Step 5: Refactor `optuna_hpo.py`

**File**: `scripts/optuna_hpo.py`

In `main()`, before calling `study.optimize()`, build context once:

```python
from src.training.dpo_trainer import build_static_context, get_tokens_path

effective_model = args.model or MODEL_NAME
device = "cuda" if torch.cuda.is_available() else "cpu"
ctx = build_static_context(
    get_tokens_path(),
    effective_model,
    device,
    DATA_PATH / "problem_index_dict.json",
)
```

Pass `ctx` into `_build_objective_fn`:

```python
objective_fn = _build_objective_fn(search, use_grid=(args.sampler == "grid"), ctx=ctx)
```

In `_build_objective_fn`, add `ctx` to the factory signature and forward it to `train_dpo`:

```python
def _build_objective_fn(search: SearchConfig, use_grid: bool, ctx: StaticTrainingContext):
    def objective(trial: optuna.Trial) -> float:
        ...
        result = train_dpo(
            ...
            ctx=ctx,
        )
```

#### `_cleanup_gpu` between trials

The existing `_cleanup_gpu()` call after each trial must NOT call `del ref_model` or move the shared ref model off GPU. Only the per-trial policy model and its optimizer state need to be freed. The current implementation (`gc.collect()` + `torch.cuda.empty_cache()`) is fine — it only frees unreferenced tensors, and since `ctx.ref_model` is still referenced, it stays on GPU.

#### Backward compatibility

When `ctx=None` (default), `train_dpo` builds context inline. All existing callers (standalone training scripts, `train_dpo.py`, tests) are unaffected.

### Cost reduction estimate

| Item | Per-trial cost | Eliminated? |
|------|---------------|-------------|
| `torch.load(tokens_path)` | ~20 s | Yes |
| `create_tokenizer` | ~3 s | Yes |
| `problem_index` JSON load | ~1 s | Yes |
| `create_ref_model` | ~45 s | Yes |
| **Total per trial** | **~70 s** | |
| **Over 20 trials** | **~23 min** | **Saved** |

---

## Implementation Order

1. **Issue 1 (prompt masking)** — implement preprocessing change first, then `log_prob` fix (keep mean), then dataset rebuild
2. **Issue 3 (pipeline decoupling)** — implement after Issue 1 is validated; requires no dataset rebuild
3. **Issue 2 (sum vs. mean)** — deferred; revisit only if residual length bias is empirically confirmed to hurt `λ_easy` calibration after Issue 1 is stable

---

## Files Modified

| File | Change |
|------|--------|
| `scripts/preprocess_dpo_data.py` | Store `prompt_lengths` tensor in `tokens.pt` |
| `src/training/dpo_trainer.py` | `log_prob`: add prompt masking + switch to sum; `_compute_batch_forward`: pass `prompt_lengths`; add `StaticTrainingContext`, `build_static_context()`, `ctx` param on `train_dpo`, `raw_data` param on `load_tokenized_datasets` |
| `src/data/preprocessing.py` | `TokenizedDPODataset.__getitem__`: expose `prompt_length`; collate function: batch it |
| `scripts/optuna_hpo.py` | Call `build_static_context()` once before `study.optimize()`; pass `ctx` through to each trial |
| All `tokens.pt` files | Must be regenerated after preprocessing change |

---

## Implementation Log — 2026-04-28

### Changes Implemented

**`scripts/preprocess_dpo_data.py`**
- Added `prompt_lengths = torch.empty(num_pairs, dtype=torch.long)` tensor alongside existing tensors
- Inside per-pair loop: tokenizes `prompt_text` with `add_special_tokens=False`, collects lengths into `prompt_length_batch`
- Saves `prompt_lengths` in the `torch.save` dict

**`src/training/dpo_trainer.py`**
- Added `StaticTrainingContext` dataclass (fields: `raw_data`, `tokenizer`, `problem_index`, `ref_model`)
- Updated `log_prob(logits, input_ids, attention_mask, prompt_lengths=None)` — zeros `shift_mask[i, :pl]` for each sample before computing mean; denominator is now response-only token count `R`
- Updated `TokenizedDPODataset.__getitem__` to include `"prompt_length": self.data["prompt_lengths"][real_idx]`
- Updated `collate_fn_tokenized` keys list to include `"prompt_length"`
- Updated `load_tokenized_datasets` signature: added `raw_data: Optional[dict] = None` as keyword-only arg (skips `torch.load` when provided); existence check conditioned on `raw_data is None`
- Updated `_move_batch_to_device` to extract and return `prompt_lengths` (now 6-tuple)
- Updated `_compute_batch_forward` to accept `prompt_lengths` and pass to all 4 `log_prob` calls
- Updated `compute_batch_loss_train` and `compute_batch_loss_eval` to unpack 6-tuple from `_move_batch_to_device`
- Added `build_static_context(tokens_path, model_name, device, problem_index_path) -> StaticTrainingContext` after `create_ref_model`
- Updated `train_dpo`: added `ctx: Optional[StaticTrainingContext] = None`; builds context inline when `None` (backward compatible); removed redundant `create_ref_model` call; uses `ctx.raw_data` for dataset loading

**`scripts/optuna_hpo.py`**
- Added imports: `DATA_PATH`, `MODEL_NAME`, `build_static_context`, `StaticTrainingContext`, `get_tokens_path`
- Updated `_build_objective_fn` signature to accept `ctx: StaticTrainingContext`; forwards `ctx=ctx` to `train_dpo`
- Updated `main()`: builds `ctx` once before `study.optimize()`, passes it to `_build_objective_fn`

### Deviations from PRD
- `TokenizedDPODataset` and `collate_fn_tokenized` are in `src/training/dpo_trainer.py` (not `src/data/preprocessing.py` as listed in the PRD's Files Modified table) — changes applied to the correct location
- Issue 2 (sum vs. mean) confirmed deferred; mean kept as-is

### Pending
- Regenerate all `tokens.pt` files: `python -m scripts.preprocess_dpo_data --force`
