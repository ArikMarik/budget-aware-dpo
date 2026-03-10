# Preprocessing Analysis & Specification: Budget-Aware DPO

## 1. The 4-Way Augmentation Matrix (Clarification)

Per the Budget-Aware DPO proposal and implementation plan:

| Complexity | Correctness | Length | Label | Rationale |
|------------|-------------|--------|-------|-----------|
| Easy (C=0) | Correct | Short | **Preferred** | Direct, concise—ideal for simple arithmetic |
| Easy (C=0) | Correct | Long/Verbose | **Rejected** | Over-reasoning on trivial tasks |
| Easy (C=0) | Incorrect | Any | **Rejected** | Wrong answer, regardless of length |
| Hard (C=1) | Correct | Long/CoT | **Preferred** | Detailed reasoning needed for proofs |
| Hard (C=1) | Correct | Short/Oversimplified | **Rejected** | Insufficient reasoning for complex problems |
| Hard (C=1) | Incorrect | Any | **Rejected** | Wrong answer, regardless of length |

**Do we need all 4 permutations per Q-A?** We need **preferred vs rejected pairs**, but we must avoid **imbalance**.

**Imbalance risk:** Without an explicit mechanism, we could accidentally have a dataset where (mostly) preferred = correct and rejected = incorrect. That reduces to **regular DPO** — we lose the innovation of length-aware preference. The model would overfit to "correct vs incorrect" and not learn the length-routing behavior.

**Required coverage:** We need enough pairs from **all three rejection reasons** so the model learns both correctness and length preferences:

| Case | Preferred | Rejected | Teaches |
|------|-----------|----------|---------|
| 1. Correct + right length | (correct, short) Easy / (correct, long) Hard | — | Baseline preferred |
| 2. Incorrect answer | (correct, X) | (incorrect, any) | Correctness matters |
| 3. Correct but wrong length | (correct, short) Easy / (correct, long) Hard | (correct, long) Easy / (correct, short) Hard | Length matters by complexity |

**Mechanism:** Synthesize pairs so we have balanced coverage of (2) and (3). If we only have natural pairs, we may get mostly (2). We must explicitly create (3) pairs — e.g. correct-long vs correct-short for Easy (short preferred), correct-long vs correct-short for Hard (long preferred) — to ensure the model learns length-routing, not just correctness.

---

## 2. Build DPO Pairs: Issues and Fixes

### 2.1 Critical Bug: `filter()` Returns Iterator

```python
preferred = filter(...)  # iterator
rejected = filter(...)   # iterator
if preferred and rejected:  # BUG: iterators are always truthy!
```

**Fix:** Convert to list: `preferred = list(filter(...))`, `rejected = list(filter(...))`.

### 2.2 `_make_short_answer` Assumes Correct Input

Current behavior: Takes a solution, extracts answer, returns `"The answer is X."`. If the input solution is **incorrect**, this produces **incorrect-short**, which must be **rejected**, not used as preferred.

**Correct usage:**
- `_make_short_answer` should only be called when the input solution is **correct** (verified via expected_answer).
- When used for Hard synthetic: (correct-long preferred, correct-short rejected) — the short is correct but oversimplified.
- Never use `_make_short_answer` on incorrect solutions to create a "preferred" — that would teach the model to prefer wrong answers.

### 2.3 Synthetic Pair Logic (Preferred-Only vs Rejected-Only)

| Case | Preferred | Rejected | Action |
|------|-----------|----------|--------|
| Both exist | ✓ | ✓ | Use natural pairs |
| Preferred only (Easy) | short | — | Synthesize verbose rejected (expand) |
| Preferred only (Hard) | long | — | Synthesize short rejected (`_make_short_answer`) |
| Rejected only | — | wrong | **Skip** — no correct solution to prefer |

### 2.4 Rejected-Only: Synthesize Minimal Correct

When we have only incorrect solutions for a problem, we need a correct preferred. If `expected_answer` exists, synthesize `"The answer is {expected}."` as preferred. For Easy this is ideal (short correct). For Hard, it teaches "correct > wrong" — valid.

### 2.5 Short Solution Given, Correct Is Long: Synthesize Reasoning? (Stub Only)

If we have a short correct solution (e.g. "The answer is 8.") and for Hard we need correct-long as preferred — we don't have it. **Synthesizing long reasoning** from a short answer would require an LLM or template.

**Decision:** Create a **stub function** `_make_long_reasoning(short_solution, expected, problem)` for future use. **For now, skip** — do not actually synthesize long answers. Skip Hard problems when we only have short correct solutions.

---

## 3. Real vs Synthesized Pairs: Separate and Combine

**Output structure:**
- Save **real_pairs** and **synthesized_pairs** as separate JSONL files (e.g. `dataset_real.jsonl`, `dataset_synthesized.jsonl`).
- Combine them to produce the **final dataset** (`dataset.jsonl`) used for training — same as current behavior.
- The separated files are **for analysis only** — to inspect composition and make decisions.

**Usage policy:** If there are enough real pairs (to be defined per experiment), we may choose to **not use synthesized pairs** at all. The separation enables this decision without reprocessing.

---

## 4. Output Statistics (Full Visibility)

After preprocessing, output **all** of the following so we have full visibility for decisions:

| Statistic | Description |
|-----------|-------------|
| `total_real_pairs` | Count of pairs from natural data (both preferred and rejected exist) |
| `total_synthesized_pairs` | Count of pairs created by synthesis |
| `total_pairs` | `real + synthesized` (final training set) |
| `real_pairs_pct` | `100 * real / total` |
| `synthesized_pairs_pct` | `100 * synthesized / total` |
| `rejected_by_correctness` | Count of pairs where rejected is incorrect (wrong answer) |
| `rejected_by_length` | Count of pairs where rejected is correct but wrong length (too long for Easy, too short for Hard) |

*Implementation note:* Each pair must be tagged with `rejection_reason: "correctness" | "length"` when built, so we can compute these counts.
| `rejected_by_correctness_pct` | `100 * rejected_by_correctness / total` |
| `rejected_by_length_pct` | `100 * rejected_by_length / total` |
| `easy_pairs`, `hard_pairs` | By complexity |
| `easy_pairs_pct`, `hard_pairs_pct` | Percentages by complexity |
| `avg_preferred_tokens`, `avg_rejected_tokens` | Token statistics |
| `skipped_groups` | Groups that could not form valid pairs |
| `rejected_by_correctness_real`, `rejected_by_correctness_synthesized` | Breakdown by source |
| `rejected_by_length_real`, `rejected_by_length_synthesized` | Breakdown by source |

Save these to `metadata.json` (or equivalent) alongside the dataset. The goal is to have **all the data in front of us** after preprocessing to decide on balancing, whether to use synthesized pairs, and next steps.

---

## 5. Expected Answer in Dataset

**OpenMathInstruct-2** provides: `problem`, `generated_solution`, `expected_answer`, `problem_source`. It does **not** provide `correctness_flag`.

**Current load_real_data.py:** Sets `correctness_flag: True` for all — **incorrect**. We must verify correctness by comparing `extract_answer(generated_solution)` with `expected_answer`.

**Action:** Add `verify_correctness(solution, expected_answer)` in preprocessing or load_real_data. Set `correctness_flag` based on that.

---

## 6. Skip Invalid Samples

Skip when:
- No `expected_answer` and we need it (e.g. for correctness check, or for synthesizing preferred from rejected-only)
- Cannot form any valid pair (e.g. Hard, preferred-only, but solution is short — we need long preferred, can't synthesize)
- Empty or malformed fields

---

## 7. Budget-Aware DPO Loss: Calculations and Assumptions

### Formula

$$R_{budget}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - \lambda(C) \cdot |y|$$

DPO loss: maximize probability that chosen is preferred over rejected. Implicit reward difference:

$$\Delta R = R_{chosen} - R_{rejected} = \beta (\log r_c - \log r_r) - \lambda (|y_c| - |y_r|)$$

where \(r = \pi_\theta/\pi_{ref}\).

Loss: \(-\log \sigma(\Delta R)\).

### Assumptions

1. **Chosen is preferred, rejected is worse.** The loss assumes the training pairs are correctly labeled. If we swap them (preferred as rejected, rejected as chosen), the model learns the wrong preference.

2. **No explicit correctness in the loss.** Correctness is encoded in the *pair construction*: we ensure chosen is correct (or better) and rejected is incorrect (or worse). The loss only sees (chosen, rejected) and pushes the model to assign higher reward to chosen.

3. **Length penalty is complexity-dependent.** For Easy (C=0): λ high → short chosen gets a boost (length_diff negative when chosen shorter). For Hard (C=1): λ ≈ 0 → length doesn't matter, correctness (via log-ratio) dominates.

4. **Token count |y|.** Uses actual token lengths of chosen/rejected. Shorter chosen for Easy reduces the penalty term, increasing ΔR.

---

## 8. Summary of Code Changes (To Implement)

1. **load_real_data.py:** Compute `correctness_flag` by comparing `extract_answer(generated_solution)` with `expected_answer`. Skip or flag when `expected_answer` is missing.

2. **preprocessing.py:**
   - Fix `filter` → `list(filter)`.
   - Add `verify_correctness()` using answer_extraction.
   - `_make_short_answer`: only call when solution is correct; add guard.
   - Add **stub** `_make_long_reasoning(short_solution, expected, problem)` — do not implement; skip Hard when only short correct available.
   - Synthetic logic: preferred-only Easy → synthesize verbose rejected; preferred-only Hard → synthesize short rejected.
   - Rejected-only: if expected_answer exists, synthesize minimal correct as preferred.
   - Skip samples that cannot form valid pairs.
   - **Balance mechanism:** Explicitly create (3) pairs (correct but wrong length) to ensure coverage; avoid overfitting to (2) only (correct vs incorrect).

3. **Output:**
   - Save `dataset_real.jsonl` and `dataset_synthesized.jsonl` separately (for analysis).
   - Combine into `dataset.jsonl` for training.
   - Output full statistics (Section 4) to `metadata.json`.

4. **Balanced set:** Use statistics to decide; consider oversampling Easy or stratified sampling if needed.
