# Report: Preprocessing Implementation (Budget-Aware DPO Spec)

**Date:** 2026-03-10

## Summary

Implemented the preprocessing plan from `docs/preprocessing_analysis_and_spec.md`: correctness verification, real/synthesized separation, full statistics, and balanced pair construction.

## Implementation

### 1. load_real_data.py

- **Already had** `verify_correctness()` and uses it in `convert_openmathinstruct()` to set `correctness_flag` from `expected_answer` (no change needed).

### 2. preprocessing.py

- **`_verify_correctness()`**: Uses `expected_answer` when `correctness_flag` is missing.
- **`label_preference()`**: Uses `_verify_correctness()` instead of `correctness_flag` default.
- **`_make_short_answer()`**: Clarified docstring; only used when solution is correct or expected is provided.
- **`_make_verbose_answer()`**: New helper for Easy synthetic rejected (verbose from short).
- **`_make_long_reasoning()`**: Stub for future long-reasoning synthesis; returns empty string.
- **`_rejection_reason()`**: Returns `"correctness"` or `"length"` for each pair.
- **`build_dpo_pairs()`**:
  - Returns `(real_pairs, synthesized_pairs, skipped_groups)`.
  - Each pair: `problem`, `chosen`, `rejected`, `complexity`, `rejection_reason`.
  - Natural pairs: preferred + rejected from data.
  - Synthetic preferred-only: Easy → verbose rejected; Hard → short rejected.
  - Synthetic rejected-only: minimal correct preferred from `expected_answer`.
  - Fixed `filter` bug (use list instead of iterator).
- **`compute_statistics()`**: Accepts `(real_pairs, synthesized_pairs, skipped_groups)` and returns full stats per spec.

### 3. preprocess_dpo_data.py

- **Output files**:
  - `dataset_real.jsonl` — real pairs.
  - `dataset_synthesized.jsonl` — synthesized pairs.
  - `dataset.jsonl` — combined (real + synthesized) for training.
  - `metadata.json` — full statistics.
- **Skip condition**: Skips only if all four files exist.

### 4. Statistics (metadata.json)

- `total_real_pairs`, `total_synthesized_pairs`, `total_pairs`
- `real_pairs_pct`, `synthesized_pairs_pct`
- `rejected_by_correctness`, `rejected_by_length` and percentages
- `rejected_by_correctness_real`, `rejected_by_correctness_synthesized`
- `rejected_by_length_real`, `rejected_by_length_synthesized`
- `easy_pairs`, `hard_pairs` and percentages
- `avg_preferred_tokens`, `avg_rejected_tokens`
- `skipped_groups`

## Test Results (Dummy Data)

```
Loaded 50 examples
Real: 18, Synthesized: 14, Skipped: 0
total_pairs: 32
rejected_by_correctness: 5 (15.62%), rejected_by_length: 27 (84.38%)
easy_pairs: 27 (84.38%), hard_pairs: 5 (15.62%)
```

Training pipeline loads `dataset.jsonl` successfully; extra keys ignored.

## Appendix: Problems Encountered and Solutions

- **Import time**: First run can be slow due to torch/transformers. No functional impact.
- **Real dataset**: Uses old format without `rejection_reason`. Reprocess with `USE_DUMMY_DATA=0` to regenerate full stats.
