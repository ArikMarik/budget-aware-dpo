# Phase 7: Real Data Preprocessing — Report

**Date:** 2026-03-02  
**Status:** Complete  
**Data:** Real data (OpenMathInstruct-2)

## Implementation Summary

- **load_real_data.py:** Loads OpenMathInstruct-2 from HuggingFace (train_1M split), converts to JSONL with teacher_token_count, correctness_flag
- **Config:** REAL_DATASET_PATH, PROCESSED_DATASET_PATH_REAL, GSM8K_TEST_PATH, MATH_TEST_PATH
- **Path routing:** get_input_path() / get_output_path() return real paths when USE_DUMMY_DATA=0
- **Synthetic pairs:** When problems have only one solution, create preferred/rejected by short-answer extraction (Easy: short preferred; Hard: CoT preferred)
- **Trainer/eval:** get_processed_dataset_path() routes to real processed data when not dummy

## Test Results

- Loaded 5000 examples from OpenMathInstruct-2 train_1M
- Preprocessing: 4994 DPO pairs (742 easy, 4252 hard)
- Avg preferred: 190 tokens; avg rejected: 20 tokens
- Output: `data/processed_dpo_dataset_real/`

## Usage

```bash
# Load real data (use --limit for quick test)
python scripts/load_real_data.py --split train_1M --limit 5000

# Preprocess (with USE_DUMMY_DATA=0)
USE_DUMMY_DATA=0 python scripts/preprocess_dpo_data.py
```

## Next Steps

- Phase 8: Full training on real data

---

## Appendix: Problems Encountered and Solutions

### Problem 1: Empty DPO pairs from preprocessing

**Issue:** Running preprocessing on 500 OpenMathInstruct-2 examples produced 0 DPO pairs. The pipeline requires each (problem, complexity) group to have both preferred and rejected solutions.

**Cause:** OpenMathInstruct-2 train_1M has mostly unique problems (one solution per problem). With 500 examples, only 6 problems had multiple solutions, and none had both short and long variants needed for natural preferred/rejected pairs.

**Solution:** Added synthetic pair creation in `build_dpo_pairs()`. When a group has only one solution, create a pair by extracting the short answer (via `extract_answer()`) and pairing it with the full solution: Easy → (short preferred, verbose rejected); Hard → (CoT preferred, oversimplified rejected).

### Problem 2: `trust_remote_code` deprecation

**Issue:** `load_dataset(..., trust_remote_code=True)` triggered a deprecation warning for OpenMathInstruct-2.

**Solution:** Removed `trust_remote_code=True` from the `load_dataset` call.

### Problem 3: MATH test dataset loading

**Issue:** `hendrycks/competition_math` may be unavailable (DMCA). Alternative dataset structures differ.

**Solution:** Added fallback chain: try `EleutherAI/hendrycks_math`, then `hendrycks/competition_math`, then `lighteval/MATH`. Handle varying field names (problem/question, solution/answer).

---

## Note: Dataset imbalance

The processed real dataset is heavily skewed toward Hard problems: 742 easy pairs vs 4252 hard pairs (~85% hard). This may affect training and evaluation. A TODO has been added to the implementation plan (User notes and todos) to investigate and address this imbalance in the future.
