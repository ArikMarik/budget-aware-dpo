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
