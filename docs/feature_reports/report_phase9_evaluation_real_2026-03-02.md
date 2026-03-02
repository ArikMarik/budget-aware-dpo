# Phase 9: Evaluation on Real Data — Report

**Date:** 2026-03-02  
**Status:** Complete  
**Data:** GSM8K test (1319), MATH test (5000)

## Implementation Summary

- **load_eval_problems_real():** Loads from GSM8K and MATH test JSONL files; complexity: gsm8k→Easy (0), math→Hard (1)
- **MATH dataset fix:** EleutherAI/hendrycks_math requires per-subject configs; concatenate all 7 subjects for full test set
- **MATH Level 4–5:** Added `math_level_4_5_accuracy` metric when level field present
- **run_evaluation.py:** Uses `USE_DUMMY_DATA` and `get_baseline_output_dir()` / `get_budget_aware_output_dir()` for checkpoint routing
- **load_real_data.py:** Added `--test-sets-only` for Phase 9 (load GSM8K+MATH without OpenMathInstruct)

## Test Results (50 problems, GSM8K only — quick verification)

| Model           | Accuracy | TPCA  | Avg Easy | Avg Hard |
|-----------------|----------|-------|----------|----------|
| Baseline DPO    | 26.00%   | 768.3 | 199.8    | 0        |
| Budget-aware DPO| 28.00%   | 696.9 | 195.1    | 0        |

*Note: --limit 50 loads first 50 from GSM8K (all Easy). Full evaluation on 6319 problems (GSM8K+MATH) requires running without --limit.*

## Usage

```bash
# 1. Load test sets (if not already done)
python scripts/load_real_data.py --test-sets-only

# 2. Run evaluation on real checkpoints
USE_DUMMY_DATA=0 python scripts/run_evaluation.py

# Quick test (50 problems)
USE_DUMMY_DATA=0 python scripts/run_evaluation.py --limit 50
```

## Output files

- `checkpoints/evaluation_results_real.json` — aggregated metrics
- `checkpoints/baseline_eval_real.json` — baseline per-problem results
- `checkpoints/budget_aware_eval_real.json` — budget-aware per-problem results

## Next Steps

- Phase 10: Visualization from real data (histograms, results table)
- Run full evaluation (no --limit) for publication-ready metrics

---

## Appendix: Problems Encountered and Solutions

### Problem 1: MATH dataset loading

**Issue:** `lighteval/MATH` and `hendrycks/competition_math` failed or unavailable. `EleutherAI/hendrycks_math` requires a config name.

**Solution:** Loaded all 7 subject configs (algebra, counting_and_probability, geometry, etc.) and concatenated via `concatenate_datasets()` for full 5000 test problems.

### Problem 2: `import os` unused

**Issue:** Leftover `import os` in run_evaluation.py after refactor.

**Solution:** Removed unused import.
