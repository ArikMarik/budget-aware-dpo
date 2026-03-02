# Phase 10: Visualization from Real Data — Report

**Date:** 2026-03-02  
**Status:** Complete  
**Data:** Real evaluation results (GSM8K, MATH)

## Implementation Summary

- **run_visualization.py:** Default mode uses `_real` suffix (baseline_eval_real.json, budget_aware_eval_real.json). Output: `reports/figures/`
- **plot_results.py:** Added optional MATH Level 4–5 column to results table when present in metrics
- **Figures:** `length_histograms_real.pdf`, `length_by_complexity_real.pdf`
- **Results table:** `results_table_real.md`

## Test Results (50 problems, GSM8K)

| Model           | Accuracy | TPCA  | Avg Easy | Avg Hard |
|-----------------|----------|-------|----------|----------|
| baseline_dpo    | 26.0%    | 768.3 | 199.8    | 0.0      |
| budget_aware_dpo| 28.0%    | 696.9 | 195.1    | 0.0      |

## Usage

```bash
# Real data (default)
python scripts/run_visualization.py

# Dummy data
python scripts/run_visualization.py --dummy
```

## Output

- `reports/figures/length_histograms_real.pdf`
- `reports/figures/length_by_complexity_real.pdf`
- `reports/figures/results_table_real.md`

## Plan Update

Phase 11 (Final ACL report) moved from implementation plan to User notes and todos.

## Next Steps

- Final report: ACL-formatted project report (User notes)

---

## Appendix: Problems Encountered and Solutions

No significant problems encountered.
