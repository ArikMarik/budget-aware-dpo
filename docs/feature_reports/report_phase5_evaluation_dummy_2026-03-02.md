# Phase 5: Evaluation & Benchmarking — Report (Dummy Data)

**Date:** 2026-03-02  
**Status:** Complete  
**Data:** Dummy data only

## Implementation Summary

- **Evaluation module:** `src/evaluation/` — answer extraction, metrics, TPCA
- **Answer extraction:** Handles "The answer is X", "#### X", "\\boxed{X}", trailing numbers
- **Metrics:** Accuracy, TPCA (Tokens Per Correct Answer), avg tokens Easy/Hard
- **Script:** `scripts/run_evaluation.py` — evaluates baseline_dpo and budget_aware_dpo checkpoints

## Test Results (Dummy Data)

- Evaluated on 10 unique problems (limit)
- Baseline DPO: Accuracy 70%, TPCA 54.9, Avg Easy 38.4
- Budget-aware DPO: Accuracy 60%, TPCA 138.8, Avg Easy 83.3
- Results saved to `checkpoints/evaluation_results_dummy.json`

## Next Steps

- Phase 6: Visualization (dummy)
- Phase 9: Evaluation on real data (GSM8K, MATH)
