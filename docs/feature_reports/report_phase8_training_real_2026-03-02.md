# Phase 8: Full Training on Real Data — Report

**Date:** 2026-03-02  
**Status:** Complete  
**Data:** Real data (4994 DPO pairs from OpenMathInstruct-2)

## Implementation Summary

- **Output dir routing:** Added `get_baseline_output_dir()` and `get_budget_aware_output_dir()` in config — when `USE_DUMMY_DATA=0`, checkpoints save to `*_real` subdirs
- **Training scripts:** Default output dir now depends on dummy vs real mode
- **Baseline DPO:** Trained on real data, saved to `checkpoints/baseline_dpo_real/`
- **Budget-aware DPO:** Trained on real data, saved to `checkpoints/budget_aware_dpo_real/`

## Test Results

- **Baseline:** 100 steps, batch 4, 4994 pairs → loss 9.63 → 0.09
- **Budget-aware:** 100 steps, batch 4, 4994 pairs → loss 4.14 → 0.09
- Both checkpoints saved with `training_config.json`, `metrics.json`

## Usage

```bash
# Full training on real data (ensure Phase 7 preprocessing done)
USE_DUMMY_DATA=0 python scripts/training/train_baseline_dpo.py --max-epochs 10
USE_DUMMY_DATA=0 python scripts/training/train_budget_aware_dpo.py --max-epochs 10
```

## Next Steps

- Phase 9: Evaluation on real data (GSM8K, MATH test sets)

---

## Appendix: Problems Encountered and Solutions

No significant problems encountered. Training pipeline ran successfully with real data. Output directory routing was added proactively to support the dummy/real distinction.
