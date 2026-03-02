# Phase 3: Custom Loss Function & Sanity Check (Overfitting) — Report

**Date:** 2026-03-02  
**Status:** Complete

## Implementation Summary

- **Budget-aware DPO loss:** Implemented in `src/models/budget_aware_dpo_loss.py`
  - R_budget(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) − λ(C)·|y|
  - Dynamic λ: high (0.05) for Easy (C=0), near zero (0.001) for Hard (C=1)
- **LoRA config:** r=128, alpha=256, target_modules q_proj/v_proj/k_proj/o_proj
- **Sanity check script:** `scripts/train_sanity_check.py` — overfits on 30 dummy pairs, 3 epochs
- **Inspection script:** `scripts/inspect_sanity_outputs.py` — loads checkpoint, generates on Easy/Hard prompts, reports token counts

## Test Results

- **Training:** Ran on GPU (cuda). Checkpoint saved to `checkpoints/sanity_overfit/`
- **Inspection:** Easy prompts → 60, 61, 256 tokens (first two compressed; third hit max_length)
- **Inspection:** Hard prompts → 256, 256, 7 tokens (two full CoT; one short)
- **Avg tokens:** Easy 125.7 | Hard 173.0
- Model successfully memorized and produced math-relevant outputs. Easy problems tend toward shorter responses; Hard problems produce detailed CoT when not truncated.

## Issues Resolved

- Sanity check previously run on CPU; re-ran on GPU
- Added PYTHONPATH for script execution from project root

## Next Steps

- Phase 4: Full model training (baseline DPO + budget-aware DPO) and experiment tracking
