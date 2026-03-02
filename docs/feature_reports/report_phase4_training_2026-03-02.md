# Phase 4: Full Model Training & Baselines — Report

**Date:** 2026-03-02  
**Status:** Complete

## Implementation Summary

- **Standard DPO loss:** `src/models/standard_dpo_loss.py` — no length penalty
- **Shared trainer:** `src/training/dpo_trainer.py` — data loading, batching, checkpointing, metrics
- **Baseline script:** `scripts/training/train_baseline_dpo.py` — standard DPO
- **Budget-aware script:** `scripts/training/train_budget_aware_dpo.py` — custom R_budget loss
- **Experiment tracking:** `training_config.json`, `metrics.json` per run
- **Checkpointing:** Every N steps (default 500), `resume_from` support

## Test Results

- Baseline DPO: 20 steps, batch 2, 18 pairs → loss 0.30 → 0.00002
- Budget-aware DPO: 20 steps, batch 2, 18 pairs → loss 0.38 → 0.006
- Both checkpoints saved to `checkpoints/baseline_dpo/` and `checkpoints/budget_aware_dpo/`

## CLI Usage

```bash
# Baseline (standard DPO)
python scripts/training/train_baseline_dpo.py --max-steps 1000 --batch-size 4

# Budget-aware DPO
python scripts/training/train_budget_aware_dpo.py --max-steps 1000 --batch-size 4

# Resume from checkpoint
python scripts/training/train_budget_aware_dpo.py --resume-from checkpoints/budget_aware_dpo/checkpoint-500
```

## Next Steps

- Phase 5: Evaluation on GSM8K/MATH, TPCA, vLLM latency
