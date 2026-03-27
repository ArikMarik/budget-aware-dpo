# Autoresearch Handoff — Live State

**Last updated**: 2026-03-27 08:15 UTC
**Branch**: `autoresearch/mar26`
**Agent status**: Iteration 4 training launched (both baseline + budget-aware)

---

## Current Runs (Iteration 4)

### Baseline
- **Run name**: `baseline_balanced_iter4`
- **PID**: 889675
- **GPU**: 0
- **Log**: `logs/baseline_balanced_iter4.log`
- **Checkpoints**: `checkpoints/baseline_balanced_iter4/`

### Budget-Aware
- **Run name**: `budget_aware_balanced_iter4`
- **PID**: 889686
- **GPU**: 1
- **Log**: `logs/budget_aware_balanced_iter4.log`
- **Checkpoints**: `checkpoints/budget_aware_balanced_iter4/`

### Hyperparameters
```
Budget: lambda_easy=5.0, lambda_hard=0.0, beta=0.1, kl_penalty=0.1 (NEW)
Both: max_epochs=3, batch_size=4, lr=1e-6 (was 1e-5), grad_accum=1
Dataset: data/processed_dpo_dataset_balanced_v4_capped (50K pairs, 6,347 unique problems, cap=50easy/100hard)
```

### Changes from iteration 3
1. **Dataset**: real_capped100 (51K, 11K problems) → balanced_v4_capped (50K, 6K problems, same pipeline as original)
2. **Learning rate**: 1e-5 → 1e-6 (10x lower to prevent model collapse)
3. **KL penalty**: 0.0 → 0.1 (prevents policy diverging from reference)
4. **Gen eval**: String comparison → tiered verify_correctness() (Tier 0+1 during training, full Tier 0+1+2 post-training)
5. **Baseline re-run**: Yes (lr changed + dataset changed)

### Expected timeline
- ~11,296 steps/epoch × ~1s/step = ~3.1 hours/epoch
- Gen eval: ~25 min after each epoch
- Total: ~10-12 hours for 3 epochs

---

## Iteration History

| Iter | Dataset | lr | Key Result | Status |
|------|---------|-----|------------|--------|
| 0 | balanced 50K | 1e-5 | Lambda too small (0.05), no divergence | Done |
| 1 | balanced 50K | 1e-5 | Lambda=5.0: accuracy +5.2% but tokens 2x UP | Done |
| 2 | capped50 10K | 1e-5 | Correct token direction but 1% accuracy | Done |
| 3 | real_capped100 51K | 1e-5 | MODEL COLLAPSE — 0% accuracy, gibberish | Done |
| 4 | balanced_v4_capped 50K | 1e-6 | Running... | Active |

---

## Rules
- **2 GPUs always occupied** during training
- **Document BEFORE acting** — iterationN.md written before launch
- **Never overwrite data/checkpoints** — new directories only
- **Poll every 20-30 min**, 1-hour for mid-epoch
- Read `autoresearch/RULES.md` for full operational rules

---

## How to resume
1. Read this file + `autoresearch/iteration4.md` for current plan
2. Check training: `ps aux | grep "train_b" | grep -v grep`
3. Poll logs: `grep -iE "Epoch.*train_loss|gen-eval" logs/*_iter4.log`
4. When done: collect results, run post-training eval, update iteration4.md, plan iter 5
