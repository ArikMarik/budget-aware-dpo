# Iteration 5 — KL=0.01 Breakthrough (Phase 1 Best: 22% Post-Train)

**Date**: 2026-03-27
**Status**: Preparing to launch
**Run name**: `budget_aware_balanced_iter5` (GPU 1 only — baseline continues from iter 4 on GPU 0)

---

## 1. Hypothesis

**KL penalty at 0.01 (10x lower than iter 4's 0.1) will allow the budget-aware model to learn while still preventing model collapse.**

Iteration 4 showed KL=0.1 was too strong — budget loss stayed flat at ~0.40 while baseline dropped to ~0.15. The model couldn't learn anything because every update was pulled back to the reference. At 0.01, the KL penalty should act as a gentle constraint rather than a straitjacket.

---

## 2. What Changed and Why

### Change 1: KL penalty 0.1 → 0.01
- **What**: `--kl-penalty 0.01`
- **Why**: At 0.1, budget loss was frozen at ~0.40 vs baseline ~0.15. WandB confirmed loss, reward_diff, and all metrics were essentially flat. The KL term dominated the gradient, preventing the DPO and length penalty signals from taking effect.
- **Risk**: 0.01 might be too weak to prevent collapse. But lr=1e-6 alone should provide stability (iter 1 at 1e-5 didn't collapse on balanced data, and we're at 10x lower lr now).

### No other changes
- Same dataset: `balanced_v4_capped` (50K, 6,347 unique problems)
- Same lr=1e-6
- Same lambda_easy=5.0, lambda_hard=0.0
- Same max_epochs=3
- Baseline continues from iter 4 (NOT re-run — only KL changed, which doesn't affect baseline)

---

## 3. Hyperparameters

```
lambda_easy=5.0, lambda_hard=0.0, beta=0.1
kl_penalty_weight=0.01 (was 0.1)
max_epochs=3, batch_size=4, lr=1e-6
gradient_accumulation_steps=1
early_stopping_patience=3
LoRA: r=128, alpha=256, dropout=0.05
Dataset: data/processed_dpo_dataset_balanced_v4_capped (50K pairs, 6,347 unique problems)
```

### Command
```bash
CUDA_VISIBLE_DEVICES=1 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_aware_balanced_iter5 --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --early-stopping-patience 3 --run-name budget_aware_balanced_iter5 --wandb \
  > logs/budget_aware_balanced_iter5.log 2>&1 &
```

### Baseline
Continuing from iteration 4 on GPU 0 (`baseline_balanced_iter4`, PID 889675). Not re-run because KL penalty only affects budget-aware.

---

## 4. Expected Outcomes

### Best case
- Budget loss tracks similarly to baseline (both declining)
- Gen eval shows >0% accuracy (model produces coherent text, not gibberish)
- Token direction signal visible (Easy shorter, Hard longer for budget vs baseline)

### Worst case
- 0.01 still too strong, or too weak (model collapses)
- If too strong: loss flat again → try 0.001
- If too weak: model collapse → the problem is lr, not KL

### Early detection criteria
- If budget loss stays within 2x of baseline loss by step 2000: KL is reasonable
- If budget loss > 3x baseline by step 2000: KL still too strong, kill and reduce
- If budget loss shows NaN/inf or gibberish gen eval: collapse, add more KL or raise lr

---

## 5. Results (in progress — budget still training)

### Baseline iter4 (continued from iter 4, killed at epoch 3)

Baseline was NOT re-launched for iter 5 — it continued from iter 4 since only KL changed (budget-only parameter).

| Epoch | Train Loss | Val Loss | Reward Diff | Accuracy | Easy Acc | Hard Acc | Tokens Easy | Tokens Hard | TPCA |
|-------|-----------|----------|-------------|----------|----------|----------|-------------|-------------|------|
| 1     | 0.1226    | 0.6655   | 5.3919      | 3%       | 4%       | 2%       | 177.3       | 186.7       | 6,067 |
| 2     | 0.0258    | 0.8194   | 6.2201      | 1%       | 2%       | 0%       | 256.0       | 252.3       | 25,413 |
| 3     | (killed)  | —        | —           | —        | —        | —        | —           | —           | — |

**Conclusion**: Baseline at lr=1e-6 is too slow — only 3% accuracy at epoch 1, then collapses into max-length gibberish by epoch 2 (tokens=256 = max). Val loss rising (0.67→0.82) confirms overfitting. Killed at epoch 3 step 825. **Not a valid comparison for budget-aware.**

**Decision**: Re-run baseline at lr=1e-5 on the same dataset (balanced_v4_capped). Iter 1 showed lr=1e-5 gives 32.4% accuracy on balanced data without collapsing. This gives a fair comparison point.

### Budget iter5 (KL=0.01) — Epoch 1 complete, epoch 2 in progress

| Epoch | Train Loss | Val Loss | Reward Diff | Accuracy | Easy Acc | Hard Acc | Tokens Easy | Tokens Hard | TPCA |
|-------|-----------|----------|-------------|----------|----------|----------|-------------|-------------|------|
| 1     | 0.2744    | 0.4528   | 0.0220      | **24%**  | **46%**  | 2%       | 202.6       | 219.3       | 879  |
| 2     | 0.2261    | 0.4561   | 0.0380      | 27%      | 52%      | 2%       | 148.3       | 242.1       | 723  |
| 3     | 0.2138    | 0.4460   | 0.0342      | **38%**  | **68%**  | **8%**   | 193.3       | 245.9       | **578** |

**Key findings so far:**
1. **KL=0.01 works!** Budget loss declined steadily (0.69→0.27 over epoch 1), unlike iter 4's KL=0.1 which was frozen at 0.40.
2. **24% accuracy at epoch 1** — 8x better than baseline at the same lr. Easy accuracy (46%) dominates.
3. **TPCA=879** — much more efficient than baseline's 6,067.
4. **Token direction NOT yet visible**: easy=203, hard=219. Budget generates slightly longer than baseline on easy (177), not shorter. May need more training or different lambda.
5. **Reward diff very low (0.022)**: Budget model learned more conservatively than baseline (5.39). The length penalty modulates the reward signal.

### New baseline (iter5b) — lr=1e-5 on GPU 0

Launched after killing baseline iter4. Same dataset (balanced_v4_capped), standard DPO, lr=1e-5. Run name: `baseline_balanced_iter5b`.

---

## 6. Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `checkpoints/budget_aware_balanced_iter5/` | Created | Budget checkpoints |
| `checkpoints/baseline_balanced_iter5b/` | Created | New baseline at lr=1e-5 |
| No code changes | — | Only CLI parameters changed |
