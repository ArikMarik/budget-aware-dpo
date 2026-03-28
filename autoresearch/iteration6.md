# Budget-Aware DPO — Auto-Research Iteration 6

**Date**: 2026-03-27
**Status**: Launching
**Run name**: `baseline_kl_iter6` (GPU 0 only — budget continues from iter 5 on GPU 1, finishing epoch 3)

---

## 1. Hypothesis

**A baseline with KL=0.01 at lr=1e-6 will learn properly (matching budget-aware accuracy), providing a fair comparison for token direction and TPCA.**

Previous baselines failed because:
- lr=1e-6 without KL: learned too slowly (3% accuracy), overfitted by epoch 2
- lr=1e-5 without KL: overfitted immediately (1% accuracy, max-length gibberish epoch 1)

Budget-aware at lr=1e-6 with KL=0.01 gets 27% accuracy and stable val_loss. The KL penalty acts as a regularizer that prevents overfitting. Adding KL to baseline should give it the same stability, making a fair comparison where the ONLY difference is the length penalty (lambda).

---

## 2. What Changed and Why

### Change 1: Add KL penalty to baseline
- **What**: Use budget-aware trainer with `--lambda-easy 0.0 --lambda-hard 0.0 --kl-penalty 0.01`
- **Why**: With lambda=0 for both, there's no length penalty — it's standard DPO + KL regularization. This matches budget-aware's regularization while removing the budget signal.
- **Implementation**: Reuse `train_budget_aware_dpo.py` with zero lambdas. The length penalty term becomes zero, leaving only DPO loss + KL penalty.

### No other changes
- Same dataset: `balanced_v4_capped` (50K, 6,347 unique problems)
- Same lr=1e-6
- Same max_epochs=3
- Same batch_size=4, grad_accum=1

---

## 3. Hyperparameters

```
lambda_easy=0.0, lambda_hard=0.0, beta=0.1
kl_penalty_weight=0.01
max_epochs=3, batch_size=4, lr=1e-6
gradient_accumulation_steps=1
early_stopping_patience=3
LoRA: r=128, alpha=256, dropout=0.05
Dataset: data/processed_dpo_dataset_balanced_v4_capped (50K pairs, 6,347 unique problems)
```

### Command
```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/baseline_kl_iter6 --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 0.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --early-stopping-patience 3 --run-name baseline_kl_iter6 --wandb \
  > logs/baseline_kl_iter6.log 2>&1 &
```

### Comparison to budget iter5
| Parameter | Baseline iter6 | Budget iter5 |
|-----------|---------------|-------------|
| lambda_easy | **0.0** | **5.0** |
| lambda_hard | 0.0 | 0.0 |
| kl_penalty | 0.01 | 0.01 |
| lr | 1e-6 | 1e-6 |
| dataset | balanced_v4_capped | balanced_v4_capped |

The ONLY difference is lambda_easy. This is the cleanest possible comparison.

---

## 4. Expected Outcomes

### Best case
- Baseline accuracy 20-30% (matching budget) — proves KL regularization helps
- Token counts similar to pre-trained model (no length direction) — baseline generates equally long for easy and hard
- Budget shows clear token direction advantage (shorter easy, longer hard) at similar accuracy

### Worst case
- Baseline still can't learn at lr=1e-6 even with KL — fundamental issue with this lr for standard DPO
- If so: the budget-aware loss IS the reason for learning (not just KL), which is also an interesting finding

---

## 5. Results

### Epoch-Level Metrics (in-training gen eval, Tier 0+1, 100 problems)

| Epoch | Train Loss | Val Loss | Reward Diff | Accuracy | Easy Acc | Hard Acc | Tokens Easy | Tokens Hard | TPCA |
|-------|-----------|----------|-------------|----------|----------|----------|-------------|-------------|------|
| 1     | 0.4164    | 0.5708   | 0.9284      | 23%      | 44%      | 2%       | 120.2       | 227.8       | 757  |
| 2     | 0.3488    | 0.5703   | 0.9553      | 35%      | 68%      | 2%       | 144.0       | 250.9       | 564  |
| 3     | 0.3327    | 0.6022   | 0.9204      | **39%**  | **76%**  | 2%       | 198.4       | 256.0       | 583  |

**Best checkpoint**: Epoch 2 (lowest val_loss=0.5703). Epoch 3 shows val_loss rising (overfitting).

### Side-by-Side with Budget iter5 (in-training gen eval)

| Metric | Baseline iter6 E2 | Baseline iter6 E3 | Budget iter5 E2 | Budget iter5 E3 |
|--------|-------------------|-------------------|-----------------|-----------------|
| Accuracy | 35% | 39% | 27% | 38% |
| Easy Acc | 68% | 76% | 52% | 68% |
| Hard Acc | 2% | 2% | 2% | **8%** |
| Tokens Easy | **144.0** | 198.4 | 148.3 | 193.3 |
| Tokens Hard | 250.9 | 256.0 | 242.1 | 245.9 |
| TPCA | **564** | 583 | 723 | 578 |
| Val Loss | 0.5703 | 0.6022 | 0.4561 | 0.4460 |

### Post-Training Eval (500 problems, Tier 0+1+2, held-out test)

| Metric | Budget iter5 (λ=5.0) | Baseline iter6 (λ=0.0) | Δ |
|--------|---------------------|----------------------|---|
| Overall accuracy | **22.0%** | 21.2% | +0.8% |
| Avg tokens easy | **177.4** | 179.4 | **-2.0 tokens** |
| Avg tokens hard | **194.8** | 198.3 | -3.5 tokens |
| TPCA | **845.9** | 890.8 | **-44.9 (5% better)** |
| MATH L4-5 | 8.2% | **11.6%** | -3.4% |

**Conclusion**: The length penalty (λ=5.0) produces a **marginal** improvement. Budget-aware is ~2 tokens shorter on easy and ~5% better TPCA, but the effect is small. The KL regularization alone (baseline) achieves nearly identical results. The budget-aware signal exists but is too weak to be a convincing paper contribution at this scale.

---

## 6. Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `checkpoints/baseline_kl_iter6/` | Created | Baseline + KL checkpoints |
| No code changes | — | Only CLI parameters changed |
