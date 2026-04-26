# Iteration 9 — 1.5B No-KL Baseline, Fair Comparison (Phase 2)

**Date**: 2026-03-29
**Branch**: `autoresearch/mar26`

## 1. Hypothesis

The 1.5B baseline with KL=0.01 (iter 7a) only achieved 6% gen-eval accuracy and was severely overfitting (train_loss=0.02 by E2). This made the comparison with 1.5B budget-aware (43% accuracy) unfair.

**Hypothesis**: A 1.5B baseline without KL penalty will perform better than the KL=0.01 version, giving us a fairer comparison for the budget-aware improvement.

## 2. Hyperparameters

```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_1.5b_noKL_iter9 \
  --model Qwen/Qwen2.5-1.5B \
  --max-epochs 3 --batch-size 2 --lr 1e-6 \
  --kl-penalty 0.0 \
  --early-stopping-patience 3 \
  --run-name baseline_1.5b_noKL_iter9 --wandb \
  > logs/baseline_1.5b_noKL_iter9.log 2>&1 &
```

## 3. Changes Made

No code changes. Same training infrastructure as iteration 7.

## 4. Results

### Epoch 1 (gen-eval)
| Metric | Value |
|--------|-------|
| train_loss | 0.1196 |
| val_loss | 0.6125 |
| accuracy | 10% |
| easy_acc | 18% |
| hard_acc | 2% |
| tokens_easy | **74.3** |
| tokens_hard | 185.4 |
| TPCA | 1298.4 |

### Epoch 2
Process died during gen-eval (likely OOM from leaked GPU memory from killed iter 7 processes). Only E1 checkpoint available.

### Post-Training Eval (E1, 500 held-out, Tier 0+1+2)
*Pending — eval running on GPU 0.*

## 5. Comparison

| Metric | 1.5B NoKL (iter9 E1) | 1.5B KL (iter7a E1) | 1.5B Budget (iter7b E2) |
|--------|---------------------|---------------------|------------------------|
| gen accuracy | **10%** | 6% | **43%** |
| easy_acc | **18%** | 12% | **74%** |
| tokens_easy | **74.3** | 149.0 | 210.7 |
| tokens_hard | 185.4 | 178.8 | 251.1 |
| TPCA | **1298.4** | 2732.0 | **537.0** |
| val_loss | **0.6125** | 0.7426 | **0.4240** |

## 6. Analysis

1. **NoKL baseline is better than KL baseline**: 10% vs 6% accuracy, 0.61 vs 0.74 val_loss. The KL penalty was hurting the 1.5B baseline.

2. **Budget-aware still dominates**: Even with the improved noKL baseline, budget-aware (43% acc, 537 TPCA) is far ahead of noKL baseline (10% acc, 1298 TPCA).

3. **Extremely short easy tokens (74.3)**: The noKL baseline generates very short responses on easy problems — even shorter than budget-aware. This might indicate the model is truncating/not answering properly, or it's generating concise but correct answers. Post-training eval will clarify.

4. **Budget-aware length penalty helps learning**: The budget model (with lambda_easy=5.0 + KL=0.01) learns 4-7x faster than either baseline variant. The length penalty seems to provide a beneficial regularization signal for the 1.5B model, not just length control.

## 7. Open Questions

1. How does noKL baseline perform on held-out data (post-training eval)?
2. Is 74.3 easy tokens genuine conciseness or truncated/garbage output?
3. Why does the budget-aware model learn so much faster than both baselines?

## 8. Next Iteration Plan

- Evaluate noKL baseline E1 on held-out test set
- If noKL baseline accuracy is reasonable on held-out, this becomes the definitive baseline comparison
- Try budget-aware with higher lambda (10 or 20) to push easy tokens shorter
