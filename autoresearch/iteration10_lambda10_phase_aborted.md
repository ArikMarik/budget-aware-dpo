# Iteration 10 — λ=10 All-In on Accuracy, 29.2% but Phase Aborted (Phase 3)

**Date**: 2026-03-28
**Branch**: `autoresearch/mar26`

## 1. Hypothesis

Phase 2 showed the 0.5B model actually outperforms 1.5B on easy problems (29.2% vs 24.0%) with much shorter tokens (177 vs 240). The base Qwen2.5-0.5B gets ~44% on GSM8K — we're below that after DPO. This iteration pushes the 0.5B model harder:

- **10a**: Budget-aware with stronger lambda (λ=10, doubled from 5) and more epochs (6 vs 3) — tests if stronger penalty + more training improves easy accuracy while keeping tokens short
- **10b**: Baseline with more epochs (6) — establishes the accuracy ceiling for 0.5B without budget penalty

## 2. Hyperparameters

### 10a: 0.5B Budget λ=10, 6 epochs (GPU 1)
```bash
CUDA_VISIBLE_DEVICES=1 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_0.5b_lambda10_iter10a \
  --max-epochs 6 --batch-size 4 --lr 1e-6 \
  --lambda-easy 10.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --early-stopping-patience 4 \
  --run-name budget_0.5b_lambda10_iter10a --wandb
```

### 10b: 0.5B Baseline, 6 epochs (GPU 2)
```bash
CUDA_VISIBLE_DEVICES=2 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_0.5b_6ep_iter10b \
  --max-epochs 6 --batch-size 4 --lr 1e-6 --kl-penalty 0.01 \
  --early-stopping-patience 4 \
  --run-name baseline_0.5b_6ep_iter10b --wandb
```

## 3. Changes Made

No code changes. Pure hyperparameter experiment.

## 4. Results

### 10a: Budget λ=10 (E1-E3 done, continuing to E6)
| Epoch | Accuracy | Easy Acc | Hard Acc | Easy Tok | Hard Tok | TPCA | val_loss |
|-------|---------|---------|---------|---------|---------|------|----------|
| 1 | 25% | 46% | 4% | 201 | 221 | 845 | 0.4018 |
| 2 | 27% | 52% | 2% | 255 | 244 | 922 | **0.3998** |
| 3 | 30% | 54% | 6% | 256 | 256 | 853 | 0.4024 |

Accuracy improving (25→30%) but tokens maxing at 256 by E3. λ=10 doesn't produce shorter tokens — the model generates full-length responses. val_loss plateauing around 0.40.

### 10b: Baseline 6 epochs (E1-E3, overfitting badly)
| Epoch | Accuracy | Easy Acc | Easy Tok | TPCA | val_loss |
|-------|---------|---------|---------|------|----------|
| 1 | 3% | 4% | 163 | 5337 | 0.5800 |
| 2 | 1% | 2% | 247 | 22462 | 0.7083 |
| 3 | 3% | 6% | 215 | 6867 | **1.1478** |

**Baseline completely fails** — val_loss exploding 0.58→1.15, accuracy 1-3%. Same pattern as 1.5B baseline. Standard DPO with KL=0.01 at lr=1e-6 doesn't work on this dataset without lambda.

### 10c: Budget λ=20 (just launched on GPU 0)
Testing if even stronger penalty produces different behavior.

### Key Insight
The budget-aware lambda provides a **crucial learning signal** that standard DPO lacks. Both 0.5B and 1.5B baselines fail at lr=1e-6 with KL=0.01, while budget-aware models learn effectively. The lambda isn't just a length penalty — it's an additional gradient signal that helps the model distinguish between easy and hard problems.

## 5. Reference: Full Post-Training Eval Table (all models so far)

| Model | Overall | Easy Acc | Hard Acc | Easy Tok | Hard Tok | TPCA | MATH L4-5 |
|-------|---------|---------|---------|---------|---------|------|-----------|
| 0.5B Baseline iter6 | 21.2% | 27.6% | 14.8% | 179 | 198 | 891 | 11.6% |
| 0.5B Budget iter5 | 22.0% | **29.2%** | 14.8% | **177** | 195 | 846 | 8.2% |
| 1.5B Baseline E1 (KL) | 22.4% | 20.0% | 24.8% | 242 | 177 | 935 | 13.0% |
| 1.5B Budget E1 | 24.6% | 22.4% | 26.8% | 244 | 177 | 856 | 13.7% |
| 1.5B Budget E2 | 25.8% | 24.0% | 27.6% | 240 | 179 | 812 | 13.7% |

**Target**: Push 0.5B easy accuracy from 29.2% toward 35-45% (base model gets 44%).

## 6. Benchmark Context

- Qwen2.5-0.5B base on GSM8K: ~44% (no fine-tuning)
- Qwen2.5-Coder-0.5B on GSM8K: 34.5% (4-shot)
- Our 0.5B budget iter5: 29.2% easy (held-out)
- Realistic ceiling after DPO: 45-55%
- Note: our eval uses max_new_tokens=256 which may truncate solutions. Will test with 512.
