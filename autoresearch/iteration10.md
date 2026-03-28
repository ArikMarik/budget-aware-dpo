# Iteration 10 — Phase 3: 0.5B All-In on Easy Accuracy (λ=10, 6 Epochs)

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

*In progress — launched at 20:20 (10a) and 20:52 (10b)*

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
