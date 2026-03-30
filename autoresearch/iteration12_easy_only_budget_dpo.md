# Iteration 12 — Easy-Only Budget-Aware DPO Training (Phase 4)

**Date**: 2026-03-30
**Phase**: 4 (Baseline Establishment & Budget-Aware on Easy)
**Status**: IN PROGRESS

---

## 1. Hypothesis

Training budget-aware DPO **only on easy problems** (complexity=0) should:
- Maximize easy accuracy (match or beat base model's 40.7% GSM8K 8-shot)
- Produce shorter solutions on easy problems (the core budget-aware effect)
- Not degrade hard-problem accuracy since we're not training on hard problems at all

The iter5 model matched base on easy (41.2%) but collapsed on hard (16.8% vs 32%). By training only on easy data, we focus the model's learning budget entirely on the task we care about.

## 2. Experiment Design

### Training
- **Dataset**: `data/processed_dpo_dataset_easy_only/` (22,742 train, 2,258 val — all complexity=0)
- **Model**: Qwen2.5-0.5B + LoRA (r=128, alpha=256)
- **Loss**: Budget-aware DPO with λ_easy=5.0, λ_hard=0.0, KL=0.01
- **Hyperparameters**: lr=1e-6, batch=4, grad_accum=4, effective_batch=16, 3 epochs
- **Checkpoint**: `checkpoints/budget_easy_only_iter12/`

### Evaluation
- 8-shot eval on 500 problems (GSM8K + MATH L1-2), full tiered LLM judge
- Compare to base model 8-shot baseline (40.7% GSM8K, 36.4% overall)

## 3. CLI

```bash
# Training
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project \
  PYTHONUNBUFFERED=1 DATASET_PATH=data/processed_dpo_dataset_easy_only \
  .venv/bin/python scripts/training/train_budget_aware_dpo.py \
  --output-dir checkpoints/budget_easy_only_iter12 \
  --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --gradient-accumulation-steps 4 --run-name budget_easy_only_iter12 --wandb

# 8-shot eval
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/budget_easy_only_iter12 \
  --output eval_results/budget_easy_only_iter12_8shot.json \
  --use-real --limit 500 --few-shot 8
```

## 4. Results

### Training ✅

| Epoch | Val Loss | Gen Eval Acc (50 easy) | Avg Tokens Easy | TPCA |
|-------|---------|----------------------|----------------|------|
| 1 | 0.0943 (best) | 36.0% | 130.5 | 362.6 |
| 2 | — | 34.0% | 143.9 | 423.4 |
| 3 | — | **42.0%** | 160.4 | 381.9 |

Training time: ~2h 13min. Epoch 3 best accuracy (42.0%), above base 8-shot (40.7%).
Token reduction strongest at E1 (130 vs 157 base), trading off with accuracy.

### 8-shot Evaluation (500 problems, balanced 250 easy + 250 hard) ✅

| Model | Easy (8-shot) | Hard (8-shot) | Overall | Avg Tok Easy | TPCA |
|-------|-------------|-------------|---------|-------------|------|
| Base (no training) | 40.7% | 32.0% | 36.4% | 156.7 | 476.2 |
| Budget iter5 (λ=5, mixed data) | 41.2% | 16.8% | 29.0% | 158.2 | 638.9 |
| **Budget easy-only (λ=5, easy data)** | **41.6%** | 16.4% | 29.0% | 155.7 | 633.4 |

## 5. Analysis

### Key Findings

1. **Easy accuracy preserved**: Both budget models match or slightly exceed the base model on easy problems (41.2-41.6% vs 40.7%). The budget-aware mechanism successfully preserves easy-problem capability.

2. **Hard collapse is universal**: Both budget models drop from 32% to ~16% on hard problems. This happens regardless of whether hard problems were in the training data. The DPO training process itself — not the data composition — causes hard degradation.

3. **Easy-only data ≈ mixed data**: Training on easy-only data (22.7K pairs) produces nearly identical results to training on mixed data (50K pairs). This suggests the hard problems in training data weren't contributing to easy accuracy.

4. **Token efficiency is modest**: Avg easy tokens ~156 for both budget models vs 157 for base — essentially no token reduction with 8-shot prompting. (With 0-shot, the in-training gen-eval showed stronger reduction at E1: 130 tokens.)

### Implications

- The budget-aware DPO mechanism works for **preserving** easy accuracy but doesn't **improve** it beyond the base model
- Hard problem degradation is a fundamental issue with the DPO training setup, not data composition
- For the paper: the story is about maintaining easy accuracy while (potentially) reducing tokens, compared to the untrained baseline

## 6. Next Steps

Options to explore:
- **Reduce training epochs** (E1 had best token reduction: 130 tokens at 36% acc)
- **Increase lambda** to push harder on token efficiency
- **Few-shot prompt in training** — match eval format to training format
- **SFT first, then budget DPO** — establish accuracy, then compress
