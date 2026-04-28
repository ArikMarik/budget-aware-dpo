# Iteration 14 — Easy-Only Budget-Aware DPO Training (Phase 4)

**Date**: 2026-04-07  
**Phase**: 4 (Baseline Establishment & Budget-Aware on Easy)  
**Status**: COMPLETE (train + eval)

---

## 1. Hypothesis

Training budget-aware DPO **only on easy problems** (complexity=0) should:
- Preserve easy accuracy while improving token efficiency on easy problems (shorter solutions)
- Avoid (or at least reduce) the hard-problem collapse seen in earlier budget-DPO runs

This iteration pushes **token efficiency selection** harder by:
- Lowering λ_easy from 5.0 → **3.0**
- Selecting the best checkpoint by **lowest `gen/avg_tokens_easy`**, but only among epochs with **`gen/accuracy_easy >= 0.55`**

## 2. Experiment Design

### Training
- **Dataset**: `data/processed_dpo_dataset_easy_only/` (easy-only)
- **Model**: Qwen2.5-0.5B + LoRA (r=128, alpha=256)
- **Loss**: Budget-aware DPO with λ_easy=3.0, λ_hard=0.0, KL=0.01
- **Hyperparameters**: lr=1e-6, batch=2, grad_accum=2, effective_batch=4, 3 epochs
- **Best model selection**: `gen_tokens_easy_with_accuracy_floor` with `accuracy_floor=0.55`
- **Checkpoint**: `checkpoints/budget_easy_only_iter14/`

### Evaluation
- 8-shot eval on 500 problems (balanced 250 easy + 250 hard)
- Output: `eval_results/budget_easy_only_iter14_8shot.json`

## 3. CLI

```bash
# Training (conda env; no venv)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  DATASET_PATH=data/processed_dpo_dataset_easy_only \
  python scripts/training/train_budget_aware_dpo.py \
  --output-dir checkpoints/budget_easy_only_iter14 \
  --max-epochs 3 --batch-size 2 --lr 1e-6 \
  --lambda-easy 3.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --gradient-accumulation-steps 2 \
  --best-model-metric gen_tokens_easy_with_accuracy_floor \
  --accuracy-floor 0.55 \
  --run-name budget_iter14 --wandb

# 8-shot eval (using best-model)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/budget_easy_only_iter14/best-model/ \
  --output eval_results/budget_easy_only_iter14_8shot.json \
  --use-real --limit 500 --few-shot 8
```

## 4. Results

### 8-shot Evaluation (500 problems, balanced 250 easy + 250 hard) ✅

From `eval_results/budget_easy_only_iter14_8shot.json`:

| Model | Easy (8-shot) | Hard (8-shot) | Overall | Avg Tok Easy | TPCA |
|-------|-------------|-------------|---------|-------------|------|
| Budget easy-only (iter14) | _(not in file; see note)_ | _(not in file; see note)_ | **30.4%** (152/500) | **155.3** | **602.8** |

**Additional metrics**
- `avg_tokens_hard`: 211.2  
- `math_level_4_5_accuracy`: 10.27% (n=146)

**Note (easy/hard accuracy):** the eval JSON stores overall accuracy plus `num_easy`/`num_hard`, but does not include easy-only vs hard-only accuracy as separate aggregate fields. If you want those, we can compute them from the `results[]` list.

## 5. Analysis (quick)

- Overall accuracy improved vs iter13 (30.4% vs 27.6%).
- TPCA improved vs iter13 (602.8 vs 664.1).
- Avg easy tokens roughly similar (155.3 vs 154.1).
- Confirm which epoch was selected by checking:
  - `checkpoints/budget_easy_only_iter14/best_model_selection.json`
  - `checkpoints/budget_easy_only_iter14/summary.json`

## 6. Next Steps

- Compute easy vs hard accuracies from the eval `results[]` to pinpoint where the gains came from.
- Compare per-epoch `metrics.json` to see how the accuracy floor affected checkpoint selection.

