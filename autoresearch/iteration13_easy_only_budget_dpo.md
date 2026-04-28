# Iteration 13 — Easy-Only Budget-Aware DPO Training (Phase 4)

**Date**: 2026-04-07  
**Phase**: 4 (Baseline Establishment & Budget-Aware on Easy)  
**Status**: COMPLETE (train + eval)

---

## 1. Hypothesis

Training budget-aware DPO **only on easy problems** (complexity=0) should:
- Preserve or improve easy accuracy
- Encourage shorter solutions on easy problems (token efficiency)
- Avoid harming hard problems (though prior runs showed hard degradation can still happen)

This iteration additionally uses **best checkpoint selection by token efficiency** (with an accuracy floor) rather than always selecting by validation loss.

## 2. Experiment Design

### Training
- **Dataset**: `data/processed_dpo_dataset_easy_only/` (easy-only)
- **Model**: Qwen2.5-0.5B + LoRA (r=128, alpha=256)
- **Loss**: Budget-aware DPO with λ_easy=5.0, λ_hard=0.0, KL=0.01
- **Hyperparameters**: lr=1e-6, batch=4, grad_accum=4, effective_batch=16, 3 epochs
- **Best model selection**: `gen_tokens_easy_with_accuracy_floor` with `accuracy_floor=0.40`
- **Checkpoint**: `checkpoints/budget_easy_only_iter13/`

### Evaluation
- 8-shot eval on 500 problems (balanced 250 easy + 250 hard)
- Output: `eval_results/budget_easy_only_iter13_8shot.json`

## 3. CLI

```bash
# Training (conda env; no venv)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  DATASET_PATH=data/processed_dpo_dataset_easy_only \
  python scripts/training/train_budget_aware_dpo.py \
  --output-dir checkpoints/budget_easy_only_iter13 \
  --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --gradient-accumulation-steps 4 \
  --best-model-metric gen_tokens_easy_with_accuracy_floor \
  --accuracy-floor 0.40 \
  --run-name budget_iter13 --wandb

# 8-shot eval (conda env; no venv)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/budget_easy_only_iter13 \
  --output eval_results/budget_easy_only_iter13_8shot.json \
  --use-real --limit 500 --few-shot 8
```

## 4. Results

### 8-shot Evaluation (500 problems, balanced 250 easy + 250 hard) ✅

From `eval_results/budget_easy_only_iter13_8shot.json`:

| Model | Easy (8-shot) | Hard (8-shot) | Overall | Avg Tok Easy | TPCA |
|-------|-------------|-------------|---------|-------------|------|
| Budget easy-only (iter13) | _(not in file; see note)_ | _(not in file; see note)_ | **27.6%** (138/500) | **154.1** | **664.1** |

**Additional metrics**
- `avg_tokens_hard`: 212.5  
- `math_level_4_5_accuracy`: 7.53% (n=146)

**Note (easy/hard accuracy):** the eval JSON stores overall accuracy plus `num_easy`/`num_hard`, but does not include easy-only vs hard-only accuracy as separate aggregate fields. If you want, I can add a tiny script/snippet to compute and print easy/hard accuracies from the `results[]` list.

## 5. Analysis (quick)

- Overall accuracy (27.6%) is low for the balanced 8-shot set, despite easy-only training.
- Token usage on easy problems is moderate (154 avg tokens).
- This run used **token-efficiency-based best checkpointing with an accuracy floor**; confirm which epoch was selected by checking:
  - `checkpoints/budget_easy_only_iter13/best_model_selection.json`
  - `checkpoints/budget_easy_only_iter13/summary.json`

## 6. Next Steps

- Compute easy vs hard accuracy from the eval `results[]` to diagnose where the drop comes from.
- Compare iter13’s selected epoch metrics vs prior iter12 (val-loss-selected) to see if metric-based selection is picking a meaningfully different checkpoint.

