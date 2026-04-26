# Autoresearch Handoff — Phase 4

**Last updated**: 2026-03-29 11:45 UTC
**Phase**: Phase 4 — Baseline Establishment & SFT Accuracy Push
**Branch**: `autoresearch/mar26`
**GPU**: 1 GPU available (GPU 0)

---

## Start Here

You are the Phase 4 agent. Read these files in order:
1. **This file** — current state and what to do next
2. **`autoresearch/PHASE4_PRD.md`** — the "bible" for this phase, all goals and tasks
3. **`autoresearch/RULES.md`** — operational rules (polling, commit protocol, etc.)
4. **`program.md`** — master experiment protocol
5. **`autoresearch/iteration11.md`** — current iteration (in progress)

---

## Current State

### What's Running
- Nothing — GPU 0 is FREE

### What's Done (Iterations 11-12)
- ✅ **Easy 0-shot eval**: GSM8K=30.4%, MATH L1=42.8%, MATH L2=27.6% (2,650 problems)
- ✅ **Hard 0-shot eval**: MATH L3=18.0%, L4=10.2%, L5=4.6% (3,669 problems)
- ✅ **8-shot base eval**: GSM8K=40.7%, overall=36.4% (500 problems) — closes gap to published 44%
- ✅ **8-shot budget iter5 eval**: Easy=41.2% (matches base), Hard=16.8% (collapsed)
- ✅ **Easy-only budget DPO trained** (iter12): E3 42% easy gen-eval
- ✅ **8-shot easy-only eval**: Easy=41.6%, Hard=16.4% — nearly identical to iter5
- ✅ Easy-only dataset created, 8-shot support, zero data leakage confirmed, files renamed

### What's Done
- ✅ `scripts/eval_base_model.py` written — base model eval with raw/LoRA-init modes
- ✅ `scripts/training/train_sft.py` written — SFT training script (cross-entropy on chosen solutions)
- ✅ `autoresearch/iteration11.md` started (needs results filled in)

### What's Next
- Awaiting user direction. Key findings documented in iteration 11 and 12.
- Options: increase λ, reduce epochs for token efficiency, SFT-then-DPO, few-shot in training

Easy-only training command:
   ```bash
   CUDA_VISIBLE_DEVICES=0 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
     PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_sft \
     --output-dir checkpoints/sft_v1 --max-epochs 3 --batch-size 4 --lr 2e-5 \
     --run-name sft_v1 --wandb > logs/sft_v1.log 2>&1 &
   ```
6. **Evaluate SFT** and iterate hyperparameters

---

## Phase 4 Goals (from PRD)

| Priority | Goal |
|----------|------|
| P0 | Evaluate base Qwen2.5-0.5B on our eval pipeline — get ground truth |
| P0 | Understand any gap vs published 44% GSM8K |
| P1 | SFT fine-tune, maximize accuracy (standard cross-entropy, NOT DPO) |
| P1 | Hyperparameter sweep until accuracy plateaus |

**What NOT to do**: No token optimization, no DPO, no budget-aware, no multi-GPU.

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval_base_model.py` | Eval raw base model (supports --with-lora-init) |
| `scripts/training/train_sft.py` | SFT training (cross-entropy on chosen solutions) |
| `scripts/eval_checkpoint.py` | Eval any LoRA checkpoint (existing) |

---

## Eval Commands

```bash
# Base model eval (raw, full dataset)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_base_model.py \
  --output eval_results/base_qwen_0.5b_full_256.json --use-real --max-new-tokens 256

# Base model eval (with untrained LoRA)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_base_model.py \
  --output eval_results/base_qwen_0.5b_lora_init_256.json --use-real --with-lora-init --max-new-tokens 256

# SFT checkpoint eval
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/sft_v1/best-model --limit 500 \
  --output eval_results/sft_v1.json --use-real
```

---

## Iteration Numbering

- Iterations 0-10: Phase 1-3 (DPO experiments)
- **Iteration 11**: Base model evaluation (IN PROGRESS)
- Iteration 12+: SFT experiments

---

## Operational Notes

- Use `.venv/bin/python` (NOT system python)
- Set `PYTHONUNBUFFERED=1` for all training/eval
- Set `PYTHONPATH=/storage/arik/nlp_final_project` for eval scripts
- Poll every 20 min, /compact at 200K context
- Commit after each completed iteration
- **Always update this HANDOFF.md** before /compact or when state changes significantly
