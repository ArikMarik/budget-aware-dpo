# autoresearch

This is an experiment to have the LLM do its own research on **Budget-Aware DPO** training.

## Goal

Make the budget-aware DPO model produce **measurably different behavior** from the standard DPO baseline:
- **Shorter responses on Easy problems** (lower avg tokens) without sacrificing accuracy
- **Preserved reasoning depth on Hard problems** (maintain accuracy and token count)
- **Lower TPCA (Tokens Per Correct Answer)** overall

Read `autoresearch/iteration0.md` for the full project context, dataset analysis, current results, and diagnosed problems.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `david` branch.
3. **Read the in-scope files** for full context:
   - `autoresearch/iteration0.md` — full project context: dataset, model, loss functions, current results, diagnosis, and suggestions.
   - `src/models/budget_aware_dpo_loss.py` — the budget-aware loss function (editable).
   - `src/models/standard_dpo_loss.py` — the baseline loss function (read-only reference).
   - `src/training/dpo_trainer.py` — training loop, metrics, evaluation (editable: hyperparameters, training logic, optimizer config, LoRA config).
   - `scripts/training/train_budget_aware_dpo.py` — budget-aware training CLI (editable: CLI args, defaults).
   - `scripts/training/train_baseline_dpo.py` — baseline training CLI (read-only unless shared params change).
   - `src/config.py` — paths, model name, constants (read-only).
   - `src/evaluation/run_evaluation.py` — post-training evaluation (read-only, this is the ground truth).
4. **Verify data exists**: Check that `data/processed_dpo_dataset_balanced/dataset.jsonl` exists and has 50,000 lines. If not, the preprocessing pipeline needs to run first.
5. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row. The baseline from iteration 0 will be the first entry.
6. **Start keep_alive.py**: If GPUs are idle, run `nohup python keep_alive.py > logs/keep_alive.log 2>&1 &`.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

This project uses **two GPUs** (GPU 0 and GPU 1). Training runs take **3-8 hours per epoch** (not 5 minutes like the template). Each experiment typically runs 2-3 epochs on 46,290 training pairs.

**What you CAN modify:**
- `src/models/budget_aware_dpo_loss.py` — loss function, normalization, lambda logic
- `src/training/dpo_trainer.py` — hyperparameters, optimizer, LoRA config, training loop, learning rate schedule, gradient accumulation
- `scripts/training/train_budget_aware_dpo.py` — CLI arguments and defaults
- `scripts/training/train_baseline_dpo.py` — only if shared hyperparameters change
- `data/` preprocessing scripts — if you need a different data split or augmentation

**What you CANNOT modify:**
- `src/evaluation/run_evaluation.py` — the ground truth evaluation (accuracy, TPCA, avg_tokens_easy, avg_tokens_hard)
- `src/evaluation/answer_extraction.py` — answer parsing
- `src/evaluation/math_grader.py` — answer verification
- The base model choice (Qwen/Qwen2.5-0.5B)

**The goal**: Make the budget-aware model show **clear divergence** from the baseline on these metrics:

| Metric | Success condition |
|--------|-------------------|
| `val/accuracy` | Budget-aware >= baseline |
| `val/accuracy_easy` vs `val/accuracy_hard` | More divergence in budget-aware |
| `train/reward_diff_easy` vs `train/reward_diff_hard` | More divergence in budget-aware |
| `train/length_penalty` | Non-negligible (visible on wandb chart) |
| TPCA (post-training eval) | Budget-aware < baseline |
| avg_tokens_easy (post-training eval) | Budget-aware << baseline |
| avg_tokens_hard (post-training eval) | Budget-aware ~= baseline |

**Simplicity criterion**: All else being equal, simpler is better. A small improvement from ugly complexity is not worth it. Removing code and getting equal results is a win.

## Training commands

**Budget-Aware DPO** (the experiment — always run):
```bash
pkill -f keep_alive.py  # kill keep-alive first
CUDA_VISIBLE_DEVICES=1 DATASET_VARIANT=balanced nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/budget_aware_balanced \
  --max-epochs 3 \
  --batch-size 4 \
  --lr 1e-5 \
  --lambda-easy 5.0 \
  --lambda-hard 0.0 \
  --early-stopping-patience 3 \
  --run-name budget_aware_balanced_N \
  --wandb \
  > logs/budget_aware_balanced.log 2>&1 &
```

**Baseline DPO** (only re-run if changing shared hyperparameters — see iteration0.md Section 11 for the decision table):
```bash
CUDA_VISIBLE_DEVICES=0 DATASET_VARIANT=balanced nohup .venv/bin/python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_balanced \
  --max-epochs 3 \
  --batch-size 4 \
  --lr 1e-5 \
  --early-stopping-patience 3 \
  --run-name baseline_balanced_N \
  --wandb \
  > logs/baseline_balanced.log 2>&1 &
```

Replace `N` with the iteration number (e.g., `budget_aware_balanced_4`).

**Kill training:**
```bash
pkill -f 'train_baseline_dpo|train_budget_aware_dpo'
```

**Start keep-alive when GPUs are idle:**
```bash
nohup python keep_alive.py > logs/keep_alive.log 2>&1 &
```

## Monitoring a run

Training runs log to `logs/baseline_balanced.log` and `logs/budget_aware_balanced.log`. Key commands:

```bash
# Check current step
grep -oP 'step=\d+' logs/budget_aware_balanced.log | tail -1

# Check epoch summaries
grep "Epoch.*val_loss" logs/budget_aware_balanced.log

# Check if training is still running
ps aux | grep 'train_baseline\|train_budget' | grep -v grep

# WandB: metrics appear at https://wandb.ai/ariksheer-tel-aviv-university/budget-aware-dpo
```

## Output format

After training completes, the key metrics come from two sources:

**1. Training logs (per-epoch)**:
```
Epoch N: train_loss=X.XXXX, val_loss=X.XXXX, reward_diff=X.XXXX
```

**2. WandB** (per-step and per-epoch): All metrics listed in `autoresearch/iteration0.md` Section 10.

**3. Post-training evaluation** (run manually on best checkpoint):
```bash
PYTHONPATH=. .venv/bin/python -m src.evaluation.run_evaluation \
  --checkpoint checkpoints/budget_aware_balanced/best_model \
  --output results/budget_aware_eval.json
```
This produces: accuracy, TPCA, avg_tokens_easy, avg_tokens_hard, math_level_4_5_accuracy.

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv` (tab-separated).

Header and columns:
```
iteration	run_name	val_loss	val_accuracy	reward_diff_easy	reward_diff_hard	length_penalty	lambda_easy	lambda_hard	other_changes	status	description
```

- `iteration`: iteration number (0, 1, 2, ...)
- `run_name`: wandb run name
- `val_loss`: best validation loss
- `val_accuracy`: validation accuracy (% preferring chosen) — 0.0 if not available
- `reward_diff_easy` / `reward_diff_hard`: epoch-end values — 0.0 if not available
- `length_penalty`: mean length penalty value — 0.0 for baseline
- `lambda_easy` / `lambda_hard`: hyperparameter values used
- `other_changes`: brief note on non-lambda changes (e.g., "lr=5e-6" or "grad_accum=4")
- `status`: `keep`, `discard`, or `crash`
- `description`: short text description

Example:
```
iteration	run_name	val_loss	val_accuracy	reward_diff_easy	reward_diff_hard	length_penalty	lambda_easy	lambda_hard	other_changes	status	description
0	baseline_balanced_3	0.8478	0.0	0.0	0.0	0.0	0.0	0.0	none	keep	standard DPO baseline
0	budget_aware_balanced_3	0.8230	0.0	0.0	0.0	0.0	0.05	0.001	none	discard	lambda too small - no divergence from baseline
1	budget_aware_balanced_4	0.0	0.0	0.0	0.0	0.0	5.0	0.0	none	keep	strong lambda - first visible divergence
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar26`).

LOOP FOREVER:

1. **Read previous iteration**: Read `autoresearch/iterationN-1.md` (or `iteration0.md` for the first run) to understand what was tried and what to do next.
2. **Plan the experiment**: Based on the previous iteration's "Next iteration plan" section, decide what to change. Focus on one variable at a time when possible.
3. **Make code changes**: Edit the in-scope files. The primary levers are:
   - `lambda_easy` and `lambda_hard` (CLI args or in loss function)
   - Learning rate, batch size, gradient accumulation
   - LoRA config (rank, alpha, dropout)
   - Loss function normalization (in `budget_aware_dpo_loss.py`)
   - Learning rate schedule (add warmup/decay in `dpo_trainer.py`)
   - Dataset changes (more data, different splits — via preprocessing scripts)
4. **git commit** the changes.
5. **Kill keep_alive.py** if running: `pkill -f keep_alive.py`
6. **Launch training** (see Training commands above). Use both GPUs if running baseline + budget-aware, or just GPU 1 for budget-aware only.
7. **Wait for training to complete** (~3-8 hours per epoch, 2-3 epochs). Monitor with:
   ```bash
   grep "Epoch.*val_loss" logs/budget_aware_balanced.log
   ```
8. **Collect results**: Extract epoch-level metrics from logs and wandb.
9. **Write `autoresearch/iterationN.md`** with all required sections (see Iteration Protocol below).
10. **Log to `autoresearch/results.tsv`**.
11. **Decide**: If the run shows improvement (more divergence from baseline, better val metrics), `keep`. Otherwise `discard` and `git reset` to previous good state.
12. **Start keep_alive.py** if GPUs will be idle before next run.
13. **Go to step 1** with the next iteration.

**Crashes**: If a run crashes, check `tail -50 logs/budget_aware_balanced.log`. Fix if trivial (typo, import). If fundamentally broken, log as `crash` and revert.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue working autonomously. If you run out of ideas, re-read `iteration0.md` suggestions, try combining previous near-misses, try more radical changes to the loss function. The loop runs until the human interrupts you.

**Between runs**: While waiting for a long training run, you can:
- Analyze previous results more deeply
- Write or update iteration documents
- Plan the next 2-3 experiments
- Run the evaluation pipeline on completed checkpoints

## Iteration Protocol

Each iteration produces `autoresearch/iterationN.md` containing:

1. **Hypothesis** — What are we testing and why? What specific change from the previous iteration?
2. **Hyperparameters** — Exact parameters used (full CLI invocation)
3. **Changes made** — Any code changes (file, line, what changed)
4. **Results** — Epoch-level metrics table (train_loss, val_loss, reward_diff, accuracy for all epochs), plus key wandb observations
5. **Comparison to baseline** — Side-by-side with baseline and previous best iteration
6. **Analysis** — Why did we see these results? Was the hypothesis confirmed?
7. **Open questions** — What remains unclear?
8. **Next iteration plan** — What to try next based on findings

## Evaluation checkpoints

After every 2-3 iterations (or when a run looks promising), run the full evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/<name>/best-model --limit 500 \
  --output eval_results/<name>.json --use-real
```

This gives the ground truth metrics: **accuracy**, **TPCA**, **avg_tokens_easy**, **avg_tokens_hard**. These are what ultimately determine if budget-aware DPO works.

## Phase 2 (starting iteration 7)

Phase 2 has 3 GPUs and 24 hours of autonomous experimentation. The primary goal is **shortening token count on easy problems**. DPO must be part of the approach. See `autoresearch/PHASE1_SUMMARY.md` for full hypotheses and `autoresearch/HANDOFF.md` for starting instructions.
