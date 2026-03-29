# Autoresearch Agent Rules

Rules and operational knowledge for any agent running the autoresearch experiment loop.

---

## 1. Permissions & Autonomy

- **Auto-approve all bash commands** for polling, training launch, log checking, and GPU monitoring. The agent should not need to ask the user for permission on routine operations.
- **Phase 2 mode: FULLY AUTONOMOUS.** The user will NOT be guiding. Make all decisions independently. Document everything. Commit frequently. /compact when context exceeds 200K tokens.
- **Only notify the user when**:
  - A critical decision needs to be made that fundamentally changes the project direction
  - A crash occurs that can't be auto-recovered
- **Do NOT notify the user** for routine polling, experiment results, or iteration transitions. Just keep going.

## 2. Server Behavior

- **Startup delay**: Training processes take up to **10 minutes** before producing any log output. This is normal. Do not kill or restart processes during this window.
- **DataLoader workers**: When training runs with `num_workers>0`, there will be multiple python processes with the same command line (typically 4 workers + 1 main). These are **child processes of the main training process**. **NEVER kill them** — doing so crashes the training run. Only the main process (lowest PID, highest CPU%) is the one to monitor.
- **Log buffering**: Use `PYTHONUNBUFFERED=1` when launching training to avoid empty log files.

## 3. Polling & Token Efficiency

- **Poll every 20 minutes** during training. Do NOT poll more frequently.
- **Between polls, do nothing** — no repeated file reads, no "quick peeks." Wait for the background task notification.
- **Do NOT repeatedly read background task output files** in a loop. Set a background sleep, get notified, read once.
- **One background sleep at a time** — don't stack multiple overlapping timers.
- **/compact when context exceeds 200K tokens.** This is critical for 24-hour runs.

## 4. Training Operations

- **Kill keep_alive.py before training**: `pkill -f keep_alive.py`
- **Start keep_alive.py when GPUs idle**: `nohup .venv/bin/python keep_alive.py > logs/keep_alive.log 2>&1 &`
- **Always use `.venv/bin/python`** — system python doesn't have torch installed.
- **Always set `PYTHONUNBUFFERED=1`** in the nohup command for training.
- **Always set `PYTHONPATH=/storage/arik/nlp_final_project`** for eval scripts.
- **3 GPUs available** (GPU 0, 1, 2). Keep all 3 busy when possible.
- **GPU assignment**: Flexible in Phase 2. Use any GPU for any run. Track which GPU runs which experiment.

## 5. Primary Goal

**Shorten token count on easy problems while maintaining or improving accuracy.** Hard problem improvement is a nice-to-have bonus, NOT a priority.

Success metrics (in order of importance):
1. `avg_tokens_easy` ↓ (budget-aware << baseline)
2. `easy_accuracy` ≥ baseline (no accuracy sacrifice)
3. `TPCA` ↓ (overall efficiency)
4. `hard_accuracy` ≥ baseline (don't regress)

**Constraint**: DPO must be part of the approach (can be modified DPO variants like SimPO).

## 6. Experiment Analysis

- **Post-training eval is ground truth.** Always use `scripts/eval_checkpoint.py` with `--use-real` for final numbers.
- **In-training gen eval** is for early detection and relative comparison only.
- **Data quality/quantity** is a concern. If 2-3 iterations show no improvement, try data pipeline changes.
- **Iteration speed**: Full runs take 6-8 hours on 0.5B. Budget time accordingly with 1.5B models (~1.5x longer).
- **Gradient instability**: lambda_easy=5.0 causes high gradient norms. Grad clipping at max_norm=1.0 handles this.

## 7. File Conventions

- **Iteration docs**: `autoresearch/iterationN.md` — one per experiment
- **Results log**: `autoresearch/results.tsv` — append after each experiment
- **Training logs**: `logs/<run_name>.log`
- **Checkpoints**: `checkpoints/<run_name>/`
- **Eval results**: `eval_results/<name>.json`
- **NEVER overwrite existing data/checkpoints/results**. Use new directories with descriptive names.

## 8. Pre-Experiment Checklist

Before launching a new experiment, check:
- [ ] HANDOFF.md updated with current state
- [ ] Previous iterationN.md is COMPLETE with all results
- [ ] New iteration(N+1).md written with hypothesis and changes BEFORE launching
- [ ] Any pending evals from previous iteration are done or running
- [ ] GPU assignment plan for all available GPUs

## 9. Documentation Rules

- **EVERY decision must be documented in iterationN.md BEFORE executing it.**
- Each iteration doc must explain: what changed, WHY it changed, what we expect, and risks.
- When multiple changes are made simultaneously, document each one separately.
- The user must be able to reconstruct every decision by reading the iteration docs alone.

## 10. Early Detection & Autonomous Action

During each poll, actively check for:
- **Stale learning**: Loss not moving
- **Model collapse**: Loss exploding, gibberish output, 256-token max gen
- **Too fast convergence**: Loss near zero early (memorization)
- **Weird values**: NaN, inf in loss or gradients

If detected: **kill the run immediately**, document findings, start next iteration with fixes.

## 11. Commit & Compact Protocol

- **Commit after each completed iteration** with descriptive messages.
- **/compact when context approaches 200K tokens.** Before compacting, ensure:
  - All current state is written to HANDOFF.md
  - Current iteration doc is up to date
  - Any in-progress plans are documented
- After /compact, re-read HANDOFF.md and RULES.md to restore context.

## 12. Key References

- `autoresearch/PHASE1_SUMMARY.md` — Full Phase 1 findings, hypotheses, and Phase 2 plan
- `autoresearch/iteration[0-6].md` — Per-iteration docs
- `src/models/budget_aware_dpo_loss.py` — Loss function (primary lever)
- `src/training/dpo_trainer.py` — Training loop
- `scripts/training/train_budget_aware_dpo.py` — CLI for training
- `scripts/eval_checkpoint.py` — Post-training eval (use with PYTHONPATH)
- `src/config.py` — Paths and model config
- `src/evaluation/run_evaluation.py` — Evaluation internals (read-only)

## 13. Eval Command Reference

```bash
# Post-training eval (500 problems, Tier 0+1+2, held-out test)
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  nohup .venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint checkpoints/<name>/best-model --limit 500 \
  --output eval_results/<name>.json --use-real \
  > logs/eval_<name>.log 2>&1 &

# Training
CUDA_VISIBLE_DEVICES=<gpu> DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
  PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/<name> --max-epochs 3 --batch-size 4 --lr 1e-6 \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --early-stopping-patience 3 --run-name <name> --wandb \
  > logs/<name>.log 2>&1 &
```
