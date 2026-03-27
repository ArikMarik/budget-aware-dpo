# Autoresearch Agent Rules

Rules and operational knowledge for any agent running the autoresearch experiment loop.

---

## 1. Permissions & Autonomy

- **Auto-approve all bash commands** for polling, training launch, log checking, and GPU monitoring. The agent should not need to ask the user for permission on routine operations.
- **Only notify the user when**:
  - An experiment completes (all epochs done or early stopping triggered)
  - A critical decision needs to be made (e.g., should we stop a run early, change strategy)
  - A crash occurs that can't be auto-recovered
- **Do NOT notify the user** for routine polling, status checks, or intermediate progress.

## 2. Server Behavior

- **Startup delay**: Training processes take up to **10 minutes** before producing any log output. This is normal. Do not kill or restart processes during this window.
- **DataLoader workers**: When training runs with `num_workers>0`, there will be multiple python processes with the same command line (typically 4 workers + 1 main). These are **child processes of the main training process**. **NEVER kill them** — doing so crashes the training run. Only the main process (lowest PID, highest CPU%) is the one to monitor.
- **Log buffering**: Use `PYTHONUNBUFFERED=1` when launching training to avoid empty log files.

## 3. Polling & Token Efficiency

- **Poll every 20 minutes** during training. Do NOT poll more frequently.
- **Between polls, do nothing** — no repeated file reads, no "quick peeks." Wait for the background task notification.
- **Do NOT repeatedly read background task output files** in a loop. Set a background sleep, get notified, read once.
- **One background sleep at a time** — don't stack multiple overlapping timers.

## 4. Training Operations

- **Kill keep_alive.py before training**: `pkill -f keep_alive.py`
- **Start keep_alive.py when GPUs idle**: `nohup .venv/bin/python keep_alive.py > logs/keep_alive.log 2>&1 &`
- **Always use `.venv/bin/python`** — system python doesn't have torch installed.
- **Always set `PYTHONUNBUFFERED=1`** in the nohup command for training.
- **GPU assignment**: Budget-aware on GPU 1, baseline on GPU 0.
- **Only re-run baseline** if shared hyperparameters change (lr, batch_size, epochs, beta, LoRA config, dataset). Lambda changes do NOT require a baseline re-run.

## 5. Experiment Analysis

- **Validation accuracy** has been poor in previous experiments. Always report it prominently and factor it into decisions.
- **Data quality/quantity** is a concern. If 2-3 iterations show no improvement, suggest data pipeline changes (more data, different splits, quality filtering).
- **Iteration speed**: Full runs take 6-8 hours. Consider using `--data-limit` for quick smoke tests before committing to full runs. Propose speedup strategies between experiments.
- **Gradient instability**: lambda_easy=5.0 causes high gradient norms (10-100+, with occasional inf/nan). Grad clipping at max_norm=1.0 handles this, but watch for loss divergence.

## 6. File Conventions

- **Iteration docs**: `autoresearch/iterationN.md` — one per experiment
- **Results log**: `autoresearch/results.tsv` — append after each experiment
- **Training logs**: `logs/budget_aware_balanced.log`, `logs/baseline_balanced.log`
- **Checkpoints**: `checkpoints/budget_aware_balanced/`, `checkpoints/baseline_balanced/`
- **Run names**: `budget_aware_balanced_N` / `baseline_balanced_N` where N = iteration number
- **NEVER overwrite existing data/checkpoints/results**. When creating new datasets or checkpoints, use new directories with descriptive names (e.g., `data/processed_dpo_dataset_balanced_v2_capped50/`, `checkpoints/budget_aware_balanced_iter2/`). Keep all previous artifacts intact for comparison.

## 7. Pre-Experiment Checklist

Before launching a new experiment, check:
- [ ] Any pending fixes (e.g., WandB metrics, logging changes)
- [ ] HANDOFF.md "Before next experiment" section
- [ ] Memory files for user requests
- [ ] iterationN.md for previous iteration is COMPLETE with all results and final conclusions
- [ ] iteration(N+1).md is written with hypothesis, all changes, and rationale BEFORE launching

## 8. Documentation Rules

- **EVERY decision must be documented in iterationN.md BEFORE executing it.** Never jump from results to launching without writing the doc first.
- Each iteration doc must explain: what changed, WHY it changed, what we expect, and what the risks are.
- When multiple changes are made simultaneously, document each one separately with its rationale.
- If a decision turns out wrong mid-iteration, document the correction and why.
- The user must be able to reconstruct every decision by reading the iteration docs alone.

## 8. Key References

- `program.md` — master experiment protocol
- `autoresearch/iteration0.md` — baseline results, diagnosis, and suggestions
- `src/models/budget_aware_dpo_loss.py` — the loss function (primary lever)
- `src/training/dpo_trainer.py` — training loop
- `scripts/training/train_budget_aware_dpo.py` — CLI for budget-aware training
- `src/config.py` — paths and model config
- `src/evaluation/run_evaluation.py` — ground truth evaluation (read-only)
