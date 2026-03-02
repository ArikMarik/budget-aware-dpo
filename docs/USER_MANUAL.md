# Budget-Aware DPO — User Manual for Project Partners

A practical guide to the project: structure, scripts, conventions, and how to work with Cursor agents.

---

## 1. Project Overview

**Goal:** Train a small math model (Qwen-2.5-0.5B) with **Budget-Aware DPO** — a custom loss that encourages short answers for easy problems and full chain-of-thought for hard ones.

**Status:** Phases 0–8 complete. Dummy pipeline works end-to-end. Real data loaded and trained. Phases 9–11 (evaluation, visualization, final report) pending.

---

## 2. Project Structure

```
nlp_final_project/
├── .cursorrules          # Project rules (Cursor agents follow these)
├── .venv/                # Python virtual environment (activate before running)
├── data/                 # Data files (dummy + real)
│   ├── dummy_openmathinstruct.jsonl
│   ├── processed_dpo_dataset/      # Dummy DPO pairs
│   ├── real_openmathinstruct.jsonl # Real (gitignored)
│   └── processed_dpo_dataset_real/ # Real DPO pairs (gitignored)
├── checkpoints/          # Model checkpoints (gitignored)
├── docs/
│   ├── USER_MANUAL.md    # This file
│   ├── feature_reports/  # Phase reports (report_phaseN_*.md)
│   ├── misc/             # Misc docs (e.g. knowledge_distillation_vs_dpo.md)
│   └── GIT_SETUP.md
├── scripts/              # Runnable scripts
│   ├── generate_dummy_data.py
│   ├── check_model_load.py
│   ├── preprocess_dpo_data.py
│   ├── train_sanity_check.py
│   ├── load_real_data.py
│   ├── run_evaluation.py
│   ├── run_visualization.py
│   ├── inspect_sanity_outputs.py
│   ├── run_all_dummy.sh
│   └── training/
│       ├── train_baseline_dpo.py
│       └── train_budget_aware_dpo.py
├── src/
│   ├── config.py         # Paths, USE_DUMMY_DATA, model name
│   ├── utils.py          # set_seed, etc.
│   ├── data/preprocessing.py
│   ├── models/
│   │   ├── budget_aware_dpo_loss.py
│   │   └── standard_dpo_loss.py
│   ├── training/dpo_trainer.py
│   ├── evaluation/
│   └── visualization/
├── reports/figures_dummy/  # PDF figures from dummy run
└── implementation_plan.md  # Full phase plan (reference)
```

---

## 3. Environment Setup

```bash
# Clone and enter project
cd nlp_final_project

# Activate virtual environment (ALWAYS do this before running scripts)
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# Set PYTHONPATH so scripts can import src
export PYTHONPATH="$PWD:$PYTHONPATH"

# Optional: use cluster storage for data (avoid filling home quota)
export DATA_PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/<your_username>"
export CHECKPOINT_DIR="$DATA_PATH/checkpoints"
```

---

## 4. Dummy vs Real Data

| Mode | Env var | Data | Checkpoints |
|------|---------|------|-------------|
| Dummy | `USE_DUMMY_DATA=1` | 50 synthetic examples | `baseline_dpo/`, `budget_aware_dpo/` |
| Real | `USE_DUMMY_DATA=0` | OpenMathInstruct-2 | `baseline_dpo_real/`, `budget_aware_dpo_real/` |

All scripts respect `USE_DUMMY_DATA`. Set it before running.

---

## 5. Available Scripts

### Run Everything (Dummy Only)

```bash
USE_DUMMY_DATA=1 ./scripts/run_all_dummy.sh
```

Runs: generate dummy data → model load check → preprocessing → sanity check (overfitting). ~2–5 min.

---

### Individual Scripts

| Script | Purpose | Dummy | Real |
|--------|---------|-------|------|
| `generate_dummy_data.py` | Create 50 synthetic examples | ✓ | — |
| `check_model_load.py` | Verify Qwen-2.5-0.5B loads | ✓ | ✓ |
| `load_real_data.py` | Load OpenMathInstruct-2 from HuggingFace | — | ✓ |
| `preprocess_dpo_data.py` | Build DPO pairs (4-way augmentation) | ✓ | ✓ |
| `train_sanity_check.py` | Overfit on 30 pairs (sanity check) | ✓ | — |
| `train_baseline_dpo.py` | Train standard DPO | ✓ | ✓ |
| `train_budget_aware_dpo.py` | Train budget-aware DPO | ✓ | ✓ |
| `run_evaluation.py` | Evaluate checkpoints (accuracy, TPCA) | ✓ | ✓ |
| `run_visualization.py` | Generate figures | ✓ | ✓ |
| `inspect_sanity_outputs.py` | Inspect sanity checkpoint outputs | ✓ | — |

---

### Example Commands

```bash
# Dummy pipeline (quick)
USE_DUMMY_DATA=1 python scripts/generate_dummy_data.py
USE_DUMMY_DATA=1 python scripts/preprocess_dpo_data.py
USE_DUMMY_DATA=1 python scripts/train_sanity_check.py

# Real data pipeline
python scripts/load_real_data.py --split train_1M --limit 5000
USE_DUMMY_DATA=0 python scripts/preprocess_dpo_data.py
USE_DUMMY_DATA=0 python scripts/training/train_baseline_dpo.py --max-steps 1000
USE_DUMMY_DATA=0 python scripts/training/train_budget_aware_dpo.py --max-steps 1000

# Evaluation
USE_DUMMY_DATA=1 python scripts/run_evaluation.py --dummy
USE_DUMMY_DATA=0 python scripts/run_evaluation.py

# Visualization
python scripts/run_visualization.py --dummy
python scripts/run_visualization.py --output-dir reports/figures
```

---

## 6. Conventions Used

### Code

- **Seeds:** All scripts call `set_seed(42)` at the start.
- **Environment:** Always run inside `.venv`. Never install system-wide.
- **Imports:** Use `from src.X import Y`. Set `PYTHONPATH` to project root.

### Data

- **Paths:** Configurable via `DATA_PATH`, `CHECKPOINT_DIR` env vars.
- **Format:** JSONL for datasets. Each line is a JSON object.

### Reports

- **Location:** `docs/feature_reports/report_phaseN_<feature>_<date>.md`
- **Naming:** Reports include `_dummy` or `_real` suffix when data type matters.
- **Appendix:** Each report has "Appendix: Problems Encountered and Solutions" (per `.cursorrules`).

### Git

- **Commits:** One commit per logical change. Clear messages (e.g. `feat: ...`, `docs: ...`).
- **Ignored:** `.venv/`, `checkpoints/`, large data files (see `.gitignore`).

---

## 7. Working with Cursor Agents

### Project Rules (`.cursorrules`)

Cursor agents are instructed to:

1. **Environment:** Always use `.venv`, never install system-wide.
2. **Data paths:** Route data to cluster storage when configured.
3. **Reproducibility:** Fix random seeds in all scripts.
4. **Reporting:** Generate a report after each phase.
5. **Report appendix:** Include "Problems Encountered and Solutions" in phase reports.
6. **Git:** Commit after each tested step; one commit per logical change.
7. **Documentation:** Keep a "white paper" focus; avoid dead-end work logs.

### How to Communicate with Cursor

- **Ask mode:** "Explain X", "What does Y do?", "Is Z implemented?" — no edits.
- **Agent mode:** "Implement X", "Fix Y", "Add Z" — agent can edit files and run commands.
- **Reference files:** Use `@filename` to point the agent at specific files.
- **Reference docs:** `@implementation_plan.md`, `@docs/USER_MANUAL.md` for context.

### Useful Prompts

- "Add a new script for X and report in docs/feature_reports."
- "Run the dummy pipeline and report any errors."
- "Explain the budget-aware DPO loss and where it's used."
- "Update the implementation plan and feature reports for Phase N."

---

## 8. What Was Done So Far

| Phase | Status | Summary |
|-------|--------|---------|
| 0 | ✅ | Virtual env, `.cursorrules`, cowsay test |
| 1 | ✅ | Dummy data (50 examples), model load check, `run_all_dummy.sh` |
| 2 | ✅ | 4-way augmentation, complexity classification, DPO pairs |
| 3 | ✅ | Budget-aware DPO loss, LoRA r=128, sanity check run |
| 4 | ✅ | Baseline + budget-aware training scripts |
| 5 | ✅ | Evaluation (accuracy, TPCA) |
| 6 | ✅ | Visualization (histograms, results table) |
| 7 | ✅ | Real data (OpenMathInstruct-2), synthetic DPO pairs |
| 8 | ✅ | Full training on real data (short run verified) |
| 9 | ⏳ | Evaluation on real data (GSM8K, MATH) |
| 10 | ⏳ | Visualization from real data |
| 11 | ⏳ | Final ACL report |

---

## 9. Key Concepts

- **Budget-Aware DPO:** DPO loss + length penalty. Easy problems → shorter answers; Hard problems → full CoT.
- **Complexity (C):** C=0 Easy (GSM8K-like), C=1 Hard (MATH-like). Based on `problem_source` or token count.
- **DPO pairs:** (problem, chosen, rejected, complexity). Chosen = preferred; Rejected = dispreferred.
- **Reference model:** Frozen copy of base model. Used for log-ratios in DPO, not knowledge distillation.

See `docs/misc/knowledge_distillation_vs_dpo.md` for clarification on DPO vs distillation.

---

## 10. Quick Start

```bash
# 1. Setup
cd nlp_final_project
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# 2. Run dummy pipeline (fast)
USE_DUMMY_DATA=1 ./scripts/run_all_dummy.sh

# 3. Inspect sanity outputs
USE_DUMMY_DATA=1 python scripts/inspect_sanity_outputs.py

# 4. Run evaluation (optional)
USE_DUMMY_DATA=1 python scripts/run_evaluation.py --dummy
python scripts/run_visualization.py --dummy
```

---

## 11. Troubleshooting

- **ModuleNotFoundError: No module named 'src'** → Set `PYTHONPATH="$PWD:$PYTHONPATH"`.
- **FileNotFoundError: processed dataset** → Run `preprocess_dpo_data.py` first.
- **CUDA out of memory** → Reduce `--batch-size` or use `--data-limit`.
- **Real data not found** → Run `load_real_data.py` before preprocessing with `USE_DUMMY_DATA=0`.

---

## 12. Further Reading

- `implementation_plan.md` — Full phase plan and senior engineer tips
- `docs/feature_reports/` — Phase reports with implementation details
- `docs/misc/knowledge_distillation_vs_dpo.md` — DPO vs distillation
- `docs/GIT_SETUP.md` — Git setup
