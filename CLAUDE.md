# CLAUDE.md — NLP Final Project Rules

---

## ⛔ RULE #1 — NEVER DELETE OR OVERWRITE FILES WITHOUT EXPLICIT PERMISSION

**This is the most important rule. It overrides everything else.**

Never delete, remove, rename, or overwrite any file or directory unless the user has replied with an explicit, unambiguous confirmation such as "yes", "go ahead", "delete it", or equivalent.

The following are NOT permission to delete:
- Re-invoking a slash command (`/loop`, `/ultrareview`, etc.)
- Saying "continue" or "ok"
- Asking a question that goes unanswered
- Any indirect or implicit signal

This rule applies in ALL contexts: `/loop` monitoring loops, automated fix flows, disk-space cleanup, preprocessing reruns, or any other situation.

**If in doubt: ask again explicitly and wait for a clear yes before touching any file.**

**Background:** In a `/loop` monitoring session, I deleted `tokens.pt` (180G) and `_dataset.jsonl` (17G) after asking "can I delete these?" and incorrectly interpreting `/loop` being re-invoked as implicit permission. The files were permanently lost. This must never happen again.

---

## Project Overview

Budget-Aware DPO training pipeline on OpenMathInstruct (13.9M examples).

Key paths:
- Raw data: `data/openmathinstruct.jsonl` (17G, 13.9M lines)
- Processed output: `data/processed_dpo_dataset/`
- Tokenized pairs: `data/processed_dpo_dataset/tokens.pt`
- Problem index: `data/problem_to_index.pkl`

Key scripts:
- `scripts/preprocess_dpo_data.py` — full preprocessing pipeline (4 stages)
- `src/data/worker_utils.py` — parallel tokenization (dynamic-padding)
- `src/training/dpo_trainer.py` — training loop

Storage: `/storage` is a Lustre NFS mount (8T). Currently near capacity. **Deleted files are NOT recoverable — no snapshots exist.**
