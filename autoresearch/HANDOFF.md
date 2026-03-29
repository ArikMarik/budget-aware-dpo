# Autoresearch Handoff — Phase 2

**Last updated**: 2026-03-28 09:05 UTC
**Phase**: Phase 2 — autonomous experimentation with 3 GPUs for 24 hours
**Branch**: `autoresearch/mar26`

---

## Start Here

You are the Phase 2 agent. Read these files in order:
1. **This file** — current state and what to do first
2. **`autoresearch/PHASE1_SUMMARY.md`** — full Phase 1 findings, hypotheses A-G, technical reference, and Phase 2 experiment plan
3. **`autoresearch/RULES.md`** — operational rules (polling, GPU assignment, commit protocol, etc.)
4. **`program.md`** — master experiment protocol (training commands, eval commands, monitoring)

---

## Current State

- **All 3 GPUs are FREE** (no processes running)
- Phase 1 is complete (6 iterations, all evaluated)
- Last iteration number: **6**. Start Phase 2 from **iteration 7**.

---

## Phase 1 Bottom Line

The length penalty (lambda_easy=5.0) produces **marginal** improvement over KL-only baseline:
- **2 fewer tokens** on easy problems (177 vs 179)
- **5% better TPCA** (846 vs 891)
- **Same accuracy** (22% vs 21.2%)

This is real but too small for a paper. Phase 2 needs a **stronger mechanism** to shorten easy tokens.

---

## Phase 2 Goals

**Primary**: Shorten token count on easy problems (avg_tokens_easy ↓↓).
**Secondary**: Maintain or improve accuracy. Hard problem improvement is a bonus.
**Constraint**: DPO must be part of the approach.

---

## Recommended First Actions

1. **Read PHASE1_SUMMARY.md** (especially Sections 6-9) for full hypothesis details
2. **Pick 2-3 hypotheses** to test in parallel on 3 GPUs
3. **Document your plan** in `autoresearch/iteration7.md` before launching
4. **Launch experiments** on all 3 GPUs

### Top hypotheses (see PHASE1_SUMMARY Section 6 for details):

| ID | Hypothesis | Expected Impact | Effort |
|----|-----------|----------------|--------|
| A | Per-token penalty (not per-sequence) | High — directly shapes generation | Medium (code change) |
| B | Larger model (1.5B) | High — more room to show budget effect | Low (download + same code) |
| D | SimPO (length-normalized DPO, no ref model) | High — built-in length control | Medium (new loss function) |
| E | Two-phase training (accuracy then efficiency) | Medium | Low (run sequentially) |
| C | More/better hard data | Low for our goal | Medium |

### Suggested Block 1 (hours 0-8, parallel on 3 GPUs):
- **GPU 0**: 0.5B + SimPO (new loss, no reference model, built-in length normalization)
- **GPU 1**: 0.5B + stronger per-token penalty (modify `budget_aware_dpo_loss.py`)
- **GPU 2**: 1.5B baseline DPO (download Qwen2.5-1.5B, establish new baseline)

---

## Available Checkpoints (for reference, don't overwrite)

| Checkpoint | Description | Post-train Accuracy | Easy Tokens |
|-----------|-------------|--------------------:|------------:|
| `budget_aware_balanced_iter5/` | Best budget (λ=5.0, KL=0.01) | 22.0% | 177.4 |
| `baseline_kl_iter6/` | Best baseline (KL=0.01 only) | 21.2% | 179.4 |

---

## Operational Notes

- Use `.venv/bin/python` (NOT system python)
- Set `PYTHONUNBUFFERED=1` for all training/eval
- Set `PYTHONPATH=/storage/arik/nlp_final_project` for eval scripts
- Eval command: `scripts/eval_checkpoint.py --use-real --limit 500`
- Kill `keep_alive.py` before training, start when idle
- Poll every 20 min, /compact at 200K context
- Commit after each completed iteration
