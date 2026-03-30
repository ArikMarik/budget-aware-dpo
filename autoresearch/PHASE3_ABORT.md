# Phase 3 (Budget-Aware Token Optimization) — ABORTED

**Date**: 2026-03-29
**Decision**: User aborted Phase 3 mid-iteration 10.

## Reason

Accuracies are unsatisfactory. The best held-out results after 10 iterations:
- 0.5B budget: 22.0% overall, 29.2% easy (iter5)
- 1.5B budget: 25.8% overall (iter7)
- Base Qwen2.5-0.5B reportedly gets ~44% on GSM8K with no fine-tuning

DPO training is producing models **worse than the base model** on easy problems. Before optimizing token count, we need to understand why and establish proper baselines.

## State at Abort

- **Iteration 10a** (λ=10, 6 epochs): E3 complete, 30% in-training acc but 256 max tokens
- **Iteration 10b** (baseline 6ep): completely failed (1-3% acc)
- **Iteration 10c** (λ=20): launched but not evaluated
- No post-training eval run for any iter 10 model

## What Carries Forward

- All evaluation infrastructure (tiered eval with LLM judge) is solid
- Dataset (balanced_v4_capped, 50K pairs) is ready
- GSM8K test (1,319 problems) and MATH test (5,000 problems) are available
- Key insight: budget-aware lambda acts as a learning signal, not just length penalty

## Next Phase

Phase 4: Baseline establishment + SFT accuracy push. See `PHASE4_PRD.md`.
