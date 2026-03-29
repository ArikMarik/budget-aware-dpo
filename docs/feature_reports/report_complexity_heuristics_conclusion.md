# Conclusion Report: Recommended Complexity Heuristics

**Date:** 2026-03-11  
**Purpose:** Synthesize recommendations from the complexity heuristics analysis and next-update considerations into a single, actionable conclusion.

---

## Executive Summary

The analysis of 50,000 training examples (81.5% MATH, 18.5% GSM8K) and the review of preprocessing considerations lead to a unified set of recommendations for complexity classification. The key changes are: **(1)** replace word count with tiktoken-based token counts, **(2)** adopt level-based classification for MATH when available, **(3)** use empirically derived token thresholds, and **(4)** treat CoT indicators as an optional secondary signal.

---

## 1. Recommended Complexity Heuristics

### 1.1 Primary: MATH Level (When Available)

Use the MATH dataset’s 5-level difficulty scale (AoPS) as the primary signal when `level` is present:

| MATH Level | Complexity (C) | Rationale |
|------------|----------------|-----------|
| Level 1–2  | Easy (C=0)     | Easiest problems; avg tokens 168–214 |
| Level 3    | Token fallback  | Medium; use P25/P75 token thresholds |
| Level 4–5  | Hard (C=1)     | Hardest problems; avg tokens 293–346 |

**Caveat:** `level` exists only for MATH problems. For GSM8K or other sources without level, use token-based heuristics.

---

### 1.2 Token-Based Thresholds (tiktoken cl100k_base)

Replace `approx_tokens(text) = len(text.split())` with tiktoken `cl100k_base` (GPT-4/Claude compatible).

| Parameter              | Current (word count) | Recommended (tiktoken) | Notes |
|------------------------|----------------------|------------------------|-------|
| EASY_TOKEN_THRESHOLD   | 50                   | **70**                 | Below = short/direct answer |
| HARD_TOKEN_THRESHOLD   | 80                   | **130**                | Above = long CoT-style reasoning |

**Rationale:** Training data shows GSM8K P25 ≈ 106, P50 ≈ 134; MATH Level 1 P50 ≈ 151. The 70/130 thresholds separate short vs. long reasoning while staying compatible with both distributions.

---

### 1.3 Percentile-Based Alternative (Per-Dataset)

For dataset-specific tuning, use training percentiles:

| Category | Easy (below) | Hard (above) |
|----------|--------------|--------------|
| MATH     | P25 ≈ 185    | P75 ≈ 427    |
| GSM8K    | P25 ≈ 106    | P75 ≈ 175    |

**Implementation:** Compute percentiles on the training set; use them when preprocessing that dataset. Make thresholds configurable (config or env).

---

### 1.4 CoT Indicators (Optional Secondary Signal)

CoT patterns (e.g., "First", "Step", "Therefore", "Thus", "Hence", "So we", "Let me think", "In summary", "Consequently") correlate weakly with difficulty:

- MATH avg CoT count: 1.7; GSM8K: 1.5
- Level 1 avg: 0.9; Level 5 avg: 1.4

**Recommendation:** Use as a secondary signal only, not as a replacement for token count or level.

---

## 2. Decision Logic (Recommended Flow)

```
1. If MATH and level is present:
   - Level 1–2 → C=0 (Easy)
   - Level 4–5 → C=1 (Hard)
   - Level 3 → go to step 2

2. Compute teacher_token_count = len(tiktoken_cl100k_base.encode(teacher_answer))

3. If teacher_token_count < EASY_TOKEN_THRESHOLD (70):
   → C=0 (Easy)
   Else if teacher_token_count > HARD_TOKEN_THRESHOLD (130):
   → C=1 (Hard)
   Else:
   → C=0.5 or treat as medium (implementation-dependent)
```

---

## 3. Implementation Checklist

| Task | Priority | Notes |
|------|----------|-------|
| Replace `approx_tokens` with tiktoken `cl100k_base` | High | Use `len(tokenizer.encode(text))` |
| Add `level` to `load_math_test()` output | Done | Per next_update_considerations |
| Use level-based classification when `level` present | High | Level 1–2 → Easy, 4–5 → Hard |
| Set EASY_TOKEN_THRESHOLD=70, HARD_TOKEN_THRESHOLD=130 | High | Make configurable |
| Make thresholds configurable (config/env) | Medium | For future tuning |
| Optional: per-dataset percentile thresholds | Low | For advanced use |
| Optional: CoT indicators as secondary signal | Low | Not required for initial rollout |

---

## 4. Summary Table

| Heuristic | Value | Scope |
|-----------|-------|-------|
| Tokenizer | tiktoken `cl100k_base` | All |
| EASY_TOKEN_THRESHOLD | 70 | When level unavailable or Level 3 |
| HARD_TOKEN_THRESHOLD | 130 | When level unavailable or Level 3 |
| MATH Level 1–2 | C=0 (Easy) | MATH only |
| MATH Level 3 | Token fallback | MATH only |
| MATH Level 4–5 | C=1 (Hard) | MATH only |
| CoT indicators | Optional secondary | All |

---

## Appendix: Problems Encountered and Solutions

This conclusion report synthesizes existing analysis; no new implementation was performed. The recommendations are derived from:

- **report_complexity_heuristics_analysis.md:** Empirical token and level statistics on 50,000 training examples.
- **docs/next_update_considerations.md:** Design considerations for tokenization, level usage, and threshold configurability.

No significant problems encountered during the synthesis.
