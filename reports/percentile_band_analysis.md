# Percentile-Band Analysis for DPO Preference Labeling

Source: `data/problem_index.json` — 606,922 problems. This report answers two questions:

1. Is the heuristic `MATH L1-2 → complexity=0`, `L4-5 → complexity=1` supported by the data-distribution of teacher-solution lengths?
2. Given the validated complexity pools, what per-problem percentile band should define `preferred` solutions?

## 1. Validating the complexity heuristic

Pooled per-solution token-count percentiles, plus comparison to two reference groups: **gsm8k** (canonical Easy) and **math_L5** (canonical Hard).

- `d_vs_*` = Cohen's d (pooled). Sign indicates direction (positive = longer than ref). |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
- `overlap_*` ∈ [0, 1] = fraction of probability mass shared with the reference (1 = identical, 0 = disjoint).
- `ks_*` = max |CDF difference| with the reference (0 = identical, 1 = disjoint).

| group | # problems | # solutions | mean | p10 | p25 | p50 | p75 | p90 | p95 | p99 | d vs gsm8k | d vs L5 | overlap gsm8k | overlap L5 | KS gsm8k | KS L5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 7,426 | 458,822 | 154 | 92 | 114 | 145 | 183 | 227 | 258 | 328 | +0.00 | -1.42 | 1.00 | 0.35 | 0.00 | 0.65 |
| augmented_gsm8k | 73,636 | 2,111,683 | 166 | 95 | 118 | 152 | 197 | 250 | 289 | 409 | +0.17 | -1.75 | 0.94 | 0.40 | 0.06 | 0.60 |
| math_L1 | 633 | 226,058 | 187 | 95 | 120 | 162 | 227 | 309 | 370 | 512 | +0.45 | -1.05 | 0.84 | 0.50 | 0.16 | 0.50 |
| math_L2 | 1,637 | 621,104 | 226 | 108 | 143 | 198 | 276 | 377 | 454 | 646 | +0.73 | -0.85 | 0.68 | 0.63 | 0.32 | 0.37 |
| math_L3 | 728 | 272,013 | 274 | 126 | 170 | 239 | 344 | 463 | 541 | 801 | +1.20 | -0.48 | 0.55 | 0.78 | 0.46 | 0.22 |
| math_L4 | 2,005 | 714,159 | 298 | 140 | 191 | 267 | 370 | 488 | 580 | 845 | +1.16 | -0.35 | 0.46 | 0.85 | 0.54 | 0.15 |
| math_L5 | 2,430 | 645,855 | 356 | 164 | 227 | 321 | 445 | 585 | 720 | 953 | +1.42 | +0.00 | 0.35 | 1.00 | 0.65 | 0.00 |
| augmented_math_L1 | 17,185 | 383,263 | 281 | 113 | 154 | 229 | 352 | 523 | 657 | 921 | +0.99 | -0.42 | 0.59 | 0.75 | 0.41 | 0.25 |
| augmented_math_L2 | 52,816 | 1,123,921 | 313 | 127 | 177 | 261 | 396 | 578 | 712 | 947 | +0.98 | -0.23 | 0.50 | 0.84 | 0.50 | 0.16 |
| augmented_math_L4 | 73,545 | 1,348,374 | 371 | 155 | 220 | 323 | 476 | 664 | 790 | 977 | +1.22 | +0.08 | 0.37 | 0.92 | 0.63 | 0.06 |
| augmented_math_L5 | 84,796 | 1,295,122 | 405 | 168 | 241 | 358 | 527 | 723 | 843 | 1001 | +1.34 | +0.24 | 0.32 | 0.89 | 0.68 | 0.11 |
| augmented_math_NoLevel | 290,083 | 4,771,477 | 367 | 145 | 208 | 313 | 478 | 679 | 808 | 986 | +1.05 | +0.05 | 0.40 | 0.89 | 0.60 | 0.06 |

## 2. Within-problem percentile stats by complexity pool

For each problem with ≥ 5 teacher solutions, compute its per-problem percentiles p5, p10, …, p95. Then across problems report the **median** (p50) absolute token count at each within-problem percentile, plus the p10/p90 across problems (i.e., variability from one problem to another).

### Complexity 0 (Easy) pool

| within-problem pct | problems | median abs tokens | p10 across probs | p90 across probs |
|---:|---:|---:|---:|---:|
| 5 | 192,322 | 131 | 82 | 264 |
| 10 | 192,322 | 141 | 87 | 284 |
| 15 | 192,322 | 148 | 91 | 301 |
| 20 | 192,322 | 155 | 94 | 316 |
| 25 | 192,322 | 161 | 98 | 330 |
| 30 | 192,322 | 166 | 101 | 344 |
| 40 | 192,322 | 177 | 106 | 371 |
| 50 | 192,322 | 188 | 112 | 399 |
| 60 | 192,322 | 199 | 117 | 430 |
| 70 | 192,322 | 212 | 123 | 466 |
| 75 | 192,322 | 218 | 127 | 488 |
| 80 | 192,322 | 227 | 131 | 513 |
| 85 | 192,322 | 237 | 135 | 545 |
| 90 | 192,322 | 249 | 140 | 587 |
| 95 | 192,322 | 268 | 148 | 650 |

### Complexity 1 (Hard) pool

| within-problem pct | problems | median abs tokens | p10 across probs | p90 across probs |
|---:|---:|---:|---:|---:|
| 5 | 314,806 | 262 | 142 | 470 |
| 10 | 314,806 | 286 | 159 | 500 |
| 15 | 314,806 | 305 | 173 | 527 |
| 20 | 314,806 | 322 | 185 | 551 |
| 25 | 314,806 | 338 | 196 | 574 |
| 30 | 314,806 | 354 | 207 | 596 |
| 40 | 314,806 | 383 | 227 | 639 |
| 50 | 314,806 | 413 | 246 | 683 |
| 60 | 314,806 | 446 | 265 | 729 |
| 70 | 314,806 | 483 | 286 | 781 |
| 75 | 314,806 | 505 | 297 | 808 |
| 80 | 314,806 | 531 | 311 | 839 |
| 85 | 314,806 | 562 | 326 | 872 |
| 90 | 314,806 | 603 | 345 | 908 |
| 95 | 314,806 | 660 | 372 | 950 |

### Per-atomic-group, median absolute tokens at within-problem pct

Each cell is the *median across problems* of that problem's within-problem percentile. A group where within-problem p50 is already small (e.g. 150) is a group where the typical teacher solution is short.

| group | pP5 | pP10 | pP15 | pP20 | pP25 | pP30 | pP40 | pP50 | pP60 | pP70 | pP75 | pP80 | pP85 | pP90 | pP95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k | 111 | 117 | 122 | 126 | 130 | 132 | 138 | 144 | 150 | 156 | 160 | 164 | 169 | 176 | 185 |
| augmented_gsm8k | 122 | 128 | 132 | 136 | 140 | 143 | 150 | 156 | 162 | 169 | 173 | 178 | 183 | 190 | 200 |
| math_L1 | 96 | 105 | 112 | 119 | 126 | 131 | 142 | 155 | 166 | 179 | 189 | 199 | 212 | 227 | 252 |
| math_L2 | 115 | 131 | 142 | 152 | 161 | 170 | 184 | 197 | 214 | 230 | 240 | 252 | 266 | 285 | 317 |
| math_L3 | 146 | 167 | 182 | 192 | 203 | 211 | 228 | 248 | 268 | 290 | 303 | 315 | 335 | 361 | 405 |
| math_L4 | 170 | 193 | 208 | 222 | 234 | 245 | 266 | 288 | 308 | 333 | 347 | 366 | 385 | 412 | 456 |
| math_L5 | 210 | 242 | 268 | 288 | 304 | 321 | 352 | 383 | 417 | 455 | 474 | 494 | 520 | 562 | 640 |
| augmented_math_L1 | 172 | 188 | 201 | 212 | 222 | 232 | 250 | 268 | 288 | 312 | 325 | 341 | 361 | 385 | 423 |
| augmented_math_L2 | 196 | 213 | 228 | 241 | 252 | 263 | 283 | 304 | 327 | 353 | 368 | 385 | 406 | 433 | 475 |
| augmented_math_L4 | 238 | 259 | 276 | 292 | 306 | 319 | 345 | 370 | 399 | 432 | 450 | 472 | 499 | 534 | 583 |
| augmented_math_L5 | 257 | 281 | 301 | 318 | 334 | 350 | 379 | 409 | 442 | 480 | 500 | 526 | 557 | 596 | 653 |
| augmented_math_NoLevel | 234 | 256 | 274 | 289 | 303 | 317 | 344 | 370 | 400 | 434 | 453 | 476 | 505 | 542 | 595 |

## 3. Candidate-band simulation on each complexity pool

For each candidate band we label every solution in every eligible problem (≥ 5 solutions) and aggregate: what % of correct solutions land in the band; what absolute token-count distribution the preferred, rejected-short, and rejected-long buckets have; and what fraction of problems end up with at least one preferred solution (low = band is so narrow that many problems produce no positive example).

### Complexity 0 (Easy) — candidate bands

| band | eligible probs | probs w/ any preferred (%) | preferred (%) | preferred med tok | pref p10/p90 | rej-short med | rej-long med |
|---|---:|---:|---:|---:|---:|---:|---:|
| [5, 30] | 192,322 | 100.0 | 24.7 | 138 | 87/239 | 106 | 196 |
| [5, 40] | 192,322 | 100.0 | 34.6 | 144 | 90/251 | 106 | 203 |
| [10, 30] | 192,322 | 99.9 | 19.8 | 143 | 90/247 | 115 | 196 |
| [10, 40] | 192,322 | 100.0 | 29.7 | 149 | 93/259 | 115 | 203 |
| [10, 50] | 192,322 | 100.0 | 40.5 | 154 | 96/269 | 115 | 211 |
| [15, 45] | 192,322 | 100.0 | 30.1 | 154 | 96/267 | 121 | 206 |
| [20, 50] | 192,322 | 100.0 | 30.9 | 160 | 100/280 | 126 | 211 |

### Complexity 1 (Hard) — candidate bands

| band | eligible probs | probs w/ any preferred (%) | preferred (%) | preferred med tok | pref p10/p90 | rej-short med | rej-long med |
|---|---:|---:|---:|---:|---:|---:|---:|
| [50, 90] | 314,806 | 100.0 | 41.5 | 405 | 239/707 | 270 | 586 |
| [55, 90] | 314,806 | 100.0 | 35.7 | 414 | 245/722 | 279 | 586 |
| [55, 95] | 314,806 | 100.0 | 40.3 | 424 | 250/743 | 279 | 648 |
| [60, 90] | 314,806 | 100.0 | 31.0 | 423 | 251/735 | 285 | 586 |
| [60, 95] | 314,806 | 100.0 | 35.6 | 434 | 256/756 | 285 | 648 |
| [65, 95] | 314,806 | 100.0 | 30.2 | 444 | 263/771 | 293 | 648 |
| [70, 95] | 314,806 | 99.9 | 25.3 | 455 | 270/787 | 299 | 648 |

## 4. Diagnostic: solutions-per-problem distribution

- complexity=0: problems=204,761 | mean=31.7 | p10=7 p50=29 p90=56 | max=2370
- complexity=1: problems=402,161 | mean=18.6 | p10=3 p50=11 p90=31 | max=9457

## 5. Rebalancing Hard: switching to **short-preferring, conservative**

### 5.1 Motivation

The §3 Hard-pool bands (all of [50, 90], [55, 90], [60, 90], [60, 95], ...) are *long-preferring*: preferred solutions sit above the median, rejected below. Combined with `lambda_hard ≈ 0.001` in `src/models/budget_aware_dpo_loss.py`, the training objective had **no counter-pressure on length** for Hard — the DPO signal alone pushed generations longer.

New goal:

- **Hard chosen = the shortest solution that still preserves essential reasoning.** Both Easy and Hard now push toward conciseness.
- **Hard gets its own active length penalty**, separate from `lambda_easy` and smaller than it, so Hard compresses less aggressively than Easy (constraint: don't degrade the model into a "guess").

### 5.2 Safety-floor analysis for Hard-short

Within-problem percentile → median absolute tokens, with p10-across-problems as the worst-case floor (from §2 Hard pool):

| within-prob pct | median abs tokens | p10 across probs | risk assessment |
|---:|---:|---:|---|
| 5  | 262 | 142 | Ultra-short; "lucky-short" risk. **Unsafe as a floor** — some problems have p5 as low as 142 tokens (≈ 1 reasoning step). |
| 10 | 286 | 159 | Still risky — p10 across probs is only 159. |
| 15 | 305 | 173 | Borderline-safe. |
| **20** | **322** | **185** | **Safe** — even the bottom-10% problem has 185 tokens of CoT at within-problem p20. |
| 25 | 338 | 196 | Comfortably safe but bites into the preferred bandwidth. |

**Conclusion — low edge = p20.** This rejects the degenerate bottom tail while preserving ~80% of the pool as eligible.

### 5.3 Upper-edge analysis for Hard-short

The upper edge must sit below the Hard median (413 tokens) so "preferred" is visibly shorter than "rejected":

| within-prob pct | median abs tokens | compression vs pool median (413) |
|---:|---:|---:|
| 35 | ~369 | 89% (11% reduction) |
| 40 | 383 | 93% (7%) |
| **45** | **~398** | **96% (4%)** |
| 50 | 413 | 100% — no shortening pressure |

**Conclusion — high edge = p45.** Leaves a 4% headroom below the pool median; combined with the p20 floor, the preferred mass sits at mean rank ≈ (20+45)/2 = 32.5 → median ≈ 360 tokens ≈ 87 % of Hard median.

### 5.4 Candidate-band simulation on the Hard pool (short-preferring)

Projecting §2 Hard-pool within-problem percentile → median token quantities onto several short-preferring bands (complement of the long-preferring bands in §3):

| band | coverage (~% preferred) | pref median abs | rej-short median (< low) | rej-long median (> high) | safety (> 90% probs have a preferred) | compression vs pool median |
|---|---:|---:|---:|---:|---:|---:|
| [5, 30]  | ~25% | ~330 | n/a (no floor) | ~413+ | yes | 80% |
| [10, 35] | ~25% | ~345 | ~286 | ~413+ | yes | 84% |
| [15, 40] | ~25% | ~355 | ~305 | ~446 | yes | 86% |
| **[20, 45]** | **~25%** | **~360** | **~305** | **~446** | **yes (~100%)** | **87%** |
| [20, 50] | ~30% | ~370 | ~305 | ~446 | yes | 90% *(too little pressure)* |
| [25, 50] | ~25% | ~375 | ~322 | ~446 | yes | 91% *(too little pressure)* |

**Chosen band: `[20, 45]`.** Tight enough to exert shortening pressure (87% of pool median), loose enough at the floor (p20 = 322 tokens median, worst-case 185) to avoid step-skipping.

### 5.5 Final recommended bands

| Flag | Old band (long-preferring) | New band (short-preferring, conservative) | Pref target median | Compression vs pool median |
|---|---|---|---:|---:|
| Easy (C=0) | `[10, 40]` | `[10, 40]` *(unchanged)* | ~150 tokens | ~82% of Easy median (~188) |
| Hard (C=1) | `[60, 92]` | **`[20, 45]`** | ~360 tokens | ~87% of Hard median (413) |

The Hard band is deliberately gentler than Easy (87% vs 82% compression; safety floor p20 vs p10) to satisfy the "don't skip essential reasoning" constraint.

### 5.6 Training-side changes

With chosen now short in **both** complexity flags, the existing penalty

```
length_penalty = lambda(C) · (chosen_len − rejected_len) / avg_len
```

(`src/models/budget_aware_dpo_loss.py:54-55`) immediately becomes productive for Hard — the term is negative (chosen shorter than rejected), so the reward difference gets a positive boost that reinforces the short chosen. The previous setting `lambda_hard = 0.001` was tuned for the Hard-prefers-long world and must be raised.

| Param | Old default | New default | Rationale |
|---|---:|---:|---|
| `lambda_easy` | 0.05 | 0.05 *(unchanged)* | Working value for Easy. |
| `lambda_hard` | 0.001 | **0.03** | Active Hard shortening pressure at ~60% of Easy — conservative per the no-step-skipping constraint. |

The pre-flip loss function and the pre-flip preference thresholds are **commented out, not deleted**, in `budget_aware_dpo_loss.py` and `src/data/preprocessing.py`, so the original Hard-prefers-long behavior remains recoverable for comparison runs.

### 5.7 Known follow-up (out of scope for this change)

The inline length-ratio filter in `build_dpo_pairs` (`src/data/preprocessing.py:473-477`) uses `_length_ratio = 1/length_ratio` for Hard, which assumed chosen was the *longer* solution. Under the new Hard-short paradigm that branch should mirror the Easy case (`chosen_length · length_ratio ≤ rejected_length`). The standalone `filter_pairs_by_length_ratio` function is already correct; the inline filter needs a separate fix when the new datasets are rebuilt.
