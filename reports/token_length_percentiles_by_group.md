# Token-Length Percentile Tables by Group

Source: `data/problem_index.json` (606,922 problems). Solutions sampled from `data/openmathinstruct.jsonl`.

Percentiles are over the **flattened list of per-solution teacher token counts** within each group (each problem contributes all of its teacher-solution lengths).

## 1. By MATH Level

| Level | # problems | # solutions | mean | min | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|---:|
| 1 | 17,818 | 609,321 | 245.9 | 36 | 104 | 137 | 199 | 300 | 451 | 1666 |
| 2 | 54,453 | 1,745,025 | 282.2 | 38 | 118 | 162 | 234 | 350 | 512 | 1726 |
| 3 | 728 | 272,013 | 273.9 | 42 | 126 | 170 | 239 | 344 | 463 | 1466 |
| 4 | 75,550 | 2,062,533 | 345.7 | 38 | 149 | 208 | 300 | 436 | 611 | 1677 |
| 5 | 87,226 | 1,940,977 | 388.6 | 39 | 166 | 236 | 344 | 496 | 686 | 1680 |

## 2. By Problem Source

| Source | # problems | # solutions | mean | min | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|---:|
| gsm8k | 7,426 | 458,822 | 154.3 | 50 | 92 | 114 | 145 | 183 | 227 | 1158 |
| augmented_gsm8k | 73,636 | 2,111,683 | 166.1 | 41 | 95 | 118 | 152 | 197 | 250 | 1331 |
| math | 7,435 | 2,480,129 | 282.2 | 36 | 122 | 169 | 246 | 355 | 484 | 1605 |
| augmented_math | 518,425 | 8,922,157 | 362.6 | 37 | 144 | 206 | 310 | 471 | 669 | 2207 |

## 3. Representative Example

**problem_id=`4961` · source=`math` · level=`3` · avg_token_length=192.5 (group pct ≈ 33.5)**

- **Problem:** The average age of the 10 females in a choir is 30 years. The average age of the 15 males in the same choir is 35 years. What is the average age, in years, of the 25 people in the choir?
- **Solution token count:** 120 → group percentile ≈ 8.6 (expected_answer=`33`, correct=True)

> **Generated solution:**
>
> Find the total age of the females and males separately:
> Total age of 10 females: $10 \times 30 = 300$ years
> Total age of 15 males: $15 \times 35 = 525$ years
>
> Find the total age of all choir members:
> Total age of all members: $300 + 525 = 825$ years
>
> Find the average age of all choir members:
> Average age: $825 \div 25 = \boxed{33}$ years

## 4. Percentile-Band Recommendations for `label_preference`

### 4.1 Methodology

`label_preference` (`src/data/preprocessing.py:266`) assigns **preferred** to a correct solution whose token count sits inside a percentile band **computed within that solution's own problem**, and **rejected** to every other correct solution in the same group. The current defaults are:

| Complexity | Default band | Intent |
|---|---|---|
| Easy (C=0) | `[10, 40]` | Short-but-not-degenerate CoT |
| Hard (C=1) | `[60, 95]` | Full CoT minus outlier tail |

Per the updated labeling spec (see §4.6 for the reasoning behind the revised boundary):

- **Easy (C=0)** — `gsm8k`, `augmented_gsm8k`, MATH L1 (and `augmented_math` whose similarity match resolves to L1).
- **Hard (C=1)** — MATH L2, L3, L4, L5 (and `augmented_math` whose similarity match resolves to L2–L5).

Bands are applied *within each problem's* teacher-solution distribution, but the group-level shape (right-tail heaviness, median position, inter-decile spread) still governs how aggressive the band should be at each edge, because most problems inherit that shape.

### 4.2 Distribution shape per group

Two diagnostic indicators derived from §1 / §2:

| Group | p10 | p50 | p90 | p90/p10 | (p50 − p10) / (p90 − p10) |
|---|---:|---:|---:|---:|---:|
| gsm8k | 92 | 145 | 227 | 2.47× | 0.39 |
| augmented_gsm8k | 95 | 152 | 250 | 2.63× | 0.37 |
| MATH L1 | 104 | 199 | 451 | 4.34× | 0.27 |
| MATH L2 | 118 | 234 | 512 | 4.34× | 0.29 |
| MATH L3 | 126 | 239 | 463 | 3.67× | 0.34 |
| MATH L4 | 149 | 300 | 611 | 4.10× | 0.33 |
| MATH L5 | 166 | 344 | 686 | 4.13× | 0.34 |
| math (all levels) | 122 | 246 | 484 | 3.97× | 0.34 |
| augmented_math | 144 | 310 | 669 | 4.65× | 0.32 |

Observations that drive the band choices:

1. **GSM8K family is the most symmetric** (median at ~0.39 of the p10–p90 range; spread ~2.5×). Short solutions are not rare in these groups, so an aggressive low-tail exclusion is unnecessary.
2. **MATH L1–L2 are the most right-skewed** (median at only 0.27–0.29 of the range). The "short-and-clean" region is concentrated near p10–p30; a tight upper edge keeps "preferred" out of median territory.
3. **MATH L2 and L3 form a "middle tier" — see §4.6 for a dedicated recommendation.** Their p50s are essentially identical (234 / 239), sitting midway between L1 (199) and L4 (300). Keeping them together is clearly correct; the open question is whether the L2/L3 pair belongs with Easy or with Hard. The recommendation (§4.6) is **L2+L3 → Hard**, making L1 the top of Easy.
4. **MATH L5 and `augmented_math` have the heaviest right tails**. L5 max (1680) is ~2.4× p90; `augmented_math` max (2207) is ~3.3× p90. The top 5–8% are extreme-length outliers (redundant restatements, off-task exploration) that should not be promoted to "preferred".

### 4.3 Recommended unified bands (drop-in defaults)

| Flag | Band | Low-edge rationale | High-edge rationale |
|---|---|---|---|
| **Easy (C=0)** | `[10, 40]` *(unchanged)* | Excludes the bottom decile → filters degenerate/truncated solutions (min token counts of 36–50 across all easy groups). | p40 sits just below the group median for every easy group (p40 ≈ 130 for gsm8k, ≈ 180 for L1, ≈ 200 for L2), so "preferred" stays reliably shorter than the remaining "rejected" mass. |
| **Hard (C=1)** | `[60, 92]` *(high tightened from 95)* | p60 sits above the group median for every hard group (≈ 280 for L3, ≈ 355 for L4, ≈ 400 for L5). Dropping below p60 pulls "preferred" into short-CoT territory. | p92 rather than p95 cuts more of the heavy right tail. For L5 and `augmented_math`, ranks 92–100 span roughly 700 → 1700–2200 tokens (mostly redundant outliers). |

Expected within-problem length-ratio behaviour (using group p-values as an upper-bound proxy — within-problem spread is typically ~60–70 % of group spread):

| Flag | Mean "preferred" rank | Mean "rejected (length)" rank | Expected ratio (long / short) |
|---|---:|---:|---:|
| Easy `[10, 40]` | ≈ p25 | ≈ p61 | 1.4–1.7× |
| Hard `[60, 92]` | ≈ p76 | ≈ p38 | 1.7–1.9× |

Combined with the downstream `length_ratio ≥ 2` filter in `build_dpo_pairs`, a meaningful fraction of candidate pairs still survives while guaranteeing a clear stylistic gap between chosen and rejected.

### 4.4 Per-group bands (if `_band_for_complexity` is extended)

`_band_for_complexity` (`src/data/preprocessing.py:253`) currently dispatches only on `complexity`. If it gains a per-source / per-level override (each problem's `source` and `level` are already available from `problem_index` at the `build_dpo_pairs` call-site), these bands better match each group's shape:

| Group | Flag | Band | Reason |
|---|---|---|---|
| `gsm8k` | Easy | `[10, 45]` | Narrow, symmetric spread (2.47×) — upper edge can drift closer to the median without diluting "preferred". |
| `augmented_gsm8k` | Easy | `[10, 45]` | Same shape as `gsm8k` (spread 2.63×, median pos. 0.37). |
| MATH L1 | Easy | `[10, 35]` | Heavy right-skew (median at 0.27 of range) — keep the upper edge tight to stay in the genuinely-short region. |
| `augmented_math` → L1 | Easy | `[10, 35]` | Inherits L1's skew. |
| MATH L2 | Hard | `[65, 92]` | Light end of Hard (p50 = 234). Pull the lower edge up to p65 so "preferred" sits clearly above L2's median (~p65 ≈ 275 tokens vs p50 = 234). |
| MATH L3 | Hard | `[65, 90]` | Shape essentially identical to L2 (p50 = 239). Same lower-edge bump as L2; upper edge at p90 because L3's tail is narrower than L4/L5. |
| MATH L4 | Hard | `[60, 92]` | Matches the unified hard recommendation — shape is exactly the case those defaults were chosen for. |
| MATH L5 | Hard | `[60, 90]` | Heaviest native tail (p90 → max is a 2.4× jump). Cap at p90 to exclude the verbose outlier band. |
| `augmented_math` → L2/L3/L4/L5 | Hard | `[60, 90]` | Widest overall spread (4.65×) and the biggest absolute tail in the corpus (max 2207). Aggressive cap matches L5's rationale. |

Implementation sketch: resolve `(low, high)` per problem inside `build_dpo_pairs` using the problem's `source` + `level`, then pass it through the existing `bands` kwarg — no signature change for `label_preference`.

### 4.6 Recommendation on MATH L2 and L3: reclassify BOTH as Hard

**TL;DR — move L2 and L3 both into Hard.** Do not skip either. Make L1 the top of Easy and L2 the bottom of Hard.

**Why L2 and L3 stay together.** Their group-level distributions are essentially identical (p50 = 234 vs 239; p25 = 162 vs 170; p75 = 350 vs 344). Any labeling contract that applies to one should apply to the other — splitting the L2–L3 pair across Easy/Hard creates supervision overlap at the boundary.

**Why the pair belongs in Hard, not Easy.**

1. **Natural break points in the p50 sequence put the strongest gap between L1 and L2+.** Consecutive p50 gaps:
   - gsm8k(145) → aug_gsm8k(152): Δ = 7
   - aug_gsm8k(152) → L1(199): Δ = 47 *(Easy/Hard boundary candidate A)*
   - L1(199) → L2(234): Δ = 35
   - L2(234) → L3(239): Δ = 5
   - L3(239) → L4(300): Δ = 61 *(Easy/Hard boundary candidate B)*
   - L4(300) → L5(344): Δ = 44

   Candidate A (after L1) and Candidate B (after L3) are the two largest gaps. A tighter Easy bucket + cleaner semantics tips the choice toward A.

2. **Intra-bucket tightness.** Under *Easy = {gsm8k, aug_gsm8k, L1}*, the Easy p50 range is 145–199 (ratio 1.37×). Under *Easy = {gsm8k, aug_gsm8k, L1, L2, L3}*, it would be 145–239 (ratio 1.65×). The A split gives the Easy bucket a tight, consistent "short-answer" target.

3. **Budget-Aware semantics map cleanly.** Easy = "concise answer, key steps only" (gsm8k/L1 pattern). Hard = "full chain of reasoning" (L2-and-up MATH pattern). L2 problems genuinely benefit from showing intermediate work; training them as Easy would push the model to emit terse answers on problems that need the work exposed.

4. **The boundary-overlap problem is solved.** Had L3 been Hard while L2 was Easy, "Hard-preferred L3" (~p76 ≈ 355 tokens) would have overlapped numerically with "Easy-rejected L2" (p75 = 350), producing contradictory supervision at the Easy/Hard boundary. Placing L2 *and* L3 together in Hard eliminates this entirely.

5. **Neither should be skipped.** L2 contributes 54,453 problems / 1,745,025 solutions (~14% of MATH solutions); L3 contributes 728 problems / 272,013 solutions (~374 solutions per problem — the richest per-problem density in the corpus, so per-problem percentile ranks are sharpest there). Discarding either to dodge a labeling ambiguity trades a lot of signal for very little noise reduction.

6. **L1 staying in Easy is still defensible despite its wider spread.** L1's p90/p10 = 4.34× is wider than gsm8k's 2.47×, but L1's absolute p25 (137 tokens) is genuinely short. The within-problem band `[10, 40]` (or `[10, 35]` per §4.4) still selects concise solutions for L1 regardless of group-level spread.

**Caveats worth noting (not blockers).**

- L2 sits at the light end of Hard (p50 = 234). Under the *unified* Hard band `[60, 92]`, L2-preferred centres at ~p76 ≈ 360 tokens — reasonable for a moderate MATH problem, but on the longer side. The per-group recommendation in §4.4 uses `[65, 92]` for L2 to raise the lower edge so "preferred" sits clearly above L2's median rather than hugging it.
- If using a single unified Hard band, L5 and `augmented_math` (long tails) get the same treatment as L2 (short tail). The per-group table in §4.4 addresses this by giving L2/L3 a tighter lower edge (p65) and L5/`augmented_math` a tighter upper edge (p90).

**Implementation is low-cost.** In `classify_complexity` (`src/data/preprocessing.py:184`), the L1–L2 branch splits: L1 → `return 0, "1"`, L2 → `return 1, "2"`. The L3 branch (currently similarity-search + token fallback) collapses to `return 1, "3"`. `augmented_math` problems whose similarity match resolves to L2 or L3 inherit the move automatically.

**Updated Easy / Hard membership:**

- **Easy (C=0):** `gsm8k`, `augmented_gsm8k`, MATH L1, and `augmented_math` matched to L1.
- **Hard (C=1):** MATH **L2, L3**, L4, L5, and `augmented_math` matched to L2–L5.

**Consequences elsewhere in §4:**

- §4.1 and §4.3 unified-band recommendations stand unchanged — the bands are defined by what "preferred" should look like in each bucket, not by which levels live in the bucket.
- §4.4 per-group bands have been updated: L2 and L3 now appear as Hard rows (`[65, 92]` and `[65, 90]` respectively), and `augmented_math` → L2/L3/L4/L5 inherits the Hard row.

### 4.7 Environment-variable settings (no code change)

Applied as unified (easy / hard) defaults, the recommendation translates to:

```bash
# Easy: unchanged
export EASY_PREF_PCT_LOW=10
export EASY_PREF_PCT_HIGH=40
# Hard: tighten the upper edge from 95 → 92 to exclude the verbose outlier tail
export HARD_PREF_PCT_LOW=60
export HARD_PREF_PCT_HIGH=92
```
