# Dataset Analysis: `processed_dpo_dataset_balanced`

**Date**: 2025-03-25
**Dataset**: `data/processed_dpo_dataset_balanced/dataset.jsonl`
**Total samples**: 50,000 (25,000 Easy + 25,000 Hard)

---

## 1. Overview

The balanced DPO dataset contains preference pairs (chosen vs rejected) for math problems at two complexity levels. Each pair teaches the model which response to prefer.

| Complexity | Count  | Rejection Reason     | What the pair teaches                      |
|------------|--------|----------------------|--------------------------------------------|
| Easy (0)   | 25,000 | **100% length-based** | Prefer short, correct answers over verbose ones |
| Hard (1)   | 25,000 | **100% incorrectness** | Prefer correct answers over wrong ones     |

This clean split is by design in the preprocessing pipeline (`src/data/preprocessing.py`):
- **Easy** problems: both chosen and rejected are correct, but rejected is a verbose/redundant version. The model should learn to be concise.
- **Hard** problems: chosen is correct, rejected is incorrect. The model should learn accuracy.

---

## 2. Token Length Statistics

### 2.1 Easy Problems (rejected for length)

The signal is clear: chosen responses are short and tight, rejected are 2-3x longer.

| Metric               | Chosen (short, correct) | Rejected (verbose, correct) |
|----------------------|------------------------|-----------------------------|
| Mean                 | 60 tokens              | 151 tokens                  |
| Std                  | 8                      | 53                          |
| Min                  | 38                     | 99                          |
| Max                  | 70                     | 1,121                       |

**Percentiles — Chosen:**

| P5  | P10 | P25 | P50 | P75 | P90 | P95 | P99 |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 46  | 49  | 54  | 62  | 67  | 69  | 70  | 70  |

**Percentiles — Rejected:**

| P5  | P10 | P25 | P50 | P75 | P90 | P95  | P99 |
|-----|-----|-----|-----|-----|-----|------|-----|
| 117 | 121 | 126 | 130 | 162 | 190 | 224  | 359 |

**Length difference (rejected − chosen):**
- Mean: +91 tokens
- Median: +77 tokens
- P95: +160 tokens
- Max: +1,058 tokens (extreme outlier with very verbose rejected)

**Ratio (rejected / chosen):** mean 2.5x, median 2.4x, up to 17.8x

**Key insight**: Easy chosen responses cluster tightly around 50-70 tokens. The rejected responses are consistently 2-3x longer, sometimes with extreme long tails (99th percentile: 359 tokens). This gives the budget-aware loss a strong, consistent length signal to learn from.

### 2.2 Hard Problems (rejected for incorrectness)

The length pattern is **reversed**: correct answers are longer than incorrect ones.

| Metric               | Chosen (correct)       | Rejected (incorrect)        |
|----------------------|------------------------|-----------------------------|
| Mean                 | 799 tokens             | 543 tokens                  |
| Std                  | 138                    | 210                         |
| Min                  | 378                    | 131                         |
| Max                  | 1,363                  | 1,176                       |

**Percentiles — Chosen (correct):**

| P5  | P10 | P25 | P50 | P75 | P90 | P95   | P99   |
|-----|-----|-----|-----|-----|-----|-------|-------|
| 542 | 602 | 713 | 804 | 898 | 973 | 1,016 | 1,076 |

**Percentiles — Rejected (incorrect):**

| P5  | P10 | P25 | P50 | P75 | P90 | P95 | P99   |
|-----|-----|-----|-----|-----|-----|-----|-------|
| 229 | 277 | 388 | 510 | 695 | 845 | 902 | 1,039 |

**Length difference (rejected − chosen):**
- Mean: **−255 tokens** (rejected is shorter)
- Median: −269 tokens
- P10: −533 tokens (rejected much shorter)
- P90: +45 tokens (only 10% of cases have longer rejected)

**Ratio (rejected / chosen):** mean 0.7x, median 0.6x, down to 0.1x

**Key insight**: For hard problems, the correct answer requires more detailed reasoning (800 tokens on average), while incorrect answers tend to be shorter (543 tokens) — they often take shortcuts or make errors partway through. A naive length penalty would push the model toward these shorter, wrong answers. This is exactly why `lambda_hard` must be near zero.

---

## 3. Correctness Verification of Hard Rejections

We verified the "incorrect" label by extracting the final `\boxed{...}` answer from both chosen and rejected responses:

| Category                        | Count  | Percentage |
|---------------------------------|--------|------------|
| Different final answer          | 18,750 | 75.0%      |
| Same final answer               | 6,246  | 25.0%      |
| Missing boxed answer            | 4      | 0.0%       |

**75% of Hard pairs** have clearly different boxed answers — the rejected solution arrives at a wrong numerical result.

**25% of Hard pairs** have the same boxed answer. In these cases, the "incorrect" label comes from the preprocessing pipeline's evaluation logic, not just from comparing final answers. Possible reasons:
- The rejected response may have flawed intermediate reasoning even if the final answer matches
- The boxed answer extraction may not capture formatting differences
- The evaluation in the preprocessing pipeline uses a more sophisticated correctness check than simple string matching

### 3.1 Examples of Hard Pairs

#### Example A — Different answers (clear incorrectness)

**Problem**: *What is the largest value of n less than 100,000 for which 8(n−2)^5 − n² + 14n − 24 is a multiple of 5?*

**Chosen (correct, 661 words) → answer: 99,997**
The correct solution expands the expression, reduces mod 5, and determines n must end in 2 or 7.

**Rejected (incorrect, 146 words) → answer: 99,995**
The rejected solution incorrectly simplifies the mod-5 analysis, concluding n must be divisible by 5. It's much shorter because it skips the detailed case analysis that the correct solution requires.

---

#### Example B — Different answers (subtle error)

**Problem**: *The science club has 25 members: 10 boys and 15 girls. A 5-person committee is chosen at random. What is the probability that the committee has at least 1 boy and at least 1 girl?*

**Chosen (correct, 327 words) → answer: 49875/53130**
Correctly computes P(at least 1 boy and 1 girl) = 1 − P(all boys) − P(all girls).

**Rejected (incorrect, 208 words) → answer: 399/425**
Gets the same numerator/denominator (49875/53130) but then incorrectly simplifies the fraction, claiming 49875/53130 = 399/425. The actual simplified form is 3325/3542.

---

#### Example C — Same answer (reasoning quality difference)

**Problem**: *Find 53·(3 1/5 − 4 1/2) ÷ (2 3/4 + 1 2/3)*

**Chosen (correct, 285 words) → answer: −15 3/5**
Clear step-by-step arithmetic with proper fraction operations.

**Rejected (also gets −15 3/5, 151 words)**
Arrives at the same answer but with compressed reasoning. Labeled "incorrect" by the preprocessing pipeline's evaluation, possibly due to formatting or intermediate step validation.

---

#### Example D — Very short wrong answer

**Problem**: *Let N be the number of ways to write 2010 in the form 2010 = a₃·10³ + a₂·10² + a₁·10 + a₀ where 0 ≤ aᵢ ≤ 99.*

**Chosen (correct, 481 words) → answer: 202**
Systematic case analysis covering all valid coefficient combinations.

**Rejected (incorrect, 76 words) → answer: 20,201**
Only 76 words — makes a quick (wrong) combinatorial argument. This is a typical Hard rejection: the wrong answer is much shorter because it doesn't do the careful analysis needed.

---

## 4. Implications for Budget-Aware DPO Training

### Why the dataset structure matters for λ values

| Complexity | Length relationship | Ideal λ | Reasoning |
|------------|-------------------|---------|-----------|
| Easy (0) | Rejected is 2.5x longer | High (0.05) | Length penalty reinforces the preference signal — both point toward the shorter answer |
| Hard (1) | Chosen is 1.5x longer | Near-zero (0.001) | Length penalty would **oppose** the correctness signal — penalizing length means penalizing the correct answer |

### Risk of length penalty on Hard problems

For Hard problems, the correct (chosen) answer averages 799 tokens while the incorrect (rejected) averages 543 tokens. Any length penalty on Hard problems actively fights against the DPO objective:
- DPO says: "prefer the chosen (longer, correct) response"
- Length penalty says: "prefer shorter responses"
- These two signals conflict, slowing convergence on Hard problems

This explains the observation that `complexity_1_loss` decreases more slowly in budget-aware training compared to baseline: even `lambda_hard=0.001` introduces a small opposing gradient.

### Recommendation

Consider setting `lambda_hard=0.0` to completely eliminate length pressure on Hard problems, since any length penalty there works against correctness.

---

## 5. Script

Statistics generated by `scripts/analysis/dataset_stats.py`.
