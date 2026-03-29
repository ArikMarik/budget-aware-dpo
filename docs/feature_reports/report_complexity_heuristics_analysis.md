# Complexity Heuristics Analysis Report

**Purpose:** Statistics to inform complexity threshold decisions (EASY_TOKEN_THRESHOLD, HARD_TOKEN_THRESHOLD, level-based classification).

---

## 1. Dataset Composition

### 1.2 Training Data

Analyzed **50,000** examples from training dataset.

| Category | Count | Percentage |
|----------|-------|------------|
| MATH | 40,740 | 81.5% |
| GSM8k | 9,260 | 18.5% |
| **Total** | **50,000** | 100% |

---

## 2. MATH Level Distribution (Training Data)

MATH uses a 5-level difficulty scale (AoPS). Levels from MATH train lookup.

| Level | Count | Percentage |
|-------|-------|------------|
| Level 1 | 756 | 8.5% |
| Level 2 | 1,874 | 21.1% |
| Level 3 | 2,071 | 23.3% |
| Level 4 | 2,079 | 23.4% |
| Level 5 | 2,097 | 23.6% |
| Level ? | 3 | 0.0% |

---

## 3. Token Count Statistics (tiktoken cl100k_base)

Using `tiktoken` cl100k_base (GPT-4/Claude compatible) for accurate token counts.

### 3.2 Training Data: Average Tokens per Category

| Category | Count | Avg Tokens | P25 | P50 | P75 |
|----------|-------|------------|-----|-----|-----|
| MATH | 40,740 | 330.2 | 185 | 279 | 427 |
| GSM8k | 9,260 | 147.3 | 106 | 134 | 175 |


### 3.3 Training Data: Average Tokens per MATH Level

| Level | Avg Tokens | Count |
|-------|------------|-------|
| Level 1 | 168.2 | 756 |
| Level 2 | 213.5 | 1,874 |
| Level 3 | 249.8 | 2,071 |
| Level 4 | 293.3 | 2,079 |
| Level 5 | 346.1 | 2,097 |
| Level ? | 198.0 | 3 |

---

## 4. Percentile-Based Thresholds

Suggested approach: Use percentiles instead of fixed 50/80 to adapt to each dataset.

### 4.4 Training Data: Token Percentiles

| Category | P10 | P25 | P50 | P75 | P90 |
|----------|-----|-----|-----|-----|-----|
| MATH | 130 | 185 | 279 | 427 | 611 |
| GSM8k | 84 | 106 | 134 | 175 | 220 |

### 4.5 Training Data: Per-Level Token Percentiles (MATH)

**Level 1**

| Percentile | Token Count |
|------------|-------------|
| P10 | 90 |
| P25 | 109 |
| P50 | 151 |
| P75 | 208 |
| P90 | 267 |

**Level 2**

| Percentile | Token Count |
|------------|-------------|
| P10 | 103 |
| P25 | 133 |
| P50 | 183 |
| P75 | 260 |
| P90 | 362 |

**Level 3**

| Percentile | Token Count |
|------------|-------------|
| P10 | 117 |
| P25 | 158 |
| P50 | 222 |
| P75 | 310 |
| P90 | 416 |

**Level 4**

| Percentile | Token Count |
|------------|-------------|
| P10 | 138 |
| P25 | 189 |
| P50 | 265 |
| P75 | 367 |
| P90 | 473 |

**Level 5**

| Percentile | Token Count |
|------------|-------------|
| P10 | 160 |
| P25 | 216 |
| P50 | 308 |
| P75 | 437 |
| P90 | 572 |

**Level ?**

| Percentile | Token Count |
|------------|-------------|
| P10 | 157 |
| P25 | 157 |
| P50 | 172 |
| P75 | 172 |
| P90 | 172 |

---

## 5. CoT Indicators & Sentence Count (Optional Heuristics)

CoT patterns: First, Step, Therefore, Thus, Hence, "So we", "Let me think", "In summary", Consequently.
Count = total occurrences of these patterns per answer.


### 5.5 Training Data: CoT Indicator Count

| Category | Count | Avg Count | Median |
|----------|-------|-----------|--------|
| MATH | 40,740 | 1.7 | 1.0 |
| GSM8k | 9,260 | 1.5 | 1.0 |

### 5.6 Training Data: CoT Indicator Count per MATH Level

| Level | Avg Count | Median | Count |
|-------|-----------|--------|-------|
| Level 1 | 0.9 | 1.0 | 756 |
| Level 2 | 1.0 | 1.0 | 1,874 |
| Level 3 | 1.2 | 1.0 | 2,071 |
| Level 4 | 1.3 | 1.0 | 2,079 |
| Level 5 | 1.4 | 1.0 | 2,097 |
| Level ? | 0.7 | 1.0 | 3 |

### 5.7 Training Data: Sentence Count

| Category | Count | Avg Sentences | P25 | P50 | P75 |
|----------|-------|---------------|-----|-----|-----|
| MATH | 40,740 | 14.2 | 8 | 12 | 18 |
| GSM8k | 9,260 | 7.4 | 5 | 7 | 9 |

### 5.8 Training Data: Sentence Count per MATH Level

| Level | Avg Sentences | P25 | P50 | P75 | Count |
|-------|---------------|-----|-----|-----|-------|
| Level 1 | 8.6 | 6 | 8 | 11 | 756 |
| Level 2 | 10.0 | 6 | 9 | 13 | 1,874 |
| Level 3 | 11.7 | 7 | 10 | 15 | 2,071 |
| Level 4 | 13.1 | 9 | 12 | 16 | 2,079 |
| Level 5 | 14.6 | 9 | 13 | 18 | 2,097 |
| Level ? | 6.7 | 6 | 7 | 7 | 3 |

---

## 6. Recommendations for Threshold Decisions

### Current Fixed Thresholds
- EASY_TOKEN_THRESHOLD = 50 (word count)
- HARD_TOKEN_THRESHOLD = 80 (word count)

### Observations

- - **Training MATH:** P25 ≈ 185, P50 ≈ 279, P75 ≈ 427 tokens
- - **Training GSM8k:** P25 ≈ 106, P50 ≈ 134, P75 ≈ 175 tokens
- - Token count (tiktoken) differs from word count; math LaTeX uses more tokens.

### Suggested Approaches
1. **Replace word count with tiktoken** for `teacher_token_count`.
2. **Per-category percentiles:** Easy = below P25, Hard = above P75. Use **training** percentiles when preprocessing training data.
3. **Level-based (MATH):** Level 1-2 → Easy, Level 3 → Medium, Level 4-5 → Hard.
4. **CoT indicators:** Secondary signal only.

---

## 7. Concrete Conclusions: Recommended Thresholds

| Heuristic | Value | Notes |
|-----------|-------|-------|
| EASY_TOKEN_THRESHOLD | **70** | Below = short/direct (tiktoken) |
| HARD_TOKEN_THRESHOLD | **130** | Above = long CoT (tiktoken) |
| MATH Level 1-2 | C=0 | Easy |
| MATH Level 4-5 | C=1 | Hard |
| MATH Level 3 | Token fallback | Use P25/P75 |

---

## 8. Summary Table: All Levels & Heuristics

Consolidated view of key metrics for threshold decisions.


### Training Data Summary

| Source | Category | Count | Avg Tokens | P25 | P50 | P75 | Avg CoT | Avg Sent |
|--------|----------|-------|------------|-----|-----|-----|---------|----------|
| Train | MATH | 40,740 | 330.2 | 185 | 279 | 427 | 1.7 | 14.2 |
| Train | GSM8k | 9,260 | 147.3 | 106 | 134 | 175 | 1.5 | 7.4 |
| Train | Level 1 | 756 | 168.2 | 109 | 151 | 208 | 0.9 | 8.6 |
| Train | Level 2 | 1,874 | 213.5 | 133 | 183 | 260 | 1.0 | 10.0 |
| Train | Level 3 | 2,071 | 249.8 | 158 | 222 | 310 | 1.2 | 11.7 |
| Train | Level 4 | 2,079 | 293.3 | 189 | 265 | 367 | 1.3 | 13.1 |
| Train | Level 5 | 2,097 | 346.1 | 216 | 308 | 437 | 1.4 | 14.6 |
| Train | Level ? | 3 | 198.0 | 157 | 172 | 172 | 0.7 | 6.7 |

---

*Generated by scripts/analyze_complexity_heuristics.py*