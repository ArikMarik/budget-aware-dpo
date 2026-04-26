# Load Real Data Script Update

This document describes the updates made to `scripts/load_real_data.py` to support problem indexing with unique IDs, complexity classification, and solution token length tracking.

---

## Session Goal

Update `scripts/load_real_data.py` to:
1. Assign a unique integer ID to each unique problem
2. Create a new JSON file containing all solution token lengths per problem, along with its complexity classification

---

## Features Implemented

### 1. Problem Index Building

The script now builds a **problem index** that groups solutions by unique problem text and tracks:

| Field | Description |
|-------|------------|
| `problem_id` | Unique integer (0, 1, 2, ...) per unique problem |
| `problem` | The problem text |
| `problem_source` | Source dataset (math, augmented_math, gsm8k, augmented_gsm8k) |
| `level` | MATH level (1-5) or level from similar problem |
| `token_lengths` | List of all teacher token counts for this problem |
| `avg_token_length` | Average token count across all solutions |
| `complexity` | 0 (Easy) or 1 (Hard) |

### 2. Source Preference Order

When the same problem exists in multiple sources, data is selected by preference:

| Priority | Source | Rank |
|----------|--------|-----|
| 1 (best) | `math` | 0 |
| 2 | `augmented_math` | 1 |
| 3 | `gsm8k` | 2 |
| 4 (worst) | `augmented_gsm8k` | 3 |

- `problem_source` field uses the highest priority source
- Complexity is classified using that source's logic
- `token_lengths` includes ALL solutions from all sources for accurate averaging

### 3. Complexity Classification

Complexity is classified using the logic in `src/data/preprocessing.py`:

1. **GSM8K/augmented_gsm8k** → Always complexity = 0 (Easy)
2. **MATH with valid level**:
   - Level 1, 2 → 0 (Easy)
   - Level 4, 5 → 1 (Hard)
   - Level 3 → Similarity search → Token fallback
3. **Augmented MATH** → Similarity search → Token fallback
4. **Unknown source** → Token fallback (tokens > 250 = Hard)

When using similarity search, the **level is copied** from the matched original MATH problem.

### 4. Token Fallback

The average token length (across all solutions for a problem) is used for complexity classification when:
- Level 3 problems
- Augmented MATH without similar match
- Unknown problem sources

**Constants:**
- `HARD_TOKEN_THRESHOLD` = 250 tokens → complexity = 1 (Hard)

---

## Files Modified

### `src/data/preprocessing.py`

1. **`find_similar_math_problem()`**
   - Changed return type from `int | None` to `tuple[int | None, str | None]`
   - Now returns `(complexity, matched_level)`

2. **`classify_complexity()`**
   - Added optional `avg_token_length` parameter
   - Changed return type from `int` to `tuple[int, str | None]`
   - Returns `(complexity, matched_level)`
   - Uses average token length for token fallback when provided

### `scripts/load_real_data.py`

1. **Added imports**
   - `classify_complexity` from `src.data.preprocessing`

2. **Added source preference helpers**
   - `SOURCE_PREFERENCE` dict
   - `get_source_rank()` function

3. **Added `build_problem_index()` function**
   - Groups solutions by normalized problem text
   - Sorts by source preference
   - Collects all token lengths
   - Computes average token length
   - Classifies complexity
   - Returns list of problem index records

4. **Updated `main()`**
   - Added `--no-problem-index` flag to skip building index
   - Outputs to `data/problem_index.json` (regular JSON array)

---

## Output File

**Path:** `data/problem_index.json`

**Format:** Regular JSON array

```json
[
  {
    "problem_id": 0,
    "problem": "What is 2 + 2?",
    "problem_source": "math",
    "level": "2",
    "token_lengths": [145, 152, 148],
    "avg_token_length": 148.33,
    "complexity": 0
  },
  {
    "problem_id": 1,
    "problem": "Solve for x: 2x + 5 = 15",
    "problem_source": "augmented_math",
    "level": "3",
    "token_lengths": [280, 295],
    "avg_token_length": 287.5,
    "complexity": 1
  }
]
```

---

## Usage

```bash
# Load data and build problem index (default behavior)
python scripts/load_real_data.py --split train_1M

# Limit examples for quick test
python scripts/load_real_data.py --split train_1M --limit 1000

# Skip building problem index
python scripts/load_real_data.py --no-problem-index

# Skip test sets (faster)
python scripts/load_real_data.py --skip-test-sets
```

---

## Similarity Index

The complexity classification uses a pre-built FAISS index for similarity search:

- **Location:** `data/math_problem_index/`
- **Files:**
  - `index.faiss` - FAISS index with 5,849 original MATH problems
  - `metadata.jsonl` - Problem text, level, and complexity per problem
  - `config.json` - Embedding model configuration
- **Embedding model:** `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`
- **Similarity threshold:** 0.7

---

## Future Development Ideas

1. **Incremental index building:** Support adding new problems without rebuilding from scratch
2. **Problem deduplication:** Detect near-duplicate problems via similarity search
3. **Statistics output:** Add histograms of complexity distribution
4. **Filtering options:** Filter by source, complexity, or token length range
5. **Cache similarity index:** Memory-mapped loading for large datasets