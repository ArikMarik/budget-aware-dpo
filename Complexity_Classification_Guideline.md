## **Complexity Classification Guidelines**

Given a problem example (dictionary with fields like `problem`, `problem_source`, `level`, `teacher_token_count`, etc.), follow these steps to determine complexity:

---

### **Step 1: Check if GSM8K (always Easy)**

If `problem_source` contains "gsm" (e.g., "gsm8k", "augmented_gsm8k") → **complexity = 0**

This is an invariant rule - all GSM8K problems are always classified as Easy, regardless of any other attributes.

---

### **Step 2: Check Original MATH Problems**

If `problem_source` == "math":

**First, normalize the level field:**
- Extract the numeric part from the level string
- Examples: "Level 2" → "2", "Level 5" → "5", "Level ?" → invalid, "unknown" → invalid
- If level is invalid (None, empty, "?", "unknown") → skip to Step 3

**Then apply level-based rules:**
| Level | Complexity | Reason |
|-------|------------|--------|
| "1" | 0 (Easy) | Introductory MATH problems |
| "2" | 0 (Easy) | Intermediate MATH problems |
| "3" | See below | Ambiguous - use similarity search |
| "4" | 1 (Hard) | Advanced MATH problems |
| "5" | 1 (Hard) | Expert MATH problems |

**For Level 3 (ambiguous):**
1. **First**: Perform similarity search against original MATH problems
2. If a similar problem is found (similarity ≥ 0.7): use that problem's complexity
3. If no similar problem found: use token fallback

---

### **Step 3: Invalid Level OR Augmented MATH**

For any problem where:
- `problem_source` == "math" but level is invalid (not 1-5), OR
- `problem_source` == "augmented_math"

**Apply this priority:**

1. **Similarity Search First**:
   - Use a sentence embedding model to encode the problem text
   - Search against the index of 5,849 original MATH problems
   - Each indexed problem has: problem text, level (1-5), and complexity (0 or 1)
   - If cosine similarity ≥ 0.7: use the matched problem's complexity
   - The similarity index was built from original MATH problems where:
     - Level 1,2 → complexity 0
     - Level 4,5 → complexity 1

2. **Token Fallback** (if no similar match found):
   - Get the `teacher_token_count` (solution token length)
   - Average over all solutions of this problem (id)
   - If average tokens length > HARD_TOKEN_THRESHOLD → complexity = **1** (Hard)
   - Otherwise → complexity = **0** (Easy, default)

---

### **Step 4: Unknown Source (final fallback)**

For any other `problem_source` (not GSM8K, not math, not augmented_math):
- If average tokens length > HARD_TOKEN_THRESHOLD → complexity = **1**
- Otherwise → complexity = **0** (default Easy)

---

### **Detailed Similarity Search Process**

The similarity search works as follows:

1. **Load the index** (once, lazily):
   - FAISS index with 5,849 original MATH problems
   - Metadata JSONL with each problem's text, level, and pre-computed complexity

2. **For each query problem:**
   a. Encode using sentence-transformers model (e.g., "multi-qa-MiniLM-L6-cos-v1")
   b. Normalize the embedding vector (L2)
   c. Search FAISS index for top-1 most similar problem
   d. If similarity score ≥ 0.7 threshold: return that problem's complexity
   e. If similarity < 0.7: return None (no match found)

3. **Complexity assignment from index:**
   - Level "1" or "2" → complexity 0
   - Level "4" or "5" → complexity 1

---

### **Key Constants**

| Constant | Value | Usage |
|----------|-------|-------|
| `HARD_TOKEN_THRESHOLD` | 250 | Tokens > this = Hard (complexity 1) |
| `EASY_TOKEN_THRESHOLD` | 140 | Used for preference labeling, not complexity |
| `SIMILARITY_THRESHOLD` | 0.7 | Minimum cosine similarity to accept a match |
| `_VALID_MATH_LEVELS` | {"1","2","3","4","5"} | Valid level values |

---

### **Decision Flow Summary**

```
problem_source contains "gsm"?
  → YES: return 0 (Easy)

problem_source == "math"?
  → YES: normalize level
       level in {"1","2"}? → return 0
       level in {"4","5"}? → return 1
       level == "3"? → similarity → fallback
       level invalid? → similarity → fallback

problem_source == "augmented_math"?
  → YES: similarity → fallback

Otherwise:
  → token fallback (tokens > HARD_TOKEN_THRESHOLD ? 1 : 0)
```

---

**Output:** complexity is either **0 (Easy)** or **1 (Hard)**

**This ensures consistent classification: same problem text always gets the same complexity, regardless of which solution (teacher) is used.**

---