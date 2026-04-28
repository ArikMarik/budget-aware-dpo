# PRD: Switch to Qwen2.5-Math-1.5B + Match Official Evaluation & Grading

## Goal

Three coupled upgrades:
1. Replace base model (`Qwen/Qwen2.5-0.5B` → `Qwen/Qwen2.5-Math-1.5B`)
2. Replace grading logic everywhere — evaluation **and** pair construction — with the official Qwen2.5-Math pipeline (`strip_string` + `math_equal`)
3. Align the full evaluation pipeline (prompts, few-shot, generation params) to match the official Qwen2.5-Math base model evaluation setup

---

## Background

### Why the model changes
Qwen2.5-Math-1.5B is pre-trained on math-heavy corpora. Smallest available math-specific model (no 0.5B math variant exists). Both fit on 24GB GPU; 1.5B gives 3× capacity and dramatically better math baseline. Use the **base** model for DPO training — instruct variant has GRPO/RL priors that conflict with DPO.

### Why grading needs to change
Our current `verify_correctness` (in `answer_extraction.py`) used an LLM judge as a third tier because our normalization was too weak — answers like `\frac{1}{2}` vs `1/2` looked different to string/SymPy comparison. The root cause is missing `strip_string` normalization, not insufficient grading logic. The official Qwen pipeline normalizes both predicted and ground-truth answers extensively before any comparison, eliminating the need for an LLM judge.

The LLM judge also: loads a 7B model during evaluation (memory pressure), is non-deterministic, can reverse a correct SymPy `False` (false positives), and is extremely slow.

### Where grading is used
- **`src/evaluation/run_evaluation.py`**: `generate_and_evaluate` calls `verify_correctness` per problem
- **`src/data/preprocessing.py`**: `_verify_correctness` (line 152) checks `correctness_flag` first, then falls back to `verify_correctness` when the flag is absent — this is the pair construction path

Both use the same `verify_correctness` function from `src/evaluation/answer_extraction.py`. Replacing it fixes both pipelines.

### Official Qwen2.5-Math evaluation setup (base model)
- Prompt type: `cot` — raw completion format, no chat template, no system prompt
- GSM8K: 8-shot, format `"Question: {q}\nAnswer: {a}"`, separator `"\n\n\n"`
- MATH: 4-shot, same format, first 4 of 5 exemplars from `evaluation/examples.py`
- Generation: greedy (`temperature=0`), `max_new_tokens=2048`, stop at `"\n\nQuestion:"`
- Answer extraction: `\boxed{}` → "he answer is" → last number, then `strip_string` normalization
- Grading: `math_equal` — numeric (`isclose rel_tol=1e-4`, with percentage variants) → symbolic (SymPy, timeout-protected)

---

## Scope

Nine tasks across four files plus one shell command. No new architectural changes.

---

## Task 1 — Update model name in config

**File:** `src/config.py` lines 79–80

```python
# Before
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
UNSLOTH_MODEL_NAME = "unsloth/Qwen2.5-0.5B"

# After
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
UNSLOTH_MODEL_NAME = "unsloth/Qwen2.5-Math-1.5B"
```

Note: `UNSLOTH_MODEL_NAME` is never imported or used anywhere in source code. Update for consistency.

---

## Task 2 — Switch model loading dtype to bfloat16

**File:** `src/training/dpo_trainer.py`

Two locations — `create_model` (~line 451) and `create_ref_model` (~line 480). The 1.5B model in float32 uses ~6GB for weights; bfloat16 halves that. `use_mixed_precision=True` is already the default so training autocast is handled separately.

```python
# Before (both locations)
torch_dtype=torch.float32,

# After (both locations)
torch_dtype=torch.bfloat16,
```

---

## Task 3 — Add all missing parsing steps to `answer_extraction.py`

**File:** `src/evaluation/answer_extraction.py`

Keep the existing file structure and all existing functions. Add every parsing step from the official `evaluation/parser.py` that is relevant to our datasets (MATH, GSM8K, and their augmented versions). Do not add anything that only applies to datasets we don't use.

### What to include and why

**Include — these are directly called by `strip_string` and handle LaTeX math normalization:**

| Function | What it does | Why needed |
|---|---|---|
| `_fix_fracs(string)` | Fixes `\frac12` → `\frac{1}{2}` | MATH answers frequently have malformed fractions |
| `_fix_a_slash_b(string)` | Converts `1/2` → `\frac{1}{2}` | Model and GT may use different fraction notation |
| `_fix_sqrt(string)` | Fixes `\sqrt2` → `\sqrt{2}` | Same — sqrt formatting varies |
| `convert_word_number(text)` | "eight" → "8" | GSM8K problems sometimes produce word numbers |
| `unit_texts` list | ~150 unit strings | Strips units like "mph", "km", "°" from answers |
| `strip_string(string, skip_unit=False)` | Full LaTeX normalization | Core normalizer — calls all helpers above |

These six come as a package. `strip_string` calls the others internally — you cannot port `strip_string` without them.

**Include — extraction paths that apply to MATH/GSM8K model outputs:**

| Path | Trigger | Why needed |
|---|---|---|
| Cyrillic strip `text.replace("ки", "")` | Some model outputs contain Cyrillic artifacts | Defensive cleanup |
| `"final answer is $...$. I hope"` | Minerva-style phrasing | Some MATH-trained models use this phrasing |
| `"he answer is"` | Already present (as "The answer is" regex) | Equivalent — keep our version |
| `"final answer is"` | Without the Minerva `$...$` wrapper | Model fallback phrasing |
| Pre-return cleanup + `strip_string` call | Always applied | The normalization that fixes all LaTeX differences |

**Exclude — only apply to datasets we don't use:**

| What | Reason to exclude |
|---|---|
| `choice_answer_clean` | Only for multi-choice (MMLU, SAT, AQuA) |
| `extract_multi_choice_answer` | Same |
| Multi-choice routing block in `extract_answer` | `data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]` — never triggered |
| Multi-choice cleanup block | `data_name in ["sat_math", "aqua"] or "mmlu" in data_name` — never triggered |
| Chinese `"答案是"` path | Chinese-language datasets only |
| `find_box` | Only used by `extract_theoremqa_answer` |
| `clean_units` | Only used by `extract_theoremqa_answer` |
| `extract_theoremqa_answer` | TheoremQA dataset only |
| `parse_ground_truth` | Their data loading pipeline; we pre-store GT |
| `parse_question` | Their data loading pipeline |
| `run_execute` | Tool-integrated reasoning (PAL/ToRA) only |

### 3a — Add the normalization helpers and `strip_string`

Port verbatim from the official `parser.py`, add to the top of `answer_extraction.py` before the existing functions:
- `_fix_fracs(string)`
- `_fix_a_slash_b(string)`
- `_fix_sqrt(string)`
- `convert_word_number(text)` — add `from word2number import w2n` to imports
- `unit_texts` list (all entries)
- `strip_string(string, skip_unit=False)`

### 3b — Update `extract_answer`

Update the signature to add `data_name` (default `"math"`) and `use_last_number` (default `True`). For our data, `data_name` is always `"gsm8k"` or `"math"` — `skip_unit` will always be `False`.

Add to the existing function body, without removing existing paths:

1. **First line of body** — Cyrillic strip:
```python
text = text.replace("ки", "")
```

2. **After `\boxed{}` path, before `####`** — Minerva format:
```python
if "final answer is $" in text and "$. I hope" in text:
    tmp = text.split("final answer is $", 1)[1]
    pred = tmp.split("$. I hope", 1)[0].strip()
    # fall through to cleanup below
```

3. **After "The answer is" path** — plain "final answer is":
```python
elif "final answer is" in text:
    pred = text.split("final answer is")[-1].strip()
```

4. **End of function, before `return`** — cleanup and normalize (unconditional, applies to every extracted answer):
```python
pred = re.sub(r"\n\s*", "", pred)
if pred and pred[0] == ":":
    pred = pred[1:]
if pred and pred[-1] == ".":
    pred = pred[:-1]
if pred and pred[-1] == "/":
    pred = pred[:-1]
pred = strip_string(pred)
```

### 3c — Update `verify_correctness`

# user note - data_name could also be augmented_math and augmented_gsm8k. augmented_math should be treated as "math" and augmented_gsm8k as gsm8k. Make sure data_name is consistant with what we have

Update signature: add `data_name: str = "math"`,  `use_llm_judge` should be a last resort option, default it to false.

Pass `data_name` to `extract_answer`. Apply `strip_string` to the ground-truth before comparison — our pre-stored `expected_answer` values may not have been through the official normalizer:
```python
gt = strip_string(str(expected_answer))
```

Then call `math_equal(pred, gt)` (from `math_grader.py`, Task 4).

### 3d — Update all callers to pass `data_name`

`src/evaluation/run_evaluation.py`:
- Line 35 (dummy GT extraction): `extract_answer(p["chosen"], data_name="math")`
- Line 112 (generation loop): `extract_answer(response, data_name=p.get("source", "math"))`
- Line 114 (`verify_correctness`): add `data_name=p.get("source", "math")`

`src/training/dpo_trainer.py`:
- Line 580 (`verify_correctness`): add `data_name="math"`

`src/data/preprocessing.py`:
- Line 162 (`verify_correctness`): derive from example:
```python
source = str(example.get("problem_source", "")).lower()
data_name = "gsm8k" if "gsm" in source else "math"
return verify_correctness(generated_solution, expected_answer, data_name=data_name, problem=problem)
```

**Dependency to add if not installed:** `word2number`

### 3c — Update `verify_correctness` to add `data_name`, apply `strip_string` to GT, remove `use_llm_judge`

```python
def verify_correctness(
    generated_solution: str,
    expected_answer: str,
    data_name: str = "math",
    problem: str = "",
    logs: bool = True,
) -> bool:
```

- Pass `data_name` through to `extract_answer`
- Normalize the stored ground-truth through `strip_string` before comparison — our pre-stored `expected_answer` strings may not have been through the official normalizer: `gt = strip_string(str(expected_answer))`
- Remove `use_llm_judge` parameter entirely

### 3d — Update all callers to pass `data_name`

`src/evaluation/run_evaluation.py`:
- Line 35 (dummy GT extraction): `extract_answer(p["chosen"], data_name="math")`
- Line 112 (generation loop): `extract_answer(response, data_name=p.get("source", "math"))`
- Line 114 (`verify_correctness`): add `data_name=p.get("source", "math")`
- Line 18 import: remove `normalize_answer` if it is no longer called (check first)

`src/training/dpo_trainer.py`:
- Line 580 (`verify_correctness`): add `data_name="math"`

`src/data/preprocessing.py`:
- Line 162 (`verify_correctness`): derive from example and pass:
```python
source = str(example.get("problem_source", "")).lower()
data_name = "gsm8k" if "gsm" in source else "math"
return verify_correctness(generated_solution, expected_answer, data_name=data_name, problem=problem)
```

**`data_name` mapping:** `"gsm8k"` for any GSM8K source, `"math"` for MATH and everything else. For both values the extraction path is identical (neither is multi-choice); the only difference is the `skip_unit` flag in `strip_string`, which is `False` for both.

**Dependency to add if not installed:** `word2number`

---

## Task 4 — Replace `math_grader.py` with `math_equal`, remove LLM judge

**File:** `src/evaluation/math_grader.py`

**What to do:** Delete the current contents of `math_grader.py` entirely (removes the LLM judge, `verify_answer`, the `math_verify` tier). Replace with the following ported **verbatim** from the official `evaluation/grader.py` (https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/grader.py):

- `parse_digits(num)`
- `is_digit(num)`
- `str_to_pmatrix(input_str)`
- `math_equal(prediction, reference, include_percentage=True, is_close=True, timeout=False)`
- `math_equal_process(param)`
- `numeric_equal(prediction, reference)`
- `symbolic_equal(a, b)`
- `symbolic_equal_process(a, b, output_queue)`
- `call_with_timeout(func, *args, timeout=1, **kwargs)`

Do not modify any of these.

The `math_equal` cascade (already in the official code, no changes needed):
1. Exact string match (case-insensitive)
2. Numerical: `isclose(rel_tol=1e-4)`, also tries `value/100` and `value*100`
3. Symbolic: `parse_latex` → `parse_expr` → `latex2sympy`; checks `.equals()`, `simplify(a-b)==0`, float `N()`, matrix element-wise
4. Equation form: `a=b` on both sides → normalize to `a-b`
5. `call_with_timeout`: SymPy runs in a subprocess, 1-second timeout

**Remove entirely from `math_grader.py`:** the LLM judge (`Qwen/Qwen2.5-Math-7B-Instruct` load), `verify_answer`, the `math_verify` library call, the `use_llm_judge` flag.

**Remove `use_llm_judge` from all call sites:**
- `src/evaluation/answer_extraction.py`: `verify_correctness` signature (Task 3 above already handles this)
- `src/evaluation/run_evaluation.py`: `generate_and_evaluate` call to `verify_correctness`
- `src/training/dpo_trainer.py`: `generate_with_problem_info` call to `verify_correctness` (~line 580)
- `src/data/preprocessing.py`: `_verify_correctness` call to `verify_correctness` (~line 162)

**Dependencies to add if not installed:** `sympy`, `latex2sympy2`, `regex`

---

## Task 5 — Fix `few_shot_exemplars.py`

**File:** `src/evaluation/few_shot_exemplars.py`

Full replacement. Three problems in the current file:
1. `build_8shot_prompt` uses `"\n\n"` separator — official is `"\n\n\n"` (three newlines)
2. `build_0shot_prompt` uses `"Problem:/Solution:"` format — must match `"Question:/Answer:"`
3. No MATH 4-shot exemplars exist

Replace the entire file with:
- `GSM8K_8SHOT_EXEMPLARS` — keep existing 8 entries (content matches official), no change needed
- `MATH_4SHOT_EXEMPLARS` — 4 new entries ported verbatim from `evaluation/examples.py` in the Qwen2.5-Math repo (Kevin Kangaroo, circle area, unit circle max, function inverse)
- `_SHOT_SEP = "\n\n\n"` — the official separator constant
- `build_gsm8k_prompt(problem: str) -> str` — 8-shot, `"Question: {q}\nAnswer: {a}"` per shot, sep `"\n\n\n"`, test ends with `"Question: {problem}\nAnswer:"`
- `build_math_prompt(problem: str) -> str` — 4-shot, same format
- `build_0shot_prompt(problem: str) -> str` — returns `f"Question: {problem}\nAnswer:"`

The full content of `MATH_4SHOT_EXEMPLARS` (from official `examples.py`):

```python
MATH_4SHOT_EXEMPLARS = [
    {
        "question": r"Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.",
        "answer": "Let's think step by step\nKevin hops $1/3$ of the remaining distance with every hop.\nHis first hop takes $1/3$ closer.\nFor his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$.\nFor his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$.\nIn general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop.\nWe want to find how far he has hopped after five hops.\nThis is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms.\nThus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.\nThe answer is \\frac{211}{243}",
    },
    {
        "question": r"What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?",
        "answer": "Let's think step by step\nWe rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square,\nresulting in $(x+7)^2-49 + (y-2)^2-4=10$,\nor $(x+7)^2+(y-2)^2=63$.\nThis is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$\nso the area of this region is $\\pi r^2 = \\boxed{63\\pi}$.\nThe answer is 63\\pi",
    },
    {
        "question": r"If $x^2+y^2=1$, what is the largest possible value of $|x|+|y|$?",
        "answer": "Let's think step by step\nIf $(x,y)$ lies on the circle, so does $(x,-y),$ $(-x,-y),$ and $(-x,y),$ (which all give the same value of $|x| + |y|$), so we can assume that $x \\ge 0$ and $y \\ge 0.$\nThen $|x| + |y| = x + y.$ Squaring, we get\n\\[(x + y)^2 = x^2 + 2xy + y^2 = 1 + 2xy.\\]\nNote that $(x - y)^2 \\ge 0.$\nExpanding, we get $x^2 - 2xy + y^2 \\ge 0,$ so $2xy \\le x^2 + y^2 = 1.$\nHence, $1 + 2xy \\le 2,$ which means $x + y \\le \\sqrt{2}.$\nEquality occurs when $x = y = \\frac{1}{\\sqrt{2}},$ so the maximum value of $|x| + |y|$ is $\\boxed{\\sqrt{2}}.$\nThe answer is \\sqrt{2}",
    },
    {
        "question": r"If $f(x)=\frac{ax+b}{cx+d}, abcd\not=0$ and $f(f(x))=x$ for all $x$ in the domain of $f$, what is the value of $a+d$?",
        "answer": "Let's think step by step\nThe condition $f(f(x))=x$ means that $f$ is the inverse of itself, so its graph is symmetrical about the line $y = x$.\nWith a rational function of this form, we will have two asymptotes:\na vertical one at $x=-d/c$ if $cx+d$ does not divide $ax+b$,\nand a horizontal one at $y=a/c$.\nIn order for $f$ to be its own inverse, the intersection of the asymptotes must lie on the line $y=x$\nso that it and its asymptotes reflect onto themselves.\nThis means that $-d/c=a/c$, and therefore $-d=a$ and $a+d=\\boxed{0}$.\nThe answer is 0",
    },
]
```

---

## Task 6 — Align prompt format to `"Question:/Answer:"` everywhere

Two files must change so that the format used during training matches the format used during evaluation.

**File 1: `scripts/preprocess_dpo_data.py` line 59:**
```python
# Before
def _format_prompt(problem: str) -> str:
    return f"Problem: {problem}\nSolution: "

# After
def _format_prompt(problem: str) -> str:
    return f"Question: {problem}\nAnswer: "
```

**File 2: `src/evaluation/run_evaluation.py` line 83:**
```python
# Before
def default_prompt_fn(problem: str) -> str:
    return f"Problem: {problem}\nSolution:"

# After
def default_prompt_fn(problem: str) -> str:
    return f"Question: {problem}\nAnswer:"
```

`default_prompt_fn` is the fallback used for: training-time validation problems (`build_val_problems` in `dpo_trainer.py`), dummy data evaluation, and any caller that doesn't pass an explicit `prompt_fn`. All three should use the same `"Question:/Answer:"` format.

---

## Task 7 — Fix generation parameters and per-source prompt routing

# TODO - look into this _compute_val_accuracy function

**File:** `src/evaluation/run_evaluation.py`

Three changes inside `generate_and_evaluate`:

**a) `max_new_tokens` default** — 256 truncates most MATH solutions mid-chain. Official uses 2048.
```python
# Before
def generate_and_evaluate(..., max_new_tokens: int = 256, ...):
# After
def generate_and_evaluate(..., max_new_tokens: int = 2048, ...):
```

**b) Qwen2 EOS token IDs** — without these, generation may continue past `<|im_end|>` or `<|endoftext|>`. Add to `model.generate`:
```python
out = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=[151645, 151643],  # <|im_end|> and <|endoftext|>
)
```

**c) Stop word truncation** — after decoding, truncate the response at `"\n\nQuestion:"` before passing to answer extraction. This matches the official `cot` stop word and prevents the model's own generated next-question from being treated as part of the answer:
```python
response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
response = response.split("\n\nQuestion:")[0]  # add this line
```

**d) Per-source prompt routing** — route on `p.get("source")` inside the generation loop. Each problem already has `"source": "gsm8k"` or `"source": "math"` set by `load_eval_problems_real`. The `prompt_fn` parameter stays as a fallback for dummy data and training-time val (no source field).

```python
# Replace: prompt = prompt_fn(p["problem"])
# With:
source = p.get("source")
if source == "gsm8k":
    prompt = build_gsm8k_prompt(p["problem"])
elif source == "math":
    prompt = build_math_prompt(p["problem"])
else:
    prompt = prompt_fn(p["problem"])
```

Add import at top of file:
```python
from src.evaluation.few_shot_exemplars import build_gsm8k_prompt, build_math_prompt
```

---

## Task 8 — Fix training-time validation: generation params + problem cap



**File:** `src/training/dpo_trainer.py`

### 8a — Generation parameters

The `_compute_val_accuracy` function (~line 556) generates answers during training-time validation with `max_new_tokens=256`. Increase to 512 — shorter than eval to keep training fast, long enough to not truncate most solutions:

```python
# Before
max_new_tokens=256,
# After
max_new_tokens=512,
```

Also add Qwen2 EOS token IDs to this `model.generate` call so generation stops at `<|im_end|>`:
```python
eos_token_id=[151645, 151643],
```

Training-time val stays zero-shot (`default_prompt_fn`) — no few-shot context. This is intentional: training-time val needs a consistent signal, not benchmark-accurate numbers.

### 8b — Cap validation problem count

**Why this matters — timing analysis:**

# TODO - this is very similar to generate_and_evaluate - they should share the same logic (so use either 1024 cap)
# TODO - For now - keep the 1000 problem cap for now, stratified by complexity and data source

The validation loop in `_compute_val_accuracy` (line 568) runs `model.generate` one problem at a time with no batching. `build_val_problems` (line 655) iterates the entire val DataLoader and collects every unique problem ID. With a 20% val split over a large dataset this can be thousands of problems — all of which get generated over on every epoch.

Time breakdown per validation problem, before and after this PRD:

| Step | Before (LLM judge) | After (this PRD) |
|---|---|---|
| `model.generate` (1.5B) | ~1–2s (256 tokens) | ~2–3s (512 tokens) |
| `strip_string` + numeric check | — | < 1ms |
| SymPy symbolic | ~100–500ms | ~100–500ms |
| LLM judge (7B, ~30–50% of answers) | ~3–8s + VRAM pressure | **gone** |
| **Total per problem** | **~4–10s** | **~2.5–3.5s** |

For 1000 val problems:
- **Before**: ~67 min per epoch validation
- **After without cap**: ~47 min per epoch — better, but still blocks training for nearly an hour per epoch
- **After with 250-problem cap**: ~10–15 min per epoch — feasible

The LLM judge had two costs: inference time (~3–8s per judged answer) and VRAM pressure (loading a 7B model alongside the 1.5B policy + 1.5B ref model). Removing it alone cuts validation time roughly in half and frees GPU memory. But generation is still the wall — 2–3s × thousands of problems = hours.

250–300 problems gives a statistically stable accuracy signal per epoch. The signal doesn't improve meaningfully beyond that for a training-time check.

**Change:** Add `max_val_problems` parameter to `build_val_problems` with a default of `250`. Stop collecting new problems once the cap is reached:

```python
def build_val_problems(
    val_loader: DataLoader,
    problem_index: dict,
    tokenizer: PreTrainedTokenizer,
    prompt_fn: Callable = default_prompt_fn,
    max_val_problems: int = 250,          # add this
) -> list[dict]:
    seen_problem_ids = set()
    val_problems = []

    for batch in tqdm(val_loader, desc="Building val problems"):
        problem_ids = batch['problem_id'].tolist()
        for pid in problem_ids:
            if pid not in seen_problem_ids:
                seen_problem_ids.add(pid)
                # ... existing logic to build problem dict ...
                val_problems.append(...)
                if len(val_problems) >= max_val_problems:   # add this
                    return val_problems                       # add this

    return val_problems
```

The cap is applied at problem-collection time, not post-hoc, so the DataLoader loop also exits early.

---

## Task 9 — Re-run preprocessing to regenerate tokens.pt

**Why:** Two things changed that invalidate the existing `tokens.pt`:
1. `MODEL_NAME` changed → different tokenizer vocabulary (Task 1)
2. `_format_prompt` changed (`"Problem:/Solution:"` → `"Question:/Answer:"`) → different token IDs for the prompt prefix (Task 6)

```bash
python scripts/preprocess_dpo_data.py --force
```

`--force` is required to bypass the "already exists" early-exit check. The script reads `MODEL_NAME` from config and overwrites `tokens.pt` in `data/processed_dpo_dataset/`.

**Verify:** Confirm `data/processed_dpo_dataset/tokens.pt` has a newer modification timestamp after the run.

---

## Few-shot: training vs evaluation — clarification

Few-shot exemplars are **evaluation-only**. The training data is tokenized as `_format_prompt(problem) + solution` — a zero-shot prompt followed by the full solution. The model learns to continue `"Question: {problem}\nAnswer:"`.

At evaluation time, the 8-shot GSM8K or 4-shot MATH prefix is prepended. The model still sees the same `"Question: {problem}\nAnswer:"` tail — the shots just provide in-context demonstration style, not a format change.

---

## Evaluation strategy: 0-shot primary, 8-shot for benchmark alignment

### Default: use 0-shot as the primary comparison metric

0-shot eval reflects actual deployment conditions and removes the in-context length-anchoring effect of the few-shot prefix. All budget-aware comparisons (baseline vs. budget DPO) should be run 0-shot unless the accuracy threshold below is not met.

**`avg_tokens_easy` under 8-shot conditions is not the real deployment efficiency number.** The exemplars anchor output length via in-context learning. 0-shot `avg_tokens_easy` is what users would actually experience.

### 0-shot viability threshold for Qwen2.5-Math-1.5B

Qwen2.5-Math-1.5B is math-specialized (trained on >1T math tokens). It should produce parseable answers zero-shot. Evaluate the raw base model 0-shot first and check:

| Benchmark | Threshold | Interpretation |
|---|---|---|
| GSM8K 0-shot | ≥ 25% | Model is parsing problems and producing formatted answers |
| MATH 0-shot | ≥ 10% | Model is attempting structured solutions |

If both thresholds are met, use 0-shot as the primary eval. 8-shot is a supplementary pass.

### When 8-shot becomes the primary metric

On 0.5B general-purpose models (earlier iterations), 0-shot accuracy was near-unusable — the model did not reliably produce parseable answers and the few-shot prefix was critical to get meaningful signal. On a 1.5B math-specialized model this is unlikely, but if 0-shot falls below the thresholds above, promote 8-shot to primary.

### Summary

| Eval mode | Purpose | When to use |
|---|---|---|
| **0-shot** | Deployment-realistic efficiency measurement | Primary — always run first |
| **8-shot** | Comparison against published Qwen2.5-Math benchmark numbers | Supplementary; mandatory only if 0-shot accuracy is below threshold |

---

## Explicit non-changes

| Item | File | Reason |
|---|---|---|
| LoRA config (`r=128`, `alpha=256`, target modules) | `src/training/dpo_trainer.py` | Same attention module names in 1.5B; hyperparams can be tuned later |
| `_pad_token_if_needed` | `src/training/dpo_trainer.py` | Math base model has no dedicated pad token; `eos_token` fallback is correct |
| `correctness_flag` fast-path in `_verify_correctness` | `src/data/preprocessing.py` | Still valid — pre-computed flags are trusted; `verify_correctness` is only called as a fallback |
| Loss functions, data loaders, train/val split logic | various | Model-agnostic |
| `compute_metrics` in `run_evaluation.py` | `src/evaluation/run_evaluation.py` | Accuracy, TPCA, level 4-5 breakdown — all still correct |

---

## Acceptance Criteria

1. `src/config.py` `MODEL_NAME` equals `"Qwen/Qwen2.5-Math-1.5B"`.
2. Both `create_model` and `create_ref_model` use `torch_dtype=torch.bfloat16`.
3. `src/evaluation/answer_extraction.py` contains the ported helpers (`_fix_fracs`, `_fix_a_slash_b`, `_fix_sqrt`, `convert_word_number`, `unit_texts`, `strip_string`). `extract_answer` has `data_name` parameter and includes: Cyrillic strip, Minerva format, `"final answer is"`, pre-return cleanup (`\n`, `:`, `.`, `/`), and `strip_string` call. Excluded (not our datasets): `choice_answer_clean`, multi-choice blocks, Chinese path, `find_box`, `clean_units`, `extract_theoremqa_answer`. `verify_correctness` has `data_name` parameter, applies `strip_string` to GT, has no `use_llm_judge`. All callers pass `data_name`.
4. `src/evaluation/math_grader.py` contains only the verbatim-ported official functions (`math_equal` and all helpers). No LLM judge, no `verify_answer`, no `math_verify` library call.
5. `src/evaluation/few_shot_exemplars.py` has `MATH_4SHOT_EXEMPLARS` (4 entries), `build_gsm8k_prompt`, `build_math_prompt`, `build_0shot_prompt`, all using `"\n\n\n"` separator.
6. `_format_prompt` in `scripts/preprocess_dpo_data.py` returns `f"Question: {problem}\nAnswer: "`.
7. `default_prompt_fn` in `src/evaluation/run_evaluation.py` returns `f"Question: {problem}\nAnswer:"`.
8. `generate_and_evaluate` defaults `max_new_tokens=2048`, passes `eos_token_id=[151645, 151643]`, truncates response at `"\n\nQuestion:"`, routes to `build_gsm8k_prompt`/`build_math_prompt` by `p["source"]`.
9. `_compute_val_accuracy` in `dpo_trainer.py` uses `max_new_tokens=512` and `eos_token_id=[151645, 151643]`. `build_val_problems` has a `max_val_problems=250` cap that exits the DataLoader loop early once reached.
10. `data/processed_dpo_dataset/tokens.pt` regenerated via `python scripts/preprocess_dpo_data.py --force`.
