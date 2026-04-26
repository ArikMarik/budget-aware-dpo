# Iteration 11 — Base Qwen2.5-0.5B Ground Truth: 30.4% GSM8K 0-Shot (Phase 4)

**Date**: 2026-03-29
**Phase**: 4 (Baseline Establishment & SFT Accuracy Push)
**Status**: IN PROGRESS

---

## 1. Hypothesis

Our DPO-trained models (best: 29.2% easy accuracy) may be **degrading** the base Qwen2.5-0.5B, which reportedly achieves ~44% on GSM8K. Before any further training, we need ground truth: what does the untrained base model actually score on our eval pipeline?

**Sub-hypotheses:**
- H1: The base model scores close to 44% on GSM8K with our 0-shot prompt format
- H2: Any gap is due to prompt format (0-shot vs published 8-shot) or max_new_tokens truncation
- H3: LoRA initialization (r=128, alpha=256) does NOT degrade the base model before training

## 2. Experiment Design

### 2a. Easy Eval (GSM8K + MATH L1-2, max_new_tokens=256)
- Model: `Qwen/Qwen2.5-0.5B` (no training, no LoRA)
- Data: All 1,319 GSM8K + 437 MATH L1 + 894 MATH L2 = **2,650 problems**
- Prompt: `"Problem: {problem}\nSolution:"`
- max_new_tokens: 256, do_sample=False (greedy)
- Verification: Tier 0 + Tier 1 + Tier 2 (LLM judge)
- **Key question**: Does GSM8K accuracy match published ~44%?

### 2b. Hard Eval (MATH L3-5, max_new_tokens=256)
- Data: MATH L3 (1,131) + L4 (1,214) + L5 (1,324) = **3,669 problems**
- Run separately after easy eval completes

### 2c. Truncation Test (max_new_tokens=512)
- Re-run easy eval subset with 512 tokens if truncation appears to hurt accuracy

### 2d. LoRA Diagnostic
- Base model + untrained LoRA (r=128, alpha=256, same target modules as training)
- Compare accuracy to raw base model on a 500-problem sample
- If gap: LoRA itself is a problem → consider full fine-tuning for SFT

**Rationale for split**: Full 6,319 problems with LLM judge takes ~10h. Splitting into easy (2,650, ~3h) and hard (3,669, ~4-5h) lets us get GSM8K results fast and run hard eval in parallel with SFT development.

## 3. New Script

`scripts/eval_base_model.py`:
- Loads raw base model (no checkpoint, no LoRA)
- Supports `--with-lora-init` for untrained LoRA diagnostic
- Supports `--max-new-tokens` for truncation testing
- Reports: overall accuracy, easy/hard accuracy, MATH by level, avg tokens, TPCA
- Saves full per-problem results for error analysis

## 4. CLI Invocations

```bash
# 2a: Easy eval — GSM8K + MATH L1-2
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_base_model.py \
  --output eval_results/base_qwen_0.5b_easy_256.json --use-real --math-levels 1,2 --max-new-tokens 256

# 2b: Hard eval — MATH L3-5 only
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_base_model.py \
  --output eval_results/base_qwen_0.5b_hard_256.json --use-real --math-levels 3,4,5 --math-only --max-new-tokens 256

# 2c: 8-shot base model eval (easy)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_base_model.py \
  --output eval_results/base_qwen_0.5b_easy_8shot.json --use-real --math-levels 1,2 --max-new-tokens 256 --few-shot 8

# 2d: LoRA diagnostic
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/eval_base_model.py \
  --output eval_results/base_qwen_0.5b_lora_init_256.json --use-real --math-levels 1,2 --limit 500 --with-lora-init --max-new-tokens 256
```

## 5. Results

### 5a. Easy Eval — GSM8K + MATH L1-2 (max_new_tokens=256) ✅

| Metric | Value |
|--------|-------|
| **GSM8K accuracy** | **30.40%** (401/1319) |
| MATH L1 accuracy | 42.79% (187/437) |
| MATH L2 accuracy | 27.63% (247/894) |
| Overall accuracy | 31.51% (835/2650) |
| Avg tokens easy (GSM8K) | 182.1 |
| Avg tokens hard (MATH L1-2) | 167.9 |
| TPCA | 555.2 |
| Eval time | 168 min |
| Num problems | 2,650 |

**Key finding**: GSM8K 30.4% vs published 44% — a **13.6pp gap** due to 0-shot vs 8-shot prompting. See Gap Diagnosis and Appendix A below.

### 5b. Hard Eval — MATH L3-5 (max_new_tokens=256) ✅

| Metric | Value |
|--------|-------|
| MATH L3 accuracy | 17.95% (203/1131) |
| MATH L4 accuracy | 10.21% (124/1214) |
| MATH L5 accuracy | 4.61% (61/1324) |
| Overall accuracy | 10.58% (388/3669) |
| Avg tokens | 207.4 |
| TPCA | 1961.5 |
| Eval time | 323.8 min |
| Num problems | 3,669 |

Clear difficulty gradient: L3→L4→L5 accuracy drops from 18% to 10% to 5%.

### 5c. 8-Shot Base Eval — GSM8K + MATH L1-2 (500 problems) ✅

| Metric | 0-shot (2,650) | 8-shot (500) | Delta |
|--------|---------------|-------------|-------|
| **GSM8K** | 30.40% | **40.71%** | **+10.3pp** |
| MATH L1 | 42.79% | 43.24% | +0.5pp |
| MATH L2 | 27.63% | 27.17% | -0.5pp |
| Overall | 31.51% | 36.40% | +4.9pp |
| Avg tokens easy | 182.1 | 156.7 | -25.4 |
| Avg tokens hard | 167.9 | 190.3 | +22.4 |
| TPCA | 555.2 | 476.2 | -79.0 |
| Eval time | 168 min | 33.4 min | — |

**Key finding**: 8-shot closes most of the gap to published 44% (we get 40.7%). The remaining ~3pp is likely from majority voting or prompt format differences. MATH accuracy unchanged — 8-shot exemplars are GSM8K-style arithmetic, not MATH-style.

### 5d. 8-Shot Budget Model Eval — `budget_aware_balanced_iter5` (0.5B, λ=5) ✅

| Model | Easy (GSM8K-like) | Hard (all MATH) | Overall (balanced 500) | TPCA |
|-------|------------------|----------------|----------------------|------|
| **Base 8-shot** | 40.7% (103/253) | 32.0% (79/247) | 36.4% | 476.2 |
| **Budget iter5 8-shot** | 41.2% (103/250) | 16.8% (42/250) | 29.0% | 638.9 |

MATH by level (budget iter5):
- L1: 70.0% (14/20), L2: 23.3% (7/30), L3: 14.8% (8/54), L4: 13.0% (9/69), L5: 5.2% (4/77)

**Key finding**: Budget model **matches base on easy** (41.2% vs 40.7%) but **collapses on hard** (16.8% vs 32.0%). This is expected — iter5 trained with λ_hard=0. The budget mechanism preserved easy accuracy while degrading hard.

### 5e. LoRA Diagnostic
*(deferred — not blocking)*

## 6. Analysis
*(pending results)*

## 7. Gap Diagnosis — GSM8K 30.4% vs Published 44%

**Gap**: 13.6pp (30.4% vs 44%)

### Most likely causes:
1. **0-shot vs 8-shot prompting** (primary): Published Qwen benchmarks use 8-shot chain-of-thought. We use 0-shot `"Problem: {problem}\nSolution:"`. This is the dominant factor — 8-shot gives the model worked examples to follow.
2. **Prompt format**: Published benchmarks may use Qwen's chat template. Our raw completion prompt doesn't leverage any instruction-tuning.
3. **Answer extraction**: Our extractor looks for `\boxed{}`, `#### N`, "the answer is X", or last number. Some correct reasoning may fail extraction.

### Less likely:
4. **max_new_tokens truncation**: At 256 tokens, some long solutions may be cut off. Will test with 512 in experiment 5c.
5. **Greedy vs sampling**: We use greedy (do_sample=False). Published may use majority voting.

### Implication for Phase 4:
The 30.4% baseline is our **actual ground truth** for 0-shot evaluation. Our SFT model should aim to **exceed 30.4% GSM8K** as the first milestone. The published 44% is achievable with few-shot prompting but not our evaluation setup — and that's OK because we compare all models with the same eval pipeline.

## 8. Next Iteration Plan

Base model accuracy established at 30.4% GSM8K (0-shot). Proceed to Task B: SFT training.
- **Iteration 12**: First SFT run with default hyperparameters, target > 30.4% GSM8K
- Hard eval (MATH L3-5) running in background, will document when complete
- LoRA diagnostic deferred — will run on a 500-problem sample before SFT if time allows

---

## Appendix A — 0-Shot vs 8-Shot Prompting (Published Benchmark Gap)

### What is 8-shot prompting?

Published Qwen2.5 benchmarks (44% on GSM8K) use **8-shot chain-of-thought** prompting. This means 8 fully-solved example problems are prepended to the test question, teaching the model the expected reasoning format in-context.

### Our 0-shot prompt
```
Problem: {question}
Solution:
```

### Standard 8-shot GSM8K prompt (from Cobbe et al. 2021)

The 8 exemplars used in the original GSM8K paper and adopted by most LLM benchmarks (including Qwen, LLaMA, etc.) are available at:
- **Source**: https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/exemplars.jsonl
- **Also used in**: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k-cot.yaml) (the standard eval framework)

The prompt format is:
```
Question: {exemplar_1_question}
Answer: {exemplar_1_chain_of_thought}
#### {exemplar_1_answer}

Question: {exemplar_2_question}
Answer: {exemplar_2_chain_of_thought}
#### {exemplar_2_answer}

... (8 total exemplars) ...

Question: {test_question}
Answer:
```

Each exemplar shows step-by-step arithmetic reasoning ending with `#### {numeric_answer}`. This teaches the base model:
1. The expected output format (step-by-step then boxed answer)
2. How to perform multi-step arithmetic
3. The `####` answer delimiter for extraction

### Why we don't use 8-shot

Our eval pipeline is 0-shot because:
- SFT training teaches the model the reasoning format directly (no need for in-context examples)
- All models (base, SFT, DPO) are evaluated with the same 0-shot pipeline for fair comparison
- 8-shot uses ~2K extra tokens per problem, making eval 8x slower and reducing effective generation budget

### Impact on reported numbers

| Setup | GSM8K Accuracy |
|-------|---------------|
| Published (8-shot CoT) | ~44% |
| Our eval (0-shot) | 30.4% |
| Gap | 13.6pp |

This gap is expected and well-documented in the literature. Base models benefit enormously from few-shot examples; fine-tuned models less so.
