# Codebase & Data Audit Report

> Generated 2026-04-15. Read-only audit — no code was modified.

---

## Table of Contents

1. [Dataset Lineage & Inventory](#1-dataset-lineage--inventory)
2. [Script Inventory & Redundancy Map](#2-script-inventory--redundancy-map)
3. [Source Module Map (`src/`)](#3-source-module-map-src)
4. [Code Flow: End-to-End Pipelines](#4-code-flow-end-to-end-pipelines)
5. [Redundancy & Cleanup Recommendations](#5-redundancy--cleanup-recommendations)

---

## 1. Dataset Lineage & Inventory

### 1.1 Lineage Tree

```
RAW INPUTS
──────────
dummy_openmathinstruct.jsonl          50 examples, 13 KB       (synthetic, generate_dummy_data.py)
openmathinstruct.jsonl                13.97M examples, 18 GB   (HuggingFace, load_real_data.py)
openmathinstruct.jsonl.v0             13.97M examples, 18 GB   (backup of above — identical)
openmathinstruct.jsonl.limited        5K examples, 7.5 MB      (manual head/limit of above)
gsm8k_test.jsonl                      1,319 examples, 816 KB   (eval-only, load_real_data.py)
math_test.jsonl                       5,000 examples, 4.1 MB   (eval-only, load_real_data.py)


PROCESSED DPO DATASETS (derived from raw inputs)
─────────────────────────────────────────────────
openmathinstruct.jsonl (14M)
│
├─► processed_dpo_dataset/                 206M pairs, 316 GB    ← FULL cartesian pairing (HUGE, still growing)
│   └── no .pt files, no train/val split
│
├─► processed_dpo_dataset_real/            3.95M pairs, ~65 GB   ← real pairs only (preprocess_dpo_data.py)
│   ├── PRE-TOKENIZED (train_tokens.pt 49GB, val_tokens.pt 12GB)
│   ├── 88.3% easy / 11.7% hard
│   │
│   ├─► processed_dpo_dataset_balanced/              50K pairs, ~800 MB
│   │   ├── PRE-TOKENIZED
│   │   ├── 25K easy (max token delta) + 25K hard (longest chosen)
│   │   ├── Created by: subsample_balanced_pairs.py
│   │   │
│   │   ├─► processed_dpo_dataset_balanced_v2_capped/     7.4K pairs, 20 MB
│   │   │   ├── NOT tokenized
│   │   │   ├── max 30 pairs/problem from balanced
│   │   │   └── Created by: subsample_capped_pairs.py --max-pairs-per-problem 30
│   │   │
│   │   └─► processed_dpo_dataset_balanced_v3_capped50/   10.2K pairs, 161 MB
│   │       ├── PRE-TOKENIZED
│   │       ├── max 50 pairs/problem from balanced
│   │       └── Created by: subsample_capped_pairs.py --max-pairs-per-problem 50
│   │
│   ├─► processed_dpo_dataset_balanced_v4_capped/    50K pairs, ~871 MB   ★ PRIMARY TRAINING DATASET
│   │   ├── PRE-TOKENIZED
│   │   ├── 25K easy (cap=50/problem) + 25K hard (cap=100/problem)
│   │   ├── 6,347 unique problems
│   │   ├── Created by: subsample_capped_balanced.py
│   │   │
│   │   └─► processed_dpo_dataset_easy_only/         25K pairs, ~419 MB
│   │       ├── PRE-TOKENIZED
│   │       ├── complexity=0 filter from v4_capped
│   │       └── Created by: create_easy_only_dataset.py
│   │
│   └─► processed_dpo_dataset_real_capped100/        51.7K pairs, ~888 MB
│       ├── PRE-TOKENIZED
│       ├── max 100 pairs/problem, directly from real (not from balanced)
│       └── Created by: manual/ad-hoc capping
│
└─► processed_dpo_dataset_limited/         5K pairs, 9 MB
    ├── NOT tokenized
    ├── from openmathinstruct.jsonl.limited (5K source)
    └── Created by: preprocess_dpo_data.py


dummy_openmathinstruct.jsonl (50)
└─► dummy_processed_dpo_dataset/           13 pairs, 226 KB
    ├── PRE-TOKENIZED
    └── Created by: build_dpo_pairs_quick.py or preprocess_dpo_data.py
```

### 1.2 Dataset Detail Table

| Dataset | Pairs | Easy | Hard | Unique Problems | Pre-tokenized | Train/Val Split | Used In Iterations |
|---------|------:|-----:|-----:|----------------:|:---:|:---:|:---:|
| `processed_dpo_dataset` | 206M | 111M | 95M | 275K | No | No | Never (too large) |
| `processed_dpo_dataset_real` | 3.95M | 3.49M | 462K | 20K | Yes | 80/20 | iter 3 (collapse) |
| `processed_dpo_dataset_limited` | 5K | 955 | 4K | 4K | No | 80/20 | Early testing |
| `processed_dpo_dataset_balanced` | 50K | 25K | 25K | ~1.9K | Yes | 90/10 | iter 0, 1 |
| `processed_dpo_dataset_balanced_v2_capped` | 7.4K | 3.7K | 3.7K | 1.9K | No | 90/10 | None (superseded by v3) |
| `processed_dpo_dataset_balanced_v3_capped50` | 10.2K | 5.1K | 5.1K | 2.1K | Yes | 90/10 | iter 2 |
| **`processed_dpo_dataset_balanced_v4_capped`** | **50K** | **25K** | **25K** | **6.3K** | **Yes** | **90/10** | **iter 4-10 (primary)** |
| `processed_dpo_dataset_real_capped100` | 51.7K | 25.8K | 25.8K | 11.6K | Yes | 90/10 | iter 3 retry |
| `processed_dpo_dataset_easy_only` | 25K | 25K | 0 | ~6K | Yes | 91/9 | iter 12-14 |
| `dummy_processed_dpo_dataset` | 13 | 13 | 0 | 13 | Yes | 80/20 | Sanity checks |

### 1.3 DPO Pair Schema (all `dataset.jsonl` files)

```json
{
  "problem": "...",
  "chosen": "short correct answer or full CoT",
  "rejected": "verbose answer (easy) or incorrect answer (hard)",
  "complexity": 0 or 1,
  "rejection_reason": "length" or "correctness",
  "problem_source": "augmented_gsm8k" or "augmented_math",
  "expected_answer": "...",
  "problem_id": "hash",
  "chosen_tokens": 63,
  "rejected_tokens": 148,
  "length_ratio": 2.35
}
```

### 1.4 Which Datasets Are Actually Needed?

| Dataset | Verdict | Reason |
|---------|---------|--------|
| `openmathinstruct.jsonl` | **KEEP** | Source data |
| `openmathinstruct.jsonl.v0` | **REDUNDANT** | Identical backup of above |
| `openmathinstruct.jsonl.limited` | **REDUNDANT** | Can recreate with `head -5000` |
| `gsm8k_test.jsonl` | **KEEP** | Evaluation benchmark |
| `math_test.jsonl` | **KEEP** | Evaluation benchmark |
| `dummy_openmathinstruct.jsonl` | **KEEP** | Needed for `run_all_dummy.sh` |
| `dummy_processed_dpo_dataset/` | **KEEP** | Needed for sanity checks |
| `processed_dpo_dataset/` | **REDUNDANT** | 206M pairs, 316GB, never used in any iteration. Superseded by _real |
| `processed_dpo_dataset_real/` | **KEEP** (large) | Source for all balanced variants; 65GB with tokenized .pt |
| `processed_dpo_dataset_limited/` | **REDUNDANT** | Not used since early testing; not tokenized |
| `processed_dpo_dataset_balanced/` | **REDUNDANT** | Superseded by v4_capped; only used in iter 0-1 |
| `processed_dpo_dataset_balanced_v2_capped/` | **REDUNDANT** | Never used in any experiment |
| `processed_dpo_dataset_balanced_v3_capped50/` | **REDUNDANT** | Only used once (iter 2); superseded by v4 |
| **`processed_dpo_dataset_balanced_v4_capped/`** | **KEEP** | Primary training dataset (iter 4-14) |
| `processed_dpo_dataset_real_capped100/` | **REDUNDANT** | Used once (iter 3 retry); similar to v4_capped |
| `processed_dpo_dataset_easy_only/` | **KEEP** | Used in iter 12-14 |

**Storage savings if redundant datasets removed:** ~380 GB
(316GB from processed_dpo_dataset + 65GB from intermediate datasets)

---

## 2. Script Inventory & Redundancy Map

### 2.1 Data Pipeline Scripts

| Script | Purpose | Inputs | Outputs | Status |
|--------|---------|--------|---------|--------|
| `generate_dummy_data.py` | Create 50 synthetic examples | None | `dummy_openmathinstruct.jsonl` | KEEP — needed for dummy pipeline |
| `load_real_data.py` | Download HF datasets (OpenMathInstruct-2, GSM8K, MATH) | HuggingFace | `openmathinstruct.jsonl`, `gsm8k_test.jsonl`, `math_test.jsonl` | KEEP — data ingestion entry point |
| `enrich_training_data_with_levels.py` | Add MATH difficulty levels to existing JSONL | `openmathinstruct.jsonl` | Same file (in-place) | KEEP — but one-time operation (already applied) |
| `preprocess_dpo_data.py` | Build DPO pairs (4-way augmentation + tokenize) | Raw JSONL | `processed_dpo_dataset*/` | **OVERLAPS** with `build_dpo_pairs_quick.py` |
| `build_dpo_pairs_quick.py` | Build DPO pairs (standalone, no src imports for core logic) | Raw JSONL | `processed_dpo_dataset/` | **OVERLAPS** with `preprocess_dpo_data.py` |
| `subsample_balanced_pairs.py` | Quality-select 50K balanced pairs | `processed_dpo_dataset_real/` | `processed_dpo_dataset_balanced/` | **SUPERSEDED** by `subsample_capped_balanced.py` |
| `subsample_capped_pairs.py` | Cap pairs/problem then balance | `processed_dpo_dataset_balanced/` | `v2_capped/` or `v3_capped50/` | **SUPERSEDED** by `subsample_capped_balanced.py` |
| `subsample_capped_balanced.py` | Asymmetric cap + quality selection | `processed_dpo_dataset_real/` | `v4_capped/` | KEEP — created primary dataset |
| `create_easy_only_dataset.py` | Filter to complexity=0 | `v4_capped/` | `easy_only/` | KEEP — created easy-only dataset |

**Redundancy: `preprocess_dpo_data.py` vs `build_dpo_pairs_quick.py`**

Both scripts build DPO pairs from raw data. The differences:

| Aspect | `preprocess_dpo_data.py` | `build_dpo_pairs_quick.py` |
|--------|--------------------------|---------------------------|
| Imports `src.data.preprocessing` | Yes | No (inline logic) |
| Complexity classification | Via `classify_complexity()` from src | Inline heuristic (same logic) |
| Output format | `dataset.jsonl` + `metadata.json` | `dataset.jsonl` + `metadata.json` |
| Tokenization | Has commented-out tokenization code | No tokenization |
| Statistics | Via `compute_statistics()` from src | Inline statistics |

`build_dpo_pairs_quick.py` is a standalone copy of the logic in `preprocess_dpo_data.py` — created to avoid import issues, but they do the same thing.

**Redundancy: `subsample_balanced_pairs.py` vs `subsample_capped_pairs.py` vs `subsample_capped_balanced.py`**

These represent an evolution of subsampling strategies:
1. `subsample_balanced_pairs.py` — v1: simple quality-based selection (no per-problem cap)
2. `subsample_capped_pairs.py` — v2/v3: adds per-problem cap, but operates on already-balanced data
3. `subsample_capped_balanced.py` — v4: operates directly on raw `_real` data with asymmetric caps + quality selection

Only `subsample_capped_balanced.py` is needed going forward. The others produced datasets that are no longer used.

**Detail: `subsample_capped_balanced.py` selection criteria**

The script operates in two stages on the 3.95M pre-built DPO pairs from `processed_dpo_dataset_real/`:

**Stage 1 — Per-problem capping** (reduces concentration):
- Groups all pairs by `problem_id`
- Easy problems: if a problem has > 50 pairs, randomly shuffle and keep only 50
- Hard problems: if a problem has > 100 pairs, keep only 100
- Hard cap is higher because there are only ~276 unique hard problems; cap=50 would yield at most 13.8K pairs (short of the 25K target)
- After capping: ~300K easy pairs, ~27K hard pairs remain

**Stage 2 — Quality-based selection** (picks the best training signal):

*Easy pairs — ranked by character-length delta:*
```python
delta = len(rejected_text) - len(chosen_text)   # in characters, not tokens
```
Pairs are sorted by delta descending; top 25K kept. A pair with rejected=800 chars and chosen=50 chars (delta=750) is a much stronger "prefer concise" signal than one with delta=20. Character length is a fast proxy for token count.

*Hard pairs — ranked by longest chosen response:*
```python
key = len(chosen_text)   # in characters
```
Sorted descending; top 25K kept. The longest correct solutions represent the deepest chain-of-thought reasoning — the strongest "preserve detailed reasoning" signal.

**Balancing:** If one category has fewer than 25K (e.g., only 24K hard), the other is trimmed to match for a strict 50/50 split.

**v4_capped result statistics** (from `metadata.json`):

| Metric | Easy (C=0) | Hard (C=1) |
|--------|---:|---:|
| Selected pairs | 25,000 | 25,000 |
| Unique problems | 6,076 | 276 |
| Avg chosen tokens | 63.5 | 404.8 |
| Avg rejected tokens | 147.6 | 430.8 |
| Avg token delta (rejected − chosen) | 84.1 | 26.0 |

**Documentation gap:** The min/max character-length delta of the selected easy pairs is logged to stdout at runtime but not persisted in metadata.json or any report. Only the average token delta (84.1) is saved.

### 2.2 Training Scripts

| Script | Purpose | Loss Function | Status |
|--------|---------|---------------|--------|
| `train_sanity_check.py` | Overfit 30 examples to verify loss | Budget-aware DPO | KEEP — diagnostic |
| `training/train_baseline_dpo.py` | Standard DPO training | `standard_dpo_loss` | KEEP |
| `training/train_budget_aware_dpo.py` | Budget-aware DPO training | `budget_aware_dpo_loss` (or `simpo_loss`) | KEEP |
| `training/train_sft.py` | Supervised fine-tuning | Cross-entropy (HF Trainer) | KEEP (Phase 4) |

No redundancy among training scripts — each serves a distinct purpose.

### 2.3 Evaluation Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `eval_base_model.py` | Evaluate base Qwen (no training) | KEEP — baseline establishment |
| `eval_checkpoint.py` | Evaluate a single checkpoint (Tier 0+1+2) | KEEP — primary eval tool |
| `run_evaluation.py` | Evaluate baseline + budget checkpoints together | **OVERLAPS** with `eval_checkpoint.py` |

`run_evaluation.py` is an orchestrator that evaluates both baseline and budget checkpoints in one run. It was used in early phases but `eval_checkpoint.py` (which supports `--few-shot`, `--use-real`, etc.) has become the primary tool. `run_evaluation.py` could be removed.

### 2.4 Analysis & Visualization Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `analyze_complexity_heuristics.py` | Statistical analysis of token/complexity distributions | KEEP — produced heuristic thresholds |
| `analyze_dpo_results.py` | Analyze DPO dataset structure (pairs per problem, etc.) | KEEP — dataset diagnostics |
| `inspect_sanity_outputs.py` | Inspect sanity model outputs | KEEP — diagnostic |
| `run_visualization.py` | Generate figures from eval results | KEEP |
| `analysis/dataset_stats.py` | Analyze balanced dataset statistics | **OVERLAPS** with `analyze_dpo_results.py` |

`analysis/dataset_stats.py` is a simpler version of `analyze_dpo_results.py` — both analyze DPO dataset statistics but from slightly different angles. Redundant.

### 2.5 Shell Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `run_all_dummy.sh` | End-to-end dummy pipeline test | KEEP |
| `run_analyze_complexity_heuristics.sh` | Launch complexity analysis in background | KEEP — convenience wrapper |
| `run_full_training_analysis.sh` | Full 1M example analysis | KEEP |
| `run_phase4_pipeline.sh` | Phase 4 multi-stage pipeline | KEEP |
| `do_retroactive_commits.sh` | Git history creation | **REDUNDANT** — one-time use, already executed |

### 2.6 Script Dependency Graph

```
scripts/training/train_budget_aware_dpo.py
  └── src.training.dpo_trainer.train_dpo()
      ├── src.models.budget_aware_dpo_loss
      ├── src.models.simpo_loss (if --loss-type simpo)
      ├── src.data.preprocessing.split_pairs_by_problem()
      ├── src.evaluation.run_evaluation.generate_and_evaluate()
      └── src.evaluation.run_evaluation.compute_metrics()

scripts/training/train_baseline_dpo.py
  └── src.training.dpo_trainer.train_dpo()
      ├── src.models.standard_dpo_loss
      ├── src.data.preprocessing.split_pairs_by_problem()
      ├── src.evaluation.run_evaluation.generate_and_evaluate()
      └── src.evaluation.run_evaluation.compute_metrics()

scripts/eval_checkpoint.py
  └── src.evaluation.run_evaluation.evaluate_checkpoint()
      ├── src.evaluation.answer_extraction.extract_answer()
      ├── src.evaluation.answer_extraction.verify_correctness()
      │   └── src.evaluation.math_grader.verify_answer()
      │       └── LLM judge (Qwen2.5-Math-7B-Instruct)
      └── src.evaluation.few_shot_exemplars.build_8shot_prompt()

scripts/preprocess_dpo_data.py
  ├── src.data.preprocessing.build_dpo_pairs()
  │   ├── src.data.preprocessing.classify_complexity()
  │   └── src.data.preprocessing.label_preference()
  └── src.data.preprocessing.compute_statistics()

scripts/build_dpo_pairs_quick.py
  └── (inline logic — does NOT import src.data.preprocessing)

scripts/subsample_capped_balanced.py
  └── src.data.preprocessing (load_jsonl, split_pairs_by_problem, compute_statistics)

scripts/eval_base_model.py
  └── src.evaluation.run_evaluation (load_eval_problems, generate_and_evaluate, compute_metrics)
```

---

## 3. Source Module Map (`src/`)

### 3.1 Module Overview

```
src/
├── __init__.py
├── config.py                          ← Path management, env flags, model names
├── utils.py                           ← Logging, seeding, token counting
├── data/
│   ├── __init__.py
│   └── preprocessing.py              ← DPO pair building, complexity classification, statistics
├── models/
│   ├── __init__.py
│   ├── standard_dpo_loss.py          ← Baseline DPO loss (1 function)
│   ├── budget_aware_dpo_loss.py      ← Budget-aware DPO loss with length penalty (2 functions)
│   └── simpo_loss.py                 ← SimPO loss, reference-free (1 function)
├── training/
│   ├── __init__.py
│   └── dpo_trainer.py                ← Full training loop (35+ functions/classes, 1200+ lines)
├── evaluation/
│   ├── __init__.py
│   ├── answer_extraction.py          ← Answer parsing & normalization (5 functions)
│   ├── math_grader.py                ← 3-tier verification: string → SymPy → LLM judge
│   ├── few_shot_exemplars.py         ← 8-shot GSM8K prompt templates
│   └── run_evaluation.py             ← Evaluation pipeline (6 functions)
└── visualization/
    ├── __init__.py
    └── plot_results.py               ← Matplotlib figures (5 functions)
```

### 3.2 Import Dependency Graph (internal only)

```
config.py ──────────────────────────────────────────────────────────┐
                                                                    │
utils.py ─────────── imports config.MODEL_NAME                      │
    │                                                               │
    ▼                                                               │
data/preprocessing.py ─── imports utils (count_tokens, get_logger)  │
    │                 └── imports evaluation/answer_extraction       │
    │                                                               │
    ▼                                                               │
training/dpo_trainer.py ─── imports config                          │
    │                   ├── imports data/preprocessing              │
    │                   ├── imports evaluation/run_evaluation        │
    │                   ├── imports utils                            │
    │                   ├── imports models/standard_dpo_loss         │
    │                   ├── imports models/budget_aware_dpo_loss     │
    │                   └── imports models/simpo_loss                │
    │                                                               │
evaluation/answer_extraction.py ── imports evaluation/math_grader   │
evaluation/math_grader.py ──────── imports utils                    │
evaluation/run_evaluation.py ───── imports config, answer_extraction│
evaluation/few_shot_exemplars.py ─ (no internal imports)            │
                                                                    │
visualization/plot_results.py ──── (no internal imports)            │
                                                                    │
models/standard_dpo_loss.py ────── (no internal imports)            │
models/budget_aware_dpo_loss.py ── (no internal imports)            │
models/simpo_loss.py ──────────── (no internal imports)             │
```

### 3.3 Key Functions Per Module

#### `src/config.py`
| Function | Returns |
|----------|---------|
| `get_processed_dataset_path()` | Path based on `DATASET_VARIANT` env var |
| `get_tokens_path()` | `{dataset_path}/{split}_tokens.pt` |
| `get_train_pairs_path()` | `{dataset_path}/train.jsonl` |
| `get_val_pairs_path()` | `{dataset_path}/val.jsonl` |
| `get_baseline_output_dir()` | `{CHECKPOINT_DIR}/baseline_dpo` |
| `get_budget_aware_output_dir()` | `{CHECKPOINT_DIR}/budget_aware_dpo` |

#### `src/data/preprocessing.py`
| Function | Purpose |
|----------|---------|
| `classify_complexity(example)` | Returns 0 (easy) or 1 (hard) based on source/level/tokens |
| `label_preference(example, complexity)` | Labels as preferred/rejected with reason |
| `build_dpo_pairs(raw_data)` | Groups by problem, builds all valid pairs |
| `split_pairs_by_problem(pairs, val_split, seed)` | Stratified split ensuring no problem in both train+val |
| `compute_statistics(pairs)` | Full dataset statistics dict |
| `load_jsonl(path)` | Loads JSONL → list of dicts |

Complexity classification logic:
- GSM8K → always C=0 (easy)
- MATH level 1-2 → C=0 (easy)
- MATH level 4-5 → C=1 (hard)
- MATH level 3 or no level → token heuristic fallback (< 140 easy, > 210 hard)

#### `src/training/dpo_trainer.py` (main training engine)
| Class/Function | Purpose |
|----------------|---------|
| `TrainingConfig` | Dataclass: lr, epochs, batch_size, early_stopping params |
| `EarlyStopping` | Patience-based early stopping |
| `TokenizedDPODataset` | PyTorch Dataset loading pre-tokenized `.pt` data |
| `load_tokenized_datasets()` | Load `.pt`, filter by length_ratio, stratified split |
| `create_model()` | Qwen2.5 + LoRA (r=128, alpha=256) |
| `create_ref_model()` | Frozen reference model for DPO |
| `log_prob()` | Per-token log-prob → sum / seq_len |
| `compute_batch_loss_train()` | Forward pass → loss + metrics |
| `train_dpo()` | Main entry (called by both baseline and budget scripts) |
| `_run_gen_eval()` | Generation-based eval during training (50 easy + 50 hard) |
| `_get_best_value_for_epoch()` | Best model selection by configurable metric |

#### `src/evaluation/run_evaluation.py`
| Function | Purpose |
|----------|---------|
| `load_eval_problems(limit, use_real)` | Loads problems from DPO dataset or real GSM8K+MATH |
| `generate_and_evaluate(model, tokenizer, problems, ...)` | Generate completions + verify answers |
| `compute_metrics(results)` | Accuracy, TPCA, avg tokens by complexity |
| `evaluate_checkpoint(checkpoint_path, problems, ...)` | Load checkpoint → run full eval |

#### `src/evaluation/math_grader.py`
| Function | Tier | Method |
|----------|------|--------|
| `is_trivially_equal()` | 0 | Exact string/numeric match |
| `_verify_symbolic()` | 1 | `math-verify` library (SymPy) |
| `_verify_llm()` | 2 | Qwen2.5-Math-7B-Instruct judge |
| `verify_answer()` | All | Cascading: Tier 0 → 1 → 2 |

---

## 4. Code Flow: End-to-End Pipelines

### 4.1 Dummy Pipeline (`run_all_dummy.sh`)

```
1. generate_dummy_data.py
   └─► data/dummy_openmathinstruct.jsonl (50 examples)

2. check_model_load.py
   └─► Verifies Qwen2.5-0.5B loads and runs forward pass

3. preprocess_dpo_data.py (USE_DUMMY_DATA=1)
   ├── Reads dummy_openmathinstruct.jsonl
   ├── classify_complexity() → all Easy (GSM8K source)
   ├── label_preference() → length-based rejection
   ├── build_dpo_pairs() → 13 pairs
   └─► data/dummy_processed_dpo_dataset/{dataset,train,val}.jsonl + *.pt

4. train_sanity_check.py
   ├── Reads 30 pairs from dataset.jsonl
   ├── LoRA on Qwen2.5-0.5B, 3 epochs
   ├── budget_aware_dpo_loss
   └─► checkpoints/sanity_overfit/
```

### 4.2 Real Data Pipeline (what was actually used for experiments)

```
STEP 1: Data Ingestion
──────────────────────
load_real_data.py
├── Downloads nvidia/OpenMathInstruct-2 from HuggingFace
├── Downloads GSM8K test + MATH test sets
├── Enriches with MATH difficulty levels
└─► data/openmathinstruct.jsonl (14M), gsm8k_test.jsonl, math_test.jsonl

STEP 2: DPO Pair Building
──────────────────────────
preprocess_dpo_data.py  (or build_dpo_pairs_quick.py — same logic)
├── Reads openmathinstruct.jsonl
├── Groups solutions by problem_id
├── For each group: classify_complexity → label_preference → build pairs
├── Splits train/val by problem (no problem leakage)
└─► data/processed_dpo_dataset_real/{dataset,train,val}.jsonl + metadata.json

STEP 3: Dataset Refinement (multiple variants tried)
─────────────────────────────────────────────────────
subsample_capped_balanced.py              ★ PRIMARY PATH
├── Reads processed_dpo_dataset_real/dataset.jsonl (3.95M pairs)
├── Caps per problem: easy=50, hard=100
├── Selects top 25K easy (by char delta) + 25K hard (by longest chosen)
├── Splits by problem
├── Tokenizes (Qwen2.5-0.5B tokenizer, max_length=512)
└─► data/processed_dpo_dataset_balanced_v4_capped/{dataset,train,val}.jsonl + *_tokens.pt

create_easy_only_dataset.py               (Phase 4 variant)
├── Reads v4_capped/{train,val}.jsonl
├── Filters complexity=0 only
├── Re-tokenizes
└─► data/processed_dpo_dataset_easy_only/{train,val}.jsonl + *_tokens.pt

STEP 4: Training
────────────────
scripts/training/train_budget_aware_dpo.py \
  --lambda-easy 5.0 --lambda-hard 0.0 --kl-penalty 0.01 \
  --lr 1e-6 --batch-size 4 --max-epochs 3
├── Calls src.training.dpo_trainer.train_dpo(use_budget_aware=True)
├── Loads *_tokens.pt via TokenizedDPODataset
├── Creates model (Qwen2.5-0.5B + LoRA r=128) + frozen ref model
├── Per epoch:
│   ├── Training loop (gradient accumulation, mixed precision)
│   ├── Validation loop (DPO loss + per-complexity metrics)
│   ├── Generation eval (50 easy + 50 hard problems)
│   └── Checkpoint + best model selection
└─► checkpoints/{run_name}/epoch-{N}/ + best-model/

scripts/training/train_baseline_dpo.py    (same flow, no length penalty)

STEP 5: Evaluation
──────────────────
scripts/eval_checkpoint.py \
  --checkpoint checkpoints/{run}/best-model \
  --use-real --few-shot 8 --limit 500
├── Loads checkpoint (base model + LoRA adapter)
├── Loads 500 eval problems (balanced easy/hard from GSM8K+MATH test)
├── Generates completions (max 256 tokens)
├── For each:
│   ├── extract_answer() → parse \boxed{}, ####, etc.
│   ├── normalize_answer() → lowercase, strip, LaTeX normalization
│   └── verify_correctness() → Tier 0 → Tier 1 (math-verify) → Tier 2 (LLM judge)
├── compute_metrics() → accuracy, TPCA, avg tokens by complexity
└─► eval_results/{name}.json

STEP 6: Visualization
─────────────────────
scripts/run_visualization.py
├── Loads eval JSON files
├── plot_length_histograms()
├── plot_length_by_complexity()
├── generate_results_table()
└─► reports/figures/*.pdf + results_table.md
```

### 4.3 Data Loading in Training: What Actually Happens

When `train_dpo()` is called, it loads data via `load_tokenized_datasets()`:

```python
# 1. Load pre-tokenized data
data = torch.load(tokens_path)  # e.g., v4_capped/train_tokens.pt

# 2. Data contains per-pair:
#    - chosen_input_ids, chosen_attention_mask
#    - rejected_input_ids, rejected_attention_mask
#    - complexity (0 or 1)
#    - chosen_lengths, rejected_lengths

# 3. Optional: filter by length_ratio (|chosen| / |rejected|)
#    Keeps pairs where chosen is at least length_ratio × rejected length

# 4. Split into train/val (stratified by problem_id + complexity)
#    using split_pairs_by_problem() from preprocessing.py
```

**Important:** The `.pt` files contain pre-tokenized tensors. The `.jsonl` files contain the raw text pairs. Training loads `.pt` only. Evaluation loads `.jsonl` or test set files.

---

## 5. Redundancy & Cleanup Recommendations

### 5.1 Redundant Scripts

| Script | Redundant With | Recommendation |
|--------|---------------|----------------|
| `build_dpo_pairs_quick.py` | `preprocess_dpo_data.py` | **MERGE** — inline copy of preprocessing logic; keep only `preprocess_dpo_data.py` |
| `subsample_balanced_pairs.py` | `subsample_capped_balanced.py` | **REMOVE** — v1 subsampling, superseded by v4 approach |
| `subsample_capped_pairs.py` | `subsample_capped_balanced.py` | **REMOVE** — v2/v3 subsampling, superseded by v4 approach |
| `run_evaluation.py` | `eval_checkpoint.py` | **REMOVE** — old orchestrator; `eval_checkpoint.py` is more capable |
| `analysis/dataset_stats.py` | `analyze_dpo_results.py` | **REMOVE** — simpler version of same analysis |
| `do_retroactive_commits.sh` | N/A | **REMOVE** — one-time git script, already executed |

### 5.2 Redundant Datasets

| Dataset | Size | Recommendation |
|---------|------|----------------|
| `openmathinstruct.jsonl.v0` | 18 GB | **REMOVE** — identical backup |
| `openmathinstruct.jsonl.limited` | 7.5 MB | **REMOVE** — trivially recreatable |
| `processed_dpo_dataset/` | 316 GB | **REMOVE** — 206M pairs, never used, still growing |
| `processed_dpo_dataset_limited/` | 9 MB | **REMOVE** — not tokenized, not used since early testing |
| `processed_dpo_dataset_balanced/` | 800 MB | **REMOVE** — superseded by v4_capped |
| `processed_dpo_dataset_balanced_v2_capped/` | 20 MB | **REMOVE** — never used in experiments |
| `processed_dpo_dataset_balanced_v3_capped50/` | 161 MB | **REMOVE** — used once, superseded |
| `processed_dpo_dataset_real_capped100/` | 888 MB | **REMOVE** — similar to v4_capped, used once |
| **Total reclaimable** | **~336 GB** | |

### 5.3 Datasets to Keep

| Dataset | Size | Why |
|---------|------|-----|
| `openmathinstruct.jsonl` | 18 GB | Source data for any future reprocessing |
| `dummy_openmathinstruct.jsonl` | 13 KB | Dummy pipeline |
| `gsm8k_test.jsonl` | 816 KB | Evaluation |
| `math_test.jsonl` | 4.1 MB | Evaluation |
| `dummy_processed_dpo_dataset/` | 226 KB | Sanity checks |
| `processed_dpo_dataset_real/` | ~65 GB | Source for subsampling (could be re-derived but expensive) |
| `processed_dpo_dataset_balanced_v4_capped/` | 871 MB | Primary training dataset |
| `processed_dpo_dataset_easy_only/` | 419 MB | Phase 4 experiments |

### 5.4 Scripts to Keep (Final Set)

```
scripts/
├── generate_dummy_data.py              # Dummy data generation
├── load_real_data.py                   # HF data ingestion
├── enrich_training_data_with_levels.py # MATH level enrichment (one-time but keep for reference)
├── preprocess_dpo_data.py              # DPO pair building
├── subsample_capped_balanced.py        # Dataset refinement (v4 approach)
├── create_easy_only_dataset.py         # Easy-only filter
├── check_model_load.py                 # Model loading diagnostic
├── train_sanity_check.py               # Sanity overfit test
├── eval_base_model.py                  # Base model evaluation
├── eval_checkpoint.py                  # Checkpoint evaluation (primary)
├── analyze_complexity_heuristics.py    # Complexity analysis
├── analyze_dpo_results.py              # Dataset structure analysis
├── inspect_sanity_outputs.py           # Sanity output inspection
├── run_visualization.py                # Figure generation
├── run_all_dummy.sh                    # Dummy pipeline
├── run_analyze_complexity_heuristics.sh
├── run_full_training_analysis.sh
├── run_phase4_pipeline.sh
└── training/
    ├── train_baseline_dpo.py           # Standard DPO training
    ├── train_budget_aware_dpo.py       # Budget-aware DPO training
    └── train_sft.py                    # SFT training
```

**Removed (6 files):**
- `build_dpo_pairs_quick.py` (merged into `preprocess_dpo_data.py`)
- `subsample_balanced_pairs.py` (superseded)
- `subsample_capped_pairs.py` (superseded)
- `run_evaluation.py` (superseded by `eval_checkpoint.py`)
- `analysis/dataset_stats.py` (superseded by `analyze_dpo_results.py`)
- `do_retroactive_commits.sh` (one-time use)

### 5.5 Summary of Redundancies

| Category | Current Count | After Cleanup | Removed |
|----------|:---:|:---:|:---:|
| Raw input files | 6 | 4 | 2 (backup + limited) |
| Processed datasets | 9 dirs | 4 dirs | 5 dirs |
| Python scripts | 22 | 16 | 6 |
| Shell scripts | 5 | 4 | 1 |
| Disk space (datasets) | ~400+ GB | ~85 GB | ~336 GB |
