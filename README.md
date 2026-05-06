# Budget-Aware DPO

Training math LLMs to allocate inference compute based on problem difficulty — short answers for easy problems, shorter chain-of-thought for hard ones.

---

## What This Project Does

**Budget-Aware DPO** fine-tunes `Qwen/Qwen2.5-Math-1.5B` with a modified DPO loss that adds a complexity-conditioned length penalty:

```
R_budget(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) − λ(C) · (chosen_len − rejected_len) / avg_len
```


| Symbol | Meaning                                                                     |
| ------ | --------------------------------------------------------------------------- |
| `β`    | DPO temperature (controls deviation from reference)                         |
| `C`    | Complexity flag: 0 = Easy, 1 = Hard                                         |
| `λ(C)` | Length penalty: **high (~0.28)** for Easy, **near zero (~0.0001)** for Hard |


The goal: easy problems (GSM8K-style arithmetic) get concise answers; hard problems (MATH Level 4–5) keep their chain-of-thought. Measured by **TPCA** (Tokens Per Correct Answer) and **avg_tokens_easy**.

---

## Prerequisites

- Python 3.11+
- CUDA GPU (training requires GPU; A100 / RTX 6000 Ada used in development)
- HuggingFace access to `Qwen/Qwen2.5-Math-1.5B`

---

## Installation

```bash
git clone https://github.com/ArikMarik/budget-aware-dpo
cd nlp_final_project

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Required for all scripts
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

## Quick Start

End-to-end pipeline from raw data to trained model. See **Pipeline Overview** below for full details on each step.

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Step 1 — Build FAISS similarity index for deduplication
python -m scripts.build_math_problem_index

# Step 2 — Download OpenMathInstruct-2 + GSM8K/MATH test sets
python -m scripts.load_real_data --split train
python -m scripts.load_real_data --test-sets-only

# Step 3 — Token length analysis + base model ground truth
python -m scripts.analysis.analyze_prompt_token_lengths
PYTHONUNBUFFERED=1 python scripts/eval_base_model.py \
  --output eval_results/base_model.json --use-real

# Step 4 — Build DPO preference pairs
python -m scripts.preprocess_dpo_data

# Step 5 — Hyperparameter search (TPE, ~25 h on 1 GPU)
PYTHONUNBUFFERED=1 nohup python -m scripts.optuna_hpo \
  --n-trials 20 --max-epochs 3 --train-size 1000 --val-size 250 \
  --objective efficiency --accuracy-floor 0.15 \
  --max-seq-len 1536 --val-gen-batch-size 8 --sampler tpe --wandb \
  > logs/hpo_run.log 2>&1 &

# Step 6 — Deep training with best HPO config (trial 6: efficiency=1.35)
PYTHONUNBUFFERED=1 nohup python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/deep_run_1 --max-epochs 8 --batch-size 8 \
  --lr 1.05e-6 --dpo-beta 0.229 --lambda-easy 0.112 --lambda-hard 1.37e-4 \
  --kl-penalty 0.0 --gradient-accumulation-steps 2 \
  --best-model-metric efficiency --run-name deep_run_1 --wandb \
  > logs/deep_run_1.log 2>&1 &

# Step 7 — Evaluate trained checkpoint (8-shot, 500 problems)
PYTHONUNBUFFERED=1 python scripts/run_evaluation.py \
  --checkpoint-path checkpoints/deep_run_1/best-model --few-shot

# Step 8 — Generate figures
python -m scripts.run_visualization
```

---

## Project Structure

```
nlp_final_project/
├── src/
│   ├── config.py                            # Paths, model name, env flags
│   ├── utils.py                             # set_seed, logging helpers
│   ├── data/
│   │   ├── preprocessing.py                 # DPO pair construction (4-way augmentation)
│   │   └── worker_utils.py                  # Parallel tokenization (dynamic padding)
│   ├── models/
│   │   ├── budget_aware_dpo_loss.py         # Custom loss with length penalty
│   │   ├── standard_dpo_loss.py             # Baseline DPO loss
│   │   └── simpo_loss.py                    # SimPO (reference-free) — not recommended
│   ├── training/
│   │   └── dpo_trainer.py                   # Training loop, in-training gen eval, metrics
│   ├── evaluation/
│   │   ├── run_evaluation.py                # Tiered eval (Tier 0+1+2)
│   │   ├── answer_extraction.py             # Parse answers from model outputs
│   │   ├── math_grader.py                   # Symbolic + LLM grading
│   │   └── few_shot_exemplars.py            # 8-shot prompt construction
│   ├── visualization/
│   │   └── plot_results.py                  # Figures for reports
│   └── qwen_evaluation/                     # Qwen's own grader (Tier 2 LLM judge)
│       ├── grader.py
│       ├── parser.py
│       └── utils.py
├── scripts/
│   ├── build_math_problem_index.py          # FAISS similarity index (Step 1)
│   ├── load_real_data.py                    # Download data + test sets (Step 2)
│   ├── eval_base_model.py                   # Raw base model eval (Step 3)
│   ├── eval_checkpoint.py                   # Evaluate a saved LoRA checkpoint
│   ├── eval_baseline_all_configs.sh         # Sweep eval across all baseline configs
│   ├── preprocess_dpo_data.py               # Build DPO pairs (Step 4)
│   ├── optuna_hpo.py                        # Optuna HPO sweep (Step 5)
│   ├── run_evaluation.py                    # General eval — base or checkpoint (Step 7)
│   ├── run_visualization.py                 # Generate PDF figures (Step 8)
│   ├── check_model_load.py                  # Verify model loads correctly
│   ├── subsample_balanced_pairs.py          # Subsample DPO pairs with balance constraints
│   ├── subsample_capped_balanced.py         # Subsample with per-problem caps
│   ├── subsample_capped_pairs.py            # Subsample with hard pair caps
│   ├── training/
│   │   ├── train_budget_aware_dpo.py        # Budget-aware DPO CLI (Step 6)
│   │   ├── train_baseline_dpo.py            # Baseline DPO CLI
│   │   └── train_sft.py                     # SFT (supervised fine-tuning) CLI
│   └── analysis/
│       ├── analyze_prompt_token_lengths.py  # Token length analysis (Step 3)
│       ├── analyze_dpo_results.py           # Compare eval results across runs
│       ├── analyze_percentile_bands.py      # Token length percentile banding
│       ├── analyze_similarity_search.py     # FAISS dedup similarity analysis
│       ├── analyze_complexity_heuristics.py # Complexity classifier evaluation
│       ├── dataset_stats.py                 # Dataset size and composition stats
│       ├── debug_pairs_per_problem.py       # Inspect pair counts per problem
│       ├── percentile_table_by_group.py     # Token length tables by complexity
│       └── visualize_percentile_bands.py    # Percentile band plots
├── data/
│   ├── openmathinstruct.jsonl               # Raw training source (~17G, 13.9M examples)
│   ├── gsm8k_test.jsonl                     # GSM8K test set (1,319 problems)
│   ├── math_test.jsonl                      # MATH test set (5,000 problems)
│   ├── processed_dpo_dataset/               # DPO pairs (train.jsonl, val.jsonl, metadata.json)
│   └── math_problem_index/                  # FAISS index for similarity dedup
│       ├── index.faiss
│       ├── metadata.jsonl
│       └── config.json
├── checkpoints/                             # Saved LoRA adapters, one dir per run
│   └── optuna/                              # HPO trial checkpoints + SQLite study DBs
├── eval_results/                            # Per-run evaluation JSON outputs
├── reports/
│   ├── figures/                             # PDF and PNG figures for the paper
│   └── data/                               # token_length_stats.csv, outlier lists
├── notebooks/
│   ├── token_length_analysis.ipynb          # Interactive token length exploration
│   └── visualize_optuna_study.ipynb         # HPO importance + slice plots
└── requirements.txt
```

---

## Pipeline Overview

### Step 1: Build Math Problem Index

Build a FAISS similarity index over all OpenMathInstruct problems. Used during preprocessing to deduplicate near-duplicate pairs and ensure train/test separation.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m scripts.build_math_problem_index
```

Output: `data/math_problem_index/` (FAISS index + problem mapping).

---

### Step 2: Load Real Data

Download OpenMathInstruct-2 (training source) and the official GSM8K + MATH test sets.

```bash
# Full dataset (~14M examples, 17G)
python -m scripts.load_real_data --split train

# Smaller subsets for faster iteration
python -m scripts.load_real_data --split train --limit 5000

# Test sets only (for evaluation without redownloading train)
python -m scripts.load_real_data --test-sets-only
```

Output: `data/openmathinstruct.jsonl`, `data/gsm8k_test.jsonl`, `data/math_test.jsonl`.

---

### Step 3: Analysis & Baseline Evaluation

#### 3.1 Token Length Analysis

Understand the solution length distribution before committing to length thresholds.

```bash
python -m scripts.analysis.analyze_prompt_token_lengths
```

Output: `reports/data/token_length_stats.csv`, `reports/figures/token_lengths.png`.

#### 3.2 Base Model Evaluation

Establish the ground-truth accuracy of the untrained model before any fine-tuning.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m scripts.eval_base_model \
  --output eval_results/base_model.json --use-real
```

Output: `eval_results/base_model.json`. Expect ~40% GSM8K (8-shot), ~10% MATH L4-5 (zero-shot).

---

### Step 4: Data Preprocessing

Build DPO preference pairs from OpenMathInstruct-2 using **4-way augmentation**:


| Scenario       | Chosen (preferred)                         | Rejected (dispreferred) |
| -------------- | ------------------------------------------ | ----------------------- |
| Easy + Correct | Short answer (≤ `len_ratio_easy × median`) | Verbose answer          |
| Hard + Correct | Full CoT (≥ `len_ratio_hard × median`)     | Short/oversimplified    |
| Incorrect      | —                                          | Incorrect answer        |


```bash
python -m scripts.preprocess_dpo_data
```

Output: `data/processed_dpo_dataset/train.jsonl`, `val.jsonl`, `metadata.json`.

---

### Step 5: Hyperparameter Optimization (Optuna)

Run a TPE-sampled sweep over key training hyperparameters. Objective: **efficiency** = accuracy / (mean_gen_length / 1024). Studies persist in SQLite and are resumable.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python -m scripts.optuna_hpo \
  --n-trials 20 \
  --max-epochs 3 \
  --train-size 1000 \
  --val-size 250 \
  --objective efficiency \
  --accuracy-floor 0.15 \
  --max-seq-len 1536 \
  --val-gen-batch-size 8 \
  --sampler tpe \
  --wandb \
  > logs/hpo_run.log 2>&1 &

# Monitor
grep -E "Trial [0-9]+ (done|OOM|pruned)" logs/hpo_run.log
tail -f logs/hpo_run.log | grep -E "Trial|Epoch|score|efficiency"
```

Visualize results: `notebooks/visualize_optuna_study.ipynb` — produces parameter importance plots and slice plots.

---

### Step 6: Deep Training

After selecting the best hyperparameters from the HPO sweep, run a full training job.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python -m scripts.training.train_budget_aware_dpo \
  --output-dir checkpoints/deep_run_1 \
  --max-epochs 8 \
  --batch-size 8 \
  --lr <lr_from_hpo> \
  --dpo-beta <beta_from_hpo> \
  --lambda-easy <lambda_easy_from_hpo> \
  --lambda-hard <lambda_hard_from_hpo> \
  --kl-penalty 0.0 \
  --gradient-accumulation-steps <from_hpo> \
  --best-model-metric efficiency \
  --run-name deep_run_1 \
  --wandb \
  > logs/deep_run_1.log 2>&1 &
```

Baseline DPO (no length penalty, for comparison):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m scripts.training.train_baseline_dpo \
  --output-dir checkpoints/baseline_run \
  --max-epochs 8 --batch-size 8 --lr 7.9e-7 --dpo-beta 0.1 \
  --kl-penalty 0.0 --run-name baseline_run --wandb
```

---

### Step 7: Evaluation

```bash
# Evaluate a trained LoRA checkpoint (8-shot, 500 problems)
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m scripts.run_evaluation \
  --checkpoint-path checkpoints/deep_run_1/best-model \
  --few-shot --output eval_results/deep_run_1_8shot.json

# Full test set (6,319 problems, ~5–6 hours with LLM judge)
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_evaluation \
  --checkpoint-path checkpoints/deep_run_1/best-model --few-shot
```

---

### Step 8: Visualization

Generate publication-ready figures comparing response length distributions and accuracy across models.

```bash
python -m scripts.run_visualization
```

Output: `reports/figures/length_histograms_real.pdf`, `reports/figures/length_by_complexity_real.pdf`, `reports/figures/results_table_real.md`.

---

## Answer Verification (Tiered)


| Tier | Method                       | Used When               |
| ---- | ---------------------------- | ----------------------- |
| 0    | Exact string match           | Always                  |
| 1    | math-verify symbolic (SymPy) | When Tier 0 fails       |
| 2    | LLM judge (Qwen2.5-Math-7B)  | Post-training eval only |


Tier 2 is the ground truth but slow (~5s/problem). In-training gen eval uses Tier 0+1 only.

---

## Key Results

### Base Model Baselines (Qwen2.5-Math-1.5B, no fine-tuning)


| Eval Mode           | Easy Acc  | Hard Acc          | Overall   | Avg Tok Easy | TPCA |
| ------------------- | --------- | ----------------- | --------- | ------------ | ---- |
| Raw base, zero-shot | —         | 10.6% (MATH only) | —         | —            | 1961 |
| Raw base, 8-shot    | **40.7%** | **32.0%**         | **36.4%** | 156.7        | 476  |


### Budget-Aware DPO (0.5B model, Phases 1–3)


| Run                          | Easy Acc | Hard Acc | Overall | Avg Tok Easy | TPCA | Notes         |
| ---------------------------- | -------- | -------- | ------- | ------------ | ---- | ------------- |
| Baseline + KL=0.01 (iter6)   | 27.6%    | 14.8%    | 21.2%   | 179          | 891  | Fair baseline |
| Budget DPO (iter5, λ=5)      | 29.2%    | 14.8%    | 22.0%   | 177          | 846  | Phase 1 best  |
| 1.5B Budget DPO (iter7b, E2) | 24.0%    | 27.6%    | 25.8%   | 240          | 812  | Phase 2 best  |


### Easy-Only Budget DPO (0.5B model, Phase 4, 8-shot eval)


| Run                                             | Overall   | Avg Tok Easy | TPCA    | Notes                         |
| ----------------------------------------------- | --------- | ------------ | ------- | ----------------------------- |
| Budget easy-only iter12 (λ=5.0)                 | 29.0%     | 155.7        | 633     | Matches base easy acc (41.6%) |
| Budget easy-only iter13 (token-eff select)      | 27.6%     | 154.1        | 664     | Lower overall                 |
| Budget easy-only iter14 (λ=3.0, acc floor=0.55) | **30.4%** | 155.3        | **603** | Best overall                  |


### Optuna HPO v6 — Stable Trials (Qwen2.5-Math-1.5B, study `budget_dpo_hpo_0504_150740`)

15 of 20 trials completed. Trials split into two regimes: **stable** (val_loss ≥ 0.59, genuine learning) and **collapsed** (val_loss → 0, reward divergence — high efficiency is artificial). Only stable trials produce usable checkpoints.


| Trial | Efficiency | Overall   | Easy Acc  | Easy Tokens | val_loss | λ_easy | lr      | grad_accum |
| ----- | ---------- | --------- | --------- | ----------- | -------- | ------ | ------- | ---------- |
| **5** | **1.506**  | 28.8%     | 32.8%     | 150         | 0.597    | 0.192  | 1.16e-6 | 2          |
| **9** | **1.440**  | 29.2%     | 33.6%     | 146         | 0.641    | 0.108  | 1.09e-6 | 2          |
| **6** | **1.348**  | **35.2%** | **41.6%** | 187         | 0.643    | 0.112  | 1.05e-6 | 2          |
| 12    | 1.142      | **35.6%** | **43.2%** | 231         | 0.676    | 0.103  | 5.42e-7 | 1          |
| 4     | 1.104      | 30.4%     | 32.8%     | 211         | 0.630    | 0.236  | 6.53e-7 | 2          |
| 7     | 0.971      | 33.2%     | 40.8%     | 247         | 0.663    | 0.197  | 7.10e-7 | 1          |
| 2     | 0.962      | 34.8%     | 40.8%     | 250         | 0.666    | 0.215  | 8.02e-7 | 2          |
| 8     | 0.889      | 34.8%     | 42.4%     | 295         | 0.662    | 0.242  | 4.04e-7 | 1          |


Candidates for deep training (top 3 stable by efficiency): **Trial 9** (best efficiency/accuracy balance), **Trial 5** (second), **Trial 6** (highest easy accuracy at 41.6%, matches untrained base model).

**Full test set evaluation — Trial 10 (collapsed, zero-shot):**
Accuracy 20.4%, Easy acc 29.7%, Easy tokens **79**, TPCA **84.7**. High efficiency is from reward divergence, not learned brevity — hard accuracy degrades to 16.8% on full MATH test.

---

## CLI Reference

### `scripts/optuna_hpo.py`

```
--n-trials          Number of trials to run (default: 20)
--max-epochs        Epochs per trial (default: 3)
--train-size        Unique training problems per trial (default: 1000)
--val-size          Unique validation problems per trial (default: 250)
--objective         Metric to optimize: efficiency | val_loss | accuracy (default: efficiency)
--accuracy-floor    Prune trial if acc_easy < threshold (default: 0.15)
--max-seq-len       Max token sequence length (default: 1536)
--val-gen-batch-size  Generation batch size for val eval (default: 8)
--sampler           tpe | grid | random (default: tpe)
--study-name        Name for the Optuna study (auto-generated if omitted)
--wandb             Enable W&B logging
```

### `scripts/training/train_budget_aware_dpo.py`

```
--output-dir                      Checkpoint output directory
--max-epochs                      (default: 10)
--batch-size                      (default: 4)
--lr                              Learning rate (default: 1e-5)
--dpo-beta                        DPO beta (default: 0.1)
--lambda-easy                     Length penalty for easy (default: 0.05)
--lambda-hard                     Length penalty for hard (default: 0.001)
--kl-penalty                      KL divergence penalty (default: 0.0)
--gradient-accumulation-steps     (default: 1)
--best-model-metric               val_loss | efficiency | gen_tokens_easy_with_accuracy_floor
--accuracy-floor                  Floor for gen_tokens_easy_with_accuracy_floor selector
--max-seq-len                     Max token length (default: 512)
--early-stopping-patience         (default: 5)
--data-limit                      Limit training pairs (quick tests)
--run-name                        W&B run name
--wandb                           Enable W&B logging
```

### `scripts/eval_checkpoint.py`

```
--checkpoint        Path to LoRA checkpoint directory
--output            Output JSON path
--use-real          Use real GSM8K + MATH test sets
--limit             Max problems to evaluate (default: all)
--few-shot N        N-shot prompting (default: 0-shot)
--max-new-tokens    (default: 256)
```

---

## Configuration

### Environment Variables


| Variable               | Default         | Description                                  |
| ---------------------- | --------------- | -------------------------------------------- |
| `DATA_PATH`            | `./data`        | Data directory                               |
| `CHECKPOINT_DIR`       | `./checkpoints` | Checkpoint directory                         |
| `DATASET_PATH`         | (computed)      | Override dataset path (absolute or relative) |
| `CUDA_VISIBLE_DEVICES` | —               | GPU selection                                |
| `PYTHONPATH`           | (required)      | Must include project root                    |
| `WANDB_PROJECT`        | —               | W&B project name                             |
| `WANDB_MODE`           | `online`        | `online` / `offline` / `disabled`            |


### Model


| Setting           | Value                          |
| ----------------- | ------------------------------ |
| Base model        | `Qwen/Qwen2.5-Math-1.5B`       |
| LoRA rank         | r=128, alpha=256               |
| Target modules    | q_proj, v_proj, k_proj, o_proj |
| Mixed precision   | float16 (autocast)             |
| Gradient clipping | max_norm=1.0                   |


### Complexity Classification


| Class | Name | Source         | Criterion                  |
| ----- | ---- | -------------- | -------------------------- |
| C=0   | Easy | GSM8K          | Short solution token count |
| C=1   | Hard | MATH Level 2–5 | Long solution token count  |


---

## Datasets

### Training


| Dataset                                 | Size       | Description                              |
| --------------------------------------- | ---------- | ---------------------------------------- |
| `data/processed_dpo_dataset/`      | ~606K pairs | Full mixed easy+hard (1.5B tokenization) |
| `data/processed_dpo_dataset_easy_only/` | ~25K pairs | Easy problems only (complexity=0)        |


Each dataset contains `train + validation` as .pt file, and `metadata.json`.

### Evaluation (held-out, zero overlap with training)


| Dataset                 | Problems | Source                                |
| ----------------------- | -------- | ------------------------------------- |
| `data/gsm8k_test.jsonl` | 1,319    | GSM8K official test split             |
| `data/math_test.jsonl`  | 5,000    | MATH official test split (all levels) |


---

## W&B Monitoring

- **Project**: `budget-aware-dpo`
- **Entity**: `ariksheer-tel-aviv-university`

Metrics logged per training step:

- `train/reward_diff` — learning vs. collapse signal
- `train/gradient_norm` — instability detector
- `train/complexity_0_loss` / `train/complexity_1_loss` — per-class DPO loss

Metrics logged per epoch (val):

- `val/reward_diff`, `val/accuracy`, `val/accuracy_easy`, `val/accuracy_hard`
- `val/token_easy` — average tokens on easy problems
- `val/efficiency` — the HPO objective

---

## Troubleshooting

### ModuleNotFoundError: No module named 'src'

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### CUDA out of memory during HPO

- `batch_size=16` always OOMs with Qwen2.5-Math-1.5B + LoRA. Use `batch_size=8`.
- Progressive VRAM fragmentation after many trials: restart the process.
- The `--train-size` and `--val-size` flags control data per trial and affect VRAM indirectly (longer sequences stay in memory during generation).

### CUDA out of memory during training

```bash
# Reduce batch size or gradient accumulation
python -m scripts.training.train_budget_aware_dpo --batch-size 4 --gradient-accumulation-steps 1
```

### W&B login required

```bash
wandb login
# Or disable
export WANDB_MODE=disabled
```

---

## License

Research and educational use.