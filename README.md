# Budget-Aware DPO

Training small math models with budget-aware Direct Preference Optimization — short answers for easy problems, full chain-of-thought for hard problems.

---

## What This Project Does

**Budget-Aware DPO** is a training method that teaches the Qwen-2.5-0.5B model to adapt its response length based on problem difficulty:

| Problem Type | Example | Target Response |
|--------------|---------|-----------------|
| **Easy** (C=0) | GSM8K-style arithmetic | Short, direct answer |
| **Hard** (C=1) | MATH-style proofs | Full chain-of-thought |

The key innovation is a custom DPO loss with a **length penalty** term:

```
R_budget(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) − λ(C) · |y|
```

Where:
- `π_θ` = policy model (being trained)
- `π_ref` = frozen reference model
- `β` = DPO temperature (default 0.1)
- `C` = complexity flag (0=Easy, 1=Hard)
- `λ(C)` = length penalty coefficient: **high (0.05)** for Easy, **low (0.001)** for Hard
- `|y|` = response length in tokens

This encourages the model to give concise answers for simple problems while preserving detailed reasoning for complex ones.

---

## Prerequisites

- **Python** 3.11+
- **GPU** with CUDA support (training requires GPU)
- **HuggingFace account** (for downloading Qwen-2.5-0.5B model)
- **Internet access** (for downloading model weights and datasets)

---

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd nlp_final_project

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# 4. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Set PYTHONPATH (required for all scripts)
export PYTHONPATH="$PWD:$PYTHONPATH"
# Add to .bashrc for convenience: echo 'export PYTHONPATH="$PWD:$PYTHONPATH"' >> ~/.bashrc
```

### Optional: Cluster Storage Configuration

If using cluster storage (avoids filling home directory quota):

```bash
export DATA_PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/<username>"
export CHECKPOINT_DIR="$DATA_PATH/checkpoints"
export HF_HOME="$DATA_PATH/.cache/huggingface"
export PIP_CACHE_DIR="$DATA_PATH/.cache/pip"
```

---

## Quick Start

### Option A: Run Everything (Dummy Data)

```bash
cd nlp_final_project
source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run complete dummy pipeline (~2-5 minutes)
USE_DUMMY_DATA=1 ./scripts/run_all_dummy.sh
```

This runs:
1. Generate dummy data (50 synthetic examples)
2. Model load check
3. Preprocessing (4-way augmentation)
4. Sanity check (overfitting)

### Option B: Step-by-Step

```bash
# 1. Generate dummy data
USE_DUMMY_DATA=1 python -m scripts.generate_dummy_data

# 2. Verify model loads
USE_DUMMY_DATA=1 python -m scripts.check_model_load

# 3. Preprocess into DPO pairs
USE_DUMMY_DATA=1 python -m scripts.preprocess_dpo_data

# 4. Sanity check (overfit on small set)
USE_DUMMY_DATA=1 python -m scripts.train_sanity_check

# 5. Inspect sanity outputs
USE_DUMMY_DATA=1 python -m scripts.inspect_sanity_outputs
```

---

## Project Structure

```
nlp_final_project/
├── .venv/                      # Python virtual environment
├── .cursorrules                # Project rules for AI agents
├── src/
│   ├── config.py               # Paths, model name, USE_DUMMY_DATA flag
│   ├── utils.py                # Utility functions (set_seed, tiktoken)
│   ├── data/
│   │   └── preprocessing.py    # 4-way augmentation, DPO pair creation
│   ├── models/
│   │   ├── budget_aware_dpo_loss.py   # Custom loss with length penalty
│   │   └── standard_dpo_loss.py       # Standard DPO loss (baseline)
│   ├── training/
│   │   └── dpo_trainer.py      # Shared training loop
│   ├── evaluation/
│   │   ├── answer_extraction.py # Parse answers from model outputs
│   │   └── run_evaluation.py    # Accuracy, TPCA metrics
│   └── visualization/
│       └── plot_results.py     # Histograms, results tables
├── scripts/
│   ├── generate_dummy_data.py        # Create 50 synthetic examples
│   ├── check_model_load.py           # Verify Qwen2.5-0.5B loads
│   ├── load_real_data.py             # Load OpenMathInstruct-2, GSM8K, MATH
│   ├── preprocess_dpo_data.py        # Build DPO pairs (4-way augmentation)
│   ├── train_sanity_check.py         # Overfit sanity check
│   ├── inspect_sanity_outputs.py     # Inspect overfitted model outputs
│   ├── run_evaluation.py             # Evaluate checkpoints
│   ├── run_visualization.py          # Generate PDF figures
│   └── training/
│       ├── train_baseline_dpo.py       # Standard DPO baseline
│       └── train_budget_aware_dpo.py   # Budget-aware DPO (main)
├── data/
│   ├── dummy_openmathinstruct.jsonl          # Dummy data (50 examples)
│   ├── processed_dpo_dataset/                # Processed dummy DPO pairs
│   ├── real_openmathinstruct.jsonl           # Real data (gitignored)
│   ├── processed_dpo_dataset_real/           # Processed real DPO pairs
│   ├── gsm8k_test.jsonl                      # GSM8K test set
│   └── math_test.jsonl                       # MATH test set
├── checkpoints/
│   ├── baseline_dpo/                 # Dummy baseline checkpoints
│   ├── budget_aware_dpo/             # Dummy budget-aware checkpoints
│   ├── baseline_dpo_real/            # Real baseline checkpoints
│   └── budget_aware_dpo_real/        # Real budget-aware checkpoints
├── reports/
│   ├── figures/                      # PDF figures from real runs
│   └── figures_dummy/                # PDF figures from dummy runs
├── docs/
│   ├── USER_MANUAL.md                # Detailed user guide
│   ├── feature_reports/              # Phase-by-phase reports
│   └── misc/                         # Additional documentation
└── requirements.txt                  # Python dependencies
```

---

## Running the Pipeline

### Dummy vs Real Mode

All scripts respect the `USE_DUMMY_DATA` environment variable:

| Mode | Data | Checkpoints |
|------|------|-------------|
| `USE_DUMMY_DATA=1` | 50 synthetic examples | `checkpoints/*/` |
| `USE_DUMMY_DATA=0` | OpenMathInstruct-2 (5000) | `checkpoints/*_real/` |

---

### Phase 1: Generate Data

#### Dummy Data (Quick Testing)
```bash
USE_DUMMY_DATA=1 python -m scripts.generate_dummy_data
```

#### Real Data (Production)
```bash
# Load OpenMathInstruct-2 + test sets
python -m scripts.load_real_data --split train         # Full train set (~14M examples)
python -m scripts.load_real_data --split train_1M      # 1M examples
python -m scripts.load_real_data --split train_1M --limit 5000  # Load 5000 examples
python -m scripts.load_real_data --split train_2M      # 2M examples
python -m scripts.load_real_data --split train_5M      # 5M examples

# Or load only test sets (for evaluation)
python -m scripts.load_real_data --test-sets-only
```

---

### Phase 2: Preprocessing

```bash
# Dummy preprocessing
USE_DUMMY_DATA=1 python -m scripts.preprocess_dpo_data

# Real preprocessing
USE_DUMMY_DATA=0 python -m scripts.preprocess_dpo_data
```

This creates DPO pairs using **4-way augmentation**:
- **Easy + Correct:** Short answer = Preferred, verbose = Rejected
- **Hard + Correct:** Full CoT = Preferred, short = Rejected
- **Incorrect:** Always Rejected

Output: `train.jsonl`, `val.jsonl`, `train_tokens.pt`, `val_tokens.pt`, `metadata.json`

---

### Phase 3: Sanity Check (Overfitting)

```bash
# Dummy only - verify model can learn the loss
USE_DUMMY_DATA=1 python -m scripts.train_sanity_check

# Inspect outputs
USE_DUMMY_DATA=1 python -m scripts.inspect_sanity_outputs
```

Expected: Easy prompts → short tokens; Hard prompts → long CoT

---

### Phase 4: Full Training

#### Baseline DPO (Standard DPO without length penalty)
```bash
# Dummy
USE_DUMMY_DATA=1 python -m scripts.training.train_baseline_dpo --max-epochs 10 --batch-size 4

# Real
USE_DUMMY_DATA=0 python -m scripts.training.train_baseline_dpo --max-epochs 10 --batch-size 4
```

#### Budget-Aware DPO (Main Method)
```bash
# Dummy
USE_DUMMY_DATA=1 python -m scripts.training.train_budget_aware_dpo --max-epochs 10 --batch-size 4

# Real
USE_DUMMY_DATA=0 python -m scripts.training.train_budget_aware_dpo --max-epochs 10 --batch-size 4

# With W&B logging
USE_DUMMY_DATA=0 python -m scripts.training.train_budget_aware_dpo --max-epochs 10 --wandb
```

---

### Phase 5: Evaluation

```bash
# Dummy evaluation
USE_DUMMY_DATA=1 python -m scripts.run_evaluation --dummy

# Real evaluation (requires GSM8K + MATH test sets)
USE_DUMMY_DATA=0 python -m scripts.run_evaluation

# Quick test (50 problems)
USE_DUMMY_DATA=0 python -m scripts.run_evaluation --limit 50
```

Output: `checkpoints/evaluation_results_real.json`

---

### Phase 6: Visualization

```bash
# Real data (default)
python -m scripts.run_visualization

# Dummy data
python -m scripts.run_visualization --dummy
```

Output:
- `reports/figures/length_histograms_real.pdf`
- `reports/figures/length_by_complexity_real.pdf`
- `reports/figures/results_table_real.md`

---

## All CLI Commands Reference

### generate_dummy_data.py
```bash
python -m scripts.generate_dummy_data
```

### check_model_load.py
```bash
python -m scripts.check_model_load
```

### load_real_data.py
```bash
python -m scripts.load_real_data --split train         # Full train set (~14M examples)
python -m scripts.load_real_data --split train_1M      # 1M examples
python -m scripts.load_real_data --split train_1M --limit 5000  # Load 5000 examples
python -m scripts.load_real_data --split train_2M     # 2M examples
python -m scripts.load_real_data --split train_5M     # 5M examples
python -m scripts.load_real_data --test-sets-only     # Load only test sets
```

### preprocess_dpo_data.py
```bash
USE_DUMMY_DATA=1 python -m scripts.preprocess_dpo_data  # Dummy
USE_DUMMY_DATA=0 python -m scripts.preprocess_dpo_data  # Real
```

### train_sanity_check.py
```bash
USE_DUMMY_DATA=1 python -m scripts.train_sanity_check
```

### train_baseline_dpo.py
```bash
python -m scripts.training.train_baseline_dpo \
    --output-dir checkpoints/baseline_dpo \
    --max-epochs 10 \
    --batch-size 4 \
    --lr 1e-5 \
    --checkpoint-every 1 \
    --data-limit 100 \
    --seed 42 \
    --wandb \
    --dpo-beta 0.1
```

### train_budget_aware_dpo.py
```bash
python -m scripts.training.train_budget_aware_dpo \
    --output-dir checkpoints/budget_aware_dpo \
    --max-epochs 10 \
    --batch-size 4 \
    --lr 1e-5 \
    --checkpoint-every 1 \
    --data-limit 100 \
    --seed 42 \
    --wandb \
    --dpo-beta 0.1 \
    --lambda-easy 0.05 \
    --lambda-hard 0.001 \
    --early-stopping-patience 5 \
    --early-stopping-threshold 0.0
```

### run_evaluation.py
```bash
python -m scripts.run_evaluation --dummy           # Dummy evaluation
python -m scripts.run_evaluation                   # Real evaluation
python -m scripts.run_evaluation --limit 50        # Quick test
```

### run_visualization.py
```bash
python -m scripts.run_visualization    # Real data
python -m scripts.run_visualization --dummy  # Dummy data
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_DUMMY_DATA` | `0` | Use dummy (1) or real (0) data |
| `DATA_PATH` | `./data` | Data directory |
| `CHECKPOINT_DIR` | `./checkpoints` | Model checkpoints directory |
| `PYTHONPATH` | (required) | Must include project root |
| `WANDB_PROJECT` | - | W&B project name |
| `WANDB_RUN_NAME` | - | W&B run name |
| `WANDB_MODE` | `online` | W&B mode (`online`, `offline`, `disabled`) |
| `EASY_TOKEN_THRESHOLD` | `70` | Tokens below = Easy |
| `HARD_TOKEN_THRESHOLD` | `130` | Tokens above = Hard |

### Model Configuration

- **Base model:** `Qwen/Qwen2.5-0.5B`
- **Unsloth model:** `unsloth/Qwen2.5-0.5B`
- **LoRA rank:** r=128, alpha=256
- **Target modules:** q_proj, v_proj, k_proj, o_proj

---

## Key Concepts

### 1. Direct Preference Optimization (DPO)

DPO trains a model to prefer chosen responses over rejected ones. The loss uses a reference model to compute log-ratios:

```
loss = -log(σ(r_chosen - r_rejected))

where r = β · log(π_θ(y|x) / π_ref(y|x))
```

### 2. Complexity Classification

Problems are classified as:

| Flag | Name | Source | Token Count |
|------|------|--------|-------------|
| C=0 | Easy | GSM8K | < 70 tokens |
| C=1 | Hard | MATH Level 4-5 | > 130 tokens |

MATH Level 1-2 → Easy; Level 4-5 → Hard; Level 3 → token fallback

### 3. 4-Way Augmentation

For each problem, create DPO pairs based on correctness and length:

| Scenario | Chosen (Preferred) | Rejected (Dispreferred) |
|----------|-------------------|------------------------|
| Easy + Correct | Short answer | Verbose answer |
| Hard + Correct | Full CoT | Short/oversimplified |
| Incorrect | (none) | Incorrect answer |

### 4. Budget-Aware Loss

The length penalty `λ(C)` is dynamic:

```python
if complexity == 0:  # Easy
    lambda_penalty = 0.05   # Strong penalty → shorter outputs
else:  # Hard
    lambda_penalty = 0.001  # Weak penalty → preserve CoT
```

---

## Output Files

### Checkpoints
```
checkpoints/
├── baseline_dpo/
│   ├── checkpoint-500/
│   ├── training_config.json
│   └── metrics.json
├── budget_aware_dpo/
│   ├── checkpoint-500/
│   ├── training_config.json
│   └── metrics.json
├── baseline_dpo_real/
└── budget_aware_dpo_real/
```

### Evaluation Results
```
checkpoints/
├── evaluation_results_dummy.json
├── evaluation_results_real.json
├── baseline_eval_real.json
└── budget_aware_eval_real.json
```

### Figures
```
reports/figures/
├── length_histograms_real.pdf
├── length_by_complexity_real.pdf
└── results_table_real.md
```

---

## Troubleshooting

### ModuleNotFoundError: No module named 'src'

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### CUDA out of memory

```bash
# Reduce batch size
python -m scripts.training.train_budget_aware_dpo --batch-size 2

# Or limit data
python -m scripts.training.train_budget_aware_dpo --data-limit 500
```

### Real data not found

```bash
# First load the data
python -m scripts.load_real_data --split train_1M --limit 5000

# Then preprocess
USE_DUMMY_DATA=0 python -m scripts.preprocess_dpo_data
```

### GSM8K/MATH test sets not found

```bash
python -m scripts.load_real_data --test-sets-only
```

### WandB login required

```bash
wandb login
# Or use offline mode
export WANDB_MODE=offline
```

---

## Dataset Statistics

### Training Data Composition (Real)

| Category | Count | Percentage |
|----------|-------|------------|
| MATH | 40,740 | 81.5% |
| GSM8K | 9,260 | 18.5% |

### DPO Pairs (After Preprocessing)

| Type | Easy (C=0) | Hard (C=1) |
|------|------------|------------|
| Real pairs | ~742 | ~4,252 |
| Synthesized pairs | Variable | Variable |

### Token Statistics (tiktoken cl100k_base)

| Source | Avg Tokens | P25 | P50 | P75 |
|--------|------------|-----|-----|-----|
| MATH | 330.2 | 185 | 279 | 427 |
| GSM8K | 147.3 | 106 | 134 | 175 |

---

## Further Reading

- `docs/USER_MANUAL.md` — Detailed user guide
- `docs/feature_reports/` — Phase-by-phase implementation reports
- `docs/misc/knowledge_distillation_vs_dpo.md` — DPO vs distillation explanation
- `implementation_plan.md` — Full project plan

---

## License

This project is for research and educational purposes.
