"""Project configuration: data paths, dummy mode, seeds."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use dummy data when USE_DUMMY_DATA=1
USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "0") == "1"

# Data storage - cluster path or local
DATA_PATH = Path(
    os.environ.get(
        "DATA_PATH",
        PROJECT_ROOT / "data",
    )
)

DUMMY_DATASET_PATH = DATA_PATH / "dummy_openmathinstruct.jsonl"
PROCESSED_DATASET_PATH = DATA_PATH / "processed_dpo_dataset"

# Real data (Phase 7+)
REAL_DATASET_PATH = DATA_PATH / "real_openmathinstruct.jsonl"
PROCESSED_DATASET_PATH_REAL = DATA_PATH / "processed_dpo_dataset_real"
GSM8K_TEST_PATH = DATA_PATH / "gsm8k_test.jsonl"
MATH_TEST_PATH = DATA_PATH / "math_test.jsonl"


def get_processed_dataset_path() -> Path:
    """Return processed dataset path based on USE_DUMMY_DATA."""
    return PROCESSED_DATASET_PATH if USE_DUMMY_DATA else PROCESSED_DATASET_PATH_REAL


def get_tokenized_train_path() -> Path:
    """Return path to pre-tokenized training data."""
    return get_processed_dataset_path() / "train_tokens.pt"


def get_tokenized_val_path() -> Path:
    """Return path to pre-tokenized validation data."""
    return get_processed_dataset_path() / "val_tokens.pt"


def get_train_pairs_path() -> Path:
    """Return path to training pairs JSONL."""
    return get_processed_dataset_path() / "train.jsonl"


def get_val_pairs_path() -> Path:
    """Return path to validation pairs JSONL."""
    return get_processed_dataset_path() / "val.jsonl"


# Checkpoints
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))


def get_baseline_output_dir() -> Path:
    """Return baseline DPO checkpoint dir (dummy vs real)."""
    suffix = "" if USE_DUMMY_DATA else "_real"
    return CHECKPOINT_DIR / f"baseline_dpo{suffix}"


def get_budget_aware_output_dir() -> Path:
    """Return budget-aware DPO checkpoint dir (dummy vs real)."""
    suffix = "" if USE_DUMMY_DATA else "_real"
    return CHECKPOINT_DIR / f"budget_aware_dpo{suffix}"

# Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
UNSLOTH_MODEL_NAME = "unsloth/Qwen2.5-0.5B"
