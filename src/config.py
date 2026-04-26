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
DUMMY_PROCESSED_DATASET_PATH = DATA_PATH / "dummy_processed_dpo_dataset"

# Real data (Phase 7+)
DATASET_PATH = DATA_PATH / "openmathinstruct.jsonl"
DATASET_PATH_LIMITED = DATA_PATH / "openmathinstruct.jsonl.limited"

PROCESSED_DATASET_PATH = DATA_PATH / "processed_dpo_dataset"
PROCESSED_DATASET_PATH_LIMITED = DATA_PATH / "processed_dpo_dataset_limited"
PROCESSED_DATASET_PATH_BALANCED = DATA_PATH / "processed_dpo_dataset_balanced"

GSM8K_TEST_PATH = DATA_PATH / "gsm8k_test.jsonl"
MATH_TEST_PATH = DATA_PATH / "math_test.jsonl"


def get_processed_dataset_path() -> Path:
    """Return processed dataset path based on DATASET_PATH / DATASET_VARIANT / USE_DUMMY_DATA."""
    override = os.environ.get("DATASET_PATH", "")
    if override:
        return Path(override)
    variant = os.environ.get("DATASET_VARIANT", "")
    if variant == "balanced":
        return PROCESSED_DATASET_PATH_BALANCED
    return DUMMY_PROCESSED_DATASET_PATH if USE_DUMMY_DATA else PROCESSED_DATASET_PATH


def get_tokens_path() -> Path:
    """Return path to all tokenized pairs."""
    return get_processed_dataset_path() / "tokens.pt"


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
    # TODO - Temp
    suffix = "_limited"
    # suffix = "" if USE_DUMMY_DATA else "_real"
    return CHECKPOINT_DIR / f"budget_aware_dpo{suffix}"


# Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
UNSLOTH_MODEL_NAME = "unsloth/Qwen2.5-0.5B"
