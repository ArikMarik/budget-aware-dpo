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

# Checkpoints
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))

# Model
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
UNSLOTH_MODEL_NAME = "unsloth/Qwen2.5-0.5B"
