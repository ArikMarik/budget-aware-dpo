#!/usr/bin/env python3
"""
Data preprocessing: 4-way augmentation, complexity classification, DPO pair creation.
Skips processing and loads from disk if processed dataset already exists.
"""

import json
import logging
from pathlib import Path

from src.config import DUMMY_DATASET_PATH, PROCESSED_DATASET_PATH, USE_DUMMY_DATA
from src.data.preprocessing import (
    build_dpo_pairs,
    compute_statistics,
    load_jsonl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_input_path() -> Path:
    if USE_DUMMY_DATA:
        return DUMMY_DATASET_PATH
    return DUMMY_DATASET_PATH  # Fallback for now


def main():
    PROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    meta_path = PROCESSED_DATASET_PATH / "metadata.json"
    dataset_path = PROCESSED_DATASET_PATH / "dataset.jsonl"

    if meta_path.exists() and dataset_path.exists():
        logger.info("Processed dataset exists. Loading from disk.")
        with open(meta_path) as f:
            stats = json.load(f)
        logger.info("Stats: %s", stats)
        return

    input_path = get_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input data not found: {input_path}. Run generate_dummy_data.py first."
        )

    raw_data = load_jsonl(input_path)
    pairs = build_dpo_pairs(raw_data)
    stats = compute_statistics(pairs)

    logger.info("Dataset statistics: %s", stats)

    # Serialize
    with open(dataset_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved processed dataset to %s", PROCESSED_DATASET_PATH)


if __name__ == "__main__":
    main()
