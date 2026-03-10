#!/usr/bin/env python3
"""
Data preprocessing: 4-way augmentation, complexity classification, DPO pair creation.
Saves real_pairs and synthesized_pairs separately for analysis; combines into dataset.jsonl for training.
Skips processing if all output files already exist.
"""

import json
import logging
from pathlib import Path

from src.config import (
    DUMMY_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    PROCESSED_DATASET_PATH_REAL,
    REAL_DATASET_PATH,
    USE_DUMMY_DATA,
)
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
    return REAL_DATASET_PATH


def get_output_path() -> Path:
    if USE_DUMMY_DATA:
        return PROCESSED_DATASET_PATH
    return PROCESSED_DATASET_PATH_REAL


def _write_jsonl(path: Path, pairs: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    output_dir = get_output_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "dataset.jsonl"
    real_path = output_dir / "dataset_real.jsonl"
    synthesized_path = output_dir / "dataset_synthesized.jsonl"
    meta_path = output_dir / "metadata.json"

    if all(p.exists() for p in (dataset_path, real_path, synthesized_path, meta_path)):
        logger.info("Processed dataset exists at %s. Loading from disk.", output_dir)
        with open(meta_path) as f:
            stats = json.load(f)
        logger.info("Stats: %s", stats)
        return

    input_path = get_input_path()
    if not input_path.exists():
        if USE_DUMMY_DATA:
            raise FileNotFoundError(
                f"Input data not found: {input_path}. Run generate_dummy_data.py first."
            )
        raise FileNotFoundError(
            f"Input data not found: {input_path}. Run load_real_data.py first."
        )

    raw_data = load_jsonl(input_path)
    real_pairs, synthesized_pairs, skipped_groups = build_dpo_pairs(raw_data)
    all_pairs = real_pairs + synthesized_pairs
    stats = compute_statistics(real_pairs, synthesized_pairs, skipped_groups)

    logger.info("Dataset statistics: %s", stats)

    # Save separated (for analysis)
    _write_jsonl(real_path, real_pairs)
    _write_jsonl(synthesized_path, synthesized_pairs)

    # Save combined (for training)
    _write_jsonl(dataset_path, all_pairs)

    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Saved to %s: dataset.jsonl (%d), dataset_real.jsonl (%d), dataset_synthesized.jsonl (%d)",
        output_dir,
        len(all_pairs),
        len(real_pairs),
        len(synthesized_pairs),
    )


if __name__ == "__main__":
    main()
