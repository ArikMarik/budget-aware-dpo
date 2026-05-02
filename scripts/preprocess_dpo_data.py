#!/usr/bin/env python3
"""
Data preprocessing: 4-way augmentation, complexity classification, DPO pair creation.
Builds all DPO pairs, tokenizes them together, and saves to tokens.pt.
The train/val split happens at training time based on problem_id.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import (
    DUMMY_DATASET_PATH,
    DUMMY_PROCESSED_DATASET_PATH,
    PROBLEM_TO_INDEX_PATH,
    PROCESSED_DATASET_PATH,
    DATASET_PATH,
    SEED,
    USE_DUMMY_DATA,
    MODEL_NAME,
    get_tokens_paths,
    OVER_LIMIT_PROBLEMS_PATH,
)
from src.data.preprocessing import (
    build_dpo_pairs,
    compute_statistics,
    load_jsonl,
)
from src.data.worker_utils import tokenize_and_save, tokenize_dpo_pairs_parallel
from src.evaluation.few_shot_exemplars import build_zero_shot_prompt
from src.utils import get_logger, get_model_tokenizer, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

MAX_LENGTH = 2048


def get_input_path() -> Path:
    if USE_DUMMY_DATA:
        return DUMMY_DATASET_PATH
    return DATASET_PATH


def get_output_path() -> Path:
    if USE_DUMMY_DATA:
        return DUMMY_PROCESSED_DATASET_PATH
    return PROCESSED_DATASET_PATH


def _write_jsonl(path: Path, pairs: list[dict], desc: str = "Saving") -> None:
    pairs_iter = tqdm(pairs, desc=desc, unit=" pairs", file=sys.stdout)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs_iter:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess DPO data - tokenize all pairs")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if files exist")
    parser.add_argument("--problem-index", type=str, default=PROBLEM_TO_INDEX_PATH, help="Path to problem_to_index.pkl")
    parser.add_argument("--max-pairs-per-problem", type=int, default=25, help="Maximum number of DPO pairs per problem (stratified by rejection_reason), enter -1 for no limit")
    parser.add_argument("--length-ratio", type=float, default=1.5, help="Minimum length ratio between preferred and rejected solutions, default: 2.0 (1.0 = no filter)")
    parser.add_argument("--over-limit-json", type=str, default=str(OVER_LIMIT_PROBLEMS_PATH), help="Path to JSON file with problems exceeding token limit")
    parser.add_argument("--batches-per-shard", type=int, default=None, help="Batches per shard (default: auto ~100K pairs/shard)")
    args = parser.parse_args()

    set_seed(SEED)
    output_dir = get_output_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_paths = get_tokens_paths()
    dataset_path = output_dir / "dataset.jsonl" # TODO - add the path to config.py and update the code accordingly
    meta_path = output_dir / "metadata.json" # TODO - add the path to config.py and update the code accordingly

    if not args or not args.force:
        if all(tok_path.exists() for tok_path in tokens_paths) and dataset_path.exists() and meta_path.exists():
            logger.info("Processed dataset exists at %s. Use --force to regenerate.", output_dir)
            with open(meta_path) as f:
                stats = json.load(f)
            logger.info("Stats: %s", stats)
            return

    input_path = get_input_path()
    if not input_path.exists():
        if USE_DUMMY_DATA:
            logger.error("Input data not found: %s. Run generate_dummy_data.py first.", input_path)
            raise FileNotFoundError(f"Input data not found: {input_path}")
        logger.error("Input data not found: %s. Run load_real_data.py first.", input_path)
        raise FileNotFoundError(f"Input data not found: {input_path}")

    logger.info("[1/4] Loading input data...")
    raw_data = load_jsonl(input_path)
    logger.info("      Loaded %s examples", f"{len(raw_data):,}")

    logger.info("[2/4] Building DPO pairs (classify, label, group)...")
    problem_to_index_path = Path(args.problem_index)
    assert problem_to_index_path.exists(), f'You must first run the load_real_data.py script, to generate the {problem_to_index_path.name} file'
    pairs = build_dpo_pairs(raw_data, problem_to_index_path=problem_to_index_path, max_per_problem=args.max_pairs_per_problem, length_ratio=args.length_ratio, over_limit_json_path=Path(args.over_limit_json))
    logger.info("      Using problem index: %s", problem_to_index_path)
    logger.info("      Built %s total pairs", f"{len(pairs):,}")

    _write_jsonl(dataset_path, pairs, desc="Saving dataset.jsonl")
    num_unique_problems = len(set(p.get("problem_id", 0) for p in pairs))
    logger.info(f"      Unique problem IDs: {num_unique_problems:,}")

    logger.info("[3/4] Tokenizing all pairs...")
    tokenizer = get_model_tokenizer(MODEL_NAME)
    tokenize_and_save(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        batch_size=20_000,
        output_paths=tokens_paths,
    )
    total_pairs = len(pairs)
    logger.info(f"      Tokenized {total_pairs:,} pairs, saved to {output_dir}")

    logger.info("[4/4] Computing and saving statistics...")
    stats = compute_statistics(dataset_path)
    stats["seed"] = SEED
    stats["total_pairs"] = total_pairs
    stats["num_unique_problems"] = num_unique_problems

    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset statistics: %s", stats)
    logger.info("Done.")


if __name__ == "__main__":
    main()
