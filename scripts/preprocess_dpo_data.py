#!/usr/bin/env python3
"""
Data preprocessing: 4-way augmentation, complexity classification, DPO pair creation.
Saves real_pairs and synthesized_pairs separately for analysis; combines into dataset.jsonl for training.
Splits by problem (not pair) to prevent data leakage, and pre-tokenizes for efficiency.
Skips processing if all output files already exist.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import (
    DUMMY_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    PROCESSED_DATASET_PATH_REAL,
    REAL_DATASET_PATH,
    USE_DUMMY_DATA,
    MODEL_NAME,
)
from src.data.preprocessing import (
    build_dpo_pairs,
    compute_statistics,
    load_jsonl,
    split_pairs_by_problem,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

VAL_SPLIT = 0.2
SEED = 42
MAX_LENGTH = 512


def get_input_path() -> Path:
    if USE_DUMMY_DATA:
        return DUMMY_DATASET_PATH
    return REAL_DATASET_PATH


def get_output_path() -> Path:
    if USE_DUMMY_DATA:
        return PROCESSED_DATASET_PATH
    return PROCESSED_DATASET_PATH_REAL


def _write_jsonl(path: Path, pairs: list[dict], desc: str = "Saving") -> None:
    pairs_iter = tqdm(pairs, desc=desc, unit=" pairs", file=sys.stdout)
    pairs_iter = pairs
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs_iter:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _format_prompt(problem: str) -> str:
    return f"Problem: {problem}\nSolution:"


def _format_chosen(problem: str, chosen: str) -> str:
    return f"{_format_prompt(problem)} {chosen}"


def _format_rejected(problem: str, rejected: str) -> str:
    return f"{_format_prompt(problem)} {rejected}"


def tokenize_and_save(
    pairs: list[dict],
    output_dir: Path,
    tokenizer: AutoTokenizer,
    filename: str = "tokens.pt",
    max_length: int = MAX_LENGTH,
) -> Path:
    """
    Tokenize all pairs and save as torch tensors.
    Returns path to the saved tensor file.
    """
    chosen_texts = []
    rejected_texts = []
    complexities = []

    for p in tqdm(pairs, desc="Formatting texts", unit=" pairs"):
        chosen_texts.append(_format_chosen(p["problem"], p["chosen"]))
        rejected_texts.append(_format_rejected(p["problem"], p["rejected"]))
        complexities.append(p.get("complexity", 0))

    chosen_tok = tokenizer(
        chosen_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    rejected_tok = tokenizer(
        rejected_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    complexities_t = torch.tensor(complexities, dtype=torch.long)

    output_path = output_dir / filename
    torch.save(
        {
            "chosen_input_ids": chosen_tok["input_ids"],
            "chosen_attention_mask": chosen_tok["attention_mask"],
            "rejected_input_ids": rejected_tok["input_ids"],
            "rejected_attention_mask": rejected_tok["attention_mask"],
            "complexities": complexities_t,
        },
        output_path,
    )
    return output_path


def main():
    set_seed(SEED)
    output_dir = get_output_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "dataset.jsonl"
    real_path = output_dir / "dataset_real.jsonl"
    synthesized_path = output_dir / "dataset_synthesized.jsonl"
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    train_tokens_path = output_dir / "train_tokens.pt"
    val_tokens_path = output_dir / "val_tokens.pt"
    meta_path = output_dir / "metadata.json"

    if all(p.exists() for p in (train_path, val_path, train_tokens_path, val_tokens_path, meta_path)):
        logger.info("Processed dataset with tokens exists at %s. Loading from disk.", output_dir)
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

    logger.info("[1/7] Loading input data...")
    raw_data = load_jsonl(input_path)
    logger.info("      Loaded %s examples", f"{len(raw_data):,}")

    logger.info("[2/7] Building DPO pairs (classify, label, group)...")
    real_pairs, synthesized_pairs, skipped_groups = build_dpo_pairs(raw_data)
    all_pairs = real_pairs + synthesized_pairs
    logger.info("      Built %s real + %s synthesized = %s total pairs", f"{len(real_pairs):,}", f"{len(synthesized_pairs):,}", f"{len(all_pairs):,}")

    logger.info("[3/7] Splitting by problem (train/val)...")
    train_pairs, val_pairs = split_pairs_by_problem(all_pairs, VAL_SPLIT, SEED)
    logger.info("      Split: %s train pairs, %s val pairs", f"{len(train_pairs):,}", f"{len(val_pairs):,}")

    train_problems = set(p["problem"] for p in train_pairs)
    val_problems = set(p["problem"] for p in val_pairs)
    overlap = train_problems & val_problems
    if overlap:
        logger.warning("      WARNING: %s problems in both sets!", len(overlap))
    else:
        logger.info("      No problem overlap (good - no data leakage)")

    logger.info("[4/7] Saving split datasets...")
    _write_jsonl(real_path, real_pairs, desc="Saving dataset_real.jsonl")
    _write_jsonl(synthesized_path, synthesized_pairs, desc="Saving dataset_synthesized.jsonl")
    _write_jsonl(dataset_path, all_pairs, desc="Saving dataset.jsonl")
    _write_jsonl(train_path, train_pairs, desc="Saving train.jsonl")
    _write_jsonl(val_path, val_pairs, desc="Saving val.jsonl")
    logger.info("      Saved to %s", output_dir)

    logger.info("[5/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("[6/7] Tokenizing and saving...")
    train_tokens_path = tokenize_and_save(train_pairs, output_dir, tokenizer, "train_tokens.pt")
    val_tokens_path = tokenize_and_save(val_pairs, output_dir, tokenizer, "val_tokens.pt")
    logger.info("      Saved tokens to %s", output_dir)

    logger.info("[7/7] Computing and saving statistics...")
    stats = compute_statistics(real_pairs, synthesized_pairs, skipped_groups)
    stats["val_split"] = VAL_SPLIT
    stats["seed"] = SEED
    stats["num_train_pairs"] = len(train_pairs)
    stats["num_val_pairs"] = len(val_pairs)
    stats["num_train_problems"] = len(train_problems)
    stats["num_val_problems"] = len(val_problems)
    logger.info("Dataset statistics: %s", stats)

    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
