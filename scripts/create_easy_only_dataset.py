#!/usr/bin/env python3
"""
Create an easy-only (complexity=0) DPO dataset from the balanced v4 capped dataset.
Filters train/val to only keep easy problems, then re-tokenizes.

Usage:
  cd /storage/arik/nlp_final_project
  .venv/bin/python scripts/create_easy_only_dataset.py
"""
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import MODEL_NAME
from src.utils import get_logger, set_seed

logger = get_logger(__name__)
set_seed(42)

SOURCE_DIR = Path("data/processed_dpo_dataset_balanced_v4_capped")
OUTPUT_DIR = Path("data/processed_dpo_dataset_easy_only")
MAX_LENGTH = 512


def _format_prompt(problem: str) -> str:
    return f"Problem: {problem}\nSolution:"


def filter_easy(input_path: Path, output_path: Path) -> list[dict]:
    """Filter to complexity=0 only."""
    pairs = []
    with open(input_path) as f:
        for line in f:
            d = json.loads(line)
            if d["complexity"] == 0:
                pairs.append(d)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return pairs


def tokenize_and_save(pairs, output_dir, tokenizer, filename, max_length=MAX_LENGTH):
    chosen_ids_acc, chosen_masks_acc = [], []
    rejected_ids_acc, rejected_masks_acc = [], []
    complexities_all = []
    batch_size = 1000

    num_batches = (len(pairs) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Tokenizing"):
        start = batch_idx * batch_size
        batch = pairs[start:start + batch_size]

        chosen_combined, rejected_combined = [], []
        for p in batch:
            prompt = _format_prompt(p["problem"])
            chosen_combined.append(prompt + p["chosen"])
            rejected_combined.append(prompt + p["rejected"])
            complexities_all.append(p.get("complexity", 0))

        chosen_tok = tokenizer(chosen_combined, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        rejected_tok = tokenizer(rejected_combined, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        chosen_ids_acc.append(chosen_tok.input_ids)
        chosen_masks_acc.append(chosen_tok.attention_mask)
        rejected_ids_acc.append(rejected_tok.input_ids)
        rejected_masks_acc.append(rejected_tok.attention_mask)

    output_path = output_dir / filename
    torch.save({
        "chosen_input_ids": torch.cat(chosen_ids_acc),
        "chosen_attention_mask": torch.cat(chosen_masks_acc),
        "rejected_input_ids": torch.cat(rejected_ids_acc),
        "rejected_attention_mask": torch.cat(rejected_masks_acc),
        "complexities": torch.tensor(complexities_all, dtype=torch.long),
    }, output_path)
    logger.info("Saved %d tokenized examples to %s", len(pairs), output_path)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Filter
    logger.info("Filtering easy-only pairs from %s", SOURCE_DIR)
    train_pairs = filter_easy(SOURCE_DIR / "train.jsonl", OUTPUT_DIR / "train.jsonl")
    val_pairs = filter_easy(SOURCE_DIR / "val.jsonl", OUTPUT_DIR / "val.jsonl")
    logger.info("Train: %d easy pairs, Val: %d easy pairs", len(train_pairs), len(val_pairs))

    # Copy dataset.jsonl (easy only)
    all_pairs = train_pairs + val_pairs
    with open(OUTPUT_DIR / "dataset.jsonl", "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    # Write metadata
    meta = {
        "source": str(SOURCE_DIR),
        "filter": "complexity=0 (easy only)",
        "num_train": len(train_pairs),
        "num_val": len(val_pairs),
        "val_split": len(val_pairs) / (len(train_pairs) + len(val_pairs)),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Tokenize
    logger.info("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize_and_save(train_pairs, OUTPUT_DIR, tokenizer, "train_tokens.pt")
    tokenize_and_save(val_pairs, OUTPUT_DIR, tokenizer, "val_tokens.pt")

    logger.info("Done! Easy-only dataset at %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
