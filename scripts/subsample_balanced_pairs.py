#!/usr/bin/env python3
"""
Subsample balanced 50k DPO pairs from processed_dpo_dataset_real.
Selects 25k Easy (max token delta) + 25k Hard (longest chosen = deepest reasoning).
Outputs train/val JSONL + tokenized .pt files ready for training.

Design decisions:
- Uses character length as proxy for ranking (correlates >0.95 with token length for
  English math text). Exact token counts computed only for the final 50k (for the report).
- Reads from dataset.jsonl (all pairs combined) and re-splits, because selection criteria
  may redistribute which problems end up in train vs val.
- Val split 10% (not 20%) — 5k validation pairs is sufficient for a curated 50k dataset.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.config import (
    MODEL_NAME,
    PROCESSED_DATASET_PATH_BALANCED,
    PROCESSED_DATASET_PATH_REAL,
)
from src.data.preprocessing import split_pairs_by_problem
from src.utils import count_tokens, get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subsample balanced 50k DPO pairs from processed dataset."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROCESSED_DATASET_PATH_REAL,
        help="Source processed dataset directory (default: processed_dpo_dataset_real)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATASET_PATH_BALANCED,
        help="Output directory for balanced dataset (default: processed_dpo_dataset_balanced)",
    )
    parser.add_argument("--num-easy", type=int, default=25000, help="Number of Easy pairs to select")
    parser.add_argument("--num-hard", type=int, default=25000, help="Number of Hard pairs to select")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Max token length for tokenization")
    parser.add_argument("--skip-tokenization", action="store_true", help="Skip .pt generation (JSONL only)")
    return parser.parse_args()


def load_all_pairs(source_path: Path) -> list[dict]:
    """Load all pairs from dataset.jsonl line by line."""
    pairs = []
    with open(source_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading pairs", unit=" pairs"):
            pairs.append(json.loads(line))
    return pairs


def select_easy_pairs(easy_pairs: list[dict], num_easy: int) -> list[dict]:
    """Select top-N easy pairs by character-length delta (proxy for token delta)."""
    logger.info("Ranking %s Easy pairs by character-length delta...", f"{len(easy_pairs):,}")
    for p in easy_pairs:
        p["_delta"] = len(p["rejected"]) - len(p["chosen"])

    easy_pairs.sort(key=lambda p: p["_delta"], reverse=True)
    selected = easy_pairs[:num_easy]

    if selected:
        logger.info(
            "  Easy delta range: max=%d, min=%d (char-length)",
            selected[0]["_delta"],
            selected[-1]["_delta"],
        )

    # Clean up temp key
    for p in easy_pairs:
        del p["_delta"]

    return selected


def select_hard_pairs(hard_pairs: list[dict], num_hard: int) -> list[dict]:
    """Select top-N hard pairs by longest chosen response (deepest reasoning)."""
    logger.info("Ranking %s Hard pairs by chosen length...", f"{len(hard_pairs):,}")

    if len(hard_pairs) < num_hard:
        logger.warning(
            "Only %d Hard pairs available (requested %d). Taking all.",
            len(hard_pairs),
            num_hard,
        )
        return hard_pairs

    hard_pairs.sort(key=lambda p: len(p["chosen"]), reverse=True)
    selected = hard_pairs[:num_hard]

    if selected:
        logger.info(
            "  Hard chosen-length range: max=%d, min=%d (chars)",
            len(selected[0]["chosen"]),
            len(selected[-1]["chosen"]),
        )

    return selected


def compute_token_stats(pairs: list[dict]) -> dict:
    """Compute average token counts for a set of pairs using the Qwen tokenizer."""
    chosen_tokens = []
    rejected_tokens = []
    for p in tqdm(pairs, desc="Computing token stats", unit=" pairs"):
        ct = count_tokens(p["chosen"])
        rt = count_tokens(p["rejected"])
        chosen_tokens.append(ct)
        rejected_tokens.append(rt)

    avg_chosen = sum(chosen_tokens) / len(chosen_tokens) if chosen_tokens else 0
    avg_rejected = sum(rejected_tokens) / len(rejected_tokens) if rejected_tokens else 0
    avg_delta = sum(rt - ct for ct, rt in zip(chosen_tokens, rejected_tokens)) / len(chosen_tokens) if chosen_tokens else 0

    return {
        "avg_chosen_tokens": round(avg_chosen, 1),
        "avg_rejected_tokens": round(avg_rejected, 1),
        "avg_delta_tokens": round(avg_delta, 1),
    }


def _write_jsonl(path: Path, pairs: list[dict], desc: str = "Saving") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in tqdm(pairs, desc=desc, unit=" pairs"):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def _format_prompt(problem: str) -> str:
    return f"Problem: {problem}\nSolution: "


def tokenize_and_save(
    pairs: list[dict],
    output_dir: Path,
    tokenizer: PreTrainedTokenizer,
    filename: str = "tokens.pt",
    max_length: int = MAX_LENGTH,
    batch_size: int = 1000,
) -> Path:
    """Tokenize pairs and save as .pt file (mirrors preprocess_dpo_data.py)."""
    chosen_ids_acc, chosen_masks_acc = [], []
    rejected_ids_acc, rejected_masks_acc = [], []
    complexities_all = []

    num_batches = (len(pairs) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Tokenizing {filename}", unit=" batches"):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_pairs = pairs[start_idx:end_idx]

        chosen_combined, rejected_combined = [], []
        for pair in batch_pairs:
            prompt_text = _format_prompt(pair["problem"])
            chosen_combined.append(prompt_text + pair["chosen"])
            rejected_combined.append(prompt_text + pair["rejected"])
            complexities_all.append(pair.get("complexity", 0))

        chosen_tok = tokenizer(
            chosen_combined, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        rejected_tok = tokenizer(
            rejected_combined, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )

        chosen_ids_acc.append(chosen_tok.input_ids)
        chosen_masks_acc.append(chosen_tok.attention_mask)
        rejected_ids_acc.append(rejected_tok.input_ids)
        rejected_masks_acc.append(rejected_tok.attention_mask)

    output_path = output_dir / filename
    torch.save(
        {
            "chosen_input_ids": torch.cat(chosen_ids_acc),
            "chosen_attention_mask": torch.cat(chosen_masks_acc),
            "rejected_input_ids": torch.cat(rejected_ids_acc),
            "rejected_attention_mask": torch.cat(rejected_masks_acc),
            "complexities": torch.tensor(complexities_all, dtype=torch.long),
        },
        output_path,
    )
    return output_path


def print_summary(
    source_total: int,
    easy_stats: dict,
    hard_stats: dict,
    num_easy: int,
    num_hard: int,
    num_train: int,
    num_val: int,
    seed: int,
    output_dir: Path,
) -> None:
    total = num_easy + num_hard
    print(f"\n{'=' * 50}")
    print("Subsampling Summary Report")
    print(f"{'=' * 50}")
    print(f"Source: processed_dpo_dataset_real ({source_total:,} pairs)")
    print(f"Output: {output_dir.name} ({total:,} pairs)")
    print()
    print(f"{'Category':<14} | {'Count':>6} | {'Avg Chosen Tok':>14} | {'Avg Rejected Tok':>16} | {'Avg Delta':>9}")
    print(f"{'-'*14}-+-{'-'*6}-+-{'-'*14}-+-{'-'*16}-+-{'-'*9}")
    print(
        f"{'Easy (C=0)':<14} | {num_easy:>6,} | "
        f"{easy_stats['avg_chosen_tokens']:>14.1f} | "
        f"{easy_stats['avg_rejected_tokens']:>16.1f} | "
        f"{easy_stats['avg_delta_tokens']:>9.1f}"
    )
    print(
        f"{'Hard (C=1)':<14} | {num_hard:>6,} | "
        f"{hard_stats['avg_chosen_tokens']:>14.1f} | "
        f"{hard_stats['avg_rejected_tokens']:>16.1f} | "
        f"{'N/A':>9}"
    )
    print()
    print(f"Split: Train {num_train:,} / Val {num_val:,} (90/10 by problem, stratified)")
    print(f"Seed: {seed}")
    print(f"Output dir: {output_dir}/")
    print(f"{'=' * 50}\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    source_path = args.source_dir / "dataset.jsonl"
    if not source_path.exists():
        logger.error("Source dataset not found: %s", source_path)
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all pairs
    logger.info("[1/8] Loading all pairs from %s...", source_path)
    all_pairs = load_all_pairs(source_path)
    source_total = len(all_pairs)
    logger.info("  Loaded %s pairs", f"{source_total:,}")

    # Step 2: Separate by complexity
    logger.info("[2/8] Separating by complexity...")
    easy_pairs = [p for p in all_pairs if p.get("complexity", 0) == 0]
    hard_pairs = [p for p in all_pairs if p.get("complexity", 0) == 1]
    logger.info("  Easy: %s, Hard: %s", f"{len(easy_pairs):,}", f"{len(hard_pairs):,}")

    # Free memory
    del all_pairs

    # Step 3: Select Easy pairs (max character-length delta)
    logger.info("[3/8] Selecting top %d Easy pairs by max delta...", args.num_easy)
    selected_easy = select_easy_pairs(easy_pairs, args.num_easy)
    del easy_pairs

    # Step 4: Select Hard pairs (longest chosen response)
    logger.info("[4/8] Selecting top %d Hard pairs by longest chosen...", args.num_hard)
    selected_hard = select_hard_pairs(hard_pairs, args.num_hard)
    del hard_pairs

    actual_easy = len(selected_easy)
    actual_hard = len(selected_hard)

    # Step 5: Combine
    logger.info("[5/8] Combining %d Easy + %d Hard = %d total", actual_easy, actual_hard, actual_easy + actual_hard)
    selected_pairs = selected_easy + selected_hard

    # Step 6: Split 90/10 by problem
    logger.info("[6/8] Splitting 90/10 by problem (stratified)...")
    train_pairs, val_pairs = split_pairs_by_problem(selected_pairs, args.val_split, args.seed)
    logger.info("  Train: %s, Val: %s", f"{len(train_pairs):,}", f"{len(val_pairs):,}")

    # Verify no leakage
    train_problems = set(p["problem"] for p in train_pairs)
    val_problems = set(p["problem"] for p in val_pairs)
    overlap = train_problems & val_problems
    if overlap:
        logger.warning("WARNING: %d problems in both train and val!", len(overlap))
    else:
        logger.info("  No problem overlap (good - no data leakage)")

    # Step 7: Write JSONL files
    logger.info("[7/8] Writing JSONL files...")
    _write_jsonl(output_dir / "dataset.jsonl", selected_pairs, desc="Saving dataset.jsonl")
    _write_jsonl(output_dir / "train.jsonl", train_pairs, desc="Saving train.jsonl")
    _write_jsonl(output_dir / "val.jsonl", val_pairs, desc="Saving val.jsonl")

    # Compute token stats on final selected pairs
    logger.info("Computing token statistics for selected Easy pairs...")
    easy_stats = compute_token_stats(selected_easy)
    logger.info("Computing token statistics for selected Hard pairs...")
    hard_stats = compute_token_stats(selected_hard)

    # Step 8: Tokenize (optional)
    if not args.skip_tokenization:
        logger.info("[8/8] Tokenizing and saving .pt files...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenize_and_save(train_pairs, output_dir, tokenizer, "train_tokens.pt", args.max_length)
        tokenize_and_save(val_pairs, output_dir, tokenizer, "val_tokens.pt", args.max_length)
        logger.info("  Saved tokenized data to %s", output_dir)
    else:
        logger.info("[8/8] Skipping tokenization (--skip-tokenization)")

    # Write metadata.json
    metadata = {
        "total_pairs": actual_easy + actual_hard,
        "easy_pairs": actual_easy,
        "hard_pairs": actual_hard,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "val_split": args.val_split,
        "seed": args.seed,
        "avg_chosen_tokens_easy": easy_stats["avg_chosen_tokens"],
        "avg_rejected_tokens_easy": easy_stats["avg_rejected_tokens"],
        "avg_length_delta_easy": easy_stats["avg_delta_tokens"],
        "avg_chosen_tokens_hard": hard_stats["avg_chosen_tokens"],
        "avg_rejected_tokens_hard": hard_stats["avg_rejected_tokens"],
        "selection_method_easy": "max_token_delta",
        "selection_method_hard": "longest_chosen_response",
        "source_dataset": "processed_dpo_dataset_real",
        "source_total_pairs": source_total,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    # Print summary
    print_summary(
        source_total=source_total,
        easy_stats=easy_stats,
        hard_stats=hard_stats,
        num_easy=actual_easy,
        num_hard=actual_hard,
        num_train=len(train_pairs),
        num_val=len(val_pairs),
        seed=args.seed,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
