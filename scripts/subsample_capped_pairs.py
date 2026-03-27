#!/usr/bin/env python3
"""
Subsample DPO pairs with a per-problem cap to reduce concentration.

Problem: The balanced 50k dataset has severe concentration — e.g., 25k Hard pairs
come from only 203 unique problems (avg 123 pairs/problem), causing overfitting.

Solution: Cap each problem to at most N pairs (default 30), then maintain 50/50
easy/hard balance by taking min(easy_count, hard_count) from each side.

Reads from processed_dpo_dataset_balanced/dataset.jsonl and writes to a new directory.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATASET_PATH_BALANCED
from src.data.preprocessing import split_pairs_by_problem
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

DEFAULT_MAX_PAIRS = 30
DEFAULT_OUTPUT_DIR = Path("data/processed_dpo_dataset_balanced_v2_capped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subsample DPO pairs with per-problem cap to reduce concentration."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROCESSED_DATASET_PATH_BALANCED,
        help="Source dataset directory (default: processed_dpo_dataset_balanced)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: processed_dpo_dataset_balanced_v2_capped)",
    )
    parser.add_argument(
        "--max-pairs-per-problem",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help=f"Maximum pairs to keep per unique problem (default: {DEFAULT_MAX_PAIRS})",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1, same as original)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_all_pairs(source_path: Path) -> list[dict]:
    """Load all pairs from dataset.jsonl."""
    pairs: list[dict] = []
    with open(source_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading pairs", unit=" pairs"):
            pairs.append(json.loads(line))
    return pairs


def cap_pairs_per_problem(
    pairs: list[dict],
    max_pairs: int,
    seed: int,
) -> list[dict]:
    """Cap each problem to at most max_pairs, sampling randomly within each problem."""
    rng = random.Random(seed)

    # Group by problem
    by_problem: defaultdict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_problem[p["problem"]].append(p)

    capped: list[dict] = []
    problems_capped = 0
    problems_uncapped = 0

    for problem, problem_pairs in by_problem.items():
        if len(problem_pairs) > max_pairs:
            rng.shuffle(problem_pairs)
            capped.extend(problem_pairs[:max_pairs])
            problems_capped += 1
        else:
            capped.extend(problem_pairs)
            problems_uncapped += 1

    return capped


def print_distribution_stats(
    label: str,
    pairs: list[dict],
) -> None:
    """Print distribution statistics for a set of pairs."""
    problem_counts: Counter[str] = Counter()
    for p in pairs:
        problem_counts[p["problem"]] += 1

    counts = sorted(problem_counts.values(), reverse=True)
    if not counts:
        print(f"  {label}: 0 pairs, 0 problems")
        return

    total = sum(counts)
    n_problems = len(counts)
    avg = total / n_problems
    median = counts[n_problems // 2]

    print(f"  {label}:")
    print(f"    Total pairs: {total:,}")
    print(f"    Unique problems: {n_problems:,}")
    print(f"    Pairs/problem: min={min(counts)}, max={max(counts)}, "
          f"avg={avg:.1f}, median={median}")
    if len(counts) >= 10:
        print(f"    Top 10 problem counts: {counts[:10]}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    source_path = args.source_dir / "dataset.jsonl"
    if not source_path.exists():
        logger.error("Source dataset not found: %s", source_path)
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    output_dir = args.output_dir
    # Safety: never overwrite the source
    if output_dir.resolve() == args.source_dir.resolve():
        raise ValueError("Output directory must differ from source directory!")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all pairs
    logger.info("[1/5] Loading all pairs from %s...", source_path)
    all_pairs = load_all_pairs(source_path)
    total_before = len(all_pairs)
    logger.info("  Loaded %s pairs", f"{total_before:,}")

    # Step 2: Separate by complexity
    logger.info("[2/5] Separating by complexity...")
    easy_pairs = [p for p in all_pairs if p.get("complexity", 0) == 0]
    hard_pairs = [p for p in all_pairs if p.get("complexity", 0) == 1]
    del all_pairs

    print(f"\n{'=' * 60}")
    print("BEFORE capping:")
    print(f"{'=' * 60}")
    print_distribution_stats("Easy (complexity=0)", easy_pairs)
    print_distribution_stats("Hard (complexity=1)", hard_pairs)

    # Step 3: Cap each problem
    logger.info(
        "[3/5] Capping to max %d pairs per problem...",
        args.max_pairs_per_problem,
    )
    easy_capped = cap_pairs_per_problem(easy_pairs, args.max_pairs_per_problem, args.seed)
    hard_capped = cap_pairs_per_problem(hard_pairs, args.max_pairs_per_problem, args.seed)
    del easy_pairs, hard_pairs

    # Balance: take min of both sides
    min_count = min(len(easy_capped), len(hard_capped))
    rng = random.Random(args.seed)

    if len(easy_capped) > min_count:
        rng.shuffle(easy_capped)
        easy_capped = easy_capped[:min_count]
    if len(hard_capped) > min_count:
        rng.shuffle(hard_capped)
        hard_capped = hard_capped[:min_count]

    total_after = len(easy_capped) + len(hard_capped)

    print(f"\n{'=' * 60}")
    print(f"AFTER capping (max {args.max_pairs_per_problem} pairs/problem, balanced):")
    print(f"{'=' * 60}")
    print_distribution_stats("Easy (complexity=0)", easy_capped)
    print_distribution_stats("Hard (complexity=1)", hard_capped)

    # Combine
    selected_pairs = easy_capped + hard_capped

    # Step 4: Split train/val by problem
    logger.info("[4/5] Splitting train/val by problem (%.0f%% val)...", args.val_split * 100)
    train_pairs, val_pairs = split_pairs_by_problem(
        selected_pairs, args.val_split, args.seed
    )
    logger.info("  Train: %s, Val: %s", f"{len(train_pairs):,}", f"{len(val_pairs):,}")

    # Verify no leakage
    train_problems = set(p["problem"] for p in train_pairs)
    val_problems = set(p["problem"] for p in val_pairs)
    overlap = train_problems & val_problems
    if overlap:
        logger.warning("WARNING: %d problems in both train and val!", len(overlap))
    else:
        logger.info("  No problem overlap (no data leakage)")

    # Step 5: Write output files
    logger.info("[5/5] Writing output files to %s...", output_dir)

    def write_jsonl(path: Path, pairs: list[dict], desc: str = "Saving") -> None:
        with open(path, "w", encoding="utf-8") as f:
            for p in tqdm(pairs, desc=desc, unit=" pairs"):
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    write_jsonl(output_dir / "dataset.jsonl", selected_pairs, "Saving dataset.jsonl")
    write_jsonl(output_dir / "train.jsonl", train_pairs, "Saving train.jsonl")
    write_jsonl(output_dir / "val.jsonl", val_pairs, "Saving val.jsonl")

    # Compute problem-level stats for metadata
    easy_problems = set(p["problem"] for p in easy_capped)
    hard_problems = set(p["problem"] for p in hard_capped)

    # Write metadata
    metadata = {
        "total_pairs": total_after,
        "easy_pairs": len(easy_capped),
        "hard_pairs": len(hard_capped),
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "val_split": args.val_split,
        "seed": args.seed,
        "max_pairs_per_problem": args.max_pairs_per_problem,
        "unique_problems_easy": len(easy_problems),
        "unique_problems_hard": len(hard_problems),
        "unique_problems_total": len(easy_problems | hard_problems),
        "source_dataset": str(args.source_dir),
        "source_total_pairs": total_before,
        "reduction_pct": round((1 - total_after / total_before) * 100, 1),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Source:      {args.source_dir} ({total_before:,} pairs)")
    print(f"  Output:      {output_dir} ({total_after:,} pairs)")
    print(f"  Reduction:   {total_before - total_after:,} pairs removed "
          f"({metadata['reduction_pct']}%)")
    print(f"  Easy/Hard:   {len(easy_capped):,} / {len(hard_capped):,}")
    print(f"  Train/Val:   {len(train_pairs):,} / {len(val_pairs):,}")
    print(f"  Cap:         {args.max_pairs_per_problem} pairs/problem")
    print(f"  Unique problems: {metadata['unique_problems_total']:,}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
