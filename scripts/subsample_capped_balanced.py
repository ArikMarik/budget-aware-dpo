#!/usr/bin/env python3
"""
Create a balanced 50K DPO dataset with per-problem capping from the FULL source data.

Combines the quality-based selection from subsample_balanced_pairs.py with per-problem
capping to prevent any single problem from dominating the dataset.

Strategy:
- Easy pairs: cap at --easy-cap per problem (default 50), then select top 25K by max
  character-length delta (same criterion as subsample_balanced_pairs.py).
- Hard pairs: cap at --hard-cap per problem (default 100), then select top 25K by
  longest chosen response (same criterion as subsample_balanced_pairs.py).
  Hard cap is higher because there are only ~276 hard problems, so cap=50 would only
  yield ~13K pairs (not enough for 25K).

If after capping one category has fewer than its target, take all available and reduce
the other category to match (maintain 50/50 balance).

Reads from: data/processed_dpo_dataset_real/dataset.jsonl (3.9M pairs)
Writes to:  data/processed_dpo_dataset_balanced_v4_capped/
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_NAME, PROCESSED_DATASET_PATH_REAL
from src.data.preprocessing import split_pairs_by_problem
from src.utils import count_tokens, get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

MAX_LENGTH = 512
DEFAULT_OUTPUT_DIR = Path("data/processed_dpo_dataset_balanced_v4_capped")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create balanced 50K DPO dataset with per-problem capping."
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
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: processed_dpo_dataset_balanced_v4_capped)",
    )
    parser.add_argument(
        "--num-easy", type=int, default=25000, help="Target number of Easy pairs"
    )
    parser.add_argument(
        "--num-hard", type=int, default=25000, help="Target number of Hard pairs"
    )
    parser.add_argument(
        "--easy-cap",
        type=int,
        default=50,
        help="Max pairs per problem for Easy category (default: 50)",
    )
    parser.add_argument(
        "--hard-cap",
        type=int,
        default=100,
        help="Max pairs per problem for Hard category (default: 100)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Max token length for tokenization",
    )
    parser.add_argument(
        "--skip-tokenization",
        action="store_true",
        help="Skip .pt generation (JSONL only)",
    )
    return parser.parse_args()


def load_all_pairs(source_path: Path) -> list[dict]:
    """Load all pairs from dataset.jsonl line by line."""
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

    logger.info(
        "  Capped %d problems (kept %d uncapped), %d -> %d pairs",
        problems_capped,
        problems_uncapped,
        len(pairs),
        len(capped),
    )
    return capped


def select_easy_pairs(easy_pairs: list[dict], num_easy: int) -> list[dict]:
    """Select top-N easy pairs by character-length delta (proxy for token delta)."""
    logger.info(
        "Ranking %s Easy pairs by character-length delta...", f"{len(easy_pairs):,}"
    )
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

    for p in easy_pairs:
        del p["_delta"]

    return selected


def select_hard_pairs(hard_pairs: list[dict], num_hard: int) -> list[dict]:
    """Select top-N hard pairs by longest chosen response (deepest reasoning)."""
    logger.info(
        "Ranking %s Hard pairs by chosen length...", f"{len(hard_pairs):,}"
    )

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


def print_distribution_stats(label: str, pairs: list[dict]) -> None:
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
    print(
        f"    Pairs/problem: min={min(counts)}, max={max(counts)}, "
        f"avg={avg:.1f}, median={median}"
    )
    if len(counts) >= 10:
        print(f"    Top 10 problem counts: {counts[:10]}")


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
    """Tokenize pairs and save as .pt file."""
    chosen_ids_acc, chosen_masks_acc = [], []
    rejected_ids_acc, rejected_masks_acc = [], []
    complexities_all: list[int] = []

    num_batches = (len(pairs) + batch_size - 1) // batch_size

    for batch_idx in tqdm(
        range(num_batches), desc=f"Tokenizing {filename}", unit=" batches"
    ):
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
            chosen_combined,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        rejected_tok = tokenizer(
            rejected_combined,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
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


def compute_token_stats(pairs: list[dict]) -> dict[str, float]:
    """Compute average token counts for a set of pairs."""
    chosen_tokens = []
    rejected_tokens = []
    for p in tqdm(pairs, desc="Computing token stats", unit=" pairs"):
        ct = count_tokens(p["chosen"])
        rt = count_tokens(p["rejected"])
        chosen_tokens.append(ct)
        rejected_tokens.append(rt)

    avg_chosen = sum(chosen_tokens) / len(chosen_tokens) if chosen_tokens else 0
    avg_rejected = (
        sum(rejected_tokens) / len(rejected_tokens) if rejected_tokens else 0
    )
    avg_delta = (
        sum(rt - ct for ct, rt in zip(chosen_tokens, rejected_tokens))
        / len(chosen_tokens)
        if chosen_tokens
        else 0
    )

    return {
        "avg_chosen_tokens": round(avg_chosen, 1),
        "avg_rejected_tokens": round(avg_rejected, 1),
        "avg_delta_tokens": round(avg_delta, 1),
    }


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
    logger.info("[1/8] Loading all pairs from %s...", source_path)
    all_pairs = load_all_pairs(source_path)
    source_total = len(all_pairs)
    logger.info("  Loaded %s pairs", f"{source_total:,}")

    # Step 2: Separate by complexity
    logger.info("[2/8] Separating by complexity...")
    easy_pairs = [p for p in all_pairs if p.get("complexity", 0) == 0]
    hard_pairs = [p for p in all_pairs if p.get("complexity", 0) == 1]
    logger.info(
        "  Easy: %s, Hard: %s", f"{len(easy_pairs):,}", f"{len(hard_pairs):,}"
    )
    del all_pairs

    print(f"\n{'=' * 60}")
    print("BEFORE capping:")
    print(f"{'=' * 60}")
    print_distribution_stats("Easy (complexity=0)", easy_pairs)
    print_distribution_stats("Hard (complexity=1)", hard_pairs)

    # Step 3: Cap per problem (asymmetric caps)
    logger.info(
        "[3/8] Capping Easy at %d, Hard at %d pairs per problem...",
        args.easy_cap,
        args.hard_cap,
    )
    easy_capped = cap_pairs_per_problem(easy_pairs, args.easy_cap, args.seed)
    hard_capped = cap_pairs_per_problem(hard_pairs, args.hard_cap, args.seed)
    del easy_pairs, hard_pairs

    print(f"\n{'=' * 60}")
    print(
        f"AFTER capping (Easy cap={args.easy_cap}, Hard cap={args.hard_cap}):"
    )
    print(f"{'=' * 60}")
    print_distribution_stats("Easy (complexity=0)", easy_capped)
    print_distribution_stats("Hard (complexity=1)", hard_capped)

    # Step 4: Select best pairs using quality criteria (same as subsample_balanced_pairs.py)
    logger.info(
        "[4/8] Selecting top %d Easy pairs by max delta...", args.num_easy
    )
    selected_easy = select_easy_pairs(easy_capped, args.num_easy)
    del easy_capped

    logger.info(
        "[5/8] Selecting top %d Hard pairs by longest chosen...", args.num_hard
    )
    selected_hard = select_hard_pairs(hard_capped, args.num_hard)
    del hard_capped

    actual_easy = len(selected_easy)
    actual_hard = len(selected_hard)

    # Balance: if one side has fewer than target, reduce the other to match
    if actual_easy != actual_hard:
        min_count = min(actual_easy, actual_hard)
        if actual_easy > min_count:
            logger.info(
                "Balancing: reducing Easy from %d to %d to match Hard",
                actual_easy,
                min_count,
            )
            selected_easy = selected_easy[:min_count]
            actual_easy = min_count
        if actual_hard > min_count:
            logger.info(
                "Balancing: reducing Hard from %d to %d to match Easy",
                actual_hard,
                min_count,
            )
            selected_hard = selected_hard[:min_count]
            actual_hard = min_count

    print(f"\n{'=' * 60}")
    print("AFTER selection + balancing:")
    print(f"{'=' * 60}")
    print_distribution_stats("Easy (complexity=0)", selected_easy)
    print_distribution_stats("Hard (complexity=1)", selected_hard)

    # Step 5: Combine
    logger.info(
        "[6/8] Combining %d Easy + %d Hard = %d total",
        actual_easy,
        actual_hard,
        actual_easy + actual_hard,
    )
    selected_pairs = selected_easy + selected_hard

    # Step 6: Split 90/10 by problem
    logger.info("[6/8] Splitting 90/10 by problem (stratified)...")
    train_pairs, val_pairs = split_pairs_by_problem(
        selected_pairs, args.val_split, args.seed
    )
    logger.info(
        "  Train: %s, Val: %s",
        f"{len(train_pairs):,}",
        f"{len(val_pairs):,}",
    )

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
    _write_jsonl(
        output_dir / "dataset.jsonl", selected_pairs, desc="Saving dataset.jsonl"
    )
    _write_jsonl(
        output_dir / "train.jsonl", train_pairs, desc="Saving train.jsonl"
    )
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

        tokenize_and_save(
            train_pairs, output_dir, tokenizer, "train_tokens.pt", args.max_length
        )
        tokenize_and_save(
            val_pairs, output_dir, tokenizer, "val_tokens.pt", args.max_length
        )
        logger.info("  Saved tokenized data to %s", output_dir)
    else:
        logger.info("[8/8] Skipping tokenization (--skip-tokenization)")

    # Write metadata.json
    easy_problems = set(p["problem"] for p in selected_easy)
    hard_problems = set(p["problem"] for p in selected_hard)

    metadata = {
        "total_pairs": actual_easy + actual_hard,
        "easy_pairs": actual_easy,
        "hard_pairs": actual_hard,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "val_split": args.val_split,
        "seed": args.seed,
        "easy_cap_per_problem": args.easy_cap,
        "hard_cap_per_problem": args.hard_cap,
        "unique_problems_easy": len(easy_problems),
        "unique_problems_hard": len(hard_problems),
        "unique_problems_total": len(easy_problems | hard_problems),
        "avg_chosen_tokens_easy": easy_stats["avg_chosen_tokens"],
        "avg_rejected_tokens_easy": easy_stats["avg_rejected_tokens"],
        "avg_length_delta_easy": easy_stats["avg_delta_tokens"],
        "avg_chosen_tokens_hard": hard_stats["avg_chosen_tokens"],
        "avg_rejected_tokens_hard": hard_stats["avg_rejected_tokens"],
        "selection_method_easy": "max_char_delta (after cap)",
        "selection_method_hard": "longest_chosen_response (after cap)",
        "source_dataset": str(args.source_dir),
        "source_total_pairs": source_total,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Source:        {args.source_dir} ({source_total:,} pairs)")
    print(f"  Output:        {output_dir} ({actual_easy + actual_hard:,} pairs)")
    print(f"  Easy/Hard:     {actual_easy:,} / {actual_hard:,}")
    print(f"  Easy cap:      {args.easy_cap} pairs/problem")
    print(f"  Hard cap:      {args.hard_cap} pairs/problem")
    print(f"  Train/Val:     {len(train_pairs):,} / {len(val_pairs):,}")
    print(f"  Easy problems: {len(easy_problems):,}")
    print(f"  Hard problems: {len(hard_problems):,}")
    print(f"  Total problems:{len(easy_problems | hard_problems):,}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
