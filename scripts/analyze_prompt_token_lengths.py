#!/usr/bin/env python3
"""
Analyze prompt and solution token lengths from PROBLEM_TO_INDEX_PATH.
Samples problems and plots histograms of token counts per complexity (0=easy, 1=hard).

Usage:
    python scripts/analyze_prompt_token_lengths.py [--sample-size 30000] [--output reports/figures/token_lengths.png]
                   [--token-limit 1500] [--stats-csv reports/data/token_length_stats.csv]
                   [--over-limit-json reports/data/problems_over_token_limit.json]
"""

import argparse
import csv
import json
import pickle
import random
from pathlib import Path

import matplotlib

from src.data.worker_utils import count_tokens_batch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from src.config import PROBLEM_TO_INDEX_PATH, MODEL_NAME, OVER_LIMIT_PROBLEMS_PATH, SEED, TOKEN_LENGTH_STATS_PATH
from src.evaluation.few_shot_exemplars import build_zero_shot_prompt
from src.utils import get_logger, get_model_tokenizer

logger = get_logger(__name__)

PROMPT_TEMPLATE = "Question: {problem}\nAnswer: "


def load_problems(sample_size: int = 30_000) -> tuple[list[dict], int]:
    """Load problem_to_index.pkl and sample problems. Returns (sampled_problems, total_count)."""
    logger.info("Loading %s", PROBLEM_TO_INDEX_PATH)
    with open(PROBLEM_TO_INDEX_PATH, "rb") as f:
        problem_index = pickle.load(f)

    total = len(problem_index)
    logger.info("Total problems: %s", f"{total:,}")

    items = list(problem_index.values())
    sample_size = min(sample_size, total)
    random.seed(SEED)
    if sample_size < total:
        sampled = random.sample(items, sample_size)
    else:
        sampled = items
    logger.info("Sampled %s problems", f"{sample_size:,}")
    return sampled, total


def compute_prompt_token_lengths(problems: list[dict], tokenizer, token_limit: int = 1500) -> tuple[list[tuple[int, int]], list[dict]]:
    """Build prompts, tokenize, return (valid_data, over_limit_problems)."""
    prompts = [build_zero_shot_prompt(p["problem"]) for p in problems]

    logger.info("Tokenizing %s prompts...", f"{len(prompts):,}")
    token_counts = count_tokens_batch(prompts, tokenizer)

    valid_data = []
    over_limit = []
    for i, p in enumerate(problems):
        token_count = token_counts[i]
        if token_count > token_limit:
            over_limit.append({
                "problem_id": p["problem_id"],
                "complexity": p["complexity"],
                "token_count": token_count,
                "problem": p["problem"],
                "problem_source": p["problem_source"]
            })
        else:
            valid_data.append((p["complexity"], token_count))
    return valid_data, over_limit


def plot_histograms(
    prompt_data: list[tuple[int, int]],
    solution_data: list[tuple[int, int]],
    output_path: Path,
    sample_size: int = 30000,
    total_problems: int = 0,
) -> None:
    """Plot overlapping histograms of prompt and solution token lengths per complexity."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Add sample size annotation as suptitle
    sample_note = f"Sampled {sample_size:,} of {total_problems:,} total problems"
    fig.suptitle(sample_note, fontsize=11, fontweight="bold")

    # Separate data by complexity
    prompt_easy = [t for c, t in prompt_data if c == 0]
    prompt_hard = [t for c, t in prompt_data if c == 1]
    sol_easy = [t for c, t in solution_data if c == 0]
    sol_hard = [t for c, t in solution_data if c == 1]

    # Panel 1: Prompt token histogram (overlapping)
    ax = axes[0, 0]
    max_val = max(max(prompt_easy, default=0), max(prompt_hard, default=0))
    bins = np.linspace(0, max_val, 50)
    ax.hist(prompt_easy, bins=bins, alpha=0.6, label="Easy (0)", color="steelblue", edgecolor="black")
    ax.hist(prompt_hard, bins=bins, alpha=0.6, label="Hard (1)", color="darkorange", edgecolor="black")
    ax.set_xlabel("Prompt Token Length")
    ax.set_ylabel("Count")
    ax.set_title("Prompt Token Length Distribution by Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Prompt token density (KDE)
    ax = axes[0, 1]
    if prompt_easy:
        sns.kdeplot(prompt_easy, ax=ax, label="Easy (0)", color="steelblue", fill=True, alpha=0.3)
    if prompt_hard:
        sns.kdeplot(prompt_hard, ax=ax, label="Hard (1)", color="darkorange", fill=True, alpha=0.3)
    ax.set_xlabel("Prompt Token Length")
    ax.set_ylabel("Density")
    ax.set_title("Prompt Token Length Density by Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Solution token histogram (overlapping)
    ax = axes[1, 0]
    max_val = max(max(sol_easy, default=0), max(sol_hard, default=0))
    bins = np.linspace(0, max_val, 50)
    ax.hist(sol_easy, bins=bins, alpha=0.6, label="Easy (0)", color="steelblue", edgecolor="black")
    ax.hist(sol_hard, bins=bins, alpha=0.6, label="Hard (1)", color="darkorange", edgecolor="black")
    ax.set_xlabel("Solution Token Length (avg)")
    ax.set_ylabel("Count")
    ax.set_title("Solution Token Length Distribution by Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Solution token density (KDE)
    ax = axes[1, 1]
    if sol_easy:
        sns.kdeplot(sol_easy, ax=ax, label="Easy (0)", color="steelblue", fill=True, alpha=0.3)
    if sol_hard:
        sns.kdeplot(sol_hard, ax=ax, label="Hard (1)", color="darkorange", fill=True, alpha=0.3)
    ax.set_xlabel("Solution Token Length (avg)")
    ax.set_ylabel("Density")
    ax.set_title("Solution Token Length Density by Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", output_path)


def print_stats(prompt_data: list[tuple[int, int]], solution_data: list[tuple[int, int]]) -> None:
    """Print summary statistics per complexity."""
    datasets = [("Prompt", prompt_data), ("Solution (avg)", solution_data)]

    print("\n" + "=" * 70)
    print("TOKEN LENGTH STATISTICS (PROMPT & SOLUTION)")
    print("=" * 70)

    for name, data in datasets:
        easy = [t for c, t in data if c == 0]
        hard = [t for c, t in data if c == 1]

        print(f"\n--- {name} ---")
        for label, tokens in [("Easy (0)", easy), ("Hard (1)", hard)]:
            if not tokens:
                continue
            arr = np.array(tokens)
            print(f"\n  {label} (n={len(tokens):,})")
            print(f"    Mean:   {arr.mean():.1f}")
            print(f"    Median: {np.median(arr):.1f}")
            print(f"    Std:    {arr.std():.1f}")
            print(f"    Min:    {arr.min()}")
            print(f"    Max:    {arr.max()}")
            print(f"    P25:    {np.percentile(arr, 25):.1f}")
            print(f"    P75:    {np.percentile(arr, 75):.1f}")
            print(f"    P90:    {np.percentile(arr, 90):.1f}")

    print("\n" + "=" * 70)


def save_stats_csv(prompt_data: list[tuple[int, int]], solution_data: list[tuple[int, int]], output_path: Path) -> None:
    """Save summary statistics to CSV file."""
    datasets = [("prompt", prompt_data), ("solution", solution_data)]
    complexities = [("easy", 0), ("hard", 1)]

    rows = []
    metrics = ["mean", "median", "std", "min", "max", "p25", "p75", "p90"]

    for metric in metrics:
        row = {"metric": metric}
        for ds_name, data in datasets:
            for comp_name, comp_val in complexities:
                tokens = [t for c, t in data if c == comp_val]
                if not tokens:
                    row[f"{comp_name}_{ds_name}"] = ""
                    continue
                arr = np.array(tokens)
                if metric == "mean":
                    row[f"{comp_name}_{ds_name}"] = f"{arr.mean():.1f}"
                elif metric == "median":
                    row[f"{comp_name}_{ds_name}"] = f"{np.median(arr):.1f}"
                elif metric == "std":
                    row[f"{comp_name}_{ds_name}"] = f"{arr.std():.1f}"
                elif metric == "min":
                    row[f"{comp_name}_{ds_name}"] = str(arr.min())
                elif metric == "max":
                    row[f"{comp_name}_{ds_name}"] = str(arr.max())
                elif metric == "p25":
                    row[f"{comp_name}_{ds_name}"] = f"{np.percentile(arr, 25):.1f}"
                elif metric == "p75":
                    row[f"{comp_name}_{ds_name}"] = f"{np.percentile(arr, 75):.1f}"
                elif metric == "p90":
                    row[f"{comp_name}_{ds_name}"] = f"{np.percentile(arr, 90):.1f}"
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "easy_prompt", "hard_prompt", "easy_solution", "hard_solution"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved summary statistics to %s", output_path)


def save_over_limit_json(over_limit: list[dict], output_path: Path) -> None:
    """Save problems exceeding token limit to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(over_limit, f, indent=2)
    logger.info("Saved %s over-limit problems to %s", len(over_limit), output_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt and solution token lengths by complexity")
    parser.add_argument("--sample-size", type=int, default=30000, help="Number of problems to sample (default: 30000)")
    parser.add_argument("--output", type=str, default="reports/figures/token_lengths.png", help="Output path for histogram")
    parser.add_argument("--token-limit", type=int, default=1500, help="Token count limit for flagging problems (default: 1500)")
    parser.add_argument("--stats-csv", type=str, default=str(TOKEN_LENGTH_STATS_PATH), help="Output path for summary statistics CSV")
    parser.add_argument("--over-limit-json", type=str, default=str(OVER_LIMIT_PROBLEMS_PATH), help="Output path for problems exceeding token limit")
    args = parser.parse_args()

    problems, total_problems = load_problems(args.sample_size)
    sample_size = len(problems)

    # Load tokenizer for prompt tokenization
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = get_model_tokenizer()

    # Compute prompt token lengths
    prompt_data, over_limit = compute_prompt_token_lengths(problems, tokenizer, args.token_limit)

    # Extract solution token lengths (avg_token_length from pickle)
    solution_data = [(p["complexity"], p["avg_token_length"]) for p in problems]

    # Print stats
    print_stats(prompt_data, solution_data)

    # Save summary statistics to CSV
    save_stats_csv(prompt_data, solution_data, Path(args.stats_csv))

    # Save over-limit problems to JSON
    if over_limit:
        save_over_limit_json(over_limit, Path(args.over_limit_json))
        print(f"\n⚠️  Found {len(over_limit)} problems exceeding token limit ({args.token_limit})")

    # Plot with sample size annotation
    plot_histograms(prompt_data, solution_data, Path(args.output),
                    sample_size=sample_size, total_problems=total_problems)


if __name__ == "__main__":
    main()
