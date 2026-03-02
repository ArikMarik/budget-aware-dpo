#!/usr/bin/env python3
"""
Generate figures from evaluation results.
Usage: python scripts/run_visualization.py --dummy
Output: reports/figures_dummy/
"""

import argparse
from pathlib import Path

from src.config import CHECKPOINT_DIR
from src.visualization.plot_results import generate_figures, generate_results_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", help="Use dummy evaluation results")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    suffix = "_dummy" if args.dummy else ""
    eval_dir = CHECKPOINT_DIR
    output_dir = Path(args.output_dir or f"reports/figures{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figures
    paths = generate_figures(eval_dir, output_dir, suffix=suffix)
    print(f"Generated figures: {paths}")

    # Results table
    metrics_path = CHECKPOINT_DIR / f"evaluation_results{suffix}.json"
    if metrics_path.exists():
        table_path = output_dir / f"results_table{suffix}.md"
        generate_results_table(metrics_path, table_path)
        print(f"Results table: {table_path}")


if __name__ == "__main__":
    main()
