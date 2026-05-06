#!/usr/bin/env python3
"""
Generate figures from evaluation results.
Usage:
  Dummy: python scripts/run_visualization.py --dummy
  Real:  python scripts/run_visualization.py
  Custom: python scripts/run_visualization.py --baseline-results path/to/baseline.json --budget-results path/to/budget.json --suffix _v1 --output-dir reports/v1
Output: reports/figures_dummy/ or reports/figures/ or custom output dir
"""

import argparse
import json
import tempfile
from pathlib import Path

from src.utils import get_logger
from src.visualization.plot_results import generate_figures, generate_results_table

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate figures from evaluation results")
    parser.add_argument("--baseline-results", type=Path, default=Path("eval_results/baseline_zeroShot_noLora.json"),
                        help="Path to baseline per-sample evaluation results JSON (default: %(default)s)")
    parser.add_argument("--budget-results", type=Path, default=Path("eval_results/budget_aware_dpo_zeroShot_epoch_3.json"),
                        help="Path to budget-aware per-sample evaluation results JSON (default: %(default)s)")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Suffix for output figure filenames (default: _dummy if --dummy, else _real)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for figures (default: reports/figures_dummy if --dummy, else reports/figures)")
    parser.add_argument("--dummy", action="store_true", help="Use dummy evaluation results (sets default suffix to _dummy and output dir to reports/figures_dummy)")
    args = parser.parse_args()

    # Determine suffix
    if args.suffix is not None:
        suffix = args.suffix
    else:
        suffix = "_dummy" if args.dummy else ""

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path("reports/figures_dummy" if args.dummy else "reports/figures")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figures
    try:
        paths = generate_figures(
            baseline_results_path=args.baseline_results,
            budget_results_path=args.budget_results,
            output_dir=output_dir,
            suffix=suffix
        )
        logger.info("Generated figures: %s", paths)
    except FileNotFoundError as e:
        logger.error("Failed to generate figures: %s", e)
        raise

    # Results table - combine metrics from both result files
    combined_metrics = {}
    result_files = [
        ("baseline", args.baseline_results),
        ("budget_aware", args.budget_results),
    ]

    for model_name, results_path in result_files:
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            if metrics:
                combined_metrics[model_name] = metrics
            else:
                logger.warning("No metrics found in %s", results_path)
        else:
            logger.warning("Results file not found: %s", results_path)

    if combined_metrics:
        combined_data = {"metrics": combined_metrics}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=output_dir) as tmp:
            json.dump(combined_data, tmp)
            tmp_path = Path(tmp.name)

        try:
            table_path = output_dir / f"results_table{suffix}.md"
            generate_results_table(tmp_path, table_path)
            logger.info("Results table: %s", table_path)
        finally:
            tmp_path.unlink()
    else:
        logger.warning("No metrics to generate results table")


if __name__ == "__main__":
    main()
