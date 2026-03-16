#!/usr/bin/env python3
"""
Run evaluation on baseline and budget-aware checkpoints.
Usage:
  Dummy: USE_DUMMY_DATA=1 python scripts/run_evaluation.py --dummy
  Real:  USE_DUMMY_DATA=0 python scripts/run_evaluation.py
Output: checkpoints/evaluation_results_dummy.json or evaluation_results_real.json
"""

import argparse
import json
from pathlib import Path

from src.config import (
    CHECKPOINT_DIR,
    USE_DUMMY_DATA,
    get_baseline_output_dir,
    get_budget_aware_output_dir,
)
from src.evaluation.run_evaluation import (
    evaluate_checkpoint,
    load_eval_problems,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

set_seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (for quick test)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dummy", action="store_true", help="Force dummy mode (overrides USE_DUMMY_DATA)")
    args = parser.parse_args()

    use_real = not args.dummy and not USE_DUMMY_DATA
    problems = load_eval_problems(limit=args.limit, use_real=use_real)
    if not problems:
        raise RuntimeError("No evaluation problems. Run preprocess_dpo_data.py (dummy) or load_real_data.py (real).")

    output_dir = Path(args.output or str(CHECKPOINT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dummy" if args.dummy or USE_DUMMY_DATA else "_real"
    results_path = output_dir / f"evaluation_results{suffix}.json"

    # Checkpoint paths: --dummy overrides env, else use config routing
    if args.dummy:
        baseline_path = CHECKPOINT_DIR / "baseline_dpo"
        budget_path = CHECKPOINT_DIR / "budget_aware_dpo"
    else:
        baseline_path = get_baseline_output_dir()
        budget_path = get_budget_aware_output_dir()
    all_metrics = {}

    # Baseline
    if baseline_path.exists():
        logger.info("Evaluating %s...", baseline_path.name)
        m = evaluate_checkpoint(
            baseline_path,
            problems,
            output_path=output_dir / f"baseline_eval{suffix}.json",
        )
        all_metrics["baseline_dpo"] = m
        line = f"  Accuracy: {m['accuracy']:.2%} | TPCA: {m['tpca']:.1f} | Avg Easy: {m['avg_tokens_easy']:.1f} | Avg Hard: {m['avg_tokens_hard']:.1f}"
        if "math_level_4_5_accuracy" in m:
            line += f" | MATH L4-5: {m['math_level_4_5_accuracy']:.2%}"
        logger.info(line)

    # Budget-aware
    if budget_path.exists():
        logger.info("Evaluating %s...", budget_path.name)
        m = evaluate_checkpoint(
            budget_path,
            problems,
            output_path=output_dir / f"budget_aware_eval{suffix}.json",
        )
        all_metrics["budget_aware_dpo"] = m
        line = f"  Accuracy: {m['accuracy']:.2%} | TPCA: {m['tpca']:.1f} | Avg Easy: {m['avg_tokens_easy']:.1f} | Avg Hard: {m['avg_tokens_hard']:.1f}"
        if "math_level_4_5_accuracy" in m:
            line += f" | MATH L4-5: {m['math_level_4_5_accuracy']:.2%}"
        logger.info(line)

    # Sanity (optional)
    sanity_path = CHECKPOINT_DIR / "sanity_overfit"
    if sanity_path.exists() and not all_metrics:
        logger.info("Evaluating sanity_overfit (no baseline/budget checkpoints yet)...")
        m = evaluate_checkpoint(
            sanity_path,
            problems,
            output_path=output_dir / f"sanity_eval{suffix}.json",
        )
        all_metrics["sanity_overfit"] = m
        logger.info("  Accuracy: %.2f | TPCA: %.1f", m['accuracy'], m['tpca'])

    with open(results_path, "w") as f:
        json.dump({"metrics": all_metrics, "num_problems": len(problems)}, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
