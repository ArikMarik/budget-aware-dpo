#!/usr/bin/env python3
"""
Run evaluation on baseline and budget-aware checkpoints.
Usage: USE_DUMMY_DATA=1 python scripts/run_evaluation.py
Output: checkpoints/evaluation_results_dummy.json
"""

import argparse
import json
from pathlib import Path

from src.config import CHECKPOINT_DIR
from src.evaluation.run_evaluation import (
    evaluate_checkpoint,
    load_eval_problems,
)
from src.utils import set_seed

set_seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (for quick test)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dummy", action="store_true", help="Use dummy checkpoints")
    args = parser.parse_args()

    problems = load_eval_problems(limit=args.limit)
    if not problems:
        raise RuntimeError("No evaluation problems. Run preprocess_dpo_data.py first.")

    output_dir = Path(args.output or str(CHECKPOINT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dummy" if args.dummy else ""
    results_path = output_dir / f"evaluation_results{suffix}.json"

    all_metrics = {}

    # Baseline
    baseline_path = CHECKPOINT_DIR / "baseline_dpo"
    if baseline_path.exists():
        print("Evaluating baseline_dpo...")
        m = evaluate_checkpoint(
            baseline_path,
            problems,
            output_path=output_dir / f"baseline_eval{suffix}.json",
        )
        all_metrics["baseline_dpo"] = m
        print(f"  Accuracy: {m['accuracy']:.2%} | TPCA: {m['tpca']:.1f} | Avg Easy: {m['avg_tokens_easy']:.1f} | Avg Hard: {m['avg_tokens_hard']:.1f}")

    # Budget-aware
    budget_path = CHECKPOINT_DIR / "budget_aware_dpo"
    if budget_path.exists():
        print("Evaluating budget_aware_dpo...")
        m = evaluate_checkpoint(
            budget_path,
            problems,
            output_path=output_dir / f"budget_aware_eval{suffix}.json",
        )
        all_metrics["budget_aware_dpo"] = m
        print(f"  Accuracy: {m['accuracy']:.2%} | TPCA: {m['tpca']:.1f} | Avg Easy: {m['avg_tokens_easy']:.1f} | Avg Hard: {m['avg_tokens_hard']:.1f}")

    # Sanity (optional)
    sanity_path = CHECKPOINT_DIR / "sanity_overfit"
    if sanity_path.exists() and not all_metrics:
        print("Evaluating sanity_overfit (no baseline/budget checkpoints yet)...")
        m = evaluate_checkpoint(
            sanity_path,
            problems,
            output_path=output_dir / f"sanity_eval{suffix}.json",
        )
        all_metrics["sanity_overfit"] = m
        print(f"  Accuracy: {m['accuracy']:.2%} | TPCA: {m['tpca']:.1f}")

    with open(results_path, "w") as f:
        json.dump({"metrics": all_metrics, "num_problems": len(problems)}, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
