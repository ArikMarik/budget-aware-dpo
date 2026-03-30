#!/usr/bin/env python3
"""
Evaluate a single checkpoint with full Tier 0+1+2 verification (including LLM judge).
Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoint.py \
    --checkpoint checkpoints/budget_aware_balanced_iter5 \
    --limit 500 --output eval_results/budget_iter5_tier012.json
"""
import argparse
import json
import random
from pathlib import Path

from src.evaluation.run_evaluation import (
    evaluate_checkpoint,
    load_eval_problems,
)
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)
set_seed(42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--limit", type=int, default=500, help="Number of problems")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--use-real", action="store_true", help="Use GSM8K+MATH test sets (not training data)")
    parser.add_argument("--base-model", type=str, default=None, help="Base model name (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--few-shot", type=int, default=0, choices=[0, 8],
                        help="Number of few-shot exemplars (0=zero-shot, 8=standard GSM8K 8-shot)")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    problems = load_eval_problems(limit=None, use_real=args.use_real)
    logger.info("Loaded %d problems total", len(problems))

    # Sample balanced: try to get equal easy/hard
    easy = [p for p in problems if p["complexity"] == 0]
    hard = [p for p in problems if p["complexity"] == 1]
    random.shuffle(easy)
    random.shuffle(hard)

    half = args.limit // 2
    sampled = easy[:half] + hard[:half]
    if len(sampled) < args.limit:
        # If one category is short, fill from the other
        remaining = args.limit - len(sampled)
        if len(easy) > half:
            sampled += easy[half:half + remaining]
        elif len(hard) > half:
            sampled += hard[half:half + remaining]
    random.shuffle(sampled)
    logger.info("Sampled %d problems (%d easy, %d hard)",
                len(sampled),
                sum(1 for p in sampled if p["complexity"] == 0),
                sum(1 for p in sampled if p["complexity"] == 1))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_fn = None
    if args.few_shot == 8:
        from src.evaluation.few_shot_exemplars import build_8shot_prompt
        prompt_fn = build_8shot_prompt
        logger.info("Using 8-shot chain-of-thought prompting")

    logger.info("Evaluating %s with %d problems (Tier 0+1+2, LLM judge ON)...", checkpoint_path, len(sampled))
    metrics = evaluate_checkpoint(
        checkpoint_path,
        sampled,
        output_path=output_path,
        base_model=args.base_model,
        prompt_fn=prompt_fn,
    )

    logger.info("=== RESULTS ===")
    logger.info("Accuracy: %.2f%% (easy=%.2f%%, hard=%.2f%%)",
                metrics["accuracy"] * 100,
                metrics.get("easy_accuracy", 0) * 100,
                metrics.get("hard_accuracy", 0) * 100)
    logger.info("Avg tokens: easy=%.1f, hard=%.1f",
                metrics.get("avg_tokens_easy", 0),
                metrics.get("avg_tokens_hard", 0))
    logger.info("TPCA: %.1f", metrics.get("tpca", float("inf")))
    if "math_level_4_5_accuracy" in metrics:
        logger.info("MATH L4-5 accuracy: %.2f%%", metrics["math_level_4_5_accuracy"] * 100)
    logger.info("Full results saved to %s", output_path)


if __name__ == "__main__":
    main()
