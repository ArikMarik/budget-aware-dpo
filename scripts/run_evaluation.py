#!/usr/bin/env python3
"""
General evaluation script for both saved checkpoints and the pretrained baseline model.

Usage:
  # Evaluate a saved checkpoint (default: budget-aware checkpoint)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/run_evaluation.py \
    --checkpoint-path checkpoints/budget_aware_dpo_full/checkpoint-epoch-3

  # Evaluate the pretrained base model
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/run_evaluation.py --use-base-model --few-shot

  # Custom output path
  .venv/bin/python scripts/run_evaluation.py --use-base-model --output eval_results/custom.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME, SEED
from src.evaluation.few_shot_exemplars import build_few_shots_prompt, build_zero_shot_prompt
from src.evaluation.run_evaluation import (
    compute_metrics,
    evaluate_checkpoint,
    generate_and_evaluate,
    load_eval_problems_real,
)
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)
set_seed(SEED)


def generate_default_output(checkpoint_path, use_base_model, max_new_tokens, few_shot):
    """Generate default output path based on flags."""
    output_dir = Path("eval_results")
    prompt_tag = "few_shot" if few_shot else "zero_shot"

    if use_base_model:
        name = "baseline"
    else:
        if checkpoint_path:
            name = Path(checkpoint_path).name
        else:
            name = "budget_aware_dpo_from_checkpoint-epoch-3"

    return str(output_dir / f"{name}_m{max_new_tokens}_{prompt_tag}.json")


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint or base model on real test data")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (auto-generated if not provided)")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to saved checkpoint")
    parser.add_argument("--use-base-model", action="store_true", help="Evaluate pretrained baseline (no training)")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--few-shot", action="store_true", help="Use few-shot prompting (default: zero-shot)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation")
    args = parser.parse_args()

    # Conflict check
    if args.use_base_model and args.checkpoint_path:
        parser.error("Cannot use both --use-base-model and --checkpoint-path")

    # Generate default output path
    if args.output is None:
        args.output = generate_default_output(
            args.checkpoint_path, args.use_base_model, args.max_new_tokens, args.few_shot
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load real test data
    logger.info("Loading real test data (GSM8K + MATH)...")
    problems = load_eval_problems_real()
    easy = [p for p in problems if p["complexity"] == 0]
    hard = [p for p in problems if p["complexity"] == 1]
    logger.info("Loaded %d problems (%d easy, %d hard)", len(problems), len(easy), len(hard))

    # Set prompt function
    if args.few_shot:
        prompt_fn = build_few_shots_prompt
        prompt_mode = "few-shot"
        logger.info("Using few-shot prompting (8-shot GSM8K, 4-shot MATH)")
    else:
        prompt_fn = build_zero_shot_prompt
        prompt_mode = "zero-shot"
        logger.info("Using zero-shot prompting")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()

    # Evaluation modes
    if args.use_base_model:
        logger.info("Loading base model: %s", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()
        mode = f"baseline+{prompt_mode}"

        logger.info("Mode: %s | max_new_tokens=%d | batch_size=%d | problems=%d",
                     mode, args.max_new_tokens, args.batch_size, len(problems))

        results = generate_and_evaluate(
            model, tokenizer, problems,
            max_new_tokens=args.max_new_tokens,
            prompt_fn=prompt_fn,
            batch_size=args.batch_size,
        )
        metrics = compute_metrics(results)
        metrics["model"] = MODEL_NAME
        metrics["mode"] = mode

    else:
        checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else \
            Path("checkpoints/budget_aware_dpo_full/checkpoint-epoch-3")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("Evaluating checkpoint: %s", checkpoint_path)
        mode = f"checkpoint+{prompt_mode}"

        out = evaluate_checkpoint(
            checkpoint_path,
            problems,
            output_path=None,
            prompt_fn=prompt_fn,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        metrics = out["metrics"]
        results = out["results"]
        metrics["model"] = str(checkpoint_path)
        metrics["mode"] = mode

    elapsed = time.time() - start
    logger.info("Evaluation completed in %.1f minutes", elapsed / 60)

    # Add metadata
    metrics["max_new_tokens"] = args.max_new_tokens
    metrics["batch_size"] = args.batch_size
    metrics["eval_time_minutes"] = elapsed / 60

    # Save results
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS — %s", mode)
    logger.info("=" * 60)
    logger.info("Overall accuracy:  %.2f%% (%d/%d)",
                metrics["accuracy"] * 100, metrics["num_correct"], metrics["num_total"])
    logger.info("Easy (GSM8K) acc:  %.2f%% (%d/%d)",
                metrics["accuracy_easy"] * 100, metrics["num_easy_correct"], metrics["num_easy"])
    logger.info("Hard (MATH) acc:   %.2f%% (%d/%d)",
                metrics["accuracy_hard"] * 100, metrics["num_hard_correct"], metrics["num_hard"])
    logger.info("Avg tokens easy:   %.1f", metrics["avg_tokens_easy"])
    logger.info("Avg tokens hard:   %.1f", metrics["avg_tokens_hard"])
    logger.info("TPCA:              %.1f", metrics["tpca"])
    logger.info("Efficiency score:  %.4f (acc / (avg_tokens / max_tokens))", metrics["efficiency"])
    if metrics["math_by_level"]:
        logger.info("MATH by level:")
        for level in sorted(metrics["math_by_level"]):
            v = metrics["math_by_level"][level]
            logger.info("  %s: %.2f%% (%d/%d)", level, v["accuracy"] * 100, v["correct"], v["total"])
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
