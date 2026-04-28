#!/usr/bin/env python3
"""
Evaluate the base Qwen2.5-Math-1.5B model (no LoRA, no training) on the full eval pipeline.
Supports two modes:
  1. Raw base model (default)
  2. Base model + untrained LoRA adapter (--with-lora-init)

Usage:
  # Raw base model, full dataset
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/eval_base_model.py \
    --output eval_results/base_qwen_1.5b_math_full.json --max-new-tokens 1024

  # With untrained LoRA (diagnostic)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/storage/arik/nlp_final_project PYTHONUNBUFFERED=1 \
    .venv/bin/python scripts/eval_base_model.py \
    --output eval_results/base_qwen_1.5b_math_lora_init.json --with-lora-init --max-new-tokens 1024
"""
import argparse
from functools import partial
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_NAME
from src.evaluation.few_shot_exemplars import build_few_shots_prompt, build_zero_shot_prompt
from src.evaluation.run_evaluation import (
    compute_metrics,
    generate_and_evaluate,
    load_eval_problems_real,
)
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)
set_seed(42)


def main():
    parser = argparse.ArgumentParser(description="Evaluate base model (no training)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--with-lora-init", action="store_true",
                        help="Apply untrained LoRA adapter (diagnostic: does LoRA init degrade base?)")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name")
    parser.add_argument("--math-levels", type=str, default=None,
                        help="Comma-separated MATH levels to include (e.g., '1,2' or '3,4,5'). GSM8K always included unless --math-only.")
    parser.add_argument("--math-only", action="store_true",
                        help="Exclude GSM8K, only evaluate MATH problems")
    parser.add_argument("--math-limit", type=int, default=None,
                        help="Limit number of MATH problems per level (sample randomly)")
    parser.add_argument("--zero-shot", action="store_true", help="Use zero-shot prompting instead of few-shot exemplars")
    args = parser.parse_args()

    model_name = args.model_name or MODEL_NAME
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load problems
    problems = load_eval_problems_real()

    # Filter by level/source
    import random
    random.seed(42)

    if args.math_only:
        problems = [p for p in problems if p.get("source") == "math"]

    if args.math_levels:
        allowed_levels = set(f"Level {l.strip()}" for l in args.math_levels.split(","))
        gsm8k = [p for p in problems if p.get("source") == "gsm8k"]
        math_filtered = [p for p in problems if p.get("source") == "math" and str(p.get("level", "")).strip() in allowed_levels]
        problems = ([] if args.math_only else gsm8k) + math_filtered

    if args.math_limit and not args.math_only:
        # Sample MATH problems per level, keep all GSM8K
        gsm8k = [p for p in problems if p.get("source") == "gsm8k"]
        math_problems = [p for p in problems if p.get("source") == "math"]
        # Group by level and sample
        by_level = {}
        for p in math_problems:
            lvl = str(p.get("level", "unknown"))
            by_level.setdefault(lvl, []).append(p)
        sampled_math = []
        for lvl, ps in by_level.items():
            random.shuffle(ps)
            sampled_math.extend(ps[:args.math_limit])
        problems = gsm8k + sampled_math
    elif args.math_limit and args.math_only:
        by_level = {}
        for p in problems:
            lvl = str(p.get("level", "unknown"))
            by_level.setdefault(lvl, []).append(p)
        sampled = []
        for lvl, ps in by_level.items():
            random.shuffle(ps)
            sampled.extend(ps[:args.math_limit])
        problems = sampled

    if args.limit:
        random.shuffle(problems)
        problems = problems[:args.limit]

    easy = [p for p in problems if p["complexity"] == 0]
    hard = [p for p in problems if p["complexity"] == 1]
    logger.info("Loaded %d problems (%d easy, %d hard)", len(problems), len(easy), len(hard))

    # Load base model
    logger.info("Loading base model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if args.with_lora_init:
        from peft import LoraConfig, get_peft_model
        # Same LoRA config as training (r=128, alpha=256)
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("LoRA applied (untrained). Trainable: %d / %d (%.2f%%)",
                     trainable, total, 100.0 * trainable / total)

    model.eval()
    mode = "base+untrained_lora" if args.with_lora_init else "raw_base"

    # Setup prompt function
    if not args.zero_shot:
        prompt_fn = build_few_shots_prompt
        mode += "+few-shot"
        logger.info("Using 8-shot chain-of-thought prompting (GSM8K exemplars) and 4-shot MATH exemplars")
    else:
        prompt_fn = build_zero_shot_prompt
        mode += "+zero-shot"
        logger.info("Using 0-shot prompting (no exemplars)")

    logger.info("Mode: %s | max_new_tokens=%d | problems=%d", mode, args.max_new_tokens, len(problems))

    # Run evaluation
    start = time.time()
    results = generate_and_evaluate(
        model, tokenizer, problems,
        max_new_tokens=args.max_new_tokens,
        prompt_fn=prompt_fn,
        batch_size=64,
    )
    elapsed = time.time() - start
    logger.info("Evaluation completed in %.1f minutes", elapsed / 60)

    # Compute metrics
    metrics = compute_metrics(results)

    # Add metadata
    metrics["model"] = model_name
    metrics["mode"] = mode
    metrics["max_new_tokens"] = args.max_new_tokens
    metrics["eval_time_minutes"] = elapsed / 60

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("BASE MODEL EVALUATION — %s", mode)
    logger.info("=" * 60)
    logger.info("Overall accuracy:  %.2f%% (%d/%d)", metrics["accuracy"] * 100, metrics["num_correct"], metrics["num_total"])
    logger.info("Easy (GSM8K) acc:  %.2f%% (%d/%d)", metrics["easy_accuracy"] * 100, metrics["num_easy_correct"], metrics["num_easy"])
    logger.info("Hard (MATH) acc:   %.2f%% (%d/%d)", metrics["hard_accuracy"] * 100, metrics["num_hard_correct"], metrics["num_hard"])
    logger.info("Avg tokens easy:   %.1f", metrics["avg_tokens_easy"])
    logger.info("Avg tokens hard:   %.1f", metrics["avg_tokens_hard"])
    logger.info("TPCA:              %.1f", metrics["tpca"])
    if metrics["math_by_level"]:
        logger.info("MATH by level:")
        for level in sorted(metrics["math_by_level"]):
            v = metrics["math_by_level"][level]
            logger.info("  %s: %.2f%% (%d/%d)", level, v["accuracy"] * 100, v["correct"], v["total"])
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
