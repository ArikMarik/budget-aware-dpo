#!/usr/bin/env python3
"""
Train budget-aware DPO (custom R_budget loss with length penalty).
"""

import argparse
from pathlib import Path

from src.config import get_budget_aware_output_dir
from src.training.dpo_trainer import train_dpo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--data-limit", type=int, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--lambda-easy", type=float, default=0.05)
    parser.add_argument("--lambda-hard", type=float, default=0.001)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name (auto-generated if omitted)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or str(get_budget_aware_output_dir()))
    train_dpo(
        use_budget_aware=True,
        output_dir=output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        data_limit=args.data_limit,
        resume_from=args.resume_from,
        seed=args.seed,
        use_wandb=args.wandb,
        run_name=args.run_name,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        dpo_beta=args.dpo_beta,
        lambda_easy=args.lambda_easy,
        lambda_hard=args.lambda_hard,
        use_mixed_precision=not args.no_mixed_precision,
        compile_model=args.compile_model,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
