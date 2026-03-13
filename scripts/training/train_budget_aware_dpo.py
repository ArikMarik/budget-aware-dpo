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
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: budget_aware_dpo or budget_aware_dpo_real)")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50, help="Log metrics every N steps (default: 50)")
    parser.add_argument("--data-limit", type=int, default=None, help="Limit samples (for dummy)")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0, help="Early stopping threshold (default: 0.0)")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta parameter (default: 0.1)")
    parser.add_argument("--lambda-easy", type=float, default=0.05, help="Lambda for easy samples (default: 0.05)")
    parser.add_argument("--lambda-hard", type=float, default=0.001, help="Lambda for hard samples (default: 0.001)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or str(get_budget_aware_output_dir()))
    train_dpo(
        use_budget_aware=True,
        output_dir=output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
        log_every=args.log_every,
        data_limit=args.data_limit,
        resume_from=args.resume_from,
        seed=args.seed,
        use_wandb=args.wandb,
        val_split=args.val_split,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        dpo_beta=args.dpo_beta,
        lambda_easy=args.lambda_easy,
        lambda_hard=args.lambda_hard,
    )


if __name__ == "__main__":
    main()
