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
    parser.add_argument("--data-limit", type=int, default=None, help="Limit samples (for dummy)")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
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
        data_limit=args.data_limit,
        resume_from=args.resume_from,
        seed=args.seed,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
