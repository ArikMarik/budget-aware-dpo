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
    parser.add_argument("--val-split", type=float, default=0.2, help='Validation fraction size, must be in (0, 1) (default 0.2)')
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
    parser.add_argument("--kl-penalty", type=float, default=0.0, help="KL divergence penalty weight to prevent model collapse (0.0 = disabled)")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default=None, help="WandB run name (auto-generated if omitted)")
    parser.add_argument("--model", type=str, default=None, help="Model name/path (default: Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--loss-type", type=str, default="dpo", choices=["dpo", "simpo"], help="Loss function type")
    parser.add_argument("--length-ratio", type=float, default=1.5, help="Minimum length ratio between preferred and rejected solutions, default: 1.5 (1.0 = no filter)")
    parser.add_argument("--max-pairs-per-problem", type=int, default=10, help="Maximum number of DPO pairs per problem (stratified by rejection_reason), enter -1 for no limit")
    parser.add_argument(
        "--best-model-metric",
        type=str,
        default="val_loss",
        choices=[
            "val_loss",
            "gen_tokens_easy",
            "gen_tpca",
            "gen_tokens_easy_with_accuracy_floor",
        ],
        help="Metric used to select the best epoch/checkpoint.",
    )
    parser.add_argument(
        "--accuracy-floor",
        type=float,
        default=None,
        help="Only used with --best-model-metric=gen_tokens_easy_with_accuracy_floor. "
             "Requires gen/accuracy_easy >= this threshold to be eligible.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or str(get_budget_aware_output_dir()))
    train_dpo(
        use_budget_aware=True,
        output_dir=output_dir,
        val_split=args.val_split,
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
        kl_penalty_weight=args.kl_penalty,
        use_mixed_precision=not args.no_mixed_precision,
        compile_model=args.compile_model,
        num_workers=args.num_workers,
        model_name=args.model,
        loss_type=args.loss_type,
        length_ratio=args.length_ratio,
        max_pairs_per_problem=args.max_pairs_per_problem,
        best_model_metric=args.best_model_metric,
        accuracy_floor=args.accuracy_floor,
    )


if __name__ == "__main__":
    main()
