#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for Budget-Aware DPO training.

Search space covers training HPs (lr, dpo_beta, lambda_easy, lambda_hard,
kl_penalty_weight, batch_size, gradient_accumulation_steps, loss_type) and
data HPs (length_ratio, max_pairs_per_problem).

Each trial calls `train_dpo()` directly and returns the chosen objective.
Studies persist in SQLite so sweeps are resumable.

Usage
-----
    # Quick sweep with TPE (default)
    python -m scripts.optuna_hpo --n-trials 20 --max-epochs 2

    # Grid search over the discrete grid defined below
    python -m scripts.optuna_hpo --sampler grid --max-epochs 2

    # Resume a named study
    python -m scripts.optuna_hpo --study-name budget_dpo_v1 --n-trials 50
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from optuna.pruners import MedianPruner, NopPruner

# Make sure `src.*` imports work whether run as module or script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINT_DIR  # noqa: E402
from src.training.dpo_trainer import train_dpo  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

LOSS_TYPES = ["dpo", "simpo"]

# Grid points used when --sampler grid. Keep small — every combo is tried.
GRID_SEARCH_SPACE: dict[str, list[Any]] = {
    "lr":                          [5e-6, 1e-5, 5e-5],
    "dpo_beta":                    [0.05, 0.1, 0.2],
    "lambda_easy":                 [0.01, 0.05, 0.1],
    "lambda_hard":                 [0.001, 0.01, 0.03],
    "kl_penalty_weight":           [0.0, 0.01, 0.1],
    "batch_size":                  [4, 8],
    "gradient_accumulation_steps": [1, 2],
    "loss_type":                   ["dpo"],
    "length_ratio":                [1.0, 1.5, 2.0],
    "max_pairs_per_problem":       [20, 50, 100],
}


@dataclass
class SearchConfig:
    budget_aware: bool
    objective: str            # "tpca" | "accuracy" | "composite" | "tokens_easy"
    accuracy_floor: float     # trials with acc_easy < floor → infeasible
    max_epochs: int
    seed: int
    data_limit: Optional[int]
    model_name: Optional[str]
    num_workers: int
    use_mixed_precision: bool
    output_root: Path
    study_name: str
    use_wandb: bool


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

INFEASIBLE = float("inf")


def _compute_objective(
    best_metrics: dict[str, float],
    objective: str,
    accuracy_floor: float,
) -> float:
    """Scalar objective (minimize). Infeasible → +inf."""
    acc_easy = float(best_metrics.get("gen/accuracy_easy", 0.0) or 0.0)

    # Floor gate (use the most relevant accuracy)
    if accuracy_floor > 0.0 and acc_easy < accuracy_floor:
        return INFEASIBLE

    acc = float(best_metrics.get("gen/accuracy", 0.0) or 0.0)
    tpca = float(best_metrics.get("gen/tpca", INFEASIBLE) or INFEASIBLE)
    if not math.isfinite(tpca) and objective in {"tpca", "composite"}:
        return INFEASIBLE

    if objective == "tpca":
        return tpca
    if objective == "tokens_easy":
        tokens_easy = float(best_metrics.get("gen/avg_tokens_easy", INFEASIBLE) or INFEASIBLE)
        return tokens_easy
    if objective == "accuracy":
        return -acc  # maximize
    if objective == "val_loss":
        val_loss = float(best_metrics.get("val_loss", INFEASIBLE) or INFEASIBLE)
        return val_loss
    if objective == "composite":
        # lower tpca is better; accuracy pulls it down. α weights accuracy.
        alpha = 500.0  # 1% accuracy ≈ 5 tokens of tpca
        return tpca - alpha * acc
    raise ValueError(f"Unknown objective: {objective}")


def _sample_hyperparams(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one configuration. Shared by TPE / Random."""
    return {
        "lr":                          trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        "dpo_beta":                    trial.suggest_float("dpo_beta", 0.02, 0.5, log=True),
        "lambda_easy":                 trial.suggest_float("lambda_easy", 1e-3, 0.3, log=True),
        "lambda_hard":                 trial.suggest_float("lambda_hard", 1e-4, 0.1, log=True),
        "kl_penalty_weight":           trial.suggest_float("kl_penalty_weight", 1e-4, 1.0, log=True),
        "batch_size":                  trial.suggest_categorical("batch_size", [4, 8, 12]),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4]),
        "loss_type":                   trial.suggest_categorical("loss_type", LOSS_TYPES),
        "length_ratio":                trial.suggest_float("length_ratio", 1.5, 10.0),
        "max_pairs_per_problem":       trial.suggest_int("max_pairs_per_problem", 1, 5),
    }


def _sample_from_grid(trial: optuna.Trial) -> dict[str, Any]:
    """Sample by invoking suggest_categorical for each grid dim so GridSampler drives it."""
    return {
        name: trial.suggest_categorical(name, values)
        for name, values in GRID_SEARCH_SPACE.items()
    }


def _trial_output_dir(search: SearchConfig, trial_number: int) -> Path:
    return search.output_root / search.study_name / f"trial_{trial_number:04d}"


def _load_trial_metrics(output_dir: Path) -> dict[str, float]:
    """Read the best-model selection JSON written by train_dpo."""
    sel_path = output_dir / "best_model_selection.json"
    if sel_path.exists():
        with open(sel_path) as f:
            payload = json.load(f)
        return dict(payload.get("metrics") or {})

    # Fallback: last entry of metrics.json
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            entries = json.load(f)
        if entries:
            return dict(entries[-1])
    return {}


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _build_objective_fn(search: SearchConfig, use_grid: bool):
    def objective(trial: optuna.Trial) -> float:
        params = _sample_from_grid(trial) if use_grid else _sample_hyperparams(trial)
        output_dir = _trial_output_dir(search, trial.number)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trial %d starting. params=%s output=%s",
            trial.number, params, output_dir,
        )

        # Ensure lambda_hard < lambda_easy (soft nudge; only relevant for budget-aware).
        if search.budget_aware and params["lambda_hard"] > params["lambda_easy"]:
            params["lambda_hard"] = params["lambda_easy"] / 2.0
            trial.set_user_attr("lambda_hard_clipped", True)

        # Isolate W&B per trial
        if not search.use_wandb:
            os.environ["WANDB_MODE"] = "disabled"
        else:
            os.environ.setdefault("WANDB_PROJECT", "budget-aware-dpo-hpo")

        try:
            result = train_dpo(
                use_budget_aware=search.budget_aware,
                output_dir=output_dir,
                val_split=0.2,
                max_epochs=search.max_epochs,
                batch_size=int(params["batch_size"]),
                lr=float(params["lr"]),
                checkpoint_every=10**9,  # skip per-epoch checkpoint writes
                gradient_accumulation_steps=int(params["gradient_accumulation_steps"]),
                data_limit=search.data_limit,
                resume_from=None,
                seed=search.seed,
                use_wandb=search.use_wandb,
                run_name=f"{search.study_name}_trial{trial.number}",
                early_stopping_patience=max(2, search.max_epochs),  # effectively off for short HPO
                early_stopping_threshold=0.0,
                dpo_beta=float(params["dpo_beta"]),
                lambda_easy=float(params["lambda_easy"]),
                lambda_hard=float(params["lambda_hard"]),
                kl_penalty_weight=float(params["kl_penalty_weight"]),
                use_mixed_precision=search.use_mixed_precision,
                compile_model=False,
                num_workers=search.num_workers,
                model_name=search.model_name,
                loss_type=str(params["loss_type"]),
                best_model_metric="gen_tpca",
                accuracy_floor=None,
                length_ratio=float(params["length_ratio"]),
                max_pairs_per_problem=int(params["max_pairs_per_problem"]),
            )
        except torch.cuda.OutOfMemoryError:
            _cleanup_gpu()
            logger.warning("Trial %d OOM — pruning.", trial.number)
            raise optuna.TrialPruned("CUDA OOM")
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.error("Trial %d crashed: %s\n%s", trial.number, exc, traceback.format_exc())
            _cleanup_gpu()
            # Report as infeasible rather than failing the whole study
            trial.set_user_attr("error", str(exc))
            return INFEASIBLE

        best_metrics = _load_trial_metrics(output_dir)
        # Fall back to direct fields from train_dpo's return dict if selection file missing
        if not best_metrics:
            best_metrics = {"val_loss": float(result.get("best_val_loss", INFEASIBLE))}

        # Persist per-trial bookkeeping for later analysis
        trial.set_user_attr("best_metrics", best_metrics)
        trial.set_user_attr("params_effective", params)
        trial.set_user_attr("output_dir", str(output_dir))

        score = _compute_objective(best_metrics, search.objective, search.accuracy_floor)

        logger.info(
            "Trial %d done. score=%s best_metrics=%s",
            trial.number, score, best_metrics,
        )

        _cleanup_gpu()
        return score

    return objective


# ---------------------------------------------------------------------------
# Study construction
# ---------------------------------------------------------------------------

def _build_sampler(name: str, seed: int):
    if name == "tpe":
        return TPESampler(seed=seed, multivariate=True, group=True)
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "grid":
        return GridSampler(GRID_SEARCH_SPACE, seed=seed)
    raise ValueError(f"Unknown sampler: {name}")


def _build_pruner(enabled: bool):
    return MedianPruner(n_warmup_steps=1) if enabled else NopPruner()


def _make_storage(study_name: str, storage_arg: Optional[str]) -> str:
    if storage_arg:
        return storage_arg
    db_dir = PROJECT_ROOT / "checkpoints" / "optuna"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(db_dir / f'{study_name}.db').as_posix()}"


def _save_summary(study: optuna.Study, out_path: Path) -> None:
    rows = []
    for t in study.trials:
        rows.append({
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "params": t.params,
            "user_attrs": t.user_attrs,
        })
    payload = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "best_trial_number": (study.best_trial.number if study.best_trial else None),
        "best_value": (study.best_value if study.best_trial else None),
        "best_params": (study.best_params if study.best_trial else None),
        "trials": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote study summary to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Study / sampler
    p.add_argument("--study-name", type=str, default="budget_dpo_hpo")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL. Default: sqlite at checkpoints/optuna/<study>.db")
    p.add_argument("--sampler", choices=["tpe", "random", "grid"], default="tpe")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout", type=int, default=None, help="Wall clock seconds")
    p.add_argument("--pruner", action="store_true", help="Enable MedianPruner (best-effort)")
    p.add_argument("--seed", type=int, default=42)

    # Training knobs (fixed across trials)
    p.add_argument("--max-epochs", type=int, default=3)
    p.add_argument("--data-limit", type=int, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--baseline", action="store_true",
                   help="Tune baseline DPO (no length penalty) instead of budget-aware")

    # Objective
    p.add_argument("--objective",
                   choices=["tpca", "tokens_easy", "accuracy", "val_loss", "composite"],
                   default="tpca")
    p.add_argument("--accuracy-floor", type=float, default=0.10,
                   help="Trials with gen/accuracy_easy below this are infeasible (+inf).")

    # Output
    p.add_argument("--output-root", type=str,
                   default=str(CHECKPOINT_DIR / "optuna"))
    p.add_argument("--wandb", action="store_true")

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    search = SearchConfig(
        budget_aware=not args.baseline,
        objective=args.objective,
        accuracy_floor=args.accuracy_floor,
        max_epochs=args.max_epochs,
        seed=args.seed,
        data_limit=args.data_limit,
        model_name=args.model,
        num_workers=args.num_workers,
        use_mixed_precision=not args.no_mixed_precision,
        output_root=Path(args.output_root),
        study_name=args.study_name,
        use_wandb=args.wandb,
    )
    search.output_root.mkdir(parents=True, exist_ok=True)

    storage = _make_storage(args.study_name, args.storage)
    sampler = _build_sampler(args.sampler, args.seed)
    pruner = _build_pruner(args.pruner)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )
    study.set_user_attr("objective", args.objective)
    study.set_user_attr("budget_aware", search.budget_aware)
    study.set_user_attr("max_epochs", args.max_epochs)
    study.set_user_attr("accuracy_floor", args.accuracy_floor)

    logger.info(
        "Starting study '%s' sampler=%s n_trials=%s timeout=%s storage=%s",
        args.study_name, args.sampler, args.n_trials, args.timeout, storage,
    )

    objective_fn = _build_objective_fn(search, use_grid=(args.sampler == "grid"))

    try:
        study.optimize(
            objective_fn,
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=False,
            catch=(RuntimeError,),
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted — writing partial summary.")

    summary_path = search.output_root / search.study_name / "study_summary.json"
    _save_summary(study, summary_path)

    if study.best_trial is not None:
        logger.info("Best trial #%d value=%.6f", study.best_trial.number, study.best_value)
        logger.info("Best params: %s", study.best_params)
        logger.info("Best metrics: %s", study.best_trial.user_attrs.get("best_metrics"))
    else:
        logger.warning("No completed trials.")


if __name__ == "__main__":
    main()
