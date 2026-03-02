"""
Generate publication-ready figures from evaluation results.
Histograms of response lengths, results table.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_results(baseline_path: Path, budget_path: Path) -> tuple[list, list]:
    """Load per-sample results from baseline and budget-aware eval JSON."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(budget_path) as f:
        budget = json.load(f)
    return baseline.get("results", []), budget.get("results", [])


def plot_length_histograms(
    baseline_results: list[dict],
    budget_results: list[dict],
    output_path: Path,
) -> None:
    """Plot histograms of response token lengths for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    baseline_tokens = [r["tokens"] for r in baseline_results]
    budget_tokens = [r["tokens"] for r in budget_results]

    axes[0].hist(baseline_tokens, bins=15, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("Baseline DPO")
    axes[0].set_xlabel("Response tokens")
    axes[0].set_ylabel("Count")

    axes[1].hist(budget_tokens, bins=15, color="darkorange", edgecolor="black", alpha=0.7)
    axes[1].set_title("Budget-Aware DPO")
    axes[1].set_xlabel("Response tokens")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_length_by_complexity(
    baseline_results: list[dict],
    budget_results: list[dict],
    output_path: Path,
) -> None:
    """Bar chart: avg tokens Easy vs Hard for both models."""
    def avg_by_complexity(results: list[dict]) -> tuple[float, float]:
        easy = [r["tokens"] for r in results if r["complexity"] == 0]
        hard = [r["tokens"] for r in results if r["complexity"] == 1]
        return sum(easy) / len(easy) if easy else 0, sum(hard) / len(hard) if hard else 0

    be, bh = avg_by_complexity(baseline_results)
    bu_e, bu_h = avg_by_complexity(budget_results)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = [0, 1]
    width = 0.35
    ax.bar([i - width/2 for i in x], [be, bh], width, label="Baseline DPO", color="steelblue")
    ax.bar([i + width/2 for i in x], [bu_e, bu_h], width, label="Budget-Aware DPO", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(["Easy", "Hard"])
    ax.set_ylabel("Avg tokens")
    ax.legend()
    ax.set_title("Avg Response Length by Complexity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_results_table(metrics_path: Path, output_path: Path) -> None:
    """Generate markdown table from evaluation metrics."""
    with open(metrics_path) as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    has_math45 = any(m.get("math_level_4_5_accuracy") is not None for m in metrics.values())
    if has_math45:
        headers = "| Model | Accuracy | TPCA | Avg Tokens (Easy) | Avg Tokens (Hard) | MATH L4-5 |"
        sep = "|-------|----------|------|--------------------|-------------------|----------|"
    else:
        headers = "| Model | Accuracy | TPCA | Avg Tokens (Easy) | Avg Tokens (Hard) |"
        sep = "|-------|----------|------|--------------------|-------------------|"
    lines = [headers, sep]
    for name, m in metrics.items():
        acc = f"{m['accuracy']:.1%}"
        tpca = f"{m['tpca']:.1f}"
        easy = f"{m['avg_tokens_easy']:.1f}"
        hard = f"{m['avg_tokens_hard']:.1f}"
        row = f"| {name} | {acc} | {tpca} | {easy} | {hard} |"
        if has_math45:
            math45 = f"{m.get('math_level_4_5_accuracy', 0):.1%}" if m.get("math_level_4_5_accuracy") is not None else "—"
            row += f" {math45} |"
        lines.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_figures(
    eval_dir: Path,
    output_dir: Path,
    suffix: str = "_dummy",
) -> list[Path]:
    """Generate all figures from evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = eval_dir / f"baseline_eval{suffix}.json"
    budget_path = eval_dir / f"budget_aware_eval{suffix}.json"

    if not baseline_path.exists() or not budget_path.exists():
        raise FileNotFoundError(f"Evaluation files not found: {baseline_path}, {budget_path}")

    baseline_res, budget_res = load_eval_results(baseline_path, budget_path)
    paths = []

    p1 = output_dir / f"length_histograms{suffix}.pdf"
    plot_length_histograms(baseline_res, budget_res, p1)
    paths.append(p1)

    p2 = output_dir / f"length_by_complexity{suffix}.pdf"
    plot_length_by_complexity(baseline_res, budget_res, p2)
    paths.append(p2)

    return paths
