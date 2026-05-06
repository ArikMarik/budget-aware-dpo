"""
Generate publication-ready figures from evaluation results.
Histograms of response lengths, results table.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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
    """Plot overlaid histograms of response token lengths for both models."""
    baseline_tokens = [r["tokens"] for r in baseline_results]
    budget_tokens = [r["tokens"] for r in budget_results]

    # Shared bins for aligned comparison
    all_tokens = baseline_tokens + budget_tokens
    if all_tokens:
        min_tok = min(all_tokens)
        max_tok = max(all_tokens)
    else:
        min_tok = 0
        max_tok = 100

    n_bins = 20
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_axisbelow(True)            # draw grid below artists with default zorder
    ax.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
    ax.hist(baseline_tokens, bins=n_bins, range=(min_tok, max_tok), color="tab:blue", alpha=0.5, label="Baseline")
    ax.hist(budget_tokens, bins=n_bins, range=(min_tok, max_tok), color="tab:orange", alpha=0.5, label="Budget-Aware DPO")
    ax.set_title("Response Token Length Distribution")
    ax.set_xlabel("Response tokens")
    ax.set_ylabel("Count")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=15))

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

    # Single header row with parenthesized subcolumn names
    header = "| Model | Acc (Overall) | Acc (Easy) | Acc (Hard) | TPCA (Overall) | TPCA (Easy) | TPCA (Hard) | Tok (Total) | Tok (Easy) | Tok (Hard) | Eff (Overall) | Eff (Easy) | Eff (Hard) |"
    sep = "|-------|--------------|-----------|-----------|---------------|------------|------------|-------------|-----------|-----------|---------------|-----------|-----------|"

    lines = [header, sep]
    for name, m in metrics.items():
        # Accuracy (Overall, Easy, Hard)
        acc_overall = f"{m['accuracy']:.1%}"
        acc_easy_val = m.get('accuracy_easy')
        acc_easy = f"{acc_easy_val:.1%}" if acc_easy_val is not None else "—"
        acc_hard_val = m.get('accuracy_hard')
        acc_hard = f"{acc_hard_val:.1%}" if acc_hard_val is not None else "—"

        # TPCA (Overall, Easy, Hard)
        tpca_overall = f"{m['tpca']:.1f}"
        tpca_easy_val = m.get('avg_tokens_easy_correct')
        tpca_easy = f"{tpca_easy_val:.1f}" if tpca_easy_val is not None else "—"
        tpca_hard_val = m.get('avg_tokens_hard_correct')
        tpca_hard = f"{tpca_hard_val:.1f}" if tpca_hard_val is not None else "—"

        # Avg Tokens (Total, Easy, Hard)
        avg_total_val = m.get('average_tokens_length')
        avg_total = f"{avg_total_val:.1f}" if avg_total_val is not None else "—"
        avg_easy = f"{m['avg_tokens_easy']:.1f}"
        avg_hard = f"{m['avg_tokens_hard']:.1f}"

        # Efficiency (Overall, Easy, Hard)
        eff_overall_val = m.get('efficiency')
        eff_overall = f"{eff_overall_val:.2f}" if eff_overall_val is not None else "—"
        eff_easy_val = m.get('efficiency_easy')
        eff_easy = f"{eff_easy_val:.2f}" if eff_easy_val is not None else "—"
        eff_hard_val = m.get('efficiency_hard')
        eff_hard = f"{eff_hard_val:.2f}" if eff_hard_val is not None else "—"

        # Build row (13 fixed columns)
        row_parts = [
            name,
            acc_overall, acc_easy, acc_hard,
            tpca_overall, tpca_easy, tpca_hard,
            avg_total, avg_easy, avg_hard,
            eff_overall, eff_easy, eff_hard
        ]

        row = "| " + " | ".join(row_parts) + " |"
        lines.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_figures(
    baseline_results_path: Path,
    budget_results_path: Path,
    output_dir: Path,
    suffix: str = "_dummy",
) -> list[Path]:
    """Generate all figures from evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_results_path.exists() or not budget_results_path.exists():
        raise FileNotFoundError(f"Evaluation files not found: {baseline_results_path}, {budget_results_path}")

    baseline_res, budget_res = load_eval_results(baseline_results_path, budget_results_path)
    paths = []

    p1 = output_dir / f"length_histograms{suffix}.pdf"
    plot_length_histograms(baseline_res, budget_res, p1)
    paths.append(p1)

    p2 = output_dir / f"length_by_complexity{suffix}.pdf"
    plot_length_by_complexity(baseline_res, budget_res, p2)
    paths.append(p2)

    return paths
