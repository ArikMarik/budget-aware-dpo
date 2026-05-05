#!/usr/bin/env python3
"""
Analyze similarity search performance for augmented MATH problems.

Diagnoses why only ~4 augmented problems match the FAISS index:
1. Distribution of similarity scores for augmented problems
2. Effect of different thresholds
3. Embedding quality analysis
4. Comparison of score distributions between matching and non-matching problems
"""

from collections import defaultdict
import json
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.config import DATA_PATH, EMBEDDING_MODEL, SEED


INDEX_PATH = DATA_PATH / "math_problem_index"
OUTPUT_PATH = DATA_PATH / "similarity_analysis.json"


def load_faiss_index():
    """Load the FAISS index and metadata."""
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH / "index.faiss"))

    print("Loading metadata...")
    metadata = []
    with open(INDEX_PATH / "metadata.jsonl", "r") as f:
        for line in tqdm(f, desc="Loading metadata", unit=" problems"):
            metadata.append(json.loads(line))

    config_path = INDEX_PATH / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    model_name = config.get("embedding_model", EMBEDDING_MODEL)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"✓ Loaded index with {index.ntotal:,} problems")
    print(f"  Embedding model: {model_name}")
    print(f"  Dimension: {index.d}")

    return index, metadata, model


def load_augmented_problems(data_path: Path, limit: int | None = None):
    """Load augmented MATH problems from the dataset."""
    print(f"\nLoading augmented MATH problems from {data_path}...")

    # First pass: count lines
    total_lines = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1

    print(f"  Total lines in file: {total_lines:,}")

    # Second pass: load augmented_math problems with progress bar
    problems = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading augmented problems", unit=" examples", total=total_lines):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            source = str(ex.get("problem_source", "")).lower()
            if source == "augmented_math" and ex.get("problem"):
                problems.append(ex["problem"])

            if limit is not None and len(problems) >= limit:
                break

    print(f"✓ Loaded {len(problems):,} augmented MATH problems")
    return problems


def analyze_similarity_distribution(index, model, problems, sample_size: int = 1000):
    """Analyze the distribution of top-1 similarity scores."""
    print(f"\nAnalyzing similarity distribution (sample size: {sample_size})...")

    # Sample problems
    if sample_size and sample_size < len(problems):
        indices = np.random.choice(len(problems), sample_size, replace=False)
        sampled = [problems[i] for i in indices]
    else:
        sampled = problems
        sample_size = len(problems)

    # Encode all sampled problems with progress bar
    print("Encoding problems...")
    embeddings = model.encode(
        sampled,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(embeddings)

    # Search with progress bar
    print("Searching FAISS index...")
    k = 5  # Get top-5 for analysis
    scores_list = []
    indices_list = []

    # Process in batches for progress bar
    batch_size = 1000
    num_batches = (len(embeddings) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Searching index", unit=" batch"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(embeddings))
        batch = embeddings[start_idx:end_idx]
        scores_batch, indices_batch = index.search(batch, k=k)
        scores_list.append(scores_batch)
        indices_list.append(indices_batch)

    scores = np.vstack(scores_list)
    indices = np.vstack(indices_list)

    # Analyze top-1 scores
    top1_scores = scores[:, 0]

    stats = {
        "num_problems": len(sampled),
        "top1_mean": float(np.mean(top1_scores)),
        "top1_median": float(np.median(top1_scores)),
        "top1_std": float(np.std(top1_scores)),
        "top1_min": float(np.min(top1_scores)),
        "top1_max": float(np.max(top1_scores)),
        "top1_p10": float(np.percentile(top1_scores, 10)),
        "top1_p25": float(np.percentile(top1_scores, 25)),
        "top1_p75": float(np.percentile(top1_scores, 75)),
        "top1_p90": float(np.percentile(top1_scores, 90)),
        "top1_p95": float(np.percentile(top1_scores, 95)),
        "top1_p99": float(np.percentile(top1_scores, 99)),
    }

    # Count matches at different thresholds
    thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    threshold_counts = {}
    for thresh in thresholds:
        count = int(np.sum(top1_scores >= thresh))
        pct = 100 * count / len(top1_scores)
        threshold_counts[f"threshold_{thresh}"] = {
            "count": count,
            "percentage": round(pct, 2),
        }

    stats["threshold_analysis"] = threshold_counts

    # Histogram of scores
    hist_bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    hist_counts = np.histogram(top1_scores, bins=hist_bins)[0]
    stats["score_histogram"] = {
        "bins": hist_bins,
        "counts": hist_counts.tolist(),
    }

    # Analyze top-5 score degradation
    stats["top5_degradation"] = {}
    for i in range(1, 5):
        diff = top1_scores - scores[:, i]
        stats["top5_degradation"][f"top1_vs_top{i+1}_mean_diff"] = float(np.mean(diff))

    return stats, scores, indices


def analyze_matched_problems(metadata, problems, scores, indices, threshold: float = 0.7):
    """Analyze which augmented problems match and their characteristics."""
    print(f"\nAnalyzing matched problems (threshold={threshold})...")

    matched = scores[:, 0] >= threshold
    matched_indices = np.where(matched)[0]
    unmatched_indices = np.where(~matched)[0]

    print(f"Matched: {len(matched_indices)} ({100*len(matched_indices)/len(problems):.2f}%)")
    print(f"Unmatched: {len(unmatched_indices)} ({100*len(unmatched_indices)/len(problems):.2f}%)")

    # Analyze matched vs unmatched score distributions
    matched_scores = scores[matched_indices, 0]
    unmatched_scores = scores[unmatched_indices, 0]

    analysis = {
        "num_matched": int(len(matched_indices)),
        "num_unmatched": int(len(unmatched_indices)),
        "matched_pct": round(100 * len(matched_indices) / len(problems), 2),
        "matched_score_mean": float(np.mean(matched_scores)) if len(matched_scores) > 0 else None,
        "matched_score_std": float(np.std(matched_scores)) if len(matched_scores) > 0 else None,
        "unmatched_score_mean": float(np.mean(unmatched_scores)) if len(unmatched_scores) > 0 else None,
        "unmatched_score_std": float(np.std(unmatched_scores)) if len(unmatched_scores) > 0 else None,
    }

    # Analyze what original problems are being matched
    if len(matched_indices) > 0:
        matched_original_indices = indices[matched_indices, 0]
        matched_original_levels = defaultdict(int)
        for idx in matched_original_indices:
            if idx < len(metadata):
                level = metadata[idx].get("level")
                matched_original_levels[level] += 1

        analysis["matched_original_levels"] = dict(matched_original_levels)

        # Show some examples
        print("\nExample matches:")
        for i in range(min(5, len(matched_indices))):
            idx = matched_indices[i]
            score = scores[idx, 0]
            original_idx = indices[idx, 0]
            aug_problem = problems[idx][:100] + "..."
            if original_idx < len(metadata):
                orig_problem = metadata[original_idx]["problem"][:100] + "..."
                print(f"  Score {score:.4f}: '{aug_problem}' -> '{orig_problem}'")

    return analysis


def compare_augmented_to_original(index, metadata, model, data_path: Path, sample_size: int = 500):
    """Compare embedding similarities between augmented and original problems."""
    print("\nComparing augmented vs original problem embeddings...")

    # Load some original problems from the index
    original_problems = [m["problem"] for m in metadata[:sample_size]]

    # Load some augmented problems
    augmented = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size * 2:
                break
            ex = json.loads(line.strip())
            if ex.get("problem_source", "").lower() == "augmented_math":
                augmented.append(ex["problem"])
                if len(augmented) >= sample_size:
                    break

    if not augmented or not original_problems:
        print("Not enough data for comparison")
        return {}

    # Encode both sets with progress bars
    print("Encoding original problems...")
    orig_embeddings = model.encode(
        original_problems,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(orig_embeddings)

    print("Encoding augmented problems...")
    aug_embeddings = model.encode(
        augmented,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True,
    )
    faiss.normalize_L2(aug_embeddings)

    # Compute cross-similarities (augmented vs original)
    # Shape: (num_augmented, num_original)
    cross_similarities = np.dot(aug_embeddings, orig_embeddings.T)

    analysis = {
        "num_augmented": len(augmented),
        "num_original": len(original_problems),
        "cross_sim_mean": float(np.mean(cross_similarities)),
        "cross_sim_std": float(np.std(cross_similarities)),
        "cross_sim_min": float(np.min(cross_similarities)),
        "cross_sim_max": float(np.max(cross_similarities)),
        "cross_sim_p50": float(np.median(cross_similarities)),
        "cross_sim_p90": float(np.percentile(cross_similarities, 90)),
        "cross_sim_p95": float(np.percentile(cross_similarities, 95)),
    }

    # For each augmented problem, what's the best match to original?
    best_matches = np.max(cross_similarities, axis=1)
    analysis["best_match_mean"] = float(np.mean(best_matches))
    analysis["best_match_std"] = float(np.std(best_matches))
    analysis["best_match_p10"] = float(np.percentile(best_matches, 10))
    analysis["best_match_p25"] = float(np.percentile(best_matches, 25))
    analysis["best_match_p50"] = float(np.percentile(best_matches, 50))
    analysis["best_match_p75"] = float(np.percentile(best_matches, 75))
    analysis["best_match_p90"] = float(np.percentile(best_matches, 90))
    analysis["best_match_p95"] = float(np.percentile(best_matches, 95))

    return analysis


def plot_score_histogram(stats, ax):
    """Plot histogram of top-1 similarity scores."""
    bins = stats["score_histogram"]["bins"]
    counts = stats["score_histogram"]["counts"]

    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    width = [bins[i+1] - bins[i] for i in range(len(bins)-1)]

    bars = ax.bar(bin_centers, counts, width=width, edgecolor='black', alpha=0.7, color='steelblue')

    for thresh in [0.7, 0.75, 0.8]:
        ax.axvline(thresh, color='red', linestyle='--', alpha=0.6, linewidth=1.5,
                   label=f'threshold={thresh}' if thresh == 0.7 else None)

    ax.set_xlabel('Top-1 Similarity Score')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Similarity Scores (n={stats["num_problems"]:,})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    stats_text = f'Mean: {stats["top1_mean"]:.3f}\nMedian: {stats["top1_median"]:.3f}\nStd: {stats["top1_std"]:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_threshold_analysis(stats, ax):
    """Plot number of matches at different thresholds."""
    thresholds = []
    counts = []
    pcts = []

    for key, val in sorted(stats["threshold_analysis"].items(), key=lambda x: float(x[0].split('_')[1])):
        thresh = float(key.split('_')[1])
        thresholds.append(thresh)
        counts.append(val["count"])
        pcts.append(val["percentage"])

    ax1 = ax
    ax2 = ax.twinx()

    bars = ax1.bar(thresholds, counts, width=0.04, alpha=0.7, color='steelblue', label='Count')
    ax2.plot(thresholds, pcts, 'ro-', linewidth=2, markersize=8, label='Percentage')

    ax1.set_xlabel('Similarity Threshold')
    ax1.set_ylabel('Number of Matches', color='steelblue')
    ax2.set_ylabel('Percentage (%)', color='red')
    ax1.set_title('Matches vs Threshold')
    ax1.grid(axis='y', alpha=0.3)

    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='red')

    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct}%', ha='center', va='bottom', fontsize=8)


def plot_matched_vs_unmatched(match_analysis, ax):
    """Plot matched vs unmatched score distributions."""
    matched_mean = match_analysis["matched_score_mean"]
    matched_std = match_analysis["matched_score_std"]
    unmatched_mean = match_analysis["unmatched_score_mean"]
    unmatched_std = match_analysis["unmatched_score_std"]

    categories = ['Unmatched', 'Matched']
    means = [unmatched_mean, matched_mean]
    stds = [unmatched_std, matched_std]
    counts = [match_analysis["num_unmatched"], match_analysis["num_matched"]]

    colors = ['lightcoral', 'lightgreen']
    bars = ax.bar(categories, means, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('Mean Top-1 Similarity Score')
    ax.set_title(f'Matched vs Unmatched (threshold={match_analysis.get("matched_pct", 47.08)}%)')
    ax.grid(axis='y', alpha=0.3)

    for bar, count, std in zip(bars, counts, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'n={count:,}', ha='center', va='bottom', fontweight='bold')

    ax.axhline(0.7, color='red', linestyle='--', alpha=0.6, label='threshold=0.7')
    ax.legend()


def plot_cross_embedding_analysis(cross_analysis, ax):
    """Plot cross-embedding similarity distribution."""
    p50 = cross_analysis["cross_sim_p50"]
    p90 = cross_analysis["cross_sim_p90"]
    p95 = cross_analysis["cross_sim_p95"]
    sim_min = cross_analysis["cross_sim_min"]
    sim_max = cross_analysis["cross_sim_max"]
    mean = cross_analysis["cross_sim_mean"]

    stats_text = (
        f'Cross-Embedding Similarities\n'
        f'(Augmented vs Original)\n\n'
        f'Mean: {mean:.3f}\n'
        f'Std: {cross_analysis["cross_sim_std"]:.3f}\n'
        f'Min: {sim_min:.3f}\n'
        f'Max: {sim_max:.3f}\n\n'
        f'Percentiles:\n'
        f'  50th: {p50:.3f}\n'
        f'  90th: {p90:.3f}\n'
        f'  95th: {p95:.3f}\n\n'
        f'Best Match (per aug):\n'
        f'  Mean: {cross_analysis["best_match_mean"]:.3f}\n'
        f'  50th: {cross_analysis["best_match_p50"]:.3f}\n'
        f'  90th: {cross_analysis["best_match_p90"]:.3f}'
    )

    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Cross-Embedding Analysis Summary')


def plot_level_distribution(match_analysis, ax):
    """Plot distribution of matched original problem levels."""
    levels = match_analysis.get("matched_original_levels", {})

    if not levels:
        ax.text(0.5, 0.5, 'No level data available', transform=ax.transAxes,
                ha='center', va='center')
        return

    level_names = sorted(levels.keys())
    counts = [levels[lvl] for lvl in level_names]

    colors = plt.cm.Set3(np.linspace(0, 1, len(level_names)))
    wedges, texts, autotexts = ax.pie(counts, labels=level_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)

    ax.set_title(f'Levels of Matched Original Problems (n={sum(counts)})')


def plot_top5_degradation(stats, ax):
    """Plot top-5 score degradation."""
    degradation = stats["top5_degradation"]

    positions = [1, 2, 3, 4, 5]
    diffs = [0]
    for i in range(1, 5):
        diffs.append(degradation[f"top1_vs_top{i+1}_mean_diff"])

    ax.plot(positions, diffs, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(positions, diffs, alpha=0.3, color='steelblue')

    ax.set_xlabel('Rank')
    ax.set_ylabel('Mean Score Difference from Top-1')
    ax.set_title('Top-5 Score Degradation')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(positions)

    for pos, diff in zip(positions, diffs):
        ax.text(pos, diff + 0.002, f'{diff:.3f}', ha='center', va='bottom', fontsize=9)


def plot_best_match_distribution(cross_analysis, ax):
    """Plot distribution of best matches for augmented problems."""
    percentiles = ['p10', 'p25', 'p50', 'p75', 'p90', 'p95']
    values = [cross_analysis[f"best_match_{p}"] for p in percentiles]

    ax.plot(percentiles, values, 'go-', linewidth=2, markersize=8, label='Best match')

    for thresh in [0.6, 0.65, 0.7, 0.75]:
        ax.axhline(thresh, color='red', linestyle='--', alpha=0.4, linewidth=1)

    ax.set_xlabel('Percentile')
    ax.set_ylabel('Similarity Score')
    ax.set_title('Best Match Distribution (Augmented vs Original)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    for p, v in zip(percentiles, values):
        ax.text(p, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)


def visualize_results(json_path: Path = OUTPUT_PATH, output_dir: Path| None = None):
    """Generate visualization figures from analysis JSON file."""
    if output_dir is None:
        output_dir = DATA_PATH / "similarity_visualizations"

    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    stats = data["similarity_stats"]
    match_analysis = data["match_analysis"]
    cross_analysis = data["cross_embedding_analysis"]

    print(f"Generating visualizations in {output_dir}...")

    # 1. Score histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_score_histogram(stats, ax)
    plt.savefig(output_dir / "score_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved score_histogram.png")

    # 2. Threshold analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_threshold_analysis(stats, ax)
    plt.savefig(output_dir / "threshold_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved threshold_analysis.png")

    # 3. Matched vs unmatched
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_matched_vs_unmatched(match_analysis, ax)
    plt.savefig(output_dir / "matched_vs_unmatched.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved matched_vs_unmatched.png")

    # 4. Level distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_level_distribution(match_analysis, ax)
    plt.savefig(output_dir / "level_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved level_distribution.png")

    # 5. Best match distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_best_match_distribution(cross_analysis, ax)
    plt.savefig(output_dir / "best_match_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved best_match_distribution.png")

    # 6. Cross-embedding summary
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cross_embedding_analysis(cross_analysis, ax)
    plt.savefig(output_dir / "cross_embedding_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved cross_embedding_summary.png")

    # 7. Top-5 degradation
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_top5_degradation(stats, ax)
    plt.savefig(output_dir / "top5_degradation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved top5_degradation.png")

    # 8. Dashboard with all plots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    plot_score_histogram(stats, ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    plot_threshold_analysis(stats, ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_matched_vs_unmatched(match_analysis, ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_level_distribution(match_analysis, ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    plot_cross_embedding_analysis(cross_analysis, ax5)

    ax6 = fig.add_subplot(gs[2, 0])
    plot_top5_degradation(stats, ax6)

    ax7 = fig.add_subplot(gs[2, 1:])
    plot_best_match_distribution(cross_analysis, ax7)

    config = data["config"]
    fig.suptitle(
        f'Similarity Search Analysis Dashboard\n'
        f'Model: {config["embedding_model"]} | '
        f'Original Problems: {config["num_original_problems"]:,} | '
        f'Augmented Analyzed: {config["num_augmented_analyzed"]:,}',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.savefig(output_dir / "dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved dashboard.png")

    print(f"\nAll visualizations saved to {output_dir}/")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH / "openmathinstruct.jsonl")
    parser.add_argument("--sample-size", type=int, default=30_000, help="Number of augmented problems to sample")
    parser.add_argument("--full-analysis", action="store_true", help="Run full analysis on all problems (slow)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold to analyze")
    parser.add_argument("--viz-only", action="store_true", help="Generate visualization figures from existing JSON, skip analysis")
    args = parser.parse_args()

    # If only visualizing, skip the analysis
    if args.viz_only:
        if not OUTPUT_PATH.exists():
            print(f"Results file not found at {OUTPUT_PATH}. Run analysis first or use --visualize with --data-path.")
            return
        visualize_results(json_path=OUTPUT_PATH)
        return

    np.random.seed(SEED)

    # Load index
    if not INDEX_PATH.exists():
        print(f"Index not found at {INDEX_PATH}. Run build_math_problem_index.py first.")
        return

    index, metadata, model = load_faiss_index()

    # Load augmented problems (sample)
    if args.full_analysis:
        problems = load_augmented_problems(args.data_path, limit=None)
        sample_size = len(problems)
    else:
        problems = load_augmented_problems(args.data_path, limit=args.sample_size * 2)
        sample_size = min(args.sample_size, len(problems))
        indices = np.random.choice(len(problems), sample_size, replace=False)
        problems = [problems[i] for i in indices]

    print(f"\nAnalyzing {len(problems):,} augmented problems...")

    # Analyze similarity distribution
    stats, scores, indices = analyze_similarity_distribution(index, model, problems, sample_size=len(problems))

    # Analyze matched problems
    match_analysis = analyze_matched_problems(
        metadata, problems, scores, indices, threshold=args.threshold
    )

    # Compare augmented vs original embeddings
    cross_analysis = compare_augmented_to_original(index, metadata, model, args.data_path, sample_size=1000)

    # Compile results
    results = {
        "similarity_stats": stats,
        "match_analysis": match_analysis,
        "cross_embedding_analysis": cross_analysis,
        "config": {
            "threshold": args.threshold,
            "embedding_model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "num_original_problems": index.ntotal,
            "num_augmented_analyzed": len(problems),
        },
    }

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis saved to {OUTPUT_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original MATH problems in index: {index.ntotal:,}")
    print(f"Augmented problems analyzed: {len(problems):,}")
    print(f"\nTop-1 similarity score distribution:")
    print(f"  Mean: {stats['top1_mean']:.4f}")
    print(f"  Median: {stats['top1_median']:.4f}")
    print(f"  10th percentile: {stats['top1_p10']:.4f}")
    print(f"  90th percentile: {stats['top1_p90']:.4f}")
    print(f"\nMatches at different thresholds:")
    for thresh_str, data in stats["threshold_analysis"].items():
        print(f"  {thresh_str}: {data['count']:,} ({data['percentage']}%)")
    print(f"\nRecommendation:")
    if stats["top1_median"] < 0.6:
        print("  The embedding model may not be suitable for this task.")
        print("  Consider using a math-specific embedding model.")
    elif stats["top1_p90"] < 0.7:
        print("  The threshold of 0.7 is too high. Consider lowering to 0.6 or 0.65.")
    else:
        print("  The issue may be with the index building or embedding process.")

    # Generate visualizations if requested
    visualize_results(json_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()