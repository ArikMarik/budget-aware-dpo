#!/usr/bin/env python3
"""
Generate figures for percentile band analysis to include in the final project report.
Reads data/problem_index.json and produces PNG figures in reports/figures/.
"""

import sys
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add scripts directory to path to import analyze_percentile_bands
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_percentile_bands import (
    iter_problem_index,
    normalize_level,
    atomic_group_key,
    pooled_percentiles,
    cohens_d,
    ks_statistic,
    distribution_overlap,
    summarise_within_problem_pcts,
    simulate_band,
    WITHIN_PCTS,
    POOL_PCTS,
    EASY_BAND_GRID,
    HARD_BAND_GRID,
    MIN_SOLUTIONS_FOR_PER_PROBLEM,
)

# Configure matplotlib and seaborn for publication-ready figures
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 9

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "problem_index.json"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load and group data from problem_index.json."""
    atomic_problems: dict[str, list[dict]] = defaultdict(list)
    atomic_tokens: dict[str, list[int]] = defaultdict(list)
    complexity_problems: dict[int, list[dict]] = defaultdict(list)
    complexity_tokens: dict[int, list[int]] = defaultdict(list)
    
    total = 0
    print(f"Loading {INDEX_PATH} ...", flush=True)
    for obj in iter_problem_index(INDEX_PATH):
        total += 1
        if total % 100000 == 0:
            print(f"  Processed {total:,} problems", flush=True)
        toks = obj.get("token_lengths") or []
        if not toks:
            continue
        
        src = str(obj.get("problem_source", "")).strip().lower()
        lvl = normalize_level(obj.get("level"))
        key = atomic_group_key(src, lvl)
        atomic_problems[key].append(obj)
        atomic_tokens[key].extend(toks)
        
        c = obj.get("complexity")
        if c in (0, 1):
            complexity_problems[c].append(obj)
            complexity_tokens[c].extend(toks)
    
    print(f"Done loading. Total problems: {total:,}", flush=True)
    return atomic_problems, atomic_tokens, complexity_problems, complexity_tokens


def generate_figure1(atomic_tokens):
    """Figure 1: Token length distribution comparison (KDE) for key groups."""
    groups = ["gsm8k", "math_L1", "math_L2", "math_L4", "math_L5"]
    colors = sns.color_palette("tab10", n_colors=len(groups))
    
    plt.figure(figsize=(10, 6))
    for g, color in zip(groups, colors):
        arr = np.asarray(atomic_tokens.get(g, []), dtype=np.int64)
        if arr.size == 0:
            continue
        sns.kdeplot(arr, label=g, color=color, linewidth=2, clip=(0, None))
    
    plt.xlabel("Token Length (absolute)")
    plt.ylabel("Density")
    plt.title("Token Length Distribution by Group (KDE)")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "distribution_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved distribution_comparison.png", flush=True)


def generate_figure2(atomic_tokens):
    """Figure 2: Group distance metrics (Cohen's d, overlap) vs references."""
    groups_of_interest = [
        "gsm8k", "augmented_gsm8k",
        "math_L1", "math_L2", "math_L3", "math_L4", "math_L5",
        "augmented_math_L1", "augmented_math_L2", "augmented_math_L3",
        "augmented_math_L4", "augmented_math_L5", "augmented_math_NoLevel"
    ]
    ref_easy = np.asarray(atomic_tokens.get("gsm8k", []), dtype=np.int64)
    ref_hard = np.asarray(atomic_tokens.get("math_L5", []), dtype=np.int64)
    
    group_labels = []
    d_easy = []
    d_hard = []
    overlap_easy = []
    overlap_hard = []
    
    for g in groups_of_interest:
        arr = np.asarray(atomic_tokens.get(g, []), dtype=np.int64)
        if arr.size == 0:
            continue
        group_labels.append(g)
        d_easy.append(cohens_d(arr, ref_easy) if ref_easy.size else np.nan)
        d_hard.append(cohens_d(arr, ref_hard) if ref_hard.size else np.nan)
        overlap_easy.append(distribution_overlap(arr, ref_easy) if ref_easy.size else np.nan)
        overlap_hard.append(distribution_overlap(arr, ref_hard) if ref_hard.size else np.nan)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(group_labels))
    width = 0.35
    
    # Cohen's d subplot
    axes[0].bar(x - width/2, d_easy, width, label="Cohen's d vs gsm8k (Easy ref)", color="skyblue")
    axes[0].bar(x + width/2, d_hard, width, label="Cohen's d vs math_L5 (Hard ref)", color="salmon")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Cohen's d")
    axes[0].set_title("Group Distance from References (Cohen's d)")
    axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(group_labels, rotation=45, ha="right")
    
    # Overlap subplot
    axes[1].bar(x - width/2, overlap_easy, width, label="Overlap with gsm8k", color="skyblue")
    axes[1].bar(x + width/2, overlap_hard, width, label="Overlap with math_L5", color="salmon")
    axes[1].set_ylabel("Distribution Overlap (0-1)")
    axes[1].set_title("Group Overlap with References")
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(group_labels, rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "group_distances.png", bbox_inches="tight")
    plt.close()
    print("Saved group_distances.png", flush=True)


def generate_figure3(complexity_problems):
    """Figure 3: Within-problem percentiles by complexity pool (key figure)."""
    c0_stats = summarise_within_problem_pcts(complexity_problems[0], WITHIN_PCTS)
    c1_stats = summarise_within_problem_pcts(complexity_problems[1], WITHIN_PCTS)
    
    pcts = WITHIN_PCTS
    # C0 (Easy) data
    c0_medians = [c0_stats[p]["p50"] if c0_stats.get(p, {}).get("n", 0) > 0 else np.nan for p in pcts]
    c0_p10 = [c0_stats[p]["p10"] if c0_stats.get(p, {}).get("n", 0) > 0 else np.nan for p in pcts]
    c0_p90 = [c0_stats[p]["p90"] if c0_stats.get(p, {}).get("n", 0) > 0 else np.nan for p in pcts]
    
    # C1 (Hard) data
    c1_medians = [c1_stats[p]["p50"] if c1_stats.get(p, {}).get("n", 0) > 0 else np.nan for p in pcts]
    c1_p10 = [c1_stats[p]["p10"] if c1_stats.get(p, {}).get("n", 0) > 0 else np.nan for p in pcts]
    c1_p90 = [c1_stats[p]["p90"] if c1_stats.get(p, {}).get("n", 0) > 0 else np.nan for p in pcts]
    
    plt.figure(figsize=(10, 6))
    # Plot C0
    plt.plot(pcts, c0_medians, "o-", label="C0 (Easy) Median", color="blue", linewidth=2)
    plt.fill_between(pcts, c0_p10, c0_p90, alpha=0.2, color="blue")
    
    # Plot C1
    plt.plot(pcts, c1_medians, "s-", label="C1 (Hard) Median", color="red", linewidth=2)
    plt.fill_between(pcts, c1_p10, c1_p90, alpha=0.2, color="red")
    
    # Shade recommended bands (adjust these if different bands were chosen)
    plt.axvspan(10, 40, alpha=0.1, color="blue", label="Recommended Easy Band (10-40 pct)")
    plt.axvspan(55, 95, alpha=0.1, color="red", label="Recommended Hard Band (55-95 pct)")
    
    plt.xlabel("Within-Problem Percentile")
    plt.ylabel("Median Absolute Token Count")
    plt.title("Within-Problem Percentile vs Median Token Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "within_problem_percentiles.png", bbox_inches="tight")
    plt.close()
    print("Saved within_problem_percentiles.png", flush=True)


def generate_figure4(complexity_problems):
    """Figure 4: Band simulation results for easy and hard pools."""
    easy_sims = [simulate_band(complexity_problems[0], b) for b in EASY_BAND_GRID]
    hard_sims = [simulate_band(complexity_problems[1], b) for b in HARD_BAND_GRID]
    
    def get_pcts(sim):
        total = sim["n_solutions_seen"]
        if total == 0:
            return 0.0, 0.0, 0.0
        pref = sim["preferred_pct"]
        rej_short = (sim["rejected_short"]["n"] / total) * 100
        rej_long = (sim["rejected_long"]["n"] / total) * 100
        return pref, rej_short, rej_long
    
    # Easy band data
    easy_bands = [f"{b[0]}-{b[1]}" for b in EASY_BAND_GRID]
    easy_pref, easy_rs, easy_rl = zip(*[get_pcts(s) for s in easy_sims])
    
    # Hard band data
    hard_bands = [f"{b[0]}-{b[1]}" for b in HARD_BAND_GRID]
    hard_pref, hard_rs, hard_rl = zip(*[get_pcts(s) for s in hard_sims])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    width = 0.25
    
    # Easy subplot
    x_easy = np.arange(len(easy_bands))
    axes[0].bar(x_easy - width, easy_pref, width, label="Preferred", color="green")
    axes[0].bar(x_easy, easy_rs, width, label="Rejected (Short)", color="orange")
    axes[0].bar(x_easy + width, easy_rl, width, label="Rejected (Long)", color="red")
    axes[0].set_ylabel("Percentage of Solutions")
    axes[0].set_title("Band Simulation: Complexity 0 (Easy) Pool")
    axes[0].set_xticks(x_easy)
    axes[0].set_xticklabels(easy_bands, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_ylim(0, 100)
    
    # Hard subplot
    x_hard = np.arange(len(hard_bands))
    axes[1].bar(x_hard - width, hard_pref, width, label="Preferred", color="green")
    axes[1].bar(x_hard, hard_rs, width, label="Rejected (Short)", color="orange")
    axes[1].bar(x_hard + width, hard_rl, width, label="Rejected (Long)", color="red")
    axes[1].set_ylabel("Percentage of Solutions")
    axes[1].set_title("Band Simulation: Complexity 1 (Hard) Pool")
    axes[1].set_xticks(x_hard)
    axes[1].set_xticklabels(hard_bands, rotation=45, ha="right")
    axes[1].legend()
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "band_simulation.png", bbox_inches="tight")
    plt.close()
    print("Saved band_simulation.png", flush=True)


def generate_figure5(atomic_problems):
    """Figure 5: Within-problem percentiles by atomic group."""
    groups_of_interest = [
        "gsm8k", "augmented_gsm8k",
        "math_L1", "math_L2", "math_L3", "math_L4", "math_L5",
    ]
    within_by_group = {
        g: summarise_within_problem_pcts(atomic_problems[g], WITHIN_PCTS)
        for g in groups_of_interest
        if g in atomic_problems
    }
    
    plt.figure(figsize=(12, 7))
    colors = sns.color_palette("tab10", n_colors=len(within_by_group))
    for (g, stats), color in zip(within_by_group.items(), colors):
        medians = [stats[p]["p50"] if stats.get(p, {}).get("n", 0) > 0 else np.nan for p in WITHIN_PCTS]
        plt.plot(WITHIN_PCTS, medians, "o-", label=g, color=color, linewidth=1.5)
    
    plt.xlabel("Within-Problem Percentile")
    plt.ylabel("Median Absolute Token Count")
    plt.title("Within-Problem Percentiles by Atomic Group")
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "atomic_group_within_pct.png", bbox_inches="tight")
    plt.close()
    print("Saved atomic_group_within_pct.png", flush=True)


def main():
    atomic_problems, atomic_tokens, complexity_problems, complexity_tokens = load_data()
    
    print("\nGenerating Figure 1: Distribution comparison...", flush=True)
    generate_figure1(atomic_tokens)
    
    print("\nGenerating Figure 2: Group distances...", flush=True)
    generate_figure2(atomic_tokens)
    
    print("\nGenerating Figure 3: Within-problem percentiles...", flush=True)
    generate_figure3(complexity_problems)
    
    print("\nGenerating Figure 4: Band simulation...", flush=True)
    generate_figure4(complexity_problems)
    
    print("\nGenerating Figure 5: Atomic group within percentiles...", flush=True)
    generate_figure5(atomic_problems)
    
    print(f"\nAll figures saved to {FIGURE_DIR}", flush=True)


if __name__ == "__main__":
    main()
