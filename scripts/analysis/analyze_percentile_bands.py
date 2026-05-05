#!/usr/bin/env python3
"""
Analyse the per-problem token-length distribution in data/problem_index.json
to answer two questions:

  1. Is the heuristic "MATH Level 1-2 => complexity=0 (Easy), Level 4-5 =>
     complexity=1 (Hard)" actually supported by the data — i.e., do L1-L2
     teacher-solution lengths look like GSM8K, or like L4-L5?

  2. For the complexity groupings that survive validation, what per-problem
     percentile band should we use to label "preferred" solutions?

We compute:
  * Pooled per-solution token-count percentiles for each atomic group:
    gsm8k, augmented_gsm8k, math L1-L5, augmented_math L1-L5, augmented_math
    (no level), and whole-complexity pools.
  * Pairwise distribution-overlap statistics (median-difference, pooled-pct
    overlap, cheap KS) between each candidate Easy/Hard sub-group and the
    GSM8K / math-L5 reference groups.
  * Across problems, the distribution of absolute token counts that
    correspond to within-problem percentiles p5, p10, ..., p95 — i.e., "for
    a typical C=0 problem, what absolute length is 'the 20th-percentile
    solution'?" — so we can judge where the band boundaries should sit.
  * For each candidate band (a small grid) on the C=0 and C=1 pools,
    simulate labeling and report the fraction of correct solutions that end
    up preferred vs length-rejected, and what absolute token range that
    maps to in aggregate.

Writes reports/percentile_band_analysis.md.
"""

import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "problem_index.json"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "percentile_band_analysis.md"

# Per-problem within-distribution percentiles we want to understand
WITHIN_PCTS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]
POOL_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# Candidate band grids for the final recommendation
EASY_BAND_GRID = [(5, 30), (5, 40), (10, 30), (10, 40), (10, 50), (15, 45), (20, 50)]
HARD_BAND_GRID = [(50, 90), (55, 90), (55, 95), (60, 90), (60, 95), (65, 95), (70, 95)]

MIN_SOLUTIONS_FOR_PER_PROBLEM = 5  # need enough solutions for percentiles to be meaningful


def iter_problem_index(path: Path):
    decoder = json.JSONDecoder()
    with open(path, "r") as f:
        buf = f.read(8192)
        idx = buf.index("[")
        buf = buf[idx + 1 :]
        while True:
            buf = buf.lstrip(" \n\r\t,")
            if not buf:
                chunk = f.read(8192)
                if not chunk:
                    return
                buf += chunk
                continue
            if buf[0] == "]":
                return
            try:
                obj, end = decoder.raw_decode(buf)
                buf = buf[end:]
                yield obj
            except json.JSONDecodeError:
                chunk = f.read(8192)
                if not chunk:
                    return
                buf += chunk


def normalize_level(level) -> str | None:
    if level is None:
        return None
    s = str(level).strip()
    if "?" in s or "unknown" in s.lower():
        return None
    for num in ("1", "2", "3", "4", "5"):
        if num in s:
            return num
    return None


def atomic_group_key(source: str, level: str | None) -> str:
    """Return the atomic group label for a problem."""
    s = (source or "").strip().lower()
    if s == "gsm8k":
        return "gsm8k"
    if s == "augmented_gsm8k":
        return "augmented_gsm8k"
    if s == "math":
        return f"math_L{level}" if level else "math_NoLevel"
    if s == "augmented_math":
        return f"augmented_math_L{level}" if level else "augmented_math_NoLevel"
    return f"other:{s}"


def pooled_percentiles(arr: np.ndarray, pcts: list[int]) -> dict[int, float]:
    if arr.size == 0:
        return {p: float("nan") for p in pcts}
    vals = np.percentile(arr, pcts)
    return {p: float(v) for p, v in zip(pcts, vals)}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = math.sqrt(((a.size - 1) * s1 ** 2 + (b.size - 1) * s2 ** 2) / (a.size + b.size - 2))
    if sp == 0:
        return float("nan")
    return (a.mean() - b.mean()) / sp


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Kolmogorov-Smirnov two-sample statistic (max |F_a - F_b|). No p-value."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    merged = np.concatenate([a, b])
    xs = np.unique(merged)
    fa = np.searchsorted(np.sort(a), xs, side="right") / a.size
    fb = np.searchsorted(np.sort(b), xs, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def distribution_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram-based overlap coefficient in [0, 1]. 1 = identical, 0 = disjoint."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, 51)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(ha, hb) * widths))


def summarise_within_problem_pcts(
    problems: list[dict], pcts: list[int]
) -> dict[int, dict[str, float]]:
    """For each within-problem percentile Q in `pcts`, return stats over problems
    of the absolute token count at that percentile."""
    per_pct: dict[int, list[float]] = {p: [] for p in pcts}
    for prob in problems:
        toks = prob.get("token_lengths") or []
        if len(toks) < MIN_SOLUTIONS_FOR_PER_PROBLEM:
            continue
        arr = np.asarray(sorted(toks), dtype=np.float64)
        vals = np.percentile(arr, pcts)
        for p, v in zip(pcts, vals):
            per_pct[p].append(float(v))
    out: dict[int, dict[str, float]] = {}
    for p, vals in per_pct.items():
        a = np.asarray(vals)
        if a.size == 0:
            out[p] = {"n": 0}
            continue
        out[p] = {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }
    return out


def simulate_band(problems: list[dict], band: tuple[float, float]) -> dict:
    """For each problem with enough solutions, label each solution
    preferred/rejected based on per-problem percentile band, then aggregate."""
    lo, hi = band
    n_problems = 0
    n_solutions = 0
    n_preferred = 0
    preferred_tokens: list[int] = []
    rejected_short_tokens: list[int] = []
    rejected_long_tokens: list[int] = []
    problems_with_any_preferred = 0

    for prob in problems:
        toks = prob.get("token_lengths") or []
        if len(toks) < MIN_SOLUTIONS_FOR_PER_PROBLEM:
            continue
        n_problems += 1
        sorted_t = sorted(int(t) for t in toks)
        n = len(sorted_t)
        picked_any = False
        for t in sorted_t:
            n_solutions += 1
            rank = 100.0 * bisect_right(sorted_t, t) / n
            if lo <= rank <= hi:
                n_preferred += 1
                preferred_tokens.append(t)
                picked_any = True
            elif rank < lo:
                rejected_short_tokens.append(t)
            else:
                rejected_long_tokens.append(t)
        if picked_any:
            problems_with_any_preferred += 1

    def pct_stats(vals: list[int]) -> dict:
        if not vals:
            return {"n": 0}
        a = np.asarray(vals)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "min": int(a.min()),
            "max": int(a.max()),
        }

    return {
        "band": list(band),
        "n_problems_eligible": n_problems,
        "n_solutions_seen": n_solutions,
        "n_preferred": n_preferred,
        "preferred_pct": 100.0 * n_preferred / n_solutions if n_solutions else 0.0,
        "problems_with_preferred_pct": (
            100.0 * problems_with_any_preferred / n_problems if n_problems else 0.0
        ),
        "preferred": pct_stats(preferred_tokens),
        "rejected_short": pct_stats(rejected_short_tokens),
        "rejected_long": pct_stats(rejected_long_tokens),
    }


def main():
    print(f"Loading {INDEX_PATH} ...", flush=True)

    # Keep per-atomic-group: list of problems (each = dict with token_lengths)
    atomic_problems: dict[str, list[dict]] = defaultdict(list)
    # Also pooled per-solution tokens by atomic group
    atomic_tokens: dict[str, list[int]] = defaultdict(list)
    # Complexity-level pools (using the precomputed "complexity" field)
    complexity_problems: dict[int, list[dict]] = defaultdict(list)
    complexity_tokens: dict[int, list[int]] = defaultdict(list)

    total = 0
    for obj in iter_problem_index(INDEX_PATH):
        total += 1
        if total % 100000 == 0:
            print(f"  processed {total:,} problems", flush=True)
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

    print(f"Done. {total:,} problems; {len(atomic_problems)} atomic groups.", flush=True)

    # -----------------------------------------------------------------
    # Part 1 — validate the MATH L1-L2 => Easy heuristic
    # -----------------------------------------------------------------
    # Candidate reference groups:
    #   gsm8k (canonical "Easy")
    #   math_L5 (canonical "Hard")
    # Compare each other atomic group against both references.
    ref_easy = np.asarray(atomic_tokens.get("gsm8k", []), dtype=np.int64)
    ref_hard = np.asarray(atomic_tokens.get("math_L5", []), dtype=np.int64)

    comparison_rows = []
    # All groups we care about for the heuristic question:
    groups_of_interest = [
        "gsm8k",
        "augmented_gsm8k",
        "math_L1",
        "math_L2",
        "math_L3",
        "math_L4",
        "math_L5",
        "augmented_math_L1",
        "augmented_math_L2",
        "augmented_math_L3",
        "augmented_math_L4",
        "augmented_math_L5",
        "augmented_math_NoLevel",
    ]
    for g in groups_of_interest:
        arr = np.asarray(atomic_tokens.get(g, []), dtype=np.int64)
        if arr.size == 0:
            continue
        pcts = pooled_percentiles(arr, POOL_PCTS)
        comparison_rows.append({
            "group": g,
            "n_problems": len(atomic_problems[g]),
            "n_solutions": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            **pcts,
            "d_vs_gsm8k": cohens_d(arr, ref_easy) if ref_easy.size else float("nan"),
            "d_vs_mathL5": cohens_d(arr, ref_hard) if ref_hard.size else float("nan"),
            "ks_vs_gsm8k": ks_statistic(arr, ref_easy) if ref_easy.size else float("nan"),
            "ks_vs_mathL5": ks_statistic(arr, ref_hard) if ref_hard.size else float("nan"),
            "overlap_gsm8k": distribution_overlap(arr, ref_easy) if ref_easy.size else float("nan"),
            "overlap_mathL5": distribution_overlap(arr, ref_hard) if ref_hard.size else float("nan"),
        })

    # -----------------------------------------------------------------
    # Part 2 — within-problem percentile statistics by complexity pool
    # -----------------------------------------------------------------
    within_stats = {
        "complexity_0": summarise_within_problem_pcts(
            complexity_problems[0], WITHIN_PCTS
        ),
        "complexity_1": summarise_within_problem_pcts(
            complexity_problems[1], WITHIN_PCTS
        ),
    }
    # Also compute per atomic group to see if L1-L2 within-problem shape looks
    # like gsm8k
    within_by_group = {
        g: summarise_within_problem_pcts(atomic_problems[g], WITHIN_PCTS)
        for g in groups_of_interest
        if g in atomic_problems
    }

    # -----------------------------------------------------------------
    # Part 3 — simulate candidate bands on each complexity pool
    # -----------------------------------------------------------------
    easy_sims = [simulate_band(complexity_problems[0], b) for b in EASY_BAND_GRID]
    hard_sims = [simulate_band(complexity_problems[1], b) for b in HARD_BAND_GRID]

    # -----------------------------------------------------------------
    # Render markdown
    # -----------------------------------------------------------------
    out: list[str] = []
    out.append("# Percentile-Band Analysis for DPO Preference Labeling")
    out.append("")
    out.append(
        f"Source: `data/problem_index.json` — {total:,} problems. This report answers "
        "two questions:"
    )
    out.append("")
    out.append(
        "1. Is the heuristic `MATH L1-2 → complexity=0`, `L4-5 → complexity=1` supported "
        "by the data-distribution of teacher-solution lengths?"
    )
    out.append(
        "2. Given the validated complexity pools, what per-problem percentile band "
        "should define `preferred` solutions?"
    )
    out.append("")

    # --- Part 1 table
    out.append("## 1. Validating the complexity heuristic")
    out.append("")
    out.append(
        "Pooled per-solution token-count percentiles, plus comparison to two reference "
        "groups: **gsm8k** (canonical Easy) and **math_L5** (canonical Hard)."
    )
    out.append("")
    out.append(
        "- `d_vs_*` = Cohen's d (pooled). Sign indicates direction (positive = longer than ref). "
        "|d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large."
    )
    out.append(
        "- `overlap_*` ∈ [0, 1] = fraction of probability mass shared with the reference "
        "(1 = identical, 0 = disjoint)."
    )
    out.append("- `ks_*` = max |CDF difference| with the reference (0 = identical, 1 = disjoint).")
    out.append("")
    header = (
        "| group | # problems | # solutions | mean | p10 | p25 | p50 | p75 | p90 | p95 | p99 | "
        "d vs gsm8k | d vs L5 | overlap gsm8k | overlap L5 | KS gsm8k | KS L5 |"
    )
    sep = "|---|---:|---:|---:|" + "---:|" * 8 + "---:|---:|---:|---:|---:|---:|"
    out.append(header)
    out.append(sep)
    for r in comparison_rows:
        out.append(
            "| "
            + " | ".join(
                [
                    r["group"],
                    f"{r['n_problems']:,}",
                    f"{r['n_solutions']:,}",
                    f"{r['mean']:.0f}",
                    f"{r[10]:.0f}",
                    f"{r[25]:.0f}",
                    f"{r[50]:.0f}",
                    f"{r[75]:.0f}",
                    f"{r[90]:.0f}",
                    f"{r[95]:.0f}",
                    f"{r[99]:.0f}",
                    f"{r['d_vs_gsm8k']:+.2f}",
                    f"{r['d_vs_mathL5']:+.2f}",
                    f"{r['overlap_gsm8k']:.2f}",
                    f"{r['overlap_mathL5']:.2f}",
                    f"{r['ks_vs_gsm8k']:.2f}",
                    f"{r['ks_vs_mathL5']:.2f}",
                ]
            )
            + " |"
        )
    out.append("")

    # --- Part 2 tables
    out.append("## 2. Within-problem percentile stats by complexity pool")
    out.append("")
    out.append(
        "For each problem with ≥"
        f" {MIN_SOLUTIONS_FOR_PER_PROBLEM} teacher solutions, compute its per-problem "
        "percentiles p5, p10, …, p95. Then across problems report the **median** (p50) "
        "absolute token count at each within-problem percentile, plus the p10/p90 across "
        "problems (i.e., variability from one problem to another)."
    )
    out.append("")

    def render_within_table(title: str, stats: dict[int, dict]):
        out.append(f"### {title}")
        out.append("")
        out.append(
            "| within-problem pct | problems | median abs tokens | p10 across probs | p90 across probs |"
        )
        out.append("|---:|---:|---:|---:|---:|")
        for p in WITHIN_PCTS:
            s = stats.get(p, {})
            if s.get("n", 0) == 0:
                out.append(f"| {p} | 0 | - | - | - |")
                continue
            out.append(
                f"| {p} | {s['n']:,} | {s['p50']:.0f} | {s['p10']:.0f} | {s['p90']:.0f} |"
            )
        out.append("")

    render_within_table("Complexity 0 (Easy) pool", within_stats["complexity_0"])
    render_within_table("Complexity 1 (Hard) pool", within_stats["complexity_1"])

    # Also per atomic group — useful to see whether L1-L2 look like gsm8k on
    # the WITHIN-PROBLEM axis (tight distribution of short solutions) rather
    # than just on the pooled-length axis.
    out.append("### Per-atomic-group, median absolute tokens at within-problem pct")
    out.append("")
    out.append(
        "Each cell is the *median across problems* of that problem's within-problem "
        "percentile. A group where within-problem p50 is already small (e.g. 150) "
        "is a group where the typical teacher solution is short."
    )
    out.append("")
    header = "| group | " + " | ".join(f"pP{p}" for p in WITHIN_PCTS) + " |"
    sep = "|---|" + "---:|" * len(WITHIN_PCTS)
    out.append(header)
    out.append(sep)
    for g, stats in within_by_group.items():
        vals = []
        for p in WITHIN_PCTS:
            s = stats.get(p, {})
            vals.append(f"{s['p50']:.0f}" if s.get("n", 0) else "-")
        out.append(f"| {g} | " + " | ".join(vals) + " |")
    out.append("")

    # --- Part 3 simulation tables
    out.append("## 3. Candidate-band simulation on each complexity pool")
    out.append("")
    out.append(
        "For each candidate band we label every solution in every eligible problem "
        "(≥"
        f" {MIN_SOLUTIONS_FOR_PER_PROBLEM} solutions) and aggregate: what % of correct "
        "solutions land in the band; what absolute token-count distribution the "
        "preferred, rejected-short, and rejected-long buckets have; and what fraction "
        "of problems end up with at least one preferred solution (low = band is so "
        "narrow that many problems produce no positive example)."
    )
    out.append("")

    def render_sims(title: str, sims: list[dict]):
        out.append(f"### {title}")
        out.append("")
        out.append(
            "| band | eligible probs | probs w/ any preferred (%) | preferred (%) | "
            "preferred med tok | pref p10/p90 | rej-short med | rej-long med |"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for s in sims:
            pref = s["preferred"]
            rshort = s["rejected_short"]
            rlong = s["rejected_long"]
            out.append(
                "| "
                + f"[{s['band'][0]:.0f}, {s['band'][1]:.0f}] | "
                + f"{s['n_problems_eligible']:,} | "
                + f"{s['problems_with_preferred_pct']:.1f} | "
                + f"{s['preferred_pct']:.1f} | "
                + (f"{pref['p50']:.0f}" if pref.get('n') else '-') + " | "
                + (f"{pref['p10']:.0f}/{pref['p90']:.0f}" if pref.get('n') else '-') + " | "
                + (f"{rshort['p50']:.0f}" if rshort.get('n') else '-') + " | "
                + (f"{rlong['p50']:.0f}" if rlong.get('n') else '-') + " |"
            )
        out.append("")

    render_sims("Complexity 0 (Easy) — candidate bands", easy_sims)
    render_sims("Complexity 1 (Hard) — candidate bands", hard_sims)

    # --- Minimum-solution distribution (informational)
    out.append("## 4. Diagnostic: solutions-per-problem distribution")
    out.append("")
    for c in (0, 1):
        counts = [len(p.get("token_lengths") or []) for p in complexity_problems[c]]
        a = np.asarray(counts)
        out.append(
            f"- complexity={c}: problems={a.size:,} | "
            f"mean={a.mean():.1f} | p10={np.percentile(a,10):.0f} "
            f"p50={np.percentile(a,50):.0f} p90={np.percentile(a,90):.0f} | max={a.max()}"
        )
    out.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(out))
    print(f"Wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
