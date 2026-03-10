#!/usr/bin/env python3
"""
Analyze dataset complexity heuristics for informed threshold decisions.

Computes statistics per docs/next_update_considerations.md:
1. MATH vs GSM8k counts
2. MATH level distribution (1-5)
3. Token counts (tiktoken cl100k_base) per category and per level
4. Percentile-based thresholds per category
5. CoT indicators and sentence count (optional heuristics)

Output: Verbose Markdown report for decision-making.
"""

import argparse
import json
import re
from pathlib import Path

from src.config import (
    DATA_PATH,
    GSM8K_TEST_PATH,
    MATH_TEST_PATH,
    PROJECT_ROOT,
    REAL_DATASET_PATH,
)
from src.utils import set_seed

set_seed(42)

# CoT indicator patterns (case-insensitive)
COT_PATTERNS = [
    r"\bfirst\b",
    r"\bstep\s*\d*",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bhence\b",
    r"\bso\s+we\b",
    r"\blet\s+me\s+think\b",
    r"\bin\s+summary\b",
    r"\bconsequently\b",
]


def get_tokenizer():
    """Get tiktoken cl100k_base tokenizer (GPT-4/Claude compatible)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        raise ImportError("tiktoken required. Install with: pip install tiktoken")


def count_tokens(text: str, enc) -> int:
    """Count tokens using tiktoken."""
    return len(enc.encode(str(text) if text else ""))


def count_sentences(text: str) -> int:
    """Simple sentence count: split on . ! ? and newlines."""
    if not text or not str(text).strip():
        return 0
    # Split on sentence-ending punctuation and newlines
    parts = re.split(r"[.!?]+\s+|\n+", str(text))
    return max(1, len([p for p in parts if p.strip()]))


def _median(arr: list[int]) -> float:
    """Compute median of integer list."""
    if not arr:
        return 0.0
    s = sorted(arr)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def count_cot_indicators(text: str) -> int:
    """Count total number of CoT indicator occurrences in text (each pattern match counted)."""
    if not text:
        return 0
    text_lower = str(text).lower()
    total = 0
    for pat in COT_PATTERNS:
        total += len(re.findall(pat, text_lower, re.IGNORECASE))
    return total


def has_cot_indicators(text: str) -> tuple[bool, list[str]]:
    """Check if text contains CoT indicator phrases. Returns (has_any, list of matched)."""
    if not text:
        return False, []
    text_lower = str(text).lower()
    matched = []
    for pat in COT_PATTERNS:
        if re.search(pat, text_lower, re.IGNORECASE):
            matched.append(pat)
    return len(matched) > 0, matched


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Load JSONL file, optionally with limit."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def get_solution_text(item: dict, source: str) -> str:
    """Extract solution text from item based on source (test vs training)."""
    if source == "test":
        return item.get("answer", "")
    return item.get("generated_solution", "")


def analyze_test_sets(math_path: Path, gsm8k_path: Path, enc) -> dict:
    """Analyze MATH and GSM8k test sets."""
    math_data = load_jsonl(math_path) if math_path.exists() else []
    gsm8k_data = load_jsonl(gsm8k_path) if gsm8k_path.exists() else []

    # 1. MATH vs GSM8k counts
    counts = {
        "math_total": len(math_data),
        "gsm8k_total": len(gsm8k_data),
        "total": len(math_data) + len(gsm8k_data),
    }

    # 2. MATH level distribution
    level_counts = {}
    for item in math_data:
        level = item.get("level", "Unknown")
        level_counts[level] = level_counts.get(level, 0) + 1

    # 3. Token stats per category and per level
    math_tokens = []
    gsm8k_tokens = []
    level_tokens: dict[str, list[int]] = {}

    for item in math_data:
        sol = get_solution_text(item, "test")
        n = count_tokens(sol, enc)
        math_tokens.append(n)
        level = item.get("level", "Unknown")
        level_tokens.setdefault(level, []).append(n)

    for item in gsm8k_data:
        sol = get_solution_text(item, "test")
        n = count_tokens(sol, enc)
        gsm8k_tokens.append(n)

    # 4. Percentiles
    def percentiles(arr: list[int], ps: list[int] = [10, 25, 50, 75, 90]) -> dict[int, float]:
        if not arr:
            return {p: 0 for p in ps}
        s = sorted(arr)
        n = len(s)
        out = {}
        for p in ps:
            idx = min(int((n - 1) * p / 100), n - 1) if p < 100 else n - 1
            out[p] = float(s[idx])
        return out

    math_percentiles = percentiles(math_tokens)
    gsm8k_percentiles = percentiles(gsm8k_tokens)

    # 5. CoT indicator count (avg/median) and sentence count (per category and per MATH level)
    math_cot_counts: list[int] = []
    gsm8k_cot_counts: list[int] = []
    math_sentences = []
    gsm8k_sentences = []
    level_cot_counts: dict[str, list[int]] = {}
    level_sentences: dict[str, list[int]] = {}

    for item in math_data:
        sol = get_solution_text(item, "test")
        n_cot = count_cot_indicators(sol)
        n_sent = count_sentences(sol)
        math_cot_counts.append(n_cot)
        math_sentences.append(n_sent)
        level = item.get("level", "Unknown")
        level_cot_counts.setdefault(level, []).append(n_cot)
        level_sentences.setdefault(level, []).append(n_sent)

    for item in gsm8k_data:
        sol = get_solution_text(item, "test")
        gsm8k_cot_counts.append(count_cot_indicators(sol))
        gsm8k_sentences.append(count_sentences(sol))

    return {
        "counts": counts,
        "level_counts": level_counts,
        "math_tokens": math_tokens,
        "gsm8k_tokens": gsm8k_tokens,
        "level_tokens": level_tokens,
        "math_tokens_avg": sum(math_tokens) / len(math_tokens) if math_tokens else 0,
        "gsm8k_tokens_avg": sum(gsm8k_tokens) / len(gsm8k_tokens) if gsm8k_tokens else 0,
        "level_tokens_avg": {
            k: sum(v) / len(v) if v else 0 for k, v in level_tokens.items()
        },
        "math_percentiles": math_percentiles,
        "gsm8k_percentiles": gsm8k_percentiles,
        "level_percentiles": {k: percentiles(v) for k, v in level_tokens.items()},
        "math_cot_counts": math_cot_counts,
        "gsm8k_cot_counts": gsm8k_cot_counts,
        "level_cot_counts": level_cot_counts,
        "math_cot_avg": sum(math_cot_counts) / len(math_cot_counts) if math_cot_counts else 0,
        "gsm8k_cot_avg": sum(gsm8k_cot_counts) / len(gsm8k_cot_counts) if gsm8k_cot_counts else 0,
        "level_cot_avg": {
            k: sum(v) / len(v) if v else 0 for k, v in level_cot_counts.items()
        },
        "math_cot_median": _median(math_cot_counts),
        "gsm8k_cot_median": _median(gsm8k_cot_counts),
        "level_cot_median": {k: _median(v) for k, v in level_cot_counts.items()},
        "math_sentences_avg": sum(math_sentences) / len(math_sentences) if math_sentences else 0,
        "gsm8k_sentences_avg": sum(gsm8k_sentences) / len(gsm8k_sentences) if gsm8k_sentences else 0,
        "level_sentences_avg": {
            k: sum(v) / len(v) if v else 0 for k, v in level_sentences.items()
        },
        "level_sentences_percentiles": {
            k: percentiles(v) for k, v in level_sentences.items()
        },
        "math_sentences_percentiles": percentiles(math_sentences),
        "gsm8k_sentences_percentiles": percentiles(gsm8k_sentences),
    }


def analyze_training_data(path: Path, limit: int, enc) -> dict | None:
    """Analyze training data (optional, with limit)."""
    if not path.exists():
        return None
    data = load_jsonl(path, limit=limit)
    if not data:
        return None

    math_items = [x for x in data if "math" in str(x.get("problem_source", "")).lower()]
    gsm8k_items = [x for x in data if "gsm" in str(x.get("problem_source", "")).lower()]

    math_tokens = [count_tokens(get_solution_text(x, "train"), enc) for x in math_items]
    gsm8k_tokens = [count_tokens(get_solution_text(x, "train"), enc) for x in gsm8k_items]

    def percentiles(arr: list[int], ps: list[int] = [10, 25, 50, 75, 90]) -> dict[int, float]:
        if not arr:
            return {p: 0 for p in ps}
        s = sorted(arr)
        return {p: s[max(0, int(len(s) * p / 100) - 1)] if p < 100 else s[-1] for p in ps}

    return {
        "total": len(data),
        "math_count": len(math_items),
        "gsm8k_count": len(gsm8k_items),
        "math_tokens_avg": sum(math_tokens) / len(math_tokens) if math_tokens else 0,
        "gsm8k_tokens_avg": sum(gsm8k_tokens) / len(gsm8k_tokens) if gsm8k_tokens else 0,
        "math_percentiles": percentiles(math_tokens),
        "gsm8k_percentiles": percentiles(gsm8k_tokens),
    }


def write_report(stats: dict, training_stats: dict | None, output_path: Path) -> None:
    """Write verbose Markdown report."""
    lines = [
        "# Complexity Heuristics Analysis Report",
        "",
        "**Purpose:** Statistics to inform complexity threshold decisions (EASY_TOKEN_THRESHOLD, HARD_TOKEN_THRESHOLD, level-based classification).",
        "",
        "---",
        "",
        "## 1. Dataset Composition: MATH vs GSM8k",
        "",
        "| Category | Count | Percentage |",
        "|----------|-------|------------|",
        f"| MATH | {stats['counts']['math_total']:,} | {100 * stats['counts']['math_total'] / max(1, stats['counts']['total']):.1f}% |",
        f"| GSM8k | {stats['counts']['gsm8k_total']:,} | {100 * stats['counts']['gsm8k_total'] / max(1, stats['counts']['total']):.1f}% |",
        f"| **Total** | **{stats['counts']['total']:,}** | 100% |",
        "",
        "---",
        "",
        "## 2. MATH Level Distribution",
        "",
        "MATH uses a 5-level difficulty scale (AoPS): Level 1 = easiest, Level 5 = hardest.",
        "",
        "| Level | Count | Percentage |",
        "|-------|-------|------------|",
    ]

    total_math = stats["counts"]["math_total"]
    for level in sorted(stats["level_counts"].keys(), key=lambda x: (x if x.startswith("Level") else "Level 0")):
        cnt = stats["level_counts"][level]
        pct = 100 * cnt / max(1, total_math)
        lines.append(f"| {level} | {cnt:,} | {pct:.1f}% |")
    lines.extend(["", "---", ""])

    # 3. Token statistics
    lines.extend([
        "## 3. Token Count Statistics (tiktoken cl100k_base)",
        "",
        "Using `tiktoken` cl100k_base (GPT-4/Claude compatible) for accurate token counts.",
        "",
        "### 3.1 Average Tokens per Category",
        "",
        "| Category | Avg Tokens | Min | Max |",
        "|----------|------------|-----|-----|",
    ])

    math_t = stats["math_tokens"]
    gsm_t = stats["gsm8k_tokens"]
    lines.append(f"| MATH | {stats['math_tokens_avg']:.1f} | {min(math_t) if math_t else 0} | {max(math_t) if math_t else 0} |")
    lines.append(f"| GSM8k | {stats['gsm8k_tokens_avg']:.1f} | {min(gsm_t) if gsm_t else 0} | {max(gsm_t) if gsm_t else 0} |")

    lines.extend([
        "",
        "### 3.2 Average Tokens per MATH Level",
        "",
        "| Level | Avg Tokens | Count |",
        "|-------|------------|-------|",
    ])
    for level in sorted(stats["level_tokens_avg"].keys(), key=lambda x: (x if x.startswith("Level") else "Level 0")):
        avg = stats["level_tokens_avg"][level]
        cnt = len(stats["level_tokens"].get(level, []))
        lines.append(f"| {level} | {avg:.1f} | {cnt:,} |")
    lines.extend(["", "---", ""])

    # 4. Percentiles
    lines.extend([
        "## 4. Percentile-Based Thresholds",
        "",
        "Suggested approach: Use percentiles instead of fixed 50/80 to adapt to each dataset.",
        "",
        "### 4.1 MATH Token Percentiles",
        "",
        "| Percentile | Token Count |",
        "|------------|-------------|",
    ])
    for p, v in stats["math_percentiles"].items():
        lines.append(f"| P{p} | {v:.0f} |")
    lines.extend([
        "",
        "### 4.2 GSM8k Token Percentiles",
        "",
        "| Percentile | Token Count |",
        "|------------|-------------|",
    ])
    for p, v in stats["gsm8k_percentiles"].items():
        lines.append(f"| P{p} | {v:.0f} |")
    lines.extend([
        "",
        "### 4.3 Per-Level Token Percentiles (MATH)",
        "",
    ])
    for level in sorted(stats["level_percentiles"].keys(), key=lambda x: (x if x.startswith("Level") else "Level 0")):
        lines.append(f"**{level}**")
        lines.append("")
        lines.append("| Percentile | Token Count |")
        lines.append("|------------|-------------|")
        for p, v in stats["level_percentiles"][level].items():
            lines.append(f"| P{p} | {v:.0f} |")
        lines.append("")
    lines.extend(["---", ""])

    # 5. CoT count and sentence count
    lines.extend([
        "## 5. CoT Indicators & Sentence Count (Optional Heuristics)",
        "",
        "CoT patterns: First, Step, Therefore, Thus, Hence, \"So we\", \"Let me think\", \"In summary\", Consequently.",
        "Count = total occurrences of these patterns per answer.",
        "",
        "### 5.1 CoT Indicator Count per Category",
        "",
        "| Category | Avg Count | Median |",
        "|----------|-----------|--------|",
        f"| MATH | {stats['math_cot_avg']:.1f} | {stats['math_cot_median']:.1f} |",
        f"| GSM8k | {stats['gsm8k_cot_avg']:.1f} | {stats['gsm8k_cot_median']:.1f} |",
        "",
        "### 5.2 CoT Indicator Count per MATH Level",
        "",
        "| Level | Avg Count | Median | Count |",
        "|-------|-----------|--------|-------|",
    ])
    for level in sorted(stats["level_cot_avg"].keys(), key=lambda x: (x if x.startswith("Level") else "Level 0")):
        avg = stats["level_cot_avg"][level]
        med = stats["level_cot_median"].get(level, 0)
        cnt = len(stats["level_tokens"].get(level, []))
        lines.append(f"| {level} | {avg:.1f} | {med:.1f} | {cnt:,} |")
    lines.extend([
        "",
        "### 5.3 Sentence Count",
        "",
        "| Category | Avg Sentences | P25 | P50 | P75 |",
        "|----------|---------------|-----|-----|-----|",
    ])
    mp = stats["math_sentences_percentiles"]
    gp = stats["gsm8k_sentences_percentiles"]
    lines.append(f"| MATH | {stats['math_sentences_avg']:.1f} | {mp.get(25, 0):.0f} | {mp.get(50, 0):.0f} | {mp.get(75, 0):.0f} |")
    lines.append(f"| GSM8k | {stats['gsm8k_sentences_avg']:.1f} | {gp.get(25, 0):.0f} | {gp.get(50, 0):.0f} | {gp.get(75, 0):.0f} |")
    lines.extend([
        "",
        "### 5.4 Sentence Count per MATH Level",
        "",
        "| Level | Avg Sentences | P25 | P50 | P75 | Count |",
        "|-------|---------------|-----|-----|-----|-------|",
    ])
    for level in sorted(stats["level_sentences_avg"].keys(), key=lambda x: (x if x.startswith("Level") else "Level 0")):
        avg = stats["level_sentences_avg"][level]
        lp = stats["level_sentences_percentiles"].get(level, {})
        cnt = len(stats["level_tokens"].get(level, []))
        lines.append(f"| {level} | {avg:.1f} | {lp.get(25, 0):.0f} | {lp.get(50, 0):.0f} | {lp.get(75, 0):.0f} | {cnt:,} |")
    lines.extend(["", "---", ""])

    # 6. Recommendations
    lines.extend([
        "## 6. Recommendations for Threshold Decisions",
        "",
        "### Current Fixed Thresholds",
        "- EASY_TOKEN_THRESHOLD = 50 (word count)",
        "- HARD_TOKEN_THRESHOLD = 80 (word count)",
        "",
        "### Observations",
        f"- MATH P25 ≈ {stats['math_percentiles'].get(25, 0):.0f} tokens, P50 ≈ {stats['math_percentiles'].get(50, 0):.0f}, P75 ≈ {stats['math_percentiles'].get(75, 0):.0f}",
        f"- GSM8k P25 ≈ {stats['gsm8k_percentiles'].get(25, 0):.0f} tokens, P50 ≈ {stats['gsm8k_percentiles'].get(50, 0):.0f}, P75 ≈ {stats['gsm8k_percentiles'].get(75, 0):.0f}",
        "- Token count (tiktoken) differs from word count; math LaTeX uses more tokens.",
        "",
        "### Suggested Approaches",
        "1. **Replace word count with tiktoken** for `teacher_token_count`.",
        "2. **Per-category percentiles:** E.g. Easy = below P25, Hard = above P75 for each category.",
        "3. **Level-based (MATH):** Level 1-2 → Easy, Level 3 → Medium, Level 4-5 → Hard.",
        "4. **CoT indicators:** Secondary signal; high prevalence in both categories.",
        "",
        "---",
        "",
        "## Appendix: Training Data (Optional)",
        "",
    ])
    if training_stats:
        lines.extend([
            f"Analyzed {training_stats['total']:,} training examples (limit applied).",
            "",
            "| Category | Count | Avg Tokens | P25 | P50 | P75 |",
            "|----------|-------|------------|-----|-----|-----|",
            f"| MATH | {training_stats['math_count']:,} | {training_stats['math_tokens_avg']:.1f} | "
            f"{training_stats['math_percentiles'].get(25, 0):.0f} | {training_stats['math_percentiles'].get(50, 0):.0f} | "
            f"{training_stats['math_percentiles'].get(75, 0):.0f} |",
            f"| GSM8k | {training_stats['gsm8k_count']:,} | {training_stats['gsm8k_tokens_avg']:.1f} | "
            f"{training_stats['gsm8k_percentiles'].get(25, 0):.0f} | {training_stats['gsm8k_percentiles'].get(50, 0):.0f} | "
            f"{training_stats['gsm8k_percentiles'].get(75, 0):.0f} |",
            "",
        ])
    else:
        lines.append("Training data not analyzed (file not found or --no-training).")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by scripts/analyze_complexity_heuristics.py*")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze complexity heuristics for threshold decisions")
    parser.add_argument("--math", type=Path, default=MATH_TEST_PATH, help="MATH test JSONL path")
    parser.add_argument("--gsm8k", type=Path, default=GSM8K_TEST_PATH, help="GSM8k test JSONL path")
    parser.add_argument(
        "--training",
        type=Path,
        default=None,
        help="Training data JSONL (default: real_openmathinstruct.jsonl or dummy)",
    )
    parser.add_argument("--training-limit", type=int, default=50000, help="Max training examples to analyze")
    parser.add_argument("--no-training", action="store_true", help="Skip training data analysis")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output report path")
    args = parser.parse_args()

    enc = get_tokenizer()

    print("Analyzing test sets...")
    stats = analyze_test_sets(args.math, args.gsm8k, enc)

    from src.config import DUMMY_DATASET_PATH, USE_DUMMY_DATA

    training_path = args.training or (
        DUMMY_DATASET_PATH if USE_DUMMY_DATA else REAL_DATASET_PATH
    )
    training_stats = None
    if not args.no_training and training_path.exists():
        print(f"Analyzing training data (limit={args.training_limit})...")
        training_stats = analyze_training_data(training_path, args.training_limit, enc)

    output_path = args.output or (
        PROJECT_ROOT / "docs" / "feature_reports" / "report_complexity_heuristics_analysis.md"
    )
    write_report(stats, training_stats, output_path)
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
