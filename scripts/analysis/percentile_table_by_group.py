#!/usr/bin/env python3
"""
Build per-group percentile tables of teacher-solution token lengths from
data/problem_index.json, broken down by MATH level (1-5) and problem_source
(gsm8k, augmented_gsm8k, math, augmented_math).

For each group, pick 3 example problems near the 25/50/75 percentile of their
average token length, fetch one concrete (problem, generated_solution) pair
from data/openmathinstruct.jsonl, and record the solution's token count and
its percentile rank within the group.

Output: reports/token_length_percentiles_by_group.md
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "problem_index.json"
JSONL_PATH = PROJECT_ROOT / "data" / "openmathinstruct.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "token_length_percentiles_by_group.md"

PERCENTILES = [10, 25, 50, 75, 90]
LEVELS = ["1", "2", "3", "4", "5"]
SOURCES = ["gsm8k", "augmented_gsm8k", "math", "augmented_math"]


def iter_problem_index(path: Path):
    """Stream-parse the top-level JSON array in problem_index.json."""
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
    for num in ("1", "2", "3", "4", "5"):
        if num in s:
            return num
    return None


def group_key_source(source: str) -> str | None:
    s = (source or "").strip().lower()
    return s if s in SOURCES else None


def pct_rank(value: float, sorted_arr: np.ndarray) -> float:
    """Return the percentile rank (0-100) of value in sorted_arr."""
    if len(sorted_arr) == 0:
        return float("nan")
    idx = np.searchsorted(sorted_arr, value, side="right")
    return 100.0 * idx / len(sorted_arr)


def pick_example_ids(entries: list[dict], n: int = 3) -> list[dict]:
    """Pick n problems near the 25/50/75 percentile of avg_token_length."""
    if not entries:
        return []
    entries_sorted = sorted(entries, key=lambda e: e["avg_token_length"])
    N = len(entries_sorted)
    targets = [0.25, 0.50, 0.75][:n]
    picks = []
    seen = set()
    for t in targets:
        i = min(int(t * N), N - 1)
        # avoid dup if group is tiny
        while i in seen and i + 1 < N:
            i += 1
        seen.add(i)
        picks.append(entries_sorted[i])
    return picks


def main():
    print(f"Reading {INDEX_PATH} ...", flush=True)

    # Per-group flattened token lengths and per-group entry bags
    level_tokens: dict[str, list[int]] = defaultdict(list)
    source_tokens: dict[str, list[int]] = defaultdict(list)
    level_entries: dict[str, list[dict]] = defaultdict(list)
    source_entries: dict[str, list[dict]] = defaultdict(list)

    total = 0
    for obj in iter_problem_index(INDEX_PATH):
        total += 1
        if total % 100000 == 0:
            print(f"  processed {total:,} problems", flush=True)

        token_lengths = obj.get("token_lengths") or []
        if not token_lengths:
            continue

        lvl = normalize_level(obj.get("level"))
        src = group_key_source(obj.get("problem_source", ""))

        entry = {
            "problem_id": obj.get("problem_id"),
            "problem": obj.get("problem", ""),
            "avg_token_length": obj.get("avg_token_length", 0.0),
            "token_lengths": token_lengths,
            "level": lvl,
            "problem_source": obj.get("problem_source", ""),
            "complexity": obj.get("complexity"),
        }

        if lvl is not None:
            level_tokens[lvl].extend(token_lengths)
            level_entries[lvl].append(entry)
        if src is not None:
            source_tokens[src].extend(token_lengths)
            source_entries[src].append(entry)

    print(f"Total problems processed: {total:,}", flush=True)

    # Compute percentiles per group
    def pct_table(name: str, groups: list[str], tokens: dict[str, list[int]]):
        rows = []
        for g in groups:
            arr = np.asarray(tokens.get(g, []), dtype=np.int64)
            if arr.size == 0:
                continue
            p = np.percentile(arr, PERCENTILES)
            rows.append({
                "group": g,
                "n_problems": len(
                    level_entries[g] if name == "Level" else source_entries[g]
                ),
                "n_solutions": int(arr.size),
                "mean": float(arr.mean()),
                "min": int(arr.min()),
                "max": int(arr.max()),
                **{f"p{q}": float(v) for q, v in zip(PERCENTILES, p)},
            })
        return rows

    level_table = pct_table("Level", LEVELS, level_tokens)
    source_table = pct_table("Source", SOURCES, source_tokens)

    # Pick 3 example problems per group
    target_texts: set[str] = set()
    level_examples: dict[str, list[dict]] = {}
    source_examples: dict[str, list[dict]] = {}

    for lvl in LEVELS:
        picks = pick_example_ids(level_entries.get(lvl, []))
        level_examples[lvl] = picks
        for p in picks:
            target_texts.add(p["problem"])
    for src in SOURCES:
        picks = pick_example_ids(source_entries.get(src, []))
        source_examples[src] = picks
        for p in picks:
            target_texts.add(p["problem"])

    print(f"Target example problems to fetch solutions for: {len(target_texts)}", flush=True)

    # Single pass over JSONL to grab one solution per target problem
    solutions: dict[str, dict] = {}
    remaining = set(target_texts)
    print(f"Scanning {JSONL_PATH} for matching solutions ...", flush=True)
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not remaining:
                break
            if i % 1000000 == 0:
                print(f"  jsonl line {i:,} ... {len(remaining)} targets left", flush=True)
            # Cheap early filter — only parse if likely candidate
            # (fallback to full parse; jsonl is valid json per line)
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            prob = obj.get("problem", "")
            if prob in remaining:
                solutions[prob] = {
                    "generated_solution": obj.get("generated_solution", ""),
                    "teacher_token_count": obj.get("teacher_token_count", 0),
                    "expected_answer": obj.get("expected_answer", ""),
                    "correctness_flag": obj.get("correctness_flag", None),
                }
                remaining.discard(prob)

    print(
        f"Solutions found for {len(solutions)}/{len(target_texts)} problems",
        flush=True,
    )

    # Write markdown report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    def fmt_row(r: dict) -> str:
        return (
            f"| {r['group']} | {r['n_problems']:,} | {r['n_solutions']:,} | "
            f"{r['mean']:.1f} | {r['min']} | "
            + " | ".join(f"{r[f'p{q}']:.0f}" for q in PERCENTILES)
            + f" | {r['max']} |"
        )

    def render_examples(title: str, picks: list[dict], group_tokens: np.ndarray) -> list[str]:
        """Render 3 examples for a group with percentile of their solution length."""
        out = [f"#### Examples — {title}", ""]
        if not picks:
            out.append("_(no problems in this group)_")
            out.append("")
            return out
        sorted_arr = np.sort(group_tokens)
        for k, p in enumerate(picks, 1):
            sol = solutions.get(p["problem"])
            if sol is None:
                out.append(f"**Example {k}** — problem_id={p['problem_id']} (no matching solution found in jsonl)")
                out.append(f"- **Problem:** {p['problem'][:500]}")
                out.append("")
                continue
            tok = sol["teacher_token_count"]
            rank = pct_rank(tok, sorted_arr)
            avg = p["avg_token_length"]
            avg_rank = pct_rank(avg, sorted_arr)
            truncated_sol = sol["generated_solution"]
            if len(truncated_sol) > 1200:
                truncated_sol = truncated_sol[:1200] + " ...(truncated)"
            out.append(
                f"**Example {k}** — problem_id=`{p['problem_id']}` · "
                f"source=`{p['problem_source']}` · level=`{p['level']}` · "
                f"avg_token_length={avg:.1f} (group pct ≈ **{avg_rank:.1f}**)"
            )
            out.append("")
            out.append(f"- **Problem:** {p['problem']}")
            out.append(
                f"- **Solution token count:** {tok} → "
                f"group percentile ≈ **{rank:.1f}** "
                f"(expected_answer=`{sol['expected_answer']}`, "
                f"correct={sol['correctness_flag']})"
            )
            out.append("")
            out.append("> **Generated solution (truncated to 1200 chars):**")
            out.append(">")
            for para in truncated_sol.split("\n"):
                out.append(f"> {para}")
            out.append("")
        return out

    lines: list[str] = []
    lines.append("# Token-Length Percentile Tables by Group")
    lines.append("")
    lines.append(
        f"Source: `{INDEX_PATH.relative_to(PROJECT_ROOT)}` "
        f"({total:,} problems). Solutions sampled from "
        f"`{JSONL_PATH.relative_to(PROJECT_ROOT)}`."
    )
    lines.append("")
    lines.append(
        "Percentiles are over the **flattened list of per-solution teacher "
        "token counts** within each group (each problem contributes all of "
        "its teacher-solution lengths)."
    )
    lines.append("")

    # Level table
    lines.append("## 1. By MATH Level")
    lines.append("")
    header = (
        "| Level | # problems | # solutions | mean | min | "
        + " | ".join(f"p{q}" for q in PERCENTILES)
        + " | max |"
    )
    sep = "|---|---:|---:|---:|---:|" + ":---:|" * len(PERCENTILES) + "---:|"
    lines.append(header)
    lines.append(sep)
    for r in level_table:
        lines.append(fmt_row(r))
    lines.append("")

    for lvl in LEVELS:
        arr = np.asarray(level_tokens.get(lvl, []), dtype=np.int64)
        if arr.size == 0:
            continue
        lines.append(f"### Level {lvl}")
        lines.append("")
        lines.extend(render_examples(f"Level {lvl}", level_examples.get(lvl, []), arr))

    # Source table
    lines.append("## 2. By Problem Source")
    lines.append("")
    header = (
        "| Source | # problems | # solutions | mean | min | "
        + " | ".join(f"p{q}" for q in PERCENTILES)
        + " | max |"
    )
    lines.append(header)
    lines.append(sep)
    for r in source_table:
        lines.append(fmt_row(r))
    lines.append("")

    for src in SOURCES:
        arr = np.asarray(source_tokens.get(src, []), dtype=np.int64)
        if arr.size == 0:
            continue
        lines.append(f"### Source: `{src}`")
        lines.append("")
        lines.extend(render_examples(f"`{src}`", source_examples.get(src, []), arr))

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote report to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
