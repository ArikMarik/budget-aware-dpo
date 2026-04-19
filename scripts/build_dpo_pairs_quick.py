#!/usr/bin/env python3
"""Quick DPO pair building - standalone version without external dependencies."""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

SEED = 42
USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "0") == "1"

EASY_TOKEN_THRESHOLD = 70
HARD_TOKEN_THRESHOLD = 130
_VALID_MATH_LEVELS = {"1", "2", "3", "4", "5"}


def get_paths():
    if USE_DUMMY_DATA:
        input_path = Path("data/dummy_openmathinstruct.jsonl")
        output_dir = Path("data/dummy_processed_dpo_dataset")
    else:
        input_path = Path("data/openmathinstruct.jsonl")
        output_dir = Path("data/processed_dpo_dataset")
    return input_path, output_dir


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading", unit=" lines"):
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _extract_answer(text: str) -> str | None:
    """Extract answer from solution text."""
    if not text:
        return None
    
    # Try boxed first
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    
    # Try aligned/equation environments
    match = re.search(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", text, re.DOTALL)
    if match:
        lines = match.group(0).split("\\\\")
        for line in reversed(lines):
            if "=" in line:
                return line.split("=")[-1].strip()
    
    # Try final answer line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if "answer" in line.lower():
            parts = line.split("=")
            if len(parts) > 1:
                return parts[-1].strip()
    
    return None


def _verify_correctness_with_fallback(example: dict) -> bool:
    """Verify correctness - simplified version."""
    # If correctness_flag exists, use it
    if example.get('correctness_flag') is not None:
        return example['correctness_flag']
    
    # Otherwise compare extracted answer
    generated = example.get("generated_solution", "")
    expected = example.get("expected_answer", "")
    
    if not expected:
        return True  # Assume correct if no expected answer
    
    gen_ans = _extract_answer(generated)
    if not gen_ans:
        return True  # Can't extract, assume correct
    
    # Simple normalize and compare
    gen_norm = re.sub(r'\\[a-zA-Z]+', '', gen_ans).strip()
    exp_norm = re.sub(r'\\[a-zA-Z]+', '', expected).strip()
    
    return gen_norm == exp_norm


def _get_teacher_token_count(example: dict) -> int:
    """Get teacher token count."""
    tc = example.get("teacher_token_count")
    if tc is not None and tc != 0:
        return int(tc)
    sol = example.get("generated_solution", "")
    return len(sol.split())


def classify_complexity(example: dict) -> tuple[int, str | None]:
    """Classify problem complexity. Returns (complexity, matched_level)."""
    source = str(example.get("problem_source", "")).lower()
    
    # GSM8K: always Easy
    if "gsm" in source:
        return 0, None
    
    # MATH: use level heuristic
    if "math" in source:
        level = example.get("level")
        level_str = str(level).strip() if level is not None else ""
        if level_str in _VALID_MATH_LEVELS:
            if level_str in ("1", "2"):
                return 0, level_str
            if level_str in ("4", "5"):
                return 1, level_str
    
    # Token fallback
    tokens = _get_teacher_token_count(example)
    if tokens < EASY_TOKEN_THRESHOLD:
        return 0, None
    if tokens > HARD_TOKEN_THRESHOLD:
        return 1, None
    return 0, None


def label_preference(example: dict, complexity: int) -> tuple[str, str]:
    """Label a solution as preferred or rejected."""
    correct = _verify_correctness_with_fallback(example)
    tokens = _get_teacher_token_count(example)
    
    if not correct:
        return "rejected", "incorrect"
    
    if complexity == 0:  # Easy
        if tokens <= EASY_TOKEN_THRESHOLD:
            return "preferred", "length"
        return "rejected", "length"
    
    # Hard
    if tokens >= HARD_TOKEN_THRESHOLD:
        return "preferred", "length"
    return "rejected", "length"


def build_dpo_pairs(raw_data: list[dict]) -> list[dict]:
    """Build DPO pairs from raw data."""
    groups: dict[str, list[dict]] = defaultdict(list)
    
    for ex in tqdm(raw_data, desc="Classifying & labeling", unit=" examples"):
        c, _ = classify_complexity(ex)
        label, rejection_reason = label_preference(ex, c)
        tc = _get_teacher_token_count(ex)
        groups[ex["problem"]].append({
            **ex,
            "complexity": c,
            "label": label,
            "rejection_reason": rejection_reason,
            "_token_count": tc
        })
    
    unique_problems = list(groups.keys())
    problem_to_id = {prob: idx for idx, prob in enumerate(unique_problems)}
    
    pairs: list[dict] = []
    
    for problem, items in tqdm(groups.items(), desc="Building pairs", unit=" groups"):
        preferred = [x for x in items if x["label"] == "preferred"]
        rejected = [x for x in items if x["label"] == "rejected"]
        
        problem_id = problem_to_id.get(problem, 0)
        complexity = items[0]["complexity"]
        
        if preferred and rejected:
            for pw in preferred:
                for rj in rejected:
                    pairs.append({
                        "problem": problem,
                        "problem_id": problem_id,
                        "chosen": pw["generated_solution"],
                        "rejected": rj["generated_solution"],
                        "complexity": complexity,
                        "rejection_reason": rj["rejection_reason"],
                        "chosen_length": pw["_token_count"],
                        "rejected_length": rj["_token_count"],
                    })
    
    print(f"Created a total of {len(pairs)} pairs")
    return pairs


def compute_statistics(pairs: list[dict]) -> dict:
    """Compute statistics."""
    total = len(pairs)
    if total == 0:
        return {"total_pairs": 0}
    
    rej_correctness = 0
    rej_length = 0
    count_easy = 0
    count_hard = 0
    chosen_len_sum = 0
    rejected_len_sum = 0
    
    for p in pairs:
        rr = p.get("rejection_reason")
        if rr == "incorrect":
            rej_correctness += 1
        elif rr == "length":
            rej_length += 1
        
        c = p.get("complexity", 0)
        if c == 0:
            count_easy += 1
        else:
            count_hard += 1
        
        chosen_len_sum += p.get("chosen_length", 0)
        rejected_len_sum += p.get("rejected_length", 0)
    
    return {
        "easy_token_threshold": EASY_TOKEN_THRESHOLD,
        "hard_token_threshold": HARD_TOKEN_THRESHOLD,
        "total_pairs": total,
        "rejected_by_correctness": rej_correctness,
        "rejected_by_length": rej_length,
        "rejected_by_correctness_pct": round(100 * rej_correctness / total, 2),
        "rejected_by_length_pct": round(100 * rej_length / total, 2),
        "easy_pairs": count_easy,
        "hard_pairs": count_hard,
        "easy_pairs_pct": round(100 * count_easy / total, 2),
        "hard_pairs_pct": round(100 * count_hard / total, 2),
        "avg_chosen_length": round(chosen_len_sum / total, 2),
        "avg_rejected_length": round(rejected_len_sum / total, 2),
    }


def _write_jsonl(path: Path, pairs: list[dict], desc: str = "Saving") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in tqdm(pairs, desc=desc, unit=" pairs"):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Quick DPO pair building")
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    return parser.parse_args()


def main(args=None):
    input_path, output_dir = get_paths()
    
    dataset_path = output_dir / "dataset.jsonl"
    meta_path = output_dir / "metadata.json"
    
    # Check if exists
    if not args or not args.force:
        if dataset_path.exists() and meta_path.exists():
            print(f"Processed dataset exists at {output_dir}")
            print("Use --force to regenerate")
            with open(meta_path) as f:
                stats = json.load(f)
            print(f"Stats: {stats}")
            return
    
    if not input_path.exists():
        print(f"Input data not found: {input_path}")
        return
    
    print(f"[1/3] Loading input data...")
    raw_data = load_jsonl(input_path)
    print(f"      Loaded {len(raw_data):,} examples")
    
    print(f"[2/3] Building DPO pairs...")
    pairs = build_dpo_pairs(raw_data)
    print(f"      Built {len(pairs):,} total pairs")
    
    num_unique_problems = len(set(p.get("problem_id", 0) for p in pairs))
    print(f"      Unique problem IDs: {num_unique_problems:,}")
    
    print(f"[3/3] Computing statistics and saving...")
    stats = compute_statistics(pairs)
    stats["seed"] = SEED
    stats["total_pairs"] = len(pairs)
    stats["num_unique_problems"] = num_unique_problems
    
    _write_jsonl(dataset_path, pairs)
    
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset statistics: {stats}")
    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)