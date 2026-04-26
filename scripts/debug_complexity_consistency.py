#!/usr/bin/env python3
"""
Debug script: Find why same problem text gets different complexity scores.
Outputs only inconsistent cases (different complexity OR same complexity but different reasons).

Efficient approach: 
1. First pass: build set of duplicate problem texts (memory-efficient)
2. Second pass: stream again, classify on-the-fly, track results incrementally
"""

import sys
sys.path.insert(0, "/storage/arik/nlp_final_project")

import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from src.data.preprocessing import classify_complexity, _get_teacher_token_count, EASY_TOKEN_THRESHOLD, HARD_TOKEN_THRESHOLD


def get_reason(ex: dict) -> str:
    """Explain why classify_complexity returned what it did."""
    source = str(ex.get("problem_source", "")).lower()
    level = str(ex.get("level", "")).strip()
    tokens = _get_teacher_token_count(ex)

    if "gsm" in source or "gsm8k" in source:
        return "gsm8k_invariant"
    if "math" in source:
        if level in ("1", "2"):
            return f"math_level_{level}"
        if level in ("3",):
            return "math_level_3_fallback"
        if level in ("4", "5"):
            return f"math_level_{level}"
    if tokens < EASY_TOKEN_THRESHOLD:
        return f"token_fallback_easy_<{EASY_TOKEN_THRESHOLD}"
    if tokens > HARD_TOKEN_THRESHOLD:
        return f"token_fallback_hard_>{HARD_TOKEN_THRESHOLD}"
    return f"token_fallback_ambiguous_{EASY_TOKEN_THRESHOLD}<x<{HARD_TOKEN_THRESHOLD}"


def main():
    data_path = Path("/storage/arik/nlp_final_project/data/openmathinstruct.jsonl")
    output_path = Path("/storage/arik/nlp_final_project/data/debug_complexity_by_problem.json")

    print(f"Pass 1: Finding duplicate problem texts...")
    
    # Pass 1: identify duplicate problems (use set for memory efficiency)
    seen = set()
    duplicates = set()
    total = 0
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pass 1", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            problem = ex["problem"]
            total += 1
            
            if problem in seen:
                duplicates.add(problem)
            else:
                seen.add(problem)
    
    print(f"Total: {total:,}, Unique: {len(seen):,}, Duplicates: {len(duplicates):,}")
    del seen
    
    # Pass 2: collect classification results for duplicates
    print(f"\nPass 2: Classifying duplicate problems...")
    
    # Track: problem -> list of (complexity, reason, source, level, tokens)
    results: dict[str, list[dict]] = defaultdict(list)
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pass 2", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            problem = ex["problem"]
            
            if problem not in duplicates:
                continue
            
            c, _ = classify_complexity(ex)
            tokens = _get_teacher_token_count(ex)
            reason = get_reason(ex)
            
            results[problem].append({
                "complexity": c,
                "reason": reason,
                "problem_source": ex.get("problem_source"),
                "level": ex.get("level"),
                "teacher_token_count": tokens,
            })
    
    print(f"Classified {len(results):,} duplicate problems")
    
    # Analyze
    print(f"\nAnalyzing inconsistencies...")
    inconsistent = {}
    
    for problem, examples in results.items():
        if len(examples) < 2:
            continue
            
        complexities = [r["complexity"] for r in examples]
        reasons = [r["reason"] for r in examples]

        is_different_complexity = len(set(complexities)) > 1
        is_same_complexity_different_reason = len(set(complexities)) == 1 and len(set(reasons)) > 1

        if is_different_complexity or is_same_complexity_different_reason:
            inconsistent[problem] = {
                "examples": examples,
                "different_complexity": is_different_complexity,
                "different_reason": is_same_complexity_different_reason,
            }

    print(f"Found {len(inconsistent):,} inconsistent problems")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inconsistent, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")

    diff_complexity = sum(1 for v in inconsistent.values() if v["different_complexity"])
    diff_reason = sum(1 for v in inconsistent.values() if v["different_reason"])
    print(f"  - Different complexity: {diff_complexity}")
    print(f"  - Same complexity, different reason: {diff_reason}")


if __name__ == "__main__":
    main()