"""
4-Way Augmentation Pipeline: Complexity classification and DPO preference labeling.

Complexity Flag C:
- C=0 (Easy): GSM8K origin or low token count
- C=1 (Hard): MATH origin or high token count

Preference Labeling:
- Easy-Correct: Short direct paths = Preferred; verbose redundant = Rejected
- Hard-Correct: Detailed CoT = Preferred; oversimplified = Rejected
- Incorrect: Logically flawed = Rejected (all levels)
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.utils import set_seed

set_seed(42)

logger = logging.getLogger(__name__)

# Thresholds for complexity classification
EASY_TOKEN_THRESHOLD = 50  # Below = Easy
HARD_TOKEN_THRESHOLD = 80  # Above = Hard (MATH-like)
# Between: use problem_source (gsm8k -> Easy, math -> Hard)


def classify_complexity(example: dict) -> int:
    """
    C=0 Easy: GSM8K or low teacher token count.
    C=1 Hard: MATH or high teacher token count.
    """
    source = str(example.get("problem_source", "")).lower()
    tokens = example.get("teacher_token_count", 0) or 0

    if "gsm8k" in source or "gsm" in source:
        return 0
    if "math" in source:
        return 1
    if tokens < EASY_TOKEN_THRESHOLD:
        return 0
    if tokens >= HARD_TOKEN_THRESHOLD:
        return 1
    return 0  # Default to Easy for ambiguous


def label_preference(example: dict, complexity: int) -> str:
    """
    Returns "preferred" or "rejected" for this solution.
    """
    correct = example.get("correctness_flag", True)
    tokens = example.get("teacher_token_count", 0) or 0

    if not correct:
        return "rejected"

    if complexity == 0:  # Easy
        # Short direct = preferred; verbose = rejected
        if tokens <= EASY_TOKEN_THRESHOLD:
            return "preferred"
        return "rejected"

    # Hard
    # Detailed CoT = preferred; oversimplified = rejected
    if tokens >= HARD_TOKEN_THRESHOLD:
        return "preferred"
    return "rejected"


def _make_short_answer(solution: str, expected: str = "") -> str:
    """Create short answer string from solution or expected_answer."""
    from src.evaluation.answer_extraction import extract_answer
    ans = extract_answer(solution) or expected
    if ans:
        return f"The answer is {ans}."
    return solution[:100] + "..." if len(solution) > 100 else solution


def build_dpo_pairs(raw_data: list[dict]) -> list[dict]:
    """
    Group by (problem, complexity) and build preferred/rejected pairs.
    When natural pairs exist (multiple solutions per problem), use them.
    Otherwise create synthetic pairs: Easy = short preferred / verbose rejected;
    Hard = CoT preferred / oversimplified rejected.
    """
    from collections import defaultdict
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for ex in raw_data:
        c = classify_complexity(ex)
        label = label_preference(ex, c)
        groups[(ex["problem"], c)].append({**ex, "complexity": c, "label": label})

    pairs = []
    for (problem, complexity), items in groups.items():
        preferred = [x for x in items if x["label"] == "preferred"]
        rejected = [x for x in items if x["label"] == "rejected"]
        if preferred and rejected:
            for pw in preferred:
                for rj in rejected:
                    pairs.append({
                        "problem": problem,
                        "chosen": pw["generated_solution"],
                        "rejected": rj["generated_solution"],
                        "complexity": complexity,
                    })
        elif items:
            # Synthetic pair: create short vs long from first item
            ex = items[0]
            sol = ex["generated_solution"]
            expected = ex.get("expected_answer", "")
            short = _make_short_answer(sol, expected)
            if complexity == 0:  # Easy: short preferred, verbose rejected
                pairs.append({"problem": problem, "chosen": short, "rejected": sol, "complexity": 0})
            else:  # Hard: CoT preferred, oversimplified rejected
                pairs.append({"problem": problem, "chosen": sol, "rejected": short, "complexity": 1})
    return pairs


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def compute_statistics(pairs: list[dict]) -> dict[str, Any]:
    """Compute and log dataset statistics."""
    if len(pairs) == 0:
        return {}

    # Token lengths would need tokenizer - use word count as proxy for dummy
    def approx_tokens(x):
        return max(1, len(str(x).split()))

    preferred_lens = [approx_tokens(r["chosen"]) for r in pairs]
    rejected_lens = [approx_tokens(r["rejected"]) for r in pairs]
    complexities = [r["complexity"] for r in pairs]
    easy = sum(1 for c in complexities if c == 0)
    hard = sum(1 for c in complexities if c == 1)

    stats = {
        "total_pairs": len(pairs),
        "easy_pairs": easy,
        "hard_pairs": hard,
        "avg_preferred_tokens": sum(preferred_lens) / len(preferred_lens),
        "avg_rejected_tokens": sum(rejected_lens) / len(rejected_lens),
        "preferred_by_complexity": {
            0: sum(preferred_lens[i] for i in range(len(pairs)) if complexities[i] == 0) / max(1, easy),
            1: sum(preferred_lens[i] for i in range(len(pairs)) if complexities[i] == 1) / max(1, hard),
        },
    }
    return stats
