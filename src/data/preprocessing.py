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

from src.utils import approx_tokens, set_seed

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

    # TODO - is this what we intended
    if "gsm" in source:
        return 0
    if "math" in source:
        # TODO - not like it
    #   if tokens < EASY_TOKEN_THRESHOLD:
    #     return 0
    #   if tokens >= HARD_TOKEN_THRESHOLD:
    #     return 1
        return 1
    if tokens < EASY_TOKEN_THRESHOLD:
        return 0
    if tokens >= HARD_TOKEN_THRESHOLD:
        return 1
    return 0  # Default to Easy for ambiguous


# TODO - check that the real data has the key "correctness_flag" (it seems like only the dummy data does - PROBLEM is true)
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


# TODO - maybe reattach the end of the solution, that may include the final solution (answer)
def _make_short_answer(solution: str, expected: str = "") -> str:
    """Create short answer string from solution or expected_answer."""
    from src.evaluation.answer_extraction import extract_answer
    ans = extract_answer(solution) or expected
    if ans:
        return f"The answer is {ans}."
    short_solution_length = 100 # TODO - maybe extract into a common constant (~EASY_TOKEN_THRESHOLD)
    if len(solution) <= short_solution_length: 
        return solution
    return solution[:short_solution_length//2] + " ... " + solution[-short_solution_length//2:]


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
        # TODO - maybe we should somehow add an id per problem instead of having the full problem text
        groups[(ex["problem"], c)].append({**ex, "complexity": c, "label": label})

    pairs = []
    for (problem, complexity), items in groups.items():
        preferred = filter(lambda item: item['label'] == 'preferred', items)
        rejected = filter(lambda item: item['label'] == 'rejected', items)
        if preferred and rejected:
            for pw in preferred:
                for rj in rejected:
                    pairs.append({
                        "problem": problem,
                        "chosen": pw["generated_solution"],
                        "rejected": rj["generated_solution"],
                        "complexity": complexity,
                    })
        # TODO - if there are enough examples, it might be better to skip these cases where there is only preferred/rejected
        elif items:
            # Synthetic pair: create short vs long from first item
            # TODO - add more cases here
            # assumes correct, but no short answer so generates a short answer
            for ex in items:
                sol = ex["generated_solution"]
                expected = ex.get("expected_answer", "")
                short = _make_short_answer(sol, expected)
                if rejected and complexity == 0:  # Easy: short preferred, verbose rejected
                        pairs.append({"problem": problem, "chosen": short, "rejected": sol, "complexity": 0})
                elif preferred and complexity == 1:  # Hard: CoT preferred, oversimplified rejected
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

    preferred_lens, rejected_lens, complexities, easy, hard = [], [], [], 0, 0
    for pair in pairs:
        preferred_lens.append(approx_tokens(pair["chosen"]))
        rejected_lens.append(approx_tokens(pair["rejected"]))
        complexities.append(pair["complexity"])
        easy += pair["complexity"] == 0
        hard += pair["complexity"] == 1

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
