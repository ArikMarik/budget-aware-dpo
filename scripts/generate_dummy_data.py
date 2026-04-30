#!/usr/bin/env python3
"""
Generate synthetic examples in OpenMathInstruct-2 format.
Uses real problems from the problem_to_index so preprocessing doesn't fail.
Token counts are set to guarantee DPO pairs survive the length-ratio filter.
"""

import json
import os
import pickle
from bisect import insort
from pathlib import Path

from src.config import DATA_PATH, PROBLEM_TO_INDEX_PATH
from src.data.preprocessing import _pct_to_token_bounds, _band_for_complexity
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

set_seed(42)

DATA_ROOT = os.environ.get("DATA_PATH", DATA_PATH)
DUMMY_PATH = Path(str(DATA_ROOT)) / "dummy_openmathinstruct.jsonl"

NUM_EASY = 8
NUM_HARD = 8
LENGTH_RATIO = 2.0  # must match --length-ratio default in preprocess_dpo_data.py


def _get_preferred_bounds(v: dict) -> tuple[int, int]:
    complexity = v["complexity"]
    token_lengths = sorted(int(t) for t in v["token_lengths"])
    pct_low, pct_high = _band_for_complexity(complexity)
    return _pct_to_token_bounds(token_lengths, pct_low, pct_high)


def generate_dummy_data() -> list[dict]:
    """Generate synthetic examples using real problems from the problem index."""
    problem_to_index = pickle.load(open(PROBLEM_TO_INDEX_PATH, "rb"))

    easy_entries = [(k, v) for k, v in problem_to_index.items() if v["complexity"] == 0][:NUM_EASY]
    hard_entries = [(k, v) for k, v in problem_to_index.items() if v["complexity"] == 1][:NUM_HARD]

    examples = []

    for _, v in easy_entries:
        problem = v["problem"]
        answer = v["expected_answer"]
        source = v["problem_source"]
        low, high = _get_preferred_bounds(v)

        preferred_tokens = (low + high) // 2
        rejected_tokens = int(preferred_tokens * LENGTH_RATIO * 1.5)  # safely above ratio

        # Correct preferred (short) solution — falls in preferred band
        examples.append({
            "problem": problem,
            "generated_solution": f"The answer is {answer}.",
            "expected_answer": answer,
            "problem_source": source,
            "teacher_token_count": preferred_tokens,
            "correctness_flag": True,
        })
        # Correct verbose solution (rejected by length) — above band, ratio >= 2
        examples.append({
            "problem": problem,
            "generated_solution": (
                f"Let me think through this carefully step by step. "
                f"We need to analyze all the information given in the problem. "
                f"After working through each part systematically and checking our work, "
                f"we can confidently say that the final answer is {answer}."
            ),
            "expected_answer": answer,
            "problem_source": source,
            "teacher_token_count": rejected_tokens,
            "correctness_flag": True,
        })
        # Incorrect solution (rejected by correctness) — token count also satisfies ratio
        examples.append({
            "problem": problem,
            "generated_solution": "I'm not sure. The answer might be 42.",
            "expected_answer": answer,
            "problem_source": source,
            "teacher_token_count": rejected_tokens,
            "correctness_flag": False,
        })

    for _, v in hard_entries:
        problem = v["problem"]
        answer = v["expected_answer"]
        source = v["problem_source"]
        low, high = _get_preferred_bounds(v)

        preferred_tokens = (low + high) // 2
        rejected_tokens = int(preferred_tokens * LENGTH_RATIO * 1.5)

        # Correct preferred solution — falls in preferred band
        examples.append({
            "problem": problem,
            "generated_solution": (
                f"Applying the relevant theorems systematically, "
                f"we find that the answer is {answer}."
            ),
            "expected_answer": answer,
            "problem_source": source,
            "teacher_token_count": preferred_tokens,
            "correctness_flag": True,
        })
        # Correct verbose solution (rejected by length) — above band, ratio >= 2
        examples.append({
            "problem": problem,
            "generated_solution": (
                f"Let me work through this problem very carefully and in great detail. "
                f"First I identify all the given information. Then I apply the relevant "
                f"mathematical principles one by one. After checking each step thoroughly "
                f"and verifying there are no errors, I conclude the answer is {answer}."
            ),
            "expected_answer": answer,
            "problem_source": source,
            "teacher_token_count": rejected_tokens,
            "correctness_flag": True,
        })
        # Incorrect solution
        examples.append({
            "problem": problem,
            "generated_solution": "The answer is 0.",
            "expected_answer": answer,
            "problem_source": source,
            "teacher_token_count": rejected_tokens,
            "correctness_flag": False,
        })

    return examples


def main():
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    examples = generate_dummy_data()
    with open(DUMMY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %s dummy examples to %s", len(examples), DUMMY_PATH)


if __name__ == "__main__":
    main()
