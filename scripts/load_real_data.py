#!/usr/bin/env python3
"""
Load real datasets (OpenMathInstruct-2, GSM8K test, MATH test) and convert to JSONL.
Saves training data for DPO preprocessing; holds out test sets for Phase 9 evaluation.
"""

import argparse
import json
import re
from pathlib import Path

from src.config import DATA_PATH, GSM8K_TEST_PATH, MATH_TEST_PATH, REAL_DATASET_PATH
from src.evaluation.answer_extraction import extract_answer, normalize_answer
from src.utils import count_tokens, get_logger, set_seed

logger = get_logger(__name__)

set_seed(42)

MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


def normalize_problem(text: str) -> str:
    """Normalize problem text for matching: collapse whitespace, strip. Improves level lookup for MATH-origin problems."""
    if not text:
        return ""
    return " ".join(str(text).split())


def verify_correctness(generated_solution: str, expected_answer: str) -> bool:
    """Verify if generated_solution matches expected_answer. Returns False when expected_answer empty (cannot verify)."""
    if not expected_answer or not str(expected_answer).strip():
        return False  # No ground truth: cannot verify, treat as incorrect
    pred = extract_answer(generated_solution)
    if pred is None:
        return False
    return normalize_answer(pred) == normalize_answer(str(expected_answer))


def extract_gsm8k_answer(answer: str) -> str:
    """Extract final answer from GSM8K format (#### N)."""
    m = re.search(r"####\s*(\S+)", answer)
    return m.group(1).strip() if m else ""


def convert_openmathinstruct(item: dict, problem_to_level: dict | None = None) -> dict:
    """Convert OpenMathInstruct-2 item to our format. Add level when problem_source has math and problem matches MATH train."""
    solution = item.get("generated_solution", "")
    expected = item.get("expected_answer", "")
    out = {
        "problem": item["problem"],
        "generated_solution": solution,
        "expected_answer": expected,
        "problem_source": item.get("problem_source", "unknown"),
        "teacher_token_count": count_tokens(solution),
        "correctness_flag": verify_correctness(solution, expected),
    }
    problem = item.get("problem", "")
    source = str(item.get("problem_source", "")).lower()
    if problem_to_level and "math" in source and problem:
        out["level"] = problem_to_level.get(normalize_problem(problem), "")
    else:
        out["level"] = ""
    return out


def load_math_problem_to_level() -> dict[str, str]:
    """Load MATH train split and build problem text -> level mapping. Used to enrich OpenMathInstruct-2 with levels."""
    from datasets import load_dataset, concatenate_datasets

    try:
        parts = [
            load_dataset("EleutherAI/hendrycks_math", cfg, split="train", trust_remote_code=False)
            for cfg in MATH_CONFIGS
        ]
        ds = concatenate_datasets(parts)
    except Exception:
        try:
            ds = load_dataset("hendrycks/competition_math", split="train")
        except Exception:
            ds = load_dataset("lighteval/MATH", split="train")
    mapping = {}
    for item in ds:
        problem = item.get("problem", item.get("question", ""))
        level = item.get("level", "")
        if problem and level:
            mapping[normalize_problem(problem)] = str(level)
    return mapping


def load_openmathinstruct(split: str = "train_1M", limit: int | None = None) -> list[dict]:
    """Load OpenMathInstruct-2 from HuggingFace. Enriches MATH-origin problems with level from MATH train."""
    from datasets import load_dataset

    logger.info("Loading MATH train for level mapping...")
    problem_to_level = load_math_problem_to_level()
    logger.info("Built level map for %s MATH problems", f"{len(problem_to_level):,}")

    dataset = load_dataset("nvidia/OpenMathInstruct-2", split=split)
    examples = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break
        examples.append(convert_openmathinstruct(dict(item), problem_to_level))
    return examples


def load_gsm8k_test() -> list[dict]:
    """Load GSM8K test set for Phase 9 evaluation."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = []
    for item in ds:
        examples.append({
            "problem": item["question"],
            "answer": item["answer"],
            "expected_answer": extract_gsm8k_answer(item["answer"]),
            "problem_source": "gsm8k",
        })
    return examples


def load_math_test() -> list[dict]:
    """Load MATH test set for Phase 9 evaluation."""
    from datasets import load_dataset, concatenate_datasets

    # EleutherAI/hendrycks_math has per-subject configs; concatenate all
    try:
        parts = [
            load_dataset("EleutherAI/hendrycks_math", cfg, split="test", trust_remote_code=False)
            for cfg in MATH_CONFIGS
        ]
        ds = concatenate_datasets(parts)
    except Exception:
        try:
            ds = load_dataset("hendrycks/competition_math", split="test")
        except Exception:
            ds = load_dataset("lighteval/MATH", split="test")
    examples = []
    for item in ds:
        problem = item.get("problem", item.get("question", ""))
        solution = item.get("solution", item.get("answer", ""))
        expected = item.get("answer", "")
        if not expected and solution and "\\boxed{" in str(solution):
            m = re.search(r"\\boxed\{([^}]+)\}", str(solution))
            expected = m.group(1).strip() if m else ""
        examples.append({
            "problem": problem,
            "answer": solution,
            "expected_answer": str(expected) if expected else "",
            "problem_source": "math",
            "level": item.get("level", ""),
        })
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="OpenMathInstruct split: train_1M, train_2M, train_5M, train")
    parser.add_argument("--limit", type=int, default=None, help="Limit training examples (for quick test)")
    parser.add_argument("--skip-test-sets", action="store_true", help="Skip loading GSM8K/MATH test (faster)")
    parser.add_argument("--test-sets-only", action="store_true", help="Load only GSM8K/MATH test (for Phase 9 evaluation)")
    args = parser.parse_args()

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not args.test_sets_only:
        logger.info("Loading OpenMathInstruct-2...")
        train_data = load_openmathinstruct(split=args.split, limit=args.limit)
        logger.info("Loaded %s training examples", len(train_data))

        REAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REAL_DATASET_PATH, "w", encoding="utf-8") as f:
            for ex in train_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved to %s", REAL_DATASET_PATH)

    if args.test_sets_only or not args.skip_test_sets:
        logger.info("Loading GSM8K test...")
        gsm8k = load_gsm8k_test()
        GSM8K_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GSM8K_TEST_PATH, "w", encoding="utf-8") as f:
            for ex in gsm8k:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved %s GSM8K test to %s", len(gsm8k), GSM8K_TEST_PATH)

        logger.info("Loading MATH test...")
        math_test = load_math_test()
        with open(MATH_TEST_PATH, "w", encoding="utf-8") as f:
            for ex in math_test:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved %s MATH test to %s", len(math_test), MATH_TEST_PATH)

    if args.test_sets_only:
        logger.info("Done. Run: USE_DUMMY_DATA=0 python scripts/run_evaluation.py")
    else:
        logger.info("Done. Run preprocess_dpo_data.py with USE_DUMMY_DATA=0 to process.")


if __name__ == "__main__":
    main()
