#!/usr/bin/env python3
"""
Load real datasets (OpenMathInstruct-2, GSM8K test, MATH test) and convert to JSONL.
Saves training data for DPO preprocessing; holds out test sets for Phase 9 evaluation.
"""

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

from src.config import DATA_PATH, GSM8K_TEST_PATH, MATH_TEST_PATH, DATASET_PATH
from src.data.preprocessing import classify_complexity
from src.evaluation.answer_extraction import extract_answer, extract_gsm8k_answer, verify_correctness
from src.utils import count_tokens, get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

set_seed(42)

MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]

OPENMATH_SIZES = {
    "train_1M": 1_000_000,
    "train_2M": 2_000_000,
    "train_5M": 5_000_000,
    "train": 14_000_000,
}


def normalize_problem(text: str) -> str:
    """Normalize problem text for matching: collapse whitespace, strip. Improves level lookup for MATH-origin problems."""
    if not text:
        return ""
    return " ".join(str(text).split())


def convert_openmath_instruct(item: dict, problem_to_level: dict | None = None) -> dict:
    """Convert OpenMathInstruct-2 item to our format. Add level when problem_source has math and problem matches MATH train."""
    solution = item["generated_solution"]
    expected = item["expected_answer"]
    problem = item["problem"]
    source = item["problem_source"].lower()
    out = {
        "problem": problem,
        "generated_solution": solution,
        "expected_answer": expected,
        "problem_source": source,
        "teacher_token_count": count_tokens(solution),
        "correctness_flag": verify_correctness(solution, expected, problem=problem),
    }
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
    except Exception as e:
        logger.warning("Failed to load EleutherAI/hendrycks_math: %s. Trying fallback...", e)
        try:
            ds = load_dataset("hendrycks/competition_math", split="train")
        except Exception as e2:
            logger.warning("Failed to load hendrycks/competition_math: %s. Using final fallback...", e2)
            ds = load_dataset("lighteval/MATH", split="train")
    mapping = {}
    for item in ds:
        problem = item.get("problem", item.get("question", ""))
        level = item.get("level", "")
        if problem and level:
            mapping[normalize_problem(problem)] = str(level)
    return mapping


def load_openmath_instruct(split: str = "train_1M", limit: int | None = None) -> list[dict]:
    """Load OpenMathInstruct-2 from HuggingFace. Enriches MATH-origin problems with level from MATH train."""
    from datasets import load_dataset

    logger.info("Loading MATH train for level mapping...")
    problem_to_level = load_math_problem_to_level()
    logger.info("Built level map for %s MATH problems", f"{len(problem_to_level):,}")

    dataset = load_dataset("nvidia/OpenMathInstruct-2", split=split, streaming=True)
    total = min(limit, OPENMATH_SIZES[split]) if limit else OPENMATH_SIZES[split]
    examples = []
    for i, item in enumerate(tqdm(dataset, total=total, desc="Loading OpenMathInstruct-2")):
        if limit and i >= limit:
            break
        examples.append(convert_openmath_instruct(dict(item), problem_to_level))
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
    except Exception as e:
        logger.warning("Failed to load EleutherAI/hendrycks_math test: %s. Trying fallback...", e)
        try:
            ds = load_dataset("hendrycks/competition_math", split="test")
        except Exception as e2:
            logger.warning("Failed to load hendrycks/competition_math test: %s. Using final fallback...", e2)
            ds = load_dataset("lighteval/MATH", split="test")
    examples = []
    for item in ds:
        problem = item.get("problem", item.get("question", ""))
        solution = item.get("solution", item.get("answer", ""))
        expected = item.get("answer", "")
        if not expected and solution and "\\boxed{" in str(solution):
            expected = extract_answer(str(solution)) or ""
        examples.append({
            "problem": problem,
            "answer": solution,
            "expected_answer": str(expected) if expected else "",
            "problem_source": "math",
            "level": item.get("level", ""),
        })
    return examples


SOURCE_PREFERENCE = {"math": 0, "augmented_math": 1, "gsm8k": 2, "augmented_gsm8k": 3}

def get_source_rank(source: str) -> int:
    """Lower is better. Returns high default for unknown sources."""
    return SOURCE_PREFERENCE.get(source.lower(), 999)


def build_problem_index(raw_data: list[dict]) -> list[dict]:
    """Build a problem index by grouping solutions by normalized problem text.
    
    For each unique problem:
    - Assign a unique integer problem_id
    - Collect all solution token lengths from all sources
    - Compute average token length
    - Select data by source preference (math > augmented_math > gsm8k > augmented_gsm8k)
    - Classify complexity using the average token length
    - Copy level from similar MATH problem if available
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in raw_data:
        norm_problem = normalize_problem(ex.get("problem", ""))
        if norm_problem:
            groups[norm_problem].append(ex)
    
    result = []
    for problem_id, examples in enumerate(tqdm(groups.values(), desc="Building problem index", unit=" problems")):
        token_lengths = [ex.get("teacher_token_count", 0) for ex in examples]
        avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        
        examples_sorted = sorted(examples, key=lambda ex: get_source_rank(ex.get("problem_source", "")))
        primary = examples_sorted[0]
        
        complexity, matched_level = classify_complexity(primary, avg_token_length=avg_tokens)
        
        level = matched_level or primary.get("level", "")
        
        result.append({
            "problem_id": problem_id,
            "problem": primary.get("problem", ""),
            "problem_source": primary.get("problem_source", ""),
            "level": level,
            "token_lengths": token_lengths,
            "avg_token_length": avg_tokens,
            "complexity": complexity,
        })
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="OpenMathInstruct split: train_1M, train_2M, train_5M, train")
    parser.add_argument("--limit", type=int, default=None, help="Limit training examples (for quick test)")
    parser.add_argument("--skip-test-sets", action="store_true", help="Skip loading GSM8K/MATH test (faster)")
    parser.add_argument("--test-sets-only", action="store_true", help="Load only GSM8K/MATH test (for Phase 9 evaluation)")
    parser.add_argument("--no-problem-index", action="store_true", help="Skip building problem index JSON")
    args = parser.parse_args()

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not args.test_sets_only:
        logger.info("Loading OpenMathInstruct-2...")
        train_data = load_openmath_instruct(split=args.split, limit=args.limit)
        logger.info("Loaded %s training examples", len(train_data))

        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            for ex in train_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        logger.info("Saved to %s", DATASET_PATH)

        if not args.no_problem_index:
            logger.info("Building problem index...")
            problem_index = build_problem_index(train_data)
            logger.info("Built index for %s unique problems", len(problem_index))
            
            DATA_PATH.mkdir(parents=True, exist_ok=True)
            problem_index_path = DATA_PATH / "problem_index.json"
            with open(problem_index_path, "w", encoding="utf-8") as f:
                json.dump(problem_index, f, ensure_ascii=False)
            logger.info("Saved problem index to %s", problem_index_path)

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
