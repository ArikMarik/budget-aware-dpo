#!/usr/bin/env python3
"""
Load real datasets (OpenMathInstruct-2, GSM8K test, MATH test) and convert to JSONL.
Saves training data for DPO preprocessing; holds out test sets for Phase 9 evaluation.
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


from src.config import DATA_PATH, GSM8K_TEST_PATH, INDEX_TO_PROBLEM_PATH, MATH_TEST_PATH, DATASET_PATH, PROBLEM_TO_INDEX_PATH, PROBLEM_TO_LEVEL_PATH
from src.data.preprocessing import load_jsonl, load_math_problem_to_level, normalize_problem
from src.data.worker_utils import count_tokens_batch, classify_complexity_batch
from src.evaluation.answer_extraction import extract_answer, extract_gsm8k_answer
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

set_seed(42)

OPENMATH_SIZES = {
    "train_1M": 1_000_000,
    "train_2M": 2_000_000,
    "train_5M": 5_000_000,
    "train": 13_972_791,
}

BATCH_SIZE = 500
NUM_WORKERS = 8


def convert_openmath_instruct_item(item: dict, problem_to_level: dict | None = None, compute_correctness: bool = True) -> dict:
    """Convert OpenMathInstruct-2 item to our format. Add level when problem_source has math and problem matches MATH train."""
    from src.data.preprocessing import normalize_problem

    solution = item["generated_solution"]
    expected = item["expected_answer"]
    problem = item["problem"]
    source = item["problem_source"].lower()

    teacher_token_count = item.get("_token_count")

    is_correct = None
    if compute_correctness:
        from src.evaluation.answer_extraction import verify_correctness
        is_correct = verify_correctness(solution, expected)

    out = {
        "problem": problem,
        "generated_solution": solution,
        "expected_answer": expected,
        "problem_source": source,
        "teacher_token_count": teacher_token_count,
        "correctness_flag": is_correct,
    }
    if problem_to_level and "math" in source and problem:
        out["level"] = problem_to_level.get(normalize_problem(problem), "")
    else:
        out["level"] = ""
    return out


def _worker_convert(item: dict) -> dict:
    """Worker function that uses pre-loaded problem_to_level."""
    return convert_openmath_instruct_item(
        item,
        problem_to_level=_problem_to_level_cache,
        compute_correctness=_compute_correctness,
    )


_problem_to_level_cache: dict | None = None
_compute_correctness: bool = False


def load_openmath_instruct(split: str = "train_1M", limit: int | None = None, compute_correctness: bool = True) -> list[dict]:
    """Load OpenMathInstruct-2 from HuggingFace. Enriches MATH-origin problems with level from MATH train."""
    global _problem_to_level_cache, _compute_correctness

    import datasets
    from datasets import load_dataset
    cache_path = '/root/.cache/huggingface/datasets'
    datasets.config.HF_DATASETS_CACHE = cache_path

    dataset = load_dataset("nvidia/OpenMathInstruct-2", split=split, streaming=True, cache_dir=cache_path)
    total = min(limit, OPENMATH_SIZES[split]) if limit else OPENMATH_SIZES[split]

    logger.info("Loading MATH train for level mapping...")
    if PROBLEM_TO_LEVEL_PATH.exists():
        with open(PROBLEM_TO_LEVEL_PATH, "rb") as f:
            problem_to_level = pickle.load(f)
        logger.info("Loaded problem to level mapping from %s", PROBLEM_TO_LEVEL_PATH)
    else:
        problem_to_level = load_math_problem_to_level()
        with open(PROBLEM_TO_LEVEL_PATH, "wb") as f:
            pickle.dump(problem_to_level, f)
        logger.info("Built level map for %s MATH problems and saved to %s", f"{len(problem_to_level):,}", PROBLEM_TO_LEVEL_PATH)

    _problem_to_level_cache = problem_to_level
    _compute_correctness = compute_correctness

    items: list[dict] = []
    for i, item in enumerate(tqdm(dataset, total=total, desc="Loading OpenMathInstruct-2")):
        if limit and i >= limit:
            break
        items.append(dict(item))

    if not items:
        return []

    logger.info("Batch tokenizing %d solutions...", len(items))
    solutions = [item["generated_solution"] for item in items]
    token_counts = count_tokens_batch(solutions)
    del solutions
    for i, tc in enumerate(token_counts):
        items[i]["_token_count"] = tc
    logger.info("Tokenization complete.")

    logger.info("Sorting items by expected_answer for improved caching...")
    items.sort(key=lambda x: x.get("expected_answer", ""))

    results = process_map(
        _worker_convert,
        items,
        total=len(items),
        max_workers=NUM_WORKERS,
        chunksize=BATCH_SIZE,
        desc="Processing OpenMathInstruct-2 items",
    )

    return list(results)


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
            expected = extract_answer(str(solution))
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
    - Classify complexity using batched classification for efficiency
    - Copy level from similar MATH problem if available
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in tqdm(raw_data, desc="Grouping by normalized problem", unit=" examples"):
        norm_problem = normalize_problem(ex.get("problem", ""))
        if norm_problem:
            groups[norm_problem].append(ex)

    primary_examples = []
    problem_groups_list = []

    for examples in tqdm(groups.values(), desc="Initial Processing of problem groups", unit=" groups"):
        token_lengths = [ex.get("teacher_token_count", 0) for ex in examples]
        avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0

        examples_sorted = sorted(
            examples,
            key=lambda ex: (get_source_rank(ex.get("problem_source", "")),
                            ex.get("level").lower() if ex.get("level") else "level unknown")
        )
        primary = examples_sorted[0].copy()
        primary["_avg_token_length"] = avg_tokens
        primary_examples.append(primary)
        problem_groups_list.append((token_lengths, primary))

    complexity_results = classify_complexity_batch(primary_examples)

    result = []
    for problem_id, (complexity, matched_level) in tqdm(complexity_results.items(), desc="Building problem index", unit=" problems"):
        token_lengths, primary = problem_groups_list[problem_id]

        problem = primary.get("problem", "")
        avg_tokens = primary.get("_avg_token_length", 0)

        level = primary.get("level") if primary.get("level") else matched_level

        result.append({
            "problem_id": problem_id,
            "problem": problem,
            "normalized_problem": normalize_problem(problem),
            "problem_source": primary.get("problem_source", ""),
            "level": level,
            "token_lengths": token_lengths,
            "avg_token_length": avg_tokens,
            "complexity": complexity,
            "expected_answer": primary.get("expected_answer", ""),
        })

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force regeneration even if files exist")
    parser.add_argument("--split", default="train", help="OpenMathInstruct split: train_1M, train_2M, train_5M, train")
    parser.add_argument("--limit", type=int, default=None, help="Limit training examples (for quick test)")
    parser.add_argument("--output", type=str, default=DATASET_PATH, help="Output path for training data JSONL (default: DATASET_PATH)")
    parser.add_argument("--skip-test-sets", action="store_true", help="Skip loading GSM8K/MATH test (faster)")
    parser.add_argument("--test-sets-only", action="store_true", help="Load only GSM8K/MATH test (for Phase 9 evaluation)")
    parser.add_argument("--no-problem-index", action="store_true", help="Skip building problem index JSON")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip computing correctness flag (faster; useful for quick loading)")
    args = parser.parse_args()

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    if not args.test_sets_only:
        if args.output != DATASET_PATH:
            output_path = Path(args.output)
        elif args.split == "train" and args.limit is None:
            output_path = DATASET_PATH
        else:
            base_name = f"openmathinstruct_{args.split}"
            if args.limit:
                base_name += f"_limit_{args.limit}"
            output_path = DATA_PATH / f"{base_name}.jsonl"

        train_data = None
        if not output_path.exists() or args.force:
            logger.info("Loading OpenMathInstruct-2...")
            train_data = load_openmath_instruct(split=args.split, limit=args.limit, compute_correctness=not args.skip_correctness)
            logger.info("Loaded %s training examples", len(train_data))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for ex in train_data:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            logger.info("Saved to %s", output_path)

        if not args.no_problem_index:
            if not (PROBLEM_TO_INDEX_PATH.exists() and INDEX_TO_PROBLEM_PATH.exists()) or args.force:
                logger.info("Building problem index...")
                train_data = train_data if train_data is not None else load_jsonl(output_path)

                problem_index = build_problem_index(train_data)
                logger.info("Built index for %s unique problems", len(problem_index))

                problem_to_index_dict, index_to_problem_dict = {}, {}
                for item in problem_index:
                    # keep the original problem structure (not normalized) in the index, but key by normalized problem for lookup
                    normalized_problem = item.pop("normalized_problem")
                    problem_to_index_dict[normalized_problem] = item
                    index_to_problem_dict[item["problem_id"]] = item

                with open(PROBLEM_TO_INDEX_PATH, "wb") as f:
                    pickle.dump(problem_to_index_dict, f)
                logger.info("Saved problem index (Dict) to %s", PROBLEM_TO_INDEX_PATH)

                with open(INDEX_TO_PROBLEM_PATH, "wb") as f:
                    pickle.dump(index_to_problem_dict, f)
                logger.info("Saved reverse index to %s", INDEX_TO_PROBLEM_PATH)

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
