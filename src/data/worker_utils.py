"""
Shared utilities for parallel data loading and processing.
Provides worker initialization and batch processing helpers.
"""

import pickle
from typing import Any, Callable

from tqdm.contrib.concurrent import process_map

from src.utils import get_logger, count_tokens


logger = get_logger(__name__)

_tokenizer = None
_problem_to_level = None


def init_worker(tokenizer_path: str | None = None, problem_to_level_path: str | None = None):
    """
    Worker initializer - loads heavy objects once per worker process.
    Call this as the initializer in process_map.
    """
    global _tokenizer, _problem_to_level

    if tokenizer_path:
        from src.utils import _get_model_tokenizer
        _tokenizer = _get_model_tokenizer()

    if problem_to_level_path:
        with open(problem_to_level_path, "rb") as f:
            _problem_to_level = pickle.load(f)


def get_tokenizer():
    """Get the worker-local tokenizer (must call init_worker first)."""
    return _tokenizer


def get_problem_to_level():
    """Get the worker-local problem_to_level dict (must call init_worker first)."""
    return _problem_to_level


def count_tokens_batch(
    texts: list[str],
    tokenizer: Any = None,
    internal_batch_size: int = 10_000,
    show_progress: bool = True,
) -> list[int]:
    """
    Batch tokenize multiple texts efficiently with internal chunking.

    Args:
        texts: List of text strings to tokenize
        tokenizer: Optional tokenizer (uses _tokenizer if None)
        internal_batch_size: Number of texts to process per chunk (default: 1000)
        show_progress: Show progress bar (default: True)

    Returns:
        List of token counts, one per input text
    """
    if not texts:
        return []

    if tokenizer is None:
        tokenizer = _tokenizer
    if tokenizer is None:
        return [count_tokens(t) for t in texts]

    results = []
    num_batches = (len(texts) + internal_batch_size - 1) // internal_batch_size

    iterator = range(0, len(texts), internal_batch_size)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=num_batches, desc=f"Tokenizing {len(texts)} texts", unit="batch")

    for start in iterator:
        end = min(start + internal_batch_size, len(texts))
        chunk = texts[start:end]
        encodings = tokenizer(
            chunk,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        chunk_results = [len(enc) for enc in encodings["input_ids"]]
        results.extend(chunk_results)

    return results


def process_items_parallel(
    items: list[dict],
    convert_fn: Callable,
    max_workers: int = 25,
    batch_size: int = 100,
    desc: str = "Processing",
    initializer: Callable | None = None,
    initargs: tuple | None = None,
) -> list[dict]:
    """
    Generic parallel processing with optional worker initialization.

    Args:
        items: List of items to process
        convert_fn: Function to apply to each item
        max_workers: Number of parallel workers
        batch_size: Chunk size for process_map
        desc: Description for progress bar
        initializer: Worker init function (e.g., init_worker)
        initargs: Arguments to pass to initializer

    Returns:
        List of processed results
    """
    if initializer is not None and initargs is not None:
        results = process_map(
            convert_fn,
            items,
            total=len(items),
            max_workers=max_workers,
            chunksize=batch_size,
            desc=desc,
            initializer=initializer,
            initargs=initargs,
        )
    else:
        results = process_map(
            convert_fn,
            items,
            total=len(items),
            max_workers=max_workers,
            chunksize=batch_size,
            desc=desc,
        )
    return list(results)


def convert_openmath_item_simple(item: dict, problem_to_level: dict | None = None) -> dict:
    """
    Lightweight conversion of OpenMathInstruct item without correctness check.
    Use this when correctness_flag is not needed.
    """
    from src.data.preprocessing import normalize_problem

    if problem_to_level is None:
        problem_to_level = _problem_to_level

    solution = item["generated_solution"]
    expected = item["expected_answer"]
    problem = item["problem"]
    source = item["problem_source"].lower()

    out = {
        "problem": problem,
        "generated_solution": solution,
        "expected_answer": expected,
        "problem_source": source,
        "teacher_token_count": None,
        "correctness_flag": None,
    }

    if problem_to_level and "math" in source and problem:
        out["level"] = problem_to_level.get(normalize_problem(problem), "")
    else:
        out["level"] = ""

    return out