"""
Shared utilities for parallel data loading and processing.
Provides batch processing helpers.
"""

import os
from typing import Any

import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import PreTrainedTokenizer

from src.utils import get_logger, count_tokens, _get_model_tokenizer
from src.data.preprocessing import HARD_TOKEN_THRESHOLD, _similarity_index, _normalize_level


logger = get_logger(__name__)


def classify_complexity_batch(
    problems: list[dict],
    batch_size: int = 256,
) -> dict[int, tuple[int, str | None]]:
    """Batch classify complexity for all problems at once.

    Args:
        problems: List of primary problems (one per unique problem).
            Each problem should contain: problem, problem_source, level (optional),
            teacher_token_count or _avg_token_length.
        device: Device to use ('cuda', 'cpu', or None for auto-detect).
        batch_size: Batch size for encoding augmented_math problems.

    Returns:
        Dict mapping problem_index -> (complexity, matched_level)
    """
    results = {}

    gsm_indices = []
    math_indices = []
    augmented_math_indices = []
    unknown_indices = []

    for idx, p in enumerate(problems):
        source = str(p.get("problem_source", "")).lower()

        if "gsm" in source:
            gsm_indices.append(idx)
        elif source == "math":
            math_indices.append(idx)
        elif source == "augmented_math" and p.get("problem"):
            augmented_math_indices.append(idx)
        else:
            unknown_indices.append(idx)

    for idx in gsm_indices:
        results[idx] = (0, None)

    for idx in math_indices:
        p = problems[idx]
        level = _normalize_level(p.get("level"))
        if level and level > 1:
            results[idx] = (1, level)
        else:
            results[idx] = (0, level)

    if augmented_math_indices:
        aug_problems = [problems[idx].get("problem", "") for idx in augmented_math_indices]

        # Use centralized batch similarity search
        aug_results = _similarity_index.find_similar_batch(aug_problems, batch_size=batch_size)

        for i, idx in enumerate(augmented_math_indices):
            complexity, level = aug_results[i]
            if complexity is None:
                # Fallback to token-based
                unknown_indices.append(idx)
                continue

            results[idx] = (complexity, level)

    for idx in unknown_indices:
        p = problems[idx]
        tokens = p.get("_avg_token_length", p.get("teacher_token_count", 0))
        if tokens > HARD_TOKEN_THRESHOLD:
            results[idx] = (1, None)
        else:
            results[idx] = (0, None)

    return results


def count_tokens_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizer | None = None,
    internal_batch_size: int = 20_000,
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
        tokenizer = _get_model_tokenizer()
    if tokenizer is None:
        return [count_tokens(t) for t in tqdm(texts, desc="Tokenizing texts (one by one)")]

    results = []
    num_batches = (len(texts) + internal_batch_size - 1) // internal_batch_size

    iterator = range(0, len(texts), internal_batch_size)
    if show_progress:
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