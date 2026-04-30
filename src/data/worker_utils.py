"""
Shared utilities for parallel data loading and processing.
Provides batch processing helpers.
"""

import itertools
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer, PreTrainedTokenizer

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


def _tokenize_pair_batch(
    pairs_batch: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Tokenize a batch of DPO pairs (core tokenization logic)."""
    from src.evaluation.few_shot_exemplars import build_zero_shot_prompt

    chosen_combined, rejected_combined = [], []
    complexities_batch, rejection_reason_batch = [], []
    chosen_length_batch, rejected_length_batch = [], []
    problem_ids_batch, source_batch = [], []
    prompt_texts = []

    for pair in pairs_batch:
        prompt_text = build_zero_shot_prompt(pair["problem"])
        prompt_texts.append(prompt_text)
        chosen_combined.append(prompt_text + pair["chosen"])
        rejected_combined.append(prompt_text + pair["rejected"])
        complexities_batch.append(pair.get("complexity", 0))
        rejection_reason_batch.append(pair["rejection_reason"])
        chosen_length_batch.append(pair.get("chosen_length", 0))
        rejected_length_batch.append(pair.get("rejected_length", 0))
        problem_ids_batch.append(pair.get("problem_id", 0))
        source_batch.append(pair["problem_source"])

    chosen_tok = tokenizer(
        chosen_combined,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    rejected_tok = tokenizer(
        rejected_combined,
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    prompt_lengths = count_tokens_batch(prompt_texts, tokenizer=tokenizer, show_progress=True)

    return {
        "chosen_input_ids": [torch.tensor(enc, dtype=torch.long) for enc in chosen_tok["input_ids"]],
        "rejected_input_ids": [torch.tensor(enc, dtype=torch.long) for enc in rejected_tok["input_ids"]],
        "complexities": torch.tensor(complexities_batch, dtype=torch.long),
        "rejection_reason": torch.tensor(rejection_reason_batch, dtype=torch.long),
        "chosen_length": torch.tensor(chosen_length_batch, dtype=torch.long),
        "rejected_length": torch.tensor(rejected_length_batch, dtype=torch.long),
        "problem_ids": torch.tensor(problem_ids_batch, dtype=torch.long),
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "problem_sources": torch.tensor(source_batch, dtype=torch.long),
    }


def _tokenize_shard(args: tuple) -> dict:
    """Worker function - tokenize a shard of DPO pairs with internal batching.

    Loads tokenizer once, reuses for all batches in this shard.
    """
    shard_idx, pairs_chunk, model_name, max_length, batch_size = args

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_pairs = len(pairs_chunk)
    num_batches = (num_pairs + batch_size - 1) // batch_size

    chosen_input_ids_all: list[torch.Tensor] = []
    rejected_input_ids_all: list[torch.Tensor] = []
    complexities_all: list[torch.Tensor] = []
    rejection_reason_all: list[torch.Tensor] = []
    chosen_length_all: list[torch.Tensor] = []
    rejected_length_all: list[torch.Tensor] = []
    problem_ids_all: list[torch.Tensor] = []
    prompt_lengths_all: list[torch.Tensor] = []
    problem_sources_all: list[torch.Tensor] = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_pairs)
        batch_pairs = pairs_chunk[start:end]

        result = _tokenize_pair_batch(batch_pairs, tokenizer, max_length)

        chosen_input_ids_all.extend(result["chosen_input_ids"])
        rejected_input_ids_all.extend(result["rejected_input_ids"])
        complexities_all.append(result["complexities"])
        rejection_reason_all.append(result["rejection_reason"])
        chosen_length_all.append(result["chosen_length"])
        rejected_length_all.append(result["rejected_length"])
        problem_ids_all.append(result["problem_ids"])
        prompt_lengths_all.append(result["prompt_lengths"])
        problem_sources_all.append(result["problem_sources"])

    return {
        "shard_idx": shard_idx,
        "chosen_input_ids": chosen_input_ids_all,
        "rejected_input_ids": rejected_input_ids_all,
        "complexities": torch.cat(complexities_all),
        "rejection_reason": torch.cat(rejection_reason_all),
        "chosen_length": torch.cat(chosen_length_all),
        "rejected_length": torch.cat(rejected_length_all),
        "problem_ids": torch.cat(problem_ids_all),
        "prompt_lengths": torch.cat(prompt_lengths_all),
        "problem_sources": torch.cat(problem_sources_all),
    }


def tokenize_dpo_pairs_parallel(
    pairs: list[dict],
    model_name: str,
    output_path: Path,
    max_length: int = 512,
    num_workers: int = 32,
    batch_size: int = 10_000,
    show_progress: bool = True,
    pad_token_id: int = 0,
) -> int:
    """Tokenize DPO pairs in parallel using multiprocessing.

    Splits pairs into shards, processes each shard in a separate worker,
    then merges results into a single output file.

    Args:
        pairs: List of DPO pair dictionaries.
        model_name: HuggingFace model name for tokenizer.
        output_path: Path to save tokenized output.
        max_length: Maximum sequence length.
        num_workers: Number of parallel workers.
        batch_size: Batch size for tokenization within each shard.
        show_progress: Show progress bar.

    Returns:
        Total number of tokenized pairs.
    """
    num_pairs = len(pairs)
    shard_size = (num_pairs + num_workers - 1) // num_workers

    shard_args = [
        (shard_idx, pairs[start:start + shard_size], model_name, max_length, batch_size)
        for shard_idx in range(num_workers)
        if (start := shard_idx * shard_size) < num_pairs
    ]

    logger.info(f"Tokenizing {num_pairs:,} pairs across {len(shard_args)} workers...")

    if show_progress:
        results = process_map(
            _tokenize_shard,
            shard_args,
            max_workers=num_workers,
            desc="Tokenizing shards",
        )
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_tokenize_shard, shard_args))

    results.sort(key=lambda x: x["shard_idx"])

    chosen_input_ids = list(itertools.chain.from_iterable(r["chosen_input_ids"] for r in results))
    rejected_input_ids = list(itertools.chain.from_iterable(r["rejected_input_ids"] for r in results))
    complexities = torch.cat([r["complexities"] for r in results])
    rejection_reason = torch.cat([r["rejection_reason"] for r in results])
    chosen_length = torch.cat([r["chosen_length"] for r in results])
    rejected_length = torch.cat([r["rejected_length"] for r in results])
    problem_ids = torch.cat([r["problem_ids"] for r in results])
    prompt_lengths = torch.cat([r["prompt_lengths"] for r in results])
    problem_sources = torch.cat([r["problem_sources"] for r in results])

    torch.save(
        {
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "complexities": complexities,
            "rejection_reason": rejection_reason,
            "chosen_length": chosen_length,
            "rejected_length": rejected_length,
            "problem_ids": problem_ids,
            "prompt_lengths": prompt_lengths,
            "problem_sources": problem_sources,
            "pad_token_id": pad_token_id,
        },
        output_path,
    )

    logger.info(f"Saved {num_pairs:,} tokenized pairs to {output_path}")
    return num_pairs
