"""
Shared utilities for parallel data loading and processing.
Provides batch processing helpers.
"""

from collections import defaultdict
import itertools
import gc
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import AutoTokenizer, PreTrainedTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import CHOSEN_ENCODINGS_PATH, PROCESSED_PAIRS_INFO_PATH, REJECTED_ENCODINGS_PATH
from src.evaluation.few_shot_exemplars import build_zero_shot_prompt
from src.utils import get_logger, get_model_tokenizer
from src.data.preprocessing import HARD_TOKEN_THRESHOLD, _similarity_index, _normalize_level

logger = get_logger(__name__)


def classify_complexity_batch(
    problems: list[dict],
    batch_size: int = 256,
) -> dict[int, tuple[int, str | None]]:
    """Batch classify complexity for all problems at once."""
    # ... (keeping existing implementation)
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
        aug_results = _similarity_index.find_similar_batch(aug_problems, batch_size=batch_size)
        for i, idx in enumerate(augmented_math_indices):
            complexity, level = aug_results[i]
            if complexity is None:
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
    """Batch tokenize multiple texts efficiently with internal chunking."""
    if not texts:
        return []
    if tokenizer is None:
        tokenizer = get_model_tokenizer()

    results = []
    num_batches = (len(texts) + internal_batch_size - 1) // internal_batch_size
    iterator = range(0, len(texts), internal_batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc=f"Counting Tokens of {len(texts)} texts", unit="batch")
    for start in iterator:
        end = min(start + internal_batch_size, len(texts))
        chunk = texts[start:end]
        encodings = tokenizer(chunk, add_special_tokens=False, truncation=False, return_attention_mask=False)
        chunk_results = [len(enc) for enc in encodings["input_ids"]]
        results.extend(chunk_results)
    return results


def _tokenize_pairs_in_batches(
        pairs: list[dict],
        tokenizer: PreTrainedTokenizer,
        option: Literal["chosen", "rejected"],
        max_length: int = 2048,
        batch_size: int = 20_000,
        show_progress: bool = True,
        description: str = 'Tokenizing'
    ) -> dict[str, torch.Tensor]:
    """Tokenize a list of texts in batches, returning a list of tensors or a dict of lists of tensors."""
    results = defaultdict(list)
    total= len(pairs)

    num_batches = (total + batch_size - 1) // batch_size
    iterator = range(0, total, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc=description, unit="batch")

    for start in iterator:
        batch = pairs[start:start + batch_size]
        if option == "chosen":
            chunk = [build_zero_shot_prompt(p["problem"]) + p["chosen"] for p in batch]
        elif option == "rejected":
            chunk = [build_zero_shot_prompt(p["problem"]) + p["rejected"] for p in batch]
        else:
            raise ValueError(f'Invalid value for option - {option}')

        encodings = tokenizer(chunk, padding=False, truncation=True, max_length=max_length, return_attention_mask=False)

        for input_id in encodings['input_ids']:
            tensor_id = torch.tensor(input_id, dtype=torch.long)
            results['input_ids'].append(tensor_id)
            results['true_lengths'].append(len(tensor_id))

    return {
        'input_ids': pad_sequence(results['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id).to(torch.int32),
        'true_lengths': torch.as_tensor(results['true_lengths'], dtype=torch.int16)
    }


def tokenize_and_save(
        pairs: list[dict],
        tokenizer: PreTrainedTokenizer | None = None,
        max_length: int = 2048,
        batch_size: int = 20_000,
        show_progress: bool = True,
        output_paths: tuple[Path, Path, Path] = (CHOSEN_ENCODINGS_PATH, REJECTED_ENCODINGS_PATH, PROCESSED_PAIRS_INFO_PATH),
        reset: bool = True
    ) -> list[dict]:
    if tokenizer is None:
        tokenizer = get_model_tokenizer()

    chosen_encodings_path, rejected_encodings_path, pairs_info_path = output_paths
    num_pairs = len(pairs)
    num_batches = (num_pairs + batch_size - 1) // batch_size

    if reset or not pairs_info_path.exists():
        # Collect pairs info and prompt lengths in batches
        pairs_info = defaultdict(list)
        iterator = range(0, num_pairs, batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Collecting pairs info", unit="batch")
        for start in iterator:
            batch = pairs[start:start + batch_size]
            for p in batch:
                pairs_info["complexity"].append(p.pop("complexity"))
                pairs_info["rejection_reason"].append(p.pop("rejection_reason"))
                pairs_info["chosen_length"].append(p.pop("chosen_length"))
                pairs_info["rejected_length"].append(p.pop("rejected_length"))
                pairs_info["problem_id"].append(p.pop("problem_id"))
                pairs_info["problem_source"].append(p.pop("problem_source"))

        prompt_texts = [build_zero_shot_prompt(p["problem"]) for p in pairs]
        prompt_lengths = count_tokens_batch(prompt_texts, tokenizer, internal_batch_size=batch_size, show_progress=False)
        pairs_info["prompt_length"] = prompt_lengths

        prompt_lengths_tensor = torch.as_tensor(prompt_lengths, dtype=torch.int16)
        chosen_length_tensor = torch.as_tensor(pairs_info["chosen_length"], dtype=torch.int16)
        rejected_length_tensor = torch.as_tensor(pairs_info["rejected_length"], dtype=torch.int16)
        max_full_prompt_length = prompt_lengths_tensor + torch.maximum(chosen_length_tensor, rejected_length_tensor)
        mask_under_max_length = max_full_prompt_length < max_length

        pairs_info = {key: torch.tensor(value, dtype=torch.long)[mask_under_max_length] for key, value in pairs_info.items()}
        torch.save(pairs_info, pairs_info_path)
        del prompt_texts, prompt_lengths, pairs_info
        gc.collect()
        logger.info("Saved pairs info to %s", pairs_info_path)

    under_max_length_indices = torch.where(mask_under_max_length)[0]
    relevant_pairs = [pairs[i] for i in tqdm(under_max_length_indices, desc="Filtering pairs under max length", unit="pair")]
    logger.info(f"Filtered out {len(pairs) - len(relevant_pairs):,} pairs surpassing max length {max_length}")
    del pairs
    gc.collect()

    if reset or not chosen_encodings_path.exists():
        # Tokenize chosen prompts in batches
        chosen_encodings = _tokenize_pairs_in_batches(relevant_pairs, tokenizer, option="chosen", max_length=max_length, batch_size=batch_size, show_progress=show_progress, description="Tokenizing chosen prompts")
        iterator = range(0, num_pairs, batch_size)
        torch.save(chosen_encodings, chosen_encodings_path)
        # Remove tokenized texts from pairs to free memory
        for p in relevant_pairs:
            p.pop("chosen")
        del chosen_encodings
        gc.collect()
        logger.info("Saved tokenized chosen prompts to %s", chosen_encodings_path)

    if reset or not rejected_encodings_path.exists():
        # Tokenize rejected prompts in batches
        rejected_encodings = _tokenize_pairs_in_batches(relevant_pairs, tokenizer, option="rejected", max_length=max_length, batch_size=batch_size, show_progress=show_progress, description="Tokenizing rejected prompts")
        torch.save(rejected_encodings, rejected_encodings_path)
        # Remove tokenized texts from pairs to free memory
        for p in relevant_pairs:
            p.pop("rejected")
        del rejected_encodings
        gc.collect()
        logger.info("Saved tokenized rejected prompts to %s", rejected_encodings_path)

    logger.info(f"Tokenized a total of {num_pairs:,} pairs")

    return relevant_pairs
