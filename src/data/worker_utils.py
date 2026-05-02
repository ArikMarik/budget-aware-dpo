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


def _tokenize_pairs_in_batches(pairs: list[dict], tokenizer: PreTrainedTokenizer, option: Literal["chosen", "rejected", "base"], max_length: int = 2048, batch_size: int = 20_000, padding: bool = False, show_progress: bool = True, description: str = 'Tokenizing') -> list[torch.Tensor] | dict[str, list[torch.Tensor]]:
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
            chunk = [build_zero_shot_prompt(p["problem"]) for p in batch]
        encodings = tokenizer(chunk, padding='max_length' if padding else 'do_not_pad', truncation=True, max_length=max_length, return_attention_mask=True if padding else False)

        for key, value in encodings.items():
            results[key].extend([torch.tensor(v, dtype=torch.long) for v in value])

    return results


def tokenize_and_save(
        pairs: list[dict],
        tokenizer: PreTrainedTokenizer | None = None,
        max_length: int = 2048,
        batch_size: int = 20_000,
        padding: bool = False,
        show_progress: bool = True,
        output_paths: tuple[Path, Path, Path] = (CHOSEN_ENCODINGS_PATH, REJECTED_ENCODINGS_PATH, PROCESSED_PAIRS_INFO_PATH),
        reset: bool = True
    ) -> None:
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

        pairs_info = {key: torch.tensor(value, dtype=torch.long) for key, value in pairs_info.items()}
        torch.save(pairs_info, pairs_info_path)
        del prompt_texts, prompt_lengths, pairs_info
        gc.collect()
        logger.info("Saved pairs info to %s", pairs_info_path)

    if reset or not chosen_encodings_path.exists():
        # Tokenize chosen prompts in batches
        chosen_encodings = _tokenize_pairs_in_batches(pairs, tokenizer, option="chosen", max_length=max_length, batch_size=batch_size, padding=padding, show_progress=show_progress, description="Tokenizing chosen prompts")
        iterator = range(0, num_pairs, batch_size)
        torch.save(chosen_encodings, chosen_encodings_path)
        # Remove tokenized texts from pairs to free memory
        for p in pairs:
            p.pop("chosen")
        del chosen_encodings
        gc.collect()
        logger.info("Saved tokenized chosen prompts to %s", chosen_encodings_path)

    if reset or not rejected_encodings_path.exists():
        # Tokenize rejected prompts in batches
        rejected_encodings = _tokenize_pairs_in_batches(pairs, tokenizer, option="rejected", max_length=max_length, batch_size=batch_size, padding=padding, show_progress=show_progress, description="Tokenizing rejected prompts")
        torch.save(rejected_encodings, rejected_encodings_path)
        # Remove tokenized texts from pairs to free memory
        for p in pairs:
            p.pop("rejected")
        del rejected_encodings
        gc.collect()
        logger.info("Saved tokenized rejected prompts to %s", rejected_encodings_path)

    logger.info(f"Tokenized a total of {num_pairs:,} pairs")


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

    chosen_tok = tokenizer(chosen_combined, padding=False, truncation=True, max_length=max_length)
    rejected_tok = tokenizer(rejected_combined, padding=False, truncation=True, max_length=max_length)
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
    """Worker function - tokenize a shard of DPO pairs with internal batching."""
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

    # OPTION A: write shard result to a temp file and return only the path.
    # Returning 100K individual PyTorch tensors through the multiprocessing pipe
    # requires pickling ~1.2GB of Python objects per shard, which serializes through
    # the _result_handler thread and takes 30-90 min for 32 shards. torch.save()
    # writes the same data in seconds via its optimized binary format, and the IPC
    # payload shrinks from ~1.2GB to a ~50-byte file path string.
    # See: docs/ipc_bug_analysis.md
    shard_data = {
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
    tmp_path = os.path.join(tempfile.gettempdir(), f"dpo_shard_{shard_idx}_{os.getpid()}.pt")
    torch.save(shard_data, tmp_path)
    return {"shard_idx": shard_idx, "tmp_path": tmp_path}

    # ORIGINAL RETURN (Option A replacement — caused IPC deadlock):
    # Returning the full dict through multiprocessing pipe pickles 200K individual
    # PyTorch tensors (~1.2GB) which serializes through a single _result_handler
    # thread. With 32 workers each queuing ~1.2GB simultaneously, effective
    # throughput drops to ~10MB/s (pipe buffer contention + per-tensor pickle
    # overhead), making the full pipeline take 30-90 min on the IPC step alone.
    # return {
    #     "shard_idx": shard_idx,
    #     "chosen_input_ids": chosen_input_ids_all,
    #     "rejected_input_ids": rejected_input_ids_all,
    #     "complexities": torch.cat(complexities_all),
    #     "rejection_reason": torch.cat(rejection_reason_all),
    #     "chosen_length": torch.cat(chosen_length_all),
    #     "rejected_length": torch.cat(rejected_length_all),
    #     "problem_ids": torch.cat(problem_ids_all),
    #     "prompt_lengths": torch.cat(prompt_lengths_all),
    #     "problem_sources": torch.cat(problem_sources_all),
    # }


def tokenize_dpo_pairs_parallel(
    pairs: list[dict],
    model_name: str,
    output_path: Path,
    max_length: int = 512,
    num_workers: int = 32,
    batch_size: int = 10_000,
    show_progress: bool = True,
    pad_token_id: int = 0,
    batches_per_shard: int | None = None,
) -> int:
    """Tokenize DPO pairs in parallel using multiprocessing with incremental processing."""
    num_pairs = len(pairs)

    if batches_per_shard is None:
        batches_per_shard = max(1, 100000 // batch_size)
    shard_size = batch_size * batches_per_shard
    num_shards = (num_pairs + shard_size - 1) // shard_size

    shard_args = [
        (shard_idx, pairs[start:start + shard_size], model_name, max_length, batch_size)
        for shard_idx in range(num_shards)
        if (start := shard_idx * shard_size) < num_pairs
    ]

    logger.info(f"Tokenizing {num_pairs:,} pairs across {num_shards} shards (batch_size={batch_size}, batches_per_shard={batches_per_shard}) using {num_workers} workers...")

    all_chosen_input_ids = []
    all_rejected_input_ids = []
    all_complexities = []
    all_rejection_reason = []
    all_chosen_length = []
    all_rejected_length = []
    all_problem_ids = []
    all_prompt_lengths = []
    all_problem_sources = []
    buffered_results = {}
    next_shard_idx = 0

    def _merge_shard(result: dict) -> None:
        all_chosen_input_ids.extend(result["chosen_input_ids"])
        all_rejected_input_ids.extend(result["rejected_input_ids"])
        all_complexities.append(result["complexities"])
        all_rejection_reason.append(result["rejection_reason"])
        all_chosen_length.append(result["chosen_length"])
        all_rejected_length.append(result["rejected_length"])
        all_problem_ids.append(result["problem_ids"])
        all_prompt_lengths.append(result["prompt_lengths"])
        all_problem_sources.append(result["problem_sources"])

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_tokenize_shard, args): args for args in shard_args}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Tokenizing shards")
        for future in iterator:
            # OPTION A: load shard data from temp file written by the worker.
            # The worker returns only a file path; actual tensor data is on disk.
            raw = future.result()
            tmp_path = raw.get("tmp_path")
            if tmp_path:
                result = torch.load(tmp_path, map_location="cpu", weights_only=False)
                os.unlink(tmp_path)
            else:
                # ORIGINAL fallback: result arrived through the pipe directly.
                # This path is now dead code (workers always write temp files).
                result = raw
            shard_idx = result["shard_idx"]
            if shard_idx == next_shard_idx:
                _merge_shard(result)
                next_shard_idx += 1
                while next_shard_idx in buffered_results:
                    _merge_shard(buffered_results.pop(next_shard_idx))
                    next_shard_idx += 1
            else:
                buffered_results[shard_idx] = result

    logger.info(f"Merging {num_shards} shards into final tensors...")
    final_chosen = list(itertools.chain.from_iterable(all_chosen_input_ids))
    final_rejected = list(itertools.chain.from_iterable(all_rejected_input_ids))
    final_complexities = torch.cat(all_complexities)
    final_rejection_reason = torch.cat(all_rejection_reason)
    final_chosen_length = torch.cat(all_chosen_length)
    final_rejected_length = torch.cat(all_rejected_length)
    final_problem_ids = torch.cat(all_problem_ids)
    final_prompt_lengths = torch.cat(all_prompt_lengths)
    final_problem_sources = torch.cat(all_problem_sources)

    torch.save(
        {
            "chosen_input_ids": final_chosen,
            "rejected_input_ids": final_rejected,
            "complexities": final_complexities,
            "rejection_reason": final_rejection_reason,
            "chosen_length": final_chosen_length,
            "rejected_length": final_rejected_length,
            "problem_ids": final_problem_ids,
            "prompt_lengths": final_prompt_lengths,
            "problem_sources": final_problem_sources,
            "pad_token_id": pad_token_id,
        },
        output_path,
    )
    logger.info(f"Saved {num_pairs:,} tokenized pairs to {output_path}")
    return num_pairs
