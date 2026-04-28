#!/usr/bin/env python3
"""
Data preprocessing: 4-way augmentation, complexity classification, DPO pair creation.
Builds all DPO pairs, tokenizes them together, and saves to tokens.pt.
The train/val split happens at training time based on problem_id.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.config import (
    DUMMY_DATASET_PATH,
    DUMMY_PROCESSED_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    DATASET_PATH,
    USE_DUMMY_DATA,
    MODEL_NAME,
    get_tokens_path,
)
from src.data.preprocessing import (
    build_dpo_pairs,
    compute_statistics,
    load_jsonl,
)
from src.evaluation.few_shot_exemplars import build_zero_shot_prompt
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

SEED = 42
MAX_LENGTH = 512


def get_input_path() -> Path:
    if USE_DUMMY_DATA:
        return DUMMY_DATASET_PATH
    return DATASET_PATH


def get_output_path() -> Path:
    if USE_DUMMY_DATA:
        return DUMMY_PROCESSED_DATASET_PATH
    return PROCESSED_DATASET_PATH


def _write_jsonl(path: Path, pairs: list[dict], desc: str = "Saving") -> None:
    pairs_iter = tqdm(pairs, desc=desc, unit=" pairs", file=sys.stdout)
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs_iter:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def tokenize_and_save(
    model_name: str,
    pairs: list[dict],
    output_path: Path,
    max_length: int = MAX_LENGTH,
    batch_size: int = 10_000,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_pairs = len(pairs)
    chosen_input_ids = torch.empty((num_pairs, max_length), dtype=torch.long)
    chosen_attention_mask = torch.empty((num_pairs, max_length), dtype=torch.long)
    rejected_input_ids = torch.empty((num_pairs, max_length), dtype=torch.long)
    rejected_attention_mask = torch.empty((num_pairs, max_length), dtype=torch.long)
    complexities = torch.empty(num_pairs, dtype=torch.long)
    rejection_reason = torch.empty(num_pairs, dtype=torch.long)
    chosen_length = torch.empty(num_pairs, dtype=torch.long)
    rejected_length = torch.empty(num_pairs, dtype=torch.long)
    problem_ids = torch.empty(num_pairs, dtype=torch.long)
    prompt_lengths = torch.empty(num_pairs, dtype=torch.long)

    num_batches = (num_pairs + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Tokenizing batches", unit=" batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_pairs)
        batch_pairs = pairs[start_idx:end_idx]

        chosen_combined, rejected_combined = [], []
        complexities_batch, rejection_reason_batch = [], []
        chosen_length_batch, rejected_length_batch = [], []
        problem_ids_batch = []
        prompt_length_batch = []

        for pair in batch_pairs:
            prompt_text = build_zero_shot_prompt(pair["problem"])
            chosen_combined.append(prompt_text + pair["chosen"])
            rejected_combined.append(prompt_text + pair["rejected"])
            complexities_batch.append(pair.get("complexity", 0))
            rejection_reason_batch.append(pair["rejection_reason"])
            chosen_length_batch.append(pair.get("chosen_length", 0))
            rejected_length_batch.append(pair.get("rejected_length", 0))
            problem_ids_batch.append(pair.get("problem_id", 0))
            # add_special_tokens=False so the count P offsets the BOS shift in the full sequence,
            # making shift_mask[..., :P] zero exactly the right prompt positions.
            prompt_tok = tokenizer(prompt_text, add_special_tokens=False)
            prompt_length_batch.append(len(prompt_tok["input_ids"]))

        chosen_tok = tokenizer(chosen_combined, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        rejected_tok = tokenizer(rejected_combined, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        chosen_input_ids[start_idx:end_idx] = chosen_tok.input_ids
        chosen_attention_mask[start_idx:end_idx] = chosen_tok.attention_mask
        rejected_input_ids[start_idx:end_idx] = rejected_tok.input_ids
        rejected_attention_mask[start_idx:end_idx] = rejected_tok.attention_mask
        complexities[start_idx:end_idx] = torch.tensor(complexities_batch, dtype=torch.long)
        rejection_reason[start_idx:end_idx] = torch.tensor(rejection_reason_batch, dtype=torch.long)
        chosen_length[start_idx:end_idx] = torch.tensor(chosen_length_batch, dtype=torch.long)
        rejected_length[start_idx:end_idx] = torch.tensor(rejected_length_batch, dtype=torch.long)
        problem_ids[start_idx:end_idx] = torch.tensor(problem_ids_batch, dtype=torch.long)
        prompt_lengths[start_idx:end_idx] = torch.tensor(prompt_length_batch, dtype=torch.long)

    torch.save(
        {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "complexities": complexities,
            "rejection_reason": rejection_reason,
            "chosen_length": chosen_length,
            "rejected_length": rejected_length,
            "problem_ids": problem_ids,
            "prompt_lengths": prompt_lengths,
        },
        output_path,
    )

    logger.info(f"      Tokenized a total of {len(problem_ids):,} problems")


def main():
    parser = argparse.ArgumentParser(description="Preprocess DPO data - tokenize all pairs")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if files exist")
    parser.add_argument("--problem-index", type=str, default='data/problem_index.json', help="Path to problem_index.json")
    parser.add_argument("--max-pairs-per-problem", type=int, default=100, help="Maximum number of DPO pairs per problem (stratified by rejection_reason), enter -1 for no limit")
    parser.add_argument("--length-ratio", type=float, default=1.5, help="Minimum length ratio between preferred and rejected solutions, default: 1.5 (1.0 = no filter)")
    args = parser.parse_args()

    set_seed(SEED)
    output_dir = get_output_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = get_tokens_path()
    dataset_path = output_dir / "dataset.jsonl" # TODO - add the path to config.py and update the code accordingly
    meta_path = output_dir / "metadata.json" # TODO - add the path to config.py and update the code accordingly

    if not args or not args.force:
        if tokens_path.exists() and dataset_path.exists() and meta_path.exists():
            logger.info("Processed dataset exists at %s. Use --force to regenerate.", output_dir)
            with open(meta_path) as f:
                stats = json.load(f)
            logger.info("Stats: %s", stats)
            return

    input_path = get_input_path()
    if not input_path.exists():
        if USE_DUMMY_DATA:
            logger.error("Input data not found: %s. Run generate_dummy_data.py first.", input_path)
            raise FileNotFoundError(f"Input data not found: {input_path}")
        logger.error("Input data not found: %s. Run load_real_data.py first.", input_path)
        raise FileNotFoundError(f"Input data not found: {input_path}")

    logger.info("[1/4] Loading input data...")
    raw_data = load_jsonl(input_path)
    logger.info("      Loaded %s examples", f"{len(raw_data):,}")

    logger.info("[2/4] Building DPO pairs (classify, label, group)...")
    problem_index_path = Path(args.problem_index)
    assert problem_index_path.exists(), f'You must first run the load_real_data.py script, to generate the {problem_index_path.name} file'
    pairs = build_dpo_pairs(raw_data, problem_index_path=problem_index_path, max_per_problem=args.max_pairs_per_problem, length_ratio=args.length_ratio)
    logger.info("      Using problem index: %s", problem_index_path)
    logger.info("      Built %s total pairs", f"{len(pairs):,}")

    num_unique_problems = len(set(p.get("problem_id", 0) for p in pairs))
    logger.info(f"      Unique problem IDs: {num_unique_problems:,}")

    logger.info("[3/4] Tokenizing all pairs...")
    tokens_path = tokenize_and_save(MODEL_NAME, pairs, tokens_path)
    logger.info("      Saved to %s", output_dir)

    logger.info("[4/4] Computing and saving statistics...")
    stats = compute_statistics(pairs)
    stats["seed"] = SEED
    stats["total_pairs"] = len(pairs)
    stats["num_unique_problems"] = num_unique_problems
    logger.info("Dataset statistics: %s", stats)

    _write_jsonl(dataset_path, pairs, desc="Saving dataset.jsonl")

    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
