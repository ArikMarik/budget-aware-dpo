#!/usr/bin/env python3
"""
Enrich existing real_openmathinstruct.jsonl with MATH levels.
Loads MATH train split, builds problem->level map, adds level to each row where problem_source has math.
Use when you have existing JSONL and don't want to re-download from HuggingFace.
"""

import argparse
import json
import sys
from pathlib import Path

from src.config import REAL_DATASET_PATH
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

set_seed(42)

MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


def load_math_problem_to_level() -> dict[str, str]:
    """Load MATH train and build problem -> level mapping."""
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
            mapping[problem] = str(level)
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REAL_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Default: overwrite input")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        sys.exit(1)

    output_path = args.output or args.input
    if output_path == args.input and args.limit:
        logger.error("ERROR: Cannot overwrite input when --limit is used. Specify --output.")
        sys.exit(1)

    logger.info("Loading MATH train for level mapping...")
    problem_to_level = load_math_problem_to_level()
    logger.info("Built level map for %s MATH problems", f"{len(problem_to_level):,}")

    if output_path == args.input:
        tmp_path = args.input.with_suffix(".jsonl.tmp")
        out_f = open(tmp_path, "w", encoding="utf-8")
    else:
        out_f = open(output_path, "w", encoding="utf-8")

    matched = 0
    total_math = 0
    n = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            problem = item.get("problem", "")
            source = str(item.get("problem_source", "")).lower()
            if "math" in source:
                total_math += 1
                level = problem_to_level.get(problem, "")
                item["level"] = level
                if level:
                    matched += 1
            else:
                item["level"] = ""
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
            if (n + 1) % 100000 == 0:
                logger.info("  Processed %s lines...", f"{n + 1:,}")

    out_f.close()
    if tmp_path is not None:
        tmp_path.replace(args.input)
    logger.info("Done. Processed %s examples. MATH items: %s, with level: %s", f"{n:,}", f"{total_math:,}", f"{matched:,}")


if __name__ == "__main__":
    main()
