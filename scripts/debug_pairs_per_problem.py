#!/usr/bin/env python3
"""Debug: find 20 problems with >1000 pairs, dump problem text + 5 sample pairs each."""

import json
import random
from collections import defaultdict
from pathlib import Path

DATASET = Path("data/processed_dpo_dataset/dataset.jsonl")
OUTPUT = Path("data/processed_dpo_dataset/debug_high_pair_problems.json")
MIN_PAIRS = 1000
NUM_PROBLEMS = 20
SAMPLES_PER_PROBLEM = 5

random.seed(42)


def main():
    # Pass 1: count pairs per problem_id, keep reservoir of samples
    counts: dict[int, int] = defaultdict(int)
    problem_text: dict[int, str] = {}
    # Reservoir sample: keep up to SAMPLES_PER_PROBLEM pairs per problem
    samples: dict[int, list[dict]] = defaultdict(list)

    print(f"Streaming {DATASET} ...")
    with open(DATASET, "r") as f:
        for line_no, line in enumerate(f, 1):
            if line_no % 5_000_000 == 0:
                print(f"  ...{line_no:,} lines")
            row = json.loads(line)
            pid = row["problem_id"]
            counts[pid] += 1

            if pid not in problem_text:
                problem_text[pid] = row["problem"]

            # Reservoir sampling to keep memory bounded
            if len(samples[pid]) < SAMPLES_PER_PROBLEM:
                samples[pid].append({"chosen": row["chosen"], "rejected": row["rejected"]})
            else:
                j = random.randint(0, counts[pid] - 1)
                if j < SAMPLES_PER_PROBLEM:
                    samples[pid][j] = {"chosen": row["chosen"], "rejected": row["rejected"]}

    # Find problems with >MIN_PAIRS, sort descending, take top NUM_PROBLEMS
    heavy = sorted(
        ((pid, cnt) for pid, cnt in counts.items() if cnt > MIN_PAIRS),
        key=lambda x: -x[1],
    )[:NUM_PROBLEMS]

    print(f"\nFound {len(heavy)} problems with >{MIN_PAIRS} pairs (showing top {NUM_PROBLEMS})")

    results = []
    for pid, cnt in heavy:
        results.append({
            "problem_id": pid,
            "num_pairs": cnt,
            "problem_text": problem_text[pid],
            "sample_pairs": samples[pid],
        })
        print(f"  problem_id={pid}  pairs={cnt:,}  problem_len={len(problem_text[pid])}")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
