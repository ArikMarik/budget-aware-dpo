#!/usr/bin/env python3
"""Analyze the balanced DPO dataset: rejection reasons, token length distributions per complexity."""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

DATA_PATH = Path("data/processed_dpo_dataset_balanced/dataset.jsonl")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    records = []
    with open(DATA_PATH) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Total samples: {len(records)}\n")

    # Separate by complexity
    easy = [r for r in records if r["complexity"] == 0]
    hard = [r for r in records if r["complexity"] == 1]
    print(f"Easy (complexity=0): {len(easy)}")
    print(f"Hard (complexity=1): {len(hard)}\n")

    # Rejection reasons per complexity
    for label, subset in [("Easy", easy), ("Hard", hard)]:
        reasons = {}
        for r in subset:
            reason = r.get("rejection_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        print(f"--- {label} rejection reasons ---")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} ({100*count/len(subset):.1f}%)")
        print()

    # Token length analysis
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]

    for label, subset in [("Easy", easy), ("Hard", hard)]:
        chosen_lens = []
        rejected_lens = []
        ratios = []
        diffs = []

        for r in subset:
            c_len = len(tokenizer.encode(r["chosen"]))
            r_len = len(tokenizer.encode(r["rejected"]))
            chosen_lens.append(c_len)
            rejected_lens.append(r_len)
            diffs.append(r_len - c_len)  # positive = rejected is longer
            ratios.append(r_len / max(c_len, 1))

        chosen_lens = np.array(chosen_lens)
        rejected_lens = np.array(rejected_lens)
        diffs = np.array(diffs)
        ratios = np.array(ratios)

        print(f"{'='*60}")
        print(f"  {label} — Token Length Statistics")
        print(f"{'='*60}")

        print(f"\n  Chosen (correct, short) responses:")
        print(f"    mean={chosen_lens.mean():.0f}, std={chosen_lens.std():.0f}, "
              f"min={chosen_lens.min()}, max={chosen_lens.max()}")
        print(f"    Percentiles:")
        for p in percentiles:
            print(f"      P{p:02d}: {np.percentile(chosen_lens, p):.0f}")

        print(f"\n  Rejected responses:")
        print(f"    mean={rejected_lens.mean():.0f}, std={rejected_lens.std():.0f}, "
              f"min={rejected_lens.min()}, max={rejected_lens.max()}")
        print(f"    Percentiles:")
        for p in percentiles:
            print(f"      P{p:02d}: {np.percentile(rejected_lens, p):.0f}")

        print(f"\n  Length difference (rejected - chosen):")
        print(f"    mean={diffs.mean():.0f}, std={diffs.std():.0f}, "
              f"min={diffs.min()}, max={diffs.max()}")
        print(f"    Percentiles:")
        for p in percentiles:
            print(f"      P{p:02d}: {np.percentile(diffs, p):.0f}")

        print(f"\n  Ratio (rejected / chosen):")
        print(f"    mean={ratios.mean():.1f}x, median={np.median(ratios):.1f}x, "
              f"min={ratios.min():.1f}x, max={ratios.max():.1f}x")

        # Breakdown by rejection reason
        for reason in set(r.get("rejection_reason", "unknown") for r in subset):
            sub = [i for i, r in enumerate(subset) if r.get("rejection_reason", "unknown") == reason]
            c = chosen_lens[sub]
            r = rejected_lens[sub]
            d = diffs[sub]
            print(f"\n  Subset: rejection_reason={reason} (n={len(sub)})")
            print(f"    Chosen:   mean={c.mean():.0f}, P50={np.median(c):.0f}, P95={np.percentile(c, 95):.0f}")
            print(f"    Rejected: mean={r.mean():.0f}, P50={np.median(r):.0f}, P95={np.percentile(r, 95):.0f}")
            print(f"    Diff:     mean={d.mean():.0f}, P50={np.median(d):.0f}, P95={np.percentile(d, 95):.0f}")

        print()


if __name__ == "__main__":
    main()
