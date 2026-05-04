import torch
import numpy as np
from collections import Counter

from src.data.preprocessing import extract_balanced_set, train_validation_split


def test_extract_balanced_set(subsample, SUBSAMPLE_SIZE):
    TEST_VAL_SIZE = 500

    try:
        val_indices = extract_balanced_set(subsample, size=TEST_VAL_SIZE, seed=42)
    except Exception as e:
        print(f"❌ extract_balanced_set failed: {e}")
        import traceback; traceback.print_exc()
        return False

    print(f"✅ extract_balanced_set returned {len(val_indices)} indices")
    assert val_indices == sorted(val_indices), "Indices not sorted"
    assert all(0 <= i < SUBSAMPLE_SIZE for i in val_indices), "Invalid indices"

    pids = subsample["problem_id"].numpy()[val_indices]
    comps = subsample["complexity"].numpy()[val_indices]

    # Get unique problems and their first occurrence complexity
    unique_pids, first_indices = np.unique(pids, return_index=True)
    unique_comps = comps[first_indices]

    # Use Counter for cleaner counting
    counts = Counter(unique_comps)
    easy, hard = counts.get(0, 0), counts.get(1, 0)

    print(f"✅ Balanced: {len(unique_pids)} unique problems (easy: {easy}, hard: {hard})")
    assert abs(easy - hard) < 5, f"Easy/hard counts unbalanced, {easy = }, {hard = }"
    return True


def test_train_validation_split(subsample, SUBSAMPLE_SIZE):
    TEST_VAL_SIZE = 1000
    TEST_TRAIN_SIZE = 10_000

    try:
        train_indices, val_indices = train_validation_split(
            subsample,
            train_size=TEST_TRAIN_SIZE,
            validation_size=TEST_VAL_SIZE,
            seed=42
        )
    except Exception as e:
        print(f"❌ train_validation_split failed: {e}")
        import traceback; traceback.print_exc()
        return False

    print(f"✅ train_validation_split returned {len(train_indices)} train + {len(val_indices)} val indices")

    # Check no overlap between train and val
    overlap = set(train_indices) & set(val_indices)
    assert len(overlap) == 0, f"Overlap detected: {len(overlap)} indices in both sets"

    # Check sorted
    assert train_indices == sorted(train_indices), "Train indices not sorted"
    assert val_indices == sorted(val_indices), "Val indices not sorted"

    # Check valid indices
    assert all(0 <= i < SUBSAMPLE_SIZE for i in train_indices), "Invalid train indices"
    assert all(0 <= i < SUBSAMPLE_SIZE for i in val_indices), "Invalid val indices"

    # Check no data leakage: validation problems should not appear in train
    val_pids = set(subsample["problem_id"].numpy()[val_indices])
    train_pids = set(subsample["problem_id"].numpy()[train_indices])
    leakage = val_pids & train_pids
    assert len(leakage) == 0, f"Data leakage: {len(leakage)} problems in both sets"

    # Check balanced easy/hard in both sets
    for name, indices in [("train", train_indices), ("val", val_indices)]:
        pids = subsample["problem_id"].numpy()[indices]
        comps = subsample["complexity"].numpy()[indices]
        unique_pids, first_indices = np.unique(pids, return_index=True)
        counts = Counter(comps[first_indices])
        easy, hard = counts.get(0, 0), counts.get(1, 0)
        print(f"✅ {name}: {len(unique_pids)} unique problems (easy: {easy}, hard: {hard})")
        assert abs(easy - hard) < 5, f"{name} not balanced: easy={easy}, hard={hard}"

    return True


def main():
    PAIRS_PATH = "data/processed_dpo_dataset/pairs_info.pt"
    SUBSAMPLE_SIZE = 40_000

    print(f"Loading {PAIRS_PATH}...")
    full_pairs = torch.load(PAIRS_PATH, map_location="cpu")
    total = len(full_pairs["problem_id"])
    sub_idx = np.random.choice(total, min(SUBSAMPLE_SIZE, total), replace=False)
    subsample = {k: v[sub_idx] for k, v in full_pairs.items()}
    print(f"Subsampled to {len(sub_idx)} pairs\n")

    print("=" * 50)
    print("Testing extract_balanced_set")
    print("=" * 50)
    if not test_extract_balanced_set(subsample, SUBSAMPLE_SIZE):
        return

    print("\n" + "=" * 50)
    print("Testing train_validation_split")
    print("=" * 50)
    if not test_train_validation_split(subsample, SUBSAMPLE_SIZE):
        return

    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    main()
