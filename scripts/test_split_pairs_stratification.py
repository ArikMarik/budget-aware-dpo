import numpy as np
from collections import Counter

from src.data.preprocessing import split_pairs_by_problem


# ---- Mock pairs structure ----
class MockTensor:
    def __init__(self, arr):
        self._arr = np.array(arr)

    def __len__(self):
        return len(self._arr)

    def numpy(self):
        return self._arr


def make_mock_pairs(n_pairs=2000, n_problems=200):
    rng = np.random.default_rng(42)

    problem_ids = rng.integers(0, n_problems, size=n_pairs)
    complexities = rng.integers(0, 3, size=n_pairs)
    sources = rng.integers(0, 2, size=n_pairs)

    return {
        "problem_id": MockTensor(problem_ids),
        "complexity": MockTensor(complexities),
        "problem_source": MockTensor(sources),
    }


# ---- Your fixed function (same as before) ----
# def split_pairs_by_problem(
#     pairs: dict,
#     val_split: float,
#     seed: int = 42,
#     filtered_indices=None,
#     max_unique_problems: int = 100_000
# ):
#     from sklearn.model_selection import train_test_split

#     if filtered_indices is None:
#         filtered_indices = np.arange(len(pairs["problem_id"].numpy())).tolist()

#     problem_ids = pairs["problem_id"].numpy()[filtered_indices]
#     complexities = pairs["complexity"].numpy()[filtered_indices]
#     problem_sources = pairs["problem_source"].numpy()[filtered_indices]

#     unique_problems = np.unique(problem_ids)

#     problem_to_complexity_and_sources = {}
#     for pid in unique_problems:
#         first_idx = np.where(problem_ids == pid)[0][0]
#         problem_to_complexity_and_sources[pid] = (
#             complexities[first_idx],
#             problem_sources[first_idx],
#         )

#     # ✅ FIXED
#     problem_strata = np.array([
#         f"{problem_to_complexity_and_sources[p][0]}_{problem_to_complexity_and_sources[p][1]}"
#         for p in unique_problems
#     ])

#     if len(unique_problems) > max_unique_problems:
#         unique_problems, _, problem_strata, _ = train_test_split(
#             unique_problems,
#             problem_strata,
#             train_size=max_unique_problems,
#             stratify=problem_strata,
#             random_state=seed,
#         )

#     unique_train_problem_ids, unique_val_problem_ids = train_test_split(
#         unique_problems,
#         test_size=val_split,
#         stratify=problem_strata,
#         random_state=seed,
#     )

#     train_set = set(unique_train_problem_ids)
#     val_set = set(unique_val_problem_ids)

#     train_indices, val_indices = [], []
#     for i, pid in enumerate(problem_ids):
#         if pid in train_set:
#             train_indices.append(filtered_indices[i])
#         elif pid in val_set:
#             val_indices.append(filtered_indices[i])

#     return train_indices, val_indices


# ---- Stats helpers ----
def compute_stats(indices, pairs):
    pids = pairs["problem_id"].numpy()[indices]
    complexities = pairs["complexity"].numpy()[indices]
    sources = pairs["problem_source"].numpy()[indices]

    # problem-level stats (important!)
    unique_pids = np.unique(pids)

    # take first occurrence per problem (same logic as training)
    pid_to_cs = {}
    for pid in unique_pids:
        idx = np.where(pids == pid)[0][0]
        pid_to_cs[pid] = (complexities[idx], sources[idx])

    complexity_counts = Counter(c for c, _ in pid_to_cs.values())
    source_counts = Counter(s for _, s in pid_to_cs.values())
    joint_counts = Counter(pid_to_cs.values())

    return {
        "num_pairs": len(indices),
        "num_problems": len(unique_pids),
        "complexity": complexity_counts,
        "source": source_counts,
        "joint": joint_counts,
    }


def print_distribution(name, stats):
    print(f"\n=== {name} ===")
    print(f"Pairs: {stats['num_pairs']}, Unique problems: {stats['num_problems']}")

    print("\nComplexity distribution:")
    for k, v in sorted(stats["complexity"].items()):
        print(f"  {k}: {v}")

    print("\nSource distribution:")
    for k, v in sorted(stats["source"].items()):
        print(f"  {k}: {v}")

    print("\nJoint (complexity, source):")
    for k, v in sorted(stats["joint"].items()):
        print(f"  {k}: {v}")


def compare_distributions(global_stats, train_stats, val_stats):
    print("\n=== Stratification Check (ratios) ===")

    def normalize(counter):
        total = sum(counter.values())
        return {k: v / total for k, v in counter.items()}

    g = normalize(global_stats["joint"])
    t = normalize(train_stats["joint"])
    v = normalize(val_stats["joint"])

    keys = sorted(set(g) | set(t) | set(v))

    print("\n(complexity, source) ratios:")
    for k in keys:
        print(
            f"{k}: "
            f"global={g.get(k,0):.3f}, "
            f"train={t.get(k,0):.3f}, "
            f"val={v.get(k,0):.3f}"
        )


# ---- Run test ----
if __name__ == "__main__":
    pairs = make_mock_pairs(n_pairs=20_000, n_problems=1000)

    train_idx, val_idx = split_pairs_by_problem(pairs, val_split=0.2)

    all_idx = list(range(len(pairs["problem_id"].numpy())))

    global_stats = compute_stats(all_idx, pairs)
    train_stats = compute_stats(train_idx, pairs)
    val_stats = compute_stats(val_idx, pairs)

    print_distribution("GLOBAL", global_stats)
    print_distribution("TRAIN", train_stats)
    print_distribution("VAL", val_stats)

    compare_distributions(global_stats, train_stats, val_stats)

    # leakage check
    train_pids = set(pairs["problem_id"].numpy()[train_idx])
    val_pids = set(pairs["problem_id"].numpy()[val_idx])
    print(f"\nLeakage (should be 0): {len(train_pids & val_pids)}")