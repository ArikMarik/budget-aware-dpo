"""
4-Way Augmentation Pipeline: Complexity classification and DPO preference labeling.

Complexity Flag C:
- C=0 (Easy): GSM8K (always), MATH level 1, or low token count
- C=1 (Hard): MATH level 2-5 or high token count

GSM8K invariant: Always C=0; never affected by level or token heuristics.

Preference Labeling:
- Easy-Correct: Short direct paths = Preferred; verbose redundant = Rejected
- Hard-Correct: Shorter-safe CoT = Preferred; verbose = Rejected
- Incorrect: Logically flawed = Rejected (all levels)

See docs/preprocessing_analysis_and_spec.md and docs/PRD_next_stage_preprocessing_and_wandb.md.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm
from torch import Tensor

from src.evaluation.answer_extraction import verify_correctness
from src.utils import count_tokens, get_logger, set_seed

set_seed(42)

logger = get_logger(__name__)

# Configurable thresholds (env or defaults); Qwen tokenizer
HARD_TOKEN_THRESHOLD = int(os.environ.get("HARD_TOKEN_THRESHOLD", 370))
# Per-problem percentile bands for preference labeling.
# For each problem we rank every teacher solution by its teacher_token_count
# within that problem's own distribution; preferred solutions fall inside the
# band, all other *correct* solutions become "rejected" by length.
#   Easy (C=0): preferred = short-and-clean → default [10, 40]
#   Hard (C=1): preferred = short-safe CoT → default [20, 45]
EASY_PREF_PCT_LOW = float(os.environ.get("EASY_PREF_PCT_LOW", 10))
EASY_PREF_PCT_HIGH = float(os.environ.get("EASY_PREF_PCT_HIGH", 40))
# HARD_PREF_PCT_LOW = float(os.environ.get("HARD_PREF_PCT_LOW", 60))
# HARD_PREF_PCT_HIGH = float(os.environ.get("HARD_PREF_PCT_HIGH", 92))
HARD_PREF_PCT_LOW = float(os.environ.get("HARD_PREF_PCT_LOW", 20))
HARD_PREF_PCT_HIGH = float(os.environ.get("HARD_PREF_PCT_HIGH", 45))

REJECTION_REASONS = {'unknown': -1, 'length': 0, 'incorrect': 1}


def _normalize_level(level: Any) -> int | None:
    """Normalize level to numeric string, handling formats like 'Level 2', '2', 2, 'Level ?', etc."""
    if level is None:
        return None
    level_str = str(level).strip()
    # Extract numeric part
    for num in range(1, 6):
        if str(num) in level_str:
            return num

    # unknown/invalid levels
    return None


class SimilarityIndex:
    """Lazy-loaded FAISS index for similarity search on augmented MATH problems."""

    def __init__(self):
        self._index: Any = None
        self._metadata: list[dict] | None = None
        self._model: Any = None
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Lazy load the similarity index. Idempotent - only loads once."""
        if self._loaded:
            return

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            index_path = Path(__file__).parent.parent.parent / "data" / "math_problem_index"
            if not index_path.exists():
                logger.warning(f"Similarity index not found at {index_path}")
                self._loaded = True
                return

            self._index = faiss.read_index(str(index_path / "index.faiss"))

            self._metadata = []
            with open(index_path / "metadata.jsonl", "r") as f:
                for line in f:
                    self._metadata.append(json.loads(line))

            config_path = index_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    model_name = config.get("embedding_model", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
            else:
                model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

            self._model = SentenceTransformer(model_name)

            logger.info(f"Loaded similarity index with {self._index.ntotal} problems")
        except Exception as e:
            logger.warning(f"Failed to load similarity index: {e}")

        self._loaded = True

    def find_similar(self, problem: str, threshold: float = 0.7) -> tuple[int | None, int | None]:
        """Find similar original MATH problem. Returns (complexity, level)."""
        self.ensure_loaded()

        if self._index is None or self._metadata is None or self._model is None:
            return None, None

        try:
            import faiss

            query = self._model.encode([problem], convert_to_numpy=True)
            faiss.normalize_L2(query)

            scores, indices = self._index.search(query, k=1)

            if scores[0][0] >= threshold:
                idx = indices[0][0]
                if idx >= 0 and idx < len(self._metadata):
                    meta = self._metadata[idx]
                    return meta.get("complexity"), _normalize_level(meta.get("level"))
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")

        return None, None


_similarity_index = SimilarityIndex()


def _get_teacher_token_count(example: dict) -> int:
    """Get teacher token count; compute from generated_solution if missing."""
    tc = example.get("teacher_token_count")
    if tc is not None and tc != 0:
        return int(tc)
    sol = example.get("generated_solution", "")
    return count_tokens(sol)


def _verify_correctness(example: dict) -> bool:
    """Return precalculated correctness_flag. However if missing,
    Verify if generated_solution matches expected_answer using tiered checking."""
    if example.get('correctness_flag') is not None:
        return example['correctness_flag']

    generated_solution = example["generated_solution"]
    expected_answer = example["expected_answer"]

    return verify_correctness(
        generated_solution,
        expected_answer,
    )


def classify_complexity(example: dict, avg_token_length: float | None = None) -> tuple[int, int | None]:
    """
    Canonical decision flow:
    1. Exact match: same problem text already classified → reuse
    2. GSM8K: always C=0 (immediate; no further heuristics)
    3. MATH with level: L1 → Easy, L2–L5 → Hard
    4. Augmented MATH: similarity search → find similar original → use its complexity
    5. Unknown: token fallback → default Easy

    Returns (complexity, matched_level) tuple. matched_level is the level copied from
    similar MATH problem, or the original level if available, or None if no match.
    """
    source = str(example.get("problem_source", "")).lower()
    problem = example.get("problem", "")

    # 1. GSM8K / augmented_gsm8k — invariant: always C=0 (per requirement)
    if "gsm" in source:
        return 0, None

    # 2. Original MATH with known level
    if source == "math":
        level = _normalize_level(example.get("level"))
        if level and level > 1:
            return 1, level
        return 0, level

    # 3. Augmented MATH - use similarity search
    if source == "augmented_math" and problem:
        complexity, matched_level = _similarity_index.find_similar(problem)
        if complexity is not None:
            return complexity, matched_level

    # 4. Unknown source: token fallback only for truly unknown sources
    tokens = avg_token_length if avg_token_length is not None else _get_teacher_token_count(example)
    if tokens > HARD_TOKEN_THRESHOLD:
        return 1, None
    return 0, None  # Default Easy


def normalize_problem(text: str) -> str:
    """Normalize problem text for matching: collapse whitespace, strip. Improves level lookup for MATH-origin problems."""
    if not text:
        return ""
    return " ".join(str(text).split())


def _pct_to_token_bounds(sorted_tokens: list[int], pct_low: float, pct_high: float) -> tuple[int, int]:
    """Convert percentile band (0-100) to token count bounds using pre-sorted token list.

    Matches _percentile_rank semantics: rank = 100 * bisect_right(tokens, value) / n.
    - low bound: smallest value where rank >= pct_low
    - high bound: largest value where rank <= pct_high
    """
    from bisect import bisect_right

    n = len(sorted_tokens)
    if n == 0:
        return 0, 0
    # bisect_right gives count of values <= target; invert to get value for percentile
    low_idx = max(0, int(pct_low * n / 100) - 1)
    high_idx = min(int(pct_high * n / 100), n) - 1
    # Handle edge case where pct_high=100 (use last index)
    if high_idx < 0:
        high_idx = n - 1
    return sorted_tokens[low_idx], sorted_tokens[high_idx]


def _band_for_complexity(complexity: int) -> tuple[float, float]:
    """Resolve the (low, high) preferred-percentile band for a complexity flag."""
    if complexity == 0:
        low = EASY_PREF_PCT_LOW
        high = EASY_PREF_PCT_HIGH
    else:
        low = HARD_PREF_PCT_LOW
        high = HARD_PREF_PCT_HIGH
    return float(low), float(high)


def label_preference(
    example: dict,
    low_tokens: int,
    high_tokens: int,
) -> tuple[str, int | None]:
    """Label a solution as preferred/rejected.

    a solution is *preferred* iff its token count's percentile rank within
    the same problem's teacher-solution distribution falls inside the band
    configured for its complexity.
    Everything else correct becomes "rejected" by length — this catches both
    too-short/degenerate solutions and verbose/outlier solutions.
    """
    correct = _verify_correctness(example)
    if not correct:
        return "rejected", REJECTION_REASONS["incorrect"]

    tokens = _get_teacher_token_count(example)
    if low_tokens <= tokens <= high_tokens:
        return "preferred", None

    return "rejected", REJECTION_REASONS["length"]


def load_problem_index(path: Path) -> dict[str, dict]:
    """Load problem index from JSON and build problem_text -> problem data mapping."""
    with open(path) as f:
        index = json.load(f)
    return {normalize_problem(item["problem"]): item for item in index}


def stratified_max_pairs_per_problem_sampling(pairs: list[dict], max_per_problem: int) -> list[dict]:
    """Stratified sampling by rejection_reason to limit pairs per problem."""
    by_reason: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_reason[p["rejection_reason"]].append(p)

    total = len(pairs)
    selected: list[dict] = []
    remaining = max_per_problem

    # TODO - should we add a bias for incorrect pairs ???
    for reason, group in by_reason.items():
        quota = max(1, round(len(group) / total * max_per_problem))
        quota = min(quota, len(group), remaining)
        selected.extend(random.sample(group, quota))
        remaining -= quota
        if remaining <= 0:
            break

    return selected


def filter_pairs_by_length_ratio(pairs: list[dict], length_ratio: float | int = 2) -> list[dict]:
    """Filter pairs by length_ratio.

    Easy (complexity 0): filter if chosen_length * length_ratio > rejected_length AND rejection_reason == 0 (length)
    Hard (complexity 1): filter if rejected_length * length_ratio > chosen_length AND rejection_reason == 0
    Keep all pairs where rejection_reason != 0 (incorrect).
    """
    if length_ratio <= 1.0:
        return pairs

    filtered = []
    for p in pairs:
        complexity = p["complexity"]
        rejection_reason = p["rejection_reason"]
        chosen_length = p["chosen_length"]
        rejected_length = p["rejected_length"]

        if rejection_reason != 0:
            filtered.append(p)
            continue

        if complexity == 0:
            if chosen_length * length_ratio <= rejected_length:
                filtered.append(p)
        elif complexity == 1:
            if rejected_length * length_ratio <= chosen_length:
                filtered.append(p)

    return filtered


def compute_pair_length_ratio(preferred_length: Tensor | int, rejected_length: Tensor | int) -> Tensor | float:
    return rejected_length / preferred_length


def build_dpo_pairs(
    raw_data: list[dict],
    problem_index_path: Path,
    max_per_problem: int | None = None,
    length_ratio: float | int = 1,
) -> list[dict]:
    """
    Group by problem and build preferred/rejected pairs.
    Returns list of pairs with: problem, chosen, rejected, complexity, rejection_reason, chosen_length, rejected_length, problem_id.

    Token counts are stored for training-time filtering by length_ratio.
    problem_id is a unique integer per unique problem for stratified split.

    Args:
        raw_data: List of examples with problem, generated_solution, etc.
        problem_index_path: Path to problem_index.json. If provided, uses existing IDs and complexity.
        max_per_problem: If set (and not -1), limit number of pairs per problem using stratified sampling by rejection_reason.
    """
    problem_index = load_problem_index(problem_index_path)
    logger.info(f"Loaded problem index with {len(problem_index)} problems")

    # Pass 1: group raw examples by problem (no labeling yet — we need the whole
    # group to compute per-problem percentile ranks).
    raw_groups: dict[str, list[dict]] = defaultdict(list)
    problem_meta: dict[str, dict] = {}
    for ex in tqdm(raw_data, desc="Grouping by problem", unit=" examples"):
        problem = normalize_problem(ex["problem"])
        problem_data = problem_index.get(problem)
        if problem_data is None:
            raise ValueError(f"Problem not found in problem index:\n{problem}")
        raw_groups[problem].append(ex)
        problem_meta[problem] = problem_data

    # Pass 2: label each solution using precomputed token bounds per problem.
    # Instead of calculating percentile rank per example (O(log N) per example),
    # compute bounds once per problem (O(1)) then simple comparison.
    groups: dict[str, list[dict]] = defaultdict(list)
    for problem, items in tqdm(raw_groups.items(), desc="Labeling by per-problem percentile", unit=" problems"):
        problem_data = problem_meta[problem]
        problem_id = problem_data["problem_id"]
        level = problem_data["level"]
        # complexity = problem_data["complexity"] # TODO - bring back after running load_real_data.py
        complexity, _ = classify_complexity(problem_data, problem_data['avg_token_length'])

        index_lengths = problem_data.get("token_lengths")
        if index_lengths:
            sorted_tokens = sorted(int(t) for t in index_lengths)
        else:
            sorted_tokens = sorted(_get_teacher_token_count(ex) for ex in items)

        pct_low, pct_high = _band_for_complexity(complexity)
        low_tokens, high_tokens = _pct_to_token_bounds(sorted_tokens, pct_low, pct_high)

        for ex in items:
            label, rejection_reason = label_preference(ex, low_tokens, high_tokens)

            groups[problem].append({
                "problem_id": problem_id,
                **ex,
                "level": level,
                "complexity": complexity,
                "label": label,
                "rejection_reason": rejection_reason,
            })

    pairs: list[dict] = []

    for problem, items in tqdm(groups.items(), desc="Building pairs from groups", unit=" groups"):
        preferred, rejected = [], []
        for x in items:
            (preferred if x["label"] == "preferred" else rejected).append(x)

        problem_id = items[0]["problem_id"]
        complexity = items[0]["complexity"]

        if preferred and rejected:
            problem_pairs = []
            for pw in preferred:
                for rj in rejected:
                    # Skip pairs where the length ratio condition is not satisfied
                    if compute_pair_length_ratio(pw["teacher_token_count"], rj["teacher_token_count"]) < length_ratio:
                        continue

                    problem_pairs.append({
                        "problem": problem,
                        "problem_id": problem_id,
                        "chosen": pw["generated_solution"],
                        "rejected": rj["generated_solution"],
                        "complexity": complexity,
                        "rejection_reason": rj["rejection_reason"],
                        "chosen_length": pw["teacher_token_count"],
                        "rejected_length": rj["teacher_token_count"],
                    })

            if max_per_problem and max_per_problem > 0 and len(problem_pairs) > max_per_problem:
                problem_pairs = stratified_max_pairs_per_problem_sampling(problem_pairs, max_per_problem)

            pairs.extend(problem_pairs)

    logger.info(f'Created a total of {len(pairs):,} pairs')

    return pairs


def load_jsonl(path: Path) -> list[dict]:
    iterator = lambda f: tqdm(f, desc="Loading JSONL", unit=" lines")

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in iterator(f):
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def split_pairs_by_problem(
    pairs: dict,
    val_split: float,
    seed: int = 42,
    filtered_indices: list[int] | None = None,
    max_unique_problems: int = 100_000
) -> tuple[list[int], list[int]]:
    """
    Split pairs into train/val by unique problem to prevent data leakage.
    Ensures the same problem doesn't appear in both sets.

    Stratifies by problem-level complexity (majority complexity among pairs for each problem).
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    set_seed(seed)


    if filtered_indices is None:
        filtered_indices = np.arange(len(pairs)).tolist()

    problem_ids = pairs["problem_ids"].numpy()[filtered_indices]
    complexities = pairs["complexities"].numpy()[filtered_indices]

    # Get unique problems
    unique_problems = np.unique(problem_ids)

    # Build problem -> complexity mapping (pick first sample's complexity)
    problem_to_complexity = {}
    for pid in unique_problems:
        first_idx = np.where(problem_ids == pid)[0][0]
        problem_to_complexity[pid] = complexities[first_idx]

    problem_complexities = np.array([problem_to_complexity[p] for p in unique_problems])

    # TODO - stratify by problem_source too
    # Stratified split
    if len(unique_problems) > max_unique_problems:
        unique_problems, discarded_problems, problem_complexities, discarded_problem_complexities = train_test_split(
        unique_problems, problem_complexities,
        train_size=max_unique_problems,
        stratify=problem_complexities,
        random_state=seed,
    )
    unique_train_problem_ids, unique_val_problem_ids = train_test_split(
        unique_problems,
        test_size=val_split,
        stratify=problem_complexities,
        random_state=seed,
    )

    # Single loop - assign to train or val
    train_indices = []
    val_indices = []
    for i, pid in enumerate(problem_ids):
        if pid in unique_train_problem_ids:
            train_indices.append(filtered_indices[i])
        elif pid in unique_val_problem_ids:
            val_indices.append(filtered_indices[i])

    logger.info(
        f"Data split, filtered indices: {len(filtered_indices):,} out of {len(pairs):,} pairs\n\t"
        f"max_unique_problems={max_unique_problems}): "
        f"Train (unique problems)={len(unique_train_problem_ids)}, Val (unique problems)={len(unique_val_problem_ids)}"
    )

    return train_indices, val_indices


def compute_statistics(
    pairs: list[dict],
) -> dict[str, Any]:
    """Compute full statistics per spec (Section 4), including length ratio histogram."""
    total = len(pairs)

    if total == 0:
        return {
            'easy_preferred_percentile_low': EASY_PREF_PCT_LOW,
            'easy_preferred_percentile_high': EASY_PREF_PCT_HIGH,
            'hard_preferred_percentile_low': HARD_PREF_PCT_LOW,
            'hard_preferred_percentile_high': HARD_PREF_PCT_HIGH,
            "total_pairs": 0,
        }

    # Single pass over all_pairs with tqdm
    pairs_iter = tqdm(pairs, desc="Computing statistics", unit=" pairs")

    rej_correctness = 0
    rej_length = 0
    count_correct_easy = 0
    count_correct_hard = 0
    count_incorrect_easy = 0
    count_incorrect_hard = 0
    chosen_length_sum_easy = 0
    rejected_length_sum_easy = 0
    chosen_length_sum_hard = 0
    rejected_length_sum_hard = 0

    # For histograms
    ratios_easy = []
    ratios_hard = []
    pairs_per_problem: dict[int, int] = defaultdict(int)

    for p in pairs_iter:
        pairs_per_problem[p["problem_id"]] += 1
        rejection_reason = p["rejection_reason"]
        complexity = p["complexity"]

        if rejection_reason == REJECTION_REASONS['incorrect']:
            rej_correctness += 1
            if complexity == 0:
                count_incorrect_easy += 1
            else:
                count_incorrect_hard += 1
        elif rejection_reason == REJECTION_REASONS['length']:
            rej_length += 1

            chosen_length = p["chosen_length"]
            rejected_length = p["rejected_length"]

            if complexity == 0:
                count_correct_easy += 1
                chosen_length_sum_easy += chosen_length
                rejected_length_sum_easy += rejected_length
                ratios_easy.append(compute_pair_length_ratio(p['chosen_length'], p['rejected_length']))
            else:
                count_correct_hard += 1
                chosen_length_sum_hard += chosen_length
                rejected_length_sum_hard += rejected_length
                ratios_hard.append(compute_pair_length_ratio(p['chosen_length'], p['rejected_length']))
        else:
            raise ValueError('All pairs must be rejected by either incorrect/length')

    avg_chosen_length_easy = chosen_length_sum_easy / count_correct_easy if count_correct_easy > 0 else 0
    avg_rejected_length_easy = rejected_length_sum_easy / count_correct_easy if count_correct_easy > 0 else 0
    avg_chosen_length_hard = chosen_length_sum_hard / count_correct_hard if count_correct_hard > 0 else 0
    avg_rejected_length_hard = rejected_length_sum_hard / count_correct_hard if count_correct_hard > 0 else 0
    avg_length_ratio_easy = sum(ratios_easy) / len(ratios_easy) if ratios_easy else 0
    avg_length_ratio_hard = sum(ratios_hard) / len(ratios_hard) if ratios_hard else 0

    # Histogram bins for length ratio
    histogram_edges = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5]
    histogram_bins = []
    for i, edge in enumerate(histogram_edges):
        if i == 0:
            start = 0
            end = edge
        elif i == len(histogram_edges)-1:
            start = edge
            end = float('inf')
        else:
            start = histogram_edges[i-1]
            end = edge

        histogram_bins.append((start, end))

    def compute_histogram(ratios: list[float] | list[int], bins: list[tuple[float, float]]) -> dict:
        counts = []
        for start_bin, end_bin in bins:
            count = sum(1 for r in ratios if start_bin <= r < end_bin)
            counts.append(count)
        return {"bins": bins, "counts": counts}

    def compute_reverse_cumulative(histogram: dict) -> dict:
        bins, counts = histogram['bins'], histogram['counts']
        result = {}
        total = 0
        biggest_end = bins[-1][-1]
        for i, (start_bin, end_bin) in enumerate(bins[::-1], start=1):
            total += counts[-i]
            result[f"ratio_gte_[{start_bin}, {biggest_end})"] = total
        return result

    # Pairs-per-problem histogram
    problem_pair_counts = sorted(pairs_per_problem.values())
    pairs_per_problem_bins = [(1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, float('inf'))]
    pairs_per_problem_histogram = compute_histogram(problem_pair_counts, pairs_per_problem_bins)

    histogram_easy = compute_histogram(ratios_easy, histogram_bins) if ratios_easy else {"bins": histogram_bins, "counts": [0] * len(histogram_bins)}
    histogram_hard = compute_histogram(ratios_hard, histogram_bins) if ratios_hard else {"bins": histogram_bins, "counts": [0] * len(histogram_bins)}
    cumulative_easy = compute_reverse_cumulative(histogram_easy) if ratios_easy else {f"ratio_gte_{b}": 0 for b in histogram_bins}
    cumulative_hard = compute_reverse_cumulative(histogram_hard) if ratios_hard else {f"ratio_gte_{b}": 0 for b in histogram_bins}

    return {
        "easy_preferred_percentile_low": EASY_PREF_PCT_LOW,
        "easy_preferred_percentile_high": EASY_PREF_PCT_HIGH,
        "hard_preferred_percentile_low": HARD_PREF_PCT_LOW,
        "hard_preferred_percentile_high": HARD_PREF_PCT_HIGH,
        "total_pairs": total,
        "rejected_by_correctness": rej_correctness,
        "rejected_by_length": rej_length,
        "rejected_by_correctness_pct": round(100 * rej_correctness / total, 2),
        "rejected_by_length_pct": round(100 * rej_length / total, 2),
        "easy_pairs": count_correct_easy + count_incorrect_easy,
        "hard_pairs": count_correct_hard + count_incorrect_hard,
        "correct_easy_pairs": count_correct_easy,
        "correct_hard_pairs": count_correct_hard,
        "incorrect_easy_pairs": count_incorrect_easy,
        "incorrect_hard_pairs": count_incorrect_hard,
        "correct_easy_pairs_pct": round(100 * count_correct_easy / rej_length, 2),
        "correct_hard_pairs_pct": round(100 * count_correct_hard / rej_length, 2),
        "incorrect_easy_pairs_pct": round(100 * count_incorrect_easy / rej_correctness, 2),
        "incorrect_hard_pairs_pct": round(100 * count_incorrect_hard / rej_correctness, 2),
        "avg_chosen_length_easy": round(avg_chosen_length_easy, 2),
        "avg_rejected_length_easy": round(avg_rejected_length_easy, 2),
        "avg_chosen_length_hard": round(avg_chosen_length_hard, 2),
        "avg_rejected_length_hard": round(avg_rejected_length_hard, 2),
        "avg_length_ratio_easy": round(avg_length_ratio_easy, 2),
        "avg_length_ratio_hard": round(avg_length_ratio_hard, 2),
        "length_ratio_histogram": {
            "complexity_easy": histogram_easy,
            "complexity_hard": histogram_hard,
        },
        "length_ratio_cumulative": {
            "complexity_easy": cumulative_easy,
            "complexity_hard": cumulative_hard,
        },
        "pairs_per_problem": {
            "unique_problems": len(pairs_per_problem),
            "avg_pairs_per_problem": round(total / len(pairs_per_problem), 2),
            "max_pairs_per_problem": max(problem_pair_counts),
            "min_pairs_per_problem": min(problem_pair_counts),
            "histogram": pairs_per_problem_histogram,
        },
    }
