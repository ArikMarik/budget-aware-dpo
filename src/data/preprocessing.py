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
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from torch import Tensor

from src.config import EMBEDDING_MODEL, PROBLEM_TO_LEVEL_PATH, SEED, SIMILARITY_INDEX_DIR
from src.evaluation.answer_extraction import verify_correctness
from src.utils import count_tokens, get_logger, set_seed

set_seed(SEED)

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
SOURCE_TO_INT = {"gsm8k": 0, "math": 1, "augmented_math": 2, "augmented_gsm8k": 3}
INT_TO_SOURCE = {v: k for k, v in SOURCE_TO_INT. items ()}

MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


def _source_to_int(source: str) -> int:
    return SOURCE_TO_INT.get (source. lower(), -1) # -1 for unknown


def load_math_problem_to_level(use_cache: bool = True) -> dict[str, str]:
    """Load MATH train split and build problem text -> level mapping.

    Loads from HuggingFace with fallback sources, and caches result to
    data/problem_to_level.pkl to avoid re-downloading.

    Args:
        use_cache: If True, load from pickle cache if exists.

    Returns:
        dict mapping normalized problem text -> level string (e.g., "Level 1")
    """
    from datasets import load_dataset, concatenate_datasets

    if use_cache and PROBLEM_TO_LEVEL_PATH.exists():
        with open(PROBLEM_TO_LEVEL_PATH, "rb") as f:
            import pickle
            mapping = pickle.load(f)
        logger.info("Loaded problem to level mapping from %s (%d problems)", PROBLEM_TO_LEVEL_PATH, len(mapping))
        return mapping

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
        problem = normalize_problem(item.get("problem", item.get("question", "")))
        level = item.get("level", "")
        if problem and level:
            mapping[problem] = str(level)

    # Cache the result
    import pickle
    PROBLEM_TO_LEVEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROBLEM_TO_LEVEL_PATH, "wb") as f:
        pickle.dump(mapping, f)
    logger.info("Built and cached level map for %d MATH problems to %s", len(mapping), PROBLEM_TO_LEVEL_PATH)

    return mapping


def load_math_problems_with_complexity(use_cache: bool = True) -> dict[str, dict]:
    """Load MATH train problems with level and complexity classification.

    Args:
        use_cache: If True, load from pickle cache if exists.

    Returns:
        dict mapping problem text -> {"level": int, "complexity": int}
        Complexity: 0 (Easy) for level 1, 1 (Hard) for levels 2-5
    """
    problem_to_level = load_math_problem_to_level(use_cache=use_cache)

    math_problems = {}
    for problem, level_str in problem_to_level.items():
        level = _normalize_level(level_str)
        if level is None:
            continue
        # Level 1 = Easy (0), Level 2-5 = Hard (1)
        complexity, _ = classify_complexity({"problem": problem, "level": level_str, "problem_source": "math"})
        if problem not in math_problems:
            math_problems[problem] = {
                "level": level,
                "complexity": complexity,
            }

    return math_problems


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
        self._model_name: str = EMBEDDING_MODEL
        self._device: str = "cpu"
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Lazy load the similarity index. Idempotent - only loads once."""
        if self._loaded:
            return

        try:
            if not SIMILARITY_INDEX_DIR.exists():
                logger.warning(f"Similarity index not found at {SIMILARITY_INDEX_DIR}")
                self._loaded = True
                return

            self._index = faiss.read_index(str(SIMILARITY_INDEX_DIR / "index.faiss"))

            self._metadata = []
            with open(SIMILARITY_INDEX_DIR / "metadata.jsonl", "r") as f:
                for line in f:
                    self._metadata.append(json.loads(line))

            config_path = SIMILARITY_INDEX_DIR / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    self._model_name = config.get("embedding_model", EMBEDDING_MODEL)
            else:
                self._model_name = EMBEDDING_MODEL

            # Auto-detect GPU for SentenceTransformer (FAISS stays on CPU)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self._model_name, device=self._device)

            logger.info(f"Loaded similarity index with {self._index.ntotal} problems (model on {self._device})")
        except Exception as e:
            logger.warning(f"Failed to load similarity index: {e}")

        self._loaded = True

    def find_similar(self, problem: str, threshold: float = 0.6) -> tuple[int | None, int | None]:
        """Find similar original MATH problem. Returns (complexity, level)."""
        self.ensure_loaded()

        if self._index is None or self._metadata is None or self._model is None:
            return None, None

        try:
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

    def find_similar_batch(
        self,
        problems: list[str],
        threshold: float = 0.6,
        batch_size: int = 256,
    ) -> list[tuple[int | None, int | None]]:
        """Batch find similar MATH problems.

        Args:
            problems: List of problem texts to search for.
            threshold: Minimum similarity score (default 0.6).
            batch_size: Batch size for encoding (default 128).

        Returns:
            List of (complexity, level) tuples, one per problem.
            complexity is None and level is None when no match found.
        """
        self.ensure_loaded()

        if self._index is None or self._metadata is None or self._model is None:
            return [(None, None)] * len(problems)

        embeddings = []
        embeddings = self._model.encode(problems, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)

        faiss.normalize_L2(embeddings)

        # Process in batches for progress bar
        scores_list, indices_list = [], []
        num_batches = (len(embeddings) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Searching index", unit=" batch"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(embeddings))
            batch = embeddings[start_idx:end_idx]
            scores_batch, indices_batch = self._index.search(batch, k=1)
            scores_list.append(scores_batch)
            indices_list.append(indices_batch)

        scores = np.vstack(scores_list)
        indices = np.vstack(indices_list)

        results = []
        for i in range(len(problems)):
            if scores[i][0] >= threshold:
                meta_idx = indices[i][0]
                if meta_idx >= 0 and meta_idx < len(self._metadata):
                    meta = self._metadata[meta_idx]
                    results.append((meta.get("complexity"), _normalize_level(meta.get("level"))))
                    continue
            results.append((None, None))

        return results


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


def load_problem_to_index(path: Path) -> dict[str, dict]:
    """Load problem to index from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def stratified_max_pairs_per_problem_sampling(pairs: list[dict], max_per_problem: int, seed: int = SEED) -> list[dict]:
    """Stratified sampling by rejection_reason with weighted selection based on length_ratio."""
    by_reason: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_reason[p["rejection_reason"]].append(p)

    total = len(pairs)
    selected: list[dict] = []
    remaining = max_per_problem

    # TODO - should we add a bias for incorrect pairs ???
    rng = np.random.default_rng(seed)
    for reason, group in by_reason.items():
        quota = max(1, round(len(group) / total * max_per_problem))
        quota = min(quota, len(group), remaining)
        if quota <= 0:
            continue

        # Compute weights based on log(length_ratio), set to 0 if log(ratio) < 0
        weights = []
        for p in group:
            ratio = compute_pair_length_ratio(p["chosen_length"], p["rejected_length"])
            # avoid log(0) and set to 0 if negative
            weights.append(max(np.log(max(ratio, 1e-6)), 0.0))

        sum_weights = sum(weights)
        if sum_weights > 0:
            # Normalize to probabilities
            probabilities = np.array(weights) / sum_weights
            # Weighted sampling without replacement
            indices = rng.choice(len(group), size=quota, replace=False, p=probabilities)
        else:
            # Fallback to uniform sampling if all weights are zero
            indices = rng.choice(len(group), size=quota, replace=False)

        selected.extend(group[i] for i in indices)
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
    problem_to_index_path: Path,
    max_per_problem: int | None = None,
    length_ratio: float | int = 1,
    over_limit_json_path: Path | None = None,
) -> list[dict]:
    """
    Group by problem and build preferred/rejected pairs.
    Returns list of pairs with: problem, chosen, rejected, complexity, rejection_reason, chosen_length, rejected_length, problem_id.

    Token counts are stored for training-time filtering by length_ratio.
    problem_id is a unique integer per unique problem for stratified split.

    Args:
        raw_data: List of examples with problem, generated_solution, etc.
        problem_to_index_path: Path to problem_to_index.pkl. If provided, uses existing IDs and complexity.
        max_per_problem: If set (and not -1), limit number of pairs per problem using stratified sampling by rejection_reason.
        over_limit_json_path: Path to JSON file with problems exceeding token limit (to skip).
    """
    problem_to_index_dict = load_problem_to_index(problem_to_index_path)
    logger.info(f"Loaded problem to index with {len(problem_to_index_dict)} problems")

    # Load over-limit problem IDs to skip
    over_limit_ids = set()
    if over_limit_json_path and Path(over_limit_json_path).exists():
        with open(over_limit_json_path) as f:
            over_limit_data = json.load(f)
        over_limit_ids = {item["problem_id"] for item in over_limit_data}
        logger.info(f"Loaded {len(over_limit_ids)} over-limit problem IDs to skip")

    # Pass 1: group raw examples by problem (no labeling yet — we need the whole
    # group to compute per-problem percentile ranks).
    raw_groups: dict[str, list[dict]] = defaultdict(list)
    for ex in tqdm(raw_data, desc="Grouping by problem", unit=" examples"):
        normalized_problem = normalize_problem(ex["problem"])
        raw_groups[normalized_problem].append(ex)

    # Pass 2: label each solution using precomputed token bounds per problem.
    # Instead of calculating percentile rank per example (O(log N) per example),
    # compute bounds once per problem (O(1)) then simple comparison.
    groups: dict[str, list[dict]] = defaultdict(list)
    for normalized_problem, items in tqdm(raw_groups.items(), desc="Labeling by per-problem percentile", unit=" problems"):
        problem_data = problem_to_index_dict.get(normalized_problem)
        if problem_data is None:
            raise ValueError(f"Problem not found in problem to index:\n{normalized_problem}")

        problem_id = problem_data["problem_id"]
        level = problem_data["level"]
        complexity = problem_data["complexity"]

        # Skip problems that exceed token limit
        if problem_id in over_limit_ids:
            continue

        index_lengths = problem_data.get("token_lengths")
        if index_lengths:
            sorted_tokens = sorted(int(t) for t in index_lengths)
        else:
            sorted_tokens = sorted(_get_teacher_token_count(ex) for ex in items)

        pct_low, pct_high = _band_for_complexity(complexity)
        low_tokens, high_tokens = _pct_to_token_bounds(sorted_tokens, pct_low, pct_high)

        for ex in items:
            label, rejection_reason = label_preference(ex, low_tokens, high_tokens)

            groups[normalized_problem].append({
                "problem_id": problem_id,
                **ex,
                "level": level,
                "complexity": complexity,
                "label": label,
                "rejection_reason": rejection_reason,
            })

    pairs: list[dict] = []

    # i = 0
    for normalized_problem, items in tqdm(groups.items(), desc="Building pairs from groups", unit=" groups"):
        # i += 1
        preferred, rejected = [], []
        for x in items:
            (preferred if x["label"] == "preferred" else rejected).append(x)

        problem_id = items[0]["problem_id"]
        problem = items[0]["problem"]
        complexity = items[0]["complexity"]
        problem_source = items[0]["problem_source"]

        if preferred and rejected:
            problem_pairs = []
            skipped_ratio = 0
            for pw in preferred:
                for rj in rejected:
                    # Skip pairs where the length ratio condition is not satisfied
                    if compute_pair_length_ratio(pw["teacher_token_count"], rj["teacher_token_count"]) < length_ratio:
                        skipped_ratio += 1
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
                        "problem_source": _source_to_int(problem_source),
                    })

            if max_per_problem and max_per_problem > 0 and len(problem_pairs) > max_per_problem:
                problem_pairs = stratified_max_pairs_per_problem_sampling(problem_pairs, max_per_problem)

            pairs.extend(problem_pairs)

        # if i >= 10_000:
        #     break

    logger.info(f'Created a total of {len(pairs):,} pairs (skipped by length ratio: {skipped_ratio:,})')

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


def safe_stratified_split(
    *arrays: np.ndarray,
    strata: np.ndarray,
    test_size: float | int | None = None,
    train_size: float | int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, ...]:
    """
    Split multiple arrays handling single-member strata gracefully.

    Accepts variadic arrays (e.g., problems, strata, metadata) and splits them
    identically by stratifying on the `strata` array, handling single-member strata
    by splitting them randomly.
    """
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    if not arrays:
        raise ValueError("At least one array must be provided")

    total_items = len(arrays[0])
    for arr in arrays:
        if len(arr) != total_items:
            raise ValueError("All arrays must have the same length")

    # Calculate test count
    if test_size is not None and train_size is not None:
        raise ValueError("Cannot specify both test_size and train_size")
    if test_size is not None:
        test_item_count = int(total_items * test_size) if isinstance(test_size, float) else test_size
    elif train_size is not None:
        train_item_count = int(total_items * train_size) if isinstance(train_size, float) else train_size
        test_item_count = total_items - train_item_count
    else:
        raise ValueError("Must specify either test_size or train_size")

    # Count items per stratum using np.unique with axis=0 and return_counts
    unique_stratum_values, stratum_counts = np.unique(strata, axis=0, return_counts=True)
    stratum_to_count_mapping = {
        tuple(stratum_value): count
        for stratum_value, count in zip(unique_stratum_values, stratum_counts)
    }

    # Separate multi-member vs single-member strata
    multi_member_mask = np.array([
        stratum_to_count_mapping.get(tuple(stratum_value), 0) >= 2
        for stratum_value in tqdm(strata, desc="Checking stratum membership", unit=" strata")
    ])

    multi_member_indices = np.where(multi_member_mask)[0]
    single_member_indices = np.where(~multi_member_mask)[0]

    multi_member_strata = strata[multi_member_mask]
    single_member_strata = strata[~multi_member_mask]

    # Allocate test count proportionally, but ensure minimum for stratification
    if len(multi_member_indices) > 0:
        # Count unique strata in multi-member group
        unique_multi_strata, multi_strata_counts = np.unique(multi_member_strata, axis=0, return_counts=True)
        num_multi_classes = len(unique_multi_strata)

        # Calculate proportional test count
        proportional_test_count = round(test_item_count * len(multi_member_indices) / total_items)

        # Check if stratified split is possible (need at least 2 per class: 1 train, 1 test)
        min_for_stratify = num_multi_classes * 2

        if len(multi_member_indices) >= min_for_stratify:
            # Ensure test count >= num_classes and train count >= num_classes
            multi_member_test_count = max(num_multi_classes, min(proportional_test_count, len(multi_member_indices) - num_multi_classes))
            multi_member_test_count = max(0, min(multi_member_test_count, len(multi_member_indices)))
        else:
            # Not enough samples for stratification, will use random split
            multi_member_test_count = max(0, min(proportional_test_count, len(multi_member_indices)))
    else:
        multi_member_test_count = 0

    # Split multi-member: use stratification if possible, else random
    if len(multi_member_indices) > 0 and 0 < multi_member_test_count < len(multi_member_indices):
        # Check if we can use stratified split
        unique_multi_strata = np.unique(multi_member_strata, axis=0)
        num_multi_classes = len(unique_multi_strata)

        can_stratify = (
            len(multi_member_indices) - multi_member_test_count >= num_multi_classes and
            multi_member_test_count >= num_multi_classes
        )

        if can_stratify:
            multi_member_train_indices, multi_member_test_indices = train_test_split(
                multi_member_indices,
                test_size=multi_member_test_count / len(multi_member_indices),
                stratify=multi_member_strata,
                random_state=random_state
            )
        else:
            # Fall back to random split
            multi_member_train_indices, multi_member_test_indices = train_test_split(
                multi_member_indices,
                test_size=multi_member_test_count / len(multi_member_indices),
                random_state=random_state
            )
    elif len(multi_member_indices) > 0:
        multi_member_train_indices, multi_member_test_indices = (
            (multi_member_indices, np.array([])) if multi_member_test_count == 0
            else (np.array([]), multi_member_indices)
        )
    else:
        multi_member_train_indices, multi_member_test_indices = np.array([]), np.array([])

    single_member_test_count = test_item_count - multi_member_test_count

    # Split single-member indices randomly
    if len(single_member_indices) > 0 and 0 < single_member_test_count < len(single_member_indices):
        single_member_train_indices, single_member_test_indices = train_test_split(
            single_member_indices,
            test_size=single_member_test_count / len(single_member_indices),
            random_state=random_state
        )
    elif len(single_member_indices) > 0:
        single_member_train_indices, single_member_test_indices = (
            (single_member_indices, np.array([])) if single_member_test_count == 0
            else (np.array([]), single_member_indices)
        )
    else:
        single_member_train_indices, single_member_test_indices = np.array([]), np.array([])

    # Combine indices
    combined_train_indices = np.concatenate([
        np.array(multi_member_train_indices, dtype=int),
        np.array(single_member_train_indices, dtype=int)
    ])
    combined_test_indices = np.concatenate([
        np.array(multi_member_test_indices, dtype=int),
        np.array(single_member_test_indices, dtype=int)
    ])

    # Split all arrays using the combined indices
    result = []
    for arr in tqdm(arrays, desc="Splitting arrays", unit=" array"):
        result.append(arr[combined_train_indices])
        result.append(arr[combined_test_indices])

    return tuple(result)


def split_pairs_by_problem(
    pairs: dict,
    val_split: float,
    seed: int = SEED,
    filtered_indices: list[int] | np.ndarray | None = None,
    max_unique_problems: int = 100_000
) -> tuple[list[int], list[int]]:
    """
    Split pairs into train/val by unique problem to prevent data leakage.
    Ensures the same problem doesn't appear in both sets.

    Stratifies by problem-level complexity (majority complexity among pairs for each problem).
    """
    set_seed(seed)

    num_pairs = len(pairs["problem_id"])
    if filtered_indices is None:
        filtered_indices = np.arange(num_pairs)
    else:
        filtered_indices = np.array(filtered_indices)

    problem_ids = pairs["problem_id"].numpy()[filtered_indices]
    complexities = pairs["complexity"].numpy()[filtered_indices]
    problem_sources = pairs["problem_source"].numpy()[filtered_indices]

    # Get unique problems
    unique_problems = np.unique(problem_ids)

    # Build problem -> complexity and source mapping (pick first sample's complexity)
    problem_to_complexity_and_sources = {}
    iterator = tqdm(problem_ids, desc='Build problem -> complexity and source mapping')
    for i, pid in enumerate(iterator):
        if pid not in problem_to_complexity_and_sources:
            problem_to_complexity_and_sources[pid] = (complexities[i], problem_sources[i])
            if len(problem_to_complexity_and_sources) >= len(unique_problems):
                iterator.close()
                break

    problem_strata = np.array([problem_to_complexity_and_sources[p] for p in unique_problems])

    # Safe stratified split (handles single-member strata)
    if len(unique_problems) > max_unique_problems:
        unique_problems, discarded_problems, problem_strata, discarded_problem_strata = safe_stratified_split(
            unique_problems, problem_strata,
            strata=problem_strata,
            train_size=max_unique_problems,
            random_state=seed,
        )

    unique_train_problem_ids, unique_val_problem_ids = safe_stratified_split(
        unique_problems,
        strata=problem_strata,
        test_size=val_split,
        random_state=seed,
    )

    # Single loop - assign to train or val
    train_mask = np.isin(problem_ids, unique_train_problem_ids)
    val_mask = np.isin(problem_ids, unique_val_problem_ids)
    train_indices = filtered_indices[train_mask].tolist()
    val_indices = filtered_indices[val_mask].tolist()

    logger.info(
        f"Data split, filtered indices: {len(filtered_indices):,} out of {num_pairs:,} pairs\n\t"
        f"max_unique_problems={max_unique_problems}): "
        f"Train (unique problems)={len(unique_train_problem_ids)}, Val (unique problems)={len(unique_val_problem_ids)}"
    )

    return train_indices, val_indices


def compute_statistics(
    dataset_path: Path,
) -> dict[str, Any]:
    """Compute full statistics per spec (Section 4), including length ratio histogram."""
    pairs = load_jsonl(dataset_path)
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
