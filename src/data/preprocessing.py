"""
4-Way Augmentation Pipeline: Complexity classification and DPO preference labeling.

Complexity Flag C:
- C=0 (Easy): GSM8K (always), MATH level 1-2, or low token count
- C=1 (Hard): MATH level 4-5 or high token count

GSM8K invariant: Always C=0; never affected by level or token heuristics.

Preference Labeling:
- Easy-Correct: Short direct paths = Preferred; verbose redundant = Rejected
- Hard-Correct: Detailed CoT = Preferred; oversimplified = Rejected
- Incorrect: Logically flawed = Rejected (all levels)

See docs/preprocessing_analysis_and_spec.md and docs/PRD_next_stage_preprocessing_and_wandb.md.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.answer_extraction import verify_correctness
from src.utils import count_tokens, get_logger, set_seed

set_seed(42)

logger = get_logger(__name__)

# Configurable thresholds (env or defaults); Qwen tokenizer
EASY_TOKEN_THRESHOLD = int(os.environ.get("EASY_TOKEN_THRESHOLD", 140))
HARD_TOKEN_THRESHOLD = int(os.environ.get("HARD_TOKEN_THRESHOLD", 250))
REJECTION_REASONS = {'unknown': -1, 'length': 0, 'incorrect': 1}

_VALID_MATH_LEVELS = {"1", "2", "3", "4", "5"}


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
    problem = example["problem"]

    return verify_correctness(
        generated_solution,
        expected_answer,
        problem,
    )


def _normalize_level(level: Any) -> str | None:
    """Normalize level to numeric string, handling formats like 'Level 2', '2', 2, 'Level ?', etc."""
    if level is None:
        return None
    level_str = str(level).strip()
    # Skip unknown/invalid levels
    if "?" in level_str or "unknown" in level_str.lower():
        return None
    # Extract numeric part
    for num in ("1", "2", "3", "4", "5"):
        if num in level_str:
            return num
    return None


# Similarity search components (lazy loaded)
_similarity_index: Any = None
_similarity_metadata: list[dict] | None = None
_similarity_model: Any = None


def _load_similarity_index():
    """Lazy load the similarity index for augmented MATH problems."""
    global _similarity_index, _similarity_metadata, _similarity_model

    if _similarity_index is not None:
        return

    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        index_path = Path(__file__).parent.parent.parent / "data" / "math_problem_index"
        if not index_path.exists():
            logger.warning(f"Similarity index not found at {index_path}")
            return

        # Load FAISS index
        _similarity_index = faiss.read_index(str(index_path / "index.faiss"))

        # Load metadata
        _similarity_metadata = []
        with open(index_path / "metadata.jsonl", "r") as f:
            for line in f:
                _similarity_metadata.append(json.loads(line))

        # Load embedding model
        config_path = index_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                model_name = config.get("embedding_model", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        else:
            model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

        _similarity_model = SentenceTransformer(model_name)

        logger.info(f"Loaded similarity index with {_similarity_index.ntotal} problems")
    except Exception as e:
        logger.warning(f"Failed to load similarity index: {e}")
        _similarity_index = None
        _similarity_metadata = None
        _similarity_model = None


def find_similar_math_problem(problem: str, threshold: float = 0.7) -> tuple[int | None, str | None]:
    """
    Find similar original MATH problem for an augmented problem.
    Returns (complexity, level) tuple - complexity is int or None, level is str or None.
    """
    global _similarity_index, _similarity_metadata, _similarity_model

    if _similarity_index is None:
        _load_similarity_index()

    if _similarity_index is None or _similarity_metadata is None or _similarity_model is None:
        return None, None

    try:
        # Encode the problem
        import numpy as np
        import faiss

        query = _similarity_model.encode([problem], convert_to_numpy=True)
        faiss.normalize_L2(query)

        # Search
        scores, indices = _similarity_index.search(query, k=1)

        if scores[0][0] >= threshold:
            idx = indices[0][0]
            if idx >= 0 and idx < len(_similarity_metadata):
                meta = _similarity_metadata[idx]
                return meta.get("complexity"), meta.get("level")
    except Exception as e:
        logger.warning(f"Similarity search failed: {e}")

    return None, None


def classify_complexity(example: dict, avg_token_length: float | None = None) -> tuple[int, str | None]:
    """
    Canonical decision flow:
    1. Exact match: same problem text already classified → reuse
    2. GSM8K: always C=0 (immediate; no further heuristics)
    3. MATH with level: level-based classification
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
        level_str = _normalize_level(example.get("level"))
        if level_str in _VALID_MATH_LEVELS:
            if level_str in ("1", "2"):
                return 0, level_str
            if level_str in ("4", "5"):
                return 1, level_str
            # Level 3: similarity search first, then token fallback
            if level_str == "3" and problem:
                complexity, matched_level = find_similar_math_problem(problem, threshold=0.7)
                if complexity is not None:
                    return complexity, matched_level
            # No similar match → token fallback using HARD_TOKEN_THRESHOLD
            tokens = avg_token_length if avg_token_length is not None else _get_teacher_token_count(example)
            return (1 if tokens > HARD_TOKEN_THRESHOLD else 0), level_str

    # 3. Augmented MATH - use similarity search
    if source == "augmented_math" and problem:
        complexity, matched_level = find_similar_math_problem(problem, threshold=0.7)
        if complexity is not None:
            return complexity, matched_level

    # 4. Unknown source: token fallback only for truly unknown sources
    tokens = avg_token_length if avg_token_length is not None else _get_teacher_token_count(example)
    if tokens > HARD_TOKEN_THRESHOLD:
        return 1, None
    return 0, None  # Default Easy


def label_preference(example: dict, complexity: int) -> tuple[str, int | None]:
    """
    Returns "preferred" or "rejected" (witt rejection reason) for this solution.
    Uses Qwen tokenizer and same thresholds (70/130) as classify_complexity.
    """
    correct = _verify_correctness(example)
    tokens = _get_teacher_token_count(example)

    if not correct:
        return "rejected", REJECTION_REASONS["incorrect"]

    if complexity == 0:  # Easy
        if tokens <= EASY_TOKEN_THRESHOLD:
            return "preferred", None
        return "rejected", REJECTION_REASONS["length"]

    # Hard
    if tokens >= HARD_TOKEN_THRESHOLD:
        return "preferred", None
    return "rejected", REJECTION_REASONS["length"]


def build_dpo_pairs(raw_data: list[dict]) -> list[dict]:
    """
    Group by problem and build preferred/rejected pairs.
    Returns list of pairs with: problem, chosen, rejected, complexity, rejection_reason, chosen_length, rejected_length, problem_id.

    Token counts are stored for training-time filtering by length_ratio.
    problem_id is a unique integer per unique problem for stratified split.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in tqdm(raw_data, desc="Classifying & labeling", unit=" examples"):
        c, _ = classify_complexity(ex)
        label, rejection_reason = label_preference(ex, c)
        tc = _get_teacher_token_count(ex)
        groups[ex["problem"]].append({**ex, "complexity": c, "label": label, "rejection_reason": rejection_reason, "_token_count": tc})

    # Assign unique problem_id to each unique problem
    unique_problems = list(groups.keys())
    problem_to_id = {prob: idx for idx, prob in enumerate(unique_problems)}

    print(f'{len(unique_problems) = }')
    print(f'{len(problem_to_id) = }')
    print(f'{max(problem_to_id.values()) = }')
    skipped = 0

    pairs: list[dict] = []

    for (problem, c), items in tqdm(groups.items(), desc="Building pairs from groups", unit=" groups"):
        preferred, rejected = [], []
        for x in items:
            (preferred if x["label"] == "preferred" else rejected).append(x)

        problem_id = problem_to_id.get(problem, 0)
        complexity = items[0]["complexity"]

        if preferred and rejected:
            for pw in preferred:
                for rj in rejected:
                    pairs.append({
                        "problem": problem,
                        "problem_id": problem_id,
                        "chosen": pw["generated_solution"],
                        "rejected": rj["generated_solution"],
                        "complexity": complexity,
                        "rejection_reason": rj["rejection_reason"],
                        "chosen_length": pw["_token_count"],
                        "rejected_length": rj["_token_count"],
                    })
        else:
            skipped += 1

    print(f'{skipped = }')
    print(f'{len(pairs) = }')

    logger.info(f'Created a total of {len(pairs)} pairs')

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
    pairs: list[dict] | dict,
    val_split: float,
    seed: int = 42,
    filtered_indices: list[int] | None = None,
) -> tuple[list[int], list[int]] | tuple[list[dict], list[dict]]:
    """
    Split pairs into train/val by unique problem to prevent data leakage.
    Ensures the same problem doesn't appear in both sets.

    Stratifies by problem-level complexity (majority complexity among pairs for each problem).
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    set_seed(seed)


    if isinstance(pairs, dict):
        if filtered_indices is None:
            filtered_indices = np.arange(len(pairs)).tolist()
        problem_ids = pairs["problem_ids"].numpy()[filtered_indices]
        complexities = pairs["complexities"].numpy()[filtered_indices]
    else: # TODO - deprecate the list[dict] options
        problem_ids, complexities = [], []
        for i, pair in enumerate(pairs):
            if filtered_indices and i not in filtered_indices:
                continue

            problem_ids.append(pair["problem_ids"])
            complexities.append(pair["complexities"])

        problem_ids = np.array(problem_ids)
        complexities = np.array(complexities)

        if filtered_indices is None:
                filtered_indices = np.arange(len(pairs)).tolist()

    # Get unique problems
    unique_problems = np.unique(problem_ids)

    # Build problem -> complexity mapping (pick first sample's complexity)
    problem_to_complexity = {}
    for pid in unique_problems:
        first_idx = np.where(problem_ids == pid)[0][0]
        problem_to_complexity[pid] = complexities[first_idx]

    problem_complexities = np.array([problem_to_complexity[p] for p in unique_problems])

    # Stratified split
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
        if pid in unique_val_problem_ids:
            if isinstance(pairs, dict):
                val_indices.append(filtered_indices[i])
            else: # TODO - deprecate
                val_indices.append(pairs[filtered_indices[i]]) # NOT indices, but the actual pairs (temporary)
        else:
            if isinstance(pairs, dict):
                train_indices.append(filtered_indices[i])
            else: # TODO - deprecate
                train_indices.append(pairs[filtered_indices[i]]) # NOT indices, but the actual pairs (temporary)

    return train_indices, val_indices


def compute_pair_length_ratio(pair: dict[str, Any]) -> float | None:
    rejection_reason = pair['rejection_reason']
    if rejection_reason != REJECTION_REASONS['length']:
        return None

    complexity = pair["complexity"]
    chosen_length = pair["chosen_length"]
    rejected_length = pair["rejected_length"]

    if complexity == 0:
        if chosen_length > 0:
            return rejected_length / chosen_length
    else:
        if rejected_length > 0:
            return chosen_length / rejected_length
    return 0.0


def compute_statistics(
    pairs: list[dict],
) -> dict[str, Any]:
    """Compute full statistics per spec (Section 4), including length ratio histogram."""
    total = len(pairs)

    if total == 0:
        return {
            "easy_token_threshold": EASY_TOKEN_THRESHOLD,
            "hard_token_threshold": HARD_TOKEN_THRESHOLD,
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
                ratios_easy.append(compute_pair_length_ratio(p))
            else:
                count_correct_hard += 1
                chosen_length_sum_hard += chosen_length
                rejected_length_sum_hard += rejected_length
                ratios_hard.append(compute_pair_length_ratio(p))
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
        "easy_token_threshold": EASY_TOKEN_THRESHOLD,
        "hard_token_threshold": HARD_TOKEN_THRESHOLD,
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
