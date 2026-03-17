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

from src.evaluation.answer_extraction import extract_answer, normalize_answer, verify_correctness
from src.evaluation.math_grader import verify_answer
from src.utils import count_tokens, get_logger, set_seed

set_seed(42)

logger = get_logger(__name__)

# Configurable thresholds (env or defaults); Qwen tokenizer
EASY_TOKEN_THRESHOLD = int(os.environ.get("EASY_TOKEN_THRESHOLD", "70"))
HARD_TOKEN_THRESHOLD = int(os.environ.get("HARD_TOKEN_THRESHOLD", "130"))

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


def classify_complexity(example: dict) -> int:
    """
    Canonical decision flow (PRD §3.1):
    1. GSM8K: always C=0 (immediate; no further heuristics)
    2. MATH: level heuristic when available; else token fallback
    3. Unknown source: token heuristic only; default Easy if ambiguous
    """
    source = str(example.get("problem_source", "")).lower()

    # 1. SOURCE CHECK (GSM8K) — invariant: always Easy
    if "gsm" in source or "gsm8k" in source:
        return 0

    # 2. SOURCE CHECK (MATH)
    if "math" in source:
        level = example.get("level")
        level_str = str(level).strip() if level is not None else ""
        if level_str in _VALID_MATH_LEVELS:
            if level_str in ("1", "2"):
                return 0
            if level_str in ("4", "5"):
                return 1
            # Level 3: fall through to token fallback

    # Level missing or invalid: use token fallback / UNKNOWN SOURCE — token heuristic only
    tokens = _get_teacher_token_count(example)
    if tokens < EASY_TOKEN_THRESHOLD:
        return 0
    if tokens > HARD_TOKEN_THRESHOLD:
        return 1
    return 0  # Ambiguous medium → default Easy


def label_preference(example: dict, complexity: int) -> tuple[str, str | None]:
    """
    Returns "preferred" or "rejected" (witt rejection reason) for this solution.
    Uses Qwen tokenizer and same thresholds (70/130) as classify_complexity.
    """
    correct = _verify_correctness(example)
    tokens = _get_teacher_token_count(example)

    if not correct:
        return "rejected", "incorrect"

    if complexity == 0:  # Easy
        if tokens <= EASY_TOKEN_THRESHOLD:
            return "preferred", None
        return "rejected", "length"

    # Hard
    if tokens >= HARD_TOKEN_THRESHOLD:
        return "preferred", None
    return "rejected", "length"


def _make_short_answer(solution: str, expected: str = "") -> str:
    """
    Create short answer string. Use when solution is correct (or use expected for correct minimal).
    Only call when we have a correct solution or expected_answer for synthesizing preferred.
    """
    ans = extract_answer(solution) or (expected.strip() if expected else "")
    if ans:
        return f"The answer is {ans}."
    short_solution_length = 100
    if len(solution) <= short_solution_length:
        return solution
    return solution[: short_solution_length // 2] + " ... " + solution[-short_solution_length // 2 :]


def _make_verbose_answer(short_solution: str) -> str:
    """
    Create verbose (redundant) version from short solution. For Easy: rejected = verbose.
    Expands to 6-7 sentences with CoT indicators (first, then, later, therefore, etc.).
    """
    ans = extract_answer(short_solution) or ""
    if not ans and short_solution.strip():
        ans = short_solution.strip().rstrip(".")
    if not ans:
        return short_solution
    return (
        "Let me think step by step. "
        "First, I need to understand the problem. "
        "Then, I will work through the solution carefully. "
        "Later, I will verify each step. "
        "So we proceed methodically. "
        "Therefore, after considering all the details, "
        f"the answer is {ans}."
    )


def _make_long_reasoning(short_solution: str, expected: str, problem: str) -> str:
    """
    Synthesize long CoT-style reasoning from short answer. For Hard: preferred = long.
    Template-based expansion with 6-7 sentences and CoT indicators.
    """
    ans = extract_answer(short_solution) or (expected.strip() if expected else "")
    if not ans:
        return short_solution
    # Truncate problem for context if very long
    prob_snippet = (problem[:200] + "...") if len(problem) > 200 else problem
    return (
        "Let me work through this step by step. "
        f"First, we examine the problem: {prob_snippet} "
        "Then, we identify the key quantities and relationships. "
        "So we set up the necessary equations or reasoning. "
        "Therefore, after applying the appropriate method, "
        "we obtain the result. "
        f"Thus, the answer is {ans}."
    )


def build_dpo_pairs(raw_data: list[dict]) -> tuple[list[dict], list[dict], int]:
    """
    Group by (problem, complexity) and build preferred/rejected pairs.
    Returns (real_pairs, synthesized_pairs, skipped_groups).
    Each pair has: problem, chosen, rejected, complexity, rejection_reason.
    """
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for ex in tqdm(raw_data, desc="Classifying & labeling", unit=" examples"):
        c = classify_complexity(ex)
        label, rejection_reason = label_preference(ex, c)
        groups[(ex["problem"], c)].append({**ex, "complexity": c, "label": label, "rejection_reason": rejection_reason})

    real_pairs: list[dict] = []
    synthesized_pairs: list[dict] = []
    skipped_groups = 0

    for (problem, complexity), items in tqdm(groups.items(), desc="Building pairs from groups", unit=" groups"):
        preferred = [x for x in items if x["label"] == "preferred"]
        rejected = [x for x in items if x["label"] == "rejected"]
        expected = items[0].get("expected_answer", "") if items else ""

        # Natural pairs: both preferred and rejected exist
        # If rejection reason is correctness, the pair teaches correctness not length — useless for our goal.
        # Synthesize a length-based pair instead (complexity=0: short preferred, verbose rejected;
        # complexity=1: long preferred, short rejected).
        if preferred and rejected:
            for pw in preferred:
                for rj in rejected:
                    # if rj["rejection_reason"] == "incorrect":
                    #     logger.info(f'{"Encountered an INCORRECT answer":#^100}')
                    #     expected = rj.get("expected_answer", "").strip()
                    #     pred = extract_answer(rj.get("generated_solution", "")) # TODO - verify this
                    #     logger.info(f'Prediction: {normalize_answer(pred)}\n Expected: {normalize_answer(expected)}')

                    #     logger.info(f'{"Encountered an INCORRECT answer":#^100}')
                    #     # Replace with synthesized length-based pair
                    #     short = _make_short_answer(pw["generated_solution"], expected)
                    
                    #     synthesized_pairs.append({
                    #         "problem": problem,
                    #         "chosen": short if complexity == 0 else pw["generated_solution"],
                    #         "rejected": short if complexity == 1 else _make_verbose_answer(short),
                    #         "complexity": complexity,
                    #         "rejection_reason": rj["rejection_reason"],
                    #     })
                    # else:
                    
                    real_pairs.append({
                        "problem": problem,
                        "chosen": pw["generated_solution"],
                        "rejected": rj["generated_solution"],
                        "complexity": complexity,
                        "rejection_reason": rj["rejection_reason"],
                    })
            continue
        
        # # TODO - add this if we dont have enough real pairs
        # # Synthetic: preferred-only
        # if preferred and not rejected:
        #     for ex in preferred:
        #         sol = ex["generated_solution"] # TODO - this isn't the solution per answer - FIX IT!!!
        #         exp = ex.get("expected_answer", expected)
        #         synthesized_pairs.append({
        #             "problem": problem,
        #             "chosen": sol,
        #             "rejected": _make_verbose_answer(sol) if complexity == 0 else _make_short_answer(sol, exp),
        #             "complexity": complexity,
        #             "rejection_reason": "length",
        #         })

        # Synthetic: rejected-only — synthesize minimal correct as preferred
        # Complexity=0: preferred = short; complexity=1: preferred = long (synthesize CoT)
        # TODO - add this if we dont have enough real pairs
        # if rejected and not preferred:
        #     if not expected or not str(expected).strip():
        #         skipped_groups += 1
        #         continue
        #     for rj in rejected:
        #         if complexity == 0:
        #             preferred_synth = _make_short_answer("", expected)
        #         else:
        #             preferred_synth = _make_long_reasoning("", expected, problem)
        #         synthesized_pairs.append({
        #             "problem": problem,
        #             "chosen": preferred_synth,
        #             "rejected": rj["generated_solution"],
        #             "complexity": complexity,
        #             "rejection_reason": "correctness",
        #         })


    logger.info(f'{("Created a total of " + str(len(synthesized_pairs)) + " synthesized pairs"):#^100}')

    return real_pairs, synthesized_pairs, skipped_groups


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
    pairs: list[dict],
    val_split: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split pairs into train/val by unique problem to prevent data leakage.
    Ensures the same problem doesn't appear in both sets.

    Stratifies by problem-level complexity (majority complexity among pairs for each problem).
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    set_seed(seed)

    problem_to_pairs: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        problem_to_pairs[p["problem"]].append(p)

    unique_problems = list(problem_to_pairs.keys())
    problem_complexities = []
    for prob in unique_problems:
        prob_pairs = problem_to_pairs[prob]
        comps = [p.get("complexity", 0) for p in prob_pairs]
        problem_complexities.append(max(set(comps), key=comps.count) if comps else 0)

    problem_complexities = np.array(problem_complexities)
    # TODO - make sure problems are balanced enough
    train_problems, val_problems = train_test_split(
        unique_problems,
        test_size=val_split,
        stratify=problem_complexities,
        random_state=seed,
    )

    train_problems_set = set(train_problems)
    val_problems_set = set(val_problems)

    train_pairs = [p for p in pairs if p["problem"] in train_problems_set]
    val_pairs = [p for p in pairs if p["problem"] in val_problems_set]

    return train_pairs, val_pairs


def compute_statistics(
    real_pairs: list[dict],
    synthesized_pairs: list[dict],
    skipped_groups: int,
) -> dict[str, Any]:
    """Compute full statistics per spec (Section 4)."""
    all_pairs = real_pairs + synthesized_pairs
    total = len(all_pairs)

    if total == 0:
        return {
            "easy_token_threshold": EASY_TOKEN_THRESHOLD,
            "hard_token_threshold": HARD_TOKEN_THRESHOLD,
            "total_real_pairs": 0,
            "total_synthesized_pairs": 0,
            "total_pairs": 0,
            "skipped_groups": skipped_groups,
        }

    # Single pass over all_pairs with tqdm
    pairs_iter = tqdm(all_pairs, desc="Computing statistics", unit=" pairs")

    rej_correctness = 0
    rej_length = 0
    easy = 0
    hard = 0
    for p in pairs_iter:
        rr = p.get("rejection_reason")
        if rr == "correctness":
            rej_correctness += 1
        elif rr == "length":
            rej_length += 1
        c = p.get("complexity", 0)
        if c == 0:
            easy += 1
        else:
            hard += 1

    rej_correctness_real = sum(1 for p in real_pairs if p.get("rejection_reason") == "correctness")
    rej_correctness_synth = sum(1 for p in synthesized_pairs if p.get("rejection_reason") == "correctness")
    rej_length_real = sum(1 for p in real_pairs if p.get("rejection_reason") == "length")
    rej_length_synth = sum(1 for p in synthesized_pairs if p.get("rejection_reason") == "length")

    # Token stats (avg_preferred_tokens, avg_rejected_tokens) skipped for speed.
    # Run separately in parallel with training if needed.

    return {
        "easy_token_threshold": EASY_TOKEN_THRESHOLD,
        "hard_token_threshold": HARD_TOKEN_THRESHOLD,
        "total_real_pairs": len(real_pairs),
        "total_synthesized_pairs": len(synthesized_pairs),
        "total_pairs": total,
        "real_pairs_pct": round(100 * len(real_pairs) / total, 2),
        "synthesized_pairs_pct": round(100 * len(synthesized_pairs) / total, 2),
        "rejected_by_correctness": rej_correctness,
        "rejected_by_length": rej_length,
        "rejected_by_correctness_pct": round(100 * rej_correctness / total, 2),
        "rejected_by_length_pct": round(100 * rej_length / total, 2),
        "rejected_by_correctness_real": rej_correctness_real,
        "rejected_by_correctness_synthesized": rej_correctness_synth,
        "rejected_by_length_real": rej_length_real,
        "rejected_by_length_synthesized": rej_length_synth,
        "easy_pairs": easy,
        "hard_pairs": hard,
        "easy_pairs_pct": round(100 * easy / total, 2),
        "hard_pairs_pct": round(100 * hard / total, 2),
        "skipped_groups": skipped_groups,
    }
