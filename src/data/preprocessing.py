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
import logging
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.evaluation.answer_extraction import extract_answer, normalize_answer
from src.utils import count_tokens_tiktoken, set_seed

set_seed(42)

logger = logging.getLogger(__name__)

# Configurable thresholds (env or defaults); tiktoken cl100k_base
EASY_TOKEN_THRESHOLD = int(os.environ.get("EASY_TOKEN_THRESHOLD", "70"))
HARD_TOKEN_THRESHOLD = int(os.environ.get("HARD_TOKEN_THRESHOLD", "130"))

_VALID_MATH_LEVELS = {"1", "2", "3", "4", "5"}


def _get_teacher_token_count(example: dict) -> int:
    """Get teacher token count; compute from generated_solution if missing."""
    tc = example.get("teacher_token_count")
    if tc is not None and tc != 0:
        return int(tc)
    sol = example.get("generated_solution", "") or ""
    return count_tokens_tiktoken(sol)


def _verify_correctness(example: dict) -> bool:
    """Ensure correctness_flag is set. Compute from expected_answer if missing."""
    if "correctness_flag" in example:
        return bool(example["correctness_flag"])
    expected = example.get("expected_answer", "")
    if not expected or not str(expected).strip():
        return False
    pred = extract_answer(example.get("generated_solution", "")) # TODO - verify this
    return pred is not None and normalize_answer(pred) == normalize_answer(str(expected))


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
        # Level missing or invalid: use token fallback
        tokens = _get_teacher_token_count(example)
        if tokens < EASY_TOKEN_THRESHOLD:
            return 0
        if tokens > HARD_TOKEN_THRESHOLD:
            return 1
        return 0  # Ambiguous medium → default Easy

    # 3. UNKNOWN SOURCE — token heuristic only
    tokens = _get_teacher_token_count(example)
    if tokens < EASY_TOKEN_THRESHOLD:
        return 0
    if tokens > HARD_TOKEN_THRESHOLD:
        return 1
    return 0  # Default Easy if ambiguous


def label_preference(example: dict, complexity: int) -> str:
    """
    Returns "preferred" or "rejected" for this solution.
    Uses tiktoken and same thresholds (70/130) as classify_complexity.
    """
    correct = _verify_correctness(example)
    tokens = _get_teacher_token_count(example)

    if not correct:
        return "rejected"

    if complexity == 0:  # Easy
        if tokens <= EASY_TOKEN_THRESHOLD:
            return "preferred"
        return "rejected"

    # Hard
    if tokens >= HARD_TOKEN_THRESHOLD:
        return "preferred"
    return "rejected"


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


def _rejection_reason(rejected_item: dict, expected: str) -> str:
    """Determine why rejected: 'correctness' (wrong answer) or 'length' (correct but wrong length)."""
    if not _verify_correctness(rejected_item):
        return "correctness"
    return "length"


def build_dpo_pairs(raw_data: list[dict]) -> tuple[list[dict], list[dict], int]:
    """
    Group by (problem, complexity) and build preferred/rejected pairs.
    Returns (real_pairs, synthesized_pairs, skipped_groups).
    Each pair has: problem, chosen, rejected, complexity, rejection_reason.
    """
    from collections import defaultdict

    raw_iter = tqdm(raw_data, desc="Classifying & labeling", unit=" examples")
    raw_iter = raw_data

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for ex in raw_iter:
        c = classify_complexity(ex)
        label = label_preference(ex, c)
        groups[(ex["problem"], c)].append({**ex, "complexity": c, "label": label})

    real_pairs: list[dict] = []
    synthesized_pairs: list[dict] = []
    skipped_groups = 0

    groups_iter = tqdm(groups.items(), desc="Building pairs from groups", unit=" groups")

    for (problem, complexity), items in groups_iter:
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
                    reason = _rejection_reason(rj, expected)
                    if reason == "correctness":
                        # Replace with synthesized length-based pair
                        short = _make_short_answer(pw["generated_solution"], expected)
                        if complexity == 0:
                            synthesized_pairs.append({
                                "problem": problem,
                                "chosen": short,
                                "rejected": _make_verbose_answer(short),
                                "complexity": 0,
                                "rejection_reason": "length",
                            })
                        else:
                            # Hard: preferred = long (use actual preferred), rejected = short
                            synthesized_pairs.append({
                                "problem": problem,
                                "chosen": pw["generated_solution"],
                                "rejected": short,
                                "complexity": 1,
                                "rejection_reason": "length",
                            })
                    else:
                        real_pairs.append({
                            "problem": problem,
                            "chosen": pw["generated_solution"],
                            "rejected": rj["generated_solution"],
                            "complexity": complexity,
                            "rejection_reason": reason,
                        })
            continue

        # Synthetic: preferred-only
        if preferred and not rejected:
            for ex in preferred:
                sol = ex["generated_solution"]
                exp = ex.get("expected_answer", expected)
                if not _verify_correctness(ex):
                    continue  # Should not happen for preferred
                if complexity == 0:
                    # Easy: short preferred, synthesize verbose rejected
                    short = _make_short_answer(sol, exp)
                    verbose = _make_verbose_answer(short)
                    synthesized_pairs.append({
                        "problem": problem,
                        "chosen": short,
                        "rejected": verbose,
                        "complexity": 0,
                        "rejection_reason": "length",
                    })
                else:
                    # Hard: long preferred, synthesize short rejected
                    short = _make_short_answer(sol, exp)
                    synthesized_pairs.append({
                        "problem": problem,
                        "chosen": sol,
                        "rejected": short,
                        "complexity": 1,
                        "rejection_reason": "length",
                    })
            continue

        # Synthetic: rejected-only — synthesize minimal correct as preferred
        # Complexity=0: preferred = short; complexity=1: preferred = long (synthesize CoT)
        if rejected and not preferred:
            if not expected or not str(expected).strip():
                skipped_groups += 1
                continue
            if complexity == 0:
                preferred_synth = _make_short_answer("", expected)
            else:
                preferred_synth = _make_long_reasoning("", expected, problem)
            for rj in rejected:
                synthesized_pairs.append({
                    "problem": problem,
                    "chosen": preferred_synth,
                    "rejected": rj["generated_solution"],
                    "complexity": complexity,
                    "rejection_reason": "correctness",
                })
            continue

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
