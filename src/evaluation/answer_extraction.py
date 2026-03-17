"""
Answer extraction for GSM8K/MATH-style outputs.
Handles: "The answer is X", "#### X", "\\boxed{X}", trailing numbers.
"""

import re
from src.evaluation.math_grader import verify_answer
from src.utils import get_logger


logger = get_logger(__name__)


def extract_boxed_answer(text: str) -> str | None:
    """Extract content of the last \\boxed{...}, handling nested braces."""
    # Find the last occurrence of \boxed{
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None  # unmatched braces
    return text[start : i - 1].strip()


def extract_gsm8k_answer(answer: str) -> str:
    """Extract final answer from GSM8K format (#### N)."""
    m = re.search(r"####\s*(\S+)", answer)
    return m.group(1).strip() if m else ""


def extract_answer(text: str) -> str | None:
    """Extract final answer from model output. Returns None if not found."""
    text = text.strip()
    if not text:
        logger.info(f"No answer found in text: {text}")
        return None

    # \boxed{...} — handles nested braces, uses last occurrence
    ans = extract_boxed_answer(text)
    if ans is not None:
        return ans

    # #### 8 (GSM8K format)
    ans = extract_gsm8k_answer(text)
    if ans:
        return ans
    
    # "The answer is 8." or "The answer is 8"
    m = re.search(r"[Tt]he answer is\s*[:=]?\s*([^\s.,;]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Last number in sentence (fallback)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    logger.info(f"No answer found in text: {text}")
    return None


def normalize_answer(a: str | None) -> str:
    """Normalize for comparison: lowercase, strip whitespace, normalize LaTeX formatting."""
    if a is None:
        return ""
    s = str(a).strip().lower()
    # Remove all internal whitespace (handles \frac{289 \pi} vs \frac{289\pi})
    s = re.sub(r"\s+", "", s)
    # Normalize \% → % (handles 56\% vs 56%)
    s = s.replace("\\%", "%")
    return s


def verify_correctness(generated_solution: str, expected_answer: str, problem: str = "") -> bool:
    """Verify if generated_solution matches expected_answer using tiered checking.

    Tier 1: math-verify symbolic equivalence (handles LaTeX format differences).
    Tier 2: LLM judge with full solution context (handles base notation, etc.).
    Returns False when expected_answer is empty (cannot verify).
    """
    if not expected_answer or not str(expected_answer).strip():
        return False
    pred = extract_answer(generated_solution)
    if pred is None:
        logger.info(f"No answer found in generated solution: {generated_solution}")
        return False
    is_correct = verify_answer(
        pred=pred,
        expected=str(expected_answer),
        problem=problem,
        solution=generated_solution,
    )
    if not is_correct:
        logger.info(f"Incorrect answer: {pred} != {expected_answer}")
    return is_correct
