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
    def string_cleanup(s):
        # Pre-return cleanup + strip_string normalization (unconditional)
        s = re.sub(r"\n\s*", "", s).strip()
        if s and s[0] == ":":
            s = s[1:]
        if s and s[-1] in [".", "/"]:
            s = s[:-1]
        return s

    # Cyrillic artifact strip (some model outputs contain these)
    text = text.replace("ки", "")
    text = text.strip()
    if not text:
        return None

    # \boxed{...} — handles nested braces, uses last occurrence
    ans = extract_boxed_answer(text)
    if ans is not None:
        return string_cleanup(ans)

    # #### 8 (GSM8K format)
    ans = extract_gsm8k_answer(text)
    if ans:
        return string_cleanup(ans)

    # "The answer is X" or "the answer is X"
    m = re.search(r"[Tt]he answer is\s*[:=]?\s*([^\s.,;]+)", text, re.IGNORECASE)
    if m:
        return string_cleanup(m.group(1).strip())

    # Last number fallback
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return string_cleanup(numbers[-1])
    return None


def normalize_answer(a: str | None) -> str:
    """Normalize for comparison: lowercase, strip whitespace, normalize LaTeX formatting."""
    if a is None:
        return ""
    s = str(a).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\%", "%")
    return s


def verify_correctness(
    generated_solution: str,
    expected_answer: str,
    logs: bool = True,
) -> bool:
    """Verify if generated_solution matches expected_answer."""
    if not expected_answer or not str(expected_answer).strip():
        return False
    pred = extract_answer(generated_solution)
    if pred is None:
        if logs:
            logger.info(f"No answer found in generated solution: {generated_solution}")
        return False
    is_correct = verify_answer(pred, expected_answer)
    if logs and not is_correct:
        logger.info(f"Incorrect answer: {pred} != {expected_answer} (expected)")
    return is_correct
