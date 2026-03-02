"""
Answer extraction for GSM8K/MATH-style outputs.
Handles: "The answer is X", "#### X", "\\boxed{X}", trailing numbers.
"""

import re


def extract_answer(text: str) -> str | None:
    """Extract final answer from model output. Returns None if not found."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # "The answer is 8." or "The answer is 8"
    m = re.search(r"[Tt]he answer is\s*[:=]?\s*([^\s.,;]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # #### 8 (GSM8K format)
    m = re.search(r"####\s*(\S+)", text)
    if m:
        return m.group(1).strip()
    # \boxed{8} or \boxed{42}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # Last number in sentence (fallback)
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return None


def normalize_answer(a: str) -> str:
    """Normalize for comparison (lowercase, strip)."""
    if a is None:
        return ""
    return str(a).strip().lower()
