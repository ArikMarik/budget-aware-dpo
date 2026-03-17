"""
Tiered math answer verification.

Tier 1 — math-verify (ANTLR4 + SymPy): symbolic equivalence, handles LaTeX
         format differences (\dfrac vs \frac, whitespace, \begin{pmatrix}, etc.).
         Returns True | False | None (None = parse/timeout failure).

Tier 2 — Local LLM judge: fallback for cases Tier 1 can't handle, e.g. base
         notation (1242_6 vs 1242), uncommon LaTeX macros, trig identities.
         Receives the full solution for context so it can reason about notation.
"""
from __future__ import annotations

import functools
import os

from src.utils import get_logger

logger = get_logger(__name__)

# Configurable via env; defaults to the instruct variant of the project model
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# Max solution chars fed to the LLM (avoid runaway prompt length)
_MAX_SOLUTION_CHARS = 2000


# ---------------------------------------------------------------------------
# Tier 1: symbolic via math-verify
# ---------------------------------------------------------------------------

def _wrap(s: str) -> str:
    """Wrap in $...$ so math-verify's parser recognises it as a math expression."""
    return f"${s}$"


def _verify_symbolic(pred: str, expected: str) -> bool | None:
    """
    Returns True/False on definitive result, None if math-verify cannot parse
    or times out (caller should fall through to Tier 2).
    """
    try:
        from math_verify import parse, verify  # type: ignore[import]
        result = verify(parse(_wrap(expected)), parse(_wrap(pred)))
        return bool(result)
    except Exception as exc:
        logger.debug("math-verify inconclusive for %r vs %r: %s", pred, expected, exc)
        return None


# ---------------------------------------------------------------------------
# Tier 2: LLM judge
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_llm_judge():
    """Lazy-load the judge model once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading LLM judge: %s", LLM_JUDGE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_JUDGE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_JUDGE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("LLM judge loaded.")
    return model, tokenizer


def _verify_llm(problem: str, solution: str, pred: str, expected: str) -> bool:
    """
    Ask the LLM whether pred and expected represent the same answer, given the
    full solution context (critical for e.g. base-N notation, implicit units).
    """
    import torch

    model, tokenizer = _load_llm_judge()

    # Truncate solution to keep prompt manageable
    if len(solution) > _MAX_SOLUTION_CHARS:
        half = _MAX_SOLUTION_CHARS // 2
        solution = solution[:half] + "\n...[truncated]...\n" + solution[-half:]

    prompt = (
        "You are a math teacher checking whether a student's answer matches the expected answer.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Student's full solution:\n{solution}\n\n"
        f"Student's final answer: {pred}\n"
        f"Expected answer: {expected}\n\n"
        "Do these represent the same value? Use the full solution for context "
        "(e.g. the student may express the answer in a different base or notation "
        "that is equivalent to the expected answer).\n"
        "Reply with exactly one word: yes or no."
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    is_correct = response.startswith("yes")
    logger.info(
        "LLM judge: pred=%r expected=%r → %r (correct=%s)", pred, expected, response, is_correct
    )
    return is_correct


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def verify_answer(
    pred: str | None,
    expected: str,
    problem: str = "",
    solution: str = "",
) -> bool:
    """
    Tiered verification:
      Tier 1 — math-verify symbolic check (fast, no model load).
               True  → accept immediately.
               False → fall through to Tier 2 (may be a notation issue).
               None  → parse failure, fall through to Tier 2.
      Tier 2 — LLM judge with full solution context.

    Args:
        pred:     Extracted predicted answer string.
        expected: Ground-truth answer string.
        problem:  Original problem text (context for LLM judge).
        solution: Full generated solution text (context for LLM judge).
    """
    if pred is None:
        logger.info(f"Predicted answer is None: {pred}")
        return False
    if not expected or not str(expected).strip():
        logger.info(f"Expected answer is empty: {expected}")
        return False

    # Tier 1
    symbolic_result = _verify_symbolic(pred, expected)
    if symbolic_result is True:
        return True

    # Tier 2 (handles False AND None from Tier 1)
    logger.info(
        "Tier 1 result=%s for pred=%r vs expected=%r — invoking LLM judge",
        symbolic_result, pred, expected,
    )
    llm_result = _verify_llm(problem, solution, pred, expected)
    return llm_result
