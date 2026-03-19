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
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger

logger = get_logger(__name__)

# Configurable via env; defaults to the instruct variant of the project model
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "Qwen/Qwen2.5-Math-7B-Instruct")


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
    # Note: Added @functools.lru_cache(maxsize=1) to ensure the model 
    # actually only loads once and caches in memory.
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
    model, tokenizer = _load_llm_judge()

    # 1. System Prompt: Explain rules and mandate Chain-of-Thought format
    system_prompt = (
        "You are an expert, strict math grader. Your task is to determine if a student's answer "
        "is mathematically equivalent to the expected answer.\n\n"
        "Rules for equivalence:\n"
        "1. STRICT NUMERICAL MATCH: Values must match exactly digit-for-digit. 3.50 is NOT 350. 99,999 is NOT 99997.\n"
        "2. ALGEBRAIC EQUIVALENCE: Expressions that simplify to the exact same value are equal (e.g. 1 - \\cos^2t equals \\sin^2t).\n"
        "3. COMPLEX NUMBERS & FRACTIONS: Different forms of the same value are equal (e.g. \\frac{-1 - 5i}{13} equals -\\frac{1}{13}-\\frac{5}{13}i).\n"
        "4. IGNORE FORMATTING: LaTeX tags like \\text{}, spaces, commas, and formatting do not matter.\n\n"
        "You MUST think step-by-step first. Then, on a new line, end your response with EXACTLY 'VERDICT: yes' or 'VERDICT: no'."
    )

    # 2. Few-Shot Chain-of-Thought Examples
    # We teach the model exactly how to reason through the edge cases it is failing on.
    messages =[
        {"role": "system", "content": system_prompt},
        
        {"role": "user", "content": "Student: 306958.05\nExpected: 306,956.63\nAre they equivalent?"},
        {"role": "assistant", "content": "The student has 306958.05. The expected is 306956.63. Comparing the digits, 958 is not equal to 956. They are mathematically different numbers.\nVERDICT: no"},
        
        {"role": "user", "content": "Student: 3.50\nExpected: 350\nAre they equivalent?"},
        {"role": "assistant", "content": "The student's answer is 3.50, which is a small decimal. The expected answer is 350, which is three hundred and fifty. 3.5 is not equal to 350.\nVERDICT: no"},
        
        {"role": "user", "content": "Student: \\text{Softball, Kickball, Picnic}\nExpected: Softball,Kickball,Picnic\nAre they equivalent?"},
        {"role": "assistant", "content": "The student's answer is wrapped in a \\text{} LaTeX tag and contains spaces. If we ignore the formatting and spaces, the lists are identical: Softball, Kickball, Picnic.\nVERDICT: yes"},
        
        {"role": "user", "content": "Student: 1 - \\cos^2t\nExpected: \\sin^2t\nAre they equivalent?"},
        {"role": "assistant", "content": "According to the Pythagorean trigonometric identity, 1 - \\cos^2t is algebraically identical to \\sin^2t. They represent the exact same value.\nVERDICT: yes"},
        
        {"role": "user", "content": "Student: \\frac{-1 - 5i}{13}\nExpected: -\\frac{1}{13}-\\frac{5}{13}i\nAre they equivalent?"},
        {"role": "assistant", "content": "The student's fraction can be split into two parts: -1/13 and -5i/13. This exactly matches the expected expanded complex number form.\nVERDICT: yes"},
        
        {"role": "user", "content": (
            f"Problem context:\n{problem}\n\n"
            f"Student: {pred}\n"
            f"Expected: {expected}\n"
            "Are they equivalent?"
        )}
    ]

    # Apply the chat template so the model receives distinct System, User, and Assistant boundary tokens
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150, # Give the model room to "think"
            temperature=0.0,   # Ensure strict greedy decoding
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    
    # Strip any stray punctuation before checking if it starts with 'yes'
    clean_response = re.sub(r'[^a-z]', '', response)
    is_correct = clean_response.startswith("yes")
    
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
