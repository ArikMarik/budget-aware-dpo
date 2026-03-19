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

from math_verify import parse, verify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import get_logger

logger = get_logger(__name__)

# Configurable via env; defaults to the instruct variant of the project model
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "Qwen/Qwen2.5-Math-7B-Instruct")
# Max solution chars fed to the LLM (avoid runaway prompt length)
_MAX_SOLUTION_CHARS = 4000


# ---------------------------------------------------------------------------
# Tier 0: exact equality
# ---------------------------------------------------------------------------
def is_trivially_equal(pred: str, expected: str) -> bool:
    # 1. Strip formatting and compare directly
    p_clean = pred.replace("\\text{", "").replace("}", "").replace(" ", "").replace(",", "").replace("\\$", "")
    e_clean = expected.replace(" ", "").replace(",", "").replace("\\$", "")
    if p_clean.lower() == e_clean.lower():
        return True
        
    # 2. Try checking if they are just unequal numbers (handles 3.50 vs 350)
    try:
        return abs(float(p_clean) - float(e_clean)) < 1e-6
    except ValueError:
        pass
        
    return False

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
        return verify(parse(_wrap(expected)), parse(_wrap(pred)))
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

    # Truncate strings to keep prompt manageable
    if len(solution) > _MAX_SOLUTION_CHARS:
        half = _MAX_SOLUTION_CHARS // 2
        solution = solution[:half] + "\n...[truncated]...\n" + solution[-half:]
        
    # System Instructions
    system_content = (
        "You are an expert math grading judge. Your ONLY task is to compare a student's extracted final answer against the expected correct answer and determine if they are mathematically equivalent.\n"
        "DO NOT grade or correct the student's step-by-step solution. The student's solution is ONLY provided so you can understand their notation, base, or units if their final answer is ambiguous.\n\n"
        "Rules for equivalence:\n"
        "1. STRICT NUMERICAL MATCH: Values must match exactly digit-for-digit. 3.50 is NOT 350.\n"
        "2. ALGEBRAIC EQUIVALENCE: Expressions that simplify to the exact same value are equal (e.g. 1 - \\cos^2t equals \\sin^2t).\n"
        "3. DECIMAL APPROXIMATIONS: If the student provides a correct decimal approximation of an exact expected value (e.g. 10.196 is equivalent to 3\\sqrt{3}+5), mark them as equivalent.\n"
        "4. IGNORE FORMATTING: LaTeX tags like \\text{}, spaces, commas, and currency symbols do not matter.\n\n"
        "You MUST think step-by-step for 1-2 sentences. Then, on a new line, end your response with EXACTLY 'VERDICT: yes' or 'VERDICT: no'."
    )

    # Strict Few-Shot Flow
    messages =[
        {"role": "system", "content": system_content},
        {"role": "user", "content": "Student's Final Answer: 306958.05\nExpected Answer: 306,956.63\n\nAre they equivalent?"},
        {"role": "assistant", "content": "Comparing the digits, 306958.05 is not equal to 306956.63. They are mathematically different numbers.\nVERDICT: no"},
        {"role": "user", "content": "Student's Final Answer: \\text{Softball, Kickball, Picnic}\nExpected Answer: Softball,Kickball,Picnic\n\nAre they equivalent?"},
        {"role": "assistant", "content": "The student's answer is wrapped in a \\text{} LaTeX tag and contains spaces. Ignoring the formatting and spaces, the lists are identical.\nVERDICT: yes"},
        {"role": "user", "content": "Student's Final Answer: 1 - \\cos^2t\nExpected Answer: \\sin^2t\n\nAre they equivalent?"},
        {"role": "assistant", "content": "According to the Pythagorean trigonometric identity, 1 - \\cos^2t is algebraically identical to \\sin^2t. They represent the exact same value.\nVERDICT: yes"},
        {"role": "user", "content": "Student's Final Answer: 10.196\nExpected Answer: 3\\sqrt{3}+5\n\nAre they equivalent?"},
        {"role": "assistant", "content": "The expected answer is 3\\sqrt{3}+5. Since \\sqrt{3} is approximately 1.732, 3(1.732) + 5 = 5.196 + 5 = 10.196. The student's decimal correctly approximates the expected exact value.\nVERDICT: yes"},
        
        # Final Query: Context clearly labeled as "Student's Full Solution" and isolated.
        {"role": "user", "content": (
            f"<context>\nProblem:\n{problem}\n\n"
            f"Student's Full Solution (for notation reference only):\n{solution}\n</context>\n\n"
            f"Student's Final Answer: {pred}\n"
            f"Expected Answer: {expected}\n\n"
            "Are they equivalent? DO NOT evaluate the student's full solution for correctness. ONLY compare the Student's Final Answer to the Expected Answer. Evaluate the equivalence step-by-step."
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
            max_new_tokens=500, # Give the model room to "think"
            temperature=0.0,   # Ensure strict greedy decoding
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
    
    # Robust Regex parsing to catch "Verdict: yes", "Verdiction: yes", "verdical: yes"
    match = re.search(r"(?i)verd[a-z]*:\s*(yes|no)", response)
    if match:
        is_correct = match.group(1).lower() == "yes"
    else:
        # Extreme Fallback: find the very last occurrence of 'yes' or 'no'
        words = re.findall(r"\b(yes|no)\b", response.lower())
        is_correct = (words[-1] == "yes") if words else False
    
    logger.info(
        "LLM judge: pred=%r expected=%r\n → %r\n(correct=%s)", pred, expected, response, is_correct
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
    
    # Tier 0
    if is_trivially_equal(pred, expected):
        return True

    # Tier 1
    if _verify_symbolic(pred, expected):
        return True

    # Tier 2
    return _verify_llm(problem, solution, pred, expected)
