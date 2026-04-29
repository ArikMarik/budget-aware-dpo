import re

from math_verify import parse, verify

from src.qwen_evaluation.grader import math_equal
from src.qwen_evaluation.parser import strip_string
from src.utils import get_logger, setup_global_exception_handler


logger = get_logger(__name__)
setup_global_exception_handler(__name__)


def strip_text_commands(s: str) -> str:
    """
    Remove LaTeX text-style commands (e.g., \text, \textbf, \textit)
    by unwrapping their { ... } content. Supports nested braces,
    optional arguments [ ... ], and preserves other LaTeX commands.
    """
    result = []
    i = 0
    n = len(s)

    while i < n:
        if s[i] == '\\':
            j = i + 1

            # Read command name
            while j < n and s[j].isalpha():
                j += 1

            command = s[i+1:j]

            # Only process text-like commands
            if command.startswith("text") or command in ["overline", "underline"]:
                k = j

                # Skip optional argument [ ... ] if it exists
                if k < n and s[k] == '[':
                    bracket_depth = 1
                    k += 1
                    while k < n and bracket_depth > 0:
                        if s[k] == '[':
                            bracket_depth += 1
                        elif s[k] == ']':
                            bracket_depth -= 1
                        elif s[k] == '\\' and k + 1 < n:
                            k += 1  # skip escaped char
                        k += 1

                # Now expect {
                if k < n and s[k] == '{':
                    i = k + 1
                    brace_depth = 1
                    content = []

                    while i < n and brace_depth > 0:
                        char = s[i]

                        # Handle escaped chars like \{ \}
                        if char == '\\' and i + 1 < n:
                            content.append(s[i])
                            content.append(s[i + 1])
                            i += 2
                            continue

                        elif char == '{':
                            brace_depth += 1
                        elif char == '}':
                            brace_depth -= 1
                            if brace_depth == 0:
                                i += 1
                                break

                        if brace_depth > 0:
                            content.append(char)

                        i += 1

                    result.append(strip_text_commands(''.join(content)))
                    continue

        # Default: copy character
        result.append(s[i])
        i += 1

    return ''.join(result)


def _wrap(s: str) -> str:
    """Wrap in $...$ so math-verify's parser recognises it as a math expression."""
    return f"${s}$"


def remove_spaces(s: str) -> str:
    return re.sub(r'\s+', '', s)


def verify_answer(
    pred: str | None,
    expected: str,
) -> bool:
    if pred is None:
        return False

    no_space_pred = remove_spaces(pred)
    no_space_expected = remove_spaces(expected)

    if math_equal(no_space_pred, no_space_expected):
        return True

    stripped_no_space_pred = remove_spaces(strip_string(pred))
    stripped_no_space_expected = remove_spaces(strip_string(expected))

    if math_equal(stripped_no_space_pred, stripped_no_space_expected):
        return True

    no_text_pred = strip_text_commands(pred)
    no_text_expected = strip_text_commands(expected)

    if math_equal(remove_spaces(no_text_pred), remove_spaces(no_text_expected)) or\
        math_equal(no_space_pred, stripped_no_space_expected) or \
            math_equal(stripped_no_space_pred, no_space_expected) or \
                math_equal(strip_string(no_space_pred), strip_string(no_space_expected)):
        return True

    stripped_no_text_pred = remove_spaces(strip_string(strip_text_commands(pred)))
    stripped_no_text_expected = remove_spaces(strip_string(strip_text_commands(expected)))

    if  math_equal(stripped_no_text_pred, stripped_no_text_expected):
        return True

    try:
        return verify(parse(_wrap(no_text_expected)), parse(_wrap(no_text_pred)), timeout_seconds=2)
    except Exception as exc:
        logger.debug("math-verify inconclusive for %r vs %r: %s", pred, expected, exc)
    return False
