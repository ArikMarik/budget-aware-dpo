from src.evaluation.math_grader import remove_spaces, strip_text_commands, verify_answer
from src.qwen_evaluation.parser import strip_string


def show_all_permutations(pred, expected) -> None:
    no_text_pred = strip_text_commands(pred)
    no_text_expected = strip_text_commands(expected)

    no_space_pred = remove_spaces(pred)
    no_space_expected = remove_spaces(expected)

    stripped_no_space_pred = remove_spaces(strip_string(pred))
    stripped_no_space_expected = remove_spaces(strip_string(expected))

    stripped_no_text_pred = remove_spaces(strip_string(strip_text_commands(pred)))
    stripped_no_text_expected = remove_spaces(strip_string(strip_text_commands(expected)))

    print(f"Original:")
    print(f"pred={pred:>30}")
    print(f"expected={expected:>26}")
    print(pred == expected)
    print("-" * 20)
    print(f"No text:")
    print(f"pred={no_text_pred:>30}")
    print(f"expected={no_text_expected:>26}")
    print(no_text_pred == no_text_expected)
    print("-" * 20)
    print(f"No space:")
    print(f"pred={no_space_pred:>30}")
    print(f"expected={no_space_expected:>26}")
    print(no_space_pred == no_space_expected)
    print(f"Stripped no space:")
    print(f"pred={stripped_no_space_pred:>30}")
    print(f"expected={stripped_no_space_expected:>26}")
    print(stripped_no_space_pred == stripped_no_space_expected)
    print("-" * 20)
    print(f"Stripped no text:")
    print(f"pred={stripped_no_text_pred:>30}")
    print(f"expected={stripped_no_text_expected:>26}")
    print(stripped_no_text_pred == stripped_no_text_expected)
    print("-" * 20)
    print(verify_answer(pred, expected))


if __name__ == "__main__":
    pred = r"-\frac{5}{3}"
    expected = r"-\frac{5}{3}."

    pred = r"-15\frac{3}{5}"
    expected = r"-15+3/5"

    pred = r"\$306,\!960.29"
    expected = r"306,956.63"

    pred = r"\42,\!409"
    expected = r"42409"

    pred = r"-5/3"
    expected = r"-\frac{5}{3}."

    pred = r"\text{C.}"
    expected = r"C"

    pred = r"100\,000"
    expected = r"100000"

    show_all_permutations(pred, expected)
