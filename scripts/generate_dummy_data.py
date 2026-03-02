#!/usr/bin/env python3
"""
Generate 50 synthetic examples mimicking OpenMathInstruct-2 format.
Mix of Easy (GSM8K-like) and Hard (MATH-like) problems with simulated
teacher_token_count and correctness_flag.
"""

import json
import os
from pathlib import Path

from src.utils import set_seed

set_seed(42)

# Data storage path - use env or fallback to local
DATA_ROOT = os.environ.get(
    "DATA_PATH",
    Path(__file__).resolve().parent.parent / "data"
)
DUMMY_PATH = Path(str(DATA_ROOT)) / "dummy_openmathinstruct.jsonl"


def generate_dummy_data() -> list[dict]:
    """Generate 50 synthetic examples in OpenMathInstruct-2 format."""
    examples = []

    # --- Easy (GSM8K-like) arithmetic - 25 examples ---
    easy_problems = [
        ("Janet has 3 apples. She buys 5 more. How many apples does she have?", "8", 15),
        ("A store has 12 shirts. They sell 4. How many remain?", "8", 12),
        ("Tom has 7 marbles. His friend gives him 6. Total marbles?", "13", 18),
        ("There are 9 birds. 3 fly away. How many stay?", "6", 14),
        ("Lisa bakes 4 cookies. She bakes 4 more. Total cookies?", "8", 16),
        ("A box has 15 pencils. 7 are used. How many left?", "8", 17),
        ("Mike has 20 candies. He eats 8. Remaining?", "12", 19),
        ("There are 6 dogs. 4 more arrive. How many dogs?", "10", 15),
        ("Sarah has 14 books. She gives 5 away. How many?", "9", 18),
        ("A garden has 11 flowers. 2 wilt. How many bloom?", "9", 16),
        ("Jake collects 8 stamps. He gets 9 more. Total?", "17", 20),
        ("A class has 18 students. 6 are absent. Present?", "12", 19),
        ("Emma has 5 dolls. She buys 7 more. Total dolls?", "12", 17),
        ("There are 16 balls. 9 are red. How many others?", "7", 18),
        ("Ben has 10 toys. He loses 3. Remaining?", "7", 15),
        ("A farm has 13 chickens. 5 are sold. Left?", "8", 16),
        ("Anna picks 6 oranges. She picks 8 more. Total?", "14", 19),
        ("There are 22 students. 9 leave. How many stay?", "13", 20),
        ("Carl has 11 coins. He finds 4 more. Total?", "15", 18),
        ("A basket has 19 eggs. 7 break. Good eggs?", "12", 19),
        ("Sue has 8 stickers. She gets 6 more. Total?", "14", 17),
        ("There are 24 trees. 10 are oak. Others?", "14", 20),
        ("Dan has 17 cards. He trades 8. Remaining?", "9", 18),
        ("A zoo has 21 monkeys. 6 move. How many stay?", "15", 20),
        ("Kate has 9 beads. She buys 11 more. Total?", "20", 21),
    ]

    for i, (problem, answer, short_tokens) in enumerate(easy_problems):
        # Correct short answer (Preferred for Easy)
        examples.append({
            "problem": problem,
            "generated_solution": f"The answer is {answer}.",
            "expected_answer": answer,
            "problem_source": "gsm8k",
            "teacher_token_count": short_tokens,
            "correctness_flag": True,
        })
        # Correct verbose answer (Rejected for Easy) - alternate
        if i % 2 == 0:
            verbose = f"Let me think step by step. First, we need to understand the problem. {problem} So we carefully add the numbers. The result is {answer}. Therefore the final answer is {answer}."
            examples.append({
                "problem": problem,
                "generated_solution": verbose,
                "expected_answer": answer,
                "problem_source": "gsm8k",
                "teacher_token_count": 85,
                "correctness_flag": True,
            })

    # --- Hard (MATH-like) proofs - 25 examples ---
    hard_problems = [
        ("Prove that the sum of the first n positive integers is n(n+1)/2.", "By induction: base n=1 gives 1. Assume true for n=k. For n=k+1, sum = k(k+1)/2 + (k+1) = (k+1)(k+2)/2. QED.", 120),
        ("Find the derivative of x^3 * ln(x).", "Using product rule: 3x^2*ln(x) + x^3*(1/x) = 3x^2*ln(x) + x^2.", 95),
        ("Solve the differential equation dy/dx = 2y with y(0)=1.", "Separating variables: dy/y=2dx. Integrating: ln|y|=2x+C. y(0)=1 gives C=0. So y=e^(2x).", 110),
        ("Prove sqrt(2) is irrational.", "Assume sqrt(2)=p/q in lowest terms. Then 2q^2=p^2, so p^2 even, hence p even. Write p=2m. Then 2q^2=4m^2, so q^2 even, q even. Contradiction.", 150),
        ("Evaluate the integral of x*e^x dx.", "Integration by parts: u=x, dv=e^x dx. Result: x*e^x - e^x + C.", 88),
    ]

    for problem, solution, tokens in hard_problems:
        examples.append({
            "problem": problem,
            "generated_solution": solution,
            "expected_answer": solution.split()[-1].rstrip(".") if solution else "",
            "problem_source": "math",
            "teacher_token_count": tokens,
            "correctness_flag": True,
        })
        # Incorrect answer (Rejected)
        examples.append({
            "problem": problem,
            "generated_solution": "The answer is 42.",
            "expected_answer": "N/A",
            "problem_source": "math",
            "teacher_token_count": 5,
            "correctness_flag": False,
        })

    # Pad to 50 if needed
    while len(examples) < 50:
        examples.append({
            "problem": "What is 2 + 2?",
            "generated_solution": "4",
            "expected_answer": "4",
            "problem_source": "gsm8k",
            "teacher_token_count": 3,
            "correctness_flag": True,
        })

    return examples[:50]


def main():
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    examples = generate_dummy_data()
    with open(DUMMY_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} dummy examples to {DUMMY_PATH}")


if __name__ == "__main__":
    main()
