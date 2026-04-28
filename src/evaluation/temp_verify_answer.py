import re
import json
import random
import argparse
from pathlib import Path

from tqdm import tqdm

from src.evaluation.answer_extraction import extract_answer
from src.qwen_evaluation.grader import math_equal
from src.qwen_evaluation.parser import strip_string

DATASET_PATH = Path("/storage/arik/nlp_final_project/data/openmathinstruct.jsonl")


# -----------------------
# Dataset loading
# -----------------------

def load_openmath_samples(n: int, random_sample: bool = True) -> list[dict]:
    """Load n samples from openmathinstruct.jsonl.

    Args:
        n: Number of samples to load
        random_sample: If True, sample randomly across the dataset.
                       If False, load the first n samples.

    Returns:
        List of dicts with keys: problem, generated_solution, expected_answer
    """
    total_lines = 13_972_791

    if random_sample:
        random.seed(42)
        offsets = sorted(random.sample(range(total_lines), n))
    else:
        offsets = list(range(n))


    offsets += [43, 45, 708, 867, 888, 1466, 1467, 1616, 1861, 3055, 3139, 3513]

    samples = []
    with open(DATASET_PATH) as f:
        for i, line in tqdm(enumerate(f), total=max(offsets)):
            if len(samples) >= n:
                break
            if random_sample and i not in offsets:
                continue

            data = json.loads(line)
            samples.append({
                "problem": data.get("problem", ""),
                "generated_solution": data.get("generated_solution", ""),
                "expected_answer": data.get("expected_answer", ""),
                "problem_source": data.get("problem_source", ""),
                "level": data.get("level", ""),
            })

    return samples


# -----------------------
# Verification runner
# -----------------------

def run_verification(samples: list[dict], save_to_file: bool = False) -> dict:
    """Run verification on samples.

    Args:
        samples: List of dicts with problem, generated_solution, expected_answer
        save_to_file: If True, save results to JSON file

    Returns:
        Dict with keys: total, passed, failed, accuracy, failures
    """
    results = {
        "total": len(samples),
        "passed": 0,
        "failed": 0,
        "failures": [],
    }

    for i, sample in tqdm(enumerate(samples), total=len(samples), desc='Verification'):
        expected = str(sample.get("expected_answer", ""))
        generated = sample.get("generated_solution", "")

        pred = extract_answer(generated)

        if pred is None:
            results["failed"] += 1
            results["failures"].append({
                "index": i,
                **sample,
                "problem": sample.get("problem", "")[:200],
                "expected": expected,
                "predicted": None,
                "reason": "no_answer_extracted",
            })
            continue

        if math_equal(strip_string(pred), strip_string(expected)):
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "index": i,
                **sample,
                "problem": sample.get("problem", "")[:200],
                "expected": expected,
                "stripped_expected": strip_string(expected),
                "predicted": pred,
                "stripped_predicted": strip_string(pred),
                "reason": "mismatch",
            })

    results["accuracy"] = results["passed"] / results["total"] if results["total"] > 0 else 0

    if save_to_file:
        output_path = Path("/storage/arik/nlp_final_project/eval_results/verify_sample_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


# -----------------------
# Old test examples
# -----------------------

examples = [
    ("306958.05", "306,956.63"),
    ("\\text{Softball, Kickball, Picnic}", "Softball,Kickball,Picnic"),
    ("1 - \\cos^2t", "\\sin^2t"),
    ("10.196", "3\\sqrt{3}+5"),
]


# -----------------------
# Main CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Test answer verification on openmath-instruct2")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=True,
                      help="Sample randomly vs sequentially")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False,
                      help="Save results to JSON file")
    parser.add_argument("--test-mode", action="store_true",
                      help="Run old test examples instead of dataset")
    args = parser.parse_args()

    if args.test_mode:
        print("Running test examples (--test-mode):\n")
        for i, (a, b) in enumerate(examples, 1):
            result = equivalent(a, b)
            print(f"Example {i}:")
            print(f"  A = {a} ({strip_string(a)})")
            print(f"  B = {b} ({strip_string(b)})")
            print(f"  Equivalent? {'YES' if result else 'NO'}\n")
        return

    print(f"Loading {args.samples} samples from openmathinstruct.jsonl (random={args.random})...")
    samples = load_openmath_samples(args.samples, random_sample=args.random)
    print(f"Loaded {len(samples)} samples\n")

    print("Running verification...")
    results = run_verification(samples, save_to_file=args.save)

    print(f"\n{'='*50}")
    print(f"Results: {results['passed']}/{results['total']} ({results['accuracy']:.1%})")
    print(f"Passed: {results['passed']}, Failed: {results['failed']}")

    if results["failures"]:
        print(f"\n{'='*50}")
        print(f"Failed cases ({len(results['failures'])}):")
        for failure in results["failures"][:10]:
            print(f"\n  [{failure['index']}] {failure['reason']}")
            print(failure)

        if len(results["failures"]) > 10:
            print(f"\n  ... and {len(results['failures']) - 10} more failures")


if __name__ == "__main__":
    main()