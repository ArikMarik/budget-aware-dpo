#!/usr/bin/env python3
"""
Test script to verify count_tokens_batch returns the EXACT SAME values as count_tokens.
Tests both synthetic cases and real data from OpenMathInstruct-2.
Run: python scripts/test_tokenization_equivalence.py
"""

import sys
import time

from tqdm import tqdm

from src.utils import count_tokens, _get_model_tokenizer
from src.data.worker_utils import count_tokens_batch


def test_single_item_equivalence(tokenizer):
    """Test 1: Verify batch returns same as individual for single items."""
    test_cases = [
        "",
        "hello",
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "1234567890",
        "#### 42",
        "\\boxed{42}",
        "\\frac{1}{2}",
        "\\sqrt{x^2 + y^2}",
        "x + y = z",
        "\n\t\r",
        "   spaces   ",
        "üñîçødé",
        "π ≈ 3.14159",
        "∫_{0}^{1} x^2 dx",
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        "0",
        " " * 100,
        "αβγδεζηθικλμνξοπρστυφχψω",
        "What is the value of x if 2x + 5 = 15?",
        "Step 1: Add 5 to both sides. Step 2: Divide by 2. Step 3: The answer is 5.",
        "```\ndef hello():\n    return \"world\"\n```",
        "<function call>",
        " " + "word " * 100,
        "Étôu͔i̮s̲n͖g",
    ]

    print(f"Test 1: Single-item equivalence ({len(test_cases)} cases)")
    failures = []

    for i, text in enumerate(test_cases):
        old_result = count_tokens(str(text), tokenizer)
        new_result = count_tokens_batch([str(text)], tokenizer)[0]

        if old_result != new_result:
            failures.append({
                "index": i,
                "text": text[:50],
                "old": old_result,
                "new": new_result,
            })

    if failures:
        print(f"  FAILED: {len(failures)}/{len(test_cases)} cases")
        for f in failures:
            print(f"    [{f['index']}] '{f['text']}' old={f['old']} new={f['new']}")
        return False

    print(f"  PASSED: All {len(test_cases)} cases")
    return True


def test_batch_processing(tokenizer):
    """Test 2: Verify batch results sum equals sum of individual results."""
    print(f"\nTest 2: Batch processing verification")

    batch_size = 500
    texts = [f"test sentence number {i} with some more words to make it longer" for i in range(batch_size)]

    individual_start = time.time()
    individual_results = [count_tokens(t, tokenizer) for t in tqdm(texts, desc="  Individual")]
    individual_time = time.time() - individual_start

    batch_start = time.time()
    batch_results = count_tokens_batch(texts, tokenizer)
    batch_time = time.time() - batch_start

    sum_individual = sum(individual_results)
    sum_batch = sum(batch_results)

    all_match = individual_results == batch_results

    print(f"  Batch size: {batch_size}")
    print(f"  Individual: {individual_time:.2f}s, sum={sum_individual}")
    print(f"  Batch:      {batch_time:.2f}s, sum={sum_batch}")
    print(f"  Speedup:   {individual_time/batch_time:.1f}x")

    if all_match:
        print(f"  PASSED: All {batch_size} batch results match individual")
        return True
    else:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(individual_results, batch_results)) if a != b]
        print(f"  FAILED: {len(mismatches)} mismatches")
        for idx, old, new in mismatches[:5]:
            print(f"    [{idx}] old={old}, new={new}")
        return False


def test_real_dataset(tokenizer):
    """Test 3: Compare on real solutions from OpenMathInstruct-2."""
    print(f"\nTest 3: Real dataset (OpenMathInstruct-2 train_1M)")

    from datasets import load_dataset

    cache_path = '/root/.cache/huggingface/datasets'
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True, cache_dir=cache_path)

    num_samples = 10_000
    solutions = []

    print(f"  Loading {num_samples} solutions...")
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        solutions.append(item["generated_solution"])

    print(f"  Computing individual token counts...")
    individual_start = time.time()
    individual_results = [count_tokens(s, tokenizer) for s in tqdm(solutions, desc="  Individual")]
    individual_time = time.time() - individual_start

    print(f"  Computing batch token counts...")
    batch_start = time.time()
    batch_results = count_tokens_batch(solutions, tokenizer)
    batch_time = time.time() - batch_start

    mismatches = [(i, s[:50], old, new) for i, (s, old, new) in enumerate(zip(solutions, individual_results, batch_results)) if old != new]

    sum_individual = sum(individual_results)
    sum_batch = sum(batch_results)

    all_match = individual_results == batch_results

    print(f"  Samples: {len(solutions)}")
    print(f"  Individual: {individual_time:.2f}s, sum={sum_individual}")
    print(f"  Batch:      {batch_time:.2f}s, sum={sum_batch}")
    print(f"  Speedup:   {individual_time/batch_time:.1f}x")

    if all_match:
        print(f"  PASSED: All {len(solutions)} real solutions match")
        return True
    else:
        print(f"  FAILED: {len(mismatches)} mismatches")
        for idx, text, old, new in mismatches[:5]:
            print(f"    [{idx}] text='{text}...' old={old}, new={new}")
        return False


def main():
    print("=" * 60)
    print("Tokenization Equivalence Tests")
    print("=" * 60)

    tokenizer = _get_model_tokenizer()

    results = []

    results.append(("Single-item equivalence", test_single_item_equivalence(tokenizer)))
    results.append(("Batch processing", test_batch_processing(tokenizer)))
    results.append(("Real dataset", test_real_dataset(tokenizer)))

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    passed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1

    print(f"\n{passed}/{len(results)} tests passed")

    if passed == len(results):
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()