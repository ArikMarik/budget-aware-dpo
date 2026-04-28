"""
Run evaluation on model checkpoints: accuracy, TPCA, token counts.
Supports dummy data (processed DPO dataset) and real data (Phase 9: GSM8K, MATH).
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm

from src.config import (
    GSM8K_TEST_PATH,
    MATH_TEST_PATH,
    MODEL_NAME,
    get_processed_dataset_path,
)
from src.data.preprocessing import classify_complexity
from src.evaluation.answer_extraction import extract_answer, verify_correctness
from src.evaluation.few_shot_exemplars import build_zero_shot_prompt
from src.utils import count_tokens, set_seed

set_seed(42)


def load_eval_problems(limit: Optional[int] = None, use_real: bool = False) -> list[dict]:
    """Load evaluation problems. If use_real, load from GSM8K+MATH test sets; else from processed DPO dataset."""
    if use_real:
        return load_eval_problems_real(limit=limit)
    path = get_processed_dataset_path() / "dataset.jsonl"
    seen = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            key = p["problem"]
            if key not in seen:
                exp = extract_answer(p["chosen"])
                seen[key] = {"problem": p["problem"], "expected": exp, "complexity": p["complexity"]}
            if limit and len(seen) >= limit:
                break
    return list(seen.values())


def load_eval_problems_real(limit: Optional[int] = None) -> list[dict]:
    """Load GSM8K and MATH test sets for Phase 9 evaluation. Run load_real_data.py first."""
    problems = []
    if GSM8K_TEST_PATH.exists():
        with open(GSM8K_TEST_PATH) as f:
            for line in f:
                p = json.loads(line)
                problems.append({
                    "problem": p["problem"],
                    "expected": p.get("expected_answer", ""),
                    "complexity": classify_complexity(p)[0],
                    "source": "gsm8k",
                    "level": None,
                })
    else:
        raise FileNotFoundError(
            f"GSM8K test set not found at {GSM8K_TEST_PATH}. "
            "Run: python scripts/load_real_data.py (without --skip-test-sets)"
        )
    if MATH_TEST_PATH.exists():
        with open(MATH_TEST_PATH) as f:
            for line in f:
                p = json.loads(line)
                level = p.get("level", "")
                problems.append({
                    "problem": p["problem"],
                    "expected": p.get("expected_answer", ""),
                    "complexity": classify_complexity(p)[0],
                    "source": "math",
                    "level": level,
                })
    else:
        raise FileNotFoundError(
            f"MATH test set not found at {MATH_TEST_PATH}. "
            "Run: python scripts/load_real_data.py (without --skip-test-sets)"
        )
    if limit:
        problems = problems[:limit]
    return problems


def _process_result(args: tuple) -> dict:
    """Worker function for parallel post-processing (extract answer + verify correctness)."""
    idx, response, num_tokens, expected, complexity, level, source, problem_text = args
    pred = extract_answer(response)
    correct = verify_correctness(response, expected, logs=False)
    return {
        "idx": idx,
        "problem": problem_text[:60] + " ...",
        "complexity": complexity,
        "predicted": pred,
        "expected": expected,
        "correct": correct,
        "level": level,
        "source": source,
        "tokens": num_tokens,
    }


def _generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
) -> list[tuple[str, int]]:
    """Generate responses for a batch of prompts. Returns list of (response, num_tokens)."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[151645, 151643],
        )

    results = []
    for out in outputs:
        prompt_len = input_ids.shape[1]
        response = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        response = response.split("\n\nQuestion:")[0]
        num_tokens = count_tokens(response, tokenizer)
        results.append((response, num_tokens))
    return results


def generate_and_evaluate(
    model,
    tokenizer,
    problems: list[dict],
    max_new_tokens: int = 1024,
    prompt_fn: Optional[Callable] = None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> list[dict]:
    """Generate for each problem, extract answer, compute metrics.

    Uses batched generation for GPU parallelism and parallel post-processing
    for CPU-bound tasks (answer extraction + verification).
    """
    if prompt_fn is None:
        prompt_fn = build_zero_shot_prompt
    device = next(model.parameters()).device

    prompts = [
        prompt_fn(p["problem"], p.get("source", p.get("problem_source")))
        for p in problems
    ]

    all_responses = []
    all_num_tokens = []

    model.eval()
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + batch_size]
        batch_results = _generate_batch(model, tokenizer, batch_prompts, max_new_tokens, device)
        for response, num_tokens in batch_results:
            all_responses.append(response)
            all_num_tokens.append(num_tokens)

    post_process_args = [
        (idx, all_responses[idx], all_num_tokens[idx], p["expected"], p["complexity"], p.get("level"), p.get("source"), p["problem"])
        for idx, p in enumerate(problems)
    ]

    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            processed_results = list(executor.map(_process_result, post_process_args))
        results = processed_results
    else:
        results = [_process_result(args) for args in post_process_args]

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, TPCA, avg tokens by complexity. MATH level 4-5 when available."""
    # MATH level 4-5 retention (Phase 9)
    def is_math_level_45(level) -> bool:
        s = str(level or "").strip()
        return s in ("4", "5", "Level 4", "Level 5")

    with_expected, correct, easy_results, hard_results = [],  [], [], []
    easy_correct, hard_correct = [], []
    total_tokens = 0
    math_by_level = {}
    math_45_with_exp, math_45_correct = [], []

    for r in results:
        if r.get("expected") is not None:
            with_expected.append(r)
            if r["correct"]:
                correct.append(r)
        total_tokens += r["tokens"]
        if r["complexity"] == 0:
            easy_results.append(r)
            if r["correct"]:
                easy_correct.append(r)
        elif r["complexity"] == 1:
            hard_results.append(r)
            if r["correct"]:
                hard_correct.append(r)

        level = r.get("level")
        if level is None:
            continue
        level_str = str(level).strip()
        if level_str not in math_by_level:
            math_by_level[level_str] = {"total": 0, "correct": 0}
        math_by_level[level_str]["total"] += 1
        if r["correct"]:
            math_by_level[level_str]["correct"] += 1

        if is_math_level_45(level):
            if r.get("expected") is not None:
                math_45_with_exp.append(r)
                if r["correct"]:
                    math_45_correct.append(r)

    accuracy = len(correct) / len(with_expected) if with_expected else 0
    tpca = total_tokens / len(correct) if correct else float("inf")

    for v in math_by_level.values():
        v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0

    out = {
        "accuracy": accuracy,
        "num_correct": len(correct),
        "num_total": len(with_expected),
        "tpca": tpca,
        "total_tokens": total_tokens,
        "avg_tokens_easy": sum(r["tokens"] for r in easy_results) / len(easy_results) if easy_results else 0,
        "avg_tokens_hard": sum(r["tokens"] for r in hard_results) / len(hard_results) if hard_results else 0,
        "num_easy": len(easy_results),
        "num_hard": len(hard_results),
        "num_easy_correct": len(easy_correct),
        "num_hard_correct": len(hard_correct),
        "easy_accuracy": len(easy_correct) / len(easy_results) if easy_results else 0,
        "hard_accuracy": len(hard_correct) / len(hard_results) if hard_results else 0,
        "math_by_level": math_by_level
    }

    if math_45_with_exp:
        out["math_level_4_5_accuracy"] = len(math_45_correct) / len(math_45_with_exp)
        out["math_level_4_5_num"] = len(math_45_with_exp)

    return out


def evaluate_checkpoint(
    checkpoint_path: Path,
    problems: list[dict],
    output_path: Optional[Path] = None,
    base_model: Optional[str] = None,
    prompt_fn: Optional[callable] = None,
) -> dict:
    """Load model, run evaluation, return metrics."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = base_model or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    model.eval()

    results = generate_and_evaluate(model, tokenizer, problems, prompt_fn=prompt_fn)
    metrics = compute_metrics(results)

    out = {"metrics": metrics, "results": results}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate results for JSON (full responses can be long)
        out_save = {"metrics": metrics, "results": results}
        with open(output_path, "w") as f:
            json.dump(out_save, f, indent=2)
    return metrics
