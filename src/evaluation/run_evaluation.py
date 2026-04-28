"""
Run evaluation on model checkpoints: accuracy, TPCA, token counts.
Supports dummy data (processed DPO dataset) and real data (Phase 9: GSM8K, MATH).
"""

import json
from pathlib import Path
from typing import Callable, Optional

from sklearn import metrics
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
from src.utils import set_seed

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
                    "complexity": classify_complexity(p),
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
                    "complexity": classify_complexity(p),
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


def generate_and_evaluate(
    model,
    tokenizer,
    problems: list[dict],
    max_new_tokens: int = 2048,
    prompt_fn: Optional[Callable] = None,
) -> list[dict]:
    """Generate for each problem, extract answer, compute metrics."""
    if prompt_fn is None:
        prompt_fn = build_zero_shot_prompt
    device = next(model.parameters()).device
    results = []

    for problem in tqdm(problems, desc="Evaluating"):
        prompt = prompt_fn(problem["problem"], problem_source=problem.get("source", problem.get("problem_source")))

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[151645, 151643],
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        response = response.split("\n\nQuestion:")[0]
        num_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        pred = extract_answer(response)
        correct = verify_correctness(response, problem["expected"], logs=False)
        results.append({
            "problem": problem["problem"][:60] + " ...",
            "complexity": problem["complexity"],
            "tokens": num_tokens,
            "predicted": pred,
            "expected": problem["expected"],
            "correct": correct,
            "level": problem.get("level"),
            "source": problem.get("source"),
        })
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
