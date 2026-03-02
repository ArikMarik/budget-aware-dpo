"""
Run evaluation on model checkpoints: accuracy, TPCA, token counts.
Supports dummy data (processed DPO dataset) and real data (Phase 9).
"""

import json
from pathlib import Path
from typing import Optional

import torch

from src.config import CHECKPOINT_DIR, MODEL_NAME, get_processed_dataset_path
from src.evaluation.answer_extraction import extract_answer, normalize_answer
from src.utils import set_seed

set_seed(42)


def load_eval_problems(limit: Optional[int] = None) -> list[dict]:
    """Load unique problems with expected answer from chosen."""
    path = get_processed_dataset_path() / "dataset.jsonl"
    seen = {}
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            key = p["problem"]
            if key not in seen:
                # Extract expected from chosen ("The answer is 8." -> "8")
                exp = extract_answer(p["chosen"])
                seen[key] = {"problem": p["problem"], "expected": exp, "complexity": p["complexity"]}
            if limit and len(seen) >= limit:
                break
    return list(seen.values())


def generate_and_evaluate(
    model,
    tokenizer,
    problems: list[dict],
    max_new_tokens: int = 256,
) -> dict:
    """Generate for each problem, extract answer, compute metrics."""
    device = next(model.parameters()).device
    results = []
    for p in problems:
        prompt = f"Problem: {p['problem']}\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        num_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        pred = extract_answer(response)
        correct = (
            normalize_answer(pred) == normalize_answer(p["expected"])
            if p["expected"] else None
        )
        results.append({
            "problem": p["problem"][:60] + "...",
            "complexity": p["complexity"],
            "tokens": num_tokens,
            "predicted": pred,
            "expected": p["expected"],
            "correct": correct,
        })
    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy, TPCA, avg tokens by complexity."""
    with_expected = [r for r in results if r["expected"] is not None]
    correct = [r for r in with_expected if r["correct"]]
    total_tokens = sum(r["tokens"] for r in results)
    easy = [r for r in results if r["complexity"] == 0]
    hard = [r for r in results if r["complexity"] == 1]

    accuracy = len(correct) / len(with_expected) if with_expected else 0
    tpca = total_tokens / len(correct) if correct else float("inf")

    return {
        "accuracy": accuracy,
        "num_correct": len(correct),
        "num_total": len(with_expected),
        "tpca": tpca,
        "total_tokens": total_tokens,
        "avg_tokens_easy": sum(r["tokens"] for r in easy) / len(easy) if easy else 0,
        "avg_tokens_hard": sum(r["tokens"] for r in hard) / len(hard) if hard else 0,
        "num_easy": len(easy),
        "num_hard": len(hard),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    problems: list[dict],
    output_path: Optional[Path] = None,
) -> dict:
    """Load model, run evaluation, return metrics."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_path))
    model.eval()

    results = generate_and_evaluate(model, tokenizer, problems)
    metrics = compute_metrics(results)

    out = {"metrics": metrics, "results": results}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate results for JSON (full responses can be long)
        out_save = {"metrics": metrics, "results": results}
        with open(output_path, "w") as f:
            json.dump(out_save, f, indent=2)
    return metrics
