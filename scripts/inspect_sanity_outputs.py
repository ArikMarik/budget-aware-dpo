#!/usr/bin/env python3
"""
Inspect sanity-check model outputs: Easy prompts should yield compressed tokens;
Hard prompts should yield full chain-of-thought.
"""

import json
from pathlib import Path

from src.config import CHECKPOINT_DIR, MODEL_NAME, PROCESSED_DATASET_PATH
from src.utils import set_seed

set_seed(42)


def load_sample_pairs(limit_easy: int = 3, limit_hard: int = 3):
    """Load Easy and Hard pairs from processed dataset."""
    path = PROCESSED_DATASET_PATH / "dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    easy, hard = [], []
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            if p["complexity"] == 0 and len(easy) < limit_easy:
                easy.append(p)
            elif p["complexity"] == 1 and len(hard) < limit_hard:
                hard.append(p)
            if len(easy) >= limit_easy and len(hard) >= limit_hard:
                break
    return easy, hard


def main():
    checkpoint_dir = CHECKPOINT_DIR / "sanity_overfit"
    if not checkpoint_dir.exists():
        print(f"Checkpoint not found: {checkpoint_dir}")
        print("Run train_sanity_check.py first.")
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    model.eval()

    easy_pairs, hard_pairs = load_sample_pairs()
    results = {"easy": [], "hard": []}

    def generate(problem: str, complexity: int, max_new_tokens: int = 256):
        prompt = f"Problem: {problem}\nSolution:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        num_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        return response.strip(), num_tokens

    print("\n--- Easy prompts (expected: compressed, ~20-40 tokens) ---")
    for p in easy_pairs:
        resp, n = generate(p["problem"], 0)
        results["easy"].append({"problem": p["problem"][:80] + "...", "tokens": n, "response": resp[:200]})
        print(f"  Tokens: {n} | Response: {resp[:120]}...")

    print("\n--- Hard prompts (expected: full CoT, 100+ tokens) ---")
    for p in hard_pairs:
        resp, n = generate(p["problem"], 1)
        results["hard"].append({"problem": p["problem"][:80] + "...", "tokens": n, "response": resp[:200]})
        print(f"  Tokens: {n} | Response: {resp[:120]}...")

    avg_easy = sum(r["tokens"] for r in results["easy"]) / max(len(results["easy"]), 1)
    avg_hard = sum(r["tokens"] for r in results["hard"]) / max(len(results["hard"]), 1)
    print(f"\nAvg tokens Easy: {avg_easy:.1f} | Avg tokens Hard: {avg_hard:.1f}")

    out_path = CHECKPOINT_DIR / "sanity_inspection_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
