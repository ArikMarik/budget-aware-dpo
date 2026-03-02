#!/usr/bin/env python3
"""
Load Qwen-2.5-0.5B and run a forward pass on a single dummy example.
Uses Unsloth when available, otherwise falls back to transformers.
"""

import os
from pathlib import Path

from src.utils import set_seed

set_seed(42)

# Route model cache to persistent storage if set
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HOME", HF_HOME)


def main():
    dummy_path = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parent.parent / "data")) / "dummy_openmathinstruct.jsonl"
    if not dummy_path.exists():
        raise FileNotFoundError(f"Run generate_dummy_data.py first. Expected: {dummy_path}")

    import json
    with open(dummy_path, "r") as f:
        example = json.loads(f.readline())

    prompt = f"Problem: {example['problem']}\nSolution:"
    print(f"Prompt (truncated): {prompt[:80]}...")

    try:
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen2.5-0.5B",
            max_seq_length=512,
            dtype=None,  # auto
            load_in_4bit=False,
        )
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        print(f"Forward pass OK. Logits shape: {logits.shape}")
    except ImportError:
        # Fallback to transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32)
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"Forward pass OK (transformers). Logits shape: {outputs.logits.shape}")

    print("Model loading check passed.")


if __name__ == "__main__":
    main()
