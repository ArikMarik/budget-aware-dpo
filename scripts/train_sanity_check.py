#!/usr/bin/env python3
"""
Sanity check: Overfit on 10-50 dummy examples to verify Budget-Aware DPO loss works.
Uses high-rank LoRA (r=128) for training stability.
"""

import json
import os
from pathlib import Path

from src.config import CHECKPOINT_DIR, MODEL_NAME, PROCESSED_DATASET_PATH
from src.utils import set_seed

set_seed(42)


def load_pairs(limit: int = 30):
    pairs = []
    path = PROCESSED_DATASET_PATH / "dataset.jsonl"
    with open(path) as f:
        for line in f:
            if len(pairs) >= limit:
                break
            pairs.append(json.loads(line))
    return pairs


def main():
    pairs = load_pairs(limit=30)
    if not pairs:
        raise RuntimeError("No DPO pairs found. Run preprocess_dpo_data.py first.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"Missing dependency: {e}. Install: pip install torch transformers peft")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    # LoRA r=128 for stability
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.train()

    # Reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Format: prompt = problem, chosen/rejected = full solution
    def tokenize_pair(p):
        prompt = f"Problem: {p['problem']}\nSolution:"
        chosen = prompt + " " + p["chosen"]
        rejected = prompt + " " + p["rejected"]
        return prompt, chosen, rejected, p["complexity"]

    from src.models.budget_aware_dpo_loss import budget_aware_dpo_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 3

    for epoch in range(num_epochs):
        total_loss = 0.0
        for p in pairs:
            prompt, chosen, rejected, complexity = tokenize_pair(p)

            chosen_tok = tokenizer(chosen, return_tensors="pt", truncation=True, max_length=512).to(device)
            rejected_tok = tokenizer(rejected, return_tensors="pt", truncation=True, max_length=512).to(device)

            with torch.no_grad():
                ref_chosen = ref_model(**chosen_tok).logits
                ref_rejected = ref_model(**rejected_tok).logits

            policy_chosen = model(**chosen_tok).logits
            policy_rejected = model(**rejected_tok).logits

            # Get log probs (simplified: use last token or mean over sequence)
            def log_prob(logits, input_ids):
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                return torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1).sum(-1)

            policy_chosen_lp = log_prob(policy_chosen, chosen_tok["input_ids"])
            policy_rejected_lp = log_prob(policy_rejected, rejected_tok["input_ids"])
            ref_chosen_lp = log_prob(ref_chosen, chosen_tok["input_ids"])
            ref_rejected_lp = log_prob(ref_rejected, rejected_tok["input_ids"])

            chosen_len = chosen_tok["input_ids"].shape[1]
            rejected_len = rejected_tok["input_ids"].shape[1]
            complexities = torch.tensor([complexity], device=device, dtype=torch.long)
            chosen_lens = torch.tensor([chosen_len], device=device)
            rejected_lens = torch.tensor([rejected_len], device=device)

            loss, metrics = budget_aware_dpo_loss(
                policy_chosen_lp,
                policy_rejected_lp,
                ref_chosen_lp,
                ref_rejected_lp,
                chosen_lens,
                rejected_lens,
                complexities,
                beta=0.1,
                lambda_easy=0.05,
                lambda_hard=0.001,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f"Epoch {epoch+1} loss: {total_loss/len(pairs):.4f}")

    # Save checkpoint
    out_dir = CHECKPOINT_DIR / "sanity_overfit"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved sanity check checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
