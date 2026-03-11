"""
Shared DPO training logic: data loading, forward pass, checkpointing.
"""

import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

from src.config import CHECKPOINT_DIR, MODEL_NAME, get_processed_dataset_path
from src.utils import set_seed


def log_prob(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute sum of log probs for the sequence."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    return torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1).sum(-1)


class DPODataset(Dataset):
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_pairs(limit: Optional[int] = None) -> list[dict]:
    pairs = []
    path = get_processed_dataset_path() / "dataset.jsonl"
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
            if limit and len(pairs) >= limit:
                break
    return pairs


def collate_fn(batch: list[dict], tokenizer, max_length: int = 512):
    """Collate batch: prompt, chosen, rejected, complexity."""
    prompts, chosens, rejecteds, complexities = [], [], [], []
    for p in batch:
        prompt = f"Problem: {p['problem']}\nSolution:"
        chosen = prompt + " " + p["chosen"]
        rejected = prompt + " " + p["rejected"]
        prompts.append(prompt)
        chosens.append(chosen)
        rejecteds.append(rejected)
        complexities.append(p["complexity"])
    chosen_tok = tokenizer(
        chosens,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    rejected_tok = tokenizer(
        rejecteds,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    complexities_t = torch.tensor(complexities, dtype=torch.long)
    return chosen_tok, rejected_tok, complexities_t


def train_dpo(
    *,
    use_budget_aware: bool,
    output_dir: Path,
    max_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 1e-5,
    checkpoint_every: int = 500,
    eval_every: int = 100,
    data_limit: Optional[int] = None,
    resume_from: Optional[str] = None,
    seed: int = 42,
    use_wandb: bool = False,
) -> dict:
    """
    Train DPO (baseline or budget-aware). Returns metrics dict.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = load_pairs(limit=data_limit)
    if not pairs:
        raise RuntimeError("No DPO pairs found. Run preprocess_dpo_data.py first.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if resume_from:
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)

    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    dataset = DPODataset(pairs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    if use_budget_aware:
        from src.models.budget_aware_dpo_loss import budget_aware_dpo_loss
        def loss_fn(pc, pr, rc, rr, cl, rl, c):
            return budget_aware_dpo_loss(pc, pr, rc, rr, cl, rl, c, beta=0.1, lambda_easy=0.05, lambda_hard=0.001)
    else:
        from src.models.standard_dpo_loss import standard_dpo_loss
        def loss_fn(pc, pr, rc, rr, cl, rl, c):
            return standard_dpo_loss(pc, pr, rc, rr, beta=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    config = {
        "use_budget_aware": use_budget_aware,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "data_limit": data_limit,
        "num_pairs": len(pairs),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # W&B init (only when use_wandb=True)
    wandb_run = None
    if use_wandb:
        import wandb
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "budget-aware-dpo"),
            name=os.environ.get("WANDB_RUN_NAME"),
            config=config,
            mode=wandb_mode,
        )
        wandb_run = wandb.run

    metrics_log = []
    step = 0
    epoch = 0
    while step < max_steps:
        epoch_loss = 0.0
        for batch in dataloader:
            if step >= max_steps:
                break
            chosen_tok, rejected_tok, complexities = batch
            chosen_tok = {k: v.to(device) for k, v in chosen_tok.items()}
            rejected_tok = {k: v.to(device) for k, v in rejected_tok.items()}
            complexities = complexities.to(device)

            with torch.no_grad():
                ref_chosen = ref_model(**chosen_tok).logits
                ref_rejected = ref_model(**rejected_tok).logits
            policy_chosen = model(**chosen_tok).logits
            policy_rejected = model(**rejected_tok).logits

            policy_chosen_lp = log_prob(policy_chosen, chosen_tok["input_ids"])
            policy_rejected_lp = log_prob(policy_rejected, rejected_tok["input_ids"])
            ref_chosen_lp = log_prob(ref_chosen, chosen_tok["input_ids"])
            ref_rejected_lp = log_prob(ref_rejected, rejected_tok["input_ids"])

            pad_id = tokenizer.pad_token_id or 0
            chosen_lens_t = (chosen_tok["input_ids"] != pad_id).sum(dim=-1).float()
            rejected_lens_t = (rejected_tok["input_ids"] != pad_id).sum(dim=-1).float()
            avg_chosen_tokens = chosen_lens_t.mean().item()
            avg_rejected_tokens = rejected_lens_t.mean().item()

            loss, extra = loss_fn(
                policy_chosen_lp,
                policy_rejected_lp,
                ref_chosen_lp,
                ref_rejected_lp,
                chosen_lens_t,
                rejected_lens_t,
                complexities,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if step % eval_every == 0:
                entry = {"step": step, "loss": loss.item()}
                if extra:
                    entry.update(extra)
                metrics_log.append(entry)
                print(f"Step {step} loss: {loss.item():.4f}")

                # W&B logging
                if use_wandb and wandb_run:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/step": step,
                        "train/avg_chosen_tokens": avg_chosen_tokens,
                        "train/avg_rejected_tokens": avg_rejected_tokens,
                        "train/token_diff": avg_chosen_tokens - avg_rejected_tokens,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    if extra and "length_penalty" in extra:
                        log_dict["train/length_penalty"] = extra["length_penalty"]
                    import wandb
                    wandb.log(log_dict, step=step)

            if step % checkpoint_every == 0:
                ckpt_path = output_dir / f"checkpoint-{step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                with open(output_dir / "metrics.json", "w") as f:
                    json.dump(metrics_log, f, indent=2)
                print(f"Saved checkpoint to {ckpt_path}")

        epoch += 1
        avg = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} avg loss: {avg:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Training complete. Saved to {output_dir}")
    return {"metrics": metrics_log, "config": config}
