#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) on chosen solutions.
Standard cross-entropy loss on solution tokens only (prompt is masked).

Usage:
  CUDA_VISIBLE_DEVICES=0 DATASET_PATH=data/processed_dpo_dataset_balanced_v4_capped \
    PYTHONUNBUFFERED=1 nohup .venv/bin/python -m scripts.training.train_sft \
    --output-dir checkpoints/sft_v1 --max-epochs 3 --batch-size 4 --lr 2e-5 \
    --run-name sft_v1 --wandb \
    > logs/sft_v1.log 2>&1 &
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import wandb

from src.config import MODEL_NAME, SEED, get_processed_dataset_path
from src.evaluation.run_evaluation import (
    compute_metrics,
    generate_and_evaluate,
    load_eval_problems,
)
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)


class SFTDataset(Dataset):
    """Dataset for SFT: tokenizes problem+solution, masks prompt tokens."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 512):
        self.examples = []
        for p in pairs:
            prompt = f"Problem: {p['problem']}\nSolution:"
            solution = " " + p["chosen"]
            full_text = prompt + solution

            # Tokenize full text
            full_enc = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)

            # Find where the prompt ends to create labels mask
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            # Labels: -100 for prompt tokens (masked), actual ids for solution tokens
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            # Also mask padding
            labels[attention_mask == 0] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def load_sft_pairs(split: str = "train") -> list[dict]:
    """Load problem+chosen pairs from the DPO dataset."""
    path = get_processed_dataset_path() / f"{split}.jsonl"
    pairs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pairs.append({"problem": d["problem"], "chosen": d["chosen"]})
    return pairs


def build_lora_config(rank: int = 128, alpha: int = 256, dropout: float = 0.05) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def run_gen_eval(model, tokenizer, epoch, step, use_wandb=False):
    """Quick generation eval (50 easy + 50 hard) for early detection."""
    import random
    problems = load_eval_problems(limit=None, use_real=True)
    easy = [p for p in problems if p["complexity"] == 0]
    hard = [p for p in problems if p["complexity"] == 1]
    random.seed(SEED)
    random.shuffle(easy)
    random.shuffle(hard)
    sample = easy[:50] + hard[:50]
    random.shuffle(sample)

    model.eval()
    results = generate_and_evaluate(model, tokenizer, sample, max_new_tokens=256)
    metrics = compute_metrics(results)

    easy_r = [r for r in results if r["complexity"] == 0]
    hard_r = [r for r in results if r["complexity"] == 1]
    easy_correct = sum(1 for r in easy_r if r["correct"])
    hard_correct = sum(1 for r in hard_r if r["correct"])

    logger.info(
        "Gen eval E%d: acc=%.1f%% (easy=%.1f%% [%d/50], hard=%.1f%% [%d/50]) | avg_tok_easy=%.0f, avg_tok_hard=%.0f | TPCA=%.0f",
        epoch,
        metrics["accuracy"] * 100,
        (easy_correct / max(len(easy_r), 1)) * 100, easy_correct,
        (hard_correct / max(len(hard_r), 1)) * 100, hard_correct,
        metrics["avg_tokens_easy"],
        metrics["avg_tokens_hard"],
        metrics["tpca"],
    )

    if use_wandb:
        wandb.log({
            "gen/accuracy": metrics["accuracy"],
            "gen/easy_accuracy": easy_correct / max(len(easy_r), 1),
            "gen/hard_accuracy": hard_correct / max(len(hard_r), 1),
            "gen/avg_tokens_easy": metrics["avg_tokens_easy"],
            "gen/avg_tokens_hard": metrics["avg_tokens_hard"],
            "gen/tpca": metrics["tpca"],
        }, step=step)

    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SFT training on chosen solutions")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--gen-eval-every", type=int, default=1, help="Run gen eval every N epochs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model or MODEL_NAME

    # Load data
    logger.info("Loading SFT data from %s", get_processed_dataset_path())
    train_pairs = load_sft_pairs("train")
    val_pairs = load_sft_pairs("val")
    logger.info("Train: %d examples, Val: %d examples", len(train_pairs), len(val_pairs))

    # Load model + LoRA
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    lora_config = build_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    # Create datasets
    logger.info("Tokenizing datasets (max_length=%d)...", args.max_length)
    train_dataset = SFTDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_dataset = SFTDataset(val_pairs, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.max_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda", enabled=(not args.no_mixed_precision and device == "cuda"))
    use_amp = not args.no_mixed_precision and device == "cuda"

    steps_per_epoch = len(train_loader)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = vars(args)
    config_dict["model_name"] = model_name
    config_dict["trainable_params"] = trainable
    config_dict["total_params"] = total
    config_dict["num_train"] = len(train_pairs)
    config_dict["num_val"] = len(val_pairs)
    config_dict["total_steps"] = total_steps
    config_dict["warmup_steps"] = warmup_steps
    config_dict["steps_per_epoch"] = steps_per_epoch
    config_dict["effective_batch_size"] = effective_batch_size

    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    if args.wandb:
        run_name = args.run_name or f"sft_{args.lr}_{args.max_epochs}ep"
        wandb.init(
            project="budget-aware-dpo",
            name=run_name,
            config=config_dict,
            tags=["sft", "phase4"],
        )

    logger.info(
        "Training config: batch=%d, grad_accum=%d, effective_batch=%d, epochs=%d, steps/epoch=%d, total_steps=%d, warmup=%d",
        args.batch_size, args.gradient_accumulation_steps, effective_batch_size,
        args.max_epochs, steps_per_epoch, total_steps, warmup_steps,
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        num_batches = 0
        optimizer.zero_grad()

        start_time = time.time()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.wandb and global_step % 10 == 0:
                    wandb.log({
                        "train/loss": loss.item() * args.gradient_accumulation_steps,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }, step=global_step)

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            # Count solution tokens (non -100 labels)
            epoch_tokens += (labels != -100).sum().item()
            num_batches += 1

            if (batch_idx + 1) % 500 == 0:
                avg_loss = epoch_loss / num_batches
                logger.info("  E%d step %d/%d: loss=%.4f, lr=%.2e",
                            epoch, batch_idx + 1, steps_per_epoch, avg_loss, scheduler.get_last_lr()[0])

        train_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                val_loss_sum += outputs.loss.item()
                val_batches += 1

        val_loss = val_loss_sum / val_batches

        logger.info(
            "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, time=%.1fmin, solution_tokens=%d",
            epoch, args.max_epochs, train_loss, val_loss, epoch_time / 60, epoch_tokens,
        )

        if args.wandb:
            wandb.log({
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
                "epoch/time_min": epoch_time / 60,
                "epoch/solution_tokens": epoch_tokens,
            }, step=global_step)

        # Save checkpoint
        checkpoint_dir = output_dir / f"epoch-{epoch}"
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
        logger.info("Saved checkpoint: %s", checkpoint_dir)

        # Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_dir = output_dir / "best-model"
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            logger.info("New best model (val_loss=%.4f) saved to %s", val_loss, best_dir)
        else:
            patience_counter += 1
            logger.info("No improvement (%d/%d patience)", patience_counter, args.early_stopping_patience)

        # Generation eval
        if epoch % args.gen_eval_every == 0:
            run_gen_eval(model, tokenizer, epoch, global_step, use_wandb=args.wandb)

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info("Early stopping at epoch %d (best was epoch %d)", epoch, best_epoch)
            break

    logger.info("=" * 60)
    logger.info("Training complete. Best epoch: %d, best val_loss: %.4f", best_epoch, best_val_loss)
    logger.info("Best model: %s/best-model", output_dir)
    logger.info("=" * 60)

    if args.wandb:
        wandb.finish()

    return {"best_val_loss": best_val_loss, "best_epoch": best_epoch}


if __name__ == "__main__":
    main()
