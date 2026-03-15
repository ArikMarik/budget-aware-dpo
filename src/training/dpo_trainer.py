"""
Shared DPO training logic: data loading, forward pass, checkpointing.
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import wandb

from src.config import (
    CHECKPOINT_DIR,
    MODEL_NAME,
    get_processed_dataset_path,
    get_tokenized_train_path,
    get_tokenized_val_path,
)
from src.utils import set_seed


@dataclass
class TrainingConfig:
    use_budget_aware: bool
    max_steps: int
    batch_size: int
    lr: float
    seed: int
    data_limit: Optional[int]
    num_pairs: int
    num_train_pairs: int
    num_val_pairs: int
    val_split: float = 0.2
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    dpo_beta: float = 0.1
    lambda_easy: float = 0.05
    lambda_hard: float = 0.001

    def to_dict(self) -> dict:
        return asdict(self)


class EarlyStopping:
    def __init__(self, patience: int = 5, threshold: float = 0.0, threshold_mode: str = "rel"):
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            return False

        if self.threshold_mode == "rel":
            improved = score > self.best_score * (1 + self.threshold)
        else:
            improved = score > self.best_score + self.threshold

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def log_prob(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    return torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1).sum(-1)


def compute_batch_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: tuple,
    tokenizer: AutoTokenizer,
    loss_fn: callable,
    device: str,
    dpo_beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, float], dict[str, float]]:
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

    loss, extra = loss_fn(
        policy_chosen_lp,
        policy_rejected_lp,
        ref_chosen_lp,
        ref_rejected_lp,
        chosen_lens_t,
        rejected_lens_t,
        complexities,
    )

    logps = {
        "policy_chosen_lp": policy_chosen_lp.mean().item(),
        "policy_rejected_lp": policy_rejected_lp.mean().item(),
    }

    reward_diff = dpo_beta * (
        (policy_chosen_lp - ref_chosen_lp) - (policy_rejected_lp - ref_rejected_lp)
    ).mean().item()
    logps["reward_diff"] = reward_diff

    reward_diff_per_sample = dpo_beta * (
        (policy_chosen_lp - ref_chosen_lp) - (policy_rejected_lp - ref_rejected_lp)
    )
    per_sample_loss = -torch.nn.functional.logsigmoid(reward_diff_per_sample)

    mask_easy = (complexities == 0).float()
    mask_hard = (complexities == 1).float()

    if mask_easy.sum() > 0:
        logps["complexity_0_loss"] = (per_sample_loss * mask_easy).sum() / mask_easy.sum()
    else:
        logps["complexity_0_loss"] = 0.0

    if mask_hard.sum() > 0:
        logps["complexity_1_loss"] = (per_sample_loss * mask_hard).sum() / mask_hard.sum()
    else:
        logps["complexity_1_loss"] = 0.0

    token_stats = {
        "avg_chosen_tokens": chosen_lens_t.mean(),
        "avg_rejected_tokens": rejected_lens_t.mean(),
    }

    return loss, extra, token_stats, logps


class DPODataset(Dataset):
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]


class TokenizedDPODataset(Dataset):
    def __init__(self, tokens_path: Path):
        self.data = torch.load(tokens_path)
        self.length = self.data["chosen_input_ids"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return {
            "chosen_input_ids": self.data["chosen_input_ids"][idx],
            "chosen_attention_mask": self.data["chosen_attention_mask"][idx],
            "rejected_input_ids": self.data["rejected_input_ids"][idx],
            "rejected_attention_mask": self.data["rejected_attention_mask"][idx],
            "complexity": self.data["complexities"][idx],
        }


def load_tokenized_dataset(tokens_path: Path) -> TokenizedDPODataset:
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found at {tokens_path}. "
            "Run preprocess_dpo_data.py first."
        )
    return TokenizedDPODataset(tokens_path)


def load_pairs(limit: Optional[int] = None) -> list[dict]:
    pairs = []
    path = get_processed_dataset_path() / "dataset.jsonl"
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
            if limit and len(pairs) >= limit:
                break
    return pairs


def collate_fn_tokenized(batch):
    chosen_input_ids = torch.stack([item["chosen_input_ids"] for item in batch])
    chosen_attention_mask = torch.stack([item["chosen_attention_mask"] for item in batch])
    rejected_input_ids = torch.stack([item["rejected_input_ids"] for item in batch])
    rejected_attention_mask = torch.stack([item["rejected_attention_mask"] for item in batch])
    complexities = torch.stack([item["complexity"] for item in batch])

    chosen_tok = {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask}
    rejected_tok = {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask}
    return chosen_tok, rejected_tok, complexities


def collate_fn_raw(batch: list[dict], tokenizer: AutoTokenizer, max_length: int = 512):
    """Legacy collate function for non-tokenized data."""
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


def create_model(
    model_name: str,
    device: str,
    lora_config: Optional[LoraConfig] = None,
    resume_from: Optional[str] = None,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    if lora_config is not None:
        model = get_peft_model(model, lora_config)

    if resume_from:
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)

    model.train()

    for p in model.parameters():
        p.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def create_ref_model(model_name: str, device: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def evaluate(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    val_loader: DataLoader,
    tokenizer: AutoTokenizer,
    loss_fn: callable,
    device: str,
    use_budget_aware: bool,
    dpo_beta: float = 0.1,
) -> tuple[float, dict]:
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    total_reward_diff = 0.0
    total_complexity_0_loss = 0.0
    total_complexity_1_loss = 0.0
    num_complexity_0 = 0
    num_complexity_1 = 0
    num_batches = 0

    with torch.inference_mode():
        for batch in val_loader:
            loss, extra, token_stats, logps = compute_batch_loss(
                model, ref_model, batch, tokenizer, loss_fn, device, dpo_beta
            )
            total_loss += loss.detach()
            total_reward_diff += logps["reward_diff"]
            if isinstance(logps["complexity_0_loss"], torch.Tensor):
                total_complexity_0_loss += logps["complexity_0_loss"].item()
            else:
                total_complexity_0_loss += logps["complexity_0_loss"]
            if isinstance(logps["complexity_1_loss"], torch.Tensor):
                total_complexity_1_loss += logps["complexity_1_loss"].item()
            else:
                total_complexity_1_loss += logps["complexity_1_loss"]
            num_batches += 1

    model.train()

    avg_loss = (total_loss / num_batches).item() if num_batches > 0 else 0.0
    metrics = {
        "val/reward_diff": total_reward_diff / num_batches if num_batches > 0 else 0.0,
        "val/complexity_0_loss": total_complexity_0_loss / num_batches if num_batches > 0 else 0.0,
        "val/complexity_1_loss": total_complexity_1_loss / num_batches if num_batches > 0 else 0.0,
    }
    return avg_loss, metrics


def log_metrics(
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    avg_chosen_tokens: float,
    avg_rejected_tokens: float,
    learning_rate: float,
    extra: Optional[dict] = None,
    reward_diff: Optional[float] = None,
    policy_chosen_logps: Optional[float] = None,
    policy_rejected_logps: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    epoch: int = 0,
    complexity_0_loss: Optional[float] = None,
    complexity_1_loss: Optional[float] = None,
) -> None:
    log_dict = {
        "train/loss": train_loss,
        "train/step": step,
        "train/epoch": epoch,
        "train/avg_chosen_tokens": avg_chosen_tokens,
        "train/avg_rejected_tokens": avg_rejected_tokens,
        "train/token_diff": avg_chosen_tokens - avg_rejected_tokens,
        "train/learning_rate": learning_rate,
    }
    if reward_diff is not None:
        log_dict["train/reward_diff"] = reward_diff
    if policy_chosen_logps is not None:
        log_dict["train/policy_chosen_logps"] = policy_chosen_logps
    if policy_rejected_logps is not None:
        log_dict["train/policy_rejected_logps"] = policy_rejected_logps
    if gradient_norm is not None:
        log_dict["train/gradient_norm"] = gradient_norm
    if complexity_0_loss is not None:
        log_dict["train/complexity_0_loss"] = complexity_0_loss
    if complexity_1_loss is not None:
        log_dict["train/complexity_1_loss"] = complexity_1_loss
    if val_loss is not None:
        log_dict["val/loss"] = val_loss
    if extra and "length_penalty" in extra:
        lp = extra["length_penalty"]
        log_dict["train/length_penalty"] = lp.item() if isinstance(lp, torch.Tensor) else lp
    wandb.log(log_dict, step=step)


def save_checkpoint(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    step: int,
    metrics_log: list,
) -> None:
    ckpt_path = output_dir / f"checkpoint-{step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)


def save_best_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    best_val_loss: float,
) -> None:
    best_model_path = output_dir / "best-model"
    best_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"Saved best model to {best_model_path} (val_loss: {best_val_loss:.4f})")


def train_dpo(
    *,
    use_budget_aware: bool,
    output_dir: Path,
    max_steps: int = 1000,
    batch_size: int = 4,
    lr: float = 1e-5,
    checkpoint_every: int = 500,
    eval_every: int = 100,
    log_every: int = 50,
    data_limit: Optional[int] = None,
    resume_from: Optional[str] = None,
    seed: int = 42,
    use_wandb: bool = False,
    early_stopping_patience: int = 5,
    early_stopping_threshold: float = 0.0,
    dpo_beta: float = 0.1,
    lambda_easy: float = 0.05,
    lambda_hard: float = 0.001,
) -> dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processed_path = get_processed_dataset_path()
    meta_path = processed_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        val_split = metadata.get("val_split", 0.2)
        num_train_pairs = metadata.get("num_train_pairs", 0)
        num_val_pairs = metadata.get("num_val_pairs", 0)
    else:
        val_split = 0.2
        num_train_pairs = 0
        num_val_pairs = 0

    train_tokens_path = get_tokenized_train_path()
    val_tokens_path = get_tokenized_val_path()

    if not train_tokens_path.exists() or not val_tokens_path.exists():
        raise FileNotFoundError(
            f"Tokenized datasets not found. Run preprocess_dpo_data.py first.\n"
            f"Expected: {train_tokens_path}, {val_tokens_path}"
        )

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model, tokenizer = create_model(
        MODEL_NAME, device, lora_config=lora_config, resume_from=resume_from
    )
    ref_model = create_ref_model(MODEL_NAME, device)

    train_dataset = load_tokenized_dataset(train_tokens_path)
    val_dataset = load_tokenized_dataset(val_tokens_path)
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    print(f"Data split: Train={num_train}, Val={num_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_tokenized,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_tokenized,
    )

    if use_budget_aware:
        from src.models.budget_aware_dpo_loss import budget_aware_dpo_loss
        loss_fn = lambda pc, pr, rc, rr, cl, rl, c: budget_aware_dpo_loss(
            pc, pr, rc, rr, cl, rl, c, beta=dpo_beta, lambda_easy=lambda_easy, lambda_hard=lambda_hard
        )
    else:
        from src.models.standard_dpo_loss import standard_dpo_loss
        loss_fn = lambda pc, pr, rc, rr, cl, rl, c: standard_dpo_loss(
            pc, pr, rc, rr, beta=dpo_beta
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    config = TrainingConfig(
        use_budget_aware=use_budget_aware,
        max_steps=max_steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        data_limit=data_limit,
        num_pairs=num_train + num_val,
        num_train_pairs=num_train,
        num_val_pairs=num_val,
        val_split=val_split,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        dpo_beta=dpo_beta,
        lambda_easy=lambda_easy,
        lambda_hard=lambda_hard,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    if use_wandb:
        wandb_mode = os.environ.get("WANDB_MODE", "online")
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "budget-aware-dpo"),
            name=os.environ.get("WANDB_RUN_NAME"),
            config=config.to_dict(),
            mode=wandb_mode,
        )

    metrics_log = []
    step = 0
    epoch = 0
    best_val_loss = float("inf")
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        threshold=early_stopping_threshold,
        threshold_mode="rel",
    )
    best_model_state = None

    pbar = tqdm(total=max_steps, desc="Training", mininterval=1.0)

    while step < max_steps:
        epoch_loss = torch.tensor(0.0, device=device)
        for batch in train_loader:
            if step >= max_steps:
                break

            loss, extra, token_stats, logps = compute_batch_loss(
                model, ref_model, batch, tokenizer, loss_fn, device, dpo_beta
            )

            optimizer.zero_grad()
            loss.backward()

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()

            epoch_loss += loss.detach()
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            if step % log_every == 0 and use_wandb:
                log_metrics(
                    step=step,
                    train_loss=loss.item(),
                    val_loss=None,
                    avg_chosen_tokens=token_stats["avg_chosen_tokens"].item(),
                    avg_rejected_tokens=token_stats["avg_rejected_tokens"].item(),
                    learning_rate=optimizer.param_groups[0]["lr"],
                    extra=extra,
                    reward_diff=logps["reward_diff"],
                    policy_chosen_logps=logps["policy_chosen_lp"],
                    policy_rejected_logps=logps["policy_rejected_lp"],
                    gradient_norm=grad_norm,
                    epoch=epoch,
                    complexity_0_loss=logps["complexity_0_loss"].item() if isinstance(logps["complexity_0_loss"], torch.Tensor) else logps["complexity_0_loss"],
                    complexity_1_loss=logps["complexity_1_loss"].item() if isinstance(logps["complexity_1_loss"], torch.Tensor) else logps["complexity_1_loss"],
                )

            if step % eval_every == 0:
                loss_val = loss.item()
                entry = {"step": step, "loss": loss_val}
                if extra:
                    entry.update(extra)
                metrics_log.append(entry)
                print(f"Step {step} loss: {loss_val:.4f}")

                val_loss, val_metrics = evaluate(
                    model, ref_model, val_loader, tokenizer, loss_fn, device, use_budget_aware, dpo_beta
                )
                entry["val_loss"] = val_loss
                entry["val_reward_diff"] = val_metrics["val/reward_diff"]
                entry["val_complexity_0_loss"] = val_metrics["val/complexity_0_loss"]
                entry["val_complexity_1_loss"] = val_metrics["val/complexity_1_loss"]
                metrics_log[-1] = entry
                print(f"Step {step} val_loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"Step {step} new best val_loss: {best_val_loss:.4f}")

                if early_stopping(val_loss):
                    print(f"Early stopping triggered at step {step}")
                    pbar.close()
                    break

                if use_wandb:
                    log_dict = {
                        "step": step,
                        "train_loss": loss_val,
                        "val_loss": val_loss,
                        "avg_chosen_tokens": token_stats["avg_chosen_tokens"].item(),
                        "avg_rejected_tokens": token_stats["avg_rejected_tokens"].item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "reward_diff": logps["reward_diff"],
                        "policy_chosen_logps": logps["policy_chosen_lp"],
                        "policy_rejected_logps": logps["policy_rejected_lp"],
                        "gradient_norm": grad_norm,
                        "epoch": epoch,
                        "complexity_0_loss": logps["complexity_0_loss"].item() if isinstance(logps["complexity_0_loss"], torch.Tensor) else logps["complexity_0_loss"],
                        "complexity_1_loss": logps["complexity_1_loss"].item() if isinstance(logps["complexity_1_loss"], torch.Tensor) else logps["complexity_1_loss"],
                        "val/reward_diff": val_metrics["val/reward_diff"],
                        "val/complexity_0_loss": val_metrics["val/complexity_0_loss"],
                        "val/complexity_1_loss": val_metrics["val/complexity_1_loss"],
                    }
                    if extra and "length_penalty" in extra:
                        lp = extra["length_penalty"]
                        log_dict["train/length_penalty"] = lp.item() if isinstance(lp, torch.Tensor) else lp
                    wandb.log(log_dict, step=step)

            if step % checkpoint_every == 0:
                save_checkpoint(
                    model, tokenizer, output_dir, step, metrics_log
                )
                print(f"Saved checkpoint to {output_dir / f'checkpoint-{step}'}")

        epoch += 1
        avg = (epoch_loss / len(train_loader)).item()
        print(f"Epoch {epoch} avg loss: {avg:.4f}")

        if early_stopping.early_stop:
            break

    pbar.close()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        save_best_model(model, tokenizer, output_dir, best_val_loss)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Training complete. Saved to {output_dir}")
    return {"metrics": metrics_log, "config": config.to_dict(), "best_val_loss": best_val_loss}
