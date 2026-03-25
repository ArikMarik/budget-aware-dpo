"""
Shared DPO training logic: data loading, forward pass, checkpointing.
Optimized for GPU utilization and training efficiency.
"""

import json
import math
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Any, Callable

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import wandb

from src.config import (
    CHECKPOINT_DIR,
    MODEL_NAME,
    get_processed_dataset_path,
    get_tokenized_train_path,
    get_tokenized_val_path,
)
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)


@dataclass
class TrainingConfig:
    use_budget_aware: bool
    max_epochs: int
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
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    compile_model: bool = False
    num_workers: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


class EarlyStopping:
    def __init__(self, patience: int = 5, threshold: float = 0.0, threshold_mode: str = "rel"):
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.early_stop = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.threshold_mode == "rel":
            improved = score > self.best_score * (1 + self.threshold)
        else:
            improved = score > self.best_score + self.threshold

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False


def log_prob(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)


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
        logger.error("Tokenized dataset not found at %s. Run preprocess_dpo_data.py first.", tokens_path)
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
    return (
        torch.stack([item["chosen_input_ids"] for item in batch]),
        torch.stack([item["chosen_attention_mask"] for item in batch]),
        torch.stack([item["rejected_input_ids"] for item in batch]),
        torch.stack([item["rejected_attention_mask"] for item in batch]),
        torch.stack([item["complexity"] for item in batch]),
    )


def _pad_token_if_needed(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def create_model(
    model_name: str,
    device: str,
    lora_config: Optional[LoraConfig] = None,
    resume_from: Optional[str] = None,
    use_compile: bool = False,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
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

    if use_compile and hasattr(torch, "compile") and device == "cuda":
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def create_ref_model(model_name: str, device: str) -> nn.Module:  # type: ignore[return]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _forward_models(
    model: nn.Module,
    ref_model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    use_compile: bool,
) -> tuple:
    with torch.no_grad():
        ref_chosen = ref_model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
        ref_rejected = ref_model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

    policy_chosen = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
    policy_rejected = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

    return policy_chosen, policy_rejected, ref_chosen, ref_rejected


def compute_batch_loss_train(
    model: nn.Module,
    ref_model: nn.Module,
    batch: tuple,
    tokenizer: PreTrainedTokenizer,
    loss_fn: Callable,
    dpo_beta: float,
    use_compile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask, complexities = batch
    chosen_ids = chosen_ids.cuda(non_blocking=True)
    chosen_mask = chosen_mask.cuda(non_blocking=True)
    rejected_ids = rejected_ids.cuda(non_blocking=True)
    rejected_mask = rejected_mask.cuda(non_blocking=True)
    complexities = complexities.cuda(non_blocking=True)

    _pad_token_if_needed(tokenizer)

    policy_chosen, policy_rejected, ref_chosen, ref_rejected = _forward_models(
        model, ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, use_compile
    )

    policy_chosen_lp = log_prob(policy_chosen, chosen_ids, chosen_mask)
    policy_rejected_lp = log_prob(policy_rejected, rejected_ids, rejected_mask)
    ref_chosen_lp = log_prob(ref_chosen, chosen_ids, chosen_mask)
    ref_rejected_lp = log_prob(ref_rejected, rejected_ids, rejected_mask)

    chosen_lens = (chosen_ids != tokenizer.pad_token_id).sum(dim=-1).float()
    rejected_lens = (rejected_ids != tokenizer.pad_token_id).sum(dim=-1).float()

    loss, extra = loss_fn(
        policy_chosen_lp,
        policy_rejected_lp,
        ref_chosen_lp,
        ref_rejected_lp,
        chosen_lens,
        rejected_lens,
        complexities,
    )

    return loss, policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen_lens, extra


def compute_batch_loss_eval(
    model: nn.Module,
    ref_model: nn.Module,
    batch: tuple,
    tokenizer: PreTrainedTokenizer,
    loss_fn: Callable,
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask, complexities = batch
    chosen_ids = chosen_ids.cuda(non_blocking=True)
    chosen_mask = chosen_mask.cuda(non_blocking=True)
    rejected_ids = rejected_ids.cuda(non_blocking=True)
    rejected_mask = rejected_mask.cuda(non_blocking=True)
    complexities = complexities.cuda(non_blocking=True)

    _pad_token_if_needed(tokenizer)

    policy_chosen, policy_rejected, ref_chosen, ref_rejected = _forward_models(
        model, ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, False
    )

    policy_chosen_lp = log_prob(policy_chosen, chosen_ids, chosen_mask)
    policy_rejected_lp = log_prob(policy_rejected, rejected_ids, rejected_mask)
    ref_chosen_lp = log_prob(ref_chosen, chosen_ids, chosen_mask)
    ref_rejected_lp = log_prob(ref_rejected, rejected_ids, rejected_mask)

    chosen_lens = (chosen_ids != tokenizer.pad_token_id).sum(dim=-1).float()
    rejected_lens = (rejected_ids != tokenizer.pad_token_id).sum(dim=-1).float()

    loss, extra = loss_fn(
        policy_chosen_lp,
        policy_rejected_lp,
        ref_chosen_lp,
        ref_rejected_lp,
        chosen_lens,
        rejected_lens,
        complexities,
    )

    reward_diff_per_sample = dpo_beta * (
        (policy_chosen_lp - ref_chosen_lp) - (policy_rejected_lp - ref_rejected_lp)
    )
    per_sample_loss = -F.logsigmoid(reward_diff_per_sample)

    mask_easy = (complexities == 0).float()
    mask_hard = (complexities == 1).float()

    metrics = {
        "loss": loss.detach(),
        "reward_diff": reward_diff_per_sample.mean().detach(),
        "complexity_0_loss": (per_sample_loss * mask_easy).sum() / mask_easy.sum().clamp(min=1),
        "complexity_1_loss": (per_sample_loss * mask_hard).sum() / mask_hard.sum().clamp(min=1),
        "avg_chosen_tokens": chosen_lens.mean().detach(),
        "avg_rejected_tokens": rejected_lens.mean().detach(),
    }

    return loss, metrics


def evaluate(
    model: nn.Module,
    ref_model: nn.Module,
    val_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    loss_fn: Callable,
    dpo_beta: float,
) -> tuple[float, dict]:
    model.eval()
    total_loss = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    total_reward_diff = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    total_complexity_0 = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    total_complexity_1 = torch.zeros((), device="cuda" if torch.cuda.is_available() else "cpu")
    num_batches = 0

    with torch.inference_mode():
        for batch in val_loader:
            loss, batch_metrics = compute_batch_loss_eval(
                model, ref_model, batch, tokenizer, loss_fn, dpo_beta
            )
            total_loss += loss.detach()
            total_reward_diff += batch_metrics["reward_diff"].detach()
            total_complexity_0 += batch_metrics["complexity_0_loss"].detach()
            total_complexity_1 += batch_metrics["complexity_1_loss"].detach()
            num_batches += 1

    model.train()

    num_batches_t = max(num_batches, 1)
    avg_loss = (total_loss / num_batches_t).cpu().item()
    metrics = {
        "val/reward_diff": (total_reward_diff / num_batches_t).cpu().item(),
        "val/complexity_0_loss": (total_complexity_0 / num_batches_t).cpu().item(),
        "val/complexity_1_loss": (total_complexity_1 / num_batches_t).cpu().item(),
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
    gradient_norm: Optional[float] = None,
    epoch: int = 0,
    complexity_0_loss: Optional[float] = None,
    complexity_1_loss: Optional[float] = None,
    val_reward_diff: Optional[float] = None,
    val_complexity_0_loss: Optional[float] = None,
    val_complexity_1_loss: Optional[float] = None,
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
    if gradient_norm is not None:
        log_dict["train/gradient_norm"] = gradient_norm
    if complexity_0_loss is not None:
        log_dict["train/complexity_0_loss"] = complexity_0_loss
    if complexity_1_loss is not None:
        log_dict["train/complexity_1_loss"] = complexity_1_loss
    if val_loss is not None:
        log_dict["val/loss"] = val_loss
    if extra and "length_penalty" in extra:
        log_dict["train/length_penalty"] = extra["length_penalty"]
    if val_reward_diff is not None:
        log_dict["val/reward_diff"] = val_reward_diff
    if val_complexity_0_loss is not None:
        log_dict["val/complexity_0_loss"] = val_complexity_0_loss
    if val_complexity_1_loss is not None:
        log_dict["val/complexity_1_loss"] = val_complexity_1_loss
    wandb.log(log_dict, step=step)


def save_checkpoint(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    epoch: int,
    metrics_log: list,
) -> None:
    ckpt_path = output_dir / f"checkpoint-epoch-{epoch}"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_path))
    tokenizer.save_pretrained(ckpt_path)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info("Saved checkpoint to %s", ckpt_path)


def save_best_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    best_model_path = output_dir / "best-model"
    best_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_model_path))
    tokenizer.save_pretrained(best_model_path)
    logger.info("Saved best model to %s (val_loss: %.4f, epoch: %d)", best_model_path, best_val_loss, best_epoch)


def _init_wandb(config: TrainingConfig, run_name: Optional[str] = None) -> None:
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "budget-aware-dpo"),
        name=run_name or os.environ.get("WANDB_RUN_NAME"),
        config=config.to_dict(),
        mode=wandb_mode,
    )


def _build_loss_fn(
    use_budget_aware: bool,
    dpo_beta: float,
    lambda_easy: float,
    lambda_hard: float,
) -> Callable:
    if use_budget_aware:
        from src.models.budget_aware_dpo_loss import budget_aware_dpo_loss
        return lambda pc, pr, rc, rr, cl, rl, c: budget_aware_dpo_loss(
            pc, pr, rc, rr, cl, rl, c, beta=dpo_beta, lambda_easy=lambda_easy, lambda_hard=lambda_hard
        )
    else:
        from src.models.standard_dpo_loss import standard_dpo_loss
        return lambda pc, pr, rc, rr, cl, rl, c: standard_dpo_loss(
            pc, pr, rc, rr, beta=dpo_beta
        )


def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def _load_val_split() -> float:
    processed_path = get_processed_dataset_path()
    meta_path = processed_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        return metadata.get("val_split", 0.2)
    return 0.2


def _build_dataloaders(
    train_dataset: TokenizedDPODataset,
    val_dataset: TokenizedDPODataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_tokenized,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_tokenized,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def train_dpo(
    *,
    use_budget_aware: bool,
    output_dir: Path,
    max_epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-5,
    checkpoint_every: int = 1,
    data_limit: Optional[int] = None,
    resume_from: Optional[str] = None,
    seed: int = 42,
    use_wandb: bool = False,
    run_name: Optional[str] = None,
    early_stopping_patience: int = 5,
    early_stopping_threshold: float = 0.0,
    dpo_beta: float = 0.1,
    lambda_easy: float = 0.05,
    lambda_hard: float = 0.001,
    gradient_accumulation_steps: int = 1,
    use_mixed_precision: bool = True,
    compile_model: bool = False,
    num_workers: int = 4,
) -> dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_split = _load_val_split()

    train_tokens_path = get_tokenized_train_path()
    val_tokens_path = get_tokenized_val_path()
    _validate_datasets_exist(train_tokens_path, val_tokens_path)

    pin_memory = device == "cuda"
    model, tokenizer = create_model(
        MODEL_NAME,
        device,
        lora_config=_build_lora_config(),
        resume_from=resume_from,
        use_compile=compile_model and device == "cuda",
    )
    ref_model = create_ref_model(MODEL_NAME, device)

    train_dataset = load_tokenized_dataset(train_tokens_path)
    val_dataset = load_tokenized_dataset(val_tokens_path)
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    logger.info("Data split: Train=%s, Val=%s", num_train, num_val)

    train_loader, val_loader = _build_dataloaders(
        train_dataset, val_dataset, batch_size, num_workers, pin_memory
    )

    loss_fn = _build_loss_fn(use_budget_aware, dpo_beta, lambda_easy, lambda_hard)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    scaler = torch.amp.GradScaler("cuda", enabled=(use_mixed_precision and device == "cuda"))

    steps_per_epoch = len(train_loader)
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(
        "Training config: batch_size=%d, grad_accum=%d, effective_batch=%d, epochs=%d, steps_per_epoch=%d",
        batch_size, gradient_accumulation_steps, effective_batch_size, max_epochs, steps_per_epoch
    )

    config = TrainingConfig(
        use_budget_aware=use_budget_aware,
        max_epochs=max_epochs,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_mixed_precision=use_mixed_precision,
        compile_model=compile_model,
        num_workers=num_workers,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    if use_wandb:
        if not run_name:
            variant = os.environ.get("DATASET_VARIANT", "unknown")
            mode = "budget_aware" if use_budget_aware else "baseline"
            run_name = f"{mode}_{variant}_s{seed}"
        _init_wandb(config, run_name=run_name)

    metrics_log = []
    best_val_loss = float("inf")
    best_model_state: Optional[dict] = None
    best_epoch = 0
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        threshold=early_stopping_threshold,
        threshold_mode="rel",
    )
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32

    for epoch in range(1, max_epochs + 1):
        epoch_metrics = _run_epoch(
            model=model,
            ref_model=ref_model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            dpo_beta=dpo_beta,
            epoch=epoch,
            metrics_log=metrics_log,
            use_wandb=use_wandb,
            steps_per_epoch=steps_per_epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_mixed_precision=use_mixed_precision and device == "cuda",
            autocast_dtype=autocast_dtype,
            compile_model=compile_model,
        )

        best_val_loss, best_model_state, best_epoch = _update_best_model(
            epoch_metrics, epoch, model, best_val_loss, best_model_state, best_epoch
        )

        if epoch % checkpoint_every == 0:
            save_checkpoint(model, tokenizer, output_dir, epoch, metrics_log)

        if early_stopping(epoch_metrics["val_loss"], epoch):
            logger.info(
                "Early stopping triggered at epoch %d (best epoch: %d)",
                epoch,
                early_stopping.best_epoch
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if device != "cuda":
            model = model.to(device)
        save_best_model(model, tokenizer, output_dir, best_val_loss, best_epoch)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info("Training complete. Saved to %s", output_dir)
    return {
        "metrics": metrics_log,
        "config": config.to_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def _validate_datasets_exist(train_path: Path, val_path: Path) -> None:
    if not train_path.exists() or not val_path.exists():
        logger.error("Tokenized datasets not found. Expected train: %s, val: %s. Run preprocess_dpo_data.py first.", train_path, val_path)
        raise FileNotFoundError(
            f"Tokenized datasets not found. Run preprocess_dpo_data.py first.\n"
            f"Expected: {train_path}, {val_path}"
        )


def _run_epoch(
    model: nn.Module,
    ref_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    dpo_beta: float,
    epoch: int,
    metrics_log: list,
    use_wandb: bool,
    steps_per_epoch: int,
    gradient_accumulation_steps: int,
    use_mixed_precision: bool,
    autocast_dtype: torch.dtype,
    compile_model: bool,
) -> dict:
    model.train()
    device = next(model.parameters()).device

    accum_loss = torch.zeros((), device=device)
    accum_reward_diff = torch.zeros((), device=device)
    accum_complexity_0 = torch.zeros((), device=device)
    accum_complexity_1 = torch.zeros((), device=device)
    accum_chosen_tokens = torch.zeros((), device=device)
    accum_rejected_tokens = torch.zeros((), device=device)
    accum_length_penalty = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader), mininterval=1.0, dynamic_ncols=True)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        is_last_accum = (batch_idx + 1) % gradient_accumulation_steps == 0

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=autocast_dtype,
            enabled=use_mixed_precision,
        ):
            loss, policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen_lens, extra = compute_batch_loss_train(
                model, ref_model, batch, tokenizer, loss_fn, dpo_beta, compile_model
            )
            loss = loss / gradient_accumulation_steps

        if use_mixed_precision:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()

        with torch.no_grad():
            reward_diff_per_sample = dpo_beta * (
                (policy_chosen_lp - ref_chosen_lp) - (policy_rejected_lp - ref_rejected_lp)
            )
            per_sample_loss = -F.logsigmoid(reward_diff_per_sample)
            complexities = batch[-1].cuda(non_blocking=True)
            mask_easy = (complexities == 0).float()
            mask_hard = (complexities == 1).float()
            rejected_lens = ((batch[-2] if isinstance(batch[-2], torch.Tensor) else batch[-2]) != tokenizer.pad_token_id).sum(dim=-1).float()
            if rejected_lens.device != device:
                rejected_lens = rejected_lens.to(device)
            elif not isinstance(rejected_lens, torch.Tensor):
                rejected_lens = torch.tensor(rejected_lens, device=device, dtype=torch.float)

            accum_loss += loss.detach() * gradient_accumulation_steps
            accum_reward_diff += reward_diff_per_sample.mean().detach()
            accum_complexity_0 += (per_sample_loss * mask_easy).sum() / mask_easy.sum().clamp(min=1)
            accum_complexity_1 += (per_sample_loss * mask_hard).sum() / mask_hard.sum().clamp(min=1)
            accum_chosen_tokens += chosen_lens.mean().detach()
            accum_rejected_tokens += rejected_lens.mean().detach()
            if "length_penalty" in extra:
                accum_length_penalty += extra["length_penalty"]

        if is_last_accum:
            if use_mixed_precision:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # type: ignore[assignment]
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            current_lr = optimizer.param_groups[0]["lr"]
            num_steps_so_far = batch_idx + 1
            pbar.set_postfix({
                "step": num_steps_so_far,
                "loss": f"{(accum_loss / num_steps_so_far).item():.4f}",
                "lr": f"{current_lr:.2e}",
                "gn": f"{grad_norm.item():.2f}",
            })

            if use_wandb:
                global_step = (epoch - 1) * steps_per_epoch + num_steps_so_far
                extra_wandb = {}
                if accum_length_penalty != 0.0:
                    extra_wandb["length_penalty"] = accum_length_penalty / num_steps_so_far
                log_metrics(
                    step=global_step,
                    train_loss=(accum_loss / num_steps_so_far).item(),
                    val_loss=None,
                    avg_chosen_tokens=accum_chosen_tokens.item() / num_steps_so_far,
                    avg_rejected_tokens=accum_rejected_tokens.item() / num_steps_so_far,
                    learning_rate=current_lr,
                    reward_diff=accum_reward_diff.item() / num_steps_so_far,
                    gradient_norm=grad_norm.item(),
                    epoch=epoch,
                    complexity_0_loss=accum_complexity_0.item() / num_steps_so_far,
                    complexity_1_loss=accum_complexity_1.item() / num_steps_so_far,
                    extra=extra_wandb if extra_wandb else None,
                )

    if use_wandb:
        num_batches = len(train_loader)
        avg_train_loss = (accum_loss / num_batches).cpu().item()
        avg_reward_diff = (accum_reward_diff / num_batches).cpu().item()
        avg_complexity_0 = (accum_complexity_0 / num_batches).cpu().item()
        avg_complexity_1 = (accum_complexity_1 / num_batches).cpu().item()
        avg_chosen = (accum_chosen_tokens / num_batches).cpu().item()
        avg_rejected = (accum_rejected_tokens / num_batches).cpu().item()
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss, val_metrics = evaluate(
            model, ref_model, val_loader, tokenizer, loss_fn, dpo_beta
        )
        logger.info(
            "Epoch %d: train_loss=%.4f, val_loss=%.4f, reward_diff=%.4f",
            epoch, avg_train_loss, val_loss, val_metrics["val/reward_diff"]
        )
        entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_reward_diff": val_metrics["val/reward_diff"],
            "val_complexity_0_loss": val_metrics["val/complexity_0_loss"],
            "val_complexity_1_loss": val_metrics["val/complexity_1_loss"],
        }
        metrics_log.append(entry)

        log_metrics(
            step=(epoch * steps_per_epoch),
            train_loss=avg_train_loss,
            val_loss=val_loss,
            avg_chosen_tokens=avg_chosen,
            avg_rejected_tokens=avg_rejected,
            learning_rate=current_lr,
            reward_diff=avg_reward_diff,
            gradient_norm=grad_norm.cpu().item(),
            epoch=epoch,
            complexity_0_loss=avg_complexity_0,
            complexity_1_loss=avg_complexity_1,
            val_reward_diff=val_metrics["val/reward_diff"],
            val_complexity_0_loss=val_metrics["val/complexity_0_loss"],
            val_complexity_1_loss=val_metrics["val/complexity_1_loss"],
        )

        return {"val_loss": val_loss, "val_metrics": val_metrics, "avg_train_loss": avg_train_loss}

    avg_train_loss = (accum_loss / len(train_loader)).cpu().item()
    logger.info("Epoch %d avg train loss: %.4f", epoch, avg_train_loss)

    val_loss, val_metrics = evaluate(
        model, ref_model, val_loader, tokenizer, loss_fn, dpo_beta
    )
    logger.info(
        "Epoch %d val_loss: %.4f (reward_diff: %.4f)",
        epoch, val_loss, val_metrics["val/reward_diff"]
    )

    num_batches = len(train_loader)
    entry = {
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": val_loss,
        "val_reward_diff": val_metrics["val/reward_diff"],
        "val_complexity_0_loss": val_metrics["val/complexity_0_loss"],
        "val_complexity_1_loss": val_metrics["val/complexity_1_loss"],
    }
    metrics_log.append(entry)

    return {"val_loss": val_loss, "val_metrics": val_metrics, "avg_train_loss": avg_train_loss}


def _update_best_model(
    epoch_metrics: dict,
    epoch: int,
    model: nn.Module,
    best_val_loss: float,
    best_model_state: Optional[dict],
    best_epoch: int,
) -> tuple[float, Optional[dict], int]:
    val_loss = epoch_metrics["val_loss"]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        logger.info("New best val_loss: %.4f at epoch %d", best_val_loss, epoch)
    return best_val_loss, best_model_state, best_epoch
