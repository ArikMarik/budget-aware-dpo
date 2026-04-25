"""
Shared DPO training logic: data loading, forward pass, checkpointing.
Optimized for GPU utilization and training efficiency.
"""

from collections import defaultdict
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Callable, Literal

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
    DATA_PATH,
    MODEL_NAME,
    get_tokens_path,
    get_processed_dataset_path,
)
from src.data.preprocessing import split_pairs_by_problem
from src.evaluation.answer_extraction import extract_answer, verify_correctness
from src.evaluation.run_evaluation import (
    default_prompt_fn,
    generate_and_evaluate,
    compute_metrics,
    load_eval_problems,
)
from src.utils import get_logger, set_seed, setup_global_exception_handler

logger = get_logger(__name__)
setup_global_exception_handler(__name__)

BestModelMetric = Literal[
    "val_loss",
    "gen_tokens_easy",
    "gen_tpca",
    "gen_tokens_easy_with_accuracy_floor",
]


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
    # lambda_hard: float = 0.001
    lambda_hard: float = 0.03
    kl_penalty_weight: float = 0.0
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


class MetricsAccumulator:
    """Accumulator for training/eval metrics with support for per-complexity breakdown."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.total_loss = torch.zeros((), device=self.device)
        self.total_reward_diff = torch.zeros((), device=self.device)
        self.total_reward_diff_easy = torch.zeros((), device=self.device)
        self.total_reward_diff_hard = torch.zeros((), device=self.device)
        self.total_complexity_0 = torch.zeros((), device=self.device)
        self.total_complexity_1 = torch.zeros((), device=self.device)
        self.total_accuracy = torch.zeros((), device=self.device)
        self.total_accuracy_easy = torch.zeros((), device=self.device)
        self.total_accuracy_hard = torch.zeros((), device=self.device)
        self.total_chosen_easy = torch.zeros((), device=self.device)
        self.total_chosen_hard = torch.zeros((), device=self.device)
        self.total_rejected_easy = torch.zeros((), device=self.device)
        self.total_rejected_hard = torch.zeros((), device=self.device)
        self.total_easy_count = torch.zeros((), device=self.device)
        self.total_hard_count = torch.zeros((), device=self.device)
        self.total_chosen_tokens = torch.zeros((), device=self.device)
        self.total_rejected_tokens = torch.zeros((), device=self.device)
        self.total_length_penalty = 0.0
        self.total_kl_penalty = 0.0
        self.num_batches = 0

    def update(
        self,
        loss: torch.Tensor,
        reward_diff_per_sample: torch.Tensor,
        per_sample_loss: torch.Tensor,
        complexities: torch.Tensor,
        chosen_lens: torch.Tensor,
        rejected_lens: torch.Tensor,
        extra: Optional[dict] = None,
    ) -> None:
        mask_easy = (complexities == 0).float()
        mask_hard = (complexities == 1).float()
        easy_count = mask_easy.sum()
        hard_count = mask_hard.sum()

        batch_accuracy = (reward_diff_per_sample > 0).float()

        self.total_loss += loss.detach()
        self.total_reward_diff += reward_diff_per_sample.mean().detach()
        self.total_reward_diff_easy += (reward_diff_per_sample * mask_easy).sum().detach() / easy_count.clamp(min=1)
        self.total_reward_diff_hard += (reward_diff_per_sample * mask_hard).sum().detach() / hard_count.clamp(min=1)
        self.total_complexity_0 += (per_sample_loss * mask_easy).sum() / easy_count.clamp(min=1)
        self.total_complexity_1 += (per_sample_loss * mask_hard).sum() / hard_count.clamp(min=1)
        self.total_accuracy += batch_accuracy.mean().detach()
        self.total_accuracy_easy += (batch_accuracy * mask_easy).sum().detach() / easy_count.clamp(min=1)
        self.total_accuracy_hard += (batch_accuracy * mask_hard).sum().detach() / hard_count.clamp(min=1)
        self.total_chosen_tokens += chosen_lens.mean().detach()
        self.total_rejected_tokens += rejected_lens.mean().detach()
        self.total_chosen_easy += (chosen_lens * mask_easy).sum().detach()
        self.total_chosen_hard += (chosen_lens * mask_hard).sum().detach()
        self.total_rejected_easy += (rejected_lens * mask_easy).sum().detach()
        self.total_rejected_hard += (rejected_lens * mask_hard).sum().detach()
        self.total_easy_count += easy_count.detach()
        self.total_hard_count += hard_count.detach()
        if extra:
            if "length_penalty" in extra:
                self.total_length_penalty += extra["length_penalty"]
            if "kl_penalty" in extra:
                self.total_kl_penalty += extra["kl_penalty"]
        self.num_batches += 1

    def compute_metrics(self) -> dict[str, float]:
        n = max(self.num_batches, 1)
        easy_n = max(self.total_easy_count.clamp(min=1).cpu().item(), 1)
        hard_n = max(self.total_hard_count.clamp(min=1).cpu().item(), 1)

        return {
            "loss": (self.total_loss / n).cpu().item(),
            "reward_diff": (self.total_reward_diff / n).cpu().item(),
            "reward_diff_easy": (self.total_reward_diff_easy / n).cpu().item(),
            "reward_diff_hard": (self.total_reward_diff_hard / n).cpu().item(),
            "complexity_0_loss": (self.total_complexity_0 / n).cpu().item(),
            "complexity_1_loss": (self.total_complexity_1 / n).cpu().item(),
            "accuracy": (self.total_accuracy / n).cpu().item(),
            "accuracy_easy": (self.total_accuracy_easy / n).cpu().item(),
            "accuracy_hard": (self.total_accuracy_hard / n).cpu().item(),
            "avg_chosen_tokens": (self.total_chosen_tokens / n).cpu().item(),
            "avg_rejected_tokens": (self.total_rejected_tokens / n).cpu().item(),
            "avg_chosen_tokens_easy": self.total_chosen_easy.cpu().item() / easy_n,
            "avg_chosen_tokens_hard": self.total_chosen_hard.cpu().item() / hard_n,
            "avg_rejected_tokens_easy": self.total_rejected_easy.cpu().item() / easy_n,
            "avg_rejected_tokens_hard": self.total_rejected_hard.cpu().item() / hard_n,
            "easy_count": self.total_easy_count.cpu().item(),
            "hard_count": self.total_hard_count.cpu().item(),
            "length_penalty": self.total_length_penalty,
            "kl_penalty": self.total_kl_penalty,
        }

    def to_wandb_dict(self, n: int, grad_norm: Optional[float] = None) -> dict:
        easy_n = max(self.total_easy_count.item(), 1)
        hard_n = max(self.total_hard_count.item(), 1)
        result = {
            "reward_diff_easy": self.total_reward_diff_easy.item() / n,
            "reward_diff_hard": self.total_reward_diff_hard.item() / n,
            "accuracy": self.total_accuracy.item() / n,
            "accuracy_easy": self.total_accuracy_easy.item() / n,
            "accuracy_hard": self.total_accuracy_hard.item() / n,
            "avg_chosen_tokens_easy": self.total_chosen_easy.item() / easy_n,
            "avg_chosen_tokens_hard": self.total_chosen_hard.item() / hard_n,
            "avg_rejected_tokens_easy": self.total_rejected_easy.item() / easy_n,
            "avg_rejected_tokens_hard": self.total_rejected_hard.item() / hard_n,
        }
        if self.total_length_penalty != 0.0:
            result["length_penalty"] = self.total_length_penalty / n
        if self.total_kl_penalty != 0.0:
            result["kl_penalty"] = self.total_kl_penalty / n
        if grad_norm is not None:
            result["grad_norm"] = grad_norm
        return result

    def update_for_step(self, loss: torch.Tensor, extra: Optional[dict] = None) -> None:
        self.total_loss += loss.detach()
        if extra:
            if "length_penalty" in extra:
                self.total_length_penalty += extra["length_penalty"]
            if "kl_penalty" in extra:
                self.total_kl_penalty += extra["kl_penalty"]
        self.num_batches += 1

    def compute_per_step_wandb(self, n: int, grad_norm: float) -> dict:
        return self.to_wandb_dict(n, grad_norm)

    def compute_val_metrics(self) -> dict[str, float]:
        n = max(self.num_batches, 1)
        easy_n = max(self.total_easy_count.clamp(min=1).cpu().item(), 1)
        hard_n = max(self.total_hard_count.clamp(min=1).cpu().item(), 1)
        return {
            "val/reward_diff": (self.total_reward_diff / n).cpu().item(),
            "val/reward_diff_easy": (self.total_reward_diff_easy / n).cpu().item(),
            "val/reward_diff_hard": (self.total_reward_diff_hard / n).cpu().item(),
            "val/complexity_0_loss": (self.total_complexity_0 / n).cpu().item(),
            "val/complexity_1_loss": (self.total_complexity_1 / n).cpu().item(),
            "val/accuracy": (self.total_accuracy / n).cpu().item(),
            "val/accuracy_easy": (self.total_accuracy_easy / n).cpu().item(),
            "val/accuracy_hard": (self.total_accuracy_hard / n).cpu().item(),
            "val/avg_chosen_tokens_easy": self.total_chosen_easy.cpu().item() / easy_n,
            "val/avg_chosen_tokens_hard": self.total_chosen_hard.cpu().item() / hard_n,
            "val/avg_rejected_tokens_easy": self.total_rejected_easy.cpu().item() / easy_n,
            "val/avg_rejected_tokens_hard": self.total_rejected_hard.cpu().item() / hard_n,
        }


def log_prob(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().float()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * shift_mask).sum(-1) / shift_mask.sum(-1).clamp(min=1)


class TokenizedDPODataset(Dataset):
    """Simple dataset with pre-computed indices."""

    def __init__(self, data: dict, indices: list[int]):
        self.data = data
        self.indices = indices
        self.length = len(indices)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        return {
            "chosen_input_ids": self.data["chosen_input_ids"][real_idx],
            "chosen_attention_mask": self.data["chosen_attention_mask"][real_idx],
            "rejected_input_ids": self.data["rejected_input_ids"][real_idx],
            "rejected_attention_mask": self.data["rejected_attention_mask"][real_idx],
            "complexity": self.data["complexities"][real_idx],
            "problem_id": self.data["problem_ids"][real_idx],
        }


def _cap_pairs_per_problem(
    data: dict,
    indices: list[int],
    max_pairs_per_problem: int,
    seed: int,
) -> list[int]:
    """Cap pairs per problem_id using stratified sampling by rejection_reason."""
    import numpy as np

    rng = np.random.default_rng(seed)
    problem_ids = data["problem_ids"].numpy()
    rejection_reason = data["rejection_reason"].numpy()

    # Group the *filtered* indices by problem_id
    by_problem: dict[int, list[int]] = defaultdict(list)
    for i in indices:
        by_problem[int(problem_ids[i])].append(i)

    kept: list[int] = []
    for problem_indices in tqdm(by_problem.values(), desc=f'Restricting to {max_pairs_per_problem} pairs per Problem'):
        if len(problem_indices) <= max_pairs_per_problem:
            kept.extend(problem_indices)
            continue

        # Stratified by rejection_reason
        by_reason: dict[int, list[int]] = defaultdict(list)
        for i in problem_indices:
            by_reason[int(rejection_reason[i])].append(i)

        total = len(problem_indices)
        remaining = max_pairs_per_problem
        for reason, group in by_reason.items():
            if remaining <= 0:
                break
            quota = max(1, round(len(group) / total * max_pairs_per_problem))
            quota = min(quota, len(group), remaining)
            chosen = rng.choice(group, size=quota, replace=False).tolist()
            kept.extend(int(x) for x in chosen)
            remaining -= quota

    return sorted(kept)


def _filter_by_length_ratio(data: dict, length_ratio: float) -> list[int]:
    """Vectorized filtering using numpy."""
    import numpy as np

    if length_ratio <= 1.0:
        return np.arange(len(data["chosen_input_ids"])).tolist()

    complexities = data["complexities"].numpy()
    rejection_reason = data["rejection_reason"].numpy()
    chosen_length = data["chosen_length"].numpy()
    rejected_length = data["rejected_length"].numpy()

    mask_easy = ((complexities == 0) & (chosen_length * length_ratio <= rejected_length))
    mask_hard = ((complexities == 1) & (rejected_length * length_ratio <= chosen_length))
    not_rejected_by_length = rejection_reason != 0

    valid_mask = mask_easy | mask_hard | not_rejected_by_length

    return np.where(valid_mask)[0].tolist()


def load_tokenized_datasets(
    tokens_path: Path,
    length_ratio: float = 1.0,
    val_split: float = 0.2,
    seed: int = 42,
    max_pairs_per_problem: Optional[int] = None,
    max_unique_problems: int = 100_000
) -> tuple[TokenizedDPODataset, TokenizedDPODataset]:
    """
    Load all tokenized pairs, filter by length_ratio, optionally cap pairs per
    problem, then split by problem_id. Returns (train_dataset, val_dataset).

    Processing order:
    1. Load all data from tokens_path
    2. Apply length_ratio filter to all pairs
    3. Apply max_pairs_per_problem cap (stratified by rejection_reason)
    4. Split filtered pairs by problem_id (stratified by complexity)
    """
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found at {tokens_path}. "
            "Run preprocess_dpo_data.py first."
        )

    data = torch.load(tokens_path)

    # 1. Apply length_ratio filter (vectorized)
    filtered_indices = _filter_by_length_ratio(data, length_ratio)

    # 2. Cap pairs per problem (stratified by rejection_reason)
    if max_pairs_per_problem is not None and max_pairs_per_problem > 0:
        filtered_indices = _cap_pairs_per_problem(
            data, filtered_indices, max_pairs_per_problem, seed
        )

    # 3. Split by problem_id (stratified by complexity of filtered data)
    train_indices, val_indices = split_pairs_by_problem(
        data, val_split, seed, filtered_indices, max_unique_problems
    )

    train_dataset = TokenizedDPODataset(data, train_indices)
    val_dataset = TokenizedDPODataset(data, val_indices)

    logger.info(
        f"Data split (length_ratio={length_ratio}, "
        f"max_pairs_per_problem={max_pairs_per_problem}): "
        f"Train (pairs)={len(train_indices)}, Val (pairs)={len(val_indices)}"
    )

    return train_dataset, val_dataset


def collate_fn_tokenized(batch: list[dict]) -> dict:
    keys = ["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask", "complexity", "problem_id"]
    collated_batch = {key: [] for key in keys}
    for item in batch:
        for key, value in item.items():
            collated_batch[key].append(value)
    return {key: torch.stack(values) for key, values in collated_batch.items()}


def _pad_token_if_needed(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def create_tokenizer(model_name: str) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _pad_token_if_needed(tokenizer)
    return tokenizer

def create_model(
    model_name: str,
    device: str,
    lora_config: Optional[LoraConfig] = None,
    resume_from: Optional[str] = None,
    use_compile: bool = False,
) -> PeftModel:
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

    return model


def create_ref_model(model_name: str, device: str) -> PeftModel:
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


def _move_batch_to_device(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_ids = batch['chosen_input_ids'].cuda(non_blocking=True)
    chosen_mask = batch['chosen_attention_mask'].cuda(non_blocking=True)
    rejected_ids = batch['rejected_input_ids'].cuda(non_blocking=True)
    rejected_mask = batch['rejected_attention_mask'].cuda(non_blocking=True)
    complexities = batch['complexity'].cuda(non_blocking=True)
    return chosen_ids, chosen_mask, rejected_ids, rejected_mask, complexities


def _compute_batch_forward(
    model: nn.Module,
    ref_model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> tuple:
    _pad_token_if_needed(tokenizer)

    with torch.no_grad():
        ref_chosen = ref_model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
        ref_rejected = ref_model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

    policy_chosen = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
    policy_rejected = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

    policy_chosen_lp = log_prob(policy_chosen, chosen_ids, chosen_mask)
    policy_rejected_lp = log_prob(policy_rejected, rejected_ids, rejected_mask)
    ref_chosen_lp = log_prob(ref_chosen, chosen_ids, chosen_mask)
    ref_rejected_lp = log_prob(ref_rejected, rejected_ids, rejected_mask)

    chosen_lens = (chosen_ids != tokenizer.pad_token_id).sum(dim=-1).float()
    rejected_lens = (rejected_ids != tokenizer.pad_token_id).sum(dim=-1).float()

    return policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen_lens, rejected_lens


def compute_batch_loss_train(
    model: nn.Module,
    ref_model: nn.Module,
    batch: dict,
    tokenizer: PreTrainedTokenizer,
    loss_fn: Callable,
    use_compile: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask, complexities = _move_batch_to_device(batch)

    policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen_lens, rejected_lens = _compute_batch_forward(
        model, ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, tokenizer
    )

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


def _compute_val_accuracy(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    val_problems: list[dict],
    epoch: int,
    use_wandb: bool,
    num_batches: int,
    max_new_tokens: int = 256,
) -> dict[str, float]:
    """Run generation on val_problems and compute accuracy."""
    model.eval()
    device = next(model.parameters()).device

    results = []
    for problem_info in tqdm(val_problems, desc=f"Val gen epoch {epoch}"):
        inputs = problem_info['prompt']
        with torch.inference_mode():
            out = model.generate(
                **inputs.to(device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_length = inputs["input_ids"].shape[1]
        new_tokens = out[0][input_length:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        correct = verify_correctness(response, problem_info["expected_answer"], problem=problem_info["problem"], logs=False) if problem_info["expected_answer"] else None
        results.append({
            "problem": problem_info["problem"],
            "complexity": problem_info["complexity"],
            "correct": correct,
        })

    model.train()

    easy_results, hard_results = [], []
    correct_count = defaultdict(int)
    for r in results:
        if r["complexity"] == 1:
            hard_results.append(r)
        else:
            easy_results.append(r)

        correct_count[r['complexity']] += r["correct"]

    accuracy = sum(correct_count.values()) / max(len(results), 1)
    accuracy_easy = correct_count[0] / max(len(easy_results), 1)
    accuracy_hard = correct_count[1] / max(len(hard_results), 1)

    val_accuracy = {
        "val/accuracy": accuracy,
        "val/accuracy_easy": accuracy_easy,
        "val/accuracy_hard": accuracy_hard,
    }

    logger.info(
        "Epoch %d val accuracy: %.4f (easy=%.4f, hard=%.4f)",
        epoch, accuracy, accuracy_easy, accuracy_hard
    )

    if use_wandb:
        wandb.log(val_accuracy, step=(epoch * num_batches))

    return val_accuracy


def compute_batch_loss_eval(
    model: nn.Module,
    ref_model: nn.Module,
    batch: dict,
    tokenizer: PreTrainedTokenizer,
    loss_fn: Callable,
    dpo_beta: float,
    metrics_accumulator: MetricsAccumulator
) -> torch.Tensor:
    chosen_ids, chosen_mask, rejected_ids, rejected_mask, complexities = _move_batch_to_device(batch)

    policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen_lens, rejected_lens = _compute_batch_forward(
        model, ref_model, chosen_ids, chosen_mask, rejected_ids, rejected_mask, tokenizer
    )

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

    metrics_accumulator.update(loss, reward_diff_per_sample, per_sample_loss, complexities, chosen_lens, rejected_lens, extra)

    return loss


def build_val_problems(
    val_loader: DataLoader,
    problem_index: dict,
    tokenizer: PreTrainedTokenizer,
    prompt_fn: Callable = default_prompt_fn,
) -> list[dict]:
    """Build list of unique validation problems with pre-tokenized prompts."""
    seen_problem_ids = set()
    val_problems = []

    for batch in tqdm(val_loader, desc="Building val problems"):
        problem_ids = batch['problem_id'].tolist()
        for pid in problem_ids:
            if pid not in seen_problem_ids:
                seen_problem_ids.add(pid)
                info = problem_index[pid]
                problem_text = info["problem"]
                prompt_text = prompt_fn(problem_text)
                tokenized = tokenizer(prompt_text, return_tensors="pt")
                val_problems.append({
                    "problem_id": pid,
                    "problem": problem_text,
                    "prompt": tokenized,
                    "expected_answer": info["expected_answer"],
                    "complexity": info["complexity"],
                })

    logger.info(f"Built val_problems for {len(val_problems)} unique problems")
    return val_problems


def evaluate(
    model: nn.Module,
    ref_model: nn.Module,
    val_loader: DataLoader,
    tokenizer: PreTrainedTokenizer,
    loss_fn: Callable,
    dpo_beta: float,
) -> tuple[float, dict]:
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    accum = MetricsAccumulator(torch.device(dev))

    with torch.inference_mode():
        for batch in val_loader:
            loss = compute_batch_loss_eval(
                model, ref_model, batch, tokenizer, loss_fn, dpo_beta, accum
            )

    model.train()

    avg_loss = (accum.total_loss / accum.num_batches).cpu().item()
    metrics = accum.compute_val_metrics()
    return avg_loss, metrics


def log_metrics(
    step: int,
    train_loss: float,
    avg_chosen_tokens: float,
    avg_rejected_tokens: float,
    learning_rate: float,
    extra: Optional[dict] = None,
    reward_diff: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    epoch: int = 0,
    complexity_0_loss: Optional[float] = None,
    complexity_1_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    val_metrics: Optional[dict] = None,
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
    if extra:
        for key, value in extra.items():
            log_dict[f"train/{key}"] = value
    if val_metrics:
        log_dict.update(val_metrics)
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
    *,
    best_model_metric: BestModelMetric,
    best_epoch: int,
    best_epoch_metrics: dict[str, float],
) -> None:
    best_model_path = output_dir / "best-model"
    best_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_model_path))
    tokenizer.save_pretrained(best_model_path)
    selection_payload = {
        "best_model_metric": best_model_metric,
        "best_epoch": best_epoch,
        "metrics": best_epoch_metrics,
    }
    with open(output_dir / "best_model_selection.json", "w") as f:
        json.dump(selection_payload, f, indent=2)

    logger.info(
        "Saved best model to %s (metric=%s, epoch=%d). "
        "val_loss=%.4f, gen/accuracy=%.4f, gen/accuracy_easy=%.4f, gen/avg_tokens_easy=%.1f, gen/tpca=%.1f",
        best_model_path,
        best_model_metric,
        best_epoch,
        float(best_epoch_metrics.get("val_loss", float("nan"))),
        float(best_epoch_metrics.get("gen/accuracy", float("nan"))),
        float(best_epoch_metrics.get("gen/accuracy_easy", float("nan"))),
        float(best_epoch_metrics.get("gen/avg_tokens_easy", float("nan"))),
        float(best_epoch_metrics.get("gen/tpca", float("nan"))),
    )


def _init_wandb(config: TrainingConfig, run_name: Optional[str] = None) -> None:
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "budget-aware-dpo"),
        name=run_name or os.environ.get("WANDB_RUN_NAME"),
        config=config.to_dict(),
        mode=wandb_mode,
    )
    # Define val and gen metrics to use epoch as x-axis so they render as proper graphs
    wandb.define_metric("train/epoch")
    wandb.define_metric("val/*", step_metric="train/epoch")
    wandb.define_metric("gen/*", step_metric="train/epoch")


def _build_loss_fn(
    use_budget_aware: bool,
    dpo_beta: float,
    lambda_easy: float,
    lambda_hard: float,
    kl_penalty_weight: float = 0.0,
    loss_type: str = "dpo",
) -> Callable:
    if loss_type == "simpo":
        from src.models.simpo_loss import simpo_loss
        return lambda pc, pr, rc, rr, cl, rl, c: simpo_loss(
            pc, pr, cl, rl, c, beta=dpo_beta, gamma=0.5,
            lambda_easy=lambda_easy, lambda_hard=lambda_hard,
        )
    if use_budget_aware:
        from src.models.budget_aware_dpo_loss import budget_aware_dpo_loss
        return lambda pc, pr, rc, rr, cl, rl, c: budget_aware_dpo_loss(
            pc, pr, rc, rr, cl, rl, c, beta=dpo_beta, lambda_easy=lambda_easy, lambda_hard=lambda_hard,
            kl_penalty_weight=kl_penalty_weight,
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
    val_split: float = 0.2,
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
    lambda_hard: float = 0.03,
    kl_penalty_weight: float = 0.0,
    gradient_accumulation_steps: int = 1,
    use_mixed_precision: bool = True,
    compile_model: bool = False,
    num_workers: int = 4,
    model_name: Optional[str] = None,
    loss_type: str = "dpo",
    length_ratio: float = 1.0,
    max_pairs_per_problem: Optional[int] = 3,
    best_model_metric: BestModelMetric = "val_loss",
    accuracy_floor: Optional[float] = None,
    max_unique_problems: int = 65_000,
    problem_index_path: Path = DATA_PATH / "problem_index_dict.json",
) -> dict:
    print(f'{" STARTED DPO TRAINER ":#^100}')
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load, filter, and split in one call
    all_tokens_path = get_tokens_path()
    train_dataset, val_dataset = load_tokenized_datasets(
        all_tokens_path,
        length_ratio=length_ratio,
        val_split=val_split,
        seed=seed,
        max_pairs_per_problem=max_pairs_per_problem,
        max_unique_problems=max_unique_problems
    )

    num_train = len(train_dataset)
    num_val = len(val_dataset)
    logger.info("Data split: Train=%s, Val=%s (length_ratio=%.1f)", num_train, num_val, length_ratio)

    pin_memory = device == "cuda"

    train_loader, val_loader = _build_dataloaders(
        train_dataset, val_dataset, batch_size, num_workers, pin_memory
    )

    effective_model_name = model_name or MODEL_NAME
    tokenizer = create_tokenizer(effective_model_name)

    # Build validation problems with pre-tokenized prompts
    if problem_index_path.exists():
        with open(problem_index_path) as f:
            problem_index = json.load(f, object_hook=lambda obj: {int(k) if k.isdigit() else k: v for k, v in obj.items()})
        val_problems = build_val_problems(val_loader, problem_index, tokenizer, default_prompt_fn)
    else:
        logger.warning(f"Problem index not found at {problem_index_path}, skipping val_problems")
        val_problems = []

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
        kl_penalty_weight=kl_penalty_weight,
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
    best_epoch_metrics: Optional[dict[str, float]] = None
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        threshold=early_stopping_threshold,
        threshold_mode="rel",
    )
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info("Using model: %s (loss_type=%s)", effective_model_name, loss_type)
    model = create_model(
        effective_model_name,
        device,
        lora_config=_build_lora_config(),
        resume_from=resume_from,
        use_compile=compile_model and device == "cuda"
    )
    ref_model = create_ref_model(effective_model_name, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    loss_fn = _build_loss_fn(use_budget_aware, dpo_beta, lambda_easy, lambda_hard, kl_penalty_weight, loss_type=loss_type)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_mixed_precision and device == "cuda"))

    # TODO - what is it used for
    # Load generation eval problems once (50 easy + 50 hard)
    # gen_eval_problems = _load_gen_eval_problems(n_easy=50, n_hard=50)

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
            val_problems=val_problems,
        )

        # # Generation-based evaluation after each epoch
        # gen_metrics = _run_gen_eval(
        #     model=model,
        #     tokenizer=tokenizer,
        #     problems=gen_eval_problems,
        #     epoch=epoch,
        #     use_wandb=use_wandb,
        #     steps_per_epoch=steps_per_epoch,
        # )
        # epoch_metrics["gen_metrics"] = gen_metrics

        # Persist generation metrics into the epoch entry so we can compare epochs later.
        if metrics_log:
            # metrics_log[-1].update(gen_metrics)
            metrics_log[-1]["best_model_metric"] = best_model_metric
            if best_model_metric == "gen_tokens_easy_with_accuracy_floor":
                metrics_log[-1]["accuracy_floor"] = accuracy_floor
        else:
            logger.warning("metrics_log was empty after epoch %d; cannot persist gen metrics.", epoch)

        best_val_loss, best_model_state, best_epoch, best_epoch_metrics = _update_best_model(
            epoch_metrics=epoch_metrics,
            epoch=epoch,
            model=model,
            best_val_loss=best_val_loss,
            best_model_state=best_model_state,
            best_epoch=best_epoch,
            best_epoch_metrics=best_epoch_metrics,
            best_model_metric=best_model_metric,
            accuracy_floor=accuracy_floor,
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

    if use_wandb:
        wandb.finish()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if device != "cuda":
            model = model.to(device)
        save_best_model(
            model,
            tokenizer,
            output_dir,
            best_model_metric=best_model_metric,
            best_epoch=best_epoch,
            best_epoch_metrics=(best_epoch_metrics or {}),
        )
    else:
        logger.warning("No best model state was selected; skipping best-model save.")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info("Training complete. Saved to %s", output_dir)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "config": config.to_dict(),
                "best_model_metric": best_model_metric,
                "accuracy_floor": accuracy_floor,
                "best_epoch": best_epoch,
                "best_epoch_metrics": best_epoch_metrics,
                "best_val_loss": best_val_loss,
            },
            f,
            indent=2,
        )
    return {
        "metrics": metrics_log,
        "config": config.to_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def _get_best_value_for_epoch(
    *,
    epoch_metrics: dict,
    best_model_metric: BestModelMetric,
    accuracy_floor: Optional[float],
) -> Optional[float]:
    if best_model_metric == "val_loss":
        return float(epoch_metrics["val_loss"])

    # gen_metrics = epoch_metrics.get("gen_metrics") or {}
    # if best_model_metric == "gen_tokens_easy":
    #     return float(gen_metrics["gen/avg_tokens_easy"])
    # if best_model_metric == "gen_tpca":
    #     return float(gen_metrics["gen/tpca"])
    # if best_model_metric == "gen_tokens_easy_with_accuracy_floor":
    #     if accuracy_floor is None:
    #         raise ValueError("--accuracy-floor is required when --best-model-metric=gen_tokens_easy_with_accuracy_floor")
    #     acc_easy = float(gen_metrics["gen/accuracy_easy"])
    #     if acc_easy < float(accuracy_floor):
    #         return None
    #     return float(gen_metrics["gen/avg_tokens_easy"])

    raise ValueError(f"Unsupported best_model_metric: {best_model_metric}")


def _load_gen_eval_problems(n_easy: int = 50, n_hard: int = 50) -> list[dict]:
    """Load a balanced set of eval problems for generation-based validation."""
    all_problems = load_eval_problems(limit=None)
    easy = [p for p in all_problems if p["complexity"] == 0][:n_easy]
    hard = [p for p in all_problems if p["complexity"] == 1][:n_hard]
    problems = easy + hard
    logger.info(
        "Loaded %d gen-eval problems (%d easy, %d hard)",
        len(problems), len(easy), len(hard),
    )
    return problems


# def _run_gen_eval(
#     model: nn.Module,
#     tokenizer: PreTrainedTokenizer,
#     problems: list[dict],
#     epoch: int,
#     use_wandb: bool,
#     steps_per_epoch: int,
# ) -> dict[str, float]:
#     """Run generation-based evaluation and log results."""
#     model.eval()
#     results = generate_and_evaluate(model, tokenizer, problems, use_llm_judge=False)
#     metrics = compute_metrics(results)
#     model.train()

#     gen_metrics = {
#         "gen/accuracy": metrics["accuracy"],
#         "gen/accuracy_easy": len([r for r in results if r["complexity"] == 0 and r["correct"]]) / max(len([r for r in results if r["complexity"] == 0]), 1),
#         "gen/accuracy_hard": len([r for r in results if r["complexity"] == 1 and r["correct"]]) / max(len([r for r in results if r["complexity"] == 1]), 1),
#         "gen/avg_tokens_easy": metrics["avg_tokens_easy"],
#         "gen/avg_tokens_hard": metrics["avg_tokens_hard"],
#         "gen/tpca": metrics["tpca"],
#     }

#     logger.info(
#         "Epoch %d gen-eval: accuracy=%.4f (easy=%.4f, hard=%.4f), "
#         "avg_tokens_easy=%.1f, avg_tokens_hard=%.1f, tpca=%.1f",
#         epoch,
#         gen_metrics["gen/accuracy"],
#         gen_metrics["gen/accuracy_easy"],
#         gen_metrics["gen/accuracy_hard"],
#         gen_metrics["gen/avg_tokens_easy"],
#         gen_metrics["gen/avg_tokens_hard"],
#         gen_metrics["gen/tpca"],
#     )

#     if use_wandb:
#         wandb.log(
#             {**gen_metrics, "train/epoch": epoch},
#             step=(epoch * steps_per_epoch),
#         )

#     return gen_metrics


def _build_extra_wandb_dict(
    accum: MetricsAccumulator,
    n: int,
    grad_norm: float,
) -> dict:
    easy_n = max(accum.total_easy_count.item(), 1) or 1
    hard_n = max(accum.total_hard_count.item(), 1) or 1
    result = {
        "reward_diff_easy": accum.total_reward_diff_easy.item() / n,
        "reward_diff_hard": accum.total_reward_diff_hard.item() / n,
        "accuracy": accum.total_accuracy.item() / n,
        "accuracy_easy": accum.total_accuracy_easy.item() / n,
        "accuracy_hard": accum.total_accuracy_hard.item() / n,
        "avg_chosen_tokens_easy": accum.total_chosen_easy.item() / easy_n,
        "avg_chosen_tokens_hard": accum.total_chosen_hard.item() / hard_n,
        "avg_rejected_tokens_easy": accum.total_rejected_easy.item() / easy_n,
        "avg_rejected_tokens_hard": accum.total_rejected_hard.item() / hard_n,
    }
    if accum.total_length_penalty != 0.0:
        result["length_penalty"] = accum.total_length_penalty / n
    if accum.total_kl_penalty != 0.0:
        result["kl_penalty"] = accum.total_kl_penalty / n
    return result


def _compute_epoch_averages(
    accum: MetricsAccumulator,
    num_batches: int,
) -> dict:
    return {
        "avg_train_loss": (accum.total_loss / num_batches).cpu().item(),
        "avg_reward_diff": (accum.total_reward_diff / num_batches).cpu().item(),
        "avg_complexity_0": (accum.total_complexity_0 / num_batches).cpu().item(),
        "avg_complexity_1": (accum.total_complexity_1 / num_batches).cpu().item(),
        "avg_chosen": (accum.total_chosen_tokens / num_batches).cpu().item(),
        "avg_rejected": (accum.total_rejected_tokens / num_batches).cpu().item(),
    }


def _build_epoch_entry(
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_metrics: dict,
) -> dict:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_reward_diff": val_metrics["val/reward_diff"],
        "val_complexity_0_loss": val_metrics["val/complexity_0_loss"],
        "val_complexity_1_loss": val_metrics["val/complexity_1_loss"],
    }


def _run_epoch(
    model: PeftModel,
    ref_model: PeftModel,
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
    val_problems: Optional[list[dict]] = None,
) -> dict:
    model.train()
    device = next(model.parameters()).device
    accum = MetricsAccumulator(device)
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=num_batches, mininterval=1.0, dynamic_ncols=True)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        is_last_accum = (batch_idx + 1) % gradient_accumulation_steps == 0

        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=autocast_dtype,
            enabled=use_mixed_precision,
        ):
            loss, policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, chosen_lens, extra = compute_batch_loss_train(
                model, ref_model, batch, tokenizer, loss_fn, compile_model
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
            complexities = batch['complexity'].cuda(non_blocking=True)
            rejected_lens = (batch['rejected_attention_mask'] != tokenizer.pad_token_id).sum(dim=-1).float()
            if rejected_lens.device != device:
                rejected_lens = rejected_lens.to(device)
            elif not isinstance(rejected_lens, torch.Tensor):
                rejected_lens = torch.tensor(rejected_lens, device=device, dtype=torch.float)

            accum.update(
                loss.detach() * gradient_accumulation_steps,
                reward_diff_per_sample,
                per_sample_loss,
                complexities,
                chosen_lens,
                rejected_lens,
                extra,
            )

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
                "loss": f"{(accum.total_loss / num_steps_so_far).item():.4f}",
                "lr": f"{current_lr:.2e}",
                "gn": f"{grad_norm.item():.2f}",
            })

            if use_wandb:
                global_step = (epoch - 1) * steps_per_epoch + num_steps_so_far
                extra_wandb = _build_extra_wandb_dict(accum, num_steps_so_far, grad_norm.item())
                log_metrics(
                    step=global_step,
                    train_loss=(accum.total_loss / num_steps_so_far).item(),
                    avg_chosen_tokens=(accum.total_chosen_tokens / num_steps_so_far).item(),
                    avg_rejected_tokens=(accum.total_rejected_tokens / num_steps_so_far).item(),
                    learning_rate=current_lr,
                    reward_diff=(accum.total_reward_diff / num_steps_so_far).item(),
                    gradient_norm=grad_norm.item(),
                    epoch=epoch,
                    complexity_0_loss=(accum.total_complexity_0 / num_steps_so_far).item(),
                    complexity_1_loss=(accum.total_complexity_1 / num_steps_so_far).item(),
                    extra=extra_wandb,
                )

    averages = _compute_epoch_averages(accum, num_batches)
    val_loss, val_metrics = evaluate(
        model, ref_model, val_loader, tokenizer, loss_fn, dpo_beta
    )

    # Compute accuracy on validation problems if available
    val_accuracy = {}
    if val_problems:
        val_accuracy = _compute_val_accuracy(
            model, tokenizer, val_problems, epoch, use_wandb, steps_per_epoch
        )

    logger.info(
        "Epoch %d: train_loss=%.4f, val_loss=%.4f, reward_diff=%.4f",
        epoch, averages["avg_train_loss"], val_loss, val_metrics["val/reward_diff"]
    )
    entry = _build_epoch_entry(epoch, averages["avg_train_loss"], val_loss, val_metrics)
    metrics_log.append(entry)

    if use_wandb:
        current_lr = optimizer.param_groups[0]["lr"]

        log_metrics(
            step=(epoch * steps_per_epoch),
            train_loss=averages["avg_train_loss"],
            val_loss=val_loss,
            val_metrics=val_metrics,
            avg_chosen_tokens=averages["avg_chosen"],
            avg_rejected_tokens=averages["avg_rejected"],
            learning_rate=current_lr,
            reward_diff=averages["avg_reward_diff"],
            gradient_norm=grad_norm.cpu().item(),
            epoch=epoch,
            complexity_0_loss=averages["avg_complexity_0"],
            complexity_1_loss=averages["avg_complexity_1"],
        )

    return {"val_loss": val_loss, "val_metrics": val_metrics, "train_loss": averages["avg_train_loss"], "val_accuracy": val_accuracy}


def _update_best_model(
    *,
    epoch_metrics: dict,
    epoch: int,
    model: nn.Module,
    best_val_loss: float,
    best_model_state: Optional[dict],
    best_epoch: int,
    best_epoch_metrics: Optional[dict[str, float]],
    best_model_metric: BestModelMetric,
    accuracy_floor: Optional[float],
) -> tuple[float, Optional[dict], int, Optional[dict[str, float]]]:
    val_loss = float(epoch_metrics["val_loss"])
    candidate_value = _get_best_value_for_epoch(
        epoch_metrics=epoch_metrics,
        best_model_metric=best_model_metric,
        accuracy_floor=accuracy_floor,
    )

    # Maintain original behavior for reporting even when selecting by a different metric.
    if val_loss < best_val_loss:
        best_val_loss = val_loss

    if candidate_value is None:
        # Only happens for gen_tokens_easy_with_accuracy_floor when epoch is below floor.
        logger.info(
            "Best-model selection skipped epoch %d for metric=%s (accuracy_easy below floor=%.4f). "
            "val_loss=%.4f, gen/accuracy_easy=%.4f",
            epoch,
            best_model_metric,
            float(accuracy_floor or 0.0),
            val_loss,
            float((epoch_metrics.get("gen_metrics") or {}).get("gen/accuracy_easy", float("nan"))),
        )
        return best_val_loss, best_model_state, best_epoch, best_epoch_metrics

    should_update = False
    if best_epoch_metrics is None:
        should_update = True
    else:
        prev_value = _get_best_value_for_epoch(
            epoch_metrics={"val_loss": best_epoch_metrics.get("val_loss"), "gen_metrics": best_epoch_metrics},
            best_model_metric=best_model_metric,
            accuracy_floor=accuracy_floor,
        )
        # If prev_value is None (e.g. previous best was invalid under floor), always update.
        should_update = (prev_value is None) or (float(candidate_value) < float(prev_value))

    if should_update:
        gen_metrics = epoch_metrics.get("gen_metrics") or {}
        best_epoch_metrics = {
            "val_loss": val_loss,
            "gen/accuracy": float(gen_metrics.get("gen/accuracy", float("nan"))),
            "gen/accuracy_easy": float(gen_metrics.get("gen/accuracy_easy", float("nan"))),
            "gen/avg_tokens_easy": float(gen_metrics.get("gen/avg_tokens_easy", float("nan"))),
            "gen/tpca": float(gen_metrics.get("gen/tpca", float("nan"))),
        }
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch
        logger.info(
            "New best model by metric=%s at epoch %d (value=%.4f). "
            "val_loss=%.4f, gen/accuracy=%.4f, gen/accuracy_easy=%.4f, gen/avg_tokens_easy=%.1f, gen/tpca=%.1f",
            best_model_metric,
            epoch,
            float(candidate_value),
            best_epoch_metrics["val_loss"],
            best_epoch_metrics["gen/accuracy"],
            best_epoch_metrics["gen/accuracy_easy"],
            best_epoch_metrics["gen/avg_tokens_easy"],
            best_epoch_metrics["gen/tpca"],
        )

    return best_val_loss, best_model_state, best_epoch, best_epoch_metrics
