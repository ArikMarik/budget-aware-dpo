"""
Budget-Aware DPO Loss: R_budget(x,y) = beta * log(pi_theta(y|x)/pi_ref(y|x)) - lambda(C) * |y|

- lambda high when C=0 (Easy): penalize length
- lambda near zero when C=1 (Hard): accuracy primary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_lambda(complexity: int, lambda_easy: float = 0.1, lambda_hard: float = 0.001) -> float:
    """Dynamic lambda: high for Easy (C=0), near zero for Hard (C=1)."""
    return lambda_easy if complexity == 0 else lambda_hard


def budget_aware_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    complexities: torch.Tensor,
    beta: float = 0.1,
    lambda_easy: float = 0.1,
    lambda_hard: float = 0.001,
) -> tuple[torch.Tensor, dict]:
    """
    Budget-Aware DPO loss with length penalty.

    Implicit reward: R_budget = beta * log_ratio - lambda(C) * |y|
    DPO uses: log_sigma(beta * (log_ratio_chosen - log_ratio_rejected) - lambda * (|y_chosen| - |y_rejected|))

    For preference: we want chosen > rejected in reward.
    So: (R_chosen - R_rejected) = beta*(log_ratio_c - log_ratio_r) - lambda*(|y_c| - |y_r|)
    For Easy: lambda high, so we penalize length difference. Short chosen gets boosted.
    """
    log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
    log_ratio_rejected = policy_rejected_logps - reference_rejected_logps

    # Lambda per sample based on complexity
    lambdas = torch.where(
        complexities == 0,
        torch.full_like(complexities, lambda_easy, dtype=torch.float32),
        torch.full_like(complexities, lambda_hard, dtype=torch.float32),
    ).to(policy_chosen_logps.device)

    # Length penalty term: penalize (chosen_len - rejected_len)
    # If chosen is shorter, this term is negative -> adds to reward difference (good)
    length_diff = chosen_lengths.float() - rejected_lengths.float()
    length_penalty = lambdas * length_diff

    # DPO implicit reward difference
    # rewards_chosen - rewards_rejected
    reward_diff = beta * (log_ratio_chosen - log_ratio_rejected) - length_penalty

    # DPO loss: -log(sigma(reward_diff))
    loss = -F.logsigmoid(reward_diff).mean()

    # Log length penalty for monitoring
    length_penalty_mean = length_penalty.detach().mean().item()

    return loss, {"length_penalty": length_penalty_mean}
