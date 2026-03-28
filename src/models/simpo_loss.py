"""
SimPO (Simple Preference Optimization) loss with optional budget-aware length penalty.

SimPO reward: R(x,y) = (beta / |y|) * sum(log pi(y_t | y_<t, x)) - gamma
No reference model needed — uses length-normalized log-prob as implicit reward.

Budget-aware variant adds per-complexity length penalty on top.
"""

import torch
import torch.nn.functional as F


def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    complexities: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
    lambda_easy: float = 0.0,
    lambda_hard: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    SimPO loss with optional budget-aware length penalty.

    Note: policy_chosen_logps are already per-token averaged by log_prob(),
    so they ARE the length-normalized log-probs SimPO wants.

    R_chosen = beta * policy_chosen_logps - gamma
    R_rejected = beta * policy_rejected_logps - gamma
    loss = -log(sigmoid(R_chosen - R_rejected))

    With budget penalty:
    loss = -log(sigmoid(R_chosen - R_rejected - length_penalty))
    """
    # SimPO reward difference (gamma cancels out in the difference)
    reward_diff = beta * (policy_chosen_logps - policy_rejected_logps)

    # Optional budget-aware length penalty
    length_penalty_mean = 0.0
    if lambda_easy > 0.0 or lambda_hard > 0.0:
        lambdas = torch.where(
            complexities == 0,
            torch.full_like(complexities, lambda_easy, dtype=torch.float32),
            torch.full_like(complexities, lambda_hard, dtype=torch.float32),
        ).to(policy_chosen_logps.device)

        avg_len = (chosen_lengths.float() + rejected_lengths.float()) / 2.0
        length_diff = (chosen_lengths.float() - rejected_lengths.float()) / avg_len.clamp(min=1)
        length_penalty = lambdas * length_diff
        reward_diff = reward_diff - length_penalty
        length_penalty_mean = length_penalty.detach().mean().item()

    # Apply gamma as margin (shifts the sigmoid)
    reward_diff = reward_diff - gamma

    loss = -F.logsigmoid(reward_diff).mean()

    extra = {"length_penalty": length_penalty_mean, "simpo_gamma": gamma}
    return loss, extra
