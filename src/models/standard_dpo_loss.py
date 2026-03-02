"""
Standard DPO loss (no length penalty).
Implicit reward: R = beta * log(pi_theta(y|x)/pi_ref(y|x))
Loss: -log(sigma(R_chosen - R_rejected))
"""

import torch
import torch.nn.functional as F


def standard_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Standard DPO loss without length penalty."""
    log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
    log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
    reward_diff = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -F.logsigmoid(reward_diff).mean()
    return loss, {}
