from __future__ import annotations

from typing import Any


def grpo_loss(
    new_logprobs: Any,
    old_logprobs: Any,
    reference_logprobs: Any,
    *,
    advantage: float,
    clip_epsilon: float,
    beta: float,
):
    """Return the mean clipped GRPO loss for already-selected action tokens."""
    import torch

    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    policy_loss = -torch.minimum(unclipped, clipped)

    # Schulman-style non-negative KL estimator, with r = p_ref / p_policy.
    log_ratio = reference_logprobs - new_logprobs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (policy_loss + beta * kl).mean(), kl.mean()

