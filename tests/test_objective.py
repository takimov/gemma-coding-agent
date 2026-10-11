import torch

from gemma_grpo.objective import grpo_loss


def test_zero_advantage_and_matching_reference_has_zero_loss():
    values = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss, kl = grpo_loss(
        values,
        values.detach(),
        values.detach(),
        advantage=0.0,
        clip_epsilon=0.2,
        beta=0.01,
    )
    assert torch.isclose(loss, torch.tensor(0.0))
    assert torch.isclose(kl, torch.tensor(0.0))


def test_positive_advantage_produces_policy_gradient():
    new = torch.tensor([-1.0], requires_grad=True)
    loss, _ = grpo_loss(
        new,
        torch.tensor([-1.0]),
        torch.tensor([-1.0]),
        advantage=1.0,
        clip_epsilon=0.2,
        beta=0.0,
    )
    loss.backward()
    assert new.grad is not None
    assert new.grad.item() < 0

