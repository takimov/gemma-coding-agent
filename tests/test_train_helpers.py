import torch
from types import SimpleNamespace

import pytest

from gemma_grpo.tokenization import TokenizedTrajectory
from gemma_grpo.train import (
    _no_split_module_classes,
    _next_telemetry_path,
    _project_action_logprobs,
    _selected_logprobs,
    _validate_trajectory,
)


class _BaseModel:
    _no_split_modules = [{"Gemma4DecoderLayer"}, ("Gemma4TextScaledWordEmbedding",)]


class _PeftWrapper:
    def get_base_model(self):
        return _BaseModel()


def test_no_split_module_classes_flattens_peft_collections():
    assert _no_split_module_classes(_PeftWrapper()) == [
        "Gemma4DecoderLayer",
        "Gemma4TextScaledWordEmbedding",
    ]


def test_chunked_action_logprobs_match_full_projection_and_backpropagate():
    torch.manual_seed(7)
    head = torch.nn.Linear(3, 11, bias=False)
    hidden = torch.randn(1, 5, 3, requires_grad=True)
    labels = torch.tensor([[1, 3, 5, 7, 9]])

    actual = _project_action_logprobs(head, hidden, labels, chunk_size=2)
    expected = (
        torch.log_softmax(head(hidden).float(), dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(0)
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_chunked_action_logprobs_preserve_gemma_final_softcapping():
    torch.manual_seed(8)
    head = torch.nn.Linear(3, 11, bias=False)
    hidden = torch.randn(1, 5, 3, requires_grad=True)
    labels = torch.tensor([[1, 3, 5, 7, 9]])
    cap = 2.5

    actual = _project_action_logprobs(
        head,
        hidden,
        labels,
        chunk_size=2,
        final_logit_softcapping=cap,
    )
    logits = torch.tanh(head(hidden) / cap) * cap
    expected = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(0)
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)


def test_trajectory_positions_are_validated_before_model_loading():
    with pytest.raises(ValueError, match="between 1"):
        _validate_trajectory(TokenizedTrajectory(input_ids=[1, 2], action_positions=[2]))


def test_telemetry_file_is_not_overwritten(tmp_path):
    output = tmp_path / "adapter"
    first = tmp_path / "adapter.telemetry.jsonl"
    first.write_text("old\n", encoding="utf-8")

    assert _next_telemetry_path(output) == tmp_path / "adapter.telemetry-v2.jsonl"


class _TinyBackbone(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyCausalLm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 4)
        self.model = _TinyBackbone(self.embedding)
        self.lm_head = torch.nn.Linear(4, 13, bias=False)
        self.config = SimpleNamespace(final_logit_softcapping=None)

    def get_output_embeddings(self):
        return self.lm_head


class _TinyPeftLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _TinyCausalLm()

    def get_base_model(self):
        return self.base

    def get_input_embeddings(self):
        return self.base.embedding


def test_selected_logprobs_use_previous_positions_without_full_causal_lm_logits():
    model = _TinyPeftLike()
    trajectory = TokenizedTrajectory(input_ids=[2, 4, 6, 8], action_positions=[1, 3])

    actual = _selected_logprobs(model, trajectory, chunk_size=1)
    hidden = model.base.embedding(torch.tensor([[2, 4, 6, 8]]))
    logits = model.base.lm_head(hidden[:, [0, 2], :])
    labels = torch.tensor([[4, 8]])
    expected = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(0)
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)
