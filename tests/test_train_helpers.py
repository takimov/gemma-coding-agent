import torch
from types import SimpleNamespace

import pytest

from gemma_grpo.tokenization import TokenizedTrajectory
from gemma_grpo.train import (
    _build_selective_logprob_model,
    _distributed_rollout_logprobs,
    _no_split_module_classes,
    _next_telemetry_path,
    _policy_state_dict,
    _project_action_logprobs,
    _selected_logprobs,
    _stored_behavior_logprobs,
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


def test_stored_behavior_logprobs_are_validated():
    row = {"seed": 101, "behavior_logprobs": [-0.1, -0.2]}

    assert _stored_behavior_logprobs(row, expected_tokens=2) == [-0.1, -0.2]
    with pytest.raises(ValueError, match="expected 3"):
        _stored_behavior_logprobs(row, expected_tokens=3)


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


def test_selective_wrapper_keeps_projection_inside_model_forward():
    policy = _TinyPeftLike()
    wrapped = _build_selective_logprob_model(policy, chunk_size=1)
    trajectory = TokenizedTrajectory(input_ids=[2, 4, 6, 8], action_positions=[1, 3])

    actual = _selected_logprobs(wrapped, trajectory, chunk_size=999)
    expected = _selected_logprobs(policy, trajectory, chunk_size=1)

    assert torch.allclose(actual, expected)


def test_distributed_logprobs_match_local_path_for_one_rank():
    policy = _TinyPeftLike()
    wrapped = _build_selective_logprob_model(policy, chunk_size=1)
    segments = [
        TokenizedTrajectory(input_ids=[2, 4, 6], action_positions=[1, 2]),
        TokenizedTrajectory(input_ids=[1, 3], action_positions=[1]),
    ]

    actual, padding_anchor = _distributed_rollout_logprobs(
        wrapped,
        segments,
        chunk_size=1,
        input_device=torch.device("cpu"),
        world_size=1,
        dummy_token_id=0,
    )
    expected = torch.cat(
        [_selected_logprobs(wrapped, segment, chunk_size=1) for segment in segments]
    )

    assert torch.allclose(actual, expected)
    assert padding_anchor.item() == 0.0


def test_policy_state_dict_removes_fsdp_wrapper_prefix():
    value = torch.tensor([1.0])
    assert _policy_state_dict({"policy.base_model.lora_A": value}) == {
        "base_model.lora_A": value
    }
