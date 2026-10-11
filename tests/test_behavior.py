import sys
import threading
from types import SimpleNamespace

import pytest

from gemma_grpo.behavior import (
    _selected_prompt_logprobs,
    attach_behavior_logprobs,
    trajectory_fingerprint,
)
from gemma_grpo.tokenization import TokenizedTrajectory


def test_selected_prompt_logprobs_extract_observed_action_tokens():
    trajectory = TokenizedTrajectory(
        input_ids=[10, 20, 30, 40],
        action_positions=[1, 3],
    )
    prompt_logprobs = [
        None,
        {"20": {"logprob": -0.2, "rank": 1}},
        {"30": {"logprob": -0.3, "rank": 2}},
        {"40": {"logprob": -0.4, "rank": 1}},
    ]

    assert _selected_prompt_logprobs(prompt_logprobs, trajectory) == [-0.2, -0.4]


def test_selected_prompt_logprobs_reject_missing_observed_token():
    trajectory = TokenizedTrajectory(input_ids=[10, 20], action_positions=[1])

    with pytest.raises(ValueError, match="token 20"):
        _selected_prompt_logprobs([None, {"99": {"logprob": -1.0}}], trajectory)


def test_trajectory_fingerprint_covers_tokens_and_action_mask():
    first = [TokenizedTrajectory(input_ids=[1, 2], action_positions=[1])]
    second = [TokenizedTrajectory(input_ids=[1, 2], action_positions=[])]

    assert trajectory_fingerprint(first).startswith("tokens-sha256:")
    assert trajectory_fingerprint(first) != trajectory_fingerprint(second)


def test_behavior_scoring_runs_trainable_rows_concurrently(monkeypatch, tmp_path):
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    monkeypatch.setattr(
        "gemma_grpo.behavior._row_segments",
        lambda tokenizer, row: [TokenizedTrajectory([1, row["seed"]], [1])],
    )
    barrier = threading.Barrier(4)

    def fake_score(trajectory, **kwargs):
        barrier.wait(timeout=5)
        return [-float(trajectory.input_ids[-1])]

    monkeypatch.setattr("gemma_grpo.behavior._score_segment", fake_score)
    rows = [{"seed": seed, "trainable": True} for seed in (1, 2, 3, 4)]

    result = attach_behavior_logprobs(
        rows,
        tokenizer_path=tmp_path,
        api_base="http://localhost/v1",
        model="base",
        trust_remote_code=False,
        max_workers=4,
    )

    assert [row["behavior_logprobs"] for row in result] == [
        [-1.0],
        [-2.0],
        [-3.0],
        [-4.0],
    ]
