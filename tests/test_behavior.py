import pytest

from gemma_grpo.behavior import _selected_prompt_logprobs, trajectory_fingerprint
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
