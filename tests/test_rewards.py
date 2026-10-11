from gemma_grpo.config import RewardConfig
from gemma_grpo.rewards import score_result


CONFIG = RewardConfig(resolved=1.0, failed=0.0, empty_patch=-0.25, tool_call_penalty=0.01)


def test_success_gets_binary_reward_and_efficiency_penalty():
    outcome = score_result({"verification": {"resolved": True}}, patch="diff", tool_calls=3, config=CONFIG)
    assert outcome.scorable
    assert outcome.resolved is True
    assert outcome.reward == 0.97


def test_failed_nonempty_patch_is_zero_without_tool_penalty():
    outcome = score_result({"resolved": False}, patch="diff", tool_calls=99, config=CONFIG)
    assert outcome.reward == 0.0
    assert outcome.reason == "tests_failed"


def test_empty_patch_is_negative():
    outcome = score_result({"metrics": {"resolved": False}}, patch="", tool_calls=1, config=CONFIG)
    assert outcome.reward == -0.25


def test_no_patch_sentinel_is_negative():
    outcome = score_result({"resolved": False}, patch="NO_PATCH", tool_calls=1, config=CONFIG)
    assert outcome.reward == -0.25


def test_infrastructure_error_is_not_a_negative_example():
    outcome = score_result(
        {"status": "sandbox_error", "resolved": False}, patch="", tool_calls=0, config=CONFIG
    )
    assert not outcome.scorable
    assert outcome.reward is None


def test_worker_error_takes_precedence_over_partial_resolved_result():
    outcome = score_result(
        {
            "error": "Unexpected evaluation worker error: DockerException",
            "metrics": {"resolved": False},
        },
        patch="",
        tool_calls=0,
        config=CONFIG,
    )
    assert not outcome.scorable
    assert outcome.reward is None
    assert outcome.reason == "infrastructure_error"

