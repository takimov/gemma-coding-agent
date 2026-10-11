from dataclasses import replace
from pathlib import Path

from gemma_grpo.config import load_config
from gemma_grpo.loop import _config_for_collection, _infrastructure_failure, plan_training_config


ROOT = Path(__file__).parents[1]


def test_collection_seeds_change_without_changing_group_size():
    config = load_config(ROOT / "configs" / "grpo.yaml")
    shifted = _config_for_collection(config, 3)
    assert shifted.rollout.seeds == tuple(seed + 30_000 for seed in config.rollout.seeds)
    assert shifted.grpo.group_size == len(shifted.rollout.seeds)


def test_plan_uses_default_epochs_when_target_has_room():
    config = load_config(ROOT / "configs" / "grpo.yaml")
    planned = plan_training_config(config, trainable_group_count=5, remaining_steps=100)
    assert planned.grpo.epochs == 2
    assert planned.grpo.max_steps == 5


def test_plan_does_not_overshoot_final_step_target():
    config = load_config(ROOT / "configs" / "grpo.yaml")
    planned = plan_training_config(config, trainable_group_count=10, remaining_steps=3)
    assert planned.grpo.epochs == 1
    assert planned.grpo.max_steps == 3


def test_plan_respects_existing_group_limit():
    config = load_config(ROOT / "configs" / "grpo.yaml")
    limited = replace(config, grpo=replace(config.grpo, max_steps=2))
    planned = plan_training_config(limited, trainable_group_count=10, remaining_steps=100)
    assert planned.grpo.max_steps == 2


def test_zero_scorable_rows_are_infrastructure_failure():
    assert _infrastructure_failure([{"scorable": False}, {"scorable": False}])
    assert not _infrastructure_failure([{"scorable": False}, {"scorable": True}])


def test_legacy_misscored_worker_error_is_infrastructure_failure():
    rows = [
        {
            "scorable": True,
            "reward_reason": "empty_patch",
            "result": {"error": "Unexpected evaluation worker error: DockerException"},
        }
    ]
    assert _infrastructure_failure(rows)

