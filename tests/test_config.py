from pathlib import Path

import pytest

from gemma_grpo.config import load_config


def test_default_config_has_matching_group_and_seeds():
    config = load_config(Path(__file__).parents[1] / "configs" / "grpo.yaml")
    assert config.grpo.group_size == 4
    assert len(config.rollout.seeds) == 4
    assert config.lora.r == 16
    assert config.model.load_mode == "nf4"
    assert config.model.device_map is None
    assert config.distributed.strategy == "fsdp"
    assert config.distributed.num_processes == 4
    assert config.rollout.managed_vllm is True
    assert config.rollout.vllm_tensor_parallel_size == 1
    assert config.rollout.vllm_pipeline_parallel_size == 4
    assert config.grpo.logprob_chunk_size == 32
    assert config.grpo.monitor_interval_seconds == 5


def test_mismatched_group_size_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model: {name_or_path: model}
grpo: {group_size: 3}
rollout: {seeds: [1, 2]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="group_size"):
        load_config(path)


def test_fsdp_requires_equal_work_per_rank(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model: {name_or_path: model, load_mode: nf4, device_map: null}
distributed: {strategy: fsdp, num_processes: 3}
grpo: {group_size: 4}
rollout: {seeds: [1, 2, 3, 4]}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="divisible"):
        load_config(path)

