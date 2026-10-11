from pathlib import Path

import pytest

from gemma_grpo.config import load_config


def test_default_config_has_matching_group_and_seeds():
    config = load_config(Path(__file__).parents[1] / "configs" / "grpo.yaml")
    assert config.grpo.group_size == 4
    assert len(config.rollout.seeds) == 4
    assert config.lora.r == 16
    assert config.model.device_map == "balanced"
    assert config.model.max_memory_per_gpu_gib == 22
    assert config.model.gpu0_reserve_gib == 8
    assert config.model.other_gpu_reserve_gib == 4
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

