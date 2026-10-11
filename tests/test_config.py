from pathlib import Path

import pytest

from gemma_grpo.config import load_config


def test_default_config_has_matching_group_and_seeds():
    config = load_config(Path(__file__).parents[1] / "configs" / "grpo.yaml")
    assert config.grpo.group_size == 4
    assert len(config.rollout.seeds) == 4
    assert config.lora.r == 16
    assert config.lora.exclude_modules == ".*vision_tower.*"
    assert config.model.load_mode == "nf4"
    assert config.model.device_map is None
    assert config.distributed.strategy == "fsdp"
    assert config.distributed.num_processes == 4
    assert config.rollout.managed_vllm is True
    assert config.rollout.vllm_checkpoint_format == "compressed-tensors"
    assert config.rollout.vllm_tensor_parallel_size == 4
    assert config.rollout.vllm_pipeline_parallel_size == 1
    assert config.rollout.vllm_language_model_only is True
    assert config.rollout.vllm_max_model_len == 32768
    assert config.rollout.concurrency == 4
    assert config.rollout.vllm_max_num_seqs == 4
    assert config.rollout.vllm_enforce_eager is False
    assert config.grpo.logprob_chunk_size == 32
    assert config.grpo.monitor_interval_seconds == 5


def test_sft_config_uses_safe_offline_defaults():
    config = load_config(Path(__file__).parents[1] / "configs" / "sft.yaml")
    assert config.sft.learning_rate == 1e-5
    assert config.sft.epochs == 2
    assert config.sft.max_seq_length == 8192
    assert config.sft.max_examples_per_task == 2
    assert config.distributed.strategy == "fsdp"
    assert config.model.load_mode == "nf4"

    smoke = load_config(Path(__file__).parents[1] / "configs" / "sft-smoke.yaml")
    assert smoke.sft.max_steps == 1
    assert smoke.sft.epochs == 1
    assert smoke.lora.r == config.lora.r
    assert smoke.lora.alpha == config.lora.alpha


def test_qwen_teacher_config_discards_cross_tokenizer_ids():
    config = load_config(Path(__file__).parents[1] / "configs" / "teacher-qwen.yaml")

    assert config.rollout.vllm_served_model_name == "qwen3-coder-next-teacher"
    assert config.rollout.vllm_tool_call_parser == "qwen3_coder"
    assert config.rollout.vllm_reasoning_parser is None
    assert config.rollout.vllm_reasoning_start_str == "<think>"
    assert config.rollout.vllm_reasoning_end_str == "</think>"
    assert config.rollout.vllm_enable_thinking is False
    assert config.rollout.preserve_trace_token_ids is False
    assert config.rollout.max_tool_calls == 50
    assert config.rollout.vllm_tensor_parallel_size == 4
    assert config.rollout.vllm_max_model_len == 32768


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


def test_unknown_rollout_checkpoint_format_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
model: {name_or_path: model}
rollout: {vllm_checkpoint_format: int3}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vllm_checkpoint_format"):
        load_config(path)

