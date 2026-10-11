from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RewardConfig:
    resolved: float = 1.0
    failed: float = 0.0
    empty_patch: float = -0.25
    tool_call_penalty: float = 0.0


@dataclass(frozen=True)
class RolloutConfig:
    sandbox: str = "docker"
    image: str | None = None
    concurrency: int = 1
    max_tool_calls: int = 20
    max_time_minutes: int = 15
    timeout_seconds: int | None = None
    seeds: tuple[int, ...] = (101, 202, 303, 404)
    managed_vllm: bool = True
    vllm_port: int = 8000
    vllm_tensor_parallel_size: int = 4
    vllm_gpu_memory_utilization: float = 0.9
    vllm_startup_timeout_seconds: int = 900


@dataclass(frozen=True)
class GrpoConfig:
    learning_rate: float = 5e-6
    beta: float = 0.01
    clip_epsilon: float = 0.2
    group_size: int = 4
    epochs: int = 2
    max_steps: int | None = None
    max_seq_length: int = 32768
    logprob_chunk_size: int = 64
    max_grad_norm: float = 1.0
    seed: int = 42
    monitor_interval_seconds: float = 5.0
    monitor_console_seconds: float = 30.0


@dataclass(frozen=True)
class LoraConfigData:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    trust_remote_code: bool = False
    torch_dtype: str = "bfloat16"
    load_mode: str = "nf4"
    device_map: str | None = None


@dataclass(frozen=True)
class DistributedConfig:
    strategy: str = "fsdp"
    num_processes: int = 4
    cpu_offload: bool = False
    limit_all_gathers: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    lora: LoraConfigData
    grpo: GrpoConfig
    rollout: RolloutConfig
    reward: RewardConfig
    distributed: DistributedConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping")

    model_raw = _section(raw, "model")
    if not model_raw.get("name_or_path"):
        raise ValueError("model.name_or_path is required")
    rollout_raw = _section(raw, "rollout")
    rollout_raw["seeds"] = tuple(int(seed) for seed in rollout_raw.get("seeds", (101, 202, 303, 404)))

    config = ExperimentConfig(
        model=ModelConfig(**model_raw),
        lora=LoraConfigData(**_section(raw, "lora")),
        grpo=GrpoConfig(**_section(raw, "grpo")),
        rollout=RolloutConfig(**rollout_raw),
        reward=RewardConfig(**_section(raw, "reward")),
        distributed=DistributedConfig(**_section(raw, "distributed")),
    )
    if config.grpo.group_size != len(config.rollout.seeds):
        raise ValueError(
            "grpo.group_size must equal len(rollout.seeds); "
            f"got {config.grpo.group_size} and {len(config.rollout.seeds)}"
        )
    if config.grpo.group_size < 2:
        raise ValueError("GRPO requires at least two rollouts per group")
    if config.rollout.sandbox not in {"docker", "subprocess"}:
        raise ValueError("rollout.sandbox must be 'docker' or 'subprocess'")
    if config.grpo.epochs < 1:
        raise ValueError("grpo.epochs must be at least 1")
    if config.grpo.max_steps is not None and config.grpo.max_steps < 1:
        raise ValueError("grpo.max_steps must be positive when set")
    if config.grpo.logprob_chunk_size < 1:
        raise ValueError("grpo.logprob_chunk_size must be positive")
    if config.grpo.monitor_interval_seconds <= 0:
        raise ValueError("grpo.monitor_interval_seconds must be positive")
    if config.grpo.monitor_console_seconds <= 0:
        raise ValueError("grpo.monitor_console_seconds must be positive")
    if not 0 < config.lora.r <= 128:
        raise ValueError("lora.r must be between 1 and the competition maximum of 128")
    if config.lora.dropout != 0.0:
        raise ValueError("lora.dropout must be 0 for deterministic old/new policy log-probabilities")
    if config.model.load_mode not in {"native", "nf4"}:
        raise ValueError("model.load_mode must be 'native' or 'nf4'")
    if config.model.device_map not in {
        None,
        "auto",
        "balanced",
        "balanced_low_0",
        "sequential",
    }:
        raise ValueError(
            "model.device_map must be null or one of: auto, balanced, balanced_low_0, sequential"
        )
    if config.distributed.strategy not in {"none", "fsdp"}:
        raise ValueError("distributed.strategy must be 'none' or 'fsdp'")
    if config.distributed.num_processes < 1:
        raise ValueError("distributed.num_processes must be positive")
    if config.distributed.strategy == "fsdp":
        if config.distributed.num_processes < 2:
            raise ValueError("FSDP requires at least two processes")
        if config.model.load_mode != "nf4":
            raise ValueError("FSDP training requires model.load_mode: nf4")
        if config.model.device_map is not None:
            raise ValueError("FSDP owns device placement; set model.device_map: null")
        if config.grpo.group_size % config.distributed.num_processes:
            raise ValueError(
                "grpo.group_size must be divisible by distributed.num_processes "
                "so every FSDP rank receives the same number of trajectories"
            )
    if not 1 <= config.rollout.vllm_port <= 65535:
        raise ValueError("rollout.vllm_port must be a valid TCP port")
    if config.rollout.vllm_tensor_parallel_size < 1:
        raise ValueError("rollout.vllm_tensor_parallel_size must be positive")
    if not 0 < config.rollout.vllm_gpu_memory_utilization < 1:
        raise ValueError("rollout.vllm_gpu_memory_utilization must be between 0 and 1")
    if config.rollout.vllm_startup_timeout_seconds < 1:
        raise ValueError("rollout.vllm_startup_timeout_seconds must be positive")
    return config

