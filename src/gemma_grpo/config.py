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


@dataclass(frozen=True)
class GrpoConfig:
    learning_rate: float = 5e-6
    beta: float = 0.01
    clip_epsilon: float = 0.2
    group_size: int = 4
    epochs: int = 2
    max_steps: int | None = None
    max_seq_length: int = 32768
    max_grad_norm: float = 1.0
    seed: int = 42


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
    load_mode: str = "native"
    device_map: str = "balanced"
    max_memory_per_gpu_gib: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    lora: LoraConfigData
    grpo: GrpoConfig
    rollout: RolloutConfig
    reward: RewardConfig


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
    if not 0 < config.lora.r <= 128:
        raise ValueError("lora.r must be between 1 and the competition maximum of 128")
    if config.lora.dropout != 0.0:
        raise ValueError("lora.dropout must be 0 for deterministic old/new policy log-probabilities")
    if config.model.load_mode not in {"native", "bnb4"}:
        raise ValueError("model.load_mode must be 'native' or 'bnb4'")
    if config.model.device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        raise ValueError(
            "model.device_map must be one of: auto, balanced, balanced_low_0, sequential"
        )
    if (
        config.model.max_memory_per_gpu_gib is not None
        and config.model.max_memory_per_gpu_gib < 1
    ):
        raise ValueError("model.max_memory_per_gpu_gib must be positive")
    return config

