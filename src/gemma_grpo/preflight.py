from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from packaging.version import InvalidVersion, Version

from .config import ExperimentConfig
from .nf4 import assert_compressed_w4_checkpoint, assert_nf4_checkpoint
from .tasks import read_jsonl
from .tokenization import assert_prefix_preserving
from .vllm_server import vllm_environment


PACKAGES = (
    "swegemma",
    "adk-submission",
    "adk-eval-core",
    "transformers",
    "peft",
    "torch",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "vllm",
)
MINIMUM_PACKAGE_VERSIONS = {
    "accelerate": "1.10",
    "bitsandbytes": "0.48.1",
    "peft": "0.17",
    "safetensors": "0.6",
    "transformers": "5.0",
}
ALLOWED_TOOLS = {
    "run_command",
    "submit_patch",
    "get_status",
    "read_file",
    "edit_file",
    "write_file",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
}


class _IncludeLoader(yaml.SafeLoader):
    pass


_IncludeLoader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))


def _valid_agent_name(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is not None


def run_preflight(
    *,
    config: ExperimentConfig,
    tasks_path: Path | None,
    snapshots_dir: Path | None,
    submission_dir: Path,
    model_path: str | None,
    rollout_model_path: str | None,
    check_tokenizer: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    executable = shutil.which("swegemma")
    record("swegemma executable", executable is not None, executable or "not found on PATH")
    if executable:
        completed = subprocess.run(
            [executable, "eval", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        required_flags = [
            "--tasks",
            "--snapshots-dir",
            "--submission-dir",
            "--results-dir",
            "--task-id",
            "--max-tool-calls",
            "--max-time-minutes",
        ]
        if config.rollout.timeout_seconds is not None:
            required_flags.append("--timeout-seconds")
        if config.rollout.image:
            required_flags.append("--image")
        missing_flags = [flag for flag in required_flags if flag not in completed.stdout]
        record(
            "swegemma CLI flags",
            completed.returncode == 0 and not missing_flags,
            "ok" if not missing_flags else f"missing: {', '.join(missing_flags)}",
        )

    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        minimum = MINIMUM_PACKAGE_VERSIONS.get(package)
        compatible = version != "not installed"
        detail = version
        if compatible and minimum is not None:
            try:
                compatible = Version(version) >= Version(minimum)
            except InvalidVersion:
                compatible = False
            detail = f"{version}; required >= {minimum}"
        record(f"package {package}", compatible, detail)

    required_submission = (
        submission_dir / "agent.yaml",
        submission_dir / "prompts" / "system.md",
        submission_dir / "configs" / "sampling.yaml",
    )
    missing = [str(path) for path in required_submission if not path.is_file()]
    record("submission files", not missing, "ok" if not missing else f"missing: {missing}")
    if not missing:
        try:
            agent = yaml.load(
                (submission_dir / "agent.yaml").read_text(encoding="utf-8"), Loader=_IncludeLoader
            )
            tools = agent.get("tools", []) if isinstance(agent, dict) else []
            name = agent.get("name") if isinstance(agent, dict) else None
            record(
                "submission agent name",
                _valid_agent_name(name),
                str(name) if _valid_agent_name(name) else f"invalid Python identifier: {name!r}",
            )
            unknown_tools = sorted(tool for tool in tools if isinstance(tool, str) and tool not in ALLOWED_TOOLS)
            record(
                "submission tools",
                not unknown_tools,
                "ok" if not unknown_tools else f"not in SWEGemma registry: {unknown_tools}",
            )
            eval_config = yaml.safe_load(
                (submission_dir / "eval_config.yaml").read_text(encoding="utf-8")
            )
            evaluation = eval_config.get("evaluation") if isinstance(eval_config, dict) else None
            record(
                "eval_config schema",
                isinstance(evaluation, dict),
                "evaluation key present" if isinstance(evaluation, dict) else "missing top-level evaluation mapping",
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            record("submission YAML", False, str(error))

    if tasks_path is not None:
        try:
            tasks = read_jsonl(tasks_path)
            record("tasks.jsonl", bool(tasks), f"{len(tasks)} task(s)")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record("tasks.jsonl", False, str(error))
    if snapshots_dir is not None:
        record("snapshots directory", snapshots_dir.is_dir(), str(snapshots_dir))

    record("LoRA rank", 0 < config.lora.r <= 128, f"r={config.lora.r}; competition maximum is 128")
    record(
        "group size",
        config.grpo.group_size == len(config.rollout.seeds) and config.grpo.group_size >= 2,
        f"group_size={config.grpo.group_size}, seeds={len(config.rollout.seeds)}",
    )

    path = Path(model_path or config.model.name_or_path)
    if config.model.load_mode == "nf4":
        try:
            assert_nf4_checkpoint(
                path,
                require_fsdp_storage=config.distributed.strategy == "fsdp",
            )
            record("saved NF4 checkpoint", True, str(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record("saved NF4 checkpoint", False, str(error))
        if config.rollout.managed_vllm:
            rollout_path = rollout_model_path or model_path or config.model.name_or_path
            try:
                if config.rollout.vllm_checkpoint_format == "nf4":
                    assert_nf4_checkpoint(rollout_path, require_fsdp_storage=True)
                    check_name = "shared rollout NF4 checkpoint"
                else:
                    assert_compressed_w4_checkpoint(rollout_path)
                    check_name = "rollout QAT W4A16 checkpoint"
                record(
                    check_name,
                    True,
                    str(Path(rollout_path)),
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                check_name = (
                    "shared rollout NF4 checkpoint"
                    if config.rollout.vllm_checkpoint_format == "nf4"
                    else "rollout QAT W4A16 checkpoint"
                )
                record(check_name, False, str(error))

    try:
        import torch

        gpu_count = torch.cuda.device_count()
        required_gpus = max(
            config.distributed.num_processes
            if config.distributed.strategy == "fsdp"
            else 1,
            (
                config.rollout.vllm_tensor_parallel_size
                * config.rollout.vllm_pipeline_parallel_size
                if config.rollout.managed_vllm
                else 1
            ),
        )
        record(
            "CUDA device count",
            gpu_count >= required_gpus,
            f"visible={gpu_count}, required={required_gpus}",
        )
    except ImportError as error:
        record("CUDA device count", False, str(error))

    if config.rollout.managed_vllm:
        uses_nf4 = config.rollout.vllm_checkpoint_format == "nf4"
        environment = vllm_environment(enable_nf4_compat=uses_nf4)
        if uses_nf4:
            builtin = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from vllm.model_executor.layers.quantization.bitsandbytes "
                    "import BitsAndBytesConfig; "
                    "config=BitsAndBytesConfig.from_config({"
                    "'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', "
                    "'bnb_4bit_quant_storage': 'bfloat16'}); "
                    "assert config.bnb_4bit_quant_storage == 'uint8'; print('ok')",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            builtin_ok = builtin.returncode == 0 and builtin.stdout.strip().endswith("ok")
            record(
                "vLLM built-in BitsAndBytes",
                builtin_ok,
                "ok" if builtin_ok else builtin.stdout[-2000:],
            )
        completed = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        required_flags = [
            "--tensor-parallel-size",
            "--pipeline-parallel-size",
            "--max-model-len",
            "--max-num-seqs",
            "--served-model-name",
            "--reasoning-parser",
            "--reasoning-config",
            "--enable-lora",
            "--lora-modules",
            "--max-lora-rank",
            "--max-loras",
        ]
        if uses_nf4:
            required_flags.extend(["--quantization", "--load-format"])
        if config.rollout.vllm_language_model_only:
            required_flags.append("--language-model-only")
        if config.rollout.vllm_enforce_eager:
            required_flags.append("--enforce-eager")
        missing = [flag for flag in required_flags if flag not in completed.stdout]
        record(
            f"managed vLLM {config.rollout.vllm_checkpoint_format}/LoRA flags",
            completed.returncode == 0 and not missing,
            "ok" if not missing else f"missing: {', '.join(missing)}",
        )

    if check_tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=config.model.trust_remote_code)
            assert_prefix_preserving(tokenizer)
            record("chat template prefix preservation", True, str(path))
        except Exception as error:  # reports model/template incompatibility without hiding its type
            record("chat template prefix preservation", False, f"{type(error).__name__}: {error}")

    return {"ok": all(check["ok"] for check in checks), "checks": checks}

