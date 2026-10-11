from __future__ import annotations

import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .behavior import attach_behavior_logprobs
from .config import ExperimentConfig
from .groups import add_group_advantages
from .policy import adapter_fingerprint
from .rewards import score_result
from .swegemma_cli import SwegemmaInvocation, run_eval
from .tasks import append_jsonl, read_jsonl, select_tasks, task_id, task_prompt
from .traces import (
    count_tool_calls,
    load_trace,
    normalize_trace,
    trace_token_segments,
    trace_tool_definitions,
)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _copy_adapter(source: Path, destination: Path) -> None:
    required = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Adapter {source} is missing: {', '.join(missing)}")
    destination.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copy2(source / name, destination / name)


def _prepare_submission(
    source: Path,
    destination: Path,
    *,
    seed: int,
    adapter: Path | None,
    max_tool_calls: int,
    max_time_minutes: int,
) -> None:
    shutil.copytree(source, destination)
    sampling_path = destination / "configs" / "sampling.yaml"
    sampling = yaml.safe_load(sampling_path.read_text(encoding="utf-8")) or {}
    sampling["seed"] = seed
    sampling_path.write_text(yaml.safe_dump(sampling, sort_keys=False), encoding="utf-8")

    eval_path = destination / "eval_config.yaml"
    eval_config = yaml.safe_load(eval_path.read_text(encoding="utf-8")) or {}
    evaluation = eval_config.setdefault("evaluation", {})
    evaluation["max_tool_calls"] = max_tool_calls
    evaluation["max_time_minutes"] = max_time_minutes
    eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False), encoding="utf-8")

    if adapter is None:
        return
    adapter_target = destination / "adapters" / "main_lora"
    _copy_adapter(adapter, adapter_target)
    agent_path = destination / "agent.yaml"
    lines = agent_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("adapter:"):
            lines[index] = "adapter: main_lora"
            replaced = True
            break
    if not replaced:
        model_index = next((index for index, line in enumerate(lines) if line.startswith("model:")), 1)
        lines.insert(model_index + 1, "adapter: main_lora")
    agent_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_result(results_dir: Path, expected_task_id: str) -> dict[str, Any]:
    path = results_dir / "task_results.jsonl"
    if not path.exists():
        return {
            "instance_id": expected_task_id,
            "status": "infrastructure_error",
            "error": "swegemma did not create task_results.jsonl",
        }
    rows = read_jsonl(path)
    for row in rows:
        candidate = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "")
        if candidate == expected_task_id:
            return row
    if len(rows) == 1:
        return rows[0]
    return {
        "instance_id": expected_task_id,
        "status": "infrastructure_error",
        "error": "task result missing from task_results.jsonl",
    }


def _find_artifact(directory: Path, task: str, suffix: str) -> Path | None:
    exact = directory / f"{task}{suffix}"
    if exact.exists():
        return exact
    safe = directory / f"{_safe_name(task)}{suffix}"
    if safe.exists():
        return safe
    candidates = sorted(directory.glob(f"*{suffix}")) if directory.exists() else []
    return candidates[0] if len(candidates) == 1 else None


def collect_rollouts(
    *,
    config: ExperimentConfig,
    tasks_path: Path,
    snapshots_dir: Path,
    submission_dir: Path,
    results_root: Path,
    output_path: Path,
    iteration: int,
    selected_ids: set[str] | None = None,
    adapter: Path | None = None,
    models_yaml: Path | None = None,
    behavior_model_path: Path | None = None,
    capture_behavior_logprobs: bool = True,
) -> list[dict[str, Any]]:
    tasks = select_tasks(read_jsonl(tasks_path), selected_ids)
    policy_id = adapter_fingerprint(adapter) if adapter is not None else "base"
    rows: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; use a new iteration or remove it explicitly"
        )

    for task in tasks:
        current_task_id = task_id(task)
        group_id = f"iter-{iteration:04d}:{current_task_id}"

        def collect_seed(seed: int) -> dict[str, Any]:
            started = time.monotonic()
            print(
                f"[rollout] seed={seed} started; task={current_task_id}",
                flush=True,
            )
            run_dir = results_root / f"iter_{iteration:04d}" / _safe_name(current_task_id) / f"seed_{seed}"
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to mix a fresh rollout with existing artifacts in {run_dir}"
                )
            with tempfile.TemporaryDirectory(prefix="gemma-grpo-submission-") as temporary:
                prepared_submission = Path(temporary) / "submission"
                _prepare_submission(
                    submission_dir,
                    prepared_submission,
                    seed=seed,
                    adapter=adapter,
                    max_tool_calls=config.rollout.max_tool_calls,
                    max_time_minutes=config.rollout.max_time_minutes,
                )
                completed = run_eval(
                    SwegemmaInvocation(
                        tasks_path=tasks_path,
                        snapshots_dir=snapshots_dir,
                        submission_dir=prepared_submission,
                        results_dir=run_dir,
                        task_id=current_task_id,
                        models_yaml=models_yaml,
                        sandbox=config.rollout.sandbox,
                        image=config.rollout.image,
                        max_tool_calls=config.rollout.max_tool_calls,
                        max_time_minutes=config.rollout.max_time_minutes,
                        timeout_seconds=config.rollout.timeout_seconds,
                        # This invocation contains exactly one task. Parallelism
                        # across GRPO seeds is owned by the outer executor.
                        concurrency=1,
                    )
                )

            result = _find_result(run_dir, current_task_id)
            patch_path = _find_artifact(run_dir / "patches", current_task_id, ".patch")
            patch = patch_path.read_text(encoding="utf-8") if patch_path else ""
            trace_path = _find_artifact(run_dir / "traces", f"trace_{current_task_id}", ".json")
            trace: Mapping[str, Any] = {}
            trace_error = ""
            if trace_path:
                try:
                    trace = load_trace(trace_path)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    trace_error = str(error)
            messages = normalize_trace(trace, fallback_user_prompt=task_prompt(task))
            tool_calls = sum(len(message.get("tool_calls", [])) for message in messages)
            if not messages:
                tool_calls = count_tool_calls(trace)
            outcome = score_result(result, patch=patch, tool_calls=tool_calls, config=config.reward)
            token_segments = trace_token_segments(trace)
            has_trainable_trace = bool(token_segments) or any(
                message.get("role") == "assistant" for message in messages
            )
            reward = outcome.reward if has_trainable_trace else None
            scorable = outcome.scorable and has_trainable_trace
            reward_reason = (
                outcome.reason if has_trainable_trace else "missing_trainable_trace"
            )
            row: dict[str, Any] = {
                "task_id": current_task_id,
                "group_id": group_id,
                "iteration": iteration,
                "seed": seed,
                "policy_id": policy_id,
                "behavior_checkpoint_format": config.rollout.vllm_checkpoint_format,
                "reward": reward,
                "resolved": outcome.resolved,
                "scorable": scorable,
                "reward_reason": reward_reason,
                "tool_calls": tool_calls,
                "patch": patch,
                "messages": messages,
                "tools": trace_tool_definitions(trace),
                "token_segments": token_segments,
                "token_source": "atif_token_ids" if token_segments else "chat_template_reconstruction",
                "trace_trainable": has_trainable_trace,
                "swegemma_returncode": completed.returncode,
                "swegemma_output_tail": completed.stdout[-4000:],
                "result": result,
                "result_dir": str(run_dir),
                "trace_path": str(trace_path) if trace_path else None,
                "trace_error": trace_error or None,
            }
            print(
                f"[rollout] seed={seed} completed in "
                f"{time.monotonic() - started:.1f}s; reward={reward}; "
                f"reason={reward_reason}; returncode={completed.returncode}",
                flush=True,
            )
            return row

        worker_count = min(config.rollout.concurrency, len(config.rollout.seeds))
        print(
            f"[rollout] collecting {len(config.rollout.seeds)} seed(s) with "
            f"concurrency={worker_count}",
            flush=True,
        )
        if worker_count == 1:
            task_rows = [collect_seed(seed) for seed in config.rollout.seeds]
        else:
            # executor.map preserves the configured seed order even though the
            # independent sandboxes finish out of order.
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="gemma-rollout",
            ) as executor:
                task_rows = list(executor.map(collect_seed, config.rollout.seeds))
        rows.extend(task_rows)

    normalized = add_group_advantages(rows)
    if (
        config.rollout.managed_vllm
        and config.rollout.vllm_checkpoint_format == "compressed-tensors"
        and capture_behavior_logprobs
    ):
        if behavior_model_path is None:
            raise ValueError(
                "Native W4 rollout requires behavior_model_path to capture exact "
                "behavior-policy log-probabilities"
            )
        normalized = attach_behavior_logprobs(
            normalized,
            tokenizer_path=behavior_model_path,
            api_base=f"http://127.0.0.1:{config.rollout.vllm_port}/v1",
            model="main_lora" if adapter is not None else "gemma-4-31b-it-qat-w4a16-ct",
            trust_remote_code=config.model.trust_remote_code,
            max_workers=config.rollout.concurrency,
        )
    for row in normalized:
        append_jsonl(output_path, row)
    return normalized

