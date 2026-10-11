from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .collect import collect_rollouts
from .config import ExperimentConfig
from .policy import ADAPTER_FILES
from .tasks import read_jsonl
from .train import train_adapter
from .vllm_server import rollout_models_yaml


STATE_VERSION = 1


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _release_training_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _config_for_collection(config: ExperimentConfig, collection_index: int) -> ExperimentConfig:
    # A policy that produced no diverse rewards must be sampled again rather
    # than optimized on a zero-advantage group. Shift seeds automatically.
    seed_offset = collection_index * 10_000
    rollout = replace(
        config.rollout,
        seeds=tuple(seed + seed_offset for seed in config.rollout.seeds),
    )
    return replace(config, rollout=rollout)


def plan_training_config(
    config: ExperimentConfig, *, trainable_group_count: int, remaining_steps: int
) -> ExperimentConfig:
    """Choose whole-group epochs without exceeding the requested step target."""
    available_groups = trainable_group_count
    if config.grpo.max_steps is not None:
        available_groups = min(available_groups, config.grpo.max_steps)
    if available_groups < 1 or remaining_steps < 1:
        raise ValueError("A training plan needs at least one group and one remaining step")

    default_steps = available_groups * config.grpo.epochs
    if default_steps <= remaining_steps:
        return replace(config, grpo=replace(config.grpo, max_steps=available_groups))
    if available_groups >= remaining_steps:
        return replace(
            config,
            grpo=replace(config.grpo, epochs=1, max_steps=remaining_steps),
        )
    epochs = max(1, min(config.grpo.epochs, remaining_steps // available_groups))
    return replace(
        config,
        grpo=replace(config.grpo, epochs=epochs, max_steps=available_groups),
    )


def _infrastructure_failure(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    if not any(bool(row.get("scorable")) for row in rows):
        return True
    # Be defensive when reading rollouts produced by an older collector which
    # could misclassify a SWEGemma worker error as a scorable empty patch.
    return any(
        bool(str((row.get("result") or {}).get("error") or "").strip())
        for row in rows
        if isinstance(row.get("result"), dict)
    )


def _failure_tail(rows: list[dict[str, Any]]) -> str:
    snippets = []
    for row in rows:
        tail = str(row.get("swegemma_output_tail") or "").strip()
        if tail and tail not in snippets:
            snippets.append(tail)
    return "\n\n".join(snippets)[-6000:]


def _new_state(*, initial_adapter: Path | None, target_steps: int) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "target_optimizer_steps": target_steps,
        "optimizer_steps": 0,
        "next_collection": 0,
        "next_update": 1,
        "current_adapter": str(initial_adapter) if initial_adapter else None,
        "pending_rollout": None,
        "history": [],
        "status": "running",
    }


def run_training_loop(
    *,
    config: ExperimentConfig,
    tasks_path: Path,
    snapshots_dir: Path,
    submission_dir: Path,
    work_dir: Path,
    model_path: str | None,
    rollout_model_path: str | None,
    selected_ids: set[str] | None,
    initial_adapter: Path | None,
    models_yaml: Path | None,
    target_steps: int,
    max_collections: int,
    resume: bool,
) -> dict[str, Any]:
    if target_steps < 1:
        raise ValueError("target_steps must be positive")
    if max_collections < 1:
        raise ValueError("max_collections must be positive")
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "loop_state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"{state_path} already exists; pass --resume to continue")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"Unsupported loop state version: {state.get('version')}")
        state["target_optimizer_steps"] = target_steps
        state["status"] = "running"
    else:
        state = _new_state(initial_adapter=initial_adapter, target_steps=target_steps)
        _write_state(state_path, state)

    while int(state["optimizer_steps"]) < target_steps:
        if int(state["next_collection"]) >= max_collections:
            state["status"] = "max_collections_reached"
            _write_state(state_path, state)
            break

        pending = state.get("pending_rollout")
        if pending is None:
            collection_index = int(state["next_collection"])
            rollout_path = work_dir / "rollouts" / f"collection_{collection_index:04d}.jsonl"
            current_adapter = (
                Path(state["current_adapter"]) if state.get("current_adapter") else None
            )
            collection_config = _config_for_collection(config, collection_index)
            print(
                f"[loop] collecting #{collection_index} with policy "
                f"{current_adapter or 'base'}; steps={state['optimizer_steps']}/{target_steps}",
                flush=True,
            )
            with rollout_models_yaml(
                config=collection_config,
                model_path=rollout_model_path or model_path or config.model.name_or_path,
                adapter=current_adapter,
                work_dir=work_dir / "vllm" / f"collection_{collection_index:04d}",
                external_models_yaml=models_yaml,
            ) as active_models_yaml:
                rows = collect_rollouts(
                    config=collection_config,
                    tasks_path=tasks_path,
                    snapshots_dir=snapshots_dir,
                    submission_dir=submission_dir,
                    results_root=work_dir / "results",
                    output_path=rollout_path,
                    iteration=collection_index,
                    selected_ids=selected_ids,
                    adapter=current_adapter,
                    models_yaml=active_models_yaml,
                )
            trainable_groups = sorted(
                {str(row["group_id"]) for row in rows if row.get("trainable")}
            )
            pending = {
                "collection": collection_index,
                "path": str(rollout_path),
                "trainable_groups": len(trainable_groups),
                "policy_adapter": str(current_adapter) if current_adapter else None,
            }
            state["pending_rollout"] = pending
            state["next_collection"] = collection_index + 1
            _write_state(state_path, state)
            print(
                f"[loop] collection #{collection_index}: "
                f"{len(trainable_groups)} trainable group(s)",
                flush=True,
            )

        rollout_path = Path(pending["path"])
        rows = read_jsonl(rollout_path)
        if _infrastructure_failure(rows):
            diagnostic = _failure_tail(rows)
            state["status"] = "infrastructure_error"
            state["history"].append(
                {
                    "collection": pending["collection"],
                    "rollout": str(rollout_path),
                    "status": "infrastructure_error",
                    "optimizer_steps_total": state["optimizer_steps"],
                }
            )
            _write_state(state_path, state)
            raise RuntimeError(
                f"Collection #{pending['collection']} produced zero scorable rollouts. "
                f"SWEGemma output tail:\n{diagnostic or '(empty)'}"
            )
        trainable_group_count = len(
            {str(row["group_id"]) for row in rows if row.get("trainable")}
        )
        if trainable_group_count == 0:
            print(
                f"[loop] collection #{pending['collection']} has no reward variance; resampling",
                flush=True,
            )
            state["history"].append(
                {
                    "collection": pending["collection"],
                    "rollout": str(rollout_path),
                    "status": "no_reward_variance",
                    "optimizer_steps_total": state["optimizer_steps"],
                }
            )
            state["pending_rollout"] = None
            _write_state(state_path, state)
            continue

        remaining_steps = target_steps - int(state["optimizer_steps"])
        training_config = plan_training_config(
            config,
            trainable_group_count=trainable_group_count,
            remaining_steps=remaining_steps,
        )
        update_index = int(state["next_update"])
        output_adapter = work_dir / "adapters" / f"update_{update_index:04d}"
        resume_adapter = (
            Path(state["current_adapter"]) if state.get("current_adapter") else None
        )
        planned_steps = int(training_config.grpo.max_steps or trainable_group_count) * int(
            training_config.grpo.epochs
        )
        print(
            f"[loop] training update #{update_index}: groups={training_config.grpo.max_steps}, "
            f"epochs={training_config.grpo.epochs}, planned_steps={planned_steps}",
            flush=True,
        )
        metrics_path = output_adapter / "training_metrics.json"
        if metrics_path.is_file() and all(
            (output_adapter / name).is_file() for name in ADAPTER_FILES
        ):
            # Recovery for interruption after adapter save but before loop-state save.
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            if output_adapter.exists() and any(output_adapter.iterdir()):
                raise RuntimeError(
                    f"Incomplete adapter checkpoint at {output_adapter}. "
                    "Inspect it, then remove that directory before --resume."
                )
            summary = train_adapter(
                config=training_config,
                rollout_paths=[rollout_path],
                output_dir=output_adapter,
                model_path=model_path,
                resume_adapter=resume_adapter,
            )
        _release_training_memory()

        step_count = int(summary["optimizer_steps"])
        state["optimizer_steps"] = int(state["optimizer_steps"]) + step_count
        state["current_adapter"] = str(output_adapter)
        state["next_update"] = update_index + 1
        state["history"].append(
            {
                "collection": pending["collection"],
                "rollout": str(rollout_path),
                "status": "trained",
                "adapter": str(output_adapter),
                "optimizer_steps": step_count,
                "optimizer_steps_total": state["optimizer_steps"],
                "groups_processed": summary["groups_processed"],
            }
        )
        state["pending_rollout"] = None
        _write_state(state_path, state)
        print(
            f"[loop] update #{update_index} complete; "
            f"steps={state['optimizer_steps']}/{target_steps}",
            flush=True,
        )

    if int(state["optimizer_steps"]) >= target_steps:
        state["status"] = "complete"
        _write_state(state_path, state)
    return state

