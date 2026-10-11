from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .behavior import trajectory_fingerprint
from .config import ExperimentConfig
from .monitoring import TrainingMonitor
from .tasks import read_jsonl
from .tokenization import TokenizedTrajectory
from .train import (
    _NullMonitor,
    _broadcast_replicated_modules,
    _build_selective_logprob_model,
    _create_accelerator,
    _distributed_rollout_logprobs,
    _free_port,
    _load_model_and_tokenizer,
    _load_tokenizer,
    _next_telemetry_path,
    _policy_state_dict,
    _prepare_replicated_projection_modules,
    _row_segments,
    _wrapped_policy,
)


PreparedExample = tuple[dict[str, Any], list[TokenizedTrajectory]]


def _approved_for_sft(row: dict[str, Any]) -> bool:
    """Accept verified successes or explicitly curated expert trajectories."""
    return (row.get("resolved") is True and row.get("scorable", True) is not False) or (
        row.get("sft_approved") is True
    )


def prepare_sft_examples(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_seq_length: int,
    max_examples_per_task: int | None,
    seed: int,
) -> tuple[list[PreparedExample], list[dict[str, Any]]]:
    """Filter, tokenize, balance, and exactly deduplicate offline trajectories."""
    candidates = [dict(row) for row in rows if _approved_for_sft(row)]
    random.Random(seed).shuffle(candidates)
    per_task: defaultdict[str, int] = defaultdict(int)
    fingerprints: set[str] = set()
    prepared: list[PreparedExample] = []
    skipped: list[dict[str, Any]] = []

    for row in candidates:
        task = str(row.get("task_id") or row.get("instance_id") or row.get("group_id") or "")
        try:
            segments = _row_segments(tokenizer, row)
        except (KeyError, TypeError, ValueError) as error:
            skipped.append(
                {
                    "task_id": task,
                    "seed": row.get("seed"),
                    "reason": "invalid_trajectory",
                    "detail": str(error),
                }
            )
            continue
        lengths = [len(segment.input_ids) for segment in segments]
        if any(length > max_seq_length for length in lengths):
            skipped.append(
                {
                    "task_id": task,
                    "seed": row.get("seed"),
                    "reason": "trajectory_exceeds_max_seq_length",
                    "lengths": lengths,
                }
            )
            continue
        fingerprint = trajectory_fingerprint(segments)
        if fingerprint in fingerprints:
            skipped.append(
                {"task_id": task, "seed": row.get("seed"), "reason": "duplicate_trajectory"}
            )
            continue
        if max_examples_per_task is not None and task and per_task[task] >= max_examples_per_task:
            skipped.append({"task_id": task, "seed": row.get("seed"), "reason": "task_cap"})
            continue
        fingerprints.add(fingerprint)
        if task:
            per_task[task] += 1
        prepared.append((row, segments))

    return prepared, skipped


def _sft_loss(logprobs, *, global_tokens: int, world_size: int, padding_anchor):
    """Token-mean NLL with correct scaling for FSDP's rank-mean reduction."""
    if global_tokens < 1:
        raise ValueError("An SFT batch must contain at least one assistant token")
    return -logprobs.sum() * (float(world_size) / float(global_tokens)) + padding_anchor


def _sft_worker_entry(
    local_rank: int,
    world_size: int,
    master_port: int,
    config: ExperimentConfig,
    trajectory_paths: list[Path],
    output_dir: Path,
    model_path: str | None,
    resume_adapter: Path | None,
    telemetry_path: Path,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = "0"
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    import torch

    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        with TrainingMonitor(
            telemetry_path,
            interval_seconds=config.sft.monitor_interval_seconds,
            console_seconds=config.sft.monitor_console_seconds,
        ) as monitor:
            _train_sft_impl(
                config=config,
                trajectory_paths=trajectory_paths,
                output_dir=output_dir,
                model_path=model_path,
                resume_adapter=resume_adapter,
                monitor=monitor,
                telemetry_path=telemetry_path,
            )
    else:
        _train_sft_impl(
            config=config,
            trajectory_paths=trajectory_paths,
            output_dir=output_dir,
            model_path=model_path,
            resume_adapter=resume_adapter,
            monitor=_NullMonitor(),
            telemetry_path=telemetry_path,
        )


def train_sft(
    *,
    config: ExperimentConfig,
    trajectory_paths: list[Path],
    output_dir: Path,
    model_path: str | None = None,
    resume_adapter: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty adapter directory {output_dir}")
    telemetry_path = _next_telemetry_path(output_dir)
    print(f"[sft] telemetry: {telemetry_path}", flush=True)
    if config.distributed.strategy == "fsdp" and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        import torch
        import torch.multiprocessing as mp

        if not torch.cuda.is_available():
            raise RuntimeError("FSDP-QLoRA requires CUDA")
        visible_gpus = torch.cuda.device_count()
        if visible_gpus < config.distributed.num_processes:
            raise RuntimeError(
                f"FSDP requested {config.distributed.num_processes} processes but only "
                f"{visible_gpus} CUDA devices are visible"
            )
        print(
            f"[sft] launching FSDP-QLoRA on {config.distributed.num_processes} GPUs",
            flush=True,
        )
        mp.spawn(
            _sft_worker_entry,
            args=(
                config.distributed.num_processes,
                _free_port(),
                config,
                trajectory_paths,
                output_dir,
                model_path,
                resume_adapter,
                telemetry_path,
            ),
            nprocs=config.distributed.num_processes,
            join=True,
        )
        metrics_path = output_dir / "training_metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError("FSDP workers exited without writing training_metrics.json")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    with TrainingMonitor(
        telemetry_path,
        interval_seconds=config.sft.monitor_interval_seconds,
        console_seconds=config.sft.monitor_console_seconds,
    ) as monitor:
        return _train_sft_impl(
            config=config,
            trajectory_paths=trajectory_paths,
            output_dir=output_dir,
            model_path=model_path,
            resume_adapter=resume_adapter,
            monitor=monitor,
            telemetry_path=telemetry_path,
        )


def _train_sft_impl(
    *,
    config: ExperimentConfig,
    trajectory_paths: list[Path],
    output_dir: Path,
    model_path: str | None,
    resume_adapter: Path | None,
    monitor: TrainingMonitor,
    telemetry_path: Path,
) -> dict[str, Any]:
    import torch

    accelerator, fsdp_plugin = _create_accelerator(config)
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    if config.distributed.strategy == "fsdp":
        if world_size != config.distributed.num_processes:
            raise RuntimeError(
                f"FSDP world size is {world_size}, expected {config.distributed.num_processes}"
            )
    elif world_size != 1:
        raise RuntimeError("distributed.strategy=none must run in one process")

    random.seed(config.sft.seed)
    torch.manual_seed(config.sft.seed)
    monitor.set_phase("read_sft_trajectories")
    source_rows: list[dict[str, Any]] = []
    for path in trajectory_paths:
        source_rows.extend(read_jsonl(path))
    approved_count = sum(_approved_for_sft(row) for row in source_rows)
    if approved_count == 0:
        raise ValueError(
            "No SFT examples: provide resolved=true trajectories or explicitly curated "
            "rows with sft_approved=true"
        )

    monitor.set_phase("tokenize_sft_trajectories")
    tokenizer = _load_tokenizer(config, model_path)
    examples, skipped = prepare_sft_examples(
        tokenizer,
        source_rows,
        max_seq_length=config.sft.max_seq_length,
        max_examples_per_task=config.sft.max_examples_per_task,
        seed=config.sft.seed,
    )
    if not examples:
        raise ValueError("No approved SFT trajectory fits the configured max_seq_length")
    if accelerator.is_main_process:
        action_tokens = sum(
            len(segment.action_positions) for _, segments in examples for segment in segments
        )
        print(
            f"[sft] source={len(source_rows)} approved={approved_count} "
            f"unique_usable={len(examples)} assistant_tokens={action_tokens} "
            f"skipped={len(skipped)}",
            flush=True,
        )

    monitor.set_phase("load_and_dispatch_model")
    policy, tokenizer = _load_model_and_tokenizer(
        config,
        model_path,
        resume_adapter,
        tokenizer=tokenizer,
        monitor=monitor,
        fsdp=config.distributed.strategy == "fsdp",
    )
    model = _build_selective_logprob_model(
        policy,
        chunk_size=config.sft.logprob_chunk_size,
    )
    replicated_modules: list[Any] = []
    if fsdp_plugin is not None:
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(policy)
        replicated_modules = _prepare_replicated_projection_modules(
            policy,
            accelerator,
            fsdp_plugin,
        )

    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("The loaded policy has no trainable LoRA parameters")
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    monitor.set_phase("optimizer_init")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.sft.learning_rate)
    monitor.set_phase("fsdp_wrap")
    model, optimizer = accelerator.prepare(model, optimizer)
    if replicated_modules:
        _broadcast_replicated_modules(replicated_modules)
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"[sft] trainable parameters: {trainable_count:,}; "
            f"world_size={world_size}; global_batch={world_size}; device={accelerator.device}",
            flush=True,
        )

    dummy_token_id = tokenizer.eos_token_id
    if dummy_token_id is None:
        dummy_token_id = tokenizer.pad_token_id
    if dummy_token_id is None:
        dummy_token_id = 0

    optimizer_step = 0
    total_seen_examples = 0
    total_seen_tokens = 0
    metrics: list[dict[str, Any]] = []
    for epoch in range(1, config.sft.epochs + 1):
        epoch_examples = list(examples)
        random.Random(config.sft.seed + epoch).shuffle(epoch_examples)
        for start in range(0, len(epoch_examples), world_size):
            batch = epoch_examples[start : start + world_size]
            local = batch[rank] if rank < len(batch) else None
            row, segments = local if local is not None else ({}, [])
            local_tokens = sum(len(segment.action_positions) for segment in segments)
            token_count = torch.tensor(local_tokens, dtype=torch.int64, device=accelerator.device)
            token_count = accelerator.reduce(token_count, reduction="sum")
            global_tokens = int(token_count.item())

            optimizer.zero_grad(set_to_none=True)
            monitor.set_phase(
                f"sft_forward_backward:epoch_{epoch}:step_{optimizer_step + 1}"
            )
            logprobs, padding_anchor = _distributed_rollout_logprobs(
                model,
                segments,
                chunk_size=config.sft.logprob_chunk_size,
                input_device=accelerator.device,
                world_size=world_size,
                dummy_token_id=int(dummy_token_id),
            )
            loss = _sft_loss(
                logprobs,
                global_tokens=global_tokens,
                world_size=world_size,
                padding_anchor=padding_anchor,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite SFT loss at epoch={epoch}, task={row.get('task_id')}"
                )
            accelerator.backward(loss)
            monitor.set_phase(f"sft_optimizer:epoch_{epoch}:step_{optimizer_step + 1}")
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), config.sft.max_grad_norm
            )
            optimizer.step()
            optimizer_step += 1

            local_nll = float((-logprobs.detach().sum()).cpu())
            totals = torch.tensor(
                [local_nll, float(local_tokens), 1.0 if local is not None else 0.0],
                dtype=torch.float64,
                device=accelerator.device,
            )
            totals = accelerator.reduce(totals, reduction="sum")
            if accelerator.is_main_process:
                mean_nll = float(totals[0].item()) / max(float(totals[1].item()), 1.0)
                batch_examples = int(totals[2].item())
                batch_tokens = int(totals[1].item())
                total_seen_examples += batch_examples
                total_seen_tokens += batch_tokens
                metric = {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "examples": batch_examples,
                    "assistant_tokens": batch_tokens,
                    "mean_nll": mean_nll,
                    "perplexity": math.exp(min(mean_nll, 50.0)),
                    "grad_norm": float(
                        grad_norm.detach().cpu()
                        if hasattr(grad_norm, "detach")
                        else grad_norm
                    ),
                }
                metrics.append(metric)
                print(
                    f"[sft] step={optimizer_step} epoch={epoch} examples={batch_examples} "
                    f"tokens={batch_tokens} nll={mean_nll:.4f} "
                    f"ppl={metric['perplexity']:.2f}",
                    flush=True,
                )
            if config.sft.max_steps is not None and optimizer_step >= config.sft.max_steps:
                break
        if config.sft.max_steps is not None and optimizer_step >= config.sft.max_steps:
            break

    if optimizer_step == 0:
        raise ValueError("No SFT optimization step was executed")
    monitor.set_phase("gather_adapter")
    accelerator.wait_for_everyone()
    if fsdp_plugin is not None:
        fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    gathered_state = accelerator.get_state_dict(model)
    summary = {
        "mode": "sft",
        "source_rows": len(source_rows),
        "approved_rows": approved_count,
        "examples": len(examples),
        "skipped": skipped,
        "optimizer_steps": optimizer_step,
        "examples_seen": total_seen_examples,
        "assistant_tokens_seen": total_seen_tokens,
        "trainable_parameters": trainable_count,
        "distributed_strategy": config.distributed.strategy,
        "world_size": world_size,
        "global_batch_size": world_size,
        "logprob_chunk_size": config.sft.logprob_chunk_size,
        "telemetry_path": str(telemetry_path),
        "token_sources": dict(
            Counter(str(row.get("token_source", "unknown")) for row, _ in examples)
        ),
        "metrics": metrics,
    }
    if accelerator.is_main_process:
        monitor.set_phase("save_adapter")
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        output_policy = _wrapped_policy(unwrapped)
        output_policy.save_pretrained(
            output_dir,
            state_dict=_policy_state_dict(gathered_state),
            safe_serialization=True,
        )
        (output_dir / "training_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return summary
