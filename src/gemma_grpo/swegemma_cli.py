from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SwegemmaInvocation:
    tasks_path: Path
    snapshots_dir: Path
    submission_dir: Path
    results_dir: Path
    task_id: str
    models_yaml: Path | None = None
    sandbox: str = "docker"
    image: str | None = None
    max_tool_calls: int = 20
    max_time_minutes: int = 15
    timeout_seconds: int | None = 300
    max_turns: int | None = 80
    concurrency: int = 1
    display: str = "quiet"
    evaluator_backend: str = "cli"
    graph_dir: Path | None = None
    embeddings_dir: Path | None = None
    wheels_dir: Path | None = None
    context_cache_min_tokens: int = 2048
    context_cache_ttl_seconds: int = 1800
    context_cache_intervals: int = 10
    compaction_interval: int = 15
    compaction_overlap_size: int = 2
    compaction_token_threshold: int = 14336
    compaction_event_retention_size: int = 5


def build_eval_command(invocation: SwegemmaInvocation) -> list[str]:
    command = [
        "swegemma",
        "eval",
        "--tasks",
        str(invocation.tasks_path),
        "--snapshots-dir",
        str(invocation.snapshots_dir),
        "--submission-dir",
        str(invocation.submission_dir),
        "--results-dir",
        str(invocation.results_dir),
        "--task-id",
        invocation.task_id,
        "--sandbox",
        invocation.sandbox,
        "--max-tool-calls",
        str(invocation.max_tool_calls),
        "--max-time-minutes",
        str(invocation.max_time_minutes),
        "--concurrency",
        str(invocation.concurrency),
        "--display",
        invocation.display,
    ]
    if invocation.timeout_seconds is not None:
        command.extend(["--timeout-seconds", str(invocation.timeout_seconds)])
    if invocation.image:
        command.extend(["--image", invocation.image])
    if invocation.models_yaml:
        command.extend(["--models-yaml", str(invocation.models_yaml)])
    return command


def build_direct_eval_command(invocation: SwegemmaInvocation) -> list[str]:
    if invocation.models_yaml is None:
        raise ValueError("Direct SWEGemma evaluation requires models_yaml")
    command = [
        sys.executable,
        "-m",
        "gemma_grpo.evaluator_worker",
        "--tasks",
        str(invocation.tasks_path),
        "--snapshots-dir",
        str(invocation.snapshots_dir),
        "--submission-dir",
        str(invocation.submission_dir),
        "--results-dir",
        str(invocation.results_dir),
        "--task-id",
        invocation.task_id,
        "--models-yaml",
        str(invocation.models_yaml),
        "--sandbox",
        invocation.sandbox,
        "--max-tool-calls",
        str(invocation.max_tool_calls),
        "--max-time-minutes",
        str(invocation.max_time_minutes),
        "--context-cache-min-tokens",
        str(invocation.context_cache_min_tokens),
        "--context-cache-ttl-seconds",
        str(invocation.context_cache_ttl_seconds),
        "--context-cache-intervals",
        str(invocation.context_cache_intervals),
        "--compaction-interval",
        str(invocation.compaction_interval),
        "--compaction-overlap-size",
        str(invocation.compaction_overlap_size),
        "--compaction-token-threshold",
        str(invocation.compaction_token_threshold),
        "--compaction-event-retention-size",
        str(invocation.compaction_event_retention_size),
    ]
    if invocation.timeout_seconds is not None:
        command.extend(["--timeout-seconds", str(invocation.timeout_seconds)])
    if invocation.max_turns is not None:
        command.extend(["--max-turns", str(invocation.max_turns)])
    if invocation.image:
        command.extend(["--image", invocation.image])
    for flag, path in (
        ("--graph-dir", invocation.graph_dir),
        ("--embeddings-dir", invocation.embeddings_dir),
        ("--wheels-dir", invocation.wheels_dir),
    ):
        if path is not None:
            command.extend([flag, str(path)])
    return command


def run_eval(invocation: SwegemmaInvocation, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    invocation.results_dir.mkdir(parents=True, exist_ok=True)
    if invocation.evaluator_backend == "direct":
        command = build_direct_eval_command(invocation)
    elif invocation.evaluator_backend == "cli":
        command = build_eval_command(invocation)
    else:
        raise ValueError(f"Unknown evaluator backend: {invocation.evaluator_backend}")
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

