from __future__ import annotations

import subprocess
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
    concurrency: int = 1
    display: str = "quiet"


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


def run_eval(invocation: SwegemmaInvocation, *, check: bool = False) -> subprocess.CompletedProcess[str]:
    invocation.results_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        build_eval_command(invocation),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

