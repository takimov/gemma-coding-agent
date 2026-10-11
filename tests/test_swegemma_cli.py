from pathlib import Path

from gemma_grpo.swegemma_cli import (
    SwegemmaInvocation,
    build_direct_eval_command,
    build_eval_command,
)


def test_command_is_argument_list_and_contains_limits():
    command = build_eval_command(
        SwegemmaInvocation(
            tasks_path=Path("tasks.jsonl"),
            snapshots_dir=Path("snapshots"),
            submission_dir=Path("submission"),
            results_dir=Path("results"),
            task_id="repo__issue-1",
            image="sandbox:latest",
            models_yaml=Path("models.yaml"),
        )
    )
    assert command[:2] == ["swegemma", "eval"]
    assert command[command.index("--task-id") + 1] == "repo__issue-1"
    assert command[command.index("--max-tool-calls") + 1] == "20"
    assert command[command.index("--display") + 1] == "quiet"
    assert command[command.index("--image") + 1] == "sandbox:latest"
    assert command[command.index("--models-yaml") + 1] == "models.yaml"


def test_direct_command_enables_official_context_controls():
    command = build_direct_eval_command(
        SwegemmaInvocation(
            tasks_path=Path("tasks.jsonl"),
            snapshots_dir=Path("snapshots"),
            submission_dir=Path("submission"),
            results_dir=Path("results"),
            task_id="repo__issue-1",
            models_yaml=Path("models.yaml"),
            max_tool_calls=28,
            max_time_minutes=8,
            timeout_seconds=240,
            max_turns=80,
            graph_dir=Path("graphs"),
            embeddings_dir=Path("embeddings"),
            wheels_dir=Path("wheels"),
            evaluator_backend="direct",
        )
    )

    assert command[1:3] == ["-m", "gemma_grpo.evaluator_worker"]
    assert command[command.index("--max-tool-calls") + 1] == "28"
    assert command[command.index("--max-time-minutes") + 1] == "8"
    assert command[command.index("--max-turns") + 1] == "80"
    assert command[command.index("--graph-dir") + 1] == "graphs"
    assert command[command.index("--compaction-token-threshold") + 1] == "14336"

