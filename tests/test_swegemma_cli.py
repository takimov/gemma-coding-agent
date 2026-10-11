from pathlib import Path

from gemma_grpo.swegemma_cli import SwegemmaInvocation, build_eval_command


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

