import json
import subprocess
import threading
from dataclasses import replace

import pytest
import yaml

from gemma_grpo.collect import _prepare_submission, collect_rollouts
from gemma_grpo.config import load_config


def test_prepared_submission_updates_seed_budget_and_adapter(tmp_path):
    source = tmp_path / "source"
    (source / "configs").mkdir(parents=True)
    (source / "configs" / "sampling.yaml").write_text("seed: 1\n", encoding="utf-8")
    (source / "eval_config.yaml").write_text(
        "evaluation:\n  max_tool_calls: 99\n  max_time_minutes: 99\n", encoding="utf-8"
    )
    (source / "agent.yaml").write_text("name: a\nmodel: base\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake")
    (adapter / "training_metrics.json").write_text("{}", encoding="utf-8")

    destination = tmp_path / "prepared"
    _prepare_submission(
        source,
        destination,
        seed=202,
        adapter=adapter,
        max_tool_calls=5,
        max_time_minutes=3,
    )

    sampling = yaml.safe_load((destination / "configs" / "sampling.yaml").read_text())
    evaluation = yaml.safe_load((destination / "eval_config.yaml").read_text())["evaluation"]
    assert sampling["seed"] == 202
    assert evaluation["max_tool_calls"] == 5
    assert evaluation["max_time_minutes"] == 3
    assert "adapter: main_lora" in (destination / "agent.yaml").read_text()
    adapter_files = {path.name for path in (destination / "adapters" / "main_lora").iterdir()}
    assert adapter_files == {"adapter_config.json", "adapter_model.safetensors"}


def test_prepared_submission_retargets_offline_teacher_model(tmp_path):
    source = tmp_path / "source"
    (source / "configs").mkdir(parents=True)
    (source / "configs" / "sampling.yaml").write_text("seed: 1\n", encoding="utf-8")
    (source / "eval_config.yaml").write_text("evaluation: {}\n", encoding="utf-8")
    (source / "agent.yaml").write_text(
        "name: teacher\nmodel: gemma\ntools: []\n", encoding="utf-8"
    )

    destination = tmp_path / "prepared"
    _prepare_submission(
        source,
        destination,
        seed=101,
        adapter=None,
        max_tool_calls=20,
        max_time_minutes=15,
        model_alias="qwen3-coder-next-teacher",
    )

    assert "model: qwen3-coder-next-teacher" in (
        destination / "agent.yaml"
    ).read_text(encoding="utf-8")


def test_collect_rollouts_runs_group_seeds_concurrently_and_preserves_order(
    tmp_path, monkeypatch
):
    source = tmp_path / "submission"
    (source / "configs").mkdir(parents=True)
    (source / "configs" / "sampling.yaml").write_text("seed: 1\n", encoding="utf-8")
    (source / "eval_config.yaml").write_text("evaluation: {}\n", encoding="utf-8")
    (source / "agent.yaml").write_text("name: a\nmodel: base\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"instance_id": "task_1", "problem_statement": "fix it"}) + "\n",
        encoding="utf-8",
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    barrier = threading.Barrier(4)

    def fake_run_eval(invocation, *, check=False):
        sampling = yaml.safe_load(
            (invocation.submission_dir / "configs" / "sampling.yaml").read_text()
        )
        barrier.wait(timeout=5)
        invocation.results_dir.mkdir(parents=True, exist_ok=True)
        (invocation.results_dir / "task_results.jsonl").write_text(
            json.dumps({"instance_id": "task_1", "resolved": False}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, stdout=f"seed={sampling['seed']}")

    monkeypatch.setattr("gemma_grpo.collect.run_eval", fake_run_eval)
    config = load_config("configs/smoke.yaml")
    config = replace(config, rollout=replace(config.rollout, managed_vllm=False))

    rows = collect_rollouts(
        config=config,
        tasks_path=tasks,
        snapshots_dir=snapshots,
        submission_dir=source,
        results_root=tmp_path / "results",
        output_path=tmp_path / "rollouts.jsonl",
        iteration=0,
    )

    assert [row["seed"] for row in rows] == list(config.rollout.seeds)
    assert all(row["swegemma_returncode"] == 0 for row in rows)


def test_collect_rollouts_checkpoints_completed_task_before_later_failure(
    tmp_path, monkeypatch
):
    source = tmp_path / "submission"
    (source / "configs").mkdir(parents=True)
    (source / "configs" / "sampling.yaml").write_text("seed: 1\n", encoding="utf-8")
    (source / "eval_config.yaml").write_text("evaluation: {}\n", encoding="utf-8")
    (source / "agent.yaml").write_text("name: a\nmodel: base\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            json.dumps({"instance_id": name, "problem_statement": "fix it"})
            for name in ("task_1", "task_2")
        )
        + "\n",
        encoding="utf-8",
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    def fake_run_eval(invocation, *, check=False):
        if invocation.task_id == "task_2":
            raise RuntimeError("later sandbox failed")
        invocation.results_dir.mkdir(parents=True, exist_ok=True)
        (invocation.results_dir / "task_results.jsonl").write_text(
            json.dumps({"instance_id": invocation.task_id, "resolved": False}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, stdout="ok")

    monkeypatch.setattr("gemma_grpo.collect.run_eval", fake_run_eval)
    config = load_config("configs/smoke.yaml")
    config = replace(
        config,
        rollout=replace(config.rollout, managed_vllm=False, concurrency=1),
    )
    output = tmp_path / "rollouts.jsonl"

    with pytest.raises(RuntimeError, match="later sandbox failed"):
        collect_rollouts(
            config=config,
            tasks_path=tasks,
            snapshots_dir=snapshots,
            submission_dir=source,
            results_root=tmp_path / "results",
            output_path=output,
            iteration=0,
        )

    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 4
    assert {row["task_id"] for row in saved} == {"task_1"}


def test_collect_rejects_tool_budget_exhausted_trajectory(tmp_path, monkeypatch):
    source = tmp_path / "submission"
    (source / "configs").mkdir(parents=True)
    (source / "configs" / "sampling.yaml").write_text("seed: 1\n", encoding="utf-8")
    (source / "eval_config.yaml").write_text("evaluation: {}\n", encoding="utf-8")
    (source / "agent.yaml").write_text("name: a\nmodel: base\n", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"instance_id": "task_1", "problem_statement": "fix it"}) + "\n",
        encoding="utf-8",
    )
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    def fake_run_eval(invocation, *, check=False):
        invocation.results_dir.mkdir(parents=True, exist_ok=True)
        (invocation.results_dir / "task_results.jsonl").write_text(
            json.dumps({"instance_id": "task_1", "resolved": True}) + "\n",
            encoding="utf-8",
        )
        traces = invocation.results_dir / "traces"
        traces.mkdir()
        (traces / "trace_task_1.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {"source": "agent", "content": "trying another tool"},
                        {
                            "source": "tool",
                            "content": {
                                "status": "error",
                                "error_type": "BudgetExceeded",
                                "error_message": "Tool call budget exhausted (50 calls)",
                            },
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess([], 0, stdout="ok")

    monkeypatch.setattr("gemma_grpo.collect.run_eval", fake_run_eval)
    config = load_config("configs/smoke.yaml")
    config = replace(
        config,
        rollout=replace(config.rollout, managed_vllm=False, concurrency=1),
    )

    rows = collect_rollouts(
        config=config,
        tasks_path=tasks,
        snapshots_dir=snapshots,
        submission_dir=source,
        results_root=tmp_path / "results",
        output_path=tmp_path / "rollouts.jsonl",
        iteration=0,
    )

    assert len(rows) == 4
    assert all(row["resolved"] is True for row in rows)
    assert all(row["tool_budget_exhausted"] is True for row in rows)
    assert all(row["scorable"] is False for row in rows)
    assert all(row["reward"] is None for row in rows)
    assert all(row["reward_reason"] == "tool_budget_exhausted" for row in rows)
    assert all(row["trace_trainable"] is False for row in rows)

