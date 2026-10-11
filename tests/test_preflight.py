from pathlib import Path

import yaml

from gemma_grpo.config import (
    DistributedConfig,
    ExperimentConfig,
    GrpoConfig,
    LoraConfigData,
    ModelConfig,
    RewardConfig,
    RolloutConfig,
)
from gemma_grpo.preflight import _valid_agent_name, run_preflight


def test_agent_name_matches_google_adk_identifier_rules():
    assert _valid_agent_name("gemma_grpo_agent")
    assert _valid_agent_name("_agent2")
    assert not _valid_agent_name("gemma-grpo-agent")
    assert not _valid_agent_name("2agent")
    assert not _valid_agent_name("agent name")


def _write_submission(path: Path, *, max_output_tokens: int) -> None:
    (path / "configs").mkdir(parents=True)
    (path / "prompts").mkdir()
    (path / "agent.yaml").write_text(
        "name: gemma_grpo_agent\ntools: []\n",
        encoding="utf-8",
    )
    (path / "prompts" / "system.md").write_text("test\n", encoding="utf-8")
    (path / "configs" / "sampling.yaml").write_text(
        yaml.safe_dump({"max_output_tokens": max_output_tokens}),
        encoding="utf-8",
    )
    (path / "eval_config.yaml").write_text(
        "evaluation: {}\n",
        encoding="utf-8",
    )


def test_preflight_rejects_output_budget_without_long_tool_history_headroom(
    tmp_path: Path, monkeypatch
):
    submission = tmp_path / "submission"
    _write_submission(submission, max_output_tokens=16384)
    monkeypatch.setattr("gemma_grpo.preflight.shutil.which", lambda _: None)
    monkeypatch.setattr("gemma_grpo.preflight.importlib.metadata.version", lambda _: "99.0")

    report = run_preflight(
        config=ExperimentConfig(
            model=ModelConfig(name_or_path="unused", load_mode="native"),
            lora=LoraConfigData(),
            grpo=GrpoConfig(),
            rollout=RolloutConfig(managed_vllm=False),
            reward=RewardConfig(),
            distributed=DistributedConfig(strategy="none", num_processes=1),
        ),
        tasks_path=None,
        snapshots_dir=None,
        submission_dir=submission,
        model_path=None,
        rollout_model_path=None,
        check_tokenizer=False,
    )

    check = next(
        item for item in report["checks"] if item["name"] == "rollout output-token headroom"
    )
    assert check["ok"] is False
    assert "required <= 8192" in check["detail"]
