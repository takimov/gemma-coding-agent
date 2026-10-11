import json

import yaml

from gemma_grpo.collect import _prepare_submission


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

