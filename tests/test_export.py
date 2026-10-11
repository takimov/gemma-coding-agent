import json
import zipfile

from gemma_grpo.export import export_submission


def test_export_contains_root_agent_and_adapter(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "agent.yaml").write_text("name: test\nmodel: base\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}), encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fake")
    (adapter / "tokenizer.model").write_bytes(b"must not be exported")

    archive = export_submission(
        submission_dir=submission,
        adapter_dir=adapter,
        output_zip=tmp_path / "dist" / "submission.zip",
    )
    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        assert "agent.yaml" in names
        assert "adapters/main_lora/adapter_config.json" in names
        assert "adapters/main_lora/adapter_model.safetensors" in names
        assert "adapters/main_lora/tokenizer.model" not in names
        agent = handle.read("agent.yaml").decode()
    assert "adapter: main_lora" in agent

