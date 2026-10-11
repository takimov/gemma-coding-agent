import yaml

from gemma_grpo.config import load_config
from gemma_grpo.vllm_server import (
    ADAPTER_ALIAS,
    BASE_MODEL_ALIAS,
    ManagedVllmServer,
    VLLM_COMPAT_SITE,
    vllm_environment,
    write_models_yaml,
)


def test_models_yaml_routes_base_and_adapter_to_managed_server(tmp_path):
    path = write_models_yaml(tmp_path / "models.yaml", port=8123, adapter=tmp_path / "adapter")
    models = yaml.safe_load(path.read_text(encoding="utf-8"))["models"]

    assert models[BASE_MODEL_ALIAS]["api_base"] == "http://127.0.0.1:8123/v1"
    assert models[BASE_MODEL_ALIAS]["path"] == f"openai/{BASE_MODEL_ALIAS}"
    assert models[ADAPTER_ALIAS]["path"] == f"openai/{ADAPTER_ALIAS}"


def test_managed_vllm_uses_pipeline_parallel_for_prequantized_nf4(tmp_path, monkeypatch):
    config = load_config("configs/smoke.yaml")
    server = ManagedVllmServer(
        config=config,
        model_path=tmp_path / "model",
        adapter=None,
        work_dir=tmp_path / "run",
    )
    monkeypatch.setattr(
        server,
        "_supported_flags",
        lambda: (
            "--quantization --load-format --pipeline-parallel-size "
            "--language-model-only"
        ),
    )

    command = server._command()

    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    assert command[command.index("--pipeline-parallel-size") + 1] == "4"
    assert "--language-model-only" in command


def test_vllm_environment_installs_compat_for_spawned_workers(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")

    environment = vllm_environment()

    assert environment["VLLM_PLUGINS"] == ""
    assert environment["GEMMA_GRPO_VLLM_BNB_COMPAT"] == "1"
    assert environment["VLLM_NO_USAGE_STATS"] == "1"
    assert environment["DO_NOT_TRACK"] == "1"
    assert environment["PYTHONPATH"].split(":") == [
        str(VLLM_COMPAT_SITE),
        "/existing",
    ]
