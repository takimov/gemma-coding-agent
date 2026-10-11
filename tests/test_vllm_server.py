import yaml

from gemma_grpo.vllm_server import ADAPTER_ALIAS, BASE_MODEL_ALIAS, write_models_yaml


def test_models_yaml_routes_base_and_adapter_to_managed_server(tmp_path):
    path = write_models_yaml(tmp_path / "models.yaml", port=8123, adapter=tmp_path / "adapter")
    models = yaml.safe_load(path.read_text(encoding="utf-8"))["models"]

    assert models[BASE_MODEL_ALIAS]["api_base"] == "http://127.0.0.1:8123/v1"
    assert models[BASE_MODEL_ALIAS]["path"] == f"openai/{BASE_MODEL_ALIAS}"
    assert models[ADAPTER_ALIAS]["path"] == f"openai/{ADAPTER_ALIAS}"
