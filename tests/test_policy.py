from gemma_grpo.policy import adapter_fingerprint


def test_adapter_fingerprint_changes_with_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"one")
    first = adapter_fingerprint(adapter)
    weights.write_bytes(b"two")
    second = adapter_fingerprint(adapter)
    assert first.startswith("lora-sha256:")
    assert first != second

