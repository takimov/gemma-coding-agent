import json
import sys
from types import SimpleNamespace

import pytest

from gemma_grpo.nf4 import (
    _replace_source_quantizer_for_save,
    _save_nf4_model,
    assert_nf4_checkpoint,
    assert_vllm_nf4_checkpoint,
    checkpoint_quantization,
    prepare_vllm_nf4_checkpoint,
)


def _write_config(path, quantization):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"quantization_config": quantization}),
        encoding="utf-8",
    )


def test_saved_nf4_checkpoint_is_accepted_for_fsdp(tmp_path):
    model = tmp_path / "model"
    _write_config(
        model,
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": "bfloat16",
        },
    )

    assert_nf4_checkpoint(model)


def test_uint8_nf4_checkpoint_is_rejected_for_fsdp(tmp_path):
    model = tmp_path / "model"
    _write_config(
        model,
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": "uint8",
        },
    )

    with pytest.raises(ValueError, match="bfloat16"):
        assert_nf4_checkpoint(model)


def test_uint8_nf4_checkpoint_is_accepted_for_vllm(tmp_path):
    model = tmp_path / "model"
    _write_config(
        model,
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": "uint8",
        },
    )

    assert_vllm_nf4_checkpoint(model)


def test_bfloat16_nf4_checkpoint_is_rejected_for_vllm(tmp_path):
    model = tmp_path / "model"
    _write_config(
        model,
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": "bfloat16",
        },
    )

    with pytest.raises(ValueError, match="byte-packed"):
        assert_vllm_nf4_checkpoint(model)


def test_prepare_vllm_nf4_bitcasts_quantized_weights_without_requantizing(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_config(
        source,
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": "bfloat16",
        },
    )
    quantized = torch.tensor([1.0, -2.0], dtype=torch.bfloat16).reshape(-1, 1)
    embedding = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    safetensors_torch.save_file(
        {
            "layer.weight": quantized,
            "layer.weight.quant_state.bitsandbytes__nf4": torch.tensor([1], dtype=torch.uint8),
            "embed.weight": embedding,
        },
        source / "model.safetensors",
    )

    prepare_vllm_nf4_checkpoint(source_model=source, output_dir=target)

    saved = safetensors_torch.load_file(target / "model.safetensors")
    assert saved["layer.weight"].dtype == torch.uint8
    assert tuple(saved["layer.weight"].shape) == (4, 1)
    assert torch.equal(saved["layer.weight"].flatten(), quantized.view(torch.uint8).flatten())
    assert saved["embed.weight"].dtype == torch.bfloat16
    assert checkpoint_quantization(target)["bnb_4bit_quant_storage"] == "uint8"


def test_nf4_save_replaces_compressed_tensor_conversion_state(monkeypatch):
    created = object()

    class _AutoHfQuantizer:
        @classmethod
        def from_config(cls, config, *, pre_quantized):
            assert config is bnb_config
            assert pre_quantized is True
            return created

    monkeypatch.setitem(
        sys.modules,
        "transformers.quantizers",
        SimpleNamespace(AutoHfQuantizer=_AutoHfQuantizer),
    )
    source_quantizer = object()
    model = SimpleNamespace(
        _weight_conversions=["compressed-tensors conversion"],
        hf_quantizer=source_quantizer,
        quantization_method="compressed-tensors",
        config=SimpleNamespace(quantization_config={"quant_method": "compressed-tensors"}),
    )
    bnb_config = SimpleNamespace(quant_method="bitsandbytes")

    _replace_source_quantizer_for_save(model, bnb_config)

    assert model._weight_conversions is None
    assert model.hf_quantizer is created
    assert model.quantization_method == "bitsandbytes"
    assert model.config.quantization_config is bnb_config


def test_nf4_save_retries_non_reversible_transformers_mapping(tmp_path):
    class _Model:
        def __init__(self):
            self.calls = []

        def save_pretrained(self, directory, **kwargs):
            self.calls.append(kwargs)
            directory.mkdir(exist_ok=True)
            (directory / "partial").write_text("partial", encoding="utf-8")
            if len(self.calls) == 1:
                raise AttributeError("'NoneType' object has no attribute 'convert'")

    model = _Model()
    directory = tmp_path / "checkpoint"

    key_format = _save_nf4_model(model, directory, max_shard_size="4GB")

    assert key_format == "canonical"
    assert len(model.calls) == 2
    assert "save_original_format" not in model.calls[0]
    assert model.calls[1]["save_original_format"] is False
