import json
import sys
from types import SimpleNamespace

import pytest

from gemma_grpo.nf4 import (
    assert_compressed_w4_checkpoint,
    _replace_source_quantizer_for_save,
    _save_nf4_model,
    assert_nf4_checkpoint,
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


def test_compressed_tensors_checkpoint_is_accepted_for_native_rollout(tmp_path):
    model = tmp_path / "model"
    _write_config(model, {"quant_method": "compressed-tensors"})

    assert_compressed_w4_checkpoint(model)


def test_nf4_checkpoint_is_rejected_for_native_rollout(tmp_path):
    model = tmp_path / "model"
    _write_config(
        model,
        {
            "quant_method": "bitsandbytes",
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        },
    )

    with pytest.raises(ValueError, match="compressed-tensors"):
        assert_compressed_w4_checkpoint(model)


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
