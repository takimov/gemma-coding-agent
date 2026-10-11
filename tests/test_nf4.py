import json

import pytest

from gemma_grpo.nf4 import assert_nf4_checkpoint


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
