from types import SimpleNamespace

import pytest

from gemma_grpo.vllm_bnb_compat import _patch_config_class, _patch_loader_class


def test_config_patch_exposes_bf16_storage_as_uint8():
    class Config:
        @classmethod
        def from_config(cls, config):
            return SimpleNamespace(
                bnb_4bit_quant_storage=config["bnb_4bit_quant_storage"]
            )

    _patch_config_class(Config)
    result = Config.from_config({"bnb_4bit_quant_storage": "bfloat16"})

    assert result.bnb_4bit_quant_storage == "uint8"


def test_loader_patch_bitcasts_only_weights_with_quant_state():
    torch = pytest.importorskip("torch")
    quantized = torch.tensor([1.0, -2.0], dtype=torch.bfloat16).reshape(-1, 1)
    embedding = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)

    class Loader:
        is_pool_model = False
        target_modules = []

        @staticmethod
        def weight_mapper(name):
            return name

        def _quantized_4bit_generator(self, files, use_safetensors, states):
            states["layer.weight"] = object()
            yield "layer.weight", quantized
            yield "embed.weight", embedding

    _patch_loader_class(Loader)
    rows = list(Loader()._quantized_4bit_generator([], True, {}))

    packed = rows[0][1]
    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (4, 1)
    assert torch.equal(packed.flatten(), quantized.view(torch.uint8).flatten())
    assert rows[1][1] is embedding
