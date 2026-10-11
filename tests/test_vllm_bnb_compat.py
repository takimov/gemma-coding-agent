from types import SimpleNamespace

import pytest

from gemma_grpo.vllm_bnb_compat import (
    _complete_gemma4_tied_kv_quant_states,
    _patch_config_class,
    _patch_intermediate_tensors_class,
    _patch_loader_class,
)


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


def test_loader_patch_completes_tied_kv_state_after_iteration():
    torch = pytest.importorskip("torch")
    weight = torch.tensor([1.0], dtype=torch.bfloat16)
    k_key = "model.layers.7.self_attn.k_proj.weight"
    v_key = "model.layers.7.self_attn.v_proj.weight"
    k_state = object()

    class Loader:
        is_pool_model = False
        target_modules = []

        @staticmethod
        def weight_mapper(name):
            return name

        def _quantized_4bit_generator(self, files, use_safetensors, states):
            states[k_key] = k_state
            yield k_key, weight

    states = {}
    _patch_loader_class(Loader)
    list(Loader()._quantized_4bit_generator([], True, states))

    assert states[v_key] is k_state


def test_tied_kv_state_does_not_replace_real_v():
    k_state = object()
    v_state = object()
    states = {
        "model.layers.1.self_attn.k_proj.weight": k_state,
        "model.layers.1.self_attn.v_proj.weight": v_state,
    }

    assert _complete_gemma4_tied_kv_quant_states(states) == 0
    assert states["model.layers.1.self_attn.v_proj.weight"] is v_state


def test_intermediate_tensors_patch_adds_mapping_get_and_drops_none():
    class Intermediate:
        def __init__(self, tensors):
            self.tensors = tensors

    _patch_intermediate_tensors_class(Intermediate)
    value = object()
    intermediate = Intermediate({"hidden_states": value, "per_layer_inputs": None})

    assert intermediate.get("hidden_states") is value
    assert "per_layer_inputs" not in intermediate.tensors
    assert intermediate.get("missing") is None
    assert intermediate.get("missing", "fallback") == "fallback"


def test_intermediate_tensors_patch_is_idempotent():
    class Intermediate:
        def __init__(self, tensors):
            self.tensors = tensors

    _patch_intermediate_tensors_class(Intermediate)
    patched_init = Intermediate.__init__
    _patch_intermediate_tensors_class(Intermediate)

    assert Intermediate.__init__ is patched_init
    assert Intermediate({"optional": None}).tensors == {}
