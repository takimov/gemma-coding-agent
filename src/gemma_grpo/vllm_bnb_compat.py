from __future__ import annotations

from collections.abc import Generator
from typing import Any


_BF16_STORAGE_NAMES = {"bfloat16", "torch.bfloat16"}


def _patch_config_class(config_class: type) -> None:
    """Make vLLM allocate uint8 parameters for BF16-packed BNB checkpoints."""
    if getattr(config_class, "_gemma_grpo_bf16_storage_compat", False):
        return

    original = config_class.from_config.__func__

    def from_config(cls: type, config: dict[str, Any]) -> Any:
        compatible = dict(config)
        storage = str(compatible.get("bnb_4bit_quant_storage") or "").lower()
        if storage in _BF16_STORAGE_NAMES:
            # The serialized BF16 tensor is only a byte container. vLLM's BNB
            # kernels consume the same bytes through a uint8 Parameter.
            compatible["bnb_4bit_quant_storage"] = "uint8"
        return original(cls, compatible)

    config_class.from_config = classmethod(from_config)
    config_class._gemma_grpo_bf16_storage_compat = True


def _mapped_weight_name(loader: Any, original_name: str) -> str:
    mapped = loader.weight_mapper(original_name)
    target_modules = getattr(loader, "target_modules", ())
    if (
        getattr(loader, "is_pool_model", False)
        and target_modules
        and target_modules[0].startswith("model.")
        and not mapped.startswith("model.")
    ):
        mapped = "model." + mapped
    return mapped


def _complete_gemma4_tied_kv_quant_states(
    quant_state_dict: dict[str, Any],
) -> int:
    """Mirror K quantization metadata for Gemma 4 layers where K is V.

    Gemma 4 checkpoints omit ``v_proj`` for full-attention layers with
    ``attention_k_eq_v``.  vLLM 0.19.1 duplicates the packed K weight while
    loading the model, but its generic BitsAndBytes loader only sees the
    checkpoint names and therefore misses the V ``QuantState``.  The missing
    third shard makes the BNB linear produce Q+K instead of Q+K+V.

    Sliding-attention layers have a real V state and are left untouched.
    """
    additions: dict[str, Any] = {}
    marker = ".self_attn.k_proj.weight"
    for key, state in quant_state_dict.items():
        if marker not in key:
            continue
        value_key = key.replace(marker, ".self_attn.v_proj.weight")
        if value_key not in quant_state_dict:
            additions[value_key] = state
    quant_state_dict.update(additions)
    return len(additions)


def _patch_loader_class(loader_class: type) -> None:
    """Bitcast only pre-quantized weights while streaming them from disk."""
    if getattr(loader_class, "_gemma_grpo_bf16_storage_compat", False):
        return

    original = loader_class._quantized_4bit_generator

    def quantized_4bit_generator(
        self: Any,
        hf_weights_files: Any,
        use_safetensors: bool,
        quant_state_dict: dict[str, Any],
    ) -> Generator[tuple[str, Any], None, None]:
        import torch

        for original_name, tensor in original(
            self,
            hf_weights_files,
            use_safetensors,
            quant_state_dict,
        ):
            mapped_name = _mapped_weight_name(self, original_name)
            if mapped_name in quant_state_dict and tensor.dtype == torch.bfloat16:
                # No values are decoded or quantized here. A contiguous BF16
                # container becomes a (packed_bytes, 1) uint8 view expected by
                # vLLM 0.19's built-in BitsAndBytes weight loader.
                tensor = tensor.contiguous().view(torch.uint8).reshape(-1, 1)
            yield original_name, tensor

        added = _complete_gemma4_tied_kv_quant_states(quant_state_dict)
        if added:
            print(
                "[gemma-grpo] mirrored "
                f"{added} Gemma 4 K quantization state(s) for tied V",
                flush=True,
            )

    loader_class._quantized_4bit_generator = quantized_4bit_generator
    loader_class._gemma_grpo_bf16_storage_compat = True


def _patch_intermediate_tensors_class(intermediate_class: type) -> None:
    """Backport the mapping ``get`` operation used by Gemma4 PP in vLLM."""
    if hasattr(intermediate_class, "get"):
        return

    def get(self: Any, key: str, default: Any = None) -> Any:
        return self.tensors.get(key, default)

    intermediate_class.get = get


def install_vllm_bnb_compat() -> None:
    """Install the vLLM 0.19 BF16-storage compatibility shim once per process."""
    from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
    from vllm.model_executor.model_loader.bitsandbytes_loader import (
        BitsAndBytesModelLoader,
    )
    from vllm.sequence import IntermediateTensors

    _patch_config_class(BitsAndBytesConfig)
    _patch_loader_class(BitsAndBytesModelLoader)
    _patch_intermediate_tensors_class(IntermediateTensors)
    print(
        "[gemma-grpo] vLLM BF16-storage NF4 streaming compatibility enabled",
        flush=True,
    )
