from __future__ import annotations

import inspect
import json
import os
import shutil
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any


NF4_MANIFEST = "nf4_manifest.json"


def checkpoint_quantization(path: str | Path) -> dict[str, Any]:
    """Return a checkpoint's serialized quantization configuration."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Model checkpoint has no config.json: {path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = raw.get("quantization_config") or {}
    if not isinstance(quantization, dict):
        raise ValueError(f"Invalid quantization_config in {config_path}")
    return quantization


def assert_nf4_checkpoint(path: str | Path, *, require_fsdp_storage: bool = True) -> None:
    """Reject checkpoints that would silently trigger another quantization pass."""
    quantization = checkpoint_quantization(path)
    method = str(quantization.get("quant_method") or "").lower()
    quant_type = str(quantization.get("bnb_4bit_quant_type") or "").lower()
    if method not in {"bitsandbytes", "bitsandbytes_4bit"}:
        raise ValueError(
            f"Expected a saved bitsandbytes checkpoint at {path}, got quant_method={method!r}. "
            "Run `gemma-grpo convert-nf4 --source-model ... --output-dir ...` once first."
        )
    if not bool(quantization.get("load_in_4bit")) or quant_type != "nf4":
        raise ValueError(
            f"Checkpoint {path} is not NF4: load_in_4bit={quantization.get('load_in_4bit')!r}, "
            f"bnb_4bit_quant_type={quant_type!r}"
        )
    if require_fsdp_storage:
        storage = str(quantization.get("bnb_4bit_quant_storage") or "").lower()
        if storage not in {"bfloat16", "torch.bfloat16"}:
            raise ValueError(
                "FSDP-QLoRA requires floating quantized storage matching the model dtype; "
                f"expected bfloat16, got {storage or 'missing'} in {path}"
            )


def assert_vllm_nf4_checkpoint(path: str | Path) -> None:
    """Require the byte-packed NF4 representation consumed by vLLM 0.19."""
    assert_nf4_checkpoint(path, require_fsdp_storage=False)
    storage = str(
        checkpoint_quantization(path).get("bnb_4bit_quant_storage") or ""
    ).lower()
    if storage not in {"uint8", "torch.uint8"}:
        raise ValueError(
            "vLLM 0.19 requires byte-packed pre-quantized BNB weights; "
            f"expected uint8 storage, got {storage or 'missing'} in {path}. "
            "Run `gemma-grpo prepare-vllm-nf4 --source-model ... --output-dir ...`."
        )


def _requantize_linears_to_nf4(
    model: Any,
    *,
    compute_dtype: Any,
    quant_storage: Any,
) -> Any:
    """Convert dense CPU Linear weights to saved bitsandbytes NF4 one layer at a time."""
    import gc

    import bitsandbytes as bnb
    import torch
    from tqdm.auto import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("NF4 conversion requires one CUDA GPU as a quantization scratch device")

    output_layer = model.get_output_embeddings()
    linear_names = [
        name
        for name, module in model.named_modules()
        if name
        and isinstance(module, torch.nn.Linear)
        and not isinstance(module, bnb.nn.Linear4bit)
        and module is not output_layer
    ]
    scratch_device = torch.device("cuda:0")
    for index, name in enumerate(tqdm(linear_names, desc="Converting linears to saved NF4")):
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, child_name)
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=old.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=quant_storage,
            device="cpu",
        )
        new.weight = bnb.nn.Params4bit(
            old.weight.detach(),
            requires_grad=False,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=quant_storage,
            module=new,
        )
        if old.bias is not None:
            new.bias = torch.nn.Parameter(old.bias.detach(), requires_grad=False)
        new.requires_grad_(False)
        # Params4bit performs the actual quantization during its first CUDA move.
        new = new.to(scratch_device)
        new = new.to("cpu")
        setattr(parent, child_name, new)
        del old, new
        if (index + 1) % 32 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    model.is_loaded_in_4bit = True
    model.is_quantized = True
    return model


def _replace_source_quantizer_for_save(model: Any, bnb_config: Any) -> None:
    """Detach compressed-tensors load conversions before saving BNB weights.

    Transformers 5 stores the exact WeightConverter chain used by
    ``from_pretrained`` on ``model._weight_conversions`` and reverses it during
    ``save_pretrained``. After we replace the dense modules with Linear4bit,
    that chain still describes the source compressed-tensors checkpoint and is
    no longer reversible. Keeping it produces ``NoneType.convert`` while
    saving. Rebuild only the model's normal key mapping and expose the actual
    BNB quantizer which owns the serialized checkpoint now.
    """
    from transformers.quantizers import AutoHfQuantizer

    # ``None`` intentionally asks save_pretrained to reconstruct only the
    # architecture's ordinary key-renaming mapping. It must not reuse the
    # compressed-tensors deserialization operations from the source load.
    model._weight_conversions = None
    model.hf_quantizer = AutoHfQuantizer.from_config(
        bnb_config,
        pre_quantized=True,
    )
    model.quantization_method = bnb_config.quant_method
    model.config.quantization_config = bnb_config


def _save_nf4_model(model: Any, directory: Path, *, max_shard_size: str) -> str:
    """Save BNB weights, falling back only for a non-reversible v5 mapping."""
    kwargs = {
        "safe_serialization": True,
        "max_shard_size": max_shard_size,
    }
    try:
        model.save_pretrained(directory, **kwargs)
        return "original"
    except AttributeError as error:
        # Some Transformers 5 VLM revisions have a model-level converter whose
        # reverse operation is literally ``None``. This is independent of BNB.
        # The supported escape hatch is to save the current canonical model
        # keys instead of trying to recreate a legacy checkpoint layout.
        if "NoneType" not in str(error) or "convert" not in str(error):
            raise
        print(
            "[nf4] source key mapping is not reversible; retrying in canonical "
            "Transformers format",
            flush=True,
        )
        shutil.rmtree(directory)
        directory.mkdir(parents=True)
        model.save_pretrained(
            directory,
            save_original_format=False,
            **kwargs,
        )
        return "canonical"


def prepare_vllm_nf4_checkpoint(
    *,
    source_model: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create a byte-packed vLLM view without dequantizing NF4 weights.

    FSDP-QLoRA requires floating BNB storage so it can shard Params4bit, while
    vLLM 0.19's built-in BNB loader allocates byte-packed parameters. The two
    layouts contain identical quantized bytes. This function bitcasts only the
    serialized quantized weight tensors from BF16 to uint8; it never reconstructs
    or requantizes a dense weight.
    """
    import gc

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    source_model = Path(source_model).resolve()
    output_dir = Path(output_dir).resolve()
    assert_nf4_checkpoint(source_model, require_fsdp_storage=True)
    if output_dir.exists():
        if any(output_dir.iterdir()) if output_dir.is_dir() else True:
            raise FileExistsError(f"Refusing to overwrite existing vLLM checkpoint: {output_dir}")
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(source_model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"NF4 checkpoint has no safetensors shards: {source_model}")

    source_bytes = sum(shard.stat().st_size for shard in shards)
    free_bytes = shutil.disk_usage(output_dir.parent).free
    reserve_bytes = 2 * 1024**3
    if free_bytes < source_bytes + reserve_bytes:
        raise OSError(
            "Not enough free disk space for the byte-packed vLLM checkpoint: "
            f"need about {(source_bytes + reserve_bytes) / 2**30:.1f} GiB including "
            f"a 2 GiB reserve, have {free_bytes / 2**30:.1f} GiB in {output_dir.parent}"
        )

    checkpoint_keys: set[str] = set()
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            checkpoint_keys.update(handle.keys())
    suffixes = (
        ".quant_state.bitsandbytes__nf4",
        ".quant_state.bitsandbytes__fp4",
    )
    quantized_weights = {
        key[: -len(suffix)]
        for key in checkpoint_keys
        for suffix in suffixes
        if key.endswith(suffix)
    }
    if not quantized_weights:
        raise ValueError(f"No serialized BNB quantization states found in {source_model}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        shutil.copytree(
            source_model,
            temporary,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.safetensors"),
        )
        converted_count = 0
        for shard in shards:
            state = load_file(str(shard), device="cpu")
            converted: dict[str, Any] = {}
            for key, tensor in state.items():
                if key in quantized_weights:
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(
                            f"Expected BF16 quantized storage for {key}, got {tensor.dtype}"
                        )
                    tensor = tensor.contiguous().view(torch.uint8).reshape(-1, 1)
                    converted_count += 1
                converted[key] = tensor
            save_file(
                converted,
                str(temporary / shard.name),
                metadata={"format": "pt"},
            )
            del state, converted
            gc.collect()

        if converted_count != len(quantized_weights):
            raise RuntimeError(
                f"Converted {converted_count} NF4 weights, expected {len(quantized_weights)}"
            )
        config_path = temporary / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization_config")
        if not isinstance(quantization, dict):
            raise ValueError(f"Invalid quantization_config in {config_path}")
        quantization["bnb_4bit_quant_storage"] = "uint8"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_path = temporary / NF4_MANIFEST
        manifest = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "format": "bitsandbytes-nf4-vllm",
                "source_model": str(source_model),
                "quant_storage": "uint8",
                "conversion": "lossless-storage-bitcast",
                "converted_weights": converted_count,
                "source_safetensors_bytes": source_bytes,
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        assert_vllm_nf4_checkpoint(temporary)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"[nf4] byte-packed vLLM checkpoint ready: {output_dir} "
        f"({len(quantized_weights)} weights, no requantization)",
        flush=True,
    )
    return output_dir


def convert_to_nf4_checkpoint(
    *,
    source_model: str | Path,
    output_dir: Path,
    trust_remote_code: bool = False,
    max_shard_size: str = "4GB",
) -> Path:
    """Create the reusable BF16-storage NF4 checkpoint for FSDP-QLoRA.

    The competition compressed-tensors checkpoint is decompressed only during
    this command. ``prepare-vllm-nf4`` derives a byte-packed rollout view from
    this checkpoint without dequantization or another quantization pass.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        CompressedTensorsConfig,
    )

    source_model = Path(source_model).resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()) if output_dir.is_dir() else True:
            raise FileExistsError(f"Refusing to overwrite existing NF4 checkpoint: {output_dir}")
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source_quantization = checkpoint_quantization(source_model)
    source_method = str(source_quantization.get("quant_method") or "").lower()
    if source_method in {"bitsandbytes", "bitsandbytes_4bit"}:
        assert_nf4_checkpoint(source_model)
        raise ValueError(
            f"{source_model} is already a compatible NF4 checkpoint; use it directly"
        )
    if source_method != "compressed-tensors":
        raise ValueError(
            "The one-time converter expects the competition compressed-tensors W4 checkpoint; "
            f"got quant_method={source_method!r}"
        )

    dtype = torch.bfloat16
    compressed_parameters = inspect.signature(CompressedTensorsConfig).parameters
    if "dequantize" in compressed_parameters:
        compressed_config = CompressedTensorsConfig(dequantize=True)
    else:
        compressed_config = CompressedTensorsConfig(run_compressed=False)

    print(f"[nf4] loading and decompressing source on CPU: {source_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        source_model,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        low_cpu_mem_usage=True,
        quantization_config=compressed_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=trust_remote_code,
    )
    print("[nf4] quantizing dense linears once; storage=bfloat16, compute=bfloat16", flush=True)
    model = _requantize_linears_to_nf4(
        model,
        compute_dtype=dtype,
        quant_storage=dtype,
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=dtype,
    )
    _replace_source_quantizer_for_save(model, bnb_config)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        saved_key_format = _save_nf4_model(
            model,
            temporary,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(temporary)
        manifest = {
            "format": "bitsandbytes-nf4-fsdp",
            "source_model": str(source_model),
            "source_quant_method": source_method,
            "compute_dtype": "bfloat16",
            "quant_storage": "bfloat16",
            "double_quant": True,
            "key_format": saved_key_format,
            "packages": {},
        }
        for package in ("torch", "transformers", "bitsandbytes"):
            try:
                manifest["packages"][package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                manifest["packages"][package] = "unknown"
        (temporary / NF4_MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        assert_nf4_checkpoint(temporary)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"[nf4] reusable checkpoint ready: {output_dir}", flush=True)
    return output_dir
