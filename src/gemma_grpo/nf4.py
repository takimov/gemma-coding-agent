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


def assert_compressed_w4_checkpoint(path: str | Path) -> None:
    """Validate the official inference-optimized QAT W4 checkpoint."""
    quantization = checkpoint_quantization(path)
    method = str(quantization.get("quant_method") or "").lower()
    if method != "compressed-tensors":
        raise ValueError(
            f"Expected a compressed-tensors QAT W4 checkpoint at {path}, "
            f"got quant_method={method!r}. Pass the official "
            "gemma-4-31b-it-qat-w4a16-ct directory as --rollout-model-path."
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


def convert_to_nf4_checkpoint(
    *,
    source_model: str | Path,
    output_dir: Path,
    trust_remote_code: bool = False,
    max_shard_size: str = "4GB",
) -> Path:
    """Create the reusable BF16-storage NF4 checkpoint for FSDP-QLoRA.

    The competition compressed-tensors checkpoint is decompressed only during
    this command. Managed vLLM bitcasts its packed BF16 containers to uint8
    views while streaming them, without another on-disk checkpoint.
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
