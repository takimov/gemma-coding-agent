from __future__ import annotations

import inspect
import json
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .groups import add_group_advantages, trainable_groups
from .objective import grpo_loss
from .policy import adapter_fingerprint
from .tasks import read_jsonl
from .tokenization import TokenizedTrajectory, assert_prefix_preserving, tokenize_trajectory


def _requantize_linears_to_bnb4(model: Any, compute_dtype: Any) -> Any:
    """Convert a dense CPU model to bitsandbytes NF4 one Linear at a time.

    The source compressed-tensors checkpoint must first be dequantized on CPU.
    Each dense layer is then quantized on a scratch GPU and immediately moved
    back to CPU, so neither GPU nor system RAM needs a second full model copy.
    """
    import gc

    import bitsandbytes as bnb
    import torch
    from tqdm.auto import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("Requantizing compressed-tensors to bnb4 requires a CUDA GPU")

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
    for index, name in enumerate(tqdm(linear_names, desc="Requantizing linears to NF4")):
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
            quant_storage=torch.uint8,
            device="cpu",
        )
        new.weight = bnb.nn.Params4bit(
            old.weight.detach(),
            requires_grad=False,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=torch.uint8,
            module=new,
        )
        if old.bias is not None:
            new.bias = torch.nn.Parameter(old.bias.detach(), requires_grad=False)
        new.requires_grad_(False)
        # Params4bit performs the actual quantization on the first CUDA move.
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


def _no_split_module_classes(model: Any) -> list[str] | None:
    """Return Accelerate-compatible, flat module class names.

    PEFT wrappers may aggregate ``_no_split_modules`` from their children as
    nested sets.  Accelerate expects a flat iterable of hashable class-name
    strings and otherwise fails while constructing its device map.
    """
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    raw = getattr(base_model, "_no_split_modules", None)
    if raw is None:
        return None

    pending = [raw]
    names: set[str] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            names.add(value)
        elif isinstance(value, type):
            names.add(value.__name__)
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        else:
            raise TypeError(f"Unsupported _no_split_modules value: {value!r}")
    return sorted(names) or None


def _load_model_and_tokenizer(
    config: ExperimentConfig, model_path: str | None, resume_adapter: Path | None
):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        CompressedTensorsConfig,
    )

    path = model_path or config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=config.model.trust_remote_code,
    )
    assert_prefix_preserving(tokenizer)

    dtype = getattr(torch, config.model.torch_dtype)
    checkpoint_config = AutoConfig.from_pretrained(
        path,
        trust_remote_code=config.model.trust_remote_code,
    )
    checkpoint_quantization = getattr(checkpoint_config, "quantization_config", None)
    checkpoint_quant_method = (
        checkpoint_quantization.get("quant_method")
        if isinstance(checkpoint_quantization, dict)
        else getattr(checkpoint_quantization, "quant_method", None)
    )
    requantize_compressed_to_bnb4 = (
        checkpoint_quant_method == "compressed-tensors"
        and config.model.load_mode == "bnb4"
    )
    stage_compressed_on_cpu = (
        checkpoint_quant_method == "compressed-tensors"
        and torch.cuda.is_available()
        and config.model.device_map not in (None, "cpu")
    )
    load_kwargs: dict[str, Any] = {
        "device_map": "cpu" if stage_compressed_on_cpu else config.model.device_map,
        "trust_remote_code": config.model.trust_remote_code,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if (
        not stage_compressed_on_cpu
        and config.model.max_memory_per_gpu_gib is not None
        and torch.cuda.is_available()
    ):
        load_kwargs["max_memory"] = {
            index: f"{config.model.max_memory_per_gpu_gib}GiB"
            for index in range(torch.cuda.device_count())
        }
    if config.model.load_mode == "bnb4" and not requantize_compressed_to_bnb4:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif checkpoint_quant_method == "compressed-tensors":
        # Packed compressed-tensors checkpoints do not expose ``weight`` until
        # decompression. PEFT injects LoRA before the first forward and therefore
        # cannot wrap those packed Linear modules. Decompress during loading so
        # device-map placement sees the real bf16 sizes and PEFT receives ordinary
        # Linear layers. The dense base is frozen immediately after LoRA injection.
        compressed_parameters = inspect.signature(CompressedTensorsConfig).parameters
        if "dequantize" in compressed_parameters:
            load_kwargs["quantization_config"] = CompressedTensorsConfig(dequantize=True)
        else:
            # Transformers <=4.x exposes the same behavior under the legacy
            # loading attribute: run_compressed=False means dequantize on load.
            load_kwargs["quantization_config"] = CompressedTensorsConfig(
                run_compressed=False
            )
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    if requantize_compressed_to_bnb4:
        print("[train] converting decompressed CPU model to bitsandbytes NF4")
        model = _requantize_linears_to_bnb4(model, dtype)
        model.config.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ).to_dict()
    if config.model.load_mode == "bnb4":
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if resume_adapter is not None:
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=config.lora.target_modules,
                bias="none",
            ),
        )
    if stage_compressed_on_cpu:
        from accelerate import dispatch_model, infer_auto_device_map
        from accelerate.utils import get_balanced_memory

        # Decompression needs a packed tensor, a temporary unpacked tensor, and
        # the final bf16 tensor at the same time. Staging on CPU avoids that VRAM
        # peak. In bnb4 mode the dense layers are then requantized one at a time,
        # before dispatch. Reserve at least 4 GiB per GPU for training state.
        max_memory: dict[int | str, str] = {}
        for index in range(torch.cuda.device_count()):
            physical_gib = torch.cuda.get_device_properties(index).total_memory // (1024**3)
            requested_gib = config.model.max_memory_per_gpu_gib or physical_gib
            weight_budget_gib = min(requested_gib, max(1, physical_gib - 4))
            max_memory[index] = f"{weight_budget_gib}GiB"
        max_memory["cpu"] = "150GiB"
        no_split = _no_split_module_classes(model)
        balanced_memory = get_balanced_memory(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
            dtype=dtype,
        )
        device_map = infer_auto_device_map(
            model,
            max_memory=balanced_memory,
            no_split_module_classes=no_split,
            dtype=dtype,
        )
        offloaded = sorted(
            name
            for name, device in device_map.items()
            if str(device) in {"cpu", "disk"}
        )
        if offloaded:
            raise RuntimeError(
                "The decompressed base does not fit in the configured GPU memory; "
                f"CPU/disk assignments include: {offloaded[:5]}"
            )
        print(f"[train] dispatching decompressed model with device map: {device_map}")
        model = dispatch_model(model, device_map=device_map)
    return model, tokenizer


def _embedding_device(model: Any):
    return model.get_input_embeddings().weight.device


def _logit_selection_device(model: Any):
    """Return the device holding hidden states immediately before ``lm_head``.

    Gemma applies tensor-valued ``logits_to_keep`` to the final hidden states
    before calling ``lm_head``.  Under Accelerate model parallelism those hidden
    states live with the language-model norm, not necessarily with embeddings
    or the (tied) output head.
    """
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    for owner in (model, base_model):
        device_map = getattr(owner, "hf_device_map", None)
        if device_map:
            for name, device in device_map.items():
                if name.endswith("language_model.norm"):
                    return f"cuda:{device}" if isinstance(device, int) else device
    for name, parameter in model.named_parameters():
        if name.endswith("language_model.norm.weight"):
            return parameter.device
    return _embedding_device(model)


def _supports_selective_logits(model: Any) -> bool:
    signature = inspect.signature(model.get_base_model().forward)
    return "logits_to_keep" in signature.parameters


def _selected_logprobs(model: Any, tokenized: TokenizedTrajectory):
    import torch

    input_device = _embedding_device(model)
    input_ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=input_device)
    target_positions = torch.tensor(
        tokenized.action_positions, dtype=torch.long, device=input_device
    )
    kwargs: dict[str, Any] = {"input_ids": input_ids, "use_cache": False}
    base_signature = inspect.signature(model.get_base_model().forward)
    if "logits_to_keep" in base_signature.parameters:
        logit_positions = torch.tensor(
            [position - 1 for position in tokenized.action_positions],
            dtype=torch.long,
            device=_logit_selection_device(model),
        )
        kwargs["logits_to_keep"] = logit_positions
    elif "num_logits_to_keep" in base_signature.parameters:
        # Older model implementations only support a suffix. It is correct only
        # when all selected targets lie in that suffix, so fall back to full logits.
        logit_positions = target_positions - 1
    else:
        logit_positions = target_positions - 1
    outputs = model(**kwargs)
    logits = outputs.logits
    logit_positions = logit_positions.to(logits.device)
    if logits.shape[1] == len(logit_positions):
        selected_logits = logits
    else:
        selected_logits = logits.index_select(1, logit_positions)
    labels = input_ids.index_select(1, target_positions).to(selected_logits.device)
    return torch.log_softmax(selected_logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(0).squeeze(-1)


def _rollout_logprobs(model: Any, segments: list[TokenizedTrajectory]):
    import torch

    return torch.cat([_selected_logprobs(model, segment) for segment in segments])


def _row_segments(tokenizer: Any, row: dict[str, Any]) -> list[TokenizedTrajectory]:
    raw_segments = row.get("token_segments")
    if isinstance(raw_segments, list) and raw_segments:
        return [
            TokenizedTrajectory(
                input_ids=[int(token) for token in segment["input_ids"]],
                action_positions=[int(position) for position in segment["action_positions"]],
            )
            for segment in raw_segments
        ]
    return [tokenize_trajectory(tokenizer, row["messages"], row.get("tools"))]


def _disable_adapter(model: Any):
    method = getattr(model, "disable_adapter", None)
    return method() if method is not None else nullcontext()


def train_adapter(
    *,
    config: ExperimentConfig,
    rollout_paths: list[Path],
    output_dir: Path,
    model_path: str | None = None,
    resume_adapter: Path | None = None,
) -> dict[str, Any]:
    import torch

    random.seed(config.grpo.seed)
    torch.manual_seed(config.grpo.seed)

    source_rows = []
    for path in rollout_paths:
        source_rows.extend(read_jsonl(path))
    policy_ids = {str(row.get("policy_id", "")) for row in source_rows}
    if len(policy_ids) != 1 or "" in policy_ids:
        raise ValueError(f"Rollouts must contain exactly one non-empty policy_id; got {sorted(policy_ids)}")
    expected_policy_id = adapter_fingerprint(resume_adapter) if resume_adapter is not None else "base"
    actual_policy_id = next(iter(policy_ids))
    if actual_policy_id != expected_policy_id:
        raise ValueError(
            "Rollout policy does not match the training start policy: "
            f"rollouts={actual_policy_id}, trainer={expected_policy_id}"
        )
    rows = add_group_advantages(source_rows)
    groups = trainable_groups(rows)
    if not groups:
        raise ValueError("No trainable groups: every group has identical reward or is unscorable")

    model, tokenizer = _load_model_and_tokenizer(config, model_path, resume_adapter)
    selective_logits = _supports_selective_logits(model)
    if not selective_logits and config.grpo.max_seq_length > 8192:
        raise RuntimeError(
            "The model forward method lacks selective logits support. Reduce max_seq_length "
            "to 8192 or use a Transformers revision whose Gemma implementation accepts logits_to_keep."
        )
    nonzero_dropouts = [
        module for module in model.modules() if isinstance(module, torch.nn.Dropout) and module.p > 0
    ]
    if nonzero_dropouts:
        raise RuntimeError(
            "The loaded policy contains non-zero dropout, so cached old log-probabilities would not be "
            "deterministic. Train/export an adapter with dropout=0."
        )
    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.grpo.learning_rate)

    group_items = list(groups.items())
    random.shuffle(group_items)
    if config.grpo.max_steps is not None:
        group_items = group_items[: config.grpo.max_steps]

    metrics: list[dict[str, Any]] = []
    prepared_groups: list[
        tuple[str, list[tuple[dict[str, Any], list[TokenizedTrajectory]]]]
    ] = []
    for group_id, group_rows in group_items:
        prepared: list[tuple[dict[str, Any], list[TokenizedTrajectory]]] = []
        for row in group_rows:
            segments = _row_segments(tokenizer, row)
            if any(len(segment.input_ids) > config.grpo.max_seq_length for segment in segments):
                continue
            prepared.append((row, segments))
        if len(prepared) < 2:
            metrics.append({"group_id": group_id, "skipped": "fewer_than_two_usable_rollouts"})
            continue
        prepared_groups.append((group_id, prepared))

    if not prepared_groups:
        raise ValueError("No group contains at least two trajectories within max_seq_length")

    # Snapshot every behavior-policy and reference log-probability before the
    # first optimizer step. Recomputing old_logprobs between groups would make
    # later groups appear on-policy under a policy that did not generate them.
    cached_groups: list[
        tuple[
            str,
            list[
                tuple[
                    dict[str, Any],
                    list[TokenizedTrajectory],
                    Any,
                    Any,
                ]
            ],
        ]
    ] = []
    model.eval()
    with torch.no_grad():
        for group_id, prepared in prepared_groups:
            cached_rollouts = []
            for row, segments in prepared:
                old = _rollout_logprobs(model, segments).detach().cpu()
                with _disable_adapter(model):
                    reference = _rollout_logprobs(model, segments).detach().cpu()
                cached_rollouts.append((row, segments, old, reference))
            cached_groups.append((group_id, cached_rollouts))

    optimizer_step = 0
    model.train()
    for epoch in range(1, config.grpo.epochs + 1):
        epoch_groups = list(cached_groups)
        random.shuffle(epoch_groups)
        for group_id, cached_rollouts in epoch_groups:
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_kl = 0.0
            for row, segments, old_cpu, reference_cpu in cached_rollouts:
                new = _rollout_logprobs(model, segments)
                old = old_cpu.to(new.device)
                reference = reference_cpu.to(new.device)
                loss, kl = grpo_loss(
                    new,
                    old,
                    reference,
                    advantage=float(row["advantage"]),
                    clip_epsilon=config.grpo.clip_epsilon,
                    beta=config.grpo.beta,
                )
                (loss / len(cached_rollouts)).backward()
                total_loss += float(loss.detach().cpu())
                total_kl += float(kl.detach().cpu())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config.grpo.max_grad_norm
            )
            optimizer.step()
            optimizer_step += 1
            metrics.append(
                {
                    "optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "group_id": group_id,
                    "rollouts": len(cached_rollouts),
                    "loss": total_loss / len(cached_rollouts),
                    "kl": total_kl / len(cached_rollouts),
                    "grad_norm": float(
                        grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm
                    ),
                }
            )

    if optimizer_step == 0:
        raise ValueError("No optimization step was executed; inspect sequence lengths and trace messages")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty adapter directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    summary = {
        "groups_available": len(groups),
        "groups_processed": len(cached_groups),
        "optimizer_steps": optimizer_step,
        "token_sources": dict(Counter(str(row.get("token_source", "unknown")) for row in source_rows)),
        "metrics": metrics,
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary

