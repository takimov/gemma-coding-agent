from __future__ import annotations

import json
import os
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .groups import add_group_advantages, trainable_groups
from .monitoring import TrainingMonitor
from .nf4 import assert_nf4_checkpoint
from .objective import grpo_loss
from .policy import adapter_fingerprint
from .tasks import read_jsonl
from .tokenization import TokenizedTrajectory, assert_prefix_preserving, tokenize_trajectory


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


def _load_tokenizer(config: ExperimentConfig, model_path: str | None):
    from transformers import AutoTokenizer

    path = model_path or config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=config.model.trust_remote_code,
    )
    assert_prefix_preserving(tokenizer)
    return tokenizer


def _load_model_and_tokenizer(
    config: ExperimentConfig,
    model_path: str | None,
    resume_adapter: Path | None,
    *,
    tokenizer: Any | None = None,
    monitor: TrainingMonitor | None = None,
    fsdp: bool = False,
):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    path = model_path or config.model.name_or_path

    def phase(name: str) -> None:
        if monitor is not None:
            monitor.set_phase(name)

    if tokenizer is None:
        phase("load_tokenizer")
        tokenizer = _load_tokenizer(config, model_path)

    dtype = getattr(torch, config.model.torch_dtype)
    if config.model.load_mode == "nf4":
        assert_nf4_checkpoint(path, require_fsdp_storage=fsdp)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if not fsdp and config.model.device_map is not None:
        load_kwargs["device_map"] = config.model.device_map
    phase("load_nf4_checkpoint" if config.model.load_mode == "nf4" else "load_checkpoint")
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    if config.model.load_mode == "nf4":
        if not bool(getattr(model, "is_loaded_in_4bit", False)):
            raise RuntimeError("Saved NF4 checkpoint was not loaded as a 4-bit model")
        phase("prepare_kbit")
        model = prepare_model_for_kbit_training(model)
        # PEFT intentionally promotes non-quantized parameters to FP32. Gemma's
        # tied 262k-token embedding/lm_head would then consume about 5.6 GiB on
        # every FSDP rank. It is frozen, so keep that shared tensor in BF16 to
        # match the NF4 storage/compute dtype and halve the replicated cost.
        projection_modules = {
            id(module): module
            for module in (model.get_input_embeddings(), model.get_output_embeddings())
            if module is not None
        }
        for module in projection_modules.values():
            module.to(dtype=dtype)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if resume_adapter is not None:
        phase("load_lora")
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        phase("inject_lora")
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
    return model, tokenizer


def _embedding_device(model: Any):
    return model.get_input_embeddings().weight.device


def _causal_lm_parts(model: Any) -> tuple[Any, Any, float | None]:
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(causal_lm, "model", None)
    output_head = causal_lm.get_output_embeddings()
    if backbone is None or output_head is None:
        raise RuntimeError(
            "The loaded causal LM must expose .model and get_output_embeddings() "
            "for memory-bounded action log-probabilities"
        )
    text_config = causal_lm.config
    if hasattr(text_config, "get_text_config"):
        text_config = text_config.get_text_config()
    softcap = getattr(text_config, "final_logit_softcapping", None)
    return backbone, output_head, float(softcap) if softcap is not None else None


def _project_action_logprobs(
    output_head: Any,
    selected_hidden: Any,
    labels: Any,
    *,
    chunk_size: int,
    final_logit_softcapping: float | None = None,
):
    """Project hidden states to token log-probs without a full action×vocab tensor."""
    import torch
    from torch.utils.checkpoint import checkpoint

    def project(hidden_chunk, label_chunk):
        logits = output_head(hidden_chunk)
        if final_logit_softcapping is not None:
            logits = torch.tanh(logits / final_logit_softcapping)
            logits = logits * final_logit_softcapping
        return (
            torch.log_softmax(logits.float(), dim=-1)
            .gather(-1, label_chunk.to(logits.device).unsqueeze(-1))
            .squeeze(-1)
        )

    chunks = []
    for start in range(0, selected_hidden.shape[1], chunk_size):
        end = min(start + chunk_size, selected_hidden.shape[1])
        hidden_chunk = selected_hidden[:, start:end, :]
        label_chunk = labels[:, start:end]
        if torch.is_grad_enabled() and hidden_chunk.requires_grad:
            values = checkpoint(
                project,
                hidden_chunk,
                label_chunk,
                use_reentrant=False,
            )
        else:
            values = project(hidden_chunk, label_chunk)
        chunks.append(values)
    return torch.cat(chunks, dim=1).squeeze(0)


def _build_selective_logprob_model(policy: Any, *, chunk_size: int):
    """Wrap the complete selective-logprob path in one FSDP-visible forward.

    Keeping the vocabulary projection loop inside the root forward ensures the
    large tied lm_head is all-gathered once per trajectory instead of once per
    logit chunk.
    """
    import torch

    class SelectiveLogprobModel(torch.nn.Module):
        def __init__(self, wrapped_policy: Any, projection_chunk_size: int) -> None:
            super().__init__()
            self.policy = wrapped_policy
            self.chunk_size = projection_chunk_size
            self._gemma_grpo_selective_logprobs = True

        def forward(self, input_ids, target_positions):
            backbone, output_head, softcap = _causal_lm_parts(self.policy)
            outputs = backbone(input_ids=input_ids, use_cache=False)
            hidden_states = outputs.last_hidden_state
            logit_positions = (target_positions - 1).to(hidden_states.device)
            selected_hidden = hidden_states.index_select(1, logit_positions)
            labels = input_ids.index_select(1, target_positions.to(input_ids.device))
            return _project_action_logprobs(
                output_head,
                selected_hidden,
                labels,
                chunk_size=self.chunk_size,
                final_logit_softcapping=softcap,
            )

    return SelectiveLogprobModel(policy, chunk_size)


def _wrapped_policy(model: Any) -> Any:
    return model.policy if hasattr(model, "policy") else model


def _is_selective_logprob_model(model: Any) -> bool:
    if bool(getattr(model, "_gemma_grpo_selective_logprobs", False)):
        return True
    wrapped = getattr(model, "module", None)
    return bool(getattr(wrapped, "_gemma_grpo_selective_logprobs", False))


def _selected_logprobs(
    model: Any,
    tokenized: TokenizedTrajectory,
    *,
    chunk_size: int,
    input_device: Any | None = None,
):
    import torch

    if _is_selective_logprob_model(model):
        device = input_device
        if device is None:
            device = next(model.parameters()).device
        input_ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=device)
        target_positions = torch.tensor(
            tokenized.action_positions, dtype=torch.long, device=device
        )
        return model(input_ids=input_ids, target_positions=target_positions)

    device = input_device or _embedding_device(model)
    input_ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=device)
    target_positions = torch.tensor(
        tokenized.action_positions, dtype=torch.long, device=device
    )
    backbone, output_head, final_logit_softcapping = _causal_lm_parts(model)
    outputs = backbone(input_ids=input_ids, use_cache=False)
    hidden_states = outputs.last_hidden_state
    logit_positions = (target_positions - 1).to(hidden_states.device)
    selected_hidden = hidden_states.index_select(1, logit_positions)
    labels = input_ids.index_select(1, target_positions)
    return _project_action_logprobs(
        output_head,
        selected_hidden,
        labels,
        chunk_size=chunk_size,
        final_logit_softcapping=final_logit_softcapping,
    )


def _rollout_logprobs(
    model: Any,
    segments: list[TokenizedTrajectory],
    *,
    chunk_size: int,
    input_device: Any | None = None,
):
    import torch

    return torch.cat(
        [
            _selected_logprobs(
                model,
                segment,
                chunk_size=chunk_size,
                input_device=input_device,
            )
            for segment in segments
        ]
    )


def _distributed_rollout_logprobs(
    model: Any,
    segments: list[TokenizedTrajectory],
    *,
    chunk_size: int,
    input_device: Any,
    world_size: int,
    dummy_token_id: int,
):
    """Score a rollout while keeping FSDP forward counts equal on every rank.

    Exact ATIF traces contain one segment per assistant call, and different
    trajectories can contain different numbers of calls. FSDP collectives
    deadlock if one rank enters the wrapped decoder more often than another.
    Short ranks therefore execute tiny two-token padding forwards. The returned
    zero-valued anchor must be included in the training loss so autograd also
    traverses those padding forwards during backward.
    """
    import torch

    maximum = len(segments)
    if world_size > 1:
        count = torch.tensor(maximum, dtype=torch.int64, device=input_device)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX)
        maximum = int(count.item())
    if maximum < 1:
        raise ValueError("A rollout must contain at least one token segment")

    padded = list(segments)
    padded.extend(
        TokenizedTrajectory(
            input_ids=[dummy_token_id, dummy_token_id],
            action_positions=[1],
        )
        for _ in range(maximum - len(segments))
    )
    outputs = [
        _selected_logprobs(
            model,
            segment,
            chunk_size=chunk_size,
            input_device=input_device,
        )
        for segment in padded
    ]
    real_count = sum(len(segment.action_positions) for segment in segments)
    combined = torch.cat(outputs)
    real = combined[:real_count]
    padding_anchor = combined[real_count:].sum() * 0.0
    return real, padding_anchor


def _validate_trajectory(trajectory: TokenizedTrajectory) -> TokenizedTrajectory:
    positions = trajectory.action_positions
    if not trajectory.input_ids:
        raise ValueError("Tokenized trajectory has no input tokens")
    if not positions:
        raise ValueError("Tokenized trajectory has no action tokens")
    if positions != sorted(set(positions)):
        raise ValueError("Action positions must be unique and sorted")
    if positions[0] <= 0 or positions[-1] >= len(trajectory.input_ids):
        raise ValueError(
            "Action positions must be between 1 and len(input_ids)-1; "
            f"got range {positions[0]}..{positions[-1]} for {len(trajectory.input_ids)} tokens"
        )
    return trajectory


def _row_segments(tokenizer: Any, row: dict[str, Any]) -> list[TokenizedTrajectory]:
    raw_segments = row.get("token_segments")
    if isinstance(raw_segments, list) and raw_segments:
        return [
            _validate_trajectory(
                TokenizedTrajectory(
                    input_ids=[int(token) for token in segment["input_ids"]],
                    action_positions=[
                        int(position) for position in segment["action_positions"]
                    ],
                )
            )
            for segment in raw_segments
        ]
    return [
        _validate_trajectory(
            tokenize_trajectory(tokenizer, row["messages"], row.get("tools"))
        )
    ]


def _disable_adapter(model: Any):
    method = getattr(_wrapped_policy(model), "disable_adapter", None)
    return method() if method is not None else nullcontext()


def _policy_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip the selective-logprob wrapper prefix from a gathered state dict."""
    prefixes = ("policy.", "module.policy.", "_fsdp_wrapped_module.policy.")
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        matched = next((prefix for prefix in prefixes if key.startswith(prefix)), None)
        if matched is not None:
            result[key[len(matched) :]] = value
    if not result:
        raise RuntimeError(
            "Gathered FSDP state dict does not contain the selective model's policy prefix"
        )
    return result


def _next_telemetry_path(output_dir: Path) -> Path:
    base = output_dir.parent / f"{output_dir.name}.telemetry.jsonl"
    if not base.exists():
        return base
    for version in range(2, 10_000):
        candidate = output_dir.parent / f"{output_dir.name}.telemetry-v{version}.jsonl"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a telemetry filename next to {output_dir}")


class _NullMonitor:
    def set_phase(self, phase: str) -> None:
        del phase


def _create_accelerator(config: ExperimentConfig):
    from accelerate import Accelerator

    if config.distributed.strategy != "fsdp":
        return Accelerator(mixed_precision="no"), None

    from accelerate import FullyShardedDataParallelPlugin
    from accelerate.utils import enable_fsdp_ram_efficient_loading
    from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy, StateDictType

    enable_fsdp_ram_efficient_loading()
    plugin = FullyShardedDataParallelPlugin(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=config.distributed.cpu_offload,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        limit_all_gathers=config.distributed.limit_all_gathers,
        use_orig_params=False,
        sync_module_states=True,
        cpu_ram_efficient_loading=True,
    )
    return Accelerator(mixed_precision="no", fsdp_plugin=plugin), plugin


def _prepare_replicated_projection_modules(policy: Any, accelerator: Any, fsdp_plugin: Any):
    """Keep tied embeddings/lm_head replicated for checkpointed chunk recompute.

    Transformer gradient checkpointing re-enters decoder FSDP units through
    their own forwards. The chunked vocabulary projection is checkpointed below
    the root wrapper, so its tied weight must remain materialized through
    backward. Replicating this one BF16 tensor costs roughly 2.8 GiB per L4 and
    avoids an all-gather for every 32-token vocabulary chunk.
    """
    import torch

    input_module = policy.get_input_embeddings()
    output_module = policy.get_output_embeddings()
    if input_module is None or output_module is None:
        raise RuntimeError("FSDP policy must expose input and output embeddings")
    tied = getattr(input_module, "weight", None) is getattr(output_module, "weight", None)
    modules = [input_module]
    if output_module is not input_module:
        modules.append(output_module)
    ignored_trainable = [
        name
        for module in modules
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if ignored_trainable:
        raise RuntimeError(
            "The replicated embedding/lm_head contains trainable parameters that FSDP "
            f"would not reduce: {ignored_trainable}. Exclude the output head from LoRA."
        )
    fsdp_plugin.ignored_modules = modules

    if accelerator.is_main_process:
        input_module.to(accelerator.device)
        if not tied and output_module is not input_module:
            output_module.to(accelerator.device)
    else:
        input_module.to_empty(device=accelerator.device, recurse=False)
        if tied:
            output_module.weight = input_module.weight
            bias = getattr(output_module, "bias", None)
            if bias is not None and getattr(bias, "is_meta", False):
                output_module.bias = torch.nn.Parameter(
                    torch.empty_like(bias, device=accelerator.device),
                    requires_grad=bias.requires_grad,
                )
        elif output_module is not input_module:
            output_module.to_empty(device=accelerator.device, recurse=False)
    return modules


def _broadcast_replicated_modules(modules: list[Any]) -> None:
    import torch.distributed as dist

    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters(recurse=False):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            dist.broadcast(parameter.data, src=0)
        for buffer in module.buffers(recurse=False):
            if id(buffer) in seen:
                continue
            seen.add(id(buffer))
            dist.broadcast(buffer.data, src=0)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fsdp_worker_entry(
    local_rank: int,
    world_size: int,
    master_port: int,
    config: ExperimentConfig,
    rollout_paths: list[Path],
    output_dir: Path,
    model_path: str | None,
    resume_adapter: Path | None,
    telemetry_path: Path,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = "0"
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

    import torch

    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        with TrainingMonitor(
            telemetry_path,
            interval_seconds=config.grpo.monitor_interval_seconds,
            console_seconds=config.grpo.monitor_console_seconds,
        ) as monitor:
            _train_adapter_impl(
                config=config,
                rollout_paths=rollout_paths,
                output_dir=output_dir,
                model_path=model_path,
                resume_adapter=resume_adapter,
                monitor=monitor,
                telemetry_path=telemetry_path,
            )
    else:
        _train_adapter_impl(
            config=config,
            rollout_paths=rollout_paths,
            output_dir=output_dir,
            model_path=model_path,
            resume_adapter=resume_adapter,
            monitor=_NullMonitor(),
            telemetry_path=telemetry_path,
        )


def train_adapter(
    *,
    config: ExperimentConfig,
    rollout_paths: list[Path],
    output_dir: Path,
    model_path: str | None = None,
    resume_adapter: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty adapter directory {output_dir}")
    telemetry_path = _next_telemetry_path(output_dir)
    print(f"[train] telemetry: {telemetry_path}", flush=True)
    if config.distributed.strategy == "fsdp" and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        import torch.multiprocessing as mp

        if not __import__("torch").cuda.is_available():
            raise RuntimeError("FSDP-QLoRA requires CUDA")
        visible_gpus = __import__("torch").cuda.device_count()
        if visible_gpus < config.distributed.num_processes:
            raise RuntimeError(
                f"FSDP requested {config.distributed.num_processes} processes but only "
                f"{visible_gpus} CUDA devices are visible"
            )
        print(
            f"[train] launching FSDP-QLoRA on {config.distributed.num_processes} GPUs",
            flush=True,
        )
        mp.spawn(
            _fsdp_worker_entry,
            args=(
                config.distributed.num_processes,
                _free_port(),
                config,
                rollout_paths,
                output_dir,
                model_path,
                resume_adapter,
                telemetry_path,
            ),
            nprocs=config.distributed.num_processes,
            join=True,
        )
        metrics_path = output_dir / "training_metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError("FSDP workers exited without writing training_metrics.json")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    with TrainingMonitor(
        telemetry_path,
        interval_seconds=config.grpo.monitor_interval_seconds,
        console_seconds=config.grpo.monitor_console_seconds,
    ) as monitor:
        return _train_adapter_impl(
            config=config,
            rollout_paths=rollout_paths,
            output_dir=output_dir,
            model_path=model_path,
            resume_adapter=resume_adapter,
            monitor=monitor,
            telemetry_path=telemetry_path,
        )


def _train_adapter_impl(
    *,
    config: ExperimentConfig,
    rollout_paths: list[Path],
    output_dir: Path,
    model_path: str | None,
    resume_adapter: Path | None,
    monitor: TrainingMonitor,
    telemetry_path: Path,
) -> dict[str, Any]:
    import torch

    accelerator, fsdp_plugin = _create_accelerator(config)
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    if config.distributed.strategy == "fsdp":
        if world_size != config.distributed.num_processes:
            raise RuntimeError(
                f"FSDP world size is {world_size}, expected {config.distributed.num_processes}"
            )
    elif world_size != 1:
        raise RuntimeError("distributed.strategy=none must run in one process")

    random.seed(config.grpo.seed)
    torch.manual_seed(config.grpo.seed)

    monitor.set_phase("validate_rollouts")
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

    group_items = list(groups.items())
    random.shuffle(group_items)

    # Tokenize and reject unusable groups before paying the model-load cost.
    monitor.set_phase("tokenize_rollouts")
    tokenizer = _load_tokenizer(config, model_path)
    metrics: list[dict[str, Any]] = []
    prepared_groups: list[
        tuple[str, list[tuple[dict[str, Any], list[TokenizedTrajectory]]]]
    ] = []
    for group_id, group_rows in group_items:
        prepared: list[tuple[dict[str, Any], list[TokenizedTrajectory]]] = []
        oversized: list[dict[str, Any]] = []
        for row in group_rows:
            segments = _row_segments(tokenizer, row)
            lengths = [len(segment.input_ids) for segment in segments]
            action_tokens = sum(len(segment.action_positions) for segment in segments)
            if accelerator.is_main_process:
                print(
                    f"[train] trajectory group={group_id} seed={row.get('seed')} "
                    f"segments={len(segments)} lengths={lengths} actions={action_tokens}",
                    flush=True,
                )
            if any(len(segment.input_ids) > config.grpo.max_seq_length for segment in segments):
                oversized.append({"seed": row.get("seed"), "lengths": lengths})
            prepared.append((row, segments))
        # Advantages were normalized over the complete sampled group. Training
        # on a subset silently changes their meaning and destroys the zero-mean
        # group baseline, so reject the entire group if one rollout is too long.
        if oversized:
            metrics.append(
                {
                    "group_id": group_id,
                    "skipped": "trajectory_exceeds_max_seq_length",
                    "max_seq_length": config.grpo.max_seq_length,
                    "oversized": oversized,
                }
            )
            continue
        if len(prepared) < 2:
            metrics.append({"group_id": group_id, "skipped": "fewer_than_two_usable_rollouts"})
            continue
        prepared_groups.append((group_id, prepared))
        # max_steps counts usable optimizer groups, not arbitrary groups seen
        # before tokenization/length validation.
        if (
            config.grpo.max_steps is not None
            and len(prepared_groups) >= config.grpo.max_steps
        ):
            break

    if not prepared_groups:
        raise ValueError("No group contains at least two trajectories within max_seq_length")

    monitor.set_phase("load_and_dispatch_model")
    model, tokenizer = _load_model_and_tokenizer(
        config,
        model_path,
        resume_adapter,
        tokenizer=tokenizer,
        monitor=monitor,
        fsdp=config.distributed.strategy == "fsdp",
    )
    nonzero_dropouts = [
        module for module in model.modules() if isinstance(module, torch.nn.Dropout) and module.p > 0
    ]
    if nonzero_dropouts:
        raise RuntimeError(
            "The loaded policy contains non-zero dropout, so cached old/new policy log-probabilities "
            "would not be deterministic. Train/export an adapter with dropout=0."
        )
    policy = model
    model = _build_selective_logprob_model(
        policy,
        chunk_size=config.grpo.logprob_chunk_size,
    )
    replicated_modules: list[Any] = []
    if fsdp_plugin is not None:
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(policy)
        replicated_modules = _prepare_replicated_projection_modules(
            policy,
            accelerator,
            fsdp_plugin,
        )

    model.train()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("The loaded policy has no trainable LoRA parameters")
    monitor.set_phase("optimizer_init")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.grpo.learning_rate)
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    monitor.set_phase("fsdp_wrap")
    model, optimizer = accelerator.prepare(model, optimizer)
    if replicated_modules:
        _broadcast_replicated_modules(replicated_modules)
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"[train] trainable parameters: {trainable_count:,}; "
            f"world_size={world_size}; device={accelerator.device}",
            flush=True,
        )

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
    dummy_token_id = tokenizer.eos_token_id
    if dummy_token_id is None:
        dummy_token_id = tokenizer.pad_token_id
    if dummy_token_id is None:
        dummy_token_id = 0
    model.eval()
    with torch.no_grad():
        for group_id, prepared in prepared_groups:
            if len(prepared) % world_size:
                raise ValueError(
                    f"Group {group_id} has {len(prepared)} trajectories, which is not divisible "
                    f"by FSDP world size {world_size}"
                )
            local_prepared = prepared[rank::world_size]
            cached_rollouts = []
            for row, segments in local_prepared:
                seed = row.get("seed")
                monitor.set_phase(f"cache_old:{group_id}:seed_{seed}")
                old, _ = _distributed_rollout_logprobs(
                    model,
                    segments,
                    chunk_size=config.grpo.logprob_chunk_size,
                    input_device=accelerator.device,
                    world_size=world_size,
                    dummy_token_id=int(dummy_token_id),
                )
                old = old.detach().cpu()
                monitor.set_phase(f"cache_reference:{group_id}:seed_{seed}")
                unwrapped = accelerator.unwrap_model(model)
                with _disable_adapter(unwrapped):
                    reference, _ = _distributed_rollout_logprobs(
                        model,
                        segments,
                        chunk_size=config.grpo.logprob_chunk_size,
                        input_device=accelerator.device,
                        world_size=world_size,
                        dummy_token_id=int(dummy_token_id),
                    )
                    reference = reference.detach().cpu()
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
                seed = row.get("seed")
                monitor.set_phase(f"forward_backward:{group_id}:seed_{seed}")
                new, padding_anchor = _distributed_rollout_logprobs(
                    model,
                    segments,
                    chunk_size=config.grpo.logprob_chunk_size,
                    input_device=accelerator.device,
                    world_size=world_size,
                    dummy_token_id=int(dummy_token_id),
                )
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
                loss = loss + padding_anchor
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(kl)):
                    raise FloatingPointError(
                        f"Non-finite GRPO objective for group={group_id} seed={seed}: "
                        f"loss={float(loss.detach().cpu())}, kl={float(kl.detach().cpu())}"
                    )
                accelerator.backward(loss / len(cached_rollouts))
                total_loss += float(loss.detach().cpu())
                total_kl += float(kl.detach().cpu())
            monitor.set_phase(f"optimizer:{group_id}:epoch_{epoch}")
            grad_norm = accelerator.clip_grad_norm_(
                model.parameters(), config.grpo.max_grad_norm
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"Non-finite gradient norm for group={group_id} epoch={epoch}: "
                    f"{float(grad_norm.detach().cpu())}"
                )
            optimizer.step()
            optimizer_step += 1
            totals = torch.tensor(
                [total_loss, total_kl, float(len(cached_rollouts))],
                dtype=torch.float64,
                device=accelerator.device,
            )
            totals = accelerator.reduce(totals, reduction="sum")
            if accelerator.is_main_process:
                global_rollouts = int(totals[2].item())
                metrics.append(
                    {
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "group_id": group_id,
                        "rollouts": global_rollouts,
                        "loss": float(totals[0].item()) / global_rollouts,
                        "kl": float(totals[1].item()) / global_rollouts,
                        "grad_norm": float(
                            grad_norm.detach().cpu()
                            if hasattr(grad_norm, "detach")
                            else grad_norm
                        ),
                    }
                )

    if optimizer_step == 0:
        raise ValueError("No optimization step was executed; inspect sequence lengths and trace messages")
    monitor.set_phase("gather_adapter")
    accelerator.wait_for_everyone()
    if fsdp_plugin is not None:
        fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    gathered_state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        monitor.set_phase("save_adapter")
        output_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        output_policy = _wrapped_policy(unwrapped)
        output_policy.save_pretrained(
            output_dir,
            state_dict=_policy_state_dict(gathered_state),
            safe_serialization=True,
        )
    summary = {
        "groups_available": len(groups),
        "groups_processed": len(cached_groups),
        "optimizer_steps": optimizer_step,
        "trainable_parameters": trainable_count,
        "distributed_strategy": config.distributed.strategy,
        "world_size": world_size,
        "logprob_chunk_size": config.grpo.logprob_chunk_size,
        "telemetry_path": str(telemetry_path),
        "token_sources": dict(Counter(str(row.get("token_source", "unknown")) for row in source_rows)),
        "metrics": metrics,
    }
    if accelerator.is_main_process:
        (output_dir / "training_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return summary

