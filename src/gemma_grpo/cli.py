from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect import collect_rollouts
from .config import load_config
from .export import export_submission
from .loop import run_training_loop
from .monitoring import summarize_telemetry
from .nf4 import convert_to_nf4_checkpoint, prepare_vllm_nf4_checkpoint
from .preflight import run_preflight
from .train import train_adapter
from .vllm_server import rollout_models_yaml


def _paths(values: list[str]) -> list[Path]:
    return [Path(value).resolve() for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemma-grpo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate harness, files, packages, and tokenizer")
    preflight.add_argument("--config", default="configs/grpo.yaml")
    preflight.add_argument("--tasks")
    preflight.add_argument("--snapshots-dir")
    preflight.add_argument("--submission-dir", default="submission")
    preflight.add_argument("--model-path")
    preflight.add_argument("--rollout-model-path")
    preflight.add_argument("--skip-tokenizer", action="store_true")

    collect = subparsers.add_parser("collect", help="collect canonical SWEGemma rollouts")
    collect.add_argument("--config", default="configs/grpo.yaml")
    collect.add_argument("--tasks", required=True)
    collect.add_argument("--snapshots-dir", required=True)
    collect.add_argument("--submission-dir", default="submission")
    collect.add_argument("--results-root", default="results")
    collect.add_argument("--output", required=True)
    collect.add_argument("--iteration", type=int, required=True)
    collect.add_argument("--task-id", action="append", dest="task_ids")
    collect.add_argument("--adapter")
    collect.add_argument("--models-yaml")
    collect.add_argument("--model-path")

    convert_nf4 = subparsers.add_parser(
        "convert-nf4",
        help="convert the competition W4 checkpoint once into reusable FSDP NF4",
    )
    convert_nf4.add_argument("--source-model", required=True)
    convert_nf4.add_argument("--output-dir", required=True)
    convert_nf4.add_argument("--trust-remote-code", action="store_true")
    convert_nf4.add_argument("--max-shard-size", default="4GB")

    prepare_vllm_nf4 = subparsers.add_parser(
        "prepare-vllm-nf4",
        help="losslessly bitcast the FSDP NF4 checkpoint for vLLM 0.19",
    )
    prepare_vllm_nf4.add_argument("--source-model", required=True)
    prepare_vllm_nf4.add_argument("--output-dir", required=True)

    train = subparsers.add_parser("train", help="perform one on-policy artifact GRPO update")
    train.add_argument("--config", default="configs/grpo.yaml")
    train.add_argument("--rollouts", nargs="+", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--model-path")
    train.add_argument("--resume-adapter")

    telemetry = subparsers.add_parser(
        "telemetry", help="summarize phase-level CPU/RAM/GPU training telemetry"
    )
    telemetry.add_argument("--input", required=True)

    export = subparsers.add_parser("export", help="package the trained adapter as a Kaggle submission")
    export.add_argument("--submission-dir", default="submission")
    export.add_argument("--adapter", required=True)
    export.add_argument("--output", required=True)

    loop = subparsers.add_parser("loop", help="alternate fresh rollouts and GRPO until a step target")
    loop.add_argument("--config", default="configs/grpo.yaml")
    loop.add_argument("--tasks", required=True)
    loop.add_argument("--snapshots-dir", required=True)
    loop.add_argument("--submission-dir", default="submission")
    loop.add_argument("--work-dir", required=True)
    loop.add_argument("--model-path")
    loop.add_argument("--rollout-model-path")
    loop.add_argument("--task-id", action="append", dest="task_ids")
    loop.add_argument("--initial-adapter")
    loop.add_argument("--models-yaml")
    loop.add_argument("--target-steps", type=int, default=100)
    loop.add_argument("--max-collections", type=int, default=100)
    loop.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        report = run_preflight(
            config=load_config(args.config),
            tasks_path=Path(args.tasks).resolve() if args.tasks else None,
            snapshots_dir=Path(args.snapshots_dir).resolve() if args.snapshots_dir else None,
            submission_dir=Path(args.submission_dir).resolve(),
            model_path=args.model_path,
            rollout_model_path=args.rollout_model_path,
            check_tokenizer=not args.skip_tokenizer,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "collect":
        config = load_config(args.config)
        results_root = Path(args.results_root).resolve()
        adapter = Path(args.adapter).resolve() if args.adapter else None
        model_path = args.model_path or config.model.name_or_path
        external_models_yaml = Path(args.models_yaml).resolve() if args.models_yaml else None
        with rollout_models_yaml(
            config=config,
            model_path=model_path,
            adapter=adapter,
            work_dir=results_root / "_vllm" / f"iter_{args.iteration:04d}",
            external_models_yaml=external_models_yaml,
        ) as models_yaml:
            rows = collect_rollouts(
                config=config,
                tasks_path=Path(args.tasks).resolve(),
                snapshots_dir=Path(args.snapshots_dir).resolve(),
                submission_dir=Path(args.submission_dir).resolve(),
                results_root=results_root,
                output_path=Path(args.output).resolve(),
                iteration=args.iteration,
                selected_ids=set(args.task_ids) if args.task_ids else None,
                adapter=adapter,
                models_yaml=models_yaml,
            )
        summary = {
            "rollouts": len(rows),
            "scorable": sum(bool(row["scorable"]) for row in rows),
            "resolved": sum(row["resolved"] is True for row in rows),
            "trainable": sum(bool(row["trainable"]) for row in rows),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "convert-nf4":
        created = convert_to_nf4_checkpoint(
            source_model=Path(args.source_model),
            output_dir=Path(args.output_dir),
            trust_remote_code=args.trust_remote_code,
            max_shard_size=args.max_shard_size,
        )
        print(created)
        return 0
    if args.command == "prepare-vllm-nf4":
        created = prepare_vllm_nf4_checkpoint(
            source_model=Path(args.source_model),
            output_dir=Path(args.output_dir),
        )
        print(created)
        return 0
    if args.command == "train":
        summary = train_adapter(
            config=load_config(args.config),
            rollout_paths=_paths(args.rollouts),
            output_dir=Path(args.output_dir).resolve(),
            model_path=args.model_path,
            resume_adapter=Path(args.resume_adapter).resolve() if args.resume_adapter else None,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "export":
        created = export_submission(
            submission_dir=Path(args.submission_dir).resolve(),
            adapter_dir=Path(args.adapter).resolve(),
            output_zip=Path(args.output).resolve(),
        )
        print(created)
        return 0
    if args.command == "telemetry":
        summary = summarize_telemetry(Path(args.input).resolve())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "loop":
        state = run_training_loop(
            config=load_config(args.config),
            tasks_path=Path(args.tasks).resolve(),
            snapshots_dir=Path(args.snapshots_dir).resolve(),
            submission_dir=Path(args.submission_dir).resolve(),
            work_dir=Path(args.work_dir).resolve(),
            model_path=args.model_path,
            rollout_model_path=args.rollout_model_path,
            selected_ids=set(args.task_ids) if args.task_ids else None,
            initial_adapter=(
                Path(args.initial_adapter).resolve() if args.initial_adapter else None
            ),
            models_yaml=Path(args.models_yaml).resolve() if args.models_yaml else None,
            target_steps=args.target_steps,
            max_collections=args.max_collections,
            resume=args.resume,
        )
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state["status"] == "complete" else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

