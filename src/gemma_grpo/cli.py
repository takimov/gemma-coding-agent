from __future__ import annotations

import argparse
import json
from pathlib import Path

from .collect import collect_rollouts
from .config import load_config
from .export import export_submission
from .loop import run_training_loop
from .monitoring import summarize_telemetry
from .preflight import run_preflight
from .train import train_adapter


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
            check_tokenizer=not args.skip_tokenizer,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    if args.command == "collect":
        rows = collect_rollouts(
            config=load_config(args.config),
            tasks_path=Path(args.tasks).resolve(),
            snapshots_dir=Path(args.snapshots_dir).resolve(),
            submission_dir=Path(args.submission_dir).resolve(),
            results_root=Path(args.results_root).resolve(),
            output_path=Path(args.output).resolve(),
            iteration=args.iteration,
            selected_ids=set(args.task_ids) if args.task_ids else None,
            adapter=Path(args.adapter).resolve() if args.adapter else None,
            models_yaml=Path(args.models_yaml).resolve() if args.models_yaml else None,
        )
        summary = {
            "rollouts": len(rows),
            "scorable": sum(bool(row["scorable"]) for row in rows),
            "resolved": sum(row["resolved"] is True for row in rows),
            "trainable": sum(bool(row["trainable"]) for row in rows),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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

