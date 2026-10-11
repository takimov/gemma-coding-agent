from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one SWEGemma task with the official direct Evaluator API."
    )
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--snapshots-dir", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--models-yaml", required=True, type=Path)
    parser.add_argument("--sandbox", choices=("docker", "subprocess"), default="subprocess")
    parser.add_argument("--image", default="swebench-sandbox:latest")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--max-tool-calls", type=int, required=True)
    parser.add_argument("--max-time-minutes", type=float, required=True)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--wheels-dir", type=Path)
    parser.add_argument("--context-cache-min-tokens", type=int, required=True)
    parser.add_argument("--context-cache-ttl-seconds", type=int, required=True)
    parser.add_argument("--context-cache-intervals", type=int, required=True)
    parser.add_argument("--compaction-interval", type=int, required=True)
    parser.add_argument("--compaction-overlap-size", type=int, required=True)
    parser.add_argument("--compaction-token-threshold", type=int, required=True)
    parser.add_argument("--compaction-event-retention-size", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Keep the heavy Kaggle-only dependencies lazy so the local training and
    # unit-test environment does not need the competition wheelhouse.
    import litellm
    from adk_submission import discover_adapters
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import (
        ALLOWED_ADAPTER_EXTENSIONS,
        EvalConfig,
        build_submission_limits,
    )
    from swegemma.evaluate import Evaluator
    from swegemma.models import setup_gemma_model_registry

    litellm.drop_params = True
    limits, generation_constraints = build_submission_limits()
    adapters = discover_adapters(
        str(args.submission_dir),
        adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS,
    )
    models = setup_gemma_model_registry(models_yaml_path=args.models_yaml)

    config = EvalConfig(
        tasks_path=args.tasks,
        snapshots_dir=args.snapshots_dir,
        results_dir=args.results_dir,
        submission_dir=args.submission_dir,
        models=models,
        image=args.image,
        sandbox=args.sandbox,
        timeout_seconds=args.timeout_seconds,
        max_time_minutes=args.max_time_minutes,
        max_tool_calls=args.max_tool_calls,
        max_turns=args.max_turns,
        task_ids=[args.task_id],
        concurrency=1,
        display_mode="quiet",
        verbose=False,
        limits=limits,
        generation_constraints=generation_constraints,
        adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(
            min_tokens=args.context_cache_min_tokens,
            ttl_seconds=args.context_cache_ttl_seconds,
            cache_intervals=args.context_cache_intervals,
        ),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=args.compaction_interval,
            overlap_size=args.compaction_overlap_size,
            token_threshold=args.compaction_token_threshold,
            event_retention_size=args.compaction_event_retention_size,
        ),
        graph_dir=str(args.graph_dir) if args.graph_dir is not None else "data/graphs",
        embeddings_dir=(
            str(args.embeddings_dir)
            if args.embeddings_dir is not None
            else "data/embeddings"
        ),
        wheels_dir=args.wheels_dir,
    )
    result = asyncio.run(Evaluator(config).run())
    print(
        f"[direct-eval] task={args.task_id}; resolved={result.resolved}/"
        f"{result.total}; rate={result.resolution_rate:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
