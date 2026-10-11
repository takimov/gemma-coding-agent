from gemma_grpo.evaluator_worker import build_parser


def test_direct_evaluator_worker_parses_context_controls():
    args = build_parser().parse_args(
        [
            "--tasks",
            "tasks.jsonl",
            "--snapshots-dir",
            "snapshots",
            "--submission-dir",
            "submission",
            "--results-dir",
            "results",
            "--task-id",
            "task_1",
            "--models-yaml",
            "models.yaml",
            "--max-tool-calls",
            "28",
            "--max-time-minutes",
            "8",
            "--context-cache-min-tokens",
            "2048",
            "--context-cache-ttl-seconds",
            "1800",
            "--context-cache-intervals",
            "10",
            "--compaction-interval",
            "15",
            "--compaction-overlap-size",
            "2",
            "--compaction-token-threshold",
            "14336",
            "--compaction-event-retention-size",
            "5",
        ]
    )

    assert args.task_id == "task_1"
    assert args.max_tool_calls == 28
    assert args.compaction_token_threshold == 14336
