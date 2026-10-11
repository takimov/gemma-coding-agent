import torch

from gemma_grpo.cli import build_parser
from gemma_grpo.sft import _approved_for_sft, _sft_loss, prepare_sft_examples


def _row(task, seed, tokens, *, resolved=True, approved=False):
    return {
        "task_id": task,
        "seed": seed,
        "resolved": resolved,
        "scorable": True,
        "sft_approved": approved,
        "token_source": "atif_token_ids",
        "token_segments": [
            {"input_ids": tokens, "action_positions": [len(tokens) - 1]}
        ],
    }


def test_sft_accepts_only_verified_or_explicitly_curated_rows():
    assert _approved_for_sft({"resolved": True, "scorable": True})
    assert not _approved_for_sft({"resolved": True, "scorable": False})
    assert not _approved_for_sft({"resolved": False, "scorable": True})
    assert _approved_for_sft({"resolved": False, "sft_approved": True})


def test_prepare_sft_examples_deduplicates_balances_and_rejects_long_rows():
    rows = [
        _row("a", 1, [1, 2]),
        _row("a", 2, [1, 3]),
        _row("a", 3, [1, 4]),
        _row("a", 4, [1, 5]),
        _row("b", 1, [1, 2]),
        _row("c", 1, [1, 2, 3, 4, 5]),
        _row("failed", 1, [8, 9], resolved=False),
        _row("curated", 1, [6, 7], resolved=False, approved=True),
    ]

    examples, skipped = prepare_sft_examples(
        tokenizer=None,
        rows=rows,
        max_seq_length=4,
        max_examples_per_task=2,
        seed=42,
    )

    tasks = [row["task_id"] for row, _ in examples]
    assert tasks.count("a") == 2
    assert "curated" in tasks
    assert "failed" not in tasks
    reasons = {row["reason"] for row in skipped}
    assert reasons == {
        "duplicate_trajectory",
        "task_cap",
        "trajectory_exceeds_max_seq_length",
    }


def test_sft_loss_is_global_token_mean_after_rank_average():
    logprobs = torch.tensor([-1.0, -3.0])
    loss = _sft_loss(
        logprobs,
        global_tokens=4,
        world_size=2,
        padding_anchor=torch.tensor(0.0),
    )
    assert loss.item() == 2.0


def test_cli_exposes_offline_sft_command():
    args = build_parser().parse_args(
        [
            "sft",
            "--trajectories",
            "one.jsonl",
            "two.jsonl",
            "--output-dir",
            "adapter",
        ]
    )
    assert args.command == "sft"
    assert args.trajectories == ["one.jsonl", "two.jsonl"]
