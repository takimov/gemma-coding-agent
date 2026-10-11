from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import sqrt
from typing import Any


def add_group_advantages(
    rows: Iterable[Mapping[str, Any]], *, eps: float = 1e-8
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if not row.get("scorable", True) or row.get("reward") is None:
            row["advantage"] = None
            row["trainable"] = False
            output.append(row)
            continue
        group_id = str(row.get("group_id", ""))
        if not group_id:
            raise ValueError("Every scorable rollout must have a non-empty group_id")
        grouped[group_id].append(row)

    for group_rows in grouped.values():
        rewards = [float(row["reward"]) for row in group_rows]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = sqrt(variance)
        trainable = std > eps and len(group_rows) >= 2
        for row, reward in zip(group_rows, rewards, strict=True):
            row["advantage"] = (reward - mean) / max(std, eps) if trainable else 0.0
            row["trainable"] = trainable
            output.append(row)
    return output


def trainable_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        if row.get("trainable") and row.get("advantage") is not None:
            grouped[str(row["group_id"])].append(row)
    return dict(grouped)

