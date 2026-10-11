from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ID_KEYS = ("instance_id", "task_id", "id")
PROMPT_KEYS = ("problem_statement", "prompt", "issue", "description")


def task_id(task: Mapping[str, Any]) -> str:
    for key in ID_KEYS:
        value = task.get(key)
        if value is not None and str(value):
            return str(value)
    raise ValueError(f"Task has no identifier in {ID_KEYS}: {task}")


def task_prompt(task: Mapping[str, Any]) -> str:
    for key in PROMPT_KEYS:
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def select_tasks(
    tasks: Iterable[Mapping[str, Any]], selected_ids: set[str] | None
) -> list[dict[str, Any]]:
    rows = [dict(task) for task in tasks]
    if not selected_ids:
        return rows
    selected = [task for task in rows if task_id(task) in selected_ids]
    found = {task_id(task) for task in selected}
    missing = selected_ids - found
    if missing:
        raise ValueError(f"Unknown task ids: {', '.join(sorted(missing))}")
    return selected


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")

