from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .config import RewardConfig


@dataclass(frozen=True)
class RewardOutcome:
    reward: float | None
    resolved: bool | None
    scorable: bool
    reason: str


def _walk(value: Any):
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def find_bool(result: Mapping[str, Any], names: set[str]) -> bool | None:
    for value in _walk(result):
        if not isinstance(value, Mapping):
            continue
        for key, child in value.items():
            if str(key).lower() in names and isinstance(child, bool):
                return child
            if str(key).lower() in names and child in (0, 1):
                return bool(child)
    return None


def has_infrastructure_error(result: Mapping[str, Any]) -> bool:
    explicit = find_bool(result, {"infrastructure_error", "infra_error"})
    if explicit is not None:
        return explicit
    status = str(result.get("status", "")).lower()
    if status in {"infra_error", "infrastructure_error", "sandbox_error"}:
        return True
    # SWEGemma reserves the top-level `error` field for evaluation-worker
    # failures. Ordinary agent/test failures are represented by resolved=false
    # and remain valid reward signals. Check this before inspecting any nested
    # resolved flag: a worker error may contain a partial default result.
    error = result.get("error")
    return error is not None and bool(str(error).strip())


def score_result(
    result: Mapping[str, Any],
    *,
    patch: str,
    tool_calls: int,
    config: RewardConfig,
) -> RewardOutcome:
    if has_infrastructure_error(result):
        return RewardOutcome(None, None, False, "infrastructure_error")

    resolved = find_bool(result, {"resolved", "tests_resolved"})
    if resolved is None and isinstance(result.get("success"), bool):
        # Some harness revisions use a top-level success flag. Do not search for
        # this name recursively: individual tool calls may also report success.
        resolved = bool(result["success"])
    if resolved is None:
        return RewardOutcome(None, None, False, "missing_verification_result")

    if resolved:
        reward = config.resolved - config.tool_call_penalty * max(tool_calls, 0)
        return RewardOutcome(reward, True, True, "resolved")
    if patch.strip() in {"", "NO_PATCH"}:
        return RewardOutcome(config.empty_patch, False, True, "empty_patch")
    return RewardOutcome(config.failed, False, True, "tests_failed")

