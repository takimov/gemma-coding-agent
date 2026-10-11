from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


VALID_ROLES = {"system", "user", "assistant", "tool"}
TOOL_BUDGET_MARKERS = (
    "budgetexceeded",
    "tool call budget exhausted",
    "tool-call budget exhausted",
)

_ERROR_TEXT_KEYS = {
    "error",
    "error_message",
    "error_type",
    "exception",
    "failure",
    "status",
}
_TOOL_ROLES = {"tool", "environment", "observation", "tool_result"}


def has_tool_budget_exhaustion(value: Any) -> bool:
    """Detect an explicit SWEGemma tool-budget rejection.

    Trace artifacts also contain system/user instructions. Searching every
    string recursively makes a prompt that merely mentions a budget error look
    exhausted. Only diagnostic fields and tool observations are evidence; a
    top-level string remains supported for captured process output.
    """

    def contains_marker(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        normalized = text.casefold()
        return any(marker in normalized for marker in TOOL_BUDGET_MARKERS)

    def inspect(current: Any) -> bool:
        if isinstance(current, Mapping):
            role = str(
                current.get("role")
                or current.get("source")
                or current.get("type")
                or ""
            ).casefold()
            for key, child in current.items():
                normalized_key = str(key).casefold()
                if normalized_key in _ERROR_TEXT_KEYS and contains_marker(child):
                    return True
                if role in _TOOL_ROLES and normalized_key in {
                    "content",
                    "message",
                    "text",
                    "output",
                    "result",
                    "observation",
                }:
                    if contains_marker(child):
                        return True
                if isinstance(child, Mapping | list | tuple) and inspect(child):
                    return True
            return False
        if isinstance(current, list | tuple):
            return any(inspect(child) for child in current)
        return False

    if isinstance(value, str):
        return contains_marker(value)
    return inspect(value)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, Mapping) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(_as_text(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        for key in ("content", "text", "message", "output", "result", "observation"):
            if key in value:
                text = _as_text(value[key])
                if text:
                    return text
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _normalize_tool_call(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"type": "function", "function": {"name": "unknown", "arguments": _as_text(value)}}
    function = value.get("function") if isinstance(value.get("function"), Mapping) else {}
    name = function.get("name") or value.get("function_name") or value.get("name") or "unknown"
    arguments = function.get("arguments", value.get("arguments", value.get("args", {})))
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }
    call_id = value.get("id") or value.get("tool_call_id")
    if call_id is not None:
        normalized["id"] = str(call_id)
    return normalized


def _normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    values = value if isinstance(value, list) else [value]
    return [_normalize_tool_call(item) for item in values]


def _candidate_messages(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    messages: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        role = str(item.get("role", "")).lower()
        if role not in VALID_ROLES:
            return None
        message = {"role": role, "content": _as_text(item.get("content", item.get("message", "")))}
        if item.get("tool_calls"):
            message["tool_calls"] = _normalize_tool_calls(item["tool_calls"])
        if item.get("tool_call_id"):
            message["tool_call_id"] = item["tool_call_id"]
        if item.get("name"):
            message["name"] = item["name"]
        messages.append(message)
    return messages


def _find_messages(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, Mapping):
        for key in ("messages", "conversation", "history"):
            candidate = _candidate_messages(value.get(key))
            if candidate:
                return candidate
        for child in value.values():
            found = _find_messages(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_messages(child)
            if found:
                return found
    return None


def _find_steps(value: Any) -> list[Mapping[str, Any]] | None:
    if isinstance(value, Mapping):
        steps = value.get("steps")
        if isinstance(steps, list) and all(isinstance(step, Mapping) for step in steps):
            return steps
        for child in value.values():
            found = _find_steps(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_steps(child)
            if found is not None:
                return found
    return None


def _step_role(step: Mapping[str, Any]) -> str:
    raw = str(step.get("role") or step.get("source") or step.get("type") or "").lower()
    if raw in {"agent", "model", "assistant", "ai"}:
        return "assistant"
    if raw in {"tool", "environment", "observation", "tool_result"}:
        return "tool"
    if raw in {"human", "user", "task"}:
        return "user"
    if raw == "system":
        return "system"
    if step.get("tool_call_id") or step.get("observation") is not None:
        return "tool"
    return "assistant"


def _messages_from_steps(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for step in steps:
        role = _step_role(step)
        payload = step.get("message", step.get("content", step.get("text")))
        if payload is None and role == "tool":
            payload = step.get("observation", step.get("result", step.get("output")))
        content = _as_text(payload)
        thought = _as_text(
            step.get("reasoning_content", step.get("thought", step.get("reasoning", "")))
        )
        if thought and role == "assistant" and thought not in content:
            content = f"{thought}\n{content}".strip()
        message: dict[str, Any] = {"role": role, "content": content}
        tool_calls = step.get("tool_calls") or step.get("tool_call")
        if tool_calls:
            message["tool_calls"] = _normalize_tool_calls(tool_calls)
        if step.get("tool_call_id"):
            message["tool_call_id"] = step["tool_call_id"]
        if step.get("name") or step.get("tool_name"):
            message["name"] = step.get("name") or step.get("tool_name")
        if content or tool_calls:
            messages.append(message)
        observation = step.get("observation")
        if role == "assistant" and isinstance(observation, Mapping):
            results = observation.get("results", [])
            if isinstance(results, list):
                call_names: dict[str, str] = {}
                raw_calls = tool_calls if isinstance(tool_calls, list) else [tool_calls] if tool_calls else []
                for call in raw_calls:
                    if not isinstance(call, Mapping):
                        continue
                    normalized_call = _normalize_tool_call(call)
                    call_id = normalized_call.get("id")
                    if call_id:
                        call_names[str(call_id)] = str(normalized_call["function"]["name"])
                for result in results:
                    if not isinstance(result, Mapping):
                        continue
                    call_id = result.get("source_call_id")
                    tool_message: dict[str, Any] = {
                        "role": "tool",
                        "content": _as_text(result.get("content", result)),
                    }
                    if call_id is not None:
                        tool_message["tool_call_id"] = str(call_id)
                        if call_names.get(str(call_id)):
                            tool_message["name"] = call_names[str(call_id)]
                    messages.append(tool_message)
    return messages


def normalize_trace(trace: Mapping[str, Any], *, fallback_user_prompt: str = "") -> list[dict[str, Any]]:
    messages = _find_messages(trace)
    if messages is None:
        steps = _find_steps(trace) or []
        messages = _messages_from_steps(steps)
    if fallback_user_prompt and not any(message["role"] == "user" for message in messages):
        messages.insert(0, {"role": "user", "content": fallback_user_prompt})
    return messages


def trace_tool_definitions(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    agent = trace.get("agent")
    if isinstance(agent, Mapping) and isinstance(agent.get("tool_definitions"), list):
        return [dict(tool) for tool in agent["tool_definitions"] if isinstance(tool, Mapping)]
    return []


def trace_token_segments(trace: Mapping[str, Any]) -> list[dict[str, list[int]]]:
    """Extract exact tokens only when every assistant LLM call recorded them.

    Returning a partial list would silently optimize only some turns while the
    rollout reward belongs to the complete trajectory. Incomplete ATIF metrics
    therefore fall back to whole-trace chat-template reconstruction.
    """
    assistant_steps = [
        step for step in (_find_steps(trace) or []) if _step_role(step) == "assistant"
    ]
    if not assistant_steps:
        return []
    segments: list[dict[str, list[int]]] = []
    for step in assistant_steps:
        metrics = step.get("metrics")
        if not isinstance(metrics, Mapping):
            return []
        prompt = metrics.get("prompt_token_ids")
        completion = metrics.get("completion_token_ids")
        if not (
            isinstance(prompt, list)
            and isinstance(completion, list)
            and prompt
            and completion
            and all(isinstance(token, int) for token in prompt + completion)
        ):
            return []
        input_ids = list(prompt) + list(completion)
        segments.append(
            {
                "input_ids": input_ids,
                "action_positions": list(range(len(prompt), len(input_ids))),
            }
        )
    return segments


def count_tool_calls(trace: Mapping[str, Any]) -> int:
    count = 0

    def visit(value: Any) -> None:
        nonlocal count
        if isinstance(value, Mapping):
            calls = value.get("tool_calls")
            if isinstance(calls, list):
                count += len(calls)
            elif calls:
                count += 1
            elif value.get("tool_call"):
                count += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(trace)
    return count


def load_trace(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Trace at {path} is not a JSON object")
    return value

