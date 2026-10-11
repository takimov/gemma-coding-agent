from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any


@dataclass(frozen=True)
class TokenizedTrajectory:
    input_ids: list[int]
    action_positions: list[int]


def _flat_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("Tokenizer chat template mapping has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, Integral) for token in value):
        raise TypeError("Tokenizer chat template did not return a flat token-id list")
    return [int(token) for token in value]


def render_prefix(
    tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        # Gemma 4's Transformers v5 tokenizer can return processor-specific
        # nested objects when asked for a naked list.  A one-row tensor inside
        # BatchEncoding has an unambiguous shape and _flat_ids normalizes it.
        "return_dict": True,
        "return_tensors": "pt",
    }
    if tools:
        kwargs["tools"] = tools
    return _flat_ids(
        tokenizer.apply_chat_template(messages, **kwargs)
    )


def tokenize_trajectory(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> TokenizedTrajectory:
    if not messages:
        raise ValueError("Trajectory has no messages")
    previous: list[int] = []
    action_positions: list[int] = []
    for index, message in enumerate(messages):
        current = render_prefix(tokenizer, messages[: index + 1], tools)
        if current[: len(previous)] != previous:
            raise ValueError(
                "The tokenizer chat template is not prefix-preserving. "
                "Online tool trajectories cannot be scored safely with incremental assistant masks."
            )
        if message.get("role") == "assistant":
            action_positions.extend(range(len(previous), len(current)))
        previous = current
    # Position zero has no preceding causal-logit position and cannot be scored.
    action_positions = [position for position in action_positions if position > 0]
    if not action_positions:
        raise ValueError("Trajectory contains no scoreable assistant tokens")
    return TokenizedTrajectory(previous, action_positions)


def assert_prefix_preserving(tokenizer: Any) -> None:
    sample = [
        {"role": "user", "content": "Return one word."},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Now return another."},
    ]
    tokenize_trajectory(tokenizer, sample)

