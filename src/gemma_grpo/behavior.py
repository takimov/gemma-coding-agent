from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .tokenization import TokenizedTrajectory, tokenize_trajectory


def trajectory_fingerprint(segments: list[TokenizedTrajectory]) -> str:
    payload = [
        {
            "input_ids": segment.input_ids,
            "action_positions": segment.action_positions,
        }
        for segment in segments
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return "tokens-sha256:" + hashlib.sha256(encoded).hexdigest()


def _row_segments(tokenizer: Any, row: dict[str, Any]) -> list[TokenizedTrajectory]:
    raw_segments = row.get("token_segments")
    if isinstance(raw_segments, list) and raw_segments:
        return [
            TokenizedTrajectory(
                input_ids=[int(token) for token in segment["input_ids"]],
                action_positions=[int(position) for position in segment["action_positions"]],
            )
            for segment in raw_segments
        ]
    return [tokenize_trajectory(tokenizer, row["messages"], row.get("tools"))]


def _selected_prompt_logprobs(
    prompt_logprobs: Any,
    trajectory: TokenizedTrajectory,
) -> list[float]:
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(
        trajectory.input_ids
    ):
        raise ValueError(
            "vLLM returned an unexpected prompt_logprobs shape: "
            f"expected {len(trajectory.input_ids)}, got "
            f"{len(prompt_logprobs) if isinstance(prompt_logprobs, list) else type(prompt_logprobs).__name__}"
        )
    selected: list[float] = []
    for position in trajectory.action_positions:
        entry = prompt_logprobs[position]
        if not isinstance(entry, dict):
            raise ValueError(f"vLLM returned no prompt log-probability at position {position}")
        token_id = trajectory.input_ids[position]
        value = entry.get(str(token_id), entry.get(token_id))
        if isinstance(value, dict):
            value = value.get("logprob")
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"vLLM prompt log-probability for token {token_id} at position "
                f"{position} is missing"
            )
        selected.append(float(value))
    return selected


def _score_segment(
    trajectory: TokenizedTrajectory,
    *,
    api_base: str,
    model: str,
    timeout_seconds: int,
) -> list[float]:
    payload = {
        "model": model,
        "prompt": trajectory.input_ids,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 0,
        "add_special_tokens": False,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not score W4 behavior log-probabilities: {error}") from error
    choices = result.get("choices") if isinstance(result, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("vLLM behavior scoring response must contain exactly one choice")
    return _selected_prompt_logprobs(choices[0].get("prompt_logprobs"), trajectory)


def attach_behavior_logprobs(
    rows: list[dict[str, Any]],
    *,
    tokenizer_path: str | Path,
    api_base: str,
    model: str,
    trust_remote_code: bool,
    timeout_seconds: int = 300,
) -> list[dict[str, Any]]:
    """Record W4 raw-model probabilities for trainable action tokens.

    The completion endpoint receives the already-tokenized trajectory with
    ``add_special_tokens=False``. ``prompt_logprobs=0`` asks vLLM for only the
    observed token at every position, avoiding a vocabulary-sized response.
    """
    trainable = [row for row in rows if row.get("trainable")]
    if not trainable:
        return rows

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    total = len(trainable)
    for index, row in enumerate(trainable, start=1):
        print(
            f"[rollout] scoring W4 behavior logprobs {index}/{total}; "
            f"seed={row.get('seed')}",
            flush=True,
        )
        segments = _row_segments(tokenizer, row)
        values: list[float] = []
        for segment in segments:
            values.extend(
                _score_segment(
                    segment,
                    api_base=api_base,
                    model=model,
                    timeout_seconds=timeout_seconds,
                )
            )
        row["behavior_logprobs"] = values
        row["behavior_logprob_source"] = "vllm_raw_prompt_logprobs"
        row["behavior_trajectory_id"] = trajectory_fingerprint(segments)
    return rows
