from gemma_grpo.traces import (
    count_tool_calls,
    normalize_trace,
    trace_token_segments,
    trace_tool_definitions,
)


def test_native_messages_are_preserved():
    trace = {
        "trajectory": {
            "messages": [
                {"role": "user", "content": "fix"},
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": "{}"}}],
                },
                {"role": "tool", "content": "contents", "tool_call_id": "1"},
            ]
        }
    }
    messages = normalize_trace(trace)
    assert [message["role"] for message in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert count_tool_calls(trace) == 1


def test_atif_steps_fallback_and_prompt_injection():
    trace = {
        "agent": {"tool_definitions": [{"name": "search", "description": "find"}]},
        "steps": [
            {
                "source": "agent",
                "reasoning_content": "inspect",
                "tool_calls": [
                    {"tool_call_id": "x", "function_name": "search", "arguments": {"query": "a"}}
                ],
                "observation": {"results": [{"source_call_id": "x", "content": "match"}]},
                "metrics": {
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3, 4],
                },
            },
        ]
    }
    messages = normalize_trace(trace, fallback_user_prompt="issue text")
    assert messages[0] == {"role": "user", "content": "issue text"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "search"
    assert messages[2] == {
        "role": "tool",
        "content": "match",
        "tool_call_id": "x",
        "name": "search",
    }
    assert trace_tool_definitions(trace) == [{"name": "search", "description": "find"}]
    assert trace_token_segments(trace) == [
        {"input_ids": [1, 2, 3, 4], "action_positions": [2, 3]}
    ]
    assert count_tool_calls(trace) == 1

