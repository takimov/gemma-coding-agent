import pytest

from gemma_grpo.tokenization import tokenize_trajectory


class PrefixTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        tokens = [1]
        for message in messages:
            tokens.extend([len(message["role"]), len(message.get("content", "")), 2])
        return tokens


class RewritingTokenizer(PrefixTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        tokens = super().apply_chat_template(messages, **kwargs)
        if len(messages) > 1:
            tokens[0] = 9
        return tokens


class V5BatchEncodingTokenizer(PrefixTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [super().apply_chat_template(messages, **kwargs)]}


def test_assistant_span_is_marked():
    trajectory = tokenize_trajectory(
        PrefixTokenizer(),
        [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "yz"},
            {"role": "tool", "content": "ok"},
        ],
    )
    assert trajectory.action_positions == [4, 5, 6]
    assert len(trajectory.input_ids) == 10


def test_non_prefix_template_is_rejected():
    with pytest.raises(ValueError, match="not prefix-preserving"):
        tokenize_trajectory(
            RewritingTokenizer(),
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        )


def test_transformers_v5_batch_encoding_is_unwrapped():
    trajectory = tokenize_trajectory(
        V5BatchEncodingTokenizer(),
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    )
    assert trajectory.input_ids == [1, 4, 1, 2, 9, 1, 2]
    assert trajectory.action_positions == [4, 5, 6]

