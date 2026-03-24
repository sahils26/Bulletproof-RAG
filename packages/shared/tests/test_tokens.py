"""Tests for the tiktoken-based token counter."""

from shared.utils.tokens import count_tokens


def test_empty_string():
    assert count_tokens("") == 0


def test_short_text():
    tokens = count_tokens("Hello, world!")
    assert tokens > 0
    assert tokens < 10  # should be ~4 tokens


def test_longer_text():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    tokens = count_tokens(text)
    assert tokens > 50
