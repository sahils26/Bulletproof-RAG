"""Token counting utility using tiktoken for accurate LLM budget tracking."""

import tiktoken


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count the number of tokens in ``text`` using tiktoken.

    Args:
        text: The string to tokenize.
        model: The tiktoken encoding name (default ``cl100k_base``,
               used by GPT-4 / Claude-compatible tokenizers).

    Returns:
        Number of tokens. Returns 0 for empty strings.
    """
    if not text:
        return 0
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))
