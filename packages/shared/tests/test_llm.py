"""Unit tests for the LLM Service."""

from unittest.mock import AsyncMock, patch

import litellm
import pytest
from shared.config import LLMConfig
from shared.llm.service import LLMResponse, LLMService


@pytest.fixture
def llm_config(monkeypatch):
    # Clear environment to ensure test isolation
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=1000,
    )


def test_llm_response_model():
    """Test the LLMResponse pydantic model."""
    res = LLMResponse(
        content="Hello world",
        input_tokens=10,
        output_tokens=5,
        model_name="mock-model",
    )
    assert res.content == "Hello world"
    assert res.input_tokens == 10
    assert res.output_tokens == 5
    assert res.model_name == "mock-model"


@pytest.mark.asyncio
async def test_llm_service_complete_success(llm_config):
    """Test successful completion."""
    service = LLMService(llm_config)

    # Mock litellm.acompletion
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "Mocked answer"
    mock_response.get = dict(
        usage={"prompt_tokens": 15, "completion_tokens": 20},
        model="gpt-4o-mini",
    ).get

    with patch(
        "shared.llm.service.litellm.acompletion", return_value=mock_response
    ) as mock_acomplete:
        messages = [{"role": "user", "content": "Hi?"}]
        response = await service.complete(messages=messages)

        assert response.content == "Mocked answer"
        assert response.input_tokens == 15
        assert response.output_tokens == 20
        assert response.model_name == "gpt-4o-mini"

        # Verify it was called with right config
        mock_acomplete.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=1000,
            messages=messages,
        )


@pytest.mark.asyncio
async def test_llm_service_retry_on_rate_limit(llm_config):
    """Test that the service retries on RateLimitError."""
    service = LLMService(llm_config)

    class FakeRateLimitError(litellm.RateLimitError):
        def __init__(self):
            super().__init__(
                message="Rate limited",
                response=None,
                llm_provider="openai",
                model="gpt-4o-mini",
            )

    mock_success = AsyncMock()
    mock_success.choices = [AsyncMock()]
    mock_success.choices[0].message.content = "Success after retry"
    mock_success.get = dict(
        usage={"prompt_tokens": 5, "completion_tokens": 5}, model="mock"
    ).get

    # We fail once, then succeed
    with (
        patch(
            "shared.llm.service.litellm.acompletion",
            side_effect=[FakeRateLimitError(), mock_success],
        ) as mock_acomplete,
        patch("shared.llm.service.wait_exponential.__call__", return_value=0.01),
    ):
        response = await service.complete(messages=[{"role": "user", "content": "Hi"}])

        assert response.content == "Success after retry"
        assert mock_acomplete.call_count == 2
