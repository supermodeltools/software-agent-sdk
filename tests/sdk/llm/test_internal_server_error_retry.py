from unittest.mock import patch

import pytest
from litellm.exceptions import InternalServerError
from litellm.types.utils import Choices, Message as LiteLLMMessage, ModelResponse, Usage
from pydantic import SecretStr

from openhands.sdk.llm import LLM, LLMResponse, Message, TextContent
from openhands.sdk.llm.exceptions import LLMServiceUnavailableError


def create_mock_response(content: str = "Test response", response_id: str = "test-id"):
    """Helper function to create properly structured mock responses."""
    return ModelResponse(
        id=response_id,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=LiteLLMMessage(content=content, role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
        system_fingerprint="test",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.fixture
def default_config():
    return LLM(
        usage_id="test-llm",
        model="gpt-4o",
        api_key=SecretStr("test_key"),
        num_retries=2,
        retry_min_wait=1,
        retry_max_wait=2,
    )


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_completion_retries_internal_server_error(
    mock_litellm_completion, default_config
):
    """Test that InternalServerError is properly retried."""
    mock_response = create_mock_response("Retry successful after internal error")

    # Mock the litellm_completion to first raise an InternalServerError,
    # then return a successful response
    mock_litellm_completion.side_effect = [
        InternalServerError(
            message="Internal server error",
            llm_provider="test_provider",
            model="test_model",
        ),
        mock_response,
    ]

    # Create an LLM instance and call completion
    llm = LLM(
        model="gpt-4o",
        api_key=SecretStr("test_key"),
        num_retries=2,
        retry_min_wait=1,
        retry_max_wait=2,
        usage_id="test-service",
    )
    response = llm.completion(
        messages=[Message(role="user", content=[TextContent(text="Hello!")])],
    )

    # Verify that the retry was successful
    assert isinstance(response, LLMResponse)
    assert response.raw_response == mock_response
    assert mock_litellm_completion.call_count == 2  # Initial call + 1 retry


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_completion_max_retries_internal_server_error(
    mock_litellm_completion, default_config
):
    """Test that InternalServerError respects max retries and is mapped to SDK error."""
    # Mock the litellm_completion to raise InternalServerError multiple times
    mock_litellm_completion.side_effect = [
        InternalServerError(
            message="Internal server error 1",
            llm_provider="test_provider",
            model="test_model",
        ),
        InternalServerError(
            message="Internal server error 2",
            llm_provider="test_provider",
            model="test_model",
        ),
        InternalServerError(
            message="Internal server error 3",
            llm_provider="test_provider",
            model="test_model",
        ),
    ]

    # Create an LLM instance and call completion
    llm = LLM(
        model="gpt-4o",
        api_key=SecretStr("test_key"),
        num_retries=2,
        retry_min_wait=1,
        retry_max_wait=2,
        usage_id="test-service",
    )

    # The completion should raise an SDK typed error after exhausting all retries
    with pytest.raises(LLMServiceUnavailableError) as excinfo:
        llm.completion(
            messages=[Message(role="user", content=[TextContent(text="Hello!")])],
        )

    # Verify that the correct number of retries were attempted
    assert mock_litellm_completion.call_count == default_config.num_retries

    # The exception should contain internal server error information
    assert "Internal server error" in str(excinfo.value)

    # Ensure the original provider exception is preserved as the cause
    assert isinstance(excinfo.value.__cause__, InternalServerError)


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_llm_service_unavailable_error_is_retried(
    mock_litellm_completion, default_config
):
    """Test that LLMServiceUnavailableError itself is also retryable."""
    mock_response = create_mock_response("Success after service unavailable")

    # Mock the litellm_completion to first raise an InternalServerError (which gets
    # mapped to LLMServiceUnavailableError), then return a successful response
    mock_litellm_completion.side_effect = [
        InternalServerError(
            message="Service temporarily unavailable",
            llm_provider="test_provider",
            model="test_model",
        ),
        mock_response,
    ]

    # Create an LLM instance and call completion
    llm = LLM(
        model="gpt-4o",
        api_key=SecretStr("test_key"),
        num_retries=3,
        retry_min_wait=1,
        retry_max_wait=2,
        usage_id="test-service",
    )
    response = llm.completion(
        messages=[Message(role="user", content=[TextContent(text="Hello!")])],
    )

    # Verify that the retry was successful
    assert isinstance(response, LLMResponse)
    assert response.raw_response == mock_response
    assert mock_litellm_completion.call_count == 2  # Initial call + 1 retry
