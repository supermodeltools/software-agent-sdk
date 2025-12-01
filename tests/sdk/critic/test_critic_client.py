"""Tests for CriticClient api_key handling and apply_chat_template."""

import json
from unittest.mock import mock_open, patch

import pytest
from pydantic import SecretStr, ValidationError

from openhands.sdk.critic.impl.api.client import CriticClient


def test_critic_client_with_str_api_key():
    """Test CriticClient accepts str api_key and converts to SecretStr."""
    client = CriticClient(api_key="test_api_key_123")

    assert isinstance(client.api_key, SecretStr)
    assert client.api_key.get_secret_value() == "test_api_key_123"


def test_critic_client_with_secret_str_api_key():
    """Test that CriticClient accepts a SecretStr api_key directly."""
    secret_key = SecretStr("secret_api_key_456")
    client = CriticClient(api_key=secret_key)

    assert isinstance(client.api_key, SecretStr)
    assert client.api_key.get_secret_value() == "secret_api_key_456"


def test_critic_client_empty_string_api_key():
    """Test that CriticClient rejects an empty string api_key."""
    with pytest.raises(ValidationError, match="api_key must be non-empty"):
        CriticClient(api_key="")


def test_critic_client_whitespace_only_api_key():
    """Test that CriticClient rejects a whitespace-only api_key."""
    with pytest.raises(ValidationError, match="api_key must be non-empty"):
        CriticClient(api_key="   \t\n  ")


def test_critic_client_empty_secret_str_api_key():
    """Test that CriticClient rejects an empty SecretStr api_key."""
    with pytest.raises(ValidationError, match="api_key must be non-empty"):
        CriticClient(api_key=SecretStr(""))


def test_critic_client_whitespace_secret_str_api_key():
    """Test that CriticClient rejects a whitespace-only SecretStr api_key."""
    with pytest.raises(ValidationError, match="api_key must be non-empty"):
        CriticClient(api_key=SecretStr("   \t\n  "))


def test_critic_client_api_key_not_exposed_in_repr():
    """Test that the api_key is not exposed in the string representation."""
    client = CriticClient(api_key="super_secret_key")

    client_repr = repr(client)
    client_str = str(client)

    # SecretStr should hide the actual key value in repr/str
    assert "super_secret_key" not in client_repr
    assert "super_secret_key" not in client_str


def test_critic_client_api_key_preserved_after_validation():
    """Test that the api_key value is correctly preserved after validation."""
    test_key = "my_test_key_789"
    client = CriticClient(api_key=test_key)

    # Verify the key is preserved correctly
    assert isinstance(client.api_key, SecretStr)
    assert client.api_key.get_secret_value() == test_key

    # Verify it works with SecretStr input too
    secret_key = SecretStr("another_key_101112")
    client2 = CriticClient(api_key=secret_key)
    assert isinstance(client2.api_key, SecretStr)
    assert client2.api_key.get_secret_value() == "another_key_101112"


@patch("openhands.sdk.critic.impl.api.client.hf_hub_download")
@patch("pathlib.Path.open", new_callable=mock_open)
def test_apply_chat_template_without_tools(mock_file, mock_hf_download):
    """Test apply_chat_template without tools using jinja2 implementation."""
    # Mock the tokenizer config with a simple chat template
    mock_config = {
        "chat_template": (
            "{% for message in messages %}"
            "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
            "{% endfor %}"
        )
    }
    mock_file.return_value.read.return_value = json.dumps(mock_config)
    mock_hf_download.return_value = "/fake/path/tokenizer_config.json"

    client = CriticClient(api_key="test_key")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = client.apply_chat_template(messages)

    assert "<|im_start|>user\nHello<|im_end|>" in result
    assert "<|im_start|>assistant\nHi there!<|im_end|>" in result
    mock_hf_download.assert_called_once()


@patch("openhands.sdk.critic.impl.api.client.hf_hub_download")
@patch("pathlib.Path.open", new_callable=mock_open)
def test_apply_chat_template_with_tools(mock_file, mock_hf_download):
    """Test apply_chat_template with tools using jinja2 implementation."""
    # Mock tokenizer config with a template that uses tools
    mock_config = {
        "chat_template": (
            "{% if tools %}TOOLS: {{ tools | length }} tool(s)\n{% endif %}"
            "{% for message in messages %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{% endfor %}"
        )
    }
    mock_file.return_value.read.return_value = json.dumps(mock_config)
    mock_hf_download.return_value = "/fake/path/tokenizer_config.json"

    client = CriticClient(api_key="test_key")
    messages = [{"role": "user", "content": "What's the weather?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    result = client.apply_chat_template(messages, tools)  # type: ignore[arg-type]

    assert "TOOLS: 1 tool(s)" in result
    assert "user: What's the weather?" in result


@patch("openhands.sdk.critic.impl.api.client.hf_hub_download")
@patch("pathlib.Path.open", new_callable=mock_open)
def test_load_chat_template_caching(mock_file, mock_hf_download):
    """Test that chat template is cached after first load."""
    mock_config = {
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}"
    }
    mock_file.return_value.read.return_value = json.dumps(mock_config)
    mock_hf_download.return_value = "/fake/path/tokenizer_config.json"

    client = CriticClient(api_key="test_key")
    messages = [{"role": "user", "content": "Test"}]

    client.apply_chat_template(messages)
    client.apply_chat_template(messages)

    mock_hf_download.assert_called_once()


@patch("openhands.sdk.critic.impl.api.client.hf_hub_download")
@patch("pathlib.Path.open", new_callable=mock_open)
def test_load_chat_template_missing_template(mock_file, mock_hf_download):
    """Test error handling when chat_template is missing from config."""
    mock_config = {"some_other_key": "value"}
    mock_file.return_value.read.return_value = json.dumps(mock_config)
    mock_hf_download.return_value = "/fake/path/tokenizer_config.json"

    client = CriticClient(api_key="test_key")
    messages = [{"role": "user", "content": "Test"}]

    with pytest.raises(ValueError, match="No chat_template found"):
        client.apply_chat_template(messages)
