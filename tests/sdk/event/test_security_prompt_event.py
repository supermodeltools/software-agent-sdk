"""Tests for SecurityPromptEvent."""

from openhands.sdk.event import SecurityPromptEvent
from openhands.sdk.llm import TextContent


def test_security_prompt_event_creation():
    """Test SecurityPromptEvent creation and basic functionality."""
    security_event = SecurityPromptEvent(
        source="agent",
        security_prompt=TextContent(text="This is a security prompt for testing."),
    )

    assert security_event.source == "agent"
    assert (
        security_event.security_prompt.text == "This is a security prompt for testing."
    )
    assert security_event.kind == "SecurityPromptEvent"


def test_security_prompt_event_to_llm_message():
    """Test SecurityPromptEvent to_llm_message conversion."""
    security_event = SecurityPromptEvent(
        source="agent",
        security_prompt=TextContent(text="Security analyzer instructions."),
    )

    message = security_event.to_llm_message()

    assert message.role == "system"
    assert len(message.content) == 1
    content_item = message.content[0]
    assert isinstance(content_item, TextContent)
    assert content_item.text == "Security analyzer instructions."


def test_security_prompt_event_visualize():
    """Test SecurityPromptEvent visualize method."""
    security_event = SecurityPromptEvent(
        source="agent", security_prompt=TextContent(text="Security prompt content.")
    )

    visualization = security_event.visualize

    assert "Security Prompt:" in visualization
    assert "Security prompt content." in visualization


def test_security_prompt_event_str():
    """Test SecurityPromptEvent string representation."""
    security_event = SecurityPromptEvent(
        source="agent", security_prompt=TextContent(text="Security prompt content.")
    )

    str_repr = str(security_event)

    assert "SecurityPromptEvent (agent)" in str_repr
    assert "Security: Security prompt content." in str_repr
