"""Tests for ConversationErrorEvent serialization and deserialization."""

import json

from openhands.sdk.event import Event
from openhands.sdk.event.conversation_error import ConversationErrorEvent


def test_conversation_error_event_deserialization_from_json():
    """Test that ConversationErrorEvent can be deserialized from JSON.

    This reproduces the issue where ConversationErrorEvent was defined but not
    imported in the event module, causing deserialization to fail with
    "Unknown event type: ConversationErrorEvent".
    """
    # Example ConversationErrorEvent JSON from the user's data
    event_json = json.dumps(
        {
            "kind": "ConversationErrorEvent",
            "id": "adac9d93-e3e0-4f4f-8493-4d3b592f994f",
            "timestamp": "2025-12-15T20:30:45.563120",
            "source": "environment",
            "code": "LLMBadRequestError",
            "detail": (
                "litellm.BadRequestError: Error code: 400 - "
                "{'error': {'message': 'Test error message'}}"
            ),
        }
    )

    # This should deserialize successfully
    event = Event.model_validate_json(event_json)

    # Verify the event type and fields
    assert isinstance(event, ConversationErrorEvent)
    assert event.__class__.__name__ == "ConversationErrorEvent"
    assert event.id == "adac9d93-e3e0-4f4f-8493-4d3b592f994f"
    assert event.source == "environment"
    assert event.code == "LLMBadRequestError"
    assert "Test error message" in event.detail


def test_conversation_error_event_serialization():
    """Test that ConversationErrorEvent can be created and serialized."""
    # Create a ConversationErrorEvent
    event = ConversationErrorEvent(
        source="environment",
        code="TestError",
        detail="This is a test error",
    )

    # Serialize to JSON
    event_json = event.model_dump_json()
    event_dict = json.loads(event_json)

    # Verify serialization
    assert event_dict["kind"] == "ConversationErrorEvent"
    assert event_dict["source"] == "environment"
    assert event_dict["code"] == "TestError"
    assert event_dict["detail"] == "This is a test error"

    # Verify round-trip
    deserialized = Event.model_validate_json(event_json)
    assert isinstance(deserialized, ConversationErrorEvent)
    assert deserialized.__class__.__name__ == "ConversationErrorEvent"
    assert deserialized.code == "TestError"
    assert deserialized.detail == "This is a test error"


def test_conversation_error_event_exported_from_event_module():
    """Test that ConversationErrorEvent is properly exported from the event module.

    This ensures that ConversationErrorEvent is available when importing from
    openhands.sdk.event, not just from openhands.sdk.event.conversation_error.
    """
    from openhands.sdk import event

    # Should be able to access ConversationErrorEvent from the event module
    assert hasattr(event, "ConversationErrorEvent")
    assert event.ConversationErrorEvent is ConversationErrorEvent
