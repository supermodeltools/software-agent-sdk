"""Tests for condenser handling of thinking blocks with Claude thinking mode."""

from openhands.sdk.context.view import View
from openhands.sdk.event import Event
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    MessageEvent,
    ObservationEvent,
    SystemPromptEvent,
)
from openhands.sdk.llm import Message, MessageToolCall, TextContent, ThinkingBlock
from openhands.sdk.tool.builtins.think import ThinkObservation


def test_condenser_preserves_thinking_blocks_in_conversation():
    """Test that condensation preserves at least one thinking block in the conversation.

    When Claude's extended thinking mode is enabled, the conversation must have
    thinking blocks in assistant messages. After condensation, if all events with
    thinking blocks are forgotten, the next LLM call will fail with:
    'messages.N.content.0.type: Expected `thinking` or `redacted_thinking`, but found
    `tool_use`. When `thinking` is enabled, a final `assistant` message must start with
    a thinking block.'

    This test reproduces the issue where:
    1. Early events have thinking blocks
    2. Condensation forgets those events
    3. Remaining events don't have thinking blocks
    4. The conversation violates Claude's extended thinking requirements
    """
    # Create a sequence of events similar to the user's data
    events: list[Event] = []

    # Event 0: System prompt
    events.append(
        SystemPromptEvent(
            id="system-1",
            system_prompt=TextContent(text="You are a helpful assistant."),
            tools=[],
        )
    )

    # Event 1: User message
    events.append(
        MessageEvent(
            id="user-1",
            llm_message=Message(
                role="user", content=[TextContent(text="Help me analyze some code.")]
            ),
            source="user",
        )
    )

    # Event 2: ActionEvent with thinking block (will be forgotten by condensation)
    events.append(
        ActionEvent(
            id="action-1",
            source="agent",
            thought=[],
            thinking_blocks=[
                ThinkingBlock(
                    type="thinking",
                    thinking="I need to analyze the code carefully.",
                    signature="sig1",
                )
            ],
            tool_name="terminal",
            tool_call_id="call-1",
            tool_call=MessageToolCall(
                id="call-1", name="terminal", arguments="{}", origin="completion"
            ),
            llm_response_id="llm-resp-1",
        )
    )
    # Event 3: Observation for action-1
    events.append(
        ObservationEvent(
            id="obs-1",
            source="environment",
            tool_name="terminal",
            observation=ThinkObservation(content=[TextContent(text="Executed")]),
            tool_call_id="call-1",
            action_id="action-1",
        )
    )

    # Events 4-19: More action/observation pairs (will be forgotten)
    for i in range(3, 11):
        events.append(
            ActionEvent(
                id=f"action-{i}",
                source="agent",
                thought=[TextContent(text=f"Action {i}")],
                thinking_blocks=[],  # No thinking blocks
                tool_name="terminal",
                tool_call_id=f"call-{i}",
                tool_call=MessageToolCall(
                    id=f"call-{i}", name="terminal", arguments="{}", origin="completion"
                ),
                llm_response_id=f"llm-resp-{i}",
            )
        )
        events.append(
            ObservationEvent(
                id=f"obs-{i}",
                source="environment",
                tool_name="terminal",
                observation=ThinkObservation(content=[TextContent(text=f"Result {i}")]),
                tool_call_id=f"call-{i}",
                action_id=f"action-{i}",
            )
        )

    # Event 20: ActionEvent without thinking block (will be kept)
    events.append(
        ActionEvent(
            id="action-11",
            source="agent",
            thought=[TextContent(text="Final action")],
            thinking_blocks=[],  # No thinking blocks!
            tool_name="terminal",
            tool_call_id="call-11",
            tool_call=MessageToolCall(
                id="call-11", name="terminal", arguments="{}", origin="completion"
            ),
            llm_response_id="llm-resp-11",
        )
    )
    # Event 21: Observation for action-11
    events.append(
        ObservationEvent(
            id="obs-11",
            source="environment",
            tool_name="terminal",
            observation=ThinkObservation(content=[TextContent(text="Final result")]),
            tool_call_id="call-11",
            action_id="action-11",
        )
    )

    # Event 22: Condensation that forgets action/observation pairs (including ones
    # with thinking blocks)
    forgotten_ids = []
    for i in range(1, 11):
        if i == 1:
            forgotten_ids.extend(["action-1", "obs-1"])
        elif i >= 3:
            forgotten_ids.extend([f"action-{i}", f"obs-{i}"])

    condensation = Condensation(
        id="condensation-1",
        forgotten_event_ids=forgotten_ids,
        summary="The agent analyzed code and performed several actions.",
        summary_offset=2,  # Insert summary after first 2 events
        llm_response_id="llm-condenser-1",
    )
    events.append(condensation)

    # Create view from events
    view = View.from_events(events)

    # Convert to messages
    from openhands.sdk.event.base import LLMConvertibleEvent

    messages = LLMConvertibleEvent.events_to_messages(view.events)

    # Check that we have messages
    assert len(messages) > 0

    # Find assistant messages
    assistant_messages = [msg for msg in messages if msg.role == "assistant"]

    # After the fix: At least one assistant message should have thinking blocks
    has_thinking_blocks = any(
        msg.thinking_blocks and len(msg.thinking_blocks) > 0
        for msg in assistant_messages
    )

    assert has_thinking_blocks, (
        "After condensation, at least one assistant message should have thinking "
        "blocks to satisfy Claude extended thinking requirements."
    )

    # Verify that action-1 (with thinking blocks) was preserved
    action_1_in_view = any(
        isinstance(e, ActionEvent) and e.id == "action-1" for e in view.events
    )
    assert action_1_in_view, "action-1 (with thinking blocks) should be preserved"
