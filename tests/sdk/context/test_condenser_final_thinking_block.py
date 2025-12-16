"""Test condenser handling of thinking blocks in the final assistant message.

This test reproduces the issue where Claude's API rejects requests when:
1. Extended thinking is enabled (thinking=True in API params)
2. The FINAL/LAST assistant message in the conversation history lacks thinking blocks

The error message from Claude:
'messages.X.content.0.type: Expected `thinking` or `redacted_thinking`, but found
`tool_use`. When `thinking` is enabled, a final `assistant` message must start with
a thinking block'
"""

from openhands.sdk.context.view import View
from openhands.sdk.event import (
    ActionEvent,
    Condensation,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.llm import Message, MessageToolCall, TextContent, ThinkingBlock
from openhands.sdk.mcp.definition import MCPToolObservation


def create_observation(
    tool_call_id: str, action_id: str, content: str = "Success"
) -> ObservationEvent:
    """Helper to create an ObservationEvent."""
    return ObservationEvent(
        observation=MCPToolObservation.from_text(
            text=content,
            tool_name="test_tool",
        ),
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        action_id=action_id,
        source="environment",
    )


def test_condenser_strips_thinking_when_final_action_lacks_thinking():
    """Test that thinking blocks are stripped when the last ActionEvent has none.

    Scenario:
    - Early ActionEvents have thinking blocks
    - Later ActionEvents don't have thinking blocks
    - Condensation forgets the early events (with thinking)
    - After condensation, the last ActionEvent lacks thinking blocks
    - To prevent Claude API errors, ALL thinking blocks should be stripped
    """
    # Event 1: Early ActionEvent WITH thinking blocks
    early_action = ActionEvent(
        id="action-001",
        thought=[TextContent(text="Early thought")],
        thinking_blocks=[
            ThinkingBlock(
                type="thinking",
                thinking="Let me think about this problem...",
                signature="sig1",
            )
        ],
        tool_name="test_tool",
        tool_call_id="tool-001",
        tool_call=MessageToolCall(
            id="tool-001",
            name="test_tool",
            arguments='{"arg": "value1"}',
            origin="completion",
        ),
        llm_response_id="resp-001",
        source="agent",
    )

    # Event 2: Observation for early action
    early_obs = create_observation("tool-001", early_action.id, "Result 1")

    # Event 3: User message
    user_msg_1 = MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text="First question")]),
        source="user",
    )

    # Event 4: Later ActionEvent WITHOUT thinking blocks
    late_action = ActionEvent(
        id="action-002",
        thought=[TextContent(text="Late thought")],
        thinking_blocks=[],  # No thinking blocks!
        tool_name="test_tool",
        tool_call_id="tool-002",
        tool_call=MessageToolCall(
            id="tool-002",
            name="test_tool",
            arguments='{"arg": "value2"}',
            origin="completion",
        ),
        llm_response_id="resp-002",
        source="agent",
    )

    # Event 5: Observation for late action
    late_obs = create_observation("tool-002", late_action.id, "Result 2")

    # Event 6: User message
    user_msg_2 = MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text="Second question")]),
        source="user",
    )

    # Event 7: Condensation that forgets the early action (with thinking)
    condensation = Condensation(
        forgotten_event_ids=[early_action.id, early_obs.id, user_msg_1.id],
        llm_response_id="cond-resp-001",
    )

    events = [
        early_action,
        early_obs,
        user_msg_1,
        late_action,
        late_obs,
        user_msg_2,
        condensation,
    ]

    # Create view after condensation
    view = View.from_events(events)

    # After condensation, only late_action, late_obs, and user_msg_2 should be kept
    assert len(view.events) == 3
    # Find the action event
    action_events = [e for e in view.events if isinstance(e, ActionEvent)]
    assert len(action_events) == 1
    last_action = action_events[0]
    assert last_action.id == late_action.id

    # The key assertion: thinking blocks should be stripped to prevent Claude API errors
    # Since the last ActionEvent has no thinking blocks, ALL thinking blocks should be
    # removed to maintain consistency
    assert len(last_action.thinking_blocks) == 0

    # Also verify via to_llm_message
    llm_msg = last_action.to_llm_message()
    assert len(llm_msg.thinking_blocks) == 0


def test_condenser_preserves_thinking_when_final_action_has_thinking():
    """Test that thinking blocks are preserved when the last ActionEvent has them.

    Scenario:
    - Early ActionEvents don't have thinking blocks
    - Later ActionEvents DO have thinking blocks
    - Condensation forgets the early events (without thinking)
    - After condensation, the last ActionEvent HAS thinking blocks
    - Thinking blocks should be preserved (Claude will accept this)
    """
    # Event 1: Early ActionEvent WITHOUT thinking blocks
    early_action = ActionEvent(
        id="action-001",
        thought=[TextContent(text="Early thought")],
        thinking_blocks=[],
        tool_name="test_tool",
        tool_call_id="tool-001",
        tool_call=MessageToolCall(
            id="tool-001",
            name="test_tool",
            arguments='{"arg": "value1"}',
            origin="completion",
        ),
        llm_response_id="resp-001",
        source="agent",
    )

    # Event 2: Observation for early action
    early_obs = create_observation("tool-001", early_action.id, "Result 1")

    # Event 3: User message
    user_msg_1 = MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text="First question")]),
        source="user",
    )

    # Event 4: Later ActionEvent WITH thinking blocks
    late_action = ActionEvent(
        id="action-002",
        thought=[TextContent(text="Late thought")],
        thinking_blocks=[
            ThinkingBlock(
                type="thinking",
                thinking="Let me analyze this carefully...",
                signature="sig2",
            )
        ],
        tool_name="test_tool",
        tool_call_id="tool-002",
        tool_call=MessageToolCall(
            id="tool-002",
            name="test_tool",
            arguments='{"arg": "value2"}',
            origin="completion",
        ),
        llm_response_id="resp-002",
        source="agent",
    )

    # Event 5: Observation for late action
    late_obs = create_observation("tool-002", late_action.id, "Result 2")

    # Event 6: User message
    user_msg_2 = MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text="Second question")]),
        source="user",
    )

    # Event 7: Condensation that forgets the early action (without thinking)
    condensation = Condensation(
        forgotten_event_ids=[early_action.id, early_obs.id, user_msg_1.id],
        llm_response_id="cond-resp-001",
    )

    events = [
        early_action,
        early_obs,
        user_msg_1,
        late_action,
        late_obs,
        user_msg_2,
        condensation,
    ]

    # Create view after condensation
    view = View.from_events(events)

    # After condensation, only late_action, late_obs, and user_msg_2 should be kept
    assert len(view.events) == 3
    # Find the action event
    action_events = [e for e in view.events if isinstance(e, ActionEvent)]
    assert len(action_events) == 1
    last_action = action_events[0]
    assert last_action.id == late_action.id

    # Thinking blocks should be PRESERVED since the last ActionEvent has them
    assert len(last_action.thinking_blocks) == 1
    llm_msg = last_action.to_llm_message()
    assert len(llm_msg.thinking_blocks) == 1
    block = llm_msg.thinking_blocks[0]
    assert isinstance(block, ThinkingBlock)
    assert block.thinking == "Let me analyze this carefully..."
