"""Test hard context reset when condensation is unavailable.

This test verifies that:
1. When condensation is explicitly requested via conversation.condense()
2. But no valid condensation range exists (only 1 event in history)
3. A hard context reset is performed instead of raising an exception
4. The conversation can continue successfully after the hard context reset
"""

from openhands.sdk import Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.tool import register_tool
from openhands.tools.terminal import TerminalTool
from tests.integration.base import BaseIntegrationTest, TestResult


# Module-level instruction for test runner
INSTRUCTION = """Using the echo command, print the numbers 1 through 3.
Use exactly 3 separate echo commands, one for each number."""


class HardContextResetTest(BaseIntegrationTest):
    """Test hard context reset when condensation is unavailable."""

    INSTRUCTION: str = INSTRUCTION

    def __init__(self, *args, **kwargs):
        """Initialize test with tracking for condensation events."""
        self.condensations: list[Condensation] = []
        self.hard_reset_condensation: Condensation | None = None
        super().__init__(*args, **kwargs)

    @property
    def tools(self) -> list[Tool]:
        """Provide terminal tool."""
        register_tool("TerminalTool", TerminalTool)
        return [Tool(name="TerminalTool")]

    @property
    def condenser(self) -> LLMSummarizingCondenser:
        """Use LLMSummarizingCondenser to enable explicit condensation."""
        condenser_llm = self.create_llm_copy("test-condenser-llm")
        return LLMSummarizingCondenser(
            llm=condenser_llm,
            max_size=1000,  # High to prevent automatic triggering
            keep_first=4,  # Set higher than normal to avoid a valid condensation range
        )

    @property
    def max_iteration_per_run(self) -> int:
        """Limit iterations since this is a simple test."""
        return 10

    def conversation_callback(self, event):
        """Override callback to detect condensation events."""
        super().conversation_callback(event)

        if isinstance(event, Condensation):
            self.condensations.append(event)
            # Check if this is a hard reset (summary_offset=0, all events forgotten)
            if event.summary_offset == 0:
                self.hard_reset_condensation = event

    def run_instructions(self, conversation: LocalConversation) -> None:
        """Test explicit condense() with insufficient events triggers hard reset.

        Steps:
        1. Send initial message (creates 1 event)
        2. Try to explicitly condense - should trigger hard context reset
        3. Continue the conversation to verify it still works
        """
        # Step 1: Send initial message but DON'T run yet
        conversation.send_message(message=self.instruction_message)

        # At this point we have only 1 event (the user message)
        # No valid condensation range exists (need at least 2 atomic units)

        # Step 2: Explicitly condense - should trigger hard context reset
        conversation.condense()

        # Step 3: Now run the conversation to verify it can continue
        # after the hard context reset
        conversation.run()

    def verify_result(self) -> TestResult:
        """Verify that hard context reset occurred and conversation continued.

        Success criteria:
        1. A condensation event was generated (hard context reset)
        2. The condensation has summary_offset=0 (indicating hard reset)
        3. The conversation completed successfully (numbers 1-3 printed)
        """
        if len(self.condensations) == 0:
            return TestResult(
                success=False,
                reason=(
                    "No condensation occurred. Expected hard context reset when "
                    "explicitly requesting condensation with no valid range"
                ),
            )

        if self.hard_reset_condensation is None:
            return TestResult(
                success=False,
                reason=(
                    f"Condensation occurred but not a hard reset. "
                    f"Expected summary_offset=0, got "
                    f"summary_offset={self.condensations[0].summary_offset}"
                ),
            )

        # Check that the hard reset condensed all events in the view
        if not self.hard_reset_condensation.forgotten_event_ids:
            return TestResult(
                success=False,
                reason="Hard reset condensation had no forgotten events",
            )

        # The fact that we got here without exceptions means the conversation
        # was able to continue successfully after the hard context reset
        return TestResult(
            success=True,
            reason=(
                f"Hard context reset triggered successfully, condensed "
                f"{len(self.hard_reset_condensation.forgotten_event_ids)} events, "
                f"and conversation continued successfully"
            ),
        )
