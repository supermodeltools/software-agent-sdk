"""Test that an agent can wait for a background process to finish.

This integration test verifies that the agent can:
1. Start a background process with output redirection
2. Capture the PID and wait for completion using tail --pid
3. Check the exit status after the process completes
"""

import os

from openhands.sdk import get_logger
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from tests.integration.base import BaseIntegrationTest, TestResult
from tests.integration.behavior_utils import get_conversation_summary
from tests.integration.utils.llm_judge import judge_agent_behavior


INSTRUCTION = (
    "Run the script 'long_task.sh' in the background using 'nohup'. Redirect output "
    "to 'output.log'. Capture the PID using '$!' and then wait for it to finish using "
    "the shell 'wait' builtin. After it finishes, tell me the final result from the "
    "output file and whether the script succeeded or failed based on the exit code "
    "from 'wait'."
)

# A script that takes a few seconds to complete and writes output
SCRIPT_CONTENT = """#!/bin/bash
echo "Starting long task..."
sleep 1
echo "Step 1 complete"
sleep 1
echo "Step 2 complete"
sleep 1
echo "FINAL_RESULT: SUCCESS_12345"
exit 0
"""


logger = get_logger(__name__)


class WaitForBackgroundProcessTest(BaseIntegrationTest):
    """Test that an agent can wait for a background process to finish."""

    INSTRUCTION: str = INSTRUCTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.script_path: str = os.path.join(self.workspace, "long_task.sh")
        self.output_path: str = os.path.join(self.workspace, "output.log")

    @property
    def tools(self) -> list[Tool]:
        """List of tools available to the agent."""
        register_tool("TerminalTool", TerminalTool)
        register_tool("FileEditorTool", FileEditorTool)
        return [
            Tool(name="TerminalTool"),
            Tool(name="FileEditorTool"),
        ]

    def setup(self) -> None:
        """Create the long-running script for the agent to execute."""
        with open(self.script_path, "w") as f:
            f.write(SCRIPT_CONTENT)
        os.chmod(self.script_path, 0o755)
        logger.info(f"Created long-running script at {self.script_path}")

    def verify_result(self) -> TestResult:
        """Verify that the agent waited for the process and got the result."""
        if not os.path.exists(self.output_path):
            return TestResult(
                success=False,
                reason=(
                    "Expected output.log not found. Agent may not have executed script."
                ),
            )

        with open(self.output_path) as f:
            content = f.read()

        if "FINAL_RESULT: SUCCESS_12345" not in content:
            return TestResult(
                success=False,
                reason=f"output.log missing expected result. Content: {content}",
            )

        conversation_summary = get_conversation_summary(self.collected_events)
        evaluation_criteria = (
            "Approve only if the agent used the required background-process waiting "
            "pattern:\n"
            "1) used nohup to start the process in the background with output "
            "redirection\n"
            "2) captured PID via $!\n"
            "3) waited for completion using the shell wait builtin (wait $PID)\n"
            "The agent may chain commands with && or ; as long as the above are "
            "clearly used."
        )

        judgment = judge_agent_behavior(
            user_instruction=INSTRUCTION,
            conversation_summary=conversation_summary,
            evaluation_criteria=evaluation_criteria,
        )

        self.add_judge_usage(
            prompt_tokens=judgment.prompt_tokens,
            completion_tokens=judgment.completion_tokens,
            cost=judgment.cost,
        )

        if not judgment.approved:
            return TestResult(
                success=False,
                reason=(
                    "Agent did not demonstrate the required nohup/PID/wait pattern. "
                    f"Judge reasoning: {judgment.reasoning} "
                    f"(confidence={judgment.confidence:.2f})"
                ),
            )

        return TestResult(
            success=True,
            reason=(
                "Agent produced the expected output and used the required "
                "nohup/PID/wait pattern. "
                f"Judge reasoning: {judgment.reasoning} "
                f"(confidence={judgment.confidence:.2f})"
            ),
        )
