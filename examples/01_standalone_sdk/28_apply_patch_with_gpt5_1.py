"""Example: Using ApplyPatch tool with GPT-5.1 models via direct OpenAI API.

This demonstrates adding the ApplyPatch tool to the agent and guiding the
model through a richer sequence of file operations using 'apply_patch' text:

- Create multiple files (FACTS.txt, NOTES.md)
- Apply multi-hunk edits to a single file
- Apply a single patch that touches multiple files
- Mix add / update / delete operations in one patch

Notes:
- Works with any GPT-5.1 family model (names start with "gpt-5.1").
- Uses direct OpenAI API through LiteLLM's LLM wrapper with no base_url.
- Requires OPENAI_API_KEY in the environment (or LLM_API_KEY fallback).
"""

from __future__ import annotations

import os

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, get_logger
from openhands.sdk.tool import Tool
from openhands.tools.apply_patch import ApplyPatchTool
from openhands.tools.task_tracker import TaskTrackerTool

# from openhands.tools.preset.default import register_default_tools
from openhands.tools.terminal import TerminalTool


logger = get_logger(__name__)

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
assert api_key, "Set OPENAI_API_KEY (or LLM_API_KEY) in your environment."

# Choose a GPT-5.1 model; mini is cost-effective for examples
default_model = "openai/gpt-5.1-codex-mini"
model = os.getenv("LLM_MODEL", default_model)
assert model.startswith("openai/gpt-5.1"), "Model must be an openai gpt-5.1 variant"

llm = LLM(
    model=model,
    api_key=SecretStr(api_key),
    reasoning_summary=None,  # avoid OpenAI org verification requirement
    log_completions=True,  # enable telemetry to log input/output payloads
)

# Explicitly register tool classes so Tool(name=...) can resolve
# They self-register into the global registry on import
_ = (TerminalTool, TaskTrackerTool, ApplyPatchTool)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name="terminal"),
        Tool(name="task_tracker"),
        Tool(name="apply_patch"),
    ],
    system_prompt_kwargs={"cli_mode": True},
)

conversation = Conversation(agent=agent, workspace=os.getcwd())

# Compose instructions guiding the model to exercise richer ApplyPatch behavior.
prompt = (
    "You have access to an apply_patch tool that edits files using unified patches. "
    "Use it to perform the following sequence of operations in as few patches as "
    "reasonable, while keeping each patch valid and focused:\n\n"
    "1) Create two files:\n"
    "   - FACTS.txt containing exactly two lines:\n"
    "       OpenHands SDK integrates tools.\n"
    "       ApplyPatch can edit multiple files.\n"
    "   - NOTES.md containing exactly three lines:\n"
    "       # Notes\n"
    "       - Initial point A\n"
    "       - Initial point B\n\n"
    "2) Apply a multi-hunk update to NOTES.md in a single *** Update File block:\n"
    "   - Change the text 'Initial point A' to 'Updated point A'.\n"
    "   - Append a new bullet '- Added via multi-hunk patch.' after "
    "'Initial point B'.\n\n"
    "3) Apply a single patch that updates BOTH FACTS.txt and NOTES.md at once:\n"
    "   - In FACTS.txt, append a third line: 'Multi-file patches are supported.'.\n"
    "   - In NOTES.md, append a final line: 'Summary: multi-file patch applied.'.\n\n"
    "4) Finally, use one more patch that mixes operations across files:\n"
    "   - Add a TEMP.txt file containing a single line: 'Temporary file'.\n"
    "   - Append a line 'Cleanup step ran.' to FACTS.txt.\n"
    "   - Delete TEMP.txt.\n\n"
    "Important rules:\n"
    "- Only call the tool using the apply_patch text format between "
    "'*** Begin Patch' and '*** End Patch'.\n"
    "- Use '*** Add File', '*** Update File', and '*** Delete File' sections as "
    "described in the GPT-5.1 apply_patch guide.\n"
    "- When updating a file, include enough context lines so the patch can be "
    "applied even if whitespace varies slightly.\n"
)

conversation.send_message(prompt)
conversation.run()

print("Conversation finished.")
print(f"EXAMPLE_COST: {llm.metrics.accumulated_cost}")
