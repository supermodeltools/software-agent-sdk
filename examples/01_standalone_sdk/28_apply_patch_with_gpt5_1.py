"""Example: Using ApplyPatch tool with GPT-5.1 models via direct OpenAI API.

This demonstrates adding a new ApplyPatch tool to the agent and guiding the
model to create, modify, and delete a FACTS.txt file using 'apply_patch' text.

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
from openhands.tools.preset.default import register_default_tools


logger = get_logger(__name__)

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
assert api_key, "Set OPENAI_API_KEY (or LLM_API_KEY) in your environment."

# Choose a GPT-5.1 model; mini is cost-effective for examples
default_model = "openai/gpt-5.1-codex-mini"
model = os.getenv("LLM_MODEL", default_model)
assert model.startswith("openai/gpt-5.1"), "Model must be an openai gpt-5.1 variant"

# Force Chat Completions path by using a non-Responses model alias if needed
if model.startswith("openai/gpt-5.1"):
    # Litellm treats 'openai/gpt-5.1' via Responses; to avoid the Responses tool-output
    # coupling for this example, we can strip the provider prefix for chat path.
    # However, leave as-is to try Responses first; if it errors, instruct user below.
    pass

llm = LLM(
    model=model,
    api_key=SecretStr(api_key),
    native_tool_calling=True,  # enable native tool calling (Responses API)
    reasoning_summary=None,  # avoid OpenAI org verification requirement
)

# Ensure default tools are registered (terminal/file_editor/task_tracker)
register_default_tools(enable_browser=False)

# Add our new ApplyPatchTool by name
additional_tools = [Tool(name=ApplyPatchTool.name)]

agent = Agent(
    llm=llm,
    tools=[
        # Keep the standard console-friendly tools, plus our ApplyPatch
        Tool(name="terminal"),  # TerminalTool.name resolves to "terminal"
        Tool(name="file_editor"),  # FileEditorTool
        Tool(name="task_tracker"),  # TaskTrackerTool
        *additional_tools,
    ],
    system_prompt_kwargs={"cli_mode": True},
)

conversation = Conversation(agent=agent, workspace=os.getcwd())

# Compose instructions guiding the model to use the new tool
prompt = (
    "Use the ApplyPatch tool to: "
    "1) create a FACTS.txt with a single line 'OpenHands SDK integrates tools.'; "
    "2) modify FACTS.txt by appending a second line 'ApplyPatch works.'; "
    "3) delete FACTS.txt. "
    "Only use the apply_patch format between '*** Begin Patch' and '*** End Patch' "
    "when calling the tool."
)

conversation.send_message(prompt)
conversation.run()

print("Conversation finished.")
print(f"EXAMPLE_COST: {llm.metrics.accumulated_cost}")
