"""Example demonstrating critic-based evaluation of agent actions.

This shows how to configure an agent with a critic to evaluate action quality
in real-time. The critic scores are displayed in the conversation visualizer.
"""

import os

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.critic import APIBasedCritic
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool


llm = LLM(
    model=os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", None),
)

critic = APIBasedCritic(
    server_url=os.getenv("CRITIC_SERVER_URL", ""),
    api_key=os.getenv("CRITIC_API_KEY", ""),
    model_name=os.getenv("CRITIC_MODEL_NAME", ""),
    pass_tools_definitions=True,
    has_success_label=True,
)


# Configure agent with critic
agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ],
    # Add critic to evaluate agent actions
    critic=critic,
    # finish_and_message: evaluate on FinishAction and agent MessageEvent (default)
    critic_evaluation_mode="finish_and_message",
)

cwd = os.getcwd()
conversation = Conversation(agent=agent, workspace=cwd)

conversation.send_message(
    "Create a file called GREETING.txt with a friendly greeting message."
)
conversation.run()

print("\nAll done! Check the output above for 'Critic Score' in the visualizer.")
