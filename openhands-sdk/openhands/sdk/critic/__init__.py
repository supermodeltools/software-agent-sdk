from openhands.sdk.critic.base import CriticBase, CriticResult
from openhands.sdk.critic.impl import (
    AgentFinishedCritic,
    EmptyPatchCritic,
    PassCritic,
)
from openhands.sdk.critic.registry import CriticRegistry


# Register default critics
CriticRegistry.register("finish_with_patch", AgentFinishedCritic)
CriticRegistry.register("empty_patch_critic", EmptyPatchCritic)
CriticRegistry.register("pass", PassCritic)


__all__ = [
    "CriticBase",
    "CriticResult",
    "AgentFinishedCritic",
    "EmptyPatchCritic",
    "PassCritic",
    "CriticRegistry",
]
