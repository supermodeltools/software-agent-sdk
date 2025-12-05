from openhands.sdk.critic.base import CriticBase
from openhands.sdk.critic.result import CriticResult


__all__ = [
    "CriticBase",
    "CriticResult",
    "AgentFinishedCritic",
    "APIBasedCritic",
    "EmptyPatchCritic",
    "PassCritic",
]


def __getattr__(name: str):
    """Lazy import impl classes to avoid circular dependency."""
    if name == "AgentFinishedCritic":
        from openhands.sdk.critic.impl import AgentFinishedCritic

        return AgentFinishedCritic
    elif name == "APIBasedCritic":
        from openhands.sdk.critic.impl import APIBasedCritic

        return APIBasedCritic
    elif name == "EmptyPatchCritic":
        from openhands.sdk.critic.impl import EmptyPatchCritic

        return EmptyPatchCritic
    elif name == "PassCritic":
        from openhands.sdk.critic.impl import PassCritic

        return PassCritic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
