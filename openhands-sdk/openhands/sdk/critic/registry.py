"""Registry for critic implementations."""

from threading import RLock

from openhands.sdk.critic.base import CriticBase
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

_LOCK = RLock()
_REG: dict[str, type[CriticBase]] = {}


class CriticRegistry:
    """Registry for managing critic implementations."""

    @staticmethod
    def register(name: str, critic_class: type[CriticBase]) -> None:
        """
        Register a critic implementation.

        Args:
            name: The name to register the critic under
            critic_class: The critic class to register

        Raises:
            ValueError: If name is empty or critic_class is not a CriticBase subclass
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Critic name must be a non-empty string")

        if not isinstance(critic_class, type) or not issubclass(
            critic_class, CriticBase
        ):
            raise TypeError(
                f"Critic must be a subclass of CriticBase, got {type(critic_class)}"
            )

        with _LOCK:
            if name in _REG:
                logger.warning(f"Duplicate critic name registered: {name}")
            _REG[name] = critic_class

    @staticmethod
    def get(name: str) -> type[CriticBase]:
        """
        Get a critic class by name.

        Args:
            name: The name of the critic to retrieve

        Returns:
            The critic class

        Raises:
            KeyError: If the critic is not registered
        """
        with _LOCK:
            critic_class = _REG.get(name)

        if critic_class is None:
            raise KeyError(f"Critic '{name}' is not registered")

        return critic_class

    @staticmethod
    def create(name: str) -> CriticBase:
        """
        Create a new instance of a critic by name.

        Args:
            name: The name of the critic to create

        Returns:
            A new instance of the critic

        Raises:
            KeyError: If the critic is not registered
        """
        critic_class = CriticRegistry.get(name)
        return critic_class()

    @staticmethod
    def list_registered() -> list[str]:
        """
        Get a list of all registered critic names.

        Returns:
            A list of registered critic names
        """
        with _LOCK:
            return list(_REG.keys())

    @staticmethod
    def clear() -> None:
        """Clear all registered critics. Useful for testing."""
        with _LOCK:
            _REG.clear()
