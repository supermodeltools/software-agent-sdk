"""Tests for critic implementations and registry."""

import json

import pytest

from openhands.sdk.critic import (
    AgentFinishedCritic,
    CriticBase,
    CriticRegistry,
    CriticResult,
    EmptyPatchCritic,
    PassCritic,
)
from openhands.sdk.event import ActionEvent
from openhands.sdk.llm import MessageToolCall, TextContent
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.tool.schema import Action


# Define a dummy action class once to avoid duplicate kind errors
class DummyAction(Action):
    """A simple dummy action for testing purposes."""

    pass


def test_critic_result_success_threshold():
    """Test that CriticResult determines success based on threshold."""
    # Score above threshold should be success
    result_success = CriticResult(score=0.8, message="Success")
    assert result_success.success is True

    # Score at threshold should be success
    result_at_threshold = CriticResult(score=0.5, message="At threshold")
    assert result_at_threshold.success is True

    # Score below threshold should not be success
    result_fail = CriticResult(score=0.3, message="Fail")
    assert result_fail.success is False


def test_critic_result_validation():
    """Test that CriticResult validates score bounds."""
    # Valid scores
    CriticResult(score=0.0, message="Min")
    CriticResult(score=1.0, message="Max")

    # Invalid scores should raise validation error
    with pytest.raises(Exception):  # Pydantic ValidationError
        CriticResult(score=-0.1, message="Below min")

    with pytest.raises(Exception):  # Pydantic ValidationError
        CriticResult(score=1.1, message="Above max")


def test_pass_critic_always_succeeds():
    """Test that PassCritic always returns success."""
    critic = PassCritic()

    # Empty events and no patch
    result = critic.evaluate([], None)
    assert result.score == 1.0
    assert result.success is True

    # With events but no patch
    events = [
        ActionEvent(
            thought=[TextContent(text="thinking")],
            tool_name="test",
            tool_call_id="test_id",
            tool_call=MessageToolCall(
                id="test_id",
                name="test",
                arguments=json.dumps({}),
                origin="completion",
            ),
            llm_response_id="resp_123",
        )
    ]
    result = critic.evaluate(events, None)
    assert result.score == 1.0
    assert result.success is True

    # With events and patch
    result = critic.evaluate(events, "some patch")
    assert result.score == 1.0
    assert result.success is True


def test_empty_patch_critic_with_empty_patch():
    """Test EmptyPatchCritic returns failure for empty patches."""
    critic = EmptyPatchCritic()

    # None patch
    result = critic.evaluate([], None)
    assert result.score == 0.0
    assert result.success is False
    assert result.message is not None
    assert "empty" in result.message.lower()

    # Empty string patch
    result = critic.evaluate([], "")
    assert result.score == 0.0
    assert result.success is False

    # Whitespace-only patch
    result = critic.evaluate([], "   \n\t  ")
    assert result.score == 0.0
    assert result.success is False


def test_empty_patch_critic_with_non_empty_patch():
    """Test EmptyPatchCritic returns success for non-empty patches."""
    critic = EmptyPatchCritic()

    patch = """
    diff --git a/file.py b/file.py
    index abc123..def456 100644
    --- a/file.py
    +++ b/file.py
    @@ -1,3 +1,4 @@
    +# New line
     print("hello")
    """

    result = critic.evaluate([], patch)
    assert result.score == 1.0
    assert result.success is True
    assert result.message is not None
    assert "non-empty" in result.message.lower()


def test_agent_finished_critic_with_empty_patch():
    """Test AgentFinishedCritic fails when patch is empty."""
    critic = AgentFinishedCritic()

    # Create events with FinishAction
    finish_action = FinishAction(message="Task completed")
    events = [
        ActionEvent(
            thought=[TextContent(text="I finished the task")],
            action=finish_action,
            tool_name="finish",
            tool_call_id="finish_id",
            tool_call=MessageToolCall(
                id="finish_id",
                name="finish",
                arguments=json.dumps({"message": "Task completed"}),
                origin="completion",
            ),
            llm_response_id="resp_finish",
        )
    ]

    # Should fail with empty patch even though agent finished
    result = critic.evaluate(events, None)
    assert result.score == 0.0
    assert result.success is False
    assert result.message is not None
    assert "empty" in result.message.lower()


def test_agent_finished_critic_without_finish_action():
    """Test AgentFinishedCritic fails when no FinishAction present."""
    critic = AgentFinishedCritic()

    patch = "diff --git a/file.py"

    # Empty events
    result = critic.evaluate([], patch)
    assert result.score == 0.0
    assert result.success is False

    # Events without FinishAction
    other_action = DummyAction()
    events = [
        ActionEvent(
            thought=[TextContent(text="doing something")],
            action=other_action,
            tool_name="other",
            tool_call_id="other_id",
            tool_call=MessageToolCall(
                id="other_id",
                name="other",
                arguments=json.dumps({}),
                origin="completion",
            ),
            llm_response_id="resp_other",
        )
    ]

    result = critic.evaluate(events, patch)
    assert result.score == 0.0
    assert result.success is False
    assert result.message is not None
    assert "finish" in result.message.lower()


def test_agent_finished_critic_success():
    """Test AgentFinishedCritic succeeds with FinishAction and non-empty patch."""
    critic = AgentFinishedCritic()

    patch = """
    diff --git a/file.py b/file.py
    --- a/file.py
    +++ b/file.py
    @@ -1 +1,2 @@
     original line
    +new line
    """

    finish_action = FinishAction(message="Task completed successfully")
    events = [
        ActionEvent(
            thought=[TextContent(text="Starting task")],
            action=None,
            tool_name="read",
            tool_call_id="read_id",
            tool_call=MessageToolCall(
                id="read_id",
                name="read",
                arguments=json.dumps({}),
                origin="completion",
            ),
            llm_response_id="resp_read",
        ),
        ActionEvent(
            thought=[TextContent(text="Finishing task")],
            action=finish_action,
            tool_name="finish",
            tool_call_id="finish_id",
            tool_call=MessageToolCall(
                id="finish_id",
                name="finish",
                arguments=json.dumps({"message": "Task completed successfully"}),
                origin="completion",
            ),
            llm_response_id="resp_finish_success",
        ),
    ]

    result = critic.evaluate(events, patch)
    assert result.score == 1.0
    assert result.success is True


def test_agent_finished_critic_last_action_not_finish():
    """Test AgentFinishedCritic fails when last action is not FinishAction."""
    critic = AgentFinishedCritic()

    patch = "diff --git a/file.py"

    finish_action = FinishAction(message="Task completed")
    other_action = DummyAction()

    # FinishAction is not the last action
    events = [
        ActionEvent(
            thought=[TextContent(text="Finishing")],
            action=finish_action,
            tool_name="finish",
            tool_call_id="finish_id",
            tool_call=MessageToolCall(
                id="finish_id",
                name="finish",
                arguments=json.dumps({"message": "Task completed"}),
                origin="completion",
            ),
            llm_response_id="resp_finish_mid",
        ),
        ActionEvent(
            thought=[TextContent(text="Doing more")],
            action=other_action,
            tool_name="other",
            tool_call_id="other_id",
            tool_call=MessageToolCall(
                id="other_id",
                name="other",
                arguments=json.dumps({}),
                origin="completion",
            ),
            llm_response_id="resp_other_last",
        ),
    ]

    result = critic.evaluate(events, patch)
    assert result.score == 0.0
    assert result.success is False


def test_critic_registry_register():
    """Test registering critics in the registry."""
    # Register a critic (not clearing to preserve defaults)
    CriticRegistry.register("test_pass", PassCritic)

    # Should be able to retrieve it
    critic_class = CriticRegistry.get("test_pass")
    assert critic_class is PassCritic

    # Should be in the list
    assert "test_pass" in CriticRegistry.list_registered()


def test_critic_registry_register_invalid():
    """Test that registering invalid critics raises errors."""
    # Empty name
    with pytest.raises(ValueError):
        CriticRegistry.register("", PassCritic)

    # Non-string name
    with pytest.raises(ValueError):
        CriticRegistry.register(123, PassCritic)  # type: ignore

    # Not a class
    with pytest.raises(TypeError):
        CriticRegistry.register("invalid", PassCritic())  # type: ignore

    # Not a CriticBase subclass
    with pytest.raises(TypeError):
        CriticRegistry.register("invalid", str)  # type: ignore


def test_critic_registry_get_not_found():
    """Test that getting unregistered critic raises KeyError."""
    with pytest.raises(KeyError):
        CriticRegistry.get("nonexistent_critic_xyz123")


def test_critic_registry_create():
    """Test creating critic instances from registry."""
    # Use existing registration
    # Create instance
    critic = CriticRegistry.create("pass")
    assert isinstance(critic, PassCritic)
    assert isinstance(critic, CriticBase)

    # Each call should create a new instance
    critic2 = CriticRegistry.create("pass")
    assert critic is not critic2


def test_critic_registry_default_registrations():
    """Test that default critics are registered on import."""
    # These should be registered by __init__.py
    assert "finish_with_patch" in CriticRegistry.list_registered()
    assert "empty_patch_critic" in CriticRegistry.list_registered()
    assert "pass" in CriticRegistry.list_registered()

    # Verify they can be created
    finish_critic = CriticRegistry.create("finish_with_patch")
    assert isinstance(finish_critic, AgentFinishedCritic)

    empty_critic = CriticRegistry.create("empty_patch_critic")
    assert isinstance(empty_critic, EmptyPatchCritic)

    pass_critic = CriticRegistry.create("pass")
    assert isinstance(pass_critic, PassCritic)


def test_critic_registry_duplicate_warning(caplog):
    """Test that registering duplicate names logs a warning."""
    # Use a unique name that won't conflict
    test_name = "duplicate_test_critic_xyz"

    # First registration
    CriticRegistry.register(test_name, PassCritic)

    # Second registration should log warning
    CriticRegistry.register(test_name, EmptyPatchCritic)

    # Should use the latest registration
    critic_class = CriticRegistry.get(test_name)
    assert critic_class is EmptyPatchCritic


def test_critic_base_is_abstract():
    """Test that CriticBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        CriticBase()  # type: ignore
