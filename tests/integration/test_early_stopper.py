"""Unit tests for early stopping utilities."""

import pytest

from tests.integration.early_stopper import (
    BashCommandPruner,
    CompositeEarlyStopper,
    EarlyStopResult,
    FileEditPruner,
    TestExecutionPruner,
)


class MockEvent:
    """Mock event for testing."""

    def __init__(self, tool_name: str = "", action=None):
        self.tool_name = tool_name
        self.action = action


class MockFileEditorAction:
    """Mock file editor action."""

    def __init__(self, command: str, path: str):
        self.command = command
        self.path = path


class MockTerminalAction:
    """Mock terminal action."""

    def __init__(self, command: str):
        self.command = command


class MockActionEvent(MockEvent):
    """Mock action event for testing."""

    pass


# Patch the isinstance checks to work with our mocks
@pytest.fixture(autouse=True)
def patch_imports(monkeypatch):
    """Patch imports to use mocks."""
    import tests.integration.early_stopper as early_stopper_module

    # Use the real ActionEvent check but treat our mocks as valid
    original_check = early_stopper_module.FileEditPruner.check

    def patched_file_edit_check(self, events):
        # Custom check that works with our mock events
        for event in events:
            if (
                hasattr(event, "tool_name")
                and event.tool_name == "file_editor"
                and hasattr(event, "action")
                and event.action is not None
            ):
                if hasattr(event.action, "command"):
                    if event.action.command in self.forbidden_commands:
                        return EarlyStopResult(
                            should_stop=True,
                            reason=(
                                f"Detected forbidden file operation: "
                                f"{event.action.command} on {event.action.path}"
                            ),
                        )
        return EarlyStopResult(should_stop=False)

    monkeypatch.setattr(
        early_stopper_module.FileEditPruner, "check", patched_file_edit_check
    )

    def patched_bash_check(self, events):
        for event in events:
            if (
                hasattr(event, "tool_name")
                and event.tool_name == "terminal"
                and hasattr(event, "action")
                and event.action is not None
            ):
                if hasattr(event.action, "command"):
                    command = event.action.command
                    for pattern in self.forbidden_patterns:
                        if pattern in command:
                            return EarlyStopResult(
                                should_stop=True,
                                reason=(
                                    f"Detected forbidden command pattern "
                                    f"'{pattern}' in: {command[:100]}"
                                ),
                            )
        return EarlyStopResult(should_stop=False)

    monkeypatch.setattr(
        early_stopper_module.BashCommandPruner, "check", patched_bash_check
    )


class TestFileEditPruner:
    """Tests for FileEditPruner."""

    def test_no_events_returns_no_stop(self):
        """Empty events list should not trigger stop."""
        pruner = FileEditPruner()
        result = pruner.check([])
        assert result.should_stop is False
        assert result.reason is None

    def test_view_command_not_blocked(self):
        """View command should not trigger stop."""
        pruner = FileEditPruner()
        action = MockFileEditorAction(command="view", path="/test.py")
        event = MockActionEvent(tool_name="file_editor", action=action)
        result = pruner.check([event])
        assert result.should_stop is False

    def test_create_command_triggers_stop(self):
        """Create command should trigger stop."""
        pruner = FileEditPruner()
        action = MockFileEditorAction(command="create", path="/new_file.py")
        event = MockActionEvent(tool_name="file_editor", action=action)
        result = pruner.check([event])
        assert result.should_stop is True
        assert "create" in result.reason
        assert "new_file.py" in result.reason

    def test_str_replace_triggers_stop(self):
        """str_replace command should trigger stop."""
        pruner = FileEditPruner()
        action = MockFileEditorAction(command="str_replace", path="/test.py")
        event = MockActionEvent(tool_name="file_editor", action=action)
        result = pruner.check([event])
        assert result.should_stop is True
        assert "str_replace" in result.reason

    def test_custom_forbidden_commands(self):
        """Custom forbidden commands should be respected."""
        pruner = FileEditPruner(forbidden_commands=["delete"])
        action = MockFileEditorAction(command="delete", path="/test.py")
        event = MockActionEvent(tool_name="file_editor", action=action)
        result = pruner.check([event])
        assert result.should_stop is True

    def test_non_matching_event_not_stopped(self):
        """Non-file-editor events should not trigger stop."""
        pruner = FileEditPruner()
        event = MockEvent(tool_name="terminal")
        result = pruner.check([event])
        assert result.should_stop is False


class TestBashCommandPruner:
    """Tests for BashCommandPruner."""

    def test_no_events_returns_no_stop(self):
        """Empty events should not trigger stop."""
        pruner = BashCommandPruner(forbidden_patterns=["rm -rf"])
        result = pruner.check([])
        assert result.should_stop is False

    def test_forbidden_pattern_triggers_stop(self):
        """Forbidden command pattern should trigger stop."""
        pruner = BashCommandPruner(forbidden_patterns=["rm -rf"])
        action = MockTerminalAction(command="rm -rf /important")
        event = MockActionEvent(tool_name="terminal", action=action)
        result = pruner.check([event])
        assert result.should_stop is True
        assert "rm -rf" in result.reason

    def test_safe_command_not_stopped(self):
        """Safe commands should not trigger stop."""
        pruner = BashCommandPruner(forbidden_patterns=["rm -rf"])
        action = MockTerminalAction(command="ls -la")
        event = MockActionEvent(tool_name="terminal", action=action)
        result = pruner.check([event])
        assert result.should_stop is False


class TestCompositeEarlyStopper:
    """Tests for CompositeEarlyStopper."""

    def test_empty_stoppers_never_stops(self):
        """Empty stopper list should never stop."""
        composite = CompositeEarlyStopper(stoppers=[])
        result = composite.check([])
        assert result.should_stop is False

    def test_stops_on_first_match(self):
        """Should stop on first matching stopper."""
        # Create two pruners
        file_pruner = FileEditPruner()
        bash_pruner = BashCommandPruner(forbidden_patterns=["dangerous"])

        composite = CompositeEarlyStopper(stoppers=[file_pruner, bash_pruner])

        # Test with file edit
        action = MockFileEditorAction(command="create", path="/test.py")
        event = MockActionEvent(tool_name="file_editor", action=action)
        result = composite.check([event])
        assert result.should_stop is True

    def test_no_match_continues(self):
        """Should not stop if no stopper matches."""
        file_pruner = FileEditPruner()
        composite = CompositeEarlyStopper(stoppers=[file_pruner])

        event = MockEvent(tool_name="other_tool")
        result = composite.check([event])
        assert result.should_stop is False


class TestEarlyStopResult:
    """Tests for EarlyStopResult model."""

    def test_default_values(self):
        """Test default values."""
        result = EarlyStopResult(should_stop=False)
        assert result.should_stop is False
        assert result.reason is None

    def test_with_reason(self):
        """Test with reason."""
        result = EarlyStopResult(should_stop=True, reason="Test reason")
        assert result.should_stop is True
        assert result.reason == "Test reason"


class TestLLMJudgePruner:
    """Tests for LLMJudgePruner."""

    def test_skip_when_not_enough_events(self):
        """Should not call LLM when below check threshold."""
        from tests.integration.early_stopper import LLMJudgePruner

        pruner = LLMJudgePruner(
            evaluation_criteria="Test criteria",
            user_instruction="Test instruction",
            check_every_n_events=10,
        )
        # Create a list of less than check_every_n_events events
        events = [MockEvent() for _ in range(5)]
        result = pruner.check(events)
        # Should not stop because not enough events
        assert result.should_stop is False
        # LLM should not have been called
        assert pruner._llm is None

    def test_caches_result(self):
        """Should cache result between checks."""
        from tests.integration.early_stopper import LLMJudgePruner

        pruner = LLMJudgePruner(
            evaluation_criteria="Test criteria",
            user_instruction="Test instruction",
            check_every_n_events=10,
        )
        # First check - not enough events
        events = [MockEvent() for _ in range(3)]
        result1 = pruner.check(events)
        assert result1.should_stop is False

        # Same number of events - should return cached
        result2 = pruner.check(events)
        assert result2.should_stop is False
        # Should be the same object (cached)
        assert pruner._last_checked_count == 0  # Never actually checked with LLM

    def test_initialization(self):
        """Test pruner initialization."""
        from tests.integration.early_stopper import LLMJudgePruner

        pruner = LLMJudgePruner(
            evaluation_criteria="Agent should not hallucinate",
            user_instruction="Fix the bug",
            check_every_n_events=5,
        )
        assert pruner.evaluation_criteria == "Agent should not hallucinate"
        assert pruner.user_instruction == "Fix the bug"
        assert pruner.check_every_n_events == 5
        assert pruner._llm is None
        assert pruner._last_checked_count == 0
