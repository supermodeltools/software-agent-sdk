from collections.abc import Sequence
from logging import getLogger
from typing import overload

from pydantic import BaseModel

from openhands.sdk.event import (
    Condensation,
    CondensationRequest,
    CondensationSummaryEvent,
    LLMConvertibleEvent,
)
from openhands.sdk.event.base import Event, EventID
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    ObservationBaseEvent,
)
from openhands.sdk.event.types import ToolCallID


logger = getLogger(__name__)


class View(BaseModel):
    """Linearly ordered view of events.

    Produced by a condenser to indicate the included events are ready to process as LLM
    input. Also contains fields with information from the condensation process to aid
    in deciding whether further condensation is needed.
    """

    events: list[LLMConvertibleEvent]

    unhandled_condensation_request: bool = False
    """Whether there is an unhandled condensation request in the view."""

    condensations: list[Condensation] = []
    """A list of condensations that were processed to produce the view."""

    def __len__(self) -> int:
        return len(self.events)

    @property
    def most_recent_condensation(self) -> Condensation | None:
        """Return the most recent condensation, or None if no condensations exist."""
        return self.condensations[-1] if self.condensations else None

    @property
    def summary_event_index(self) -> int | None:
        """Return the index of the summary event, or None if no summary exists."""
        recent_condensation = self.most_recent_condensation
        if (
            recent_condensation is not None
            and recent_condensation.summary is not None
            and recent_condensation.summary_offset is not None
        ):
            return recent_condensation.summary_offset
        return None

    @property
    def summary_event(self) -> CondensationSummaryEvent | None:
        """Return the summary event, or None if no summary exists."""
        if self.summary_event_index is not None:
            event = self.events[self.summary_event_index]
            if isinstance(event, CondensationSummaryEvent):
                return event
        return None

    # To preserve list-like indexing, we ideally support slicing and position-based
    # indexing. The only challenge with that is switching the return type based on the
    # input type -- we can mark the different signatures for MyPy with `@overload`
    # decorators.

    @overload
    def __getitem__(self, key: slice) -> list[LLMConvertibleEvent]: ...

    @overload
    def __getitem__(self, key: int) -> LLMConvertibleEvent: ...

    def __getitem__(
        self, key: int | slice
    ) -> LLMConvertibleEvent | list[LLMConvertibleEvent]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.events[key]
        else:
            raise ValueError(f"Invalid key type: {type(key)}")

    @staticmethod
    def _enforce_batch_atomicity(
        events: Sequence[Event],
        forgotten_event_ids: set[EventID],
    ) -> set[EventID]:
        """Ensure that if any event in a batch is forgotten, all events in that
        batch are forgotten.

        This prevents partial batches from being sent to the LLM, which can cause
        API errors when thinking blocks are separated from their tool calls.
        """
        batches: dict[EventID, list[EventID]] = {}
        for event in events:
            if isinstance(event, ActionEvent):
                llm_response_id = event.llm_response_id
                if llm_response_id not in batches:
                    batches[llm_response_id] = []
                batches[llm_response_id].append(event.id)

        updated_forgotten_ids = set(forgotten_event_ids)

        for llm_response_id, batch_event_ids in batches.items():
            # Check if any event in this batch is being forgotten
            if any(event_id in forgotten_event_ids for event_id in batch_event_ids):
                # If so, forget all events in this batch
                updated_forgotten_ids.update(batch_event_ids)
                logger.debug(
                    f"Enforcing batch atomicity: forgetting entire batch "
                    f"with llm_response_id={llm_response_id} "
                    f"({len(batch_event_ids)} events)"
                )

        return updated_forgotten_ids

    @staticmethod
    def _ensure_thinking_block_preservation(
        events: Sequence[Event],
        forgotten_event_ids: set[EventID],
    ) -> set[EventID]:
        """Ensure that at least one ActionEvent with thinking blocks is kept.

        When Claude's extended thinking mode is enabled, the conversation must
        have at least one assistant message with thinking blocks. This method
        ensures that if all kept events would lack thinking blocks, we preserve
        at least one event with thinking blocks.

        This is only applied when there are kept ActionEvents - if all ActionEvents
        are forgotten, then there's no need to preserve thinking blocks.

        Args:
            events: All events in the conversation
            forgotten_event_ids: Set of event IDs to be forgotten

        Returns:
            Updated set of forgotten event IDs (with some removed to keep thinking
            blocks)
        """
        # First check if there are any kept ActionEvents
        kept_action_events = [
            e
            for e in events
            if isinstance(e, ActionEvent) and e.id not in forgotten_event_ids
        ]

        if not kept_action_events:
            # No kept ActionEvents - no need to preserve thinking blocks
            logger.debug("No kept ActionEvents - skipping thinking block preservation")
            return forgotten_event_ids

        # Find all ActionEvents with thinking blocks
        events_with_thinking: list[tuple[int, ActionEvent]] = []
        for idx, event in enumerate(events):
            if (
                isinstance(event, ActionEvent)
                and event.thinking_blocks
                and len(event.thinking_blocks) > 0
            ):
                events_with_thinking.append((idx, event))

        if not events_with_thinking:
            # No thinking blocks in the conversation at all
            logger.debug("No events with thinking blocks found")
            return forgotten_event_ids

        logger.debug(
            f"Found {len(events_with_thinking)} events with thinking blocks: "
            f"{[e.id for _, e in events_with_thinking]}"
        )

        # Check if any kept events have thinking blocks
        has_kept_thinking_blocks = any(
            event.id not in forgotten_event_ids for _, event in events_with_thinking
        )

        if has_kept_thinking_blocks:
            # At least one event with thinking blocks is already kept
            logger.debug("At least one event with thinking blocks is already kept")
            return forgotten_event_ids

        logger.debug(
            "No kept events have thinking blocks - need to preserve at least one"
        )

        # No kept events have thinking blocks - we need to preserve at least one
        # Find the most recent event with thinking blocks that's being forgotten
        forgotten_with_thinking = [
            (idx, event)
            for idx, event in events_with_thinking
            if event.id in forgotten_event_ids
        ]

        if not forgotten_with_thinking:
            # This shouldn't happen (we checked there are thinking blocks above)
            logger.debug(
                "No forgotten events with thinking blocks (this should not happen)"
            )
            return forgotten_event_ids

        # Keep the most recent event with thinking blocks
        # Also keep its entire batch (all events with same llm_response_id)
        _, event_to_keep = forgotten_with_thinking[-1]
        logger.debug(
            f"Preserving event {event_to_keep.id} with thinking blocks "
            f"(llm_response_id={event_to_keep.llm_response_id})"
        )

        # Find all events in the same batch AND their tool result observations
        batch_event_ids = []
        for e in events:
            if isinstance(e, ActionEvent) and (
                e.llm_response_id == event_to_keep.llm_response_id
            ):
                batch_event_ids.append(e.id)

        # Also find observations that match the tool_call_ids in this batch
        tool_call_ids_in_batch = [
            e.tool_call_id
            for e in events
            if isinstance(e, ActionEvent)
            and e.llm_response_id == event_to_keep.llm_response_id
            and e.tool_call_id is not None
        ]

        for e in events:
            from openhands.sdk.event.llm_convertible import ObservationBaseEvent

            if (
                isinstance(e, ObservationBaseEvent)
                and e.tool_call_id in tool_call_ids_in_batch
            ):
                batch_event_ids.append(e.id)

        updated_forgotten_ids = set(forgotten_event_ids)
        for event_id in batch_event_ids:
            if event_id in updated_forgotten_ids:
                updated_forgotten_ids.remove(event_id)
                logger.debug(
                    f"Removing {event_id} from forgotten set to preserve "
                    f"thinking blocks"
                )

        return updated_forgotten_ids

    @staticmethod
    def filter_unmatched_tool_calls(
        events: list[LLMConvertibleEvent],
    ) -> list[LLMConvertibleEvent]:
        """Filter out unmatched tool call events.

        Removes ActionEvents and ObservationEvents that have tool_call_ids
        but don't have matching pairs.
        """
        action_tool_call_ids = View._get_action_tool_call_ids(events)
        observation_tool_call_ids = View._get_observation_tool_call_ids(events)

        return [
            event
            for event in events
            if View._should_keep_event(
                event, action_tool_call_ids, observation_tool_call_ids
            )
        ]

    @staticmethod
    def _get_action_tool_call_ids(events: list[LLMConvertibleEvent]) -> set[ToolCallID]:
        """Extract tool_call_ids from ActionEvents."""
        tool_call_ids = set()
        for event in events:
            if isinstance(event, ActionEvent) and event.tool_call_id is not None:
                tool_call_ids.add(event.tool_call_id)
        return tool_call_ids

    @staticmethod
    def _get_observation_tool_call_ids(
        events: list[LLMConvertibleEvent],
    ) -> set[ToolCallID]:
        """Extract tool_call_ids from ObservationEvents."""
        tool_call_ids = set()
        for event in events:
            if (
                isinstance(event, ObservationBaseEvent)
                and event.tool_call_id is not None
            ):
                tool_call_ids.add(event.tool_call_id)
        return tool_call_ids

    @staticmethod
    def _should_keep_event(
        event: LLMConvertibleEvent,
        action_tool_call_ids: set[ToolCallID],
        observation_tool_call_ids: set[ToolCallID],
    ) -> bool:
        """Determine if an event should be kept based on tool call matching."""
        if isinstance(event, ObservationBaseEvent):
            return event.tool_call_id in action_tool_call_ids
        elif isinstance(event, ActionEvent):
            return event.tool_call_id in observation_tool_call_ids
        else:
            return True

    @staticmethod
    def from_events(events: Sequence[Event]) -> "View":
        """Create a view from a list of events, respecting the semantics of any
        condensation events.
        """
        forgotten_event_ids: set[EventID] = set()
        condensations: list[Condensation] = []
        for event in events:
            if isinstance(event, Condensation):
                condensations.append(event)
                forgotten_event_ids.update(event.forgotten_event_ids)
                # Make sure we also forget the condensation action itself
                forgotten_event_ids.add(event.id)
            if isinstance(event, CondensationRequest):
                forgotten_event_ids.add(event.id)

        # Enforce batch atomicity: if any event in a multi-action batch is forgotten,
        # forget all events in that batch to prevent partial batches with thinking
        # blocks separated from their tool calls
        forgotten_event_ids = View._enforce_batch_atomicity(events, forgotten_event_ids)

        # Ensure at least one event with thinking blocks is preserved
        # This is required for Claude's extended thinking mode
        forgotten_event_ids = View._ensure_thinking_block_preservation(
            events, forgotten_event_ids
        )

        kept_events = [
            event
            for event in events
            if event.id not in forgotten_event_ids
            and isinstance(event, LLMConvertibleEvent)
        ]

        # If we have a summary, insert it at the specified offset.
        summary: str | None = None
        summary_offset: int | None = None

        # The relevant summary is always in the last condensation event (i.e., the most
        # recent one).
        for event in reversed(events):
            if isinstance(event, Condensation):
                if event.summary is not None and event.summary_offset is not None:
                    summary = event.summary
                    summary_offset = event.summary_offset
                    break

        if summary is not None and summary_offset is not None:
            logger.debug(f"Inserting summary at offset {summary_offset}")

            _new_summary_event = CondensationSummaryEvent(summary=summary)
            kept_events.insert(summary_offset, _new_summary_event)

        # Check for an unhandled condensation request -- these are events closer to the
        # end of the list than any condensation action.
        unhandled_condensation_request = False
        for event in reversed(events):
            if isinstance(event, Condensation):
                break
            if isinstance(event, CondensationRequest):
                unhandled_condensation_request = True
                break

        return View(
            events=View.filter_unmatched_tool_calls(kept_events),
            unhandled_condensation_request=unhandled_condensation_request,
            condensations=condensations,
        )
