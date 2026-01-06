"""Property for ensuring ActionEvent batches remain atomic."""

from collections import defaultdict
from collections.abc import Sequence

from openhands.sdk.context.view.manipulation_indices import ManipulationIndices
from openhands.sdk.context.view.properties.base import ViewPropertyBase
from openhands.sdk.event.base import Event, LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.action import ActionEvent
from openhands.sdk.event.types import EventID


class BatchAtomicityProperty(ViewPropertyBase):
    """Ensures ActionEvents sharing the same llm_response_id form an atomic unit.

    When an LLM makes a single response containing multiple tool calls, those calls
    are semantically related. If any one is forgotten (e.g., during condensation),
    all must be forgotten together to maintain consistency.
    """

    @staticmethod
    def _build_batches(events: Sequence[Event]) -> dict[EventID, list[EventID]]:
        """Build mapping of llm_response_id to ActionEvent IDs.

        Args:
            events: Sequence of events to analyze

        Returns:
            Dictionary mapping llm_response_id to list of ActionEvent IDs
        """
        batches: dict[EventID, list[EventID]] = defaultdict(list)
        for event in events:
            if isinstance(event, ActionEvent):
                batches[event.llm_response_id].append(event.id)
        return dict(batches)

    @staticmethod
    def _build_event_id_to_index(events: Sequence[Event]) -> dict[EventID, int]:
        """Build mapping of event ID to index.

        Args:
            events: Sequence of events to analyze

        Returns:
            Dictionary mapping event ID to index in the list
        """
        return {event.id: idx for idx, event in enumerate(events)}

    def enforce(
        self, current_view_events: list[LLMConvertibleEvent], all_events: list[Event]
    ) -> set[EventID]:
        """Enforce batch atomicity by marking all events in a partially-removed batch.

        If any ActionEvent in a batch is missing, this method will mark all other
        ActionEvents from that batch for removal.

        Args:
            current_view_events: Events currently in the view
            all_events: All events in the conversation

        Returns:
            Set of EventIDs to remove from the current view
        """
        # Build mappings from all events to understand complete batches
        all_batches = self._build_batches(all_events)
        view_batches = self._build_batches(current_view_events)

        events_to_remove: set[EventID] = set()

        # Check each batch in the original events
        for llm_response_id, action_ids in all_batches.items():
            # Get which actions from this batch are in the view
            actions_in_view = view_batches.get(llm_response_id, [])

            # If batch is partially present (some but not all actions)
            if actions_in_view and len(actions_in_view) < len(action_ids):
                # Remove all actions from this batch from the view
                events_to_remove.update(actions_in_view)

        return events_to_remove

    def manipulation_indices(
        self, current_view_events: list[LLMConvertibleEvent], all_events: list[Event]
    ) -> ManipulationIndices:
        """Calculate manipulation indices that respect batch atomicity.

        Returns all indices outside of batch ranges. Within a batch (from min to max
        index), no manipulation is allowed.

        Args:
            current_view_events: Events currently in the view
            all_events: All events in the conversation

        Returns:
            ManipulationIndices with all valid manipulation points
        """
        batches = self._build_batches(current_view_events)
        event_id_to_index = self._build_event_id_to_index(current_view_events)

        # Find atomic ranges for each batch
        atomic_ranges: list[tuple[int, int]] = []

        for llm_response_id, action_ids in batches.items():
            if len(action_ids) > 1:
                # Get indices for all actions in this batch
                indices = [event_id_to_index[aid] for aid in action_ids]
                min_idx = min(indices)
                max_idx = max(indices)
                atomic_ranges.append((min_idx, max_idx))

        # Build set of all valid manipulation indices
        # We can manipulate at any index not inside an atomic range
        valid_indices = set(range(len(current_view_events) + 1))

        # Remove indices that fall within atomic ranges
        for min_idx, max_idx in atomic_ranges:
            # Cannot insert/remove between min_idx and max_idx (exclusive of boundaries)
            for idx in range(min_idx + 1, max_idx + 1):
                valid_indices.discard(idx)

        return ManipulationIndices(valid_indices)
