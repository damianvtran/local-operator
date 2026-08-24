"""Owner-loop command identity reservations for mobile continuation input."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

_CommandState = Literal["pending", "accepted", "prompt-transfer"]


class CommandReservations:
    """Undurable identities, mutated only on the session owner's loop.

    Durable history remains the long-term authority. Reservations close the gap
    before queued steers reach it, then disappear once history proves them, so
    memory tracks the session's pending work rather than the session's lifetime.
    """

    def __init__(self) -> None:
        self._commands: dict[str, _CommandState] = {}

    def reserve(
        self,
        command_id: str,
        history: Iterable[Any],
        *,
        prompt_transfer: bool = False,
    ) -> bool:
        durable_ids = {
            message_id
            for message in history
            if (message_id := getattr(message, "id", None)) is not None
        }
        # Once persistence proves an accepted steer, history itself becomes the
        # reservation. Removing that in-memory copy bounds this set to commands
        # still pending in the session's own queue.
        for durable_id in durable_ids:
            self._commands.pop(durable_id, None)
        if command_id in durable_ids:
            return False
        state = self._commands.get(command_id)
        if prompt_transfer and state == "prompt-transfer":
            # Prompt rejection and steer admission happen in one owner-loop
            # callback, so no competing retry can slip through this handoff.
            self._commands[command_id] = "pending"
            return True
        if state is not None:
            return False
        self._commands[command_id] = "pending"
        return True

    def accept(self, command_id: str) -> None:
        self._commands[command_id] = "accepted"

    def reject(self, command_id: str, *, transfer_to_steer: bool = False) -> None:
        if transfer_to_steer:
            self._commands[command_id] = "prompt-transfer"
        else:
            self._commands.pop(command_id, None)

    def clear(self) -> None:
        self._commands.clear()
