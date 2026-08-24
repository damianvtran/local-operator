"""Owner-loop command identity reservations for mobile continuation input."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

MAX_PENDING_STEERS = 32

_CommandState = Literal["prompt", "steer", "prompt-transfer"]
_CommandKind = Literal["prompt", "steer"]


class CommandReservations:
    """Bounded undurable identities, mutated only on the session owner's loop.

    The transcript's append-only index is the lifetime authority.  This map
    closes only the pre-append gap, so durable callbacks remove entries without
    TTLs or eviction that could make an accepted producer identity reusable.
    """

    def __init__(self, session: Any) -> None:
        has_admitted = getattr(session, "has_admitted_command", None)
        if callable(has_admitted):
            self._has_admitted: Callable[[str], bool] = lambda command_id: bool(
                has_admitted(command_id)
            )
        else:
            # Compatibility for test/third-party protocol implementations that
            # predate the durable seam. Production Session always takes the
            # transcript-indexed branch above.
            self._has_admitted = lambda command_id: any(
                getattr(message, "id", None) == command_id for message in session.history()
            )
        self._session = session
        self._commands: dict[str, _CommandState] = {}
        self._pending_steers = 0

    def subscribe_durable(self) -> Callable[[], None]:
        subscribe = getattr(self._session, "subscribe_admitted_commands", None)
        if callable(subscribe):
            unsubscribe = subscribe(self.mark_durable)
            if callable(unsubscribe):

                def stop() -> None:
                    unsubscribe()

                return stop
        return lambda: None

    def reserve(
        self,
        command_id: str,
        *,
        kind: _CommandKind,
        prompt_transfer: bool = False,
    ) -> bool:
        if self._has_admitted(command_id):
            self.mark_durable(command_id)
            return False
        state = self._commands.get(command_id)
        if prompt_transfer and state == "prompt-transfer":
            self._reserve_steer_capacity()
            self._commands[command_id] = "steer"
            self._pending_steers += 1
            return True
        if state is not None:
            return False
        if kind == "steer":
            self._reserve_steer_capacity()
            self._pending_steers += 1
        self._commands[command_id] = kind
        return True

    def _reserve_steer_capacity(self) -> None:
        if self._pending_steers >= MAX_PENDING_STEERS:
            raise RuntimeError(
                f"steering queue is full ({MAX_PENDING_STEERS}); "
                "wait for a queued steer to be delivered"
            )

    def accept(self, command_id: str) -> None:
        # Prompt ACK runs after the append callback, so it must not recreate an
        # entry already handed to the durable ledger. Steers remain until drain.
        if self._has_admitted(command_id):
            self.mark_durable(command_id)

    def reject(self, command_id: str, *, transfer_to_steer: bool = False) -> None:
        state = self._commands.get(command_id)
        if transfer_to_steer:
            if state == "steer":
                self._pending_steers -= 1
            self._commands[command_id] = "prompt-transfer"
        else:
            self._remove(command_id)

    def mark_durable(self, command_id: str) -> None:
        """Hand one reservation to the transcript-backed lifetime ledger."""
        self._remove(command_id)

    def _remove(self, command_id: str) -> None:
        if self._commands.pop(command_id, None) == "steer":
            self._pending_steers -= 1

    def clear(self) -> None:
        self._commands.clear()
        self._pending_steers = 0
