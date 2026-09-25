"""Runtime-loop command identity reservations for mobile continuation input."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

MAX_PENDING_STEERS = 32

#: The two live states an undurable identity can be in. There is deliberately no
#: third "refused, parked for a steer retry" state: a refused command is RELEASED
#: (see :meth:`CommandReservations.reject`), because any state that answers
#: "already admitted" for a message that is in no transcript and no queue is a
#: silent drop — the retry the client makes to recover is swallowed.
_CommandKind = Literal["prompt", "steer"]


class CommandReservations:
    """Bounded undurable identities, mutated only on the session runtime's loop.

    The transcript's append-only index is the lifetime authority.  This map
    closes only the pre-append gap, so durable callbacks remove entries without
    TTLs or eviction that could make an accepted producer identity reusable.

    THE INVARIANT every method keeps: :meth:`reserve` answers False ("already
    admitted") only for an identity that is durably in the transcript or is LIVE
    here — a prompt still on its way to the append, or a steer queued for the next
    boundary. An identity whose attempt failed before admission is released, so
    the producer's retry of the SAME id really admits it.
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
        self._commands: dict[str, _CommandKind] = {}
        self._pending_steers = 0

    def subscribe_durable(self) -> Callable[[], None]:
        subscribe_durable = getattr(self._session, "subscribe_admitted_commands", None)
        unsubscribe_durable = (
            subscribe_durable(self.mark_durable) if callable(subscribe_durable) else None
        )
        subscribe_rejected = getattr(self._session, "subscribe_rejected_steering", None)
        unsubscribe_rejected = (
            subscribe_rejected(self._on_steer_rejected) if callable(subscribe_rejected) else None
        )

        def stop() -> None:
            if callable(unsubscribe_durable):
                unsubscribe_durable()
            if callable(unsubscribe_rejected):
                unsubscribe_rejected()

        return stop

    def _on_steer_rejected(self, command_id: str, _reason: str) -> None:
        """Release the retry identity and its bounded steering capacity."""
        self.reject(command_id)

    def reserve(self, command_id: str, *, kind: _CommandKind) -> bool:
        if self._has_admitted(command_id):
            self.mark_durable(command_id)
            return False
        if command_id in self._commands:
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

    def reject(self, command_id: str) -> None:
        """Release an identity whose attempt was refused before admission.

        ALWAYS a full release, including for a prompt refused because a turn
        held the session (``TurnInFlight``). That refusal used to PARK the id as
        ``prompt-transfer`` so a client could retry it as a steer — but the
        parked state also answered "already admitted" to a retry of the same id
        as a PROMPT, which is exactly what the desktop's receipt journal and the
        phone send after a 5xx. The message was in no transcript and no queue,
        and the client was told it had landed: a silent drop (QA on PR #1528,
        Q1-5; 2 of 19 race runs). A released id still serves the steer retry,
        because a free id reserves as a steer like any other.
        """
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
