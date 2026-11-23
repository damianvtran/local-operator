"""Opt-in admission for an owning session tree's logical model requests."""

from __future__ import annotations

from collections.abc import Callable


class RequestAdmissionRefused(Exception):
    """A host policy declined dispatch, not a provider error or a retry signal."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


# Called synchronously before account selection with the ID that will appear
# in request-final observation. Raise RequestAdmissionRefused to decline, or
# return an optional synchronous cleanup callback (must not raise). Cleanup runs
# even on a local preparation failure or cancellation that yields no final row.
# Neither callback may await or do I/O on the model loop.
RequestAdmission = Callable[[str], Callable[[], None] | None]
