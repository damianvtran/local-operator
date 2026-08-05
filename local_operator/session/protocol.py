"""Session public API contract.

Implemented by ``local_operator.session.session.Session`` (stream A) and
programmed against by the TUI (stream D), exec mode (stream E), and the
server facade (integration). Keeping the surface as a Protocol lets the UI
and headless modes build and test against a fake session before the real one
lands.

Event delivery semantics (ported from omp): ``subscribe`` handlers receive
``AgentEvent`` instances in emission order; a handler may be sync or async.
``agent_end`` may arrive AFTER a subsequent ``agent_start`` when a turn was
superseded — UIs must handle that (see docs/REWRITE.md, stream D).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from local_operator.harness.types import EventHandler


@runtime_checkable
class SessionProtocol(Protocol):
    """The one object every front end talks to."""

    # --- identity / state -------------------------------------------------
    @property
    def session_id(self) -> str: ...

    @property
    def agent_id(self) -> str: ...

    @property
    def is_streaming(self) -> bool: ...

    @property
    def model_label(self) -> str:
        """Human-readable ``provider/model`` for status lines."""
        ...

    # --- driving turns ----------------------------------------------------
    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
        """Run one user turn to completion (awaitable) or raise."""
        ...

    def steer(self, text: str) -> None:
        """Inject a steering message into the running turn (interrupts tool
        batches at the next boundary)."""
        ...

    def abort(self, reason: str = "interrupted") -> None:
        """Abort the running turn; the engine emits an aborted agent_end."""
        ...

    # --- events -----------------------------------------------------------
    def subscribe(self, handler: EventHandler) -> Any:
        """Register an event handler; returns an unsubscribe callable."""
        ...

    # --- lifecycle --------------------------------------------------------
    async def dispose(self) -> None: ...
