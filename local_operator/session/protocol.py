"""Session public API contract.

Implemented by ``local_operator.session.session.Session`` (stream A) and
programmed against by the TUI (stream D), exec mode (stream E), and the
server facade (integration). Keeping the surface as a Protocol lets the UI
and headless modes build and test against a fake session before the real one
lands.

Event delivery semantics: ``subscribe`` handlers receive
``AgentEvent`` instances in emission order; a handler may be sync or async.
``agent_end`` may arrive AFTER a subsequent ``agent_start`` when a turn was
superseded — UIs must handle that (see docs/REWRITE.md, stream D).
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol, runtime_checkable

from local_operator.harness.types import EventHandler, Message, ModelSpec


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

    @property
    def model(self) -> ModelSpec:
        """The active spec (provider/model_id/base_url/context_window)."""
        ...

    def set_model(self, model: ModelSpec) -> None:
        """Swap the model spec; takes effect from the next turn onward.

        The TUI's ``/model <provider>/<id>`` path calls this after building a
        new spec — the loop reads the spec fresh on every turn, so no session
        teardown is required. Also changes compaction thresholds for the new
        context window.
        """
        ...

    @property
    def goal(self) -> str:
        """The session's standing objective ("" when unset)."""
        ...

    def set_goal(self, text: str) -> str:
        """Set or clear the standing objective; returns what was stored.

        Backs ``/goal``. The objective rides the system prompt's volatile
        tail, so it applies from the next turn onward.
        """
        ...

    @property
    def conversation_name(self) -> str:
        """The conversation's title ("" until one is set or generated)."""
        ...

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        """Name the conversation; returns the title in force afterwards.

        ``user_set=True`` marks an explicit rename, which a later
        auto-generated title must not overwrite.
        """
        ...

    async def complete_once(self, system: str, prompt: str) -> str:
        """One non-tool provider call for host-side side errands.

        Not a turn: no tools, no history, no transcript entry. Hosts use it
        for small derived text (conversation auto-naming) without rebuilding
        the provider's auth cascade. Never await it on a turn's critical path.
        """
        ...

    def history(self) -> list[Message]:
        """The conversation as replayed into LLM context.

        Read-only for RENDERING (a resumed session's transcript back on
        screen): returns the messages the loop sees, in order — user prompts,
        assistant replies, tool results. A front end mounts them as blocks;
        it must NOT mutate them. Empty before the first prompt on a fresh
        session; on ``--resume`` it carries the prior conversation.
        """
        ...

    # --- driving turns ----------------------------------------------------
    async def prompt(self, text: str) -> None:
        """Run one user turn to completion (awaitable) or raise."""
        ...

    async def seed_history(self, messages: list[Message]) -> None:
        """Prime the conversation from a host-supplied history.

        Once-only and pre-prompt: a no-op once the context carries messages
        (transcript replay populated them) or after the first turn. The server
        facade needs it for the two paths where the transcript is not the
        history source — stateless chat and non-persisted agent chat — so the
        provider sees the same history the response envelope echoes.
        """
        ...

    def steer(self, text: str) -> None:
        """Inject a steering message into the running turn (interrupts tool
        batches at the next boundary)."""
        ...

    def abort(self, reason: str = "interrupted") -> None:
        """Abort the running turn; the engine emits an aborted agent_end."""
        ...

    def set_approval_handler(self, handler: Callable[[str, str], Awaitable[bool]] | None) -> None:
        """Replace the host's tool-approval gate for write/exec tier tools.

        A front end that OWNS the terminal must own approvals with it: the
        default gate reads a y/N answer off stdin, which a full-screen UI has
        taken over, so leaving it installed hangs the turn instead of asking
        anyone. The handler is read when the per-turn tool context is built, so
        installing one mid-session applies from the next tool call. ``None``
        restores auto-approval (what ``--yolo`` already does).
        """
        ...

    # --- events -----------------------------------------------------------
    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        """Register an event handler; returns an unsubscribe callable."""
        ...

    # --- lifecycle --------------------------------------------------------
    async def dispose(self) -> None: ...
