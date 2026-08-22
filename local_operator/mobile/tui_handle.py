"""The TUI's side of the mobile bridge.

:class:`TuiSessionHandle` adapts a running :class:`~local_operator.tui.app.OperatorApp`
to the registrant's :class:`~local_operator.mobile.registrant.SessionHandle`
contract, so a phone can drive the same session the terminal is showing.

Two rules shape everything here:

- **Textual owns its thread.** Every mutation of app state goes through
  ``app.call_from_thread`` (the registrant's methods run on its own loop);
  reads of plain Python session state are safe directly.
- **The phone is a second front end, not a second session.** Prompts,
  interrupts, model switches and slash commands route through the app's own
  code paths (``_submit_prompt``, ``_interrupt``, ``_run_slash_command`` …)
  so the terminal screen reflects everything the phone did — the user walking
  back to their desk sees the turn they started from the phone, mid-stream.

Approval/ask prompts on the phone are v1-simple: when the TUI's own card is
up, the projection carries it (via :meth:`note_pending`) so the phone shows
"waiting in terminal"; phone-answering a TUI-mounted card is a later round,
because racing two front ends over one modal needs a resolution protocol the
TUI's card does not yet have. Daemon-owned sessions already answer from the
phone; the terminal-first session degrades to "see it, go to the desk".
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.registrant import SessionHandle, image_blocks
from local_operator.mobile.types import PendingRequest, SessionProjection

if TYPE_CHECKING:
    from local_operator.tui.app import OperatorApp

logger = logging.getLogger(__name__)


# Decode wire images via the shared mobile-contract helper (registrant.py);
# kept as a module alias so existing call sites stay short.
_image_blocks = image_blocks


class TuiSessionHandle(SessionHandle):
    def __init__(self, app: "OperatorApp") -> None:
        self._app = app
        session = app._session
        if session is None:
            raise RuntimeError("TUI session has not finished starting")
        self._projection = SessionProjection(
            session_id=session.session_id,
            pid=0,
            kind="tui",
            conversation_name=getattr(session, "conversation_name", "") or "",
            cwd=_session_cwd(session),
            model_label=session.model_label,
            model_selector=_selector(session),
            effort=_current_effort(session),
            effort_ladder=_ladder(session),
        )
        self._fold = ProjectionFold(self._projection)
        self._on_projection: Callable[[], None] | None = None
        self._unsubscribe: Callable[[], None] | None = None

    def _session(self) -> Any:
        """The app's current session. A property method (not cached) because
        /new, /resume and /reload REPLACE the session object — the phone must
        follow the rebind."""
        session = self._app._session
        if session is None:
            raise RuntimeError("session is still starting")
        return session

    def rebind(self) -> None:
        """Re-point the bridge at the app's NEW session after /new, /resume
        or /reload: re-subscribe the fold and reset the projection so no row
        of the old conversation leaks into the new one's phone view."""
        session = self._session()
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("mobile unsubscribe failed", exc_info=True)
            self._unsubscribe = None
        self._projection.session_id = session.session_id
        self._projection.transcript.clear()
        self._projection.todos.clear()
        self._projection.subagents.clear()
        self._projection.pending = None
        self._fold = ProjectionFold(self._projection)
        if self._on_projection is not None:
            self.subscribe(self._on_projection)

    # -- SessionHandle -----------------------------------------------------------

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection: Callable[[], None]) -> Callable[[], None]:
        self._on_projection = on_projection
        session = self._session()

        def handler(event: Any) -> None:
            # Events fire on the Textual loop; the fold is synchronous and
            # the notify crosses threads. Wrap defensively: a folding bug
            # must never take down the agent's event feed.
            try:
                self._fold.fold_event(event)
                self._refresh_state()
                self._refresh_todos()
            except Exception:  # noqa: BLE001
                logger.debug("mobile fold failed", exc_info=True)
            if self._on_projection is not None:
                self._on_projection()

        unsubscribe = session.subscribe(handler)
        try:
            self._fold.fold_history(session.history())
        except Exception:  # noqa: BLE001
            logger.debug("mobile history fold failed", exc_info=True)
        # Seed the live flag ONCE at attach: a phone that subscribes mid-turn
        # never witnessed the AgentStartEvent, so the fold alone would start on
        # a stale ``streaming=False``. After this the fold's own lifecycle
        # events (start/end/turn-end) are the sole authority — see
        # ``_reconcile_streaming`` for why per-event reads are poison.
        self._reconcile_streaming()
        self._unsubscribe = unsubscribe
        return unsubscribe

    # -- mutations: every one hops to the Textual thread ---------------------------

    async def prompt(self, text: str, images: list[dict[str, str]] | None = None) -> str:
        image_blocks = _image_blocks(images)

        def submit() -> None:
            self._app._submit_prompt(text, image_blocks, None)

        await self._on_app(submit)
        self._fold.note_user_message(text)
        if self._on_projection is not None:
            self._on_projection()
        return "prompt sent"

    async def steer(self, text: str, images: list[dict[str, str]] | None = None) -> str:
        image_blocks = _image_blocks(images)

        def do_steer() -> None:
            self._session().steer(text, image_blocks)

        await self._on_app(do_steer)
        self._fold.note_user_message(text, steer=True)
        if self._on_projection is not None:
            self._on_projection()
        return "steering queued"

    async def abort(self) -> str:
        await self._on_app(self._app._interrupt)
        return "stopping"

    async def set_model(self, provider: str, model_id: str) -> str:
        def apply() -> None:
            self._app._run_slash_command(f"/model {provider}/{model_id}")

        await self._on_app(apply)
        self._refresh_state()
        return f"model: {self._projection.model_label}"

    async def set_effort(self, effort: str) -> str:
        def apply() -> None:
            self._app._run_slash_command(f"/effort {effort}")

        await self._on_app(apply)
        self._refresh_state()
        return f"effort: {effort}"

    async def slash(self, command: str, args: str) -> str:
        line = f"/{command}" + (f" {args}" if args else "")

        def apply() -> None:
            self._app._run_slash_command(line)

        await self._on_app(apply)
        self._refresh_state()
        return f"ran {line}"

    async def new_conversation(self) -> str:
        def apply() -> None:
            self._app._run_slash_command("/new")

        await self._on_app(apply)
        # /new rebuilds the session; the seed's identity fields change with it.
        await self.refresh()
        return "new conversation"

    async def resume_session(self, session_id: str) -> str:
        def apply() -> None:
            self._app._run_slash_command(f"/resume {session_id}")

        await self._on_app(apply)
        await self.refresh()
        return f"resumed {session_id}"

    async def approval_answer(self, request_id: str, approved: bool, remember: bool) -> str:
        raise ValueError("this approval is on the terminal — answer it there")

    async def ask_answer(self, request_id: str, value: str) -> str:
        raise ValueError("this question is on the terminal — answer it there")

    async def refresh(self) -> None:
        """Re-seed identity fields after /new, /resume, /model, /rename."""
        session = self._session()
        self._projection.session_id = session.session_id
        self._fold.set_state(
            conversation_name=getattr(session, "conversation_name", "") or None,
            cwd=str(_session_cwd(session)),
        )
        self._refresh_state()
        self._refresh_todos()
        # A command may have started or stopped a turn (prompt, abort, /new,
        # /resume). Command boundaries are safe to reconcile from the session
        # flag: no terminal event is mid-flight here, unlike the per-event
        # path. See ``_reconcile_streaming``.
        self._reconcile_streaming()

    # -- pending-prompt mirroring ----------------------------------------------------

    def note_pending(self, pending: PendingRequest | None) -> None:
        """The TUI calls this when its own approval/ask card mounts or
        resolves, so the phone shows the wait (and who must answer it)."""
        self._fold.set_pending(pending)
        if self._on_projection is not None:
            self._on_projection()

    # -- internals ---------------------------------------------------------------------

    async def _on_app(self, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on the Textual thread and await its result, BOUNDED:
        ``call_from_thread`` enqueues, and an app inside a modal's nested pump
        or a blocked handler never runs the callback — an unbounded await here
        would wedge this session's whole serialized dispatch behind one stuck
        command. Ten seconds is generous for a UI hop and turns the wedge
        into an error the phone can show."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def wrapped() -> None:
            try:
                loop.call_soon_threadsafe(_set_unless_done, future, fn(), None)
            except Exception as exc:  # noqa: BLE001 — the error IS the answer
                loop.call_soon_threadsafe(_set_unless_done, future, None, exc)

        self._app.call_from_thread(wrapped)
        return await asyncio.wait_for(future, timeout=10.0)

    def _refresh_state(self) -> None:
        session = self._session()
        self._fold.set_state(
            model_label=session.model_label,
            model_selector=_selector(session),
            effort=_current_effort(session),
            effort_ladder=_ladder(session),
            # Re-read the title on every push: it is generated in the
            # background and lands (session.set_conversation_name) AFTER the
            # projection was first built, so seeding it once at startup leaves
            # the phone on "untitled" forever. Cheap attribute read; the fold
            # already bumps the epoch only when something actually changed.
            conversation_name=getattr(session, "conversation_name", "") or None,
            # NOTE: ``streaming`` is deliberately NOT set here. This runs after
            # EVERY folded event, and the session flips ``is_streaming`` to
            # False only in the turn's ``finally`` block -- AFTER the
            # ``AgentEndEvent`` has already been emitted and folded. Reading
            # the still-True flag on that terminal event overwrote the fold's
            # correct ``streaming=False`` with True, and because the end event
            # is the last event of the turn, no later push ever corrected it:
            # the phone stayed pinned to "in progress" forever. The fold's own
            # lifecycle events are authoritative for ``streaming``;
            # ``_reconcile_streaming`` covers attach and command boundaries.
        )

    def _reconcile_streaming(self) -> None:
        """Align ``streaming`` with the session's ``is_streaming`` flag at the
        two moments the fold cannot be trusted on its own: initial attach (the
        AgentStartEvent may predate the subscription) and command boundaries
        (a prompt/abort/new/resume just changed turn state).

        Crucially this is NOT called from the per-event handler. There the
        terminal ``AgentEndEvent`` fires while ``is_streaming`` is still True
        (the flag clears in the turn's ``finally``), so a reconcile there would
        re-stick the projection to True with no later event to fix it -- the
        exact bug this method's absence from the hot path prevents.
        """
        try:
            session = self._session()
        except RuntimeError:
            return
        self._fold.set_state(streaming=bool(getattr(session, "is_streaming", False)))

    def _refresh_todos(self) -> None:
        try:
            from local_operator.tools.builtin import TODO_STORE

            self._fold.set_todos(list(TODO_STORE.get(self._session().session_id, [])))
        except Exception:  # noqa: BLE001
            logger.debug("mobile todo refresh failed", exc_info=True)


def _set_unless_done(
    future: "asyncio.Future[Any]", value: Any, error: BaseException | None
) -> None:
    if future.done():
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(value)


def _session_cwd(session: Any) -> str:
    for attr in ("cwd", "working_directory", "current_working_directory"):
        value = getattr(session, attr, None)
        if value:
            return str(value)
    # The session keeps its cwd on the tool context.
    context = getattr(session, "context", None)
    if context is not None and getattr(context, "cwd", None):
        return str(context.cwd)
    import os

    return os.getcwd()


def _selector(session: Any) -> str:
    try:
        spec = session.model
        return f"{spec.provider}/{spec.model_id}"
    except Exception:  # noqa: BLE001
        return ""


def _current_effort(session: Any) -> str:
    try:
        return session.model.reasoning_effort or ""
    except Exception:  # noqa: BLE001
        return ""


def _ladder(session: Any) -> list[str]:
    try:
        return list(session.model.reasoning_efforts)
    except Exception:  # noqa: BLE001
        return []
