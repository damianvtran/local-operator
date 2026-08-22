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

Ask prompts are answerable from the phone. When the TUI mounts an ``ask``
picker, the app calls :meth:`note_ask_pending`, which projects the first
question as a ``kind="ask"`` :class:`PendingRequest` (mirroring the daemon's
:mod:`owned` ask gate); the phone renders the card and can answer it via the
``ask_answer`` control op. :meth:`ask_answer` resolves the LIVE picker through
its own ``settle`` path on the Textual loop, so the terminal screen comes down
too and exactly one answer wins whichever front end got there first. When the
picker settles by any route, :meth:`note_ask_settled` clears the phone card.

Approvals on a TUI session are NOT answerable from the phone: the TUI boots in
auto-approve, so it never parks an approval card the phone would need to reach
(:meth:`approval_answer` stays a terminal-only stub). Answering approvals over
mobile is a daemon-owned-session capability today (see :mod:`owned`).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import TYPE_CHECKING, Any, Callable

from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.registrant import SessionHandle, image_blocks
from local_operator.mobile.types import (
    SessionProjection,
    ask_pending_request,
)

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
        # request_id -> the live AskPickerScreen for every ask picker this
        # handle has projected to the phone. Keyed by a token_hex request id
        # (owned.py's scheme) because the phone answers by request id and never
        # sees the widget. The current question (and thus the answer key) is
        # read LIVE off ``card.question`` on each answer, because a
        # multi-question picker advances between phone answers (U1) — a stashed
        # id would resolve every answer under Q0's key. ``ask`` is ``exclusive``
        # so this normally holds at most one entry, but a dict keeps pop-by-id
        # honest. Mutated ONLY on the Textual loop (mount/settle) and read there
        # too (``ask_answer`` hops onto it before touching this), so the
        # single-winner race is decided by one owner, not by dict atomicity.
        self._ask_pending: dict[str, Any] = {}

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
        # A /new or /resume mid-ask abandons the old picker; drop its mapping
        # so a late phone answer for a question that no longer exists reports
        # "no longer waiting" instead of settling into the new conversation.
        self._ask_pending.clear()
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
        # A TUI session boots in auto-approve (OperatorApp._load_approvals_default),
        # so it never parks an approval card the phone could reach. Answering
        # approvals over mobile is a daemon-owned-session capability (owned.py);
        # keep the honest terminal-only stub rather than pretend otherwise.
        raise ValueError("this approval is on the terminal — answer it there")

    async def ask_answer(self, request_id: str, value: str) -> str:
        """Answer the CURRENT question of a live TUI ask picker from the phone.

        Called on the registrant/daemon loop, so the whole resolve hops ONTO
        the Textual loop via :meth:`_on_app`: the pending-map read, the
        single-winner guard, and the picker advance/settle all run there,
        against the one owner that also mounts and settles the card. That is
        what makes the race safe without locking — by the time this callback
        runs, either the picker is still live (answer it) or the terminal
        already answered (``settled`` is set / the entry is gone).

        Multi-question asks advance question-by-question (U1). A phone answer
        drives the picker's own :meth:`AskPickerScreen.answer_current`, the
        external-answer twin of the terminal's Enter: it records this question's
        answer and either advances the SAME picker to the next question — in
        which case we RE-PROJECT the new current question so the phone shows it
        — or, on the last question, settles the whole card. Settling resolves
        the very future ``request_user_choice`` awaits, so the terminal screen
        also comes down and the tool call returns the full answer map.

        Race message (U4): if the terminal (or a stop/teardown) already settled
        this card, report that it was "already answered on the terminal" — a
        human, reconciling message rather than the developer-worded "no longer
        waiting", so the phone user learns a different answer won rather than
        seeing the card silently vanish.
        """

        def resolve() -> str:
            card = self._ask_pending.get(request_id)
            if card is None:
                raise ValueError("that question was already answered on the terminal")
            # The terminal may have settled this card in the window before its
            # unmount cleared our mapping. ``settle`` is idempotent, but a
            # second answer must not report success for a choice the terminal
            # actually made — refuse and let the phone show the real outcome.
            if getattr(card, "settled", False):
                raise ValueError("that question was already answered on the terminal")
            # answer_current takes the chosen text for the CURRENT question:
            # for options that is the tapped label, for free-text/secret the
            # typed value. An empty value means "nothing chosen" (settles with
            # None on Q0, keeps partials past it) — parity with owned.py.
            settled = card.answer_current([value] if value else [])
            if settled:
                return "answered"
            # The picker advanced to the next question; project it so the phone
            # follows the terminal to Q2..Qn instead of thinking it is done.
            self._project_ask_question(request_id, card)
            return "answered"

        return await self._on_app(resolve)

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

    # -- ask-picker mirroring ----------------------------------------------------

    def note_ask_pending(self, card: Any) -> None:
        """Project a freshly mounted TUI ask picker to the phone as an
        answerable card.

        Called on the Textual loop from ``OperatorApp.request_user_choice``
        immediately after the picker mounts, so the card sits on its first
        question. Touching ``_fold`` here is safe: fold mutations are
        synchronous and only ``_notify`` crosses threads (the registrant
        coalesces the push onto its own loop).

        A ``token_hex`` request id (owned.py's scheme); the card's CURRENT
        question is what gets projected — with its options + descriptions (U3),
        the ``secret`` flag (D1/U2, never the value), and the question position
        for the "N of M" header (U1). A multi-question ask re-projects its next
        question from :meth:`ask_answer` as the picker advances.
        """
        # Public property, not the private ``_questions`` list (UX minor-2): at
        # mount ``card.question`` IS the first question, and reading through the
        # property keeps this bridge off the widget's internals.
        if getattr(card, "question", None) is None:
            return
        request_id = secrets.token_hex(8)
        self._ask_pending[request_id] = card
        # Parity with owned.py's gate: log when more than one question rides a
        # single card, so the operator can see a multi-part ask went to the
        # phone (UX minor-1).
        total = self._question_total(card)
        if total > 1:
            logger.info("mobile tui ask: %d questions, projecting question-by-question", total)
        self._push_current_question(request_id, card)
        if self._on_projection is not None:
            self._on_projection()

    def _project_ask_question(self, request_id: str, card: Any) -> None:
        """Re-project the picker's now-current question after it advanced.

        The push model is snapshot/repaint, so replacing the pending card with
        the next question is all it takes for the phone to follow the terminal
        from Q1 to Q2..Qn (U1). Same-id push (``set_pending`` semantics via
        pop+push) so the card updates in place rather than stacking."""
        self._fold.pop_pending(request_id)
        self._push_current_question(request_id, card)
        if self._on_projection is not None:
            self._on_projection()

    def _push_current_question(self, request_id: str, card: Any) -> None:
        """Push the card's CURRENT question onto the fold as the pending ask.

        The one construction seam, shared by the mount and the advance, built
        through :func:`ask_pending_request` so the TUI and owned projections
        cannot drift."""
        self._fold.push_pending(
            ask_pending_request(
                request_id,
                card.question,
                question_index=self._question_index(card),
                question_total=self._question_total(card),
            )
        )

    @staticmethod
    def _question_index(card: Any) -> int:
        return int(getattr(card, "question_index", 0) or 0)

    @staticmethod
    def _question_total(card: Any) -> int:
        # ``_questions`` is the only source of the count; the public surface
        # exposes the current index but not the total. Fall back to 1 so a
        # duck-typed test stand-in still projects a coherent single-question
        # header.
        questions = getattr(card, "_questions", None)
        return len(questions) if questions else 1

    def note_ask_settled(self, card: Any) -> None:
        """Clear the phone card for a TUI ask picker that just came down.

        Called on the Textual loop from ``request_user_choice``'s finally block,
        so it covers EVERY settle route — a terminal answer, Escape, a
        cancelled tool call, and teardown — with one seam. Card-scoped and
        idempotent: it pops exactly the request this card was projected under,
        so a settle can never clear a sibling and a second call is a no-op.
        """
        request_id = self._request_id_for_card(card)
        if request_id is None:
            return
        self._ask_pending.pop(request_id, None)
        self._fold.pop_pending(request_id)
        if self._on_projection is not None:
            self._on_projection()

    def _request_id_for_card(self, card: Any) -> str | None:
        for request_id, pending_card in self._ask_pending.items():
            if pending_card is card:
                return request_id
        return None

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
            streaming=bool(getattr(session, "is_streaming", False)),
        )

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
