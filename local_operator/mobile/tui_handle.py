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

Protocol-v4 full-TUI followers can answer approvals mounted by the owner TUI.
The approval is carried as follower-only pending state (never added to daemon
projection frames, preserving the phone path byte-for-byte) and
:meth:`approval_answer` resolves the owner's real ``ApprovalPrompt`` on the
Textual loop, so exactly one front end wins.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from local_operator.mobile.command_reservation import CommandReservations
from local_operator.mobile.projection import ProjectionFold, fold_messages_to_entries
from local_operator.mobile.registrant import SessionHandle, image_blocks
from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    ask_pending_request,
)

if TYPE_CHECKING:
    from local_operator.tui.app import OperatorApp

logger = logging.getLogger(__name__)


async def _await_future(future: asyncio.Future[Any]) -> Any:
    """Await an owner-loop future from the registrant's bridge coroutine."""
    return await future


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
            model_label=_effective_label(session),
            model_selector=_selector(session),
            effort=_current_effort(session),
            effort_ladder=_ladder(session),
        )
        self._fold = ProjectionFold(self._projection)
        self._on_projection: Callable[[], None] | None = None
        self._unsubscribe: Callable[[], None] | None = None
        # v4 raw-event relay subscription, separate from the projection fold:
        # the registrant serializes/fans events to full-TUI followers while
        # the phone continues receiving only projections. Rebound beside the
        # projection subscription on session swaps.
        self._on_event: Callable[[dict[str, Any]], None] | None = None
        self._unsubscribe_events: Callable[[], None] | None = None
        self._unsubscribe_detail_changes: Callable[[], None] | None = None
        # Mutated only on Textual's loop, making admission atomic even though
        # several registrant coroutines may cross from its socket thread.
        self._command_reservations = CommandReservations(session)
        self._unsubscribe_admitted_commands = self._command_reservations.subscribe_durable()
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
        # Approval pending state is follower-only. The phone path did not
        # receive TUI approvals before protocol v4 and must stay byte-identical;
        # Registrant overlays this field only onto event-client projection
        # frames. Ask pending remains in the ordinary fold because phones
        # already supported TUI asks before this change.
        self._event_pending: PendingRequest | None = None
        # Child transcripts can contain thousands of attachment-backed entries.
        # Keep their I/O off Textual's loop and bound work to one coalescing
        # worker per child so a streaming burst cannot queue stale full reads.
        self._detail_tasks: dict[str, asyncio.Task[None]] = {}
        self._detail_generations: dict[str, int] = {}
        self._detail_fingerprints: dict[str, tuple[int, int]] = {}

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
        if self._unsubscribe_events is not None:
            try:
                self._unsubscribe_events()
            except Exception:  # noqa: BLE001
                logger.debug("mobile event unsubscribe failed", exc_info=True)
            self._unsubscribe_events = None
        if self._unsubscribe_detail_changes is not None:
            self._unsubscribe_detail_changes()
            self._unsubscribe_detail_changes = None
        self._projection.session_id = session.session_id
        self._projection.transcript.clear()
        self._projection.todos.clear()
        self._projection.subagents.clear()
        self._projection.pending = None
        self._event_pending = None
        self._cancel_detail_tasks()
        self._fold = ProjectionFold(self._projection)
        # A /new or /resume mid-ask abandons the old picker; drop its mapping
        # so a late phone answer for a question that no longer exists reports
        # "no longer waiting" instead of settling into the new conversation.
        self._ask_pending.clear()
        self._unsubscribe_admitted_commands()
        self._command_reservations.clear()
        self._command_reservations = CommandReservations(session)
        self._unsubscribe_admitted_commands = self._command_reservations.subscribe_durable()
        if self._on_projection is not None:
            self.subscribe(self._on_projection)
        if self._on_event is not None:
            self.subscribe_events(self._on_event)

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
                job_id = getattr(event, "job_id", None)
                if isinstance(job_id, str):
                    self._invalidate_subagent_detail(job_id)
            except Exception:  # noqa: BLE001
                logger.debug("mobile fold failed", exc_info=True)
            if self._on_projection is not None:
                self._on_projection()

        unsubscribe = session.subscribe(handler)
        comms = getattr(session, "_subagent_comms", None)
        subscribe_details = getattr(comms, "subscribe_detail_changes", None)
        if callable(subscribe_details):
            unsubscribe_details = subscribe_details(self._invalidate_subagent_detail)
            if callable(unsubscribe_details):
                self._unsubscribe_detail_changes = cast(Callable[[], None], unsubscribe_details)
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
        self._refresh_state()
        self._warm_subagent_details()
        self._unsubscribe = unsubscribe
        return unsubscribe

    def subscribe_events(self, on_event: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Feed serialized AgentEvents to the registrant's v4 relay.

        Serialization happens on the Textual/session loop where the event is
        emitted. The callback itself is thread-safe (Registrant._relay_event
        only schedules onto its own loop), so this preserves producer order
        without moving pydantic objects across threads.
        """
        self._on_event = on_event

        def handler(event: Any) -> None:
            try:
                on_event(event.model_dump(mode="json"))
            except Exception:  # noqa: BLE001 — relay is additive, never a gate
                logger.debug("mobile event serialization failed", exc_info=True)

        unsubscribe = self._session().subscribe(handler)
        self._unsubscribe_events = unsubscribe
        return unsubscribe

    # -- mutations: every one hops to the Textual thread ---------------------------

    async def prompt(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        command_id: str | None = None,
    ) -> str:
        if not command_id:
            raise ValueError("command_id is required")
        image_blocks = _image_blocks(images)

        def begin_prompt() -> tuple[asyncio.AbstractEventLoop, asyncio.Future[None], Any] | None:
            session = self._session()
            if not self._command_reservations.reserve(command_id, kind="prompt"):
                return None
            owner_loop = asyncio.get_running_loop()
            admitted: asyncio.Future[None] = owner_loop.create_future()

            async def run_turn() -> None:
                try:
                    fields: dict[str, Any] = {
                        "message_id": command_id,
                        "admitted": admitted,
                    }
                    if "producer_command_id" in inspect.signature(session.prompt).parameters:
                        fields["producer_command_id"] = command_id
                    await session.prompt(text, image_blocks, **fields)
                except BaseException as exc:
                    if not admitted.done():
                        self._command_reservations.reject(
                            command_id,
                            transfer_to_steer="already streaming" in str(exc),
                        )
                        admitted.set_exception(exc)
                    raise

            return owner_loop, admitted, asyncio.run_coroutine_threadsafe(run_turn(), owner_loop)

        started = await self._on_app(begin_prompt)
        if started is None:
            return "already admitted"
        owner_loop, admitted, turn = started
        try:
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(_await_future(admitted), owner_loop)
            )
        except Exception:
            turn.cancel()
            raise
        await self._on_app(lambda: self._command_reservations.accept(command_id))
        return "prompt admitted"

    async def steer(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        command_id: str | None = None,
    ) -> str:
        if not command_id:
            raise ValueError("command_id is required")
        image_blocks = _image_blocks(images)

        def do_steer() -> bool:
            session = self._session()
            if not self._command_reservations.reserve(
                command_id,
                kind="steer",
                prompt_transfer=True,
            ):
                return False
            try:
                fields: dict[str, Any] = {"message_id": command_id}
                if "producer_command_id" in inspect.signature(session.steer).parameters:
                    fields["producer_command_id"] = command_id
                session.steer(text, image_blocks, **fields)
            except Exception:
                self._command_reservations.reject(command_id)
                raise
            self._command_reservations.accept(command_id)
            return True

        if not await self._on_app(do_steer):
            return "already admitted"
        self._fold.note_user_message(text, steer=True)
        if self._on_projection is not None:
            self._on_projection()
        return "steering queued"

    async def receive_peer_message(
        self,
        text: str,
        *,
        mode: str = "mailbox",
        wake: bool = False,
        sender: dict[str, Any] | None = None,
    ) -> str:
        # Session.receive_peer_message is a COROUTINE that must run on the owner
        # event loop (it touches _context.messages, the transcript, and may
        # spawn a turn). `_on_app` only runs SYNC callables on the Textual
        # thread, so we reuse the prompt() machinery: a sync shim scheduled on
        # the app captures the owner loop and schedules the coroutine there with
        # run_coroutine_threadsafe, and we await its result from this bridge
        # coroutine. Do NOT call the coroutine directly off-loop.
        sender = sender or {}

        def schedule() -> "Future[str]":
            session = self._session()
            owner_loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(
                session.receive_peer_message(text, mode=mode, wake=wake, sender=sender),
                owner_loop,
            )

        fut = await self._on_app(schedule)
        detail = await asyncio.wrap_future(fut)
        # Optimistic phone echo, matching steer(): put the card on the
        # projection now so an attached phone paints it without waiting for the
        # next projection repaint.
        self._fold.note_peer_message(text, sender=sender)
        if self._on_projection is not None:
            self._on_projection()
        return str(detail)

    async def recall_steer(self, command_id: str) -> str:
        """Recall one queued steer by the Message id its producer supplied."""

        def do_recall() -> bool:
            session = self._session()
            for message in session.queued_steering():
                if str(getattr(message, "id", "")) == command_id:
                    return bool(session.recall_steering(message))
            return False

        if not await self._on_app(do_recall):
            raise ValueError("that steering message is no longer queued")
        self._command_reservations.reject(command_id)
        return "steering recalled"

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
        """Settle the owner's real ApprovalPrompt from another front end.

        The prompt's ``resolve`` is idempotent and runs on Textual's loop, so
        terminal, phone and follower answers share one arbitration point. A
        stale request id is rejected rather than applied to the next prompt.
        """

        def settle() -> bool:
            prompt = self._app._approval
            if prompt is None or prompt.answered:
                return False
            if getattr(prompt, "_mobile_request_id", "") != request_id:
                return False
            prompt.resolve(approved, answer="y" if approved else "n")
            return True

        if not await self._on_app(settle):
            raise ValueError("that approval is no longer waiting")
        return "approved" if approved else "denied"

    async def ask_answer(
        self, request_id: str, value: str, question_index: int | None = None
    ) -> str:
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

        ``question_index`` is the question the phone was DISPLAYING when the user
        tapped. It is the U8 guard, mirroring the composer path's
        ``_HeldAnswerKey.question_index``: a multi-question picker advances (and
        the phone re-projects) between answers, so a tap already in flight when
        a terminal advance lands must NOT be recorded against the question that
        moved into its place. We refuse when it no longer matches the picker's
        live index rather than misattribute the value; the phone then repaints
        to the current question from the re-projection. ``None`` (an older
        client) skips the check and answers the current question, the pre-guard
        behaviour.

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
            # U8 guard: the phone answered whatever question it was showing, but
            # the terminal may have advanced the card since. Answering against a
            # moved-on question would key the value to the WRONG question, so
            # refuse and let the re-projection repaint the phone to the current
            # one.
            if question_index is not None:
                live_index = int(getattr(card, "question_index", 0) or 0)
                if live_index != question_index:
                    raise ValueError("that question moved on — here is the current one")
            # answer_current takes the chosen text for the CURRENT question:
            # for options that is the tapped label, for free-text/secret the
            # typed value. An empty value means "nothing chosen" (settles with
            # None on Q0, keeps partials past it) — parity with owned.py.
            settled = card.answer_current([value] if value else [])
            if settled:
                return "answered"
            # The picker advanced to the next question; project it so the phone
            # follows to Q2..Qn instead of thinking it is done. (The picker also
            # posts QuestionAdvanced, which re-projects too — this is the
            # immediate, deterministic path; the message is belt-and-suspenders
            # for terminal-driven advances.)
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
        # A command may have started or stopped a turn (prompt, abort, /new,
        # /resume). Command boundaries are safe to reconcile from the session
        # flag: no terminal event is mid-flight here, unlike the per-event
        # path. See ``_reconcile_streaming``.
        self._reconcile_streaming()

    # -- approval / ask mirroring ---------------------------------------------

    def note_approval_pending(self, card: Any) -> None:
        """Project the owner's real approval prompt to v4 followers.

        The request id is stored on the prompt itself because approvals are
        serialized and the prompt is the one arbitration object every answer
        route ultimately resolves. The phone also sees this projection, gaining
        parity rather than a second approval-specific channel.
        """
        request_id = secrets.token_hex(8)
        setattr(card, "_mobile_request_id", request_id)
        self._event_pending = PendingRequest(
            request_id=request_id,
            kind="approval",
            title=str(getattr(card, "tool_name", "") or "tool approval"),
            detail=str(getattr(card, "description", "") or ""),
        )
        if self._on_projection is not None:
            self._on_projection()

    def note_approval_settled(self, card: Any) -> None:
        """Remove exactly this prompt's projected approval, on every exit path."""
        request_id = str(getattr(card, "_mobile_request_id", "") or "")
        if not request_id:
            return
        if self._event_pending is not None and self._event_pending.request_id == request_id:
            self._event_pending = None
        if self._on_projection is not None:
            self._on_projection()

    @property
    def event_pending(self) -> PendingRequest | None:
        """Follower-only gate overlaid by Registrant on event clients."""
        return self._event_pending

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

    def note_ask_advanced(self, card: Any) -> None:
        """Re-project a picker that advanced to its next question, by ANY route.

        Called on the Textual loop from ``OperatorApp`` when the picker posts
        ``QuestionAdvanced`` — a terminal Enter as well as a phone-routed
        answer. Re-projecting on the TERMINAL advance is what closes U8: without
        it the phone kept showing the previous question after the terminal
        moved on, and a tap there resolved against the question the terminal had
        advanced to. Card-scoped and idempotent: a card the handle never
        projected (or one already settled/popped) is a no-op.
        """
        request_id = self._request_id_for_card(card)
        if request_id is None:
            return
        self._project_ask_question(request_id, card)

    def _project_ask_question(self, request_id: str, card: Any) -> None:
        """Re-project the picker's now-current question after it advanced.

        The push model is snapshot/repaint, so replacing the pending card with
        the current question is all it takes for the phone to follow from Q1 to
        Q2..Qn (U1/U8). Same-id push (``set_pending`` semantics via pop+push) so
        the card updates in place rather than stacking, and a mid-ask reconnect
        snapshots the CURRENT question because the fold now holds it."""
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

    def note_title(self, name: str) -> None:
        """Push a title that landed off the event stream.

        Generated titles and provisional stand-ins never emit an AgentEvent,
        so the per-event refresh never sees them. The TUI calls this the
        moment the band updates so the phone's header and list follow.
        """
        label = (name or "").strip()
        self._fold.set_state(conversation_name=label or None)
        if self._on_projection is not None:
            self._on_projection()

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

    def _cancel_detail_tasks(self) -> None:
        for task in self._detail_tasks.values():
            task.cancel()
        self._detail_tasks.clear()
        self._detail_generations.clear()
        self._detail_fingerprints.clear()

    def _warm_subagent_details(self) -> None:
        """Adopt restored descendants without delaying the initial projection."""
        comms = getattr(self._session(), "_subagent_comms", None)
        if comms is None:
            return
        for node in comms.nodes():
            if node.session_dir is not None:
                self._invalidate_subagent_detail(node.job_id)

    def _invalidate_subagent_detail(self, job_id: str) -> None:
        """Coalesce child mutations behind one generation-guarded worker."""
        comms = getattr(self._session(), "_subagent_comms", None)
        node = comms.node(job_id) if comms is not None else None
        if node is None or node.session_dir is None:
            return
        self._detail_generations[job_id] = self._detail_generations.get(job_id, 0) + 1
        task = self._detail_tasks.get(job_id)
        if task is None or task.done():
            self._detail_tasks[job_id] = asyncio.create_task(
                self._hydrate_subagent_detail(job_id),
                name=f"mobile-detail-{job_id}",
            )

    async def _hydrate_subagent_detail(self, job_id: str) -> None:
        """Hydrate only the invalidated child, repeating once if it moved."""
        try:
            while True:
                generation = self._detail_generations.get(job_id, 0)
                comms = getattr(self._session(), "_subagent_comms", None)
                node = comms.node(job_id) if comms is not None else None
                if comms is None or node is None or node.session_dir is None:
                    return
                session_dir = str(node.session_dir)
                try:
                    result = await asyncio.to_thread(_load_subagent_detail, session_dir)
                except _DetailChangedDuringHydration:
                    # Appends and atomic compaction can overlap a worker read.
                    # Retry in this same coalesced task so no later event is
                    # required to recover the newest stable generation.
                    await asyncio.sleep(0)
                    continue
                if generation != self._detail_generations.get(job_id):
                    continue
                current = comms.node(job_id)
                if current is None or str(current.session_dir) != session_dir:
                    return
                fingerprint, transcript, todos = result
                if self._detail_fingerprints.get(session_dir) != fingerprint:
                    self._detail_fingerprints[session_dir] = fingerprint
                    if self._fold.set_subagent_hydrated_details(job_id, transcript, todos):
                        if self._on_projection is not None:
                            self._on_projection()
                return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - child detail is additive
            logger.debug("mobile child-detail hydration failed", exc_info=True)
        finally:
            task = self._detail_tasks.get(job_id)
            if task is asyncio.current_task():
                self._detail_tasks.pop(job_id, None)

    def _refresh_state(self) -> None:
        session = self._session()
        self._fold.set_state(
            model_label=_effective_label(session),
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
        comms = getattr(session, "_subagent_comms", None)
        if comms is not None:
            self._fold.set_subagent_details(comms)

    def _reconcile_streaming(self) -> None:
        """Seed/align ``streaming`` from the session flag at attach and command
        boundaries, delegating the safety rule to ``ProjectionFold`` (which
        owns lifecycle authority and ignores the flag once it has folded a
        turn-terminal event -- see ``ProjectionFold.reconcile_streaming``).

        Deliberately NOT called from the per-event handler: the fold's own
        lifecycle events own ``streaming`` there.
        """
        try:
            session = self._session()
        except RuntimeError:
            return
        self._fold.reconcile_streaming(bool(getattr(session, "is_streaming", False)))

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


class _DetailChangedDuringHydration(RuntimeError):
    """Worker result became stale while its transcript was being read."""


def _load_subagent_detail(
    session_dir: str,
) -> tuple[tuple[int, int], list[Any], list[dict[str, Any]]]:
    """Read one child transcript in a worker and return its stable fingerprint."""
    from local_operator.session.transcript import TRANSCRIPT_FILENAME, Transcript

    path = Path(session_dir) / TRANSCRIPT_FILENAME
    before = path.stat()
    transcript = Transcript(session_dir)
    entries = fold_messages_to_entries(transcript.build_llm_history())
    raw_todos = (transcript.latest_custom("todo_snapshot") or {}).get("items") or []
    after = path.stat()
    # Atomic transcript replacement or an append during hydration invalidates
    # this result; the caller's next child event will request the newer detail.
    fingerprint = (after.st_size, after.st_mtime_ns)
    if (before.st_size, before.st_mtime_ns) != fingerprint:
        raise _DetailChangedDuringHydration("child transcript changed during hydration")
    return fingerprint, entries, raw_todos if isinstance(raw_todos, list) else []


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


def _effective_label(session: Any) -> str:
    """``provider/model`` of the model actually serving requests.

    A display that reads ``session.model_label`` (the selection) during a
    provider fallback names a model that is not answering — the stale
    composer chip the phone showed after a quota failover.
    """
    label = str(getattr(session, "effective_model_label", "") or "")
    return label or str(getattr(session, "model_label", "") or "")


def _selector(session: Any) -> str:
    try:
        spec = session.model
        return f"{spec.provider}/{spec.model_id}"
    except Exception:  # noqa: BLE001
        return ""


def _current_effort(session: Any) -> str:
    try:
        spec = getattr(session, "effective_model", None) or session.model
        return spec.reasoning_effort or ""
    except Exception:  # noqa: BLE001
        return ""


def _ladder(session: Any) -> list[str]:
    try:
        return list(session.model.reasoning_efforts)
    except Exception:  # noqa: BLE001
        return []
