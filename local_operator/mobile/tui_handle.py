"""The TUI's side of the mobile bridge.

:class:`TuiSessionHandle` adapts a running :class:`~local_operator.tui.app.OperatorApp`
to the session runtime's :class:`~local_operator.session.runtime.server.SessionHandle`
contract, so a phone can drive the same session the terminal is showing.

Two rules shape everything here:

- **Textual owns its thread.** Every mutation of app state goes through
  ``app.call_from_thread`` (the registrant's methods run on its own loop);
  reads of plain Python session state are safe directly.
- **The runtime's loop never waits on another thread.** ``call_from_thread``
  ENQUEUES the callback and then blocks until Textual runs it, so calling it
  directly from a coroutine parks the runtime's loop — and with it the accept,
  the welcome, ``ping`` and the heartbeat, all of which that one loop owns —
  for as long as the app is busy. Every hop therefore goes through
  :meth:`TuiSessionHandle._on_app`, which performs the blocking enqueue on a
  worker thread (``asyncio.to_thread``) and awaits the *bound* result
  (:meth:`_on_app` states the measured failure this prevents).
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

Protocol-v4 full-TUI followers can answer approvals mounted by the host TUI.
The approval is carried as follower-only pending state (never added to daemon
projection frames, preserving the phone path byte-for-byte) and
:meth:`approval_answer` resolves the host's real ``ApprovalPrompt`` on the
Textual loop, so exactly one front end wins.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, cast

from local_operator.harness.wire import bound_agent_end_for_wire
from local_operator.mobile.command_reservation import CommandReservations
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    ask_pending_request,
)
from local_operator.session.runtime.server import (
    SessionHandle,
    has_durable_history,
    image_blocks,
    image_blocks_in_thread,
)
from local_operator.session.runtime.types import RUNNING_SUBAGENT_STATUSES

if TYPE_CHECKING:
    from local_operator.tui.app import OperatorApp

logger = logging.getLogger(__name__)

#: How long ONE hop into the Textual loop may take before its caller stops
#: waiting: the blocking enqueue in :meth:`TuiSessionHandle._on_app` plus the
#: callback's own awaited result, as a single budget rather than one per leg.
#:
#: Ten seconds, chosen against the CLIENT's patience rather than by taste:
#: ``attach_client.ACK_TIMEOUT_S`` is 15 s, so a hop that has not answered by
#: ten has to fail as an error the caller can report, not keep the caller's own
#: socket parked until the client gives up first. It is deliberately NOT a
#: bound on the app-loop work the callback performs once it starts — that is
#: the app's own business and the runtime's loop is free by then (see
#: :meth:`_on_app`) — only on how long a caller waits for the answer.
_APP_HOP_TIMEOUT_S = 10.0


async def _await_future(future: asyncio.Future[Any]) -> Any:
    """Await an owner-loop future from the registrant's bridge coroutine."""
    return await future


#: Marks "the caller passed no ``budget``", so the INTERACTIVE budget is read from
#: :data:`_APP_HOP_TIMEOUT_S` at CALL time rather than captured in the signature.
#: The constant is the single knob — callers and tests patch it, and round 1's
#: budget test did exactly that — and a default evaluated at import time would
#: quietly ignore every later change to it.
_BUDGET_FROM_CONSTANT: Any = object()


def _app_hop_timeout_note(budget: float = _APP_HOP_TIMEOUT_S) -> str:
    """The sentence a hop that ran out of its budget reports.

    Names the budget and the two things a reader needs to act on it (the app is
    busy, the budget is the client's patience divided by one and a half) rather
    than a bare ``TimeoutError``, because this message is what reaches a
    follower's terminal through an error frame.

    THE PARAMETER IS NOT USED BY THE EXPIRY PATHS TODAY, and the sentence is
    therefore ALWAYS the interactive budget's. Every raise site goes through
    ``_on_app``'s local ``expired()``, which passes nothing, so a caller that
    runs its own shorter budget (the tests do, and they say so) still reports
    "within 10s". The parameter is kept because it is the knob a caller would
    pass through once a non-interactive budget exists; until then, reading this
    sentence as a report of the CALLER's deadline is wrong (review round 3,
    MINOR 2 / UX U1 / QA Q2).
    """
    return (
        f"the terminal did not answer within {budget:.0f}s "
        "(the app is busy with a turn; retry when it settles)"
    )


# Decode wire images via the shared mobile-contract helper (registrant.py);
# kept as module aliases so existing call sites stay short. The helper BOUNDS
# each image, which is CPU-bound, so async callers take the ``_async`` form and
# the one sync caller (``_decode_attachments``, reached from a Textual callback)
# keeps the direct call — see its docstring for why that stall is bounded.
_image_blocks = image_blocks
_image_blocks_async = image_blocks_in_thread


def _decode_attachments(images: list[dict[str, str]] | None) -> dict[int, Any]:
    """Rebuild the composer's index→attachment map from wire image blocks.

    Shared by ``slash_images`` and the routed-slash path so both hand the
    dispatch the same ``{1: Attachment, …}`` shape the composer would have
    produced had the images been attached locally.

    Synchronous even though the bound it now performs is CPU-bound, because it
    runs inside a Textual callback that cannot await. The stall is bounded by
    the control socket's 1 MB line cap (``_MAX_LINE_BYTES``): a payload that
    large decodes to at most a few images of a few hundred KB, ~50-100 ms of
    work, not the ~315 ms a 20 MP paste would cost.
    """
    from local_operator.tui.widgets.editor import Attachment

    return {
        index: Attachment(image=image, marker=f"[Image #{index}]")
        for index, image in enumerate(_image_blocks(images), start=1)
    }


class TuiSessionHandle(SessionHandle):
    def __init__(self, app: "OperatorApp") -> None:
        self._app = app
        session = app._session
        if session is None:
            raise RuntimeError("TUI session has not finished starting")
        # Same wiring ``ServingSessionHandle.__init__`` performs: the session
        # flips the discovery record's ``started`` bit at the top of every
        # real turn (``Session._run_turn_pipeline``), and the handle owns the
        # publish. Without it a TUI-owned ``kind="tui"`` record would stay
        # ``started=False`` for its whole life — permanently invisible to
        # broadcasts and quietly mailbox-dialled on exact sends even after
        # hundreds of turns. Guarded with ``hasattr`` so reduced hosts (test
        # doubles standing in for a Session) keep constructing; the ignore
        # matches the house pattern for duck-typed hooks
        # (``RuntimeServer``'s ``handle._registrant`` assignment) — the
        # protocol deliberately does not declare per-host hooks.
        if hasattr(session, "_publish_session_started"):
            session._publish_session_started = (  # type: ignore[attr-defined]
                self._publish_session_started
            )
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
        # (serving.py's scheme) because the phone answers by request id and never
        # sees the widget. The current question (and thus the answer key) is
        # read LIVE off ``card.question`` on each answer, because a
        # multi-question picker advances between phone answers (U1) — a stashed
        # id would resolve every answer under Q0's key. ``ask`` is ``exclusive``
        # so this normally holds at most one entry, but a dict keeps pop-by-id
        # honest. Mutated ONLY on the Textual loop (mount/settle) and read there
        # too (``ask_answer`` hops onto it before touching this), so the
        # single-winner race is decided by one loop, not by dict atomicity.
        self._ask_pending: dict[str, Any] = {}
        # Child transcripts can contain thousands of attachment-backed entries.
        # Keep their I/O off Textual's loop and bound work to one coalescing
        # worker per child so a streaming burst cannot queue stale full reads.
        self._detail_tasks: dict[str, asyncio.Task[None]] = {}
        self._detail_generations: dict[str, int] = {}
        self._detail_fingerprints: dict[str, tuple[int, int]] = {}
        #: Awaitable hop results in flight on the app loop (see ``_on_app``):
        #: held so they cannot be collected before they settle the caller.
        self._late_hop_tasks: set[asyncio.Task[Any]] = set()
        # The in-flight app-loop hop of a ``request_stop``, held so the loop
        # cannot collect it before it lands — the receipt is returned WITHOUT
        # waiting for it (see ``_detach_stop_hop``).
        self._stop_hop_task: asyncio.Task[Any] | None = None

    def _session(self) -> Any:
        """The app's current session. A property method (not cached) because
        /new, /resume and /reload REPLACE the session object — the phone must
        follow the rebind."""
        session = self._app._session
        if session is None:
            raise RuntimeError("session is still starting")
        return session

    def subagent_counts(self) -> tuple[int | None, int | None]:
        """``(running, queued)`` subagent trajectories for the record.

        The ``kind="tui"`` twin of ``ServingSessionHandle.subagent_counts``, so a
        full TUI runtime contributes to ``/info``'s fleet tally instead of being
        counted as a session that does not report. Implemented rather than left
        out because this handle already reaches the roster (see
        ``_warm_subagent_details``), and an unimplemented probe would publish
        ``None`` for the most common kind of window on this host.

        Reads the same ``RUNNING_SUBAGENT_STATUSES`` predicate the owned handle
        and ``/info`` use — a private set here would let the record and the tree
        drawn beneath it disagree. The statuses come from
        ``SubagentComms.status_counts()``, one linear walk shared with the owned
        handle's publisher (this loop used to build every ``SubagentNode`` in the
        registry, once per event, to read one string each). ``(None, None)`` on
        any failure, never ``(0, 0)``: this runs before the session finishes
        starting, and a not-yet-readable roster is not a measurement of zero.
        """
        try:
            comms = getattr(self._session(), "subagent_comms", None)
            if comms is None:
                return (None, None)
            counts = comms.status_counts()
        except Exception:  # noqa: BLE001 — a stale count never breaks the app
            logger.debug("could not read the subagent roster", exc_info=True)
            return (None, None)
        running = sum(counts.get(status, 0) for status in RUNNING_SUBAGENT_STATUSES)
        queued = counts.get("queued", 0)
        return (running, queued)

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
        # The registrant outlives the session object, so two pieces of
        # per-session state must follow the swap. First, the started hook:
        # the NEW session object must get it or (on a TUI-owned record) it
        # would never flip. Second, the registrant's ``started`` bit itself,
        # which cannot simply carry over — it describes the OLD conversation:
        # a ``/new`` after a working conversation must drop back to ``False``
        # (the composer window the flag exists for), while a ``/resume`` must
        # read ``True`` because that conversation already ran turns and a
        # peer wake could always reach it. Re-seeded from the NEW session's
        # own durable history so both directions come out right.
        if hasattr(session, "_publish_session_started"):
            session._publish_session_started = self._publish_session_started
        registrant = getattr(self, "_registrant", None)
        reseed = getattr(registrant, "reset_record_started", None)
        if callable(reseed):
            try:
                reseed(has_durable_history(session))
            except Exception:  # noqa: BLE001 — a stale bit must never break /new
                logger.debug("could not re-seed the started bit on rebind", exc_info=True)

    def _publish_session_started(self) -> None:
        """Flip the record's ``started`` bit once this session runs a real turn.

        The ``kind="tui"`` twin of ``ServingSessionHandle``'s hook, called from
        ``Session._run_turn_pipeline`` at the top of every turn; the
        registrant's ``set_record_started`` de-duplicates so only the first
        turn publishes. Defensive in the same shape as the owned hook: a
        reduced host with no registrant, or one predating the setter, is a
        no-op rather than a failed turn.
        """
        registrant = getattr(self, "_registrant", None)
        setter = getattr(registrant, "set_record_started", None)
        if not callable(setter):
            return
        try:
            setter(True)
        except Exception:  # noqa: BLE001 — a stale flag is not worth a turn
            logger.debug("could not publish the started state", exc_info=True)

    # -- SessionHandle -----------------------------------------------------------

    @property
    def session_projection_seed(self) -> SessionProjection:
        """The projection skeleton: identity fields the runtime folds onto.

        A pure read, deliberately: the runtime reads this for IDENTITY (to stamp
        ``kind``, and to build its own sink fold over the object), and a getter
        that also mutates is the shape that has twice produced this PR's defects
        — a reader changing state it did not know it touched (review round 4,
        NIT 2). Re-dating is :meth:`redate_from_phase`, called where the age
        becomes a frame.
        """
        return self._projection

    def redate_from_phase(self) -> None:
        """Re-date the band's age through the fold the EVENTS ARE FED into.

        The runtime calls this on the frame it is about to serialize (probing for
        the member, so a reduced handle without a fold simply has none): the age
        is written when the phase moves, so a viewer — or a push — arriving
        mid-phase must be served the phase's age AT THAT MOMENT rather than the
        number from the last edge (review round 3, MAJOR 1 and round 4's BLOCKER,
        both on this hand-off).
        """
        self._fold.redate_from_phase()

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
        # And seed the CLOCKS from the same attach, for the same reason: a phone
        # subscribing mid-turn never witnessed the ``tool_execution_start`` (or
        # the phase edge) either, so the fold's first event would date work that
        # is already running from the phone's arrival — the reported band
        # reading ``0s`` and counting up. One-shot, and probed: a session that
        # cannot answer seeds nothing. See ``ProjectionFold.reconcile_clocks``.
        self._fold.reconcile_clocks(session)
        self._refresh_state()
        self._warm_subagent_details()
        self._unsubscribe = unsubscribe
        return unsubscribe

    async def refresh_attention(self) -> dict[str, Any]:
        state = await self._on_app(lambda: self._session().refresh_attention())
        self._projection.attention = state
        return state

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        state = await self._on_app(lambda: self._session().acknowledge_attention(token))
        self._projection.attention = state
        return state

    @property
    def frontend_state_seed(self):  # type: ignore[no-untyped-def]
        """Canonical state seed for full-TUI attach clients only."""
        return self._session().frontend_state

    async def subscribe_frontend(
        self, on_update, *, display_window=False
    ):  # type: ignore[no-untyped-def]
        """Capture snapshot+subscription atomically on Textual's owner loop.

        UNBOUNDED ON PURPOSE (``budget=None``), and the reason is the defect UX
        round 2 measured. This bind is the one hop whose caller is not a client
        waiting for an answer: it runs in the runtime's ``_serve_frontend_sync``
        task, no request is parked on it, and the connection is already usable
        without it — the welcome is on the wire, ``ping`` and the four control
        verbs the runtime admits pre-sync (``stop``, ``abort``, ``steer``,
        ``cancel`` — its ``_SYNC_PRIORITY_OPS`` set) are served while it is
        pending, and
        ``frontend_sync_pending`` keeps everything heavier refused until it
        lands. Giving it the interactive budget therefore bought nothing and cost
        the session: with a busy terminal, a viewer was welcomed in 0.00 s and
        then KILLED at exactly 10.01 s (the bind's expiry took the whole
        connection down with it, and the person was told their session had
        died), measured twice, the second time with no other client attached. The
        parent had no bound here and its viewer was still served at +26.6 s into
        a 25 s freeze, so bounding it was a regression in this PR's own case.
        The budget protects callers who are waiting; nobody waits for this.
        """
        return await self._on_app(
            lambda: self._session().subscribe_frontend(on_update, display_window=display_window),
            budget=None,
        )

    async def record_shell(self, command: str, result: Any) -> None:
        await self._on_app(lambda: self._session().record_shell(command, result))

    async def history_page(self, before: str, anchor: str = "") -> dict[str, Any]:
        return await self._on_app(lambda: self._session().history_page(before, anchor))

    def subscribe_events(self, on_event: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Feed serialized AgentEvents to the registrant's v4 relay.

        Serialization happens on the Textual/session loop where the event is
        emitted. The callback itself is thread-safe (RuntimeServer._relay_event
        only schedules onto its own loop), so this preserves producer order
        without moving pydantic objects across threads.
        """
        self._on_event = on_event

        def handler(event: Any) -> None:
            try:
                # Bounded here for the same reason the serving handle bounds: this
                # is a wire encoder — the OTHER implementation of the handle
                # capability ``RuntimeServer`` relays (``server.py``
                # ``_on_connection``), used when the session is owned by a TUI
                # rather than by the server. An ``agent_end`` carries the whole
                # turn, so leaving it unbounded here would leave the bound
                # bypassable by whichever host happens to own the session.
                on_event(
                    bound_agent_end_for_wire(
                        event.model_dump(mode="json"),
                        session_id=getattr(self._session(), "session_id", None),
                    )
                )
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
        image_blocks = await _image_blocks_async(images)

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
                        from local_operator.session.errors import TurnInFlight

                        self._command_reservations.reject(
                            command_id,
                            transfer_to_steer=(
                                isinstance(exc, TurnInFlight) or "already streaming" in str(exc)
                            ),
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
        image_blocks = await _image_blocks_async(images)

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
        # Keyed by the id `do_steer` handed the session (always supplied on
        # this path), so the drain's MessageStartEvent upgrades THIS row
        # instead of being matched against the transcript tail — by delivery
        # time a steer's echo is several assistant/tool rows back (issue #231).
        self._fold.note_user_message(text, steer=True, message_id=command_id)
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
        # Session.receive_peer_message is a COROUTINE that must run on the host's
        # event loop (it touches _context.messages, the transcript, and may
        # spawn a turn). `_on_app` only runs SYNC callables on the Textual
        # thread, so we reuse the prompt() machinery: a sync shim scheduled on
        # the app captures the host loop and schedules the coroutine there with
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

    async def request_stop(self) -> str:
        """The graceful rung of the kill switch, for a TUI-OWNED session.

        A TUI process is not a runtime process: the session ends, the
        terminal stays. So the ``stop`` op here routes into the exact path
        bare ``/stop`` takes in this app (deny gates → abort → dispose →
        release the lease → tear the registrant down → show the session
        cold with the ``/resume`` receipt), and the process survives with
        the session ended beneath it. Without this hook the ladder saw a
        runtime that "cannot stop itself gracefully", confirmed identity
        over this very socket, and SIGTERMed the terminal (seen live: every
        open ``lop`` window died on ``lop stop --all``).

        Scheduled, not awaited: the stop tears down the registrant that is
        serving THIS request, so awaiting its completion from inside the
        dispatch would hold the socket open past the point where anything
        can answer on it. The ack means "underway"; the ladder's exit-wait
        (a pid-alive poll) never applies to a TUI, whose record is simply
        unpublished — ``_await_pid_exit`` is what makes the caller observe
        the session as gone, through the reaped record on its next scan.

        THE RECEIPT DOES NOT WAIT FOR THE APP LOOP (review round 1, UX U2).
        It used to: the hop into Textual was awaited, so the reply arrived only
        once the app had run the scheduling step — which is precisely the work
        a BUSY app cannot service. Measured over a real socket on a working
        session, rung 1 of the ladder timed out at its 15 s
        (``OwnerAckTimeout``) and ``lop stop`` escalated to the signal rung,
        killing a host whose hook would have ended the session politely
        (exit 143). A receipt is a STATEMENT about what is about to happen, not
        a report of work completed, so it is built here from identity reads and
        the hop that PERFORMS the stop is detached (``_detach_stop_hop``). The
        ladder's record reap is still the confirmation, so an early ack cannot
        make a caller believe more than it should.
        """

        def schedule() -> str:
            self._app.run_worker(self._app._stop_local_session(), thread=False, group="session")
            return "scheduled"

        # Identity reads, off the app loop: the same latitude the runtime takes
        # when it reads ``session_projection_seed`` for identity, and the reason
        # this receipt can be built without hopping. ``_app`` is read from the
        # runtime's loop, which is never Textual's (see the module docstring).
        session = self._app._session
        sid = str(getattr(session, "session_id", "") or "")
        name = str(getattr(session, "conversation_name", "") or sid)
        self._detach_stop_hop(schedule)
        # The follower's receipt: what ended and the way back, the same
        # line the host's own transcript paints.
        reopen = f"/resume {sid}" if sid else "/resume"
        return f'stopping "{name}" — {reopen} reopens it'

    def _detach_stop_hop(self, schedule: Callable[[], str]) -> None:
        """Run the stop's app-loop hop WITHOUT holding the caller's reply.

        One slot rather than a set: this is rung 1 of the stop ladder and the
        ladder dials once per session, so a second concurrent stop of the same
        TUI is the same stop. The task is held so the loop cannot collect it
        mid-hop, and its exception is consumed here — an unretrieved task
        exception is an asyncio warning, not a diagnostic. A hop that fails is
        logged at WARNING, because it is the one state in which the session
        stays up and the ladder's signal rung becomes the thing that ends it.
        """
        task = asyncio.ensure_future(self._on_app(schedule))
        self._stop_hop_task = task

        def settle(completed: asyncio.Task[Any]) -> None:
            if self._stop_hop_task is completed:
                self._stop_hop_task = None
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.warning("the TUI stop hop did not run — %s", error)

        task.add_done_callback(settle)

    async def set_model(self, provider: str, model_id: str) -> str:
        return await self.set_model_effort(provider, model_id, None)

    async def set_model_effort(self, provider: str, model_id: str, effort: str | None) -> str:
        """Select a model on the TUI-hosted session, and its level if one came.

        Two slash commands rather than one because that is what this owner's
        control surface is: the TUI applies a model and an effort through
        separate commands, and ``/effort`` validates the level against the model
        it has just been switched to. The effort half runs only when a level was
        chosen, so a model-only switch — every caller before the draft's chips
        existed — runs exactly the one command it always ran.
        """

        def apply() -> None:
            self._app._run_slash_command(f"/model {provider}/{model_id}")
            if effort:
                self._app._run_slash_command(f"/effort {effort}")

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
        return await self.slash_images(command, args, None)

    async def complete_aside(
        self,
        turns: list[dict[str, Any]],
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Run an off-record provider request on the authoritative session.

        The TUI-hosted runtime's half of the aside seam, and the SAME two rules
        as ``ServingSessionHandle.complete_aside`` apply for the same reasons:

        * the LAST user turn is wrapped in ``ASIDE_PROMPT`` HERE, so a remote
          caller (the phone's quick-ask, the desktop app attached to this
          session) cannot forget the instruction. The TUI's own ``/btw`` calls
          ``Session.complete_aside`` directly and wraps for itself, so nothing
          is wrapped twice on this handle.
        * ``on_delta`` is forwarded only when the running session advertises it
          — probed, not assumed: the app can host a session built before the
          parameter existed, and the thread hop below would turn the TypeError
          into a failed aside rather than the settled answer it should get.
        """
        from local_operator.harness.types import Message
        from local_operator.session.aside import wrap_aside_turns

        messages = wrap_aside_turns([Message.model_validate(turn) for turn in turns])
        session = self._session()
        owner_loop = await self._on_app(asyncio.get_running_loop)
        fields: dict[str, Any] = {}
        if (
            on_delta is not None
            and "on_delta" in inspect.signature(session.complete_aside).parameters
        ):
            fields["on_delta"] = on_delta
        future = asyncio.run_coroutine_threadsafe(
            session.complete_aside(messages, **fields), owner_loop
        )
        return await asyncio.wrap_future(future)

    async def slash_images(
        self,
        command: str,
        args: str,
        images: list[dict[str, str]] | None,
    ) -> str:
        line = f"/{command}" + (f" {args}" if args else "")

        def apply() -> None:
            self._app._run_slash_command(line, _decode_attachments(images))

        await self._on_app(apply)
        self._refresh_state()
        return f"ran {line}"

    async def run_slash_authoritative(
        self,
        command: str,
        args: str,
        images: list[dict[str, str]] | None,
        *,
        locality: str = "local",
        consumers: Iterable[str] | None = None,
        may_loosen: bool | None = None,
    ) -> dict[str, Any]:
        """Run one shared slash command and return its typed outcome.

        The host-side backend for a follower's ``route_shared_slash``. Unlike
        ``slash_images`` — which runs the command's UI in the HOST's terminal
        and returns a ``ran /…`` receipt — this asks the app for a
        :class:`SlashResult` payload the INVOKING terminal renders locally, so
        ``/goal``/``/rename``/``/mcp``/``/context`` answer where they were
        typed. The producer is a coroutine, so it is scheduled on the app loop
        and its completion awaited here.

        That await used to be genuinely unbounded, to let an MCP grant's
        browser round trip finish. It never worked: the invoking client gives
        up after ``ACK_TIMEOUT_S`` (15 s), so a grant that waited on a human
        left the follower with a timeout while the exchange ran on invisibly.
        The grant verbs now return a receipt immediately and report their
        settled outcome as a ``NoticeEvent`` (see ``local_operator.mcp.grants``),
        so nothing reaching here blocks on a person.

        ``locality`` is the invoking client's declared position; it is passed
        through to the app so the grant verbs can tell a terminal on this
        machine from a relayed remote device.

        ``may_loosen`` is forwarded for the same reason and to the same end as on
        ``ServingSessionHandle`` (issue #1310; design round 2 D10, UX round 2
        U9): the app's reports name remedies, and a follower whose connection
        cannot carry `/approvals auto` must not be offered it by the report the
        app builds on its behalf. It comes from the registrant's seam, which is
        the only place that knows what this connection may do.

        ``consumers`` is forwarded for the same reason and to the same end as
        on ``ServingSessionHandle``: a session can be hosted either by a detached
        runtime or by this app, and a follower must get the same answer from
        both. A viewer that does not consume action-carrying receipts has its
        request completed HERE when this TUI is the host, exactly as the
        runtime itself would.
        """
        owner_loop = await self._on_app(asyncio.get_running_loop)
        done: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        def schedule() -> None:
            task = owner_loop.create_task(
                self._app.run_slash_authoritative(
                    command,
                    args,
                    list(images or []),
                    locality=locality,
                    consumers=consumers,
                    may_loosen=may_loosen,
                )
            )

            def finish(completed: asyncio.Task[Any]) -> None:
                if completed.cancelled():
                    done.get_loop().call_soon_threadsafe(
                        _set_unless_done, done, None, asyncio.CancelledError()
                    )
                    return
                error = completed.exception()
                done.get_loop().call_soon_threadsafe(
                    _set_unless_done,
                    done,
                    None if error else completed.result(),
                    error,
                )

            task.add_done_callback(finish)

        # Same reason as :meth:`_on_app`: this hop is taken by a follower's
        # ``route_shared_slash``, i.e. from the runtime's loop, and a direct
        # ``call_from_thread`` would park that loop — accept, ``ping`` and
        # heartbeat included — until the app services the enqueue. Bounded by
        # the same budget for the same reason (review round 1, MAJOR 1: an
        # unbounded enqueue answers the client's 15 s timeout with silence).
        # The ``done`` await below stays unbounded ON PURPOSE — it is the
        # producer's own work, and the grant verbs that used to wait on a human
        # now report their outcome as a ``NoticeEvent`` instead.
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._app.call_from_thread, schedule),
                timeout=_APP_HOP_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, TimeoutError):
            raise TimeoutError(_app_hop_timeout_note()) from None
        result = await done
        return result if isinstance(result, dict) else {"kind": "notice", "text": f"ran /{command}"}

    async def adopt_aside(self, messages: list[dict[str, Any]]) -> str:
        """Fork a follower's aside exchange into the authoritative conversation."""
        from local_operator.harness.types import Message

        parsed = [Message.model_validate(message) for message in messages]
        session = self._session()
        owner_loop = await self._on_app(asyncio.get_running_loop)
        future = asyncio.run_coroutine_threadsafe(session.adopt_aside(parsed), owner_loop)
        await asyncio.wrap_future(future)
        self._refresh_state()
        return f"forked {len(parsed) // 2} aside exchange(s) into the chat"

    def cancel_subagents_count(self) -> int:
        """Cancel every running subagent on the runtime; return the REAL count."""
        session = self._session()
        cancel = getattr(session, "cancel_subagents", None)
        if not callable(cancel):
            return 0
        result = cancel("interrupted")
        stopped = result if isinstance(result, int) else 0
        self._refresh_state()
        return stopped

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
        """Settle the host's real ApprovalPrompt from another front end.

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
        against the one loop that also mounts and settles the card. That is
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
            # None on Q0, keeps partials past it) — parity with serving.py.
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
        """Project the host's real approval prompt to v4 followers.

        The request id is stored on the prompt itself because approvals are
        serialized and the prompt is the one arbitration object every answer
        route ultimately resolves. The phone also sees this projection, gaining
        parity rather than a second approval-specific channel.
        """
        request_id = secrets.token_hex(8)
        setattr(card, "_mobile_request_id", request_id)
        self._publish_pending_gate(
            PendingRequest(
                request_id=request_id,
                kind="approval",
                title=str(getattr(card, "tool_name", "") or "tool approval"),
                detail=str(getattr(card, "description", "") or ""),
            )
        )

    def note_approval_settled(self, card: Any) -> None:
        """Remove exactly this prompt's projected approval, on every exit path."""
        request_id = str(getattr(card, "_mobile_request_id", "") or "")
        if not request_id:
            return
        state = getattr(self._session(), "frontend_state", None)
        pending = getattr(state, "pending_gate", None)
        if pending is not None and pending.request_id == request_id:
            self._publish_pending_gate(None)

    def note_ask_pending(self, card: Any) -> None:
        """Project a freshly mounted TUI ask picker to the phone as an
        answerable card.

        Called on the Textual loop from ``OperatorApp.request_user_choice``
        immediately after the picker mounts, so the card sits on its first
        question. Touching ``_fold`` here is safe: fold mutations are
        synchronous and only ``_notify`` crosses threads (the registrant
        coalesces the push onto its own loop).

        A ``token_hex`` request id (serving.py's scheme); the card's CURRENT
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
        # Parity with serving.py's gate: log when more than one question rides a
        # single card, so the operator can see a multi-part ask went to the
        # phone (UX minor-1).
        total = self._question_total(card)
        if total > 1:
            logger.info("mobile tui ask: %d questions, projecting question-by-question", total)
        self._push_current_question(request_id, card)
        self._publish_pending_gate(self._fold.projection.pending)
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
        self._publish_pending_gate(self._fold.projection.pending)
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
        self._publish_pending_gate(None)
        if self._on_projection is not None:
            self._on_projection()

    def _publish_pending_gate(self, pending: PendingRequest | None) -> None:
        """Publish the host gate into the canonical full-TUI contract."""
        session = self._session()
        store = getattr(session, "_frontend_state_store", None)
        if store is None:
            return
        payload = pending.to_json() if pending is not None else None
        store.mutate(pending_gate=payload)

    def _request_id_for_card(self, card: Any) -> str | None:
        for request_id, pending_card in self._ask_pending.items():
            if pending_card is card:
                return request_id
        return None

    # -- internals ---------------------------------------------------------------------

    async def _on_app(
        self,
        fn: Callable[[], Any],
        *,
        budget: Any = _BUDGET_FROM_CONSTANT,
    ) -> Any:
        """Run ``fn`` on the Textual thread and await its result, BOUNDED —
        and WITHOUT parking the runtime's loop on the way there.

        ``call_from_thread`` does not merely enqueue: it ENQUEUES AND THEN
        BLOCKS until Textual runs the callback. Called directly from this
        coroutine, that blocking enqueue happens on whichever thread is awaiting
        us — and for a TUI-hosted runtime that thread is the runtime's own
        event loop, so the loop stops accepting connections, stops answering
        ``ping``, stops pushing projections and stops writing the heartbeat for
        as long as the app is busy. Measured on 2026-09-18 (a real
        ``OperatorApp`` + ``RuntimeServer(kind="tui")``, the app loop held with
        a synchronous ``time.sleep``): with one client registered, the control
        thread sat inside this function's enqueue for 49.6 s while the app was
        held 50.5 s; a fresh dial got no welcome within 20 s (and the client's
        own ``ACK_TIMEOUT_S`` is 15 s), and the discovery record's heartbeat
        crossed the 45 s timeout into ``wedged`` at hold+43 s, peaking at
        52.5 s — on a pid that was alive and idle. With NO client registered the
        same 20 s hold was harmless (beat 5.1 s, ``live``), which is what
        identifies the hop — not a slow turn — as the cause.

        ``asyncio.to_thread`` is what puts the blocking enqueue INSIDE the
        bound: the runtime's loop only awaits the worker thread, so it stays
        runnable and keeps serving every OTHER connection and its own loops
        while this one hop waits. The pattern is the same one already used two
        modules away for the viewer-resume hop (``tui/app.py``: "`to_thread` is
        what puts the blocking enqueue INSIDE the bound").

        THE BOUND, AND WHAT IT DOES NOT DO. The wait here is bounded by ONE
        budget of :data:`_APP_HOP_TIMEOUT_S` covering the whole hop — the
        blocking enqueue AND the callback's awaited result. That is a
        correction of what this docstring said in review round 1 (MAJOR 1): the
        first version claimed the pre-existing ``wait_for`` around the *future*
        had "become a real bound", which it could not, because
        ``call_from_thread`` returns only once Textual has RUN the callback —
        so the ENQUEUE await is what has to sit inside the bound, and the
        future-await that follows is only the scheduling tail. It still does
        not RECALL an enqueue that is already parked: Textual cannot cancel a
        queued callback, so on expiry the callback runs anyway on the app loop
        and ``_set_unless_done`` drops its result. What expires is how long
        THIS caller waits — what a stuck app produces is an error the caller
        can report, instead of a wedge it waits out forever. The residual cost
        is one parked worker thread per expired hop until the app drains its
        queue, bounded by the executor's own ``max_workers``; before this
        change that same wait was paid by the runtime's loop instead, taking
        the control socket down with it.

        AN AWAITABLE RESULT IS AWAITED (review round 1, U1). Several callbacks
        handed to this method are async on the session
        (``refresh_attention``, ``acknowledge_attention``, ``record_shell``),
        so ``fn()`` returns a coroutine OBJECT. Handing that back as the answer
        is not a harmless no-op: it stored a coroutine in
        ``Projection.attention``, after which every projection push failed to
        serialize — the FIRST viewer was dropped the moment a SECOND dialled
        (reproduced by UX round 1 over a real socket with the product's own
        attach clients, and identically at 2484cfa4, which is what marks it
        pre-existing rather than new here), and ``record_shell``'s write silently
        never ran at all. An awaitable result is therefore scheduled as a task ON
        the app loop and this caller is settled with ITS result.

        ``budget=None`` MEANS NO BUDGET, and it is for the one caller that is
        not a client waiting for an answer: the canonical frontend bind
        (:meth:`subscribe_frontend`, reached from the runtime's
        ``_serve_frontend_sync``). See that method for why an interactive budget
        must not apply there — UX round 2 measured what applying it costs
        (a viewer welcomed in 0.00 s and then killed at exactly 10.01 s because
        the terminal was busy, with the product reporting its session as dead).
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def settle(value: Any, error: BaseException | None) -> None:
            loop.call_soon_threadsafe(_set_unless_done, future, value, error)

        def expired(error: BaseException | None = None) -> TimeoutError:
            """Give up on this hop, and DISARM the answer slot on the way out.

            Marking the future done is what makes a late result safe: the callback
            runs anyway (Textual cannot cancel a queued one), and
            ``_set_unless_done`` releases anything IT owns rather than stashing an
            unread value — see ``_release_orphan_result`` for the subscription
            that leaked without this. All three expiry paths go through here so
            none of them can forget.
            """
            if not future.done():
                future.cancel()
            return TimeoutError(_app_hop_timeout_note())

        def wrapped() -> None:
            # RUNS ON THE APP LOOP: ``call_from_thread`` invokes it there, so
            # ``get_running_loop()`` below is Textual's loop and a task created
            # on it runs the callback on the thread that owns the widgets.
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 — the error IS the answer
                settle(None, exc)
                return
            if not inspect.isawaitable(result):
                settle(result, None)
                return

            def finish(completed: asyncio.Task[Any]) -> None:
                # HELD UNTIL IT LANDS, like every other task this file keeps a
                # reference to (``_detail_tasks``, ``_stop_hop_task``): an
                # unreferenced task can be collected mid-await, and this one
                # carries the ANSWER the caller is waiting for — losing it would
                # leave the hop parked to its deadline. Mutated only on the app
                # loop (``wrapped`` and this callback both run there).
                self._late_hop_tasks.discard(completed)
                if completed.cancelled():
                    settle(None, asyncio.CancelledError())
                    return
                error = completed.exception()
                settle(None if error else completed.result(), error)

            task = asyncio.ensure_future(result)
            self._late_hop_tasks.add(task)
            task.add_done_callback(finish)

        # ``fn`` still runs on the Textual loop — only the WAIT moved off this
        # one. That distinction is load-bearing: the admission sections these
        # callbacks perform (``CommandReservations.reserve`` and friends) are
        # documented as "mutated only on the session runtime's loop", and the
        # runtime's loop here IS Textual's, so they must keep running there.
        if budget is _BUDGET_FROM_CONSTANT:
            budget = _APP_HOP_TIMEOUT_S
        if budget is None:
            # No deadline at all: the caller is background setup, not a client.
            await asyncio.to_thread(self._app.call_from_thread, wrapped)
            return await future
        deadline = loop.time() + budget
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._app.call_from_thread, wrapped),
                timeout=budget,
            )
        except (asyncio.TimeoutError, TimeoutError):
            raise expired() from None
        remaining = deadline - loop.time()
        if remaining <= 0:
            # The enqueue answered, but there is nothing left of the budget for
            # the callback's own result — the same failure, reported the same
            # way rather than as a second full wait.
            raise expired()
        try:
            # THE THIRD EXPIRY PATH, and it needs the same name as the other two
            # (review round 2, MINOR 1). Without this, an enqueue serviced inside
            # the budget whose callback then did not resolve in what remained
            # raised a bare ``TimeoutError`` with an EMPTY message — reproduced
            # with a 0.3 s budget, and it is the one shape a caller cannot act
            # on, because the sentence naming the busy terminal is the whole
            # difference between "retry when it settles" and a mystery.
            return await asyncio.wait_for(future, timeout=remaining)
        except (asyncio.TimeoutError, TimeoutError):
            raise expired() from None

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
                fingerprint, todos = result
                if self._detail_fingerprints.get(session_dir) != fingerprint:
                    self._detail_fingerprints[session_dir] = fingerprint
                    # The transcript is NEVER placed on the wire (it is fetched
                    # lazily from /history), so only ``todos`` is hydrated here.
                    # Passing ``[]`` keeps the fold's signature stable while
                    # avoiding the child-history fold that would only be
                    # discarded (once here, and again on the /history fetch).
                    if self._fold.set_subagent_hydrated_details(job_id, [], todos):
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


def _release_orphan_result(value: Any) -> None:
    """Release a hop result that arrived after its caller gave up.

    An expired hop leaves its callback RUNNING — Textual cannot cancel a queued
    callback — so the value it eventually produces has no owner: the caller has
    already unwound. For most callbacks that is harmless, because the value is a
    dict the garbage collector takes. For the one that is not, the canonical
    frontend bind, the value is a live FRONTEND SUBSCRIPTION: dropping it leaves
    the session pushing canonical state at a subscriber for the life of the app.
    That is the leak UX round 2 measured — one per timed-out bind, compounding
    with each redial (baseline subscribers 2 → 3, and it stayed 3 after the
    viewer closed).

    Releasing a subscription nobody received is always correct, and it is the
    same rule the runtime applies on its side of the hop
    (``RuntimeServer._release_when_landed``). Probing for ``unsubscribe`` rather
    than naming ``FrontendSubscription`` keeps this file free of a session-layer
    import for one call, and the only results that carry one are subscriptions.
    """
    unsubscribe = getattr(value, "unsubscribe", None)
    if not callable(unsubscribe):
        return
    try:
        unsubscribe()
    except Exception:  # noqa: BLE001 — releasing must never break the hop
        logger.debug("a late hop result could not be released", exc_info=True)


def _set_unless_done(
    future: "asyncio.Future[Any]", value: Any, error: BaseException | None
) -> None:
    if future.done():
        # THE CALLER IS GONE. Settling is impossible, so the value is dropped —
        # but anything IT owns has to be released first, or the drop is a leak.
        # See ``_release_orphan_result`` for the measured shape.
        _release_orphan_result(value)
        return
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(value)


class _DetailChangedDuringHydration(RuntimeError):
    """Worker result became stale while its transcript was being read."""


def _load_subagent_detail(
    session_dir: str,
) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    """Read one child's todos in a worker and return its stable fingerprint.

    Only ``todos`` is hydrated from this path: the child transcript is no longer
    projected on the wire (it blew past the daemon's 1 MB control-frame cap and
    is fetched lazily from ``/api/sessions/{sid}/agents/{job_id}/history``
    instead — see ``ProjectionFold.set_subagent_hydrated_details``). Folding the
    full child history here only to discard it duplicated, per child, the exact
    work the /history endpoint already does on fetch, so it is not built.
    """
    from local_operator.session.transcript import (
        TRANSCRIPT_FILENAME,
        read_latest_custom,
    )

    path = Path(session_dir) / TRANSCRIPT_FILENAME
    before = path.stat()
    # One bounded backward scan instead of a whole ``Transcript`` construction:
    # hydration runs per child per event, and the reader stops at the newest
    # snapshot row rather than decoding everything above it.
    raw_todos = (read_latest_custom(session_dir, "todo_snapshot") or {}).get("items") or []
    after = path.stat()
    # Atomic transcript replacement or an append during hydration invalidates
    # this result; the caller's next child event will request the newer detail.
    fingerprint = (after.st_size, after.st_mtime_ns)
    if (before.st_size, before.st_mtime_ns) != fingerprint:
        raise _DetailChangedDuringHydration("child transcript changed during hydration")
    return fingerprint, raw_todos if isinstance(raw_todos, list) else []


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
