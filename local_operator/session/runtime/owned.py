"""Sessions the daemon owns: started from the phone, run in-process.

An owned session is a full harness ``Session`` built with the same
composition root as the CLI (:func:`session_factory.create_session`), wrapped
in the :class:`~local_operator.session.runtime.server.SessionHandle`
contract and registered through the SAME loopback socket path a TUI uses.
That last part is the design's keystone: the daemon's web layer never
branches on who owns a session, so a phone-started session and a terminal
session are indistinguishable to the UI — and connection failure handling
(re-dial, degraded, ended) has exactly one implementation.

Approval and ask gates are installed at spawn: the harness calls them when a
tool needs the user, and this bridge parks the call on a future whose
resolution arrives as a control request from the phone. The pending request
is on the projection the whole time, so a phone that opens mid-approval sees
the card — a question for the user is the most prominent thing on screen
(branding.md §7), and "the agent is waiting" must survive a phone restart.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import secrets
import uuid
from asyncio import InvalidStateError
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Coroutine, cast

from local_operator.harness.approval import (
    GATE_TIMEOUT_CUSTOM_TYPE as _GATE_TIMEOUT_CUSTOM_TYPE,
)
from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY
from local_operator.harness.types import AgentEvent, ModelChangeEvent

if TYPE_CHECKING:
    from local_operator.harness.types import ImageContent
from local_operator.mobile.command_reservation import CommandReservations
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    ask_pending_request,
)
from local_operator.session.runtime.server import SessionHandle
from local_operator.session.runtime.server import image_blocks as _image_blocks

logger = logging.getLogger(__name__)


def _resolve_gate_future(future: asyncio.Future[Any], value: Any) -> None:
    """Set a gate future's result if it can still take one, swallowing the
    InvalidStateError otherwise. Module-level so ``call_soon_threadsafe``
    callbacks (which must never raise) can share it."""
    try:
        future.set_result(value)
    except (InvalidStateError, TypeError):
        pass


#: How long an approval/ask may sit unanswered before the tool is denied and
#: the turn told why, when NOTHING CAN PRESENT THE CARD. A phone in a pocket
#: is the common case; an unbounded wait would pin the turn (and its tool
#: slot) forever.
#:
#: This is no longer the general case. Under the detached model a question can
#: be waiting for a user who is simply not at the terminal right now, and
#: denying their write tool after thirty seconds is the wrong answer to "I
#: stepped away" — see :func:`_gate_timeout_s`.
PENDING_REQUEST_TIMEOUT_S = 30.0

#: How long a PARKED gate waits, in hours, when the setting is absent. A day
#: is chosen to span an overnight: a question asked at 6pm is still answerable
#: at breakfast, which is the whole point of a session that outlives the
#: terminal. `0` in the setting means never time out.
DEFAULT_UNATTENDED_GATE_TIMEOUT_H = 24

#: Re-exported from the harness, which is the one definition all three layers
#: (this writer, the session's model render, the TUI's user render) share.
#: Kept as a module-level name here because it is part of this module's
#: published surface — the tests and the mobile bridge import it from here.
GATE_TIMEOUT_CUSTOM_TYPE = _GATE_TIMEOUT_CUSTOM_TYPE
# Socket admission is intentionally bounded: many front ends may produce input,
# but an abandoned automation loop must not grow one owner's memory forever.
MAX_QUEUED_PROMPTS = 32


@dataclass
class _PromptCommand:
    command_id: str
    text: str
    images: list["ImageContent"]
    admitted: asyncio.Future[None]

    def __iter__(self):  # type: ignore[no-untyped-def]
        # Tuple compatibility for older diagnostics that inspect the queue.
        yield self.text
        yield self.images


class OwnedSessionHandle(SessionHandle):
    """SessionHandle over an in-process Session living on ``loop``.

    The registrant drives handle methods on the OWNING loop here — the child
    process runs the registrant's socket server as a task on its one asyncio
    loop, so no cross-thread hop is needed and ``run_coroutine_threadsafe``
    never appears. The registrant's ``start_in_process`` classmethod is the
    entry point that wiring uses.
    """

    def __init__(
        self,
        session: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        cwd: str,
        auto_approve: bool = False,
    ) -> None:
        self._session = session
        self._loop = loop
        # When the owner's saved default is full-auto (``tool_approval_mode:
        # auto``), the phone must not park a card the TUI would never show —
        # the gate is answered ``True`` inline instead. Stored so a future
        # per-session toggle can flip it without reconstructing the handle.
        self._auto_approve = auto_approve
        #: The (kind, title, detail) of the gate currently parked, or None.
        #: Kept so the announcement can be re-run when the last viewer
        #: detaches — the routing decision is made when the gate opens, and
        #: without this a gate opened under a watching terminal is never
        #: announced after that terminal closes (round 3, B2).
        self._parked_announcement: tuple[str, str, str] | None = None
        #: The RuntimeServer serving this handle, set by its constructor. The
        #: gate path needs it for two things only a server knows: how many
        #: front ends could present a card right now, and how to publish the
        #: parked-gate bit into the discovery record. Declared here (rather
        #: than only assigned from outside) so it is a real attribute with a
        #: type, not a dynamic one every reader has to guess at.
        self._registrant: Any = None
        self._on_projection: Callable[[], None] | None = None
        self._projection = SessionProjection(
            session_id=session.session_id,
            pid=0,  # stamped by the registrant's record
            kind="daemon",
            # A restored session carries its stored name; a brand-new one has
            # none yet and the naming errand (see prompt()) fills it after the
            # first substantive turn. Left empty rather than a stand-in like
            # "mobile session" so the phone's own fallback (the header shows
            # "untitled", the list shows the cwd) is what the user sees until
            # the real title lands, instead of a placeholder that never moves.
            conversation_name=getattr(session, "conversation_name", "") or "",
            cwd=cwd,
            model_label=_effective_label(session),
            model_selector=_selector(session),
            effort=_current_effort(session),
            effort_ladder=_ladder(session),
        )
        self._fold = ProjectionFold(self._projection)
        # Conversation naming is a TUI-only errand today (OperatorApp owns the
        # naming worker), so a phone-started session used to stay "mobile
        # session" forever — the session list and the header both read the
        # conversation name and had nothing better to show. This latch mirrors
        # OperatorApp._name_requested: the first substantive prompt fires ONE
        # background naming call, alongside the turn it decorates.
        self._name_requested = False
        # Opener of a naming attempt that returned nothing. Isolated naming
        # often 429s on a dead primary BEFORE the turn pins a fallback; the
        # route edge re-fires this opener once a serving model exists.
        self._pending_name_text = ""
        # Strong references to detached background tasks (the naming errand),
        # so the event loop cannot garbage-collect one mid-flight and drop the
        # title silently. Each task removes itself on completion.
        self._background_tasks: set[asyncio.Future[Any]] = set()
        # request_id -> Future the gate/ask call is parked on.
        self._pending_futures: dict[str, asyncio.Future[Any]] = {}
        # request_id -> the AskQuestion.id the harness is waiting on (the
        # answer map's key — see ask_gate).
        self._pending_question_ids: dict[str, str] = {}
        # One owner process, many producers: ordinary prompts enter this FIFO
        # and exactly one drain invokes Session.prompt at a time. Session itself
        # deliberately rejects concurrent calls, so serialization belongs at
        # this control/admission boundary rather than by adding another writer.
        self._prompt_queue: deque[_PromptCommand] = deque()
        # Pending and running duplicates join the same durable-admission future;
        # completed duplicates are recognized from transcript-backed history.
        self._prompt_commands: dict[str, _PromptCommand] = {}
        self._prompt_drain_task: asyncio.Task[None] | None = None
        # Prompt and steer share one identity namespace. In particular, an idle
        # projection may race a turn start and transfer the rejected prompt's
        # identity to steer rather than admitting the same producer twice.
        self._command_reservations = CommandReservations(session)
        self._unsubscribe_admitted_commands = self._command_reservations.subscribe_durable()
        self._disposing = False
        #: Installed by the runtime process (``process.amain``): fires the
        #: process's stop event so a socket ``stop`` op exits the way SIGTERM
        #: does. ``None`` under a host that has no process to exit.
        self.on_stop_requested: Callable[[], None] | None = None
        self._install_gates()

    # -- gates -----------------------------------------------------------------

    def _install_gates(self) -> None:
        async def approval_gate(tool_name: str, description: str) -> bool:
            # Full-auto: the owner's saved default is to approve every tier,
            # exactly as the TUI adopts ``tool_approval_mode: auto`` at boot
            # (see OperatorApp._load_approvals_default). Answer inline so the
            # turn never stalls on a card no front end would present.
            if self._auto_approve:
                return True
            request_id = secrets.token_hex(8)
            future: asyncio.Future[bool] = self._loop.create_future()
            self._pending_futures[request_id] = future
            # push_pending, not set_pending: a parallel tool batch can open two
            # approvals concurrently, and each must get its own card. Clearing
            # by request_id (pop_pending) is what keeps them independent — one
            # answered card must not dismiss the sibling that is still waiting.
            self._fold.push_pending(
                PendingRequest(
                    request_id=request_id,
                    kind="approval",
                    title=tool_name,
                    detail=description,
                )
            )
            self._notify()
            self._announce_pending("approval", tool_name, description)
            try:
                return await asyncio.wait_for(future, timeout=self._gate_timeout_s())
            except TimeoutError:
                await self._record_gate_timeout(tool_name, description)
                return False
            finally:
                self._pending_futures.pop(request_id, None)
                self._fold.pop_pending(request_id)
                self._notify()
                self._announce_settled()

        async def ask_gate(questions: list[Any]) -> dict[str, list[str]] | None:
            if not questions:
                # Nothing to ask: answer NOTHING (the harness's "user escaped"
                # signal) rather than parking a card with no question on it.
                return None
            total = len(questions)
            if total > 1:
                logger.info(
                    "mobile ask gate: %d questions, projecting question-by-question",
                    total,
                )
            # Answer the questions one at a time on the SAME card so a
            # multi-question ask is answerable end to end from the phone rather
            # than the first answer resolving the whole set and dropping the
            # rest (U1). Each question parks its own future; the phone's
            # ask_answer resolves the FRONT one, and we advance to the next.
            answers: dict[str, list[str]] = {}
            for index, question in enumerate(questions):
                request_id = secrets.token_hex(8)
                future: asyncio.Future[dict[str, list[str]] | None] = self._loop.create_future()
                self._pending_futures[request_id] = future
                # The answer map is keyed by the QUESTION'S id, not our request
                # id — AskUserFn's contract answers ``question.id -> choices``.
                self._pending_question_ids[request_id] = getattr(question, "id", "") or ""
                # Built through the shared seam so this projection carries option
                # descriptions (U3), the secret flag (D1/U2, never the value),
                # and the "N of M" position (U1) identically to the TUI path.
                self._fold.push_pending(
                    ask_pending_request(
                        request_id,
                        question,
                        question_index=index,
                        question_total=total,
                    )
                )
                self._notify()
                self._announce_pending("ask", getattr(question, "text", "") or "question", "")
                try:
                    answer = await asyncio.wait_for(future, timeout=self._gate_timeout_s())
                except TimeoutError:
                    await self._record_gate_timeout(
                        "ask", getattr(question, "text", "") or "question"
                    )
                    # A timed-out question ends the whole ask: report whatever
                    # earlier questions collected (partial, like the terminal's
                    # Escape) rather than blocking forever on the next one.
                    return answers or None
                finally:
                    self._pending_futures.pop(request_id, None)
                    self._pending_question_ids.pop(request_id, None)
                    self._fold.pop_pending(request_id)
                    self._notify()
                    self._announce_settled()
                if not answer:
                    # The user answered nothing on this question. On the FIRST
                    # question that is "escaped" — fall back to the model's
                    # recommendation (None). Past it, keep the partial map, the
                    # same rule the terminal picker follows on Escape.
                    return answers or None
                answers.update(answer)
            return answers or None

        # Kept as attributes as well as registered on the session: the handle
        # is the single owner of the gate behaviour, and holding the reference
        # lets tests (and any future direct caller) exercise the exact closure
        # the harness will await, rather than a re-implementation of it.
        self._approval_gate = approval_gate
        self._ask_gate = ask_gate
        self._session.set_approval_handler(approval_gate)
        self._session.set_ask_handler(ask_gate)

    async def dispose(self) -> None:
        """Dispose the underlying session (release the claim, flush, abort).

        The child's clean-exit path calls this rather than reaching through
        to ``self._session`` so the ordering (deny gates first) stays in one
        place and hosts cannot forget the claim release."""
        self._disposing = True
        drain = self._prompt_drain_task
        if drain is not None and not drain.done():
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)
        # Anything still queued has no durable receipt. Wake every producer so
        # it can retain and retry the same identity rather than hanging forever.
        while self._prompt_queue:
            command = self._prompt_queue.popleft()
            self._prompt_commands.pop(command.command_id, None)
            if not command.admitted.done():
                command.admitted.set_exception(
                    RuntimeError("session closed before the prompt was admitted")
                )
            self._fold.note_prompt_rejected("session closed before the prompt was admitted")
        self._notify()
        self._unsubscribe_admitted_commands()
        self._command_reservations.clear()
        await self._session.dispose()

    def is_busy(self) -> bool:
        """True while the session holds work a clean exit would destroy.

        The child reaper's WORK signal (design §4.1): a turn under the lock,
        an on-demand compaction, live subagents, or a gate parked on the
        user's answer. A parked approval IS a running turn — the tool slot is
        held and the conversation is mid-flight, so a reaper that counted it
        idle would kill sessions waiting on a phone that is merely slow.

        Reads private session state (``_turn_lock``, ``_compacting``) the way
        the session's own prompt guard does; the alternative — exposing each
        flag — would widen the session's public surface for one caller."""
        if self._disposing:
            # Disposal has explicitly rejected the admission queue and owns
            # teardown now; stale provider streaming flags must not wedge exit.
            return False
        session = self._session
        if getattr(session, "is_streaming", False):
            return True
        if getattr(session, "_compacting", False):
            return True
        if getattr(session, "_turn_lock", None) is not None and session._turn_lock.locked():
            return True
        try:
            if session.running_subagents() > 0:
                return True
        except Exception:  # noqa: BLE001 — a broken counter must not wedge the reaper
            return True
        if self._prompt_queue or (
            self._prompt_drain_task is not None and not self._prompt_drain_task.done()
        ):
            return True
        if self._pending_futures:
            # Ordinary gate timeout owns the non-authorizing answer. Keeping the
            # host busy until then prevents the reaper from denying it early.
            return True
        manager = getattr(session, "jobs", None)
        if manager is not None:
            try:
                # Session.dispose tears down the whole manager, not only task jobs.
                # Background bash and capacity-queued jobs therefore carry the same
                # liveness weight as subagents until their manager row settles.
                if any(getattr(job, "status", None) == "running" for job in manager.list()):
                    return True
            except Exception:  # noqa: BLE001 — uncertainty must fail closed
                return True
        if any(not task.done() for task in self._background_tasks):
            return True
        return False

    def next_wake_due_at(self) -> int | None:
        """Epoch-ms of the earliest armed wake, or ``None`` when none is set.

        The reaper's WARMTH signal (design §6.1 term 2): a runtime whose own
        ``WakeScheduler`` will fire within ``WARM_WINDOW_S`` stays resident
        rather than exiting and paying a cold start for a wake seconds away.
        Read from the live scheduler, not the wake index — the index is a
        derived file for processes that have no session; this process has
        the truth in memory. A disposed scheduler reports no wakes so the
        reaper never waits on a schedule that can no longer fire.
        """
        scheduler = getattr(self._session, "wake_scheduler", None)
        if scheduler is None or getattr(scheduler, "disposed", False):
            return None
        try:
            schedules = scheduler.schedules
        except Exception:  # noqa: BLE001 — uncertainty must not pin the runtime
            return None
        due = [s.next_due_at for s in schedules if isinstance(s.next_due_at, int)]
        return min(due) if due else None

    def request_stop(self) -> None:
        """The graceful rung of the kill switch (``control.stop_session``).

        Runs the SAME clean-exit ordering the SIGTERM path in
        ``process.amain`` runs: deny parked gates, then let the process's
        own stop event fire so ``amain`` disposes the session (aborting any
        in-flight turn, flushing the transcript, RELEASING the sole-writer
        lease), closes the runtime (unpublishing the record) and exits.
        One implementation of the ordering, two triggers (this op and a
        signal), is the point: a stop that arrived over the socket must not
        leave different state than one that arrived as SIGTERM.

        It does NOT merely dispose and wait for the reaper: the reaper's
        drain is a residency policy (``LOP_SESSION_GRACE_S`` can be minutes),
        not an exit path, and measured against a 600 s grace the socket rung
        "succeeded" only when the caller's timeout expired and SIGTERM did
        the work. The hook the process installs (``on_stop_requested``) IS
        the exit; without one — a host that hasn't wired it, e.g. a test —
        the fallback disposes in place so the session still ends.

        Sync and non-raising by contract (see SessionHandle): called on the
        runtime loop from the ``stop`` dispatch, which acks right after.
        """
        self._deny_pending_gates()
        trigger = self.on_stop_requested
        if trigger is not None:
            try:
                trigger()
            except Exception:  # noqa: BLE001 — a stop that faults is still a stop
                logger.warning("session runtime: stop trigger failed", exc_info=True)
            return

        async def _dispose_in_place() -> None:
            try:
                await self.dispose()
            except Exception:  # noqa: BLE001
                logger.warning("session runtime: stop-path dispose failed", exc_info=True)

        self._loop.create_task(_dispose_in_place())

    def _deny_pending_gates(self) -> None:
        """Refuse every parked approval/ask so teardown cannot hang on them.

        The clean-exit ordering mirror of OperatorApp.on_unmount (deny gates
        BEFORE dispose): dispose awaits teardown, and a turn parked on an
        unanswered card would never reach it. Resolving False/None here is
        the same answer a timeout would eventually deliver, minus the wait."""
        for request_id, future in list(self._pending_futures.items()):
            if not future.done():
                # None answers an ask ("user escaped"); False would be wrong
                # there, and None is meaningless to an approval future typed
                # bool — so resolve by the gate the future serves. Both are
                # the deny answer their timeout would deliver.
                value = None if request_id in self._pending_question_ids else False
                self._loop.call_soon_threadsafe(_resolve_gate_future, future, value)
        self._pending_futures.clear()

    async def _resolve_pending(self, request_id: str, value: Any) -> None:
        """Atomically reserve and settle one gate on its owning event loop."""
        import concurrent.futures

        receipt: concurrent.futures.Future[None] = concurrent.futures.Future()

        def settle() -> None:
            future = self._pending_futures.pop(request_id, None)
            if future is None or future.done():
                receipt.set_exception(ValueError("that prompt is no longer waiting"))
                return
            try:
                future.set_result(value)
            except (InvalidStateError, TypeError):
                receipt.set_exception(ValueError("that prompt is no longer waiting"))
                return
            receipt.set_result(None)

        self._loop.call_soon_threadsafe(settle)
        await asyncio.wrap_future(receipt)

    # -- SessionHandle -----------------------------------------------------------

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    # -- v4 full-TUI capability --------------------------------------------------
    # These three are what makes ``RuntimeServer`` advertise
    # ``FRONTEND_CAPABILITY``, and therefore what makes a TUI viewer's attach
    # succeed at all: ``server.py`` advertises the capability only when the
    # handle has ``subscribe_frontend``, and hangs up on any client that asks
    # for a capability it did not advertise. ``RemoteSession`` asks for it
    # unconditionally.
    #
    # They lived ONLY on ``mobile.tui_handle.TuiSessionHandle`` — the owner
    # path this PR deletes — and were not re-homed with the rest of it, so
    # every runtime published ``capabilities: []`` and refused every viewer:
    # no message could be sent in any session. Round 1 QA (Q2) and UX (U1)
    # both found it independently against the real binary.
    #
    # The delegation is DIRECT where the mobile bridge hops threads. That
    # bridge adapts a session living on Textual's loop from a foreign thread,
    # so it must marshal; this handle IS constructed on the runtime's own loop
    # and owns its session outright (see ``spawn_owned_session``), so the hop
    # would be a round trip to the thread already executing.

    @property
    def frontend_state_seed(self) -> Any:
        """Canonical state seed for full-TUI attach clients."""
        return self._session.frontend_state

    async def subscribe_frontend(self, on_update: Callable[[Any], None]) -> Any:
        """Snapshot and subscribe atomically, on the loop that publishes.

        ``Session.subscribe_frontend`` refreshes through the publishing path
        and returns the snapshot with its sequence number, which is what lets
        every client's exact-``+1`` gap check detect transport loss. Awaited
        rather than wrapped because the caller is already on this loop.
        """
        return self._session.subscribe_frontend(on_update)

    def subscribe_events(self, on_event: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Feed serialized AgentEvents to the runtime's v4 relay.

        Serialization happens here, on the loop that emits the event, so no
        pydantic object crosses a thread boundary; ``RuntimeServer._relay_event``
        only schedules onto its own loop, so the callback is safe to call
        inline and producer order is preserved.

        Without this the handshake can succeed and the viewer still sees
        nothing stream — the capability and the relay are two halves of one
        feature, which is why they are re-homed together.
        """

        def handler(event: AgentEvent) -> None:
            try:
                on_event(event.model_dump(mode="json"))
            except Exception:  # noqa: BLE001 — the relay is additive, never a gate
                logger.debug("runtime event serialization failed", exc_info=True)

        return self._session.subscribe(handler)

    def subscribe(self, on_projection: Callable[[], None]) -> Callable[[], None]:
        self._on_projection = on_projection

        def handler(event: AgentEvent) -> None:
            # Session events fire on the daemon loop; the fold is a plain
            # state machine, so folding inline is safe. Only the repaint push
            # crosses threads.
            self._fold.fold_event(event)
            self._refresh_state()
            self._refresh_todos()
            if isinstance(event, ModelChangeEvent):
                self._retry_naming_after_route_change()
            self._notify()

        unsubscribe = self._session.subscribe(handler)
        try:
            self._fold.fold_history(self._session.history())
        except Exception:  # noqa: BLE001 — history is a convenience, not a gate
            logger.debug("owned session history fold failed", exc_info=True)
        # Seed the live flag ONCE at attach (a phone subscribing mid-turn never
        # saw the AgentStartEvent). After this the fold's own lifecycle events
        # own ``streaming`` — see ``_reconcile_streaming``.
        self._reconcile_streaming()
        return unsubscribe

    async def prompt(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        command_id: str | None = None,
    ) -> str:
        self._check_loop_thread()
        if not command_id:
            # Only old in-process callers omit the v3 field. Minting here keeps
            # their local submission valid; all wire producers retain their id.
            command_id = str(uuid.uuid4())
        existing = self._prompt_commands.get(command_id)
        if existing is not None:
            await existing.admitted
            return "already admitted"
        if not self._command_reservations.reserve(command_id, kind="prompt"):
            return "already admitted"
        if self._disposing:
            self._command_reservations.reject(command_id)
            raise RuntimeError("session is closing; prompt was not admitted")
        if len(self._prompt_queue) >= MAX_QUEUED_PROMPTS:
            self._command_reservations.reject(command_id)
            raise RuntimeError(
                f"prompt queue is full ({MAX_QUEUED_PROMPTS}); wait for an admitted turn to start"
            )
        self._maybe_name_conversation(text)
        admitted: asyncio.Future[None] = self._loop.create_future()
        command = _PromptCommand(command_id, text, _image_blocks(images), admitted)
        position = len(self._prompt_queue) + 1
        legacy_prompt = "message_id" not in inspect.signature(self._session.prompt).parameters
        # Compatibility-only fake/third-party sessions predate durable
        # admission. Production Session exposes ``message_id`` and never takes
        # this early-receipt branch.
        if legacy_prompt:
            admitted.set_result(None)
        self._prompt_commands[command_id] = command
        self._prompt_queue.append(command)
        if self._prompt_drain_task is None or self._prompt_drain_task.done():
            self._prompt_drain_task = asyncio.ensure_future(self._drain_prompt_queue())
            self._prompt_drain_task.add_done_callback(self._observe_prompt_drain)
        # ACK is the durable transcript append, never insertion into this queue.
        await admitted
        self._command_reservations.accept(command_id)
        if legacy_prompt and position > 1:
            return f"prompt queued ({position})"
        return "prompt admitted"

    def _observe_prompt_drain(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return
        # The record's ``busy`` bit must settle when the LAST turn settles,
        # and the fold's events cannot say that: the final AgentEndEvent is
        # emitted while ``_is_streaming`` is still True (the pipeline resets
        # it in a ``finally`` after the loop returns), so the last
        # event-driven publish carries True and nothing after it fires.
        # Round 2 (U6) measured the consequence — a session that had fully
        # unwound kept reading ``busy=True`` until the next event. The drain
        # task's completion is the one moment that is by definition after
        # every turn it ran.
        self._publish_busy()

    def _maybe_name_conversation(self, text: str) -> None:
        """Name a still-unnamed conversation from its first real prompt.

        The TUI's OperatorApp runs the full naming/re-titling machinery; the
        phone only needs the FIRST-name half, because a mobile session opens
        unnamed and the list/header have nothing to show until it is named.
        Mirrors OperatorApp._maybe_name_conversation: skip low-signal openers
        (a bare "hi" is usually followed by the real ask, and latching on it
        would leave the session named after the greeting), fire at most once,
        and run the call as a background task so the title arrives ALONGSIDE
        the turn rather than after it.
        """
        from local_operator.session import naming

        if self._name_requested or naming.is_low_signal(text):
            return
        if getattr(self._session, "conversation_name", ""):
            # Already named (a restored session, or a prior prompt named it).
            self._name_requested = True
            return
        # Wear the opener on the phone immediately — same stand-in the TUI
        # band shows — so the list/header are not "untitled" for the whole
        # first turn (or forever, if the isolated naming call 429s).
        label = naming.provisional_title(text)
        if label:
            self._fold.set_state(conversation_name=label)
            self._notify()
        self._name_requested = True
        # Hold a strong reference until the task settles: a bare ensure_future
        # is only weakly held by the loop and can be collected before it runs.
        task = asyncio.ensure_future(self._name_conversation_worker(text))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _name_conversation_worker(self, text: str) -> None:
        """Ask the model for a title once, cheaply, off the turn's lock.

        ``session.complete_once`` is the same isolated, single-attempt, cheap
        completion the TUI's naming worker uses — a 429 here is swallowed by
        ``generate_title`` and cannot touch the turn. On success the title is
        stored on the session (which persists it), then the projection is
        refreshed and pushed so the phone's header and list update live.
        """
        from local_operator.session import naming

        try:
            title = await naming.generate_title(text, self._session.complete_once)
        except Exception:  # noqa: BLE001 — naming is decoration; never fail a turn
            logger.debug("mobile conversation naming failed", exc_info=True)
            return
        if not title or getattr(self._session, "conversation_name", ""):
            # No title, or a user/restore named it while we were in flight:
            # allow a later substantive prompt to retry only when still unnamed.
            if not getattr(self._session, "conversation_name", ""):
                self._name_requested = False
                self._pending_name_text = text
            return
        self._pending_name_text = ""
        self._session.set_conversation_name(title, user_set=False)
        self._refresh_state()
        self._notify()

    async def _drain_prompt_queue(self) -> None:
        """Run admitted ordinary prompts in owner order, one safe turn at a time.

        Accepted input remains in memory until its turn reaches Session.prompt;
        only that call emits the user row and persists it, giving every viewer
        one shared projection row rather than one optimistic echo per producer.
        """
        while self._prompt_queue:
            command = self._prompt_queue[0]
            try:
                parameters = inspect.signature(self._session.prompt).parameters
                if "message_id" in parameters:
                    fields: dict[str, Any] = {
                        "message_id": command.command_id,
                        "admitted": command.admitted,
                    }
                    if "producer_command_id" in parameters:
                        fields["producer_command_id"] = command.command_id
                    await self._session.prompt(command.text, command.images, **fields)
                else:
                    # Legacy tests/third-party handles have no admission seam;
                    # preserve their historical queue-insertion receipt. Real
                    # Session implementations always take the durable branch.
                    if not command.admitted.done():
                        command.admitted.set_result(None)
                    await self._session.prompt(command.text, command.images)
            except asyncio.CancelledError:
                if not command.admitted.done():
                    command.admitted.set_exception(
                        RuntimeError("session closed before the prompt was admitted")
                    )
                # The in-flight admission is popped by finally, so record its
                # terminal rejection here; dispose handles only those still queued.
                self._projection.streaming = False
                self._fold.note_prompt_rejected(
                    "session closed before the admitted prompt could complete"
                )
                self._notify()
                # Cancellation is control flow and must remain cancellation.
                raise
            except Exception as exc:  # noqa: BLE001 — admitted turns need terminal handling
                if not command.admitted.done():
                    self._command_reservations.reject(
                        command.command_id,
                        transfer_to_steer="already streaming" in str(exc),
                    )
                    command.admitted.set_exception(exc)
                # Provider, transcript, and tool failures are all terminal for
                # this one admission. Surface the failure asynchronously, then
                # continue in FIFO order without retrying the failed head.
                self._projection.streaming = self._session.is_streaming
                self._fold.note_prompt_rejected(str(exc))
                self._notify()
                logger.exception("mobile prompt failed after admission")
            except BaseException:
                # KeyboardInterrupt/SystemExit retain their process semantics;
                # the finally block still removes exactly the failed admission.
                logger.critical("mobile prompt drain terminated", exc_info=True)
                raise
            finally:
                self._prompt_queue.popleft()
                self._prompt_commands.pop(command.command_id, None)

    async def steer(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        command_id: str | None = None,
    ) -> str:
        self._check_loop_thread()
        command_id = command_id or str(uuid.uuid4())
        if not self._command_reservations.reserve(
            command_id,
            kind="steer",
            prompt_transfer=True,
        ):
            return "already admitted"
        # Images ride the steer too. Producer identity follows the queued user
        # row so a reconnect cannot inject the same correction twice.
        fields: dict[str, Any] = {}
        parameters = inspect.signature(self._session.steer).parameters
        if "message_id" in parameters:
            fields["message_id"] = command_id
        if "producer_command_id" in parameters:
            fields["producer_command_id"] = command_id
        try:
            self._session.steer(text, _image_blocks(images), **fields)
        except Exception:
            # No queue insertion means no durable acceptance exists; the same
            # producer identity must remain retryable after this terminal reject.
            self._command_reservations.reject(command_id)
            raise
        self._command_reservations.accept(command_id)
        self._projection.queued_count += 1
        # Register the echo under the id the session will actually announce, so
        # the drain's MessageStartEvent upgrades THIS row rather than being
        # matched against the transcript tail (issue #231) — a steer is
        # delivered at a later tool boundary, by which point assistant and tool
        # rows have pushed the echo out of any window. Only when `message_id`
        # really reached the session: a session that mints its own id would
        # announce something this key could never match, and the fold's tail
        # fallback is the correct behaviour there.
        self._fold.note_user_message(
            text,
            steer=True,
            message_id=command_id if "message_id" in fields else None,
        )
        self._notify()
        return "steering queued"

    async def receive_peer_message(
        self,
        text: str,
        *,
        mode: str = "mailbox",
        wake: bool = False,
        sender: dict[str, Any] | None = None,
    ) -> str:
        # This handle owns an in-process Session on the registrant's own loop,
        # so the coroutine can be awaited directly (unlike the TUI handle, which
        # must hop to the owner loop). Session.receive_peer_message does its own
        # transcript/context persistence; we only mirror the phone fold the way
        # steer() does, so an attached phone paints the peer card immediately
        # rather than waiting for the next MessageStartEvent.
        self._check_loop_thread()
        detail = await self._session.receive_peer_message(
            text, mode=mode, wake=wake, sender=sender or {}
        )
        self._fold.note_peer_message(text, sender=sender or {})
        self._notify()
        return detail

    async def abort(self) -> str:
        self._check_loop_thread()
        self._session.abort("stopped from mobile")
        return "stopping"

    async def set_model(self, provider: str, model_id: str) -> str:
        self._check_loop_thread()
        from local_operator.model.configure import build_model_spec

        spec = await asyncio.to_thread(build_model_spec, provider, model_id)
        # ``explicit``: the phone's model switch is a deliberate choice, so a
        # pinned fallback route is withdrawn even when it re-selects the model
        # the fallback displaced — see ``Session.set_model``.
        self._session.set_model(spec, explicit=True)
        self._refresh_state()
        return f"model: {self._projection.model_label}"

    async def set_effort(self, effort: str) -> str:
        self._check_loop_thread()
        spec = self._session.model
        if effort not in spec.reasoning_efforts:
            ladder = ", ".join(spec.reasoning_efforts) or "no rungs"
            raise ValueError(f"{spec.model_id} accepts {ladder}, not '{effort}'")
        self._session.set_model(spec.model_copy(update={"reasoning_effort": effort}))
        self._refresh_state()
        return f"effort: {effort}"

    async def slash(self, command: str, args: str) -> str:
        """Session-level slash commands — the ones with meaning off-terminal.
        TUI chrome (/help tables, /usage panels) is the phone UI's own job."""
        self._check_loop_thread()
        if command == "goal":
            stored = self._session.set_goal(args)
            return "goal updated" if stored else "goal cleared"
        if command == "compact":
            asyncio.ensure_future(self._session.compact_now())
            return "compacting context"
        raise ValueError(f"/{command} is terminal-only here")

    async def new_conversation(self) -> str:
        raise ValueError("start a new session from the session list")

    async def resume_session(self, session_id: str) -> str:
        raise ValueError("pick the session from the session list instead")

    async def approval_answer(self, request_id: str, approved: bool, remember: bool) -> str:
        await self._resolve_pending(request_id, approved)
        return "approved" if approved else "denied"

    async def ask_answer(
        self, request_id: str, value: str, question_index: int | None = None
    ) -> str:
        # ``question_index`` is accepted for protocol parity with the TUI handle
        # (U8 guard). An owned session assigns a DISTINCT request_id per question
        # (the gate loops one future per question), so the request_id is already
        # the per-question identity: a stale tap targets an id whose future is
        # gone and is rejected below. No separate index check is needed here.
        del question_index
        # Resolve with the QUESTION id the harness asked under — never our
        # request id, which the harness never saw.
        question_id = self._pending_question_ids.get(request_id, request_id)
        try:
            await self._resolve_pending(request_id, {question_id: [value]} if value else None)
        except ValueError as exc:
            # Human, reconciling copy: a stale tap means another front end won.
            raise ValueError("that question was already answered") from exc
        return "answered"

    def has_admitted_command(self, command_id: str) -> bool:
        """Has this session already durably admitted ``command_id``?

        The DURABLE half of idempotency, and the half that survives a restart.
        ``prompt``'s own ``_prompt_commands`` map dedupes within one runtime's
        lifetime, but the case this exists for crosses lifetimes: a sender that
        crashed after the row was appended, or a wake supervisor that re-fired
        an occurrence it could not confirm, engages a NEW runtime whose
        in-memory map is empty. The transcript is what remembers.
        """
        if not command_id:
            return False
        transcript = getattr(self._session, "transcript", None)
        checker = getattr(transcript, "has_admitted_command", None)
        if not callable(checker):
            return False
        try:
            return bool(checker(command_id))
        except Exception:  # noqa: BLE001 — a dedupe probe must never fail a turn
            logger.debug("admitted-command probe failed", exc_info=True)
            return False

    def _gate_timeout_s(self) -> float | None:
        """How long THIS gate may wait. ``None`` means never time out.

        PARK, do not deny — the change the detached model forces. The old
        30-second cap assumed a gate only ever waited on a phone that might be
        in a pocket, so denying was the kind thing: the turn moved on instead
        of pinning a tool slot forever. Under this model the same wait usually
        means "the user stepped away from a session that is still running",
        and denying their write tool after thirty seconds answers a question
        nobody asked. The question is now held for
        ``runtime.unattended_gate_timeout`` hours (default 24, so it spans an
        overnight) and the user answers it when they come back.

        The short cap survives for exactly the case it was written for: no
        client can present the card at all. With a viewer attached, or a phone
        watching, something is showing the question to someone; with nothing
        attached the card exists only in this process's memory, and a bounded
        wait is still the honest behaviour there.
        """
        if self._registrant is None:
            # No control socket at all: an embedded or reduced host, where the
            # card exists only in this process's memory and no front end can
            # ever be attached to it. This is the case the ordinary cap was
            # written for, and it keeps that constant meaningful — shortening
            # it still shortens a gate, rather than being quietly ignored
            # because the policy stopped reading it.
            return PENDING_REQUEST_TIMEOUT_S
        parked = self._parked_timeout_s()
        if self._watching_surfaces():
            # A terminal OR the phone is watching: the card is on a screen
            # somebody has. Note this is deliberately kind-agnostic — the
            # question is "can this reach a person", and both surfaces can.
            return parked
        # Nothing is presenting the card. A parked gate is still preferable to
        # a denial when the user has an out-of-band way to be told about it
        # (the desktop notification), so the configured cap applies here too —
        # the short cap is reserved for the case where notification is off and
        # nobody could learn of the question at all.
        #
        # REACHABILITY IS NOT THE NOTIFY FLAG ALONE. Once announcements route
        # by surface, "reachable" means "some surface is watching OR an OS
        # notification can actually be delivered" — the watching case is
        # handled above, and this is the remaining out-of-band leg.
        from local_operator.tui.notify import notifications_enabled

        try:
            reachable = notifications_enabled()
        except Exception:  # noqa: BLE001 — an unreadable setting is "not reachable"
            reachable = False
        return parked if reachable else PENDING_REQUEST_TIMEOUT_S

    def _parked_timeout_s(self) -> float | None:
        """The configured park duration, never SHORTER than the ordinary cap.

        ``PENDING_REQUEST_TIMEOUT_S`` is the floor rather than a separate
        branch, and that keeps one property true: whatever this returns, a gate
        always waits at least as long as it did before this change. It is also
        what keeps the constant meaningful — a test (and a user) that shortens
        it to make a gate expire quickly still gets a gate that expires
        quickly, instead of silently waiting the configured 24 hours because
        the policy stopped reading the constant at all.
        """
        hours = self._unattended_gate_hours()
        if hours <= 0:
            return None
        return max(PENDING_REQUEST_TIMEOUT_S, float(hours) * 3600.0)

    def _unattended_gate_hours(self) -> int:
        """``runtime.unattended_gate_timeout`` in hours; 0 means never."""
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            values = ConfigManager(config_dir()).get_config().values
            section = values.get("runtime")
            if isinstance(section, dict) and "unattended_gate_timeout" in section:
                return max(0, int(section["unattended_gate_timeout"]))
        except Exception:  # noqa: BLE001 — a bad setting must not pin a turn
            logger.debug("could not read runtime.unattended_gate_timeout", exc_info=True)
        return DEFAULT_UNATTENDED_GATE_TIMEOUT_H

    def _install_interactivity_probe(self) -> None:
        """Let the MODEL know whether anyone can answer a question.

        The runtime is the only component that knows — it owns the control
        socket's connection table — and the session's goal-state holder is
        the established seam for live session state reaching the next turn's
        prompt (the same route ``/goal`` and ``/team`` use). Installing a
        probe rather than pushing a value keeps this O(1) in attach churn:
        the prompt closure asks at turn start, so a viewer that comes and
        goes fifty times costs exactly one line of context, and no transcript
        row is ever written for an attach or a detach.
        """
        holder = getattr(self._session, "_goal_state", None)
        if holder is None or not hasattr(holder, "interactive_probe"):
            return
        try:
            holder.interactive_probe = lambda: bool(self._watching_surfaces())
        except Exception:  # noqa: BLE001 — an unsettable holder is not fatal
            logger.debug("could not install the interactivity probe", exc_info=True)

    def _watching_surfaces(self) -> frozenset[str]:
        """Which kinds of surface are watching, for notification routing.

        Falls back to the attach COUNT when the registrant is too old to
        answer by kind: a runtime published by an older release still knows
        how many terminals are attached, and treating "some terminal" as
        "something is watching" preserves the previous behaviour exactly
        rather than inventing a toast that release never sent.
        """
        server = self._registrant
        reader = getattr(server, "watching_surfaces", None)
        if callable(reader):
            try:
                return frozenset(cast("frozenset[str]", reader()))
            except Exception:  # noqa: BLE001 — routing must never raise into a gate
                logger.debug("could not read the watching surfaces", exc_info=True)
        return frozenset({"attach"}) if self._attached_clients() > 0 else frozenset()

    def _session_id_for_resume(self) -> str:
        """The id a notification's click-through reopens, best effort."""
        server = self._registrant
        record = getattr(server, "record", None)
        session_id = getattr(record, "session_id", "") or ""
        if session_id:
            return str(session_id)
        return str(getattr(self._session, "session_id", "") or "")

    def _attached_clients(self) -> int:
        """How many front ends could present a card right now."""
        server = self._registrant
        counter = getattr(server, "attach_clients", None)
        if not callable(counter):
            return 0
        try:
            return int(cast(int, counter()))
        except Exception:  # noqa: BLE001
            return 0

    async def _record_gate_timeout(self, tool: str, description: str) -> None:
        """Append the row that says NOBODY WAS THERE.

        A denial and an expiry look identical to the model otherwise, and they
        are different facts: one is the user's decision, the other is the
        absence of one. Without this row the next turn reads "the user denied
        this" and adjusts its plan around a choice nobody made.
        """
        transcript = getattr(self._session, "transcript", None)
        append = getattr(transcript, "append_message", None)
        if not callable(append):
            return
        try:
            # A MESSAGE entry carrying a CustomMessage, not `append_custom`.
            # `build_llm_history` ignores custom ENTRIES by design, so the row
            # this method's own docstring promises would reach the model
            # reached nobody — not the model, and not the viewer that replays
            # the same history (round 1, D2/U2). A wake receipt has always
            # taken this shape for exactly that reason.
            from local_operator.harness.types import CustomMessage

            result = append(
                CustomMessage(
                    custom_type=GATE_TIMEOUT_CUSTOM_TYPE,
                    attribution="system",
                    details={
                        "tool": tool,
                        "description": description,
                        "waited_s": self._gate_timeout_s() or 0.0,
                    },
                )
            )
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 — the denial still stands
            logger.debug("could not record the unattended gate timeout", exc_info=True)

    def reannounce_pending(self) -> None:
        """Re-run the announcement for a gate that is STILL parked.

        Called by the registrant when the last viewer detaches. The routing
        decision is made once, when the gate opens, so a gate opened while
        somebody was watching correctly sent no toast — and then the user
        closed the terminal and was never told (round 3, B2). This re-runs
        the decision against the surfaces watching NOW.
        """
        parked = self._parked_announcement
        if parked is None:
            return
        self._announce_pending(*parked)

    def _announce_pending(self, kind: str, title: str, detail: str) -> None:
        """Publish that this session is WAITING FOR A PERSON, and say so.

        A parked gate holds ~283 MB resident for up to a day, so the cost has
        to be findable: the record's ``pending`` field puts it in `lop
        sessions` and sorts it first in the picker, and the notification tells
        the user out of band. A parked gate nobody can see is a process nobody
        can find.
        """
        # Remembered so a later detach can re-run this decision (B2). Held
        # until the gate settles, which is the only point the question stops
        # being owed.
        self._parked_announcement = (kind, title, detail)
        server = self._registrant
        setter = getattr(server, "set_record_pending", None)
        if callable(setter):
            try:
                setter(kind)
            except Exception:  # noqa: BLE001
                logger.debug("could not publish the pending state", exc_info=True)
        # ROUTE TO WHATEVER IS WATCHING; fall out to the OS only when nothing
        # is. The old test was `attached_clients() > 0`, which counts only
        # terminals — so a user whose PHONE was watching got a desktop toast
        # for a card already on their phone, and the desktop was the one
        # surface they were not looking at.
        #
        # Both watching surfaces deliver this card already, by different
        # means: an attached terminal paints it in-band, and the mobile relay
        # (a ``daemon`` client) carries it in the projection push that
        # ``_notify`` has already made. Neither needs a second channel, which
        # is why this is a routing decision and not a new transport.
        surfaces = self._watching_surfaces()
        if surfaces:
            return
        try:
            from local_operator.tui.notify import detached_notify

            name = getattr(self._session, "conversation_name", "") or "lop"
            detached_notify(
                f"{name} needs you",
                f"{title}: {detail}".strip().rstrip(":"),
                session_id=self._session_id_for_resume(),
            )
        except Exception:  # noqa: BLE001 — a toast must never affect the gate
            logger.debug("detached notification failed", exc_info=True)

    def _announce_settled(self) -> None:
        """Clear the waiting-for-a-person state once the gate resolves."""
        # Cleared FIRST and unconditionally: the question is no longer owed,
        # so a later detach must not resurrect a toast for it (B2).
        self._parked_announcement = None
        server = self._registrant
        setter = getattr(server, "set_record_pending", None)
        if not callable(setter):
            return
        try:
            setter(None)
        except Exception:  # noqa: BLE001
            logger.debug("could not clear the pending state", exc_info=True)

    def _publish_pending_gate(self) -> None:
        """Mirror the fold's FRONT card into the canonical full-TUI contract.

        There are two consumers of a parked gate and they read different
        places. The phone reads the projection fold (`push_pending` /
        `pop_pending`, a queue so a parallel tool batch keeps one card per
        approval). A full TUI attaching reads
        `Session.frontend_state.pending_gate` — and nothing on this path ever
        set it, so the user summoned by the toast arrived at a session with
        no question on screen and no way to answer it (round 3, U8). The
        gate then expired 24 h later as a denial.

        `TuiSessionHandle._publish_pending_gate` always published to both;
        the capability did not survive gate ownership moving into the
        runtime. Publishing the FRONT of the queue (rather than replacing the
        queue with a single slot) keeps the concurrent-approval property the
        fold exists for while giving the attach contract the card it needs.
        """
        store = getattr(self._session, "_frontend_state_store", None)
        if store is None:
            return
        try:
            # `_sync_pending` already fronts the queue onto `projection.pending`
            # for the phone's "1 of N" badge; reuse that rather than reaching
            # into the queue, so both surfaces can never disagree about which
            # card is current.
            front = self._projection.pending
            store.mutate(pending_gate=front.to_json() if front is not None else None)
        except Exception:  # noqa: BLE001 — a card is never worth failing a gate
            logger.debug("could not publish the pending gate", exc_info=True)

    async def complete_aside(self, turns: list[dict[str, Any]]) -> str:
        """Run an off-record provider request against this session.

        The aside seam: a viewer asks a question that must NOT enter the
        durable conversation (the model picker's "explain this model", the
        mobile quick-ask). It runs on the authoritative session because it
        needs the real model, credentials and context — which is why it
        cannot be answered viewer-side.

        Found by the post-U9 migration audit rather than by a review: without
        it every aside answered "this owner cannot run off-record requests".
        """
        from local_operator.harness.types import Message

        messages = [Message.model_validate(turn) for turn in turns]
        return await self._session.complete_aside(messages)

    async def adopt_aside(self, messages: list[dict[str, Any]]) -> str:
        """Fork a viewer's aside exchange into the durable conversation."""
        from local_operator.harness.types import Message

        parsed = [Message.model_validate(message) for message in messages]
        await self._session.adopt_aside(parsed)
        self._notify()
        return f"forked {len(parsed) // 2} aside exchange(s) into the chat"

    async def recall_steer(self, command_id: str) -> str:
        """Recall one queued steer by the Message id its producer supplied.

        The viewer's "unsend" for a steer that has not been consumed yet. The
        reservation is rejected as well as the message recalled, so the
        command id cannot be admitted later by a racing durable event.
        """
        recalled = False
        for message in self._session.queued_steering():
            if str(getattr(message, "id", "")) == command_id:
                recalled = bool(self._session.recall_steering(message))
                break
        if not recalled:
            raise ValueError("that steering message is no longer queued")
        self._command_reservations.reject(command_id)
        self._notify()
        return "steering recalled"

    async def slash_images(
        self,
        command: str,
        args: str,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        """Run a slash command that carries image attachments.

        The old handle ran the command's UI in the OWNER's terminal and
        returned a receipt. A runtime has no terminal, so the only commands
        reachable this way are the routed ones — this defers to the same
        dispatcher `run_slash_authoritative` uses and renders its notice as
        the receipt, rather than failing the request outright.

        Images are accepted and ignored for the routed set (none of
        `/goal`, `/rename`, `/approvals`, `/compact` consumes an attachment);
        that is stated here so a future image-consuming routed command is
        added deliberately rather than silently dropping its payload.
        """
        from local_operator.session.frontend_state import SlashResult

        result = await self._slash_result(command, args, SlashResult)
        text = getattr(result, "text", "") or f"ran /{command}"
        return str(text)

    def cancel_subagents_count(self) -> int:
        """Cancel every running subagent and return the REAL count.

        Esc's second job. `RemoteSession.cancel_subagents` swallows a failure
        to ``stopped = -1``, so a handle without this method makes Esc quietly
        do less than it says on a detached session (round 3, U9) — the turn
        ends but the children keep burning tokens.

        Re-homed from ``TuiSessionHandle``: that version hopped to the app
        loop because the session lived there. Here the session is on THIS
        loop, so the call is direct — the same reason the rest of this class
        does not need ``run_coroutine_threadsafe``.
        """
        cancel = getattr(self._session, "cancel_subagents", None)
        if not callable(cancel):
            return 0
        result = cancel("interrupted")
        stopped = result if isinstance(result, int) else 0
        self._notify()
        return stopped

    async def run_slash_authoritative(
        self,
        command: str,
        args: str,
        images: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run one shared slash command against the session and answer as data.

        The owner-side backend for a viewer's ``route_shared_slash``. These
        commands MUTATE SHARED SESSION STATE (the goal, the model, the
        approval mode, the conversation name), so they have to run where the
        session lives; a viewer-local copy would either drive nothing or
        drive a second, divergent copy of the orchestration state.

        Re-homed from ``TuiSessionHandle``, which delegated to
        ``OperatorApp.run_slash_authoritative``. A detached runtime has NO
        app — that is the whole point of this release — so the handlers are
        implemented here against the session directly. Without them every
        typed slash command answered ``this owner cannot run typed slash
        results`` on every detached session (round 3, U9): eleven commands,
        in developer vocabulary, on the branch's main path.

        The returned shape is a ``SlashResult`` dump the INVOKING terminal
        renders locally, so the receipt reads the same whether the session is
        local or detached.
        """
        from local_operator.session.frontend_state import SlashResult

        result = await self._slash_result(command, args, SlashResult)
        return result.model_dump(mode="json")

    async def _slash_result(self, command: str, args: str, SlashResult: Any) -> Any:
        """Dispatch one routed slash command. Mirrors ``OperatorApp._slash_result``.

        Only the commands a viewer ROUTES reach here; process- and
        terminal-local ones (``/quit``, ``/resume``, pickers) never leave the
        viewer. Anything not handled falls through to an honest notice rather
        than the transport's ``unknown op``, because a user typing a command
        this runtime does not implement needs to know what to do instead.
        """
        session = self._session
        if command == "goal":
            return self._goal_slash(session, args, SlashResult)
        if command == "rename":
            return self._rename_slash(session, args, SlashResult)
        if command == "effort":
            return self._effort_slash(session, args, SlashResult)
        if command == "approvals":
            return self._approvals_slash(session, args, SlashResult)
        if command == "compact":
            return self._compact_slash(session, SlashResult)
        if command == "loop":
            # The goal loop is an OperatorApp worker; a detached runtime has
            # no loop to stop, so saying "no loop is running" is the truth
            # rather than a stub. A loop started in a viewer is stopped there.
            if args.lower() in ("stop", "cancel", "abort"):
                return SlashResult(kind="notice", text="no loop is running", style="info")
            return SlashResult(
                kind="notice",
                text="/loop runs from an attached terminal — reattach and run it there",
                style="warning",
            )
        return SlashResult(
            kind="notice",
            text=f"/{command} is not available on a detached session — reattach to run it",
            style="warning",
        )

    def _goal_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        if not hasattr(session, "set_goal"):
            return SlashResult(kind="notice", text="session is still starting…", style="warning")
        if not arg:
            current = getattr(session, "goal", "")
            text = f"goal: {current}" if current else "no goal set — /goal <text> to set one"
            return SlashResult(kind="notice", text=text, style="info")
        if arg.lower() in ("clear", "none", "reset"):
            session.set_goal("")
            self._notify()
            return SlashResult(kind="notice", text="goal cleared", style="info")
        stored = session.set_goal(arg)
        self._notify()
        from local_operator.session.goal import MAX_GOAL_CHARS

        if len(stored) == MAX_GOAL_CHARS and len(arg.strip()) > MAX_GOAL_CHARS:
            return SlashResult(
                kind="notice",
                text=(
                    f"goal set — shortened to the {MAX_GOAL_CHARS}-character cap, "
                    "applies from the next turn"
                ),
                style="warning",
                data={"stored": stored},
            )
        return SlashResult(
            kind="notice",
            text="goal set — applies from the next step",
            style="info",
            data={"stored": stored},
        )

    def _rename_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        name = (arg or "").strip()
        if not name:
            current = getattr(session, "conversation_name", "") or ""
            text = f"name: {current}" if current else "no name set — /rename <text> to set one"
            return SlashResult(kind="notice", text=text, style="info")
        setter = getattr(session, "set_conversation_name", None)
        if not callable(setter):
            return SlashResult(kind="notice", text="session is still starting…", style="warning")
        stored = setter(name)
        # The name is on the discovery record, so `lop sessions` and the
        # picker must see the rename without waiting for the next heartbeat.
        # `_notify` refreshes the projection; the registrant owns the record.
        self._notify()
        republish = getattr(self._registrant, "_republish", None)
        if callable(republish):
            try:
                republish()
            except Exception:  # noqa: BLE001 — a stale name is not worth a failure
                logger.debug("could not republish the renamed record", exc_info=True)
        return SlashResult(kind="notice", text=f"renamed to {stored or name}", style="info")

    def _effort_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """Report the reasoning effort; changing it stays with the viewer.

        The app's version drives `_apply_effort`, which reaches into the
        model picker's widget state and the machine's saved default — neither
        of which exists in a runtime. Reporting is genuinely session state
        and is answered here; the mutation is declined in the same vocabulary
        `/approvals default` uses for machine-local settings, rather than
        pretending to apply and silently not.
        """
        spec = getattr(session, "model", None)
        label = getattr(session, "model_label", "") or "this model"
        if spec is None:
            return SlashResult(kind="notice", text="session is still starting…", style="warning")
        current = getattr(spec, "reasoning_effort", None)
        if not arg.strip():
            text = (
                f"effort: {current} on {label}"
                if current
                else f"effort is not adjustable on {label}"
            )
            return SlashResult(kind="notice", text=text, style="info")
        return SlashResult(
            kind="notice",
            text="/effort changes the model selection, which lives in a terminal — "
            "reattach and run it there",
            style="warning",
        )

    def _approvals_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """Report or switch the gate the RUNTIME's tools actually consult.

        `self._auto_approve` is the real gate here (see `_install_gates`), so
        unlike the viewer's own widget flag this switch is the one the engine
        honours. The persist half is declined for the same machine-locality
        reason the app gives: a default belongs to the terminal that launches
        sessions, not to a runtime that outlives it.
        """
        argument = (arg or "").strip().lower()
        if argument == "default" or argument.startswith("default "):
            return SlashResult(
                kind="notice",
                text="/approvals default persists to the local machine's config — run it "
                "on a terminal; /approvals ask|auto switches this session now",
                style="warning",
            )
        if not argument:
            live = "auto" if self._auto_approve else "ask"
            effect = (
                "every tool runs without asking"
                if self._auto_approve
                else "write and command tools prompt before running"
            )
            return SlashResult(
                kind="notice",
                text=f"tool approvals: {live} — {effect}",
                style="warning" if self._auto_approve else "info",
            )
        if argument in ("ask", "on", "prompt"):
            wanted_auto = False
        elif argument in ("auto", "off", "yolo"):
            wanted_auto = True
        else:
            return SlashResult(
                kind="notice",
                text=f"unknown approval mode {argument!r} — use ask or auto",
                style="warning",
            )
        self._auto_approve = wanted_auto
        self._notify()
        return SlashResult(
            kind="notice",
            text=(
                "tool approvals: auto — every tool runs without asking"
                if wanted_auto
                else "tool approvals: ask — write and command tools prompt before running"
            ),
            style="warning" if wanted_auto else "info",
        )

    def _compact_slash(self, session: Any, SlashResult: Any) -> Any:
        """Kick the real pass; the ACCEPT receipt is the answer.

        A long conversation compacts for minutes, which cannot be awaited
        inside a request/response op without the socket reporting failure for
        work that is actually running. The settled outcome reaches every
        terminal through the canonical compaction events instead — the same
        vocabulary a local trigger produces.
        """
        compact = cast(
            "Callable[[], Coroutine[Any, Any, Any]] | None", getattr(session, "compact_now", None)
        )
        if not callable(compact):
            return SlashResult(
                kind="notice",
                text="no session yet — there is no context to compact",
                style="warning",
            )
        task = self._loop.create_task(compact())
        # Same retention discipline as the gate tasks above: an un-retained
        # task can be collected mid-flight and the pass would vanish silently.
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return SlashResult(kind="notice", text="compacting context…", style="info")

    async def job_trajectory(self, job_id: str, offset: int, limit: int) -> dict[str, Any]:
        """One page of a child job's retained event window.

        Serves the viewer's on-demand fetch: attach snapshots ship no
        trajectories (they overflow the socket's line limit), so a follower
        that opens a subagent page asks for the rows here instead.

        ``total`` is the CURRENT retained length and ``base_seq`` the identity
        stamp of the first retained row (``TRAJECTORY_SEQ_KEY``). Both are
        needed because the window ROTATES: ``AsyncJob.trajectory`` evicts from
        the front past ``TRAJECTORY_CAP``, so an offset the caller computed one
        page ago may now name a different event. The viewer compares
        ``base_seq`` across pages and restarts the fetch when the floor moved,
        which is the same eviction problem ``job_trajectory_replacements``
        solves for the delta stream.
        """
        job = self._session.jobs.get(job_id)
        rows = list(getattr(job, "trajectory", None) or []) if job is not None else []
        total = len(rows)
        page = rows[offset : offset + limit]
        first = rows[0] if rows else None
        base_seq = first.get(TRAJECTORY_SEQ_KEY) if isinstance(first, dict) else None
        return {
            "job_id": job_id,
            "rows": page,
            "offset": offset,
            "total": total,
            "base_seq": base_seq if isinstance(base_seq, int) else None,
            # A job swept from the ledger is distinguishable from one with no
            # events yet: the page renders "no longer on the ledger" for the
            # first and "no activity" for the second.
            "known": job is not None,
        }

    async def refresh(self) -> None:
        self._refresh_state()
        self._refresh_todos()
        # Command boundary: safe to reconcile from the session flag (no
        # terminal event is mid-flight here, unlike the per-event path).
        self._reconcile_streaming()

    # -- internals ----------------------------------------------------------------

    def _notify(self) -> None:
        self._publish_busy()
        # Published HERE rather than beside each push/pop: `_notify` already
        # follows every gate mutation, so one seam keeps the phone's fold and
        # the full TUI's `pending_gate` in step and a future gate cannot
        # forget to publish one of the two (round 3, U8).
        self._publish_pending_gate()
        if self._on_projection is not None:
            self._on_projection()

    def _publish_busy(self) -> None:
        """Keep the record's ``busy`` bit in step with the session.

        `RuntimeServer.set_busy` existed with NO CALLER (round 1, U2), so the
        picker's running marker was inert: a runtime grinding through a long
        turn with no terminal open — the exact thing this release exists to
        make possible — was indistinguishable from an idle one.

        Driven from `_notify` rather than from a new observer because this
        already runs on every session event, on the session's own loop, and it
        is where the projection's own liveness is refreshed. `set_busy`
        de-duplicates, so the republish costs one comparison per event and a
        staged write only on an actual transition.

        `is_busy()` is the authority rather than a second flag: it is the same
        predicate the reaper uses to decide whether this runtime may exit, so
        the marker and the residency decision can never disagree.
        """
        server = self._registrant
        setter = getattr(server, "set_busy", None)
        if not callable(setter):
            return
        try:
            setter(self.is_busy())
        except Exception:  # noqa: BLE001 — a stale marker is not worth a turn
            logger.debug("could not publish the busy state", exc_info=True)

    def _check_loop_thread(self) -> None:
        """The registrant calls handle methods on its own loop; owned sessions
        live on the daemon loop. The registrant hops via ``run_coroutine_threadsafe``
        in the daemon's spawn path, so reaching here means we ARE on the right
        loop — assert it in dev, and let the call proceed (asyncio detects real
        cross-loop misuse loudly)."""

    def _refresh_state(self) -> None:
        self._fold.set_state(
            model_label=_effective_label(self._session),
            model_selector=_selector(self._session),
            effort=_current_effort(self._session),
            effort_ladder=_ladder(self._session),
            conversation_name=getattr(self._session, "conversation_name", "") or None,
            # NOTE: ``streaming`` is deliberately NOT set here. This runs after
            # every folded event, and the session clears ``is_streaming`` only
            # in the turn's ``finally`` -- AFTER the AgentEndEvent has been
            # emitted and folded. Reading the still-True flag on that terminal
            # event re-stuck the projection to True with no later event to fix
            # it, pinning the phone to "in progress" forever. The fold's own
            # lifecycle events are authoritative; ``_reconcile_streaming``
            # covers attach and command boundaries.
        )

    def _reconcile_streaming(self) -> None:
        """Seed/align ``streaming`` from the session flag at attach and command
        boundaries, delegating the safety rule to ``ProjectionFold`` (which
        ignores the flag once it has folded a turn-terminal event -- see
        ``ProjectionFold.reconcile_streaming``). NEVER called from the
        per-event handler: the fold's lifecycle events own ``streaming``
        there."""
        self._fold.reconcile_streaming(bool(getattr(self._session, "is_streaming", False)))

    def _retry_naming_after_route_change(self) -> None:
        """Re-fire a failed naming attempt once a fallback is actually serving.

        Isolated naming has no fallback chain, so a quota 429 on the primary
        returns None while the turn is still pinning the rescue route. The
        opener is stashed; this spends it the moment the serving model exists.
        """
        pending = self._pending_name_text
        if not pending or getattr(self._session, "conversation_name", ""):
            return
        self._pending_name_text = ""
        self._maybe_name_conversation(pending)

    def _refresh_todos(self) -> None:
        try:
            from local_operator.tools.builtin import TODO_STORE

            self._fold.set_todos(list(TODO_STORE.get(self._session.session_id, [])))
        except Exception:  # noqa: BLE001 — todos are a panel, never a failure
            logger.debug("todo refresh failed", exc_info=True)


def _effective_label(session: Any) -> str:
    """``provider/model`` of the model actually serving requests.

    A display that reads ``session.model_label`` during a provider fallback
    names a model that is not answering — the stale composer chip.
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


async def spawn_owned_session(
    loop: asyncio.AbstractEventLoop,
    *,
    cwd: str,
    provider: str | None = None,
    model_id: str | None = None,
    resume: str | None = None,
) -> OwnedSessionHandle:
    """Build a session for the phone with the CLI's composition root.

    ``resume`` names an existing session id to reopen: it flows into
    ``args.resume`` exactly as the CLI's ``--resume`` does, so the factory
    reuses that transcript directory and the session replays its history —
    the phone's "open this past conversation" button.
    """
    # These imports MUST stay function-local, and ``create_session`` most of
    # all. Do not "tidy" them to the top of the file.
    #
    # The operative reason TODAY is startup cost: ``session_factory`` is the
    # composition root and pulls the engine, the registry and the provider
    # layer behind it. Hoisting it would put the whole harness on the import
    # graph of anything that merely touches this module, and this package sits
    # on the CLI startup path.
    #
    # The second reason is latent rather than current, and is stated precisely
    # so nobody "disproves" it and hoists the line. There is no cycle at this
    # commit — ``session_factory`` has no module-scope ``local_operator.session.*``
    # imports (they are function-local or TYPE_CHECKING), and nothing under
    # ``local_operator/session/`` imports this package. A cycle becomes REAL the
    # moment either of those changes, which later PRs in this series plan to do
    # (``session.session``/``session_factory`` reaching into the runtime package
    # for engagement and arbitration). Because this module now lives *under*
    # ``session/``, that day it would surface as a partially initialised module
    # at CLI startup rather than as a clean ImportError at the edit — so the
    # function-local form is what keeps that future change cheap.
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir
    from local_operator.session_factory import create_session

    config_directory = config_dir()
    config_manager = ConfigManager(config_dir=config_directory)
    credential_manager = CredentialManager(config_dir=config_directory)
    agent_registry = AgentRegistry(config_dir=config_directory)

    # The owner's saved tool-approval default. The TUI reads the SAME key at
    # boot (OperatorApp._load_approvals_default) and adopts ``auto`` as
    # "approve every tier"; a phone-started session must honour it too, or a
    # device set to full-auto still pops an approval card the desktop would
    # not. ``yolo`` stays False so the gate is INSTALLED (a per-session toggle
    # can still switch to asking); the handle short-circuits it when auto.
    try:
        approval_mode = (
            str(config_manager.get_config_value("tool_approval_mode", "ask")).strip().lower()
        )
    except Exception:  # noqa: BLE001 — a missing/odd config means "ask", never a crash
        logger.debug("could not read tool_approval_mode; defaulting to ask", exc_info=True)
        approval_mode = "ask"
    auto_approve = approval_mode == "auto"

    args = argparse.Namespace(
        hosting=provider,
        model=model_id,
        agent_name=None,
        agent_id=None,
        yolo=False,
        train=False,
        resume=resume,
    )
    session = await create_session(
        args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=False,
        cwd=cwd,
    )
    return OwnedSessionHandle(session, loop, cwd=cwd, auto_approve=auto_approve)
