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
import time
import uuid
from asyncio import InvalidStateError
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Iterable, cast

from local_operator.harness.approval import (
    GATE_TIMEOUT_CUSTOM_TYPE as _GATE_TIMEOUT_CUSTOM_TYPE,
)
from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY
from local_operator.harness.types import AgentEvent, ModelChangeEvent

if TYPE_CHECKING:
    from local_operator.harness.types import ImageContent
    from local_operator.secrets.session import SessionRegistration

from local_operator.mobile.command_reservation import CommandReservations
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    ask_pending_request,
)
from local_operator.session.runtime.server import SessionHandle
from local_operator.session.runtime.server import (
    image_blocks_in_thread as _image_blocks_async,
)
from local_operator.session.runtime.types import (
    RUNNING_SUBAGENT_STATUSES,
    runtime_must_complete,
)
from local_operator.session.transcript import TRANSCRIPT_FILENAME

logger = logging.getLogger(__name__)


#: How many loop turns a dispatched admission is given to surface a SYNCHRONOUS
#: refusal before it is left to run detached. ``prompt`` raises its reportable
#: refusals (closing session, full queue, rejected reservation) before its first
#: await, so one turn suffices; the margin only guards a future edit that adds
#: another await to that prelude. Turns, not seconds — see
#: ``_admit_without_waiting_for_the_turn``.
_ADMISSION_PRELUDE_TURNS = 3

#: How long the ``abort`` op waits for cancelled children to actually go before
#: it reports what is left. Cancellation is one fire-and-forget task per child,
#: so a count taken at dispatch describes intent rather than outcome — the
#: overstatement review round 1 (MAJOR-1) measured. A child normally settles in
#: well under a tick, and the poll exits the moment the roster drains, so this
#: is a CEILING for the wedged case rather than a cost every stop pays. Kept
#: short because the caller (a phone, a supervisor) is blocked on the ack: a
#: child that will not die must delay the receipt by a beat, never hold it.
_ABORT_SETTLE_BUDGET_S = 1.0
_ABORT_SETTLE_POLL_S = 0.02


def _log_detached_admission(task: "asyncio.Task[str]") -> None:
    """Never let a dispatched admission become an unretrieved exception.

    The receipt has already been returned by the time this runs, so there is no
    caller left to tell: a failure here is a log line, and swallowing it
    silently is what would make the next occurrence undiagnosable.
    """
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.warning("a completed slash action's turn failed after admission: %s", error)


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


def _already_bounded(images: Any) -> bool:
    """Whether ``images`` are decoded ``ImageContent`` rather than wire dicts.

    The wire carries ``[{"data_b64": ..., "mime_type": ...}]``; in-process
    callers that have already run the bound pass ``ImageContent`` blocks. Both
    reach ``prompt``/``steer``, and telling them apart is what lets an
    already-bounded caller keep the prelude await-free (see ``prompt``).

    An EMPTY list is "already bounded" -- there is nothing to decode, and the
    hop would only cost a suspension point.
    """
    if not images:
        return True
    return not isinstance(images[0], Mapping)


def _question_prose(question: Any) -> str:
    """The human-readable prose of an ask, for surfaces that ANNOUNCE it.

    One function rather than two ``getattr`` calls because the two call sites
    below drifted together: both read a ``text`` attribute
    :class:`~local_operator.harness.types.AskQuestion` does not have and, with
    ``extra="forbid"``, cannot ever be given. Naming the read once means a
    future rename of the model's field breaks in one place instead of silently
    degrading to the fallback in several.

    ``getattr`` rather than ``question.question`` is DEFENSIVE, not a caller
    requirement: every current call site receives ``list[AskQuestion]`` straight
    from ``AskUserFn``, and no path hands this a duck-typed object (the phone's
    decode runs the other way, through ``ask_pending_request``). It is written
    this way to match the sibling read in ``mobile/types.py`` and because a gate
    announcement must degrade to a readable sentence rather than raise — an
    ``AttributeError`` here would escape into the gate path and take down a
    question a human was about to answer. The fallback is deliberately a
    sentence a human can read on a notification banner, matching
    ``ask_pending_request``'s.
    """
    return str(getattr(question, "question", "") or "the agent is asking")


@dataclass
class _PromptCommand:
    command_id: str
    text: str
    # ``None`` at runtime on an image-less prompt: ``_already_bounded`` treats
    # None as "already bounded" (nothing to decode) and passes it through, so
    # the cast that feeds this field can legitimately hand over None. Declared
    # here because pyright trusts the cast and would not catch a consumer
    # that assumed a list.
    images: list["ImageContent"] | None
    admitted: asyncio.Future[None]
    completed: asyncio.Future[bool] | None = None

    def __iter__(self):  # type: ignore[no-untyped-def]
        # Tuple compatibility for older diagnostics that inspect the queue.
        yield self.text
        yield self.images


def _read_child_todo_snapshot(directory: Any) -> list[dict[str, Any]] | None:
    """Read one historical plan without materializing or rewriting its files."""
    from local_operator.session.frontend_state import TodoItemState, TodoPhaseState
    from local_operator.session.transcript import Transcript

    try:
        transcript = Transcript(directory, defer_materialise=True)
        if not transcript.path.is_file():
            return None
        payload = transcript.latest_custom("todo_snapshot") or {}
        raw = payload.get("items", [])
        if not isinstance(raw, list):
            return None
        phases = [
            (
                TodoPhaseState.model_validate(row)
                if "items" in row
                else TodoPhaseState(name="Todos", items=[TodoItemState.model_validate(row)])
            )
            for row in raw
        ]
        return [phase.model_dump(mode="json") for phase in phases]
    except (OSError, ValueError, TypeError):
        return None


class ServingSessionHandle(SessionHandle):
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
        approval_pinned: bool = False,
        install_gates: bool = True,
        config_dir: Path | None = None,
    ) -> None:
        self._session = session
        self._goal_loop: Any = None
        self._loop_command_id: str | None = None
        self._active_prompt_command_id: str | None = None
        self._desktop_mcp: Any = None
        self._desktop_cwd = cwd
        store = getattr(session, "_frontend_state_store", None)
        if (
            store is not None
            and store.state.loop
            and store.state.loop.get("status") in {"running", "judging"}
        ):
            # Checkpoint state is not an instruction to spend more tokens. A
            # replaced owner reports interrupted work rather than resuming it.
            store.mutate(loop={**store.state.loop, "status": "interrupted"})
        self._loop = loop
        # When the owner's saved default is full-auto (``tool_approval_mode:
        # auto``), the phone must not park a card the TUI would never show —
        # the gate is answered ``True`` inline instead. Read PER DECISION by
        # the gate closure, which is what lets ``/approvals`` and a
        # ``config.yml`` change (:meth:`follow_config`) move it without
        # reconstructing the handle.
        self._auto_approve = auto_approve
        # ``approval_pinned`` marks the value as an explicit FLAG (``lop exec
        # --yolo``), which outranks the config key: a ``tool_approval_mode``
        # edit must not re-arm a gate the operator disabled on the command
        # line for this run. A separate kwarg rather than inferred from
        # ``auto_approve`` because ``spawn_owned_session`` passes the
        # CONFIG-derived value — the same ``True`` that must keep following
        # the file.
        self._approval_pinned = approval_pinned
        #: The mode a human typed with ``/approvals`` in THIS session, or
        #: ``None``. Deliberately the same shape as
        #: ``Session._explicit_model_choice``, and the symmetry is the point:
        #: the model half of this change already protects an explicit ``/model``
        #: pick from a ``config.yml`` default, and approvals is the more
        #: dangerous key of the two.
        #:
        #: It records WHICH mode was chosen, not merely THAT one was (review
        #: round 2, R6). A boolean conflated the two and the loosening guard
        #: read it as "the human chose ask", so a session whose human typed
        #: ``/approvals auto`` was pinned to ``ask`` forever by one file
        #: tightening \u2014 the operator's originating complaint ("if I change a
        #: setting I want it to go into effect for all my agents") reappearing
        #: on the very key this change is centred on.
        #:
        #: INVARIANT: non-``None`` only while the gate in force is the mode a
        #: human typed here. :meth:`_on_config_change` clears it whenever a
        #: file write MOVES the gate, because at that moment the file, not the
        #: human, owns the value \u2014 which is also what keeps the keep notice's
        #: "set with /approvals in this session" a true statement.
        self._explicit_approvals_mode: str | None = None
        self._unsubscribe_config_watch: Callable[[], None] | None = None
        #: The (kind, title, detail) of the gate currently parked, or None.
        #: Kept so the announcement can be re-run when the last viewer
        #: detaches — the routing decision is made when the gate opens, and
        #: without this a gate opened under a watching terminal is never
        #: announced after that terminal closes (round 3, B2).
        self._parked_announcement: tuple[str, str, str] | None = None
        #: In-flight MCP reloads after a `/mcp add|remove` wrote the config.
        #: Held only so a fire-and-forget task is not garbage-collected while
        #: it runs — asyncio keeps no strong reference of its own.
        self._mcp_reload_tasks: set[asyncio.Task[None]] = set()
        #: In-flight MCP grants, kept apart from the reload tasks above
        #: because their lifetimes differ by three orders of magnitude: a
        #: reload is sub-second best-effort work, a grant waits on a human for
        #: up to ten minutes. ``dispose`` cancels these; only one runs at a
        #: time (see ``_spawn_grant``).
        self._mcp_grant_tasks: set[asyncio.Task[None]] = set()
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
        #: The broker registration that authorizes THIS process's descendants to
        #: retrieve secrets, or ``None`` (§6). Held so teardown can deregister
        #: promptly rather than waiting for the process's socket to close: a
        #: runtime can outlive the session it served — a stopped ``exec`` run
        #: returning to its supervisor — and a session that is gone must stop
        #: authorizing its descendants (§2.1).
        self._secret_registration: SessionRegistration | None = None
        #: The config root this handle's session was built from, when the spawn
        #: site knows it (``spawn_owned_session``/``start_exec_control`` both
        #: do). Passed rather than re-read from ``config_dir()`` at registration
        #: time so the store-existence check and the registration share ONE
        #: base (MINOR-3): a handle whose session is rooted elsewhere can no
        #: longer check one store and register against another. ``None`` means
        #: the spawn site did not declare one; registration then falls back to
        #: this process's own ``config_dir()``, which is the same root the
        #: session was built from because the runtime is spawned with it in the
        #: environment.
        self._config_dir = config_dir
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
        #: Set once this handle has COMMITTED to retiring, by
        #: :meth:`begin_retire`. Non-empty means the admission paths refuse (see
        #: that method) — a runtime that is leaving must not start a turn it
        #: will abort one await later. Deliberately never cleared: a retirement
        #: is a one-way door for the process.
        self._retiring_cause: str = ""
        #: Installed by the runtime process (``process.amain``): fires the
        #: process's stop event so a socket ``stop`` op exits the way SIGTERM
        #: does. ``None`` under a host that has no process to exit.
        self.on_stop_requested: Callable[[], None] | None = None
        # The record's ``busy`` bit must settle when the LAST turn settles, and
        # the per-event path cannot say that: the final AgentEndEvent is
        # delivered while the turn still counts as busy (the held end is
        # flushed under ``_turn_lock``; an abort's end is emitted while
        # ``_is_streaming`` is True). ``_observe_prompt_drain`` covered turns
        # that ran through the prompt queue; every other opener (peer wake,
        # scheduled wake, background result delivery, resume catch-up) left
        # the record saying ``busy: true`` for hours (design §1.2). The
        # session's turn-boundary hook fires for all of them.
        #
        # DEFERRED by one loop iteration rather than published inline: the
        # hook runs inside ``_run_turn_pipeline``'s ``finally``, which is still
        # UNDER ``_turn_lock`` (the caller releases it on the way out), and
        # ``is_busy()`` reads that lock. ``call_soon`` runs after the pipeline
        # coroutine has returned and the ``async with`` released the lock
        # synchronously, so the publish reads the settled state
        # (``test_busy_settles`` pins the ordering). Probed so reduced
        # sessions in tests that never grew the attribute keep working.
        if hasattr(session, "on_turn_settled"):
            session.on_turn_settled = self._publish_busy_soon
        # Same shape as ``on_turn_settled``: the session flips the record's
        # ``started`` bit the first time a real turn runs (see
        # ``_run_turn_pipeline``), and the registrant owns the publish.
        if hasattr(session, "_publish_session_started"):
            session._publish_session_started = self._publish_session_started
        # Discovery/attachment does not authorize replacing a headless deny
        # gate with a parked interactive gate. Exec opts into that separately.
        #: Why the most recent admitted turn failed, for a headless caller that
        #: has no front end reading the projection. See the drain's handler.
        self._last_prompt_failure = ""
        self._gates_installed = install_gates
        if install_gates:
            self._install_gates()
        # Last, so a handle that could not be fully built never leaves a live
        # registration behind (see :meth:`_register_secret_session`). The call
        # is synchronous and can stall the loop for up to ``STARTUP_TIMEOUT_S``
        # (5 s) while a cold broker starts — bounded, one-time, and paid only
        # when a store exists at this handle's root and no daemon is listening
        # (MINOR-1). It mirrors the TUI's own boot cost deliberately, because
        # the alternative (registering off-loop) would let the handle serve
        # turns before it can authorize the descendants those turns spawn.
        self._register_secret_session()

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
                prose = _question_prose(question)
                # ``question``, not ``text``: :class:`AskQuestion` has no ``text``
                # field and sets ``extra="forbid"``, so it can never grow one, and
                # the old read always fell through to the literal "question". What
                # that actually degraded is narrower than it looks: the projection
                # card reads the field directly through ``ask_pending_request`` and
                # was always right, and ``lop sessions`` publishes the KIND
                # (``set_record_pending``) so prose never travelled there at all.
                # The reads it did break are ``_parked_announcement`` /
                # ``reannounce_pending`` and the timeout row REPLAYED TO THE MODEL.
                # It survived because this whole gate was unreachable until #868:
                # both its hosts build ``has_ui=False``, which the removed
                # ``build_ask_tool`` clause vetoed, so the body never ran.
                #
                # The prose is passed as DETAIL as well as title because the
                # notification body is composed from ``detail`` alone — a banner
                # built from the title is unreachable when detail is empty, so an
                # ask toast read the static "Waiting for your answer" while the
                # approval toast beside it named its action ("write: /etc/hosts").
                # A user who walked away could see that a decision was owed but
                # not which one.
                #
                # EXCEPT for a secret question, which stays terse deliberately. Its
                # prose names the credential being requested ("Paste your OpenAI
                # key"), and a lock-screen banner is exactly the surface that must
                # not enumerate which of the user's keys a session is missing. The
                # VALUE was never at risk here — the gate announces before any
                # answer exists — so this guards the QUESTION, and the shared
                # ``BODIES["ask"]`` vocabulary is the right fallback for it.
                announced = "" if getattr(question, "secret", False) else prose
                self._announce_pending("ask", prose, announced)
                try:
                    answer = await asyncio.wait_for(future, timeout=self._gate_timeout_s())
                except TimeoutError:
                    await self._record_gate_timeout("ask", prose, kind="ask")
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

    # -- live config -------------------------------------------------------------

    def follow_config(self, watcher: Any) -> None:
        """Subscribe the gate to ``tool_approval_mode`` changes on ``watcher``.

        The handle, not the session, owns the approval mode the RUNTIME's
        tools consult (``_auto_approve``), so it needs its own listener on the
        process :class:`~local_operator.config_watch.ConfigWatcher` beside the
        session's. Wired by the spawn sites (``spawn_owned_session``,
        ``start_exec_control``) rather than in ``__init__`` so a handle built
        around a test double never touches the real config directory.
        Idempotent; unsubscribed in :meth:`dispose`.
        """
        if self._unsubscribe_config_watch is not None:
            return
        self._unsubscribe_config_watch = watcher.subscribe(self._on_config_change)

    def _on_config_change(self, change: Any) -> None:
        """Move the gate to the mode on disk; the next DECISION sees it.

        A ``config.yml`` write is the operator's machine-wide intent ("if I
        change a setting I want it to go into effect for all my agents"), so a
        session that never made a choice of its own follows the file in BOTH
        directions. ``source`` is ignored on purpose: the runtime is never the
        process that wrote, so every delivery is another process's edit.

        **The rule is asymmetric, and only for a session that chose** (review
        round 1 R1, UX round 1 U1):

        * **Tightening (``auto`` → ``ask``) always follows the file**,
          unconditionally, in every session. Safety propagates without
          exception; a user ends up safer than they asked, which is never the
          wrong surprise.
        * **Loosening (``ask`` → ``auto``) does not move a session whose human
          typed ``/approvals ask`` in it.** That session keeps its gate and
          reads a keep notice naming the way to adopt the file instead. It is
          the CHOSEN MODE that is consulted, not merely the fact of a choice:
          a session whose human chose ``auto`` has no hardening to protect and
          follows the file in both directions like any other.

        The asymmetry is the whole point. The operator asked for settings to
        REACH running sessions, which was broken and is what this change fixes;
        they did not ask for a file write to revoke a hardening a human typed
        into a specific pane thirty seconds earlier. The parked-prompt rule
        below already encodes that principle — a card on screen is not
        auto-answered *because the human's presence outranks the file* — and it
        applies one step earlier to a human who typed the mode. This mirrors
        the model half of the same change exactly (``Session.
        _on_configured_model_changed``, ``_explicit_model_choice``, and its
        ``keeping …`` notice), and approvals is the more dangerous of the two
        keys: an explicit ``/model`` pick was already protected while an
        explicit ``/approvals ask`` was not, which reversed the safer default
        on the key where it matters more.

        Two further limits, unchanged. ``--yolo`` (``_approval_pinned``)
        outranks the key entirely — an explicit flag on this run is narrower
        and newer than a default in a file — and a prompt already PARKED
        (``_pending_futures``) is left alone: the gate reads ``_auto_approve``
        when a decision is made, so a card on screen keeps waiting for the
        human, never auto-answered on a loosening and never auto-denied on a
        tightening.

        The receipt goes out as a :class:`NoticeEvent` from THIS process, the
        one that owns the gate, so every attached viewer and the phone see
        the effect ("every tool runs without asking"). It is now the ONLY
        receipt for the event: the TUI no longer prints a value clause when a
        runtime is attached (design round 1, D1), on the same "the process that
        decided is the one the user should read" rule the ``model`` section
        already follows. Subagents inherit this gate closure, so they follow
        too.
        """
        if self._disposing or "tool_approval_mode" not in getattr(change, "changed_keys", ()):
            return
        if self._approval_pinned:
            return
        values = getattr(change, "values", {})
        mode = str(values.get("tool_approval_mode", "ask")).strip().lower()
        if mode not in ("ask", "auto"):
            # An unknown spelling is not "ask" by accident: the watcher only
            # delivers parseable files, so this is a typo, and the safe answer
            # is to keep the mode in force rather than guess.
            logger.warning("tool_approval_mode=%r ignored; use ask or auto", mode)
            return
        wanted_auto = mode == "auto"
        if wanted_auto == self._auto_approve:
            return
        if wanted_auto and self._explicit_approvals_mode == "ask":
            # LOOSENING against a human's explicit hardening: keep the gate and
            # say so. Shaped like the model half's keep notice — what is kept,
            # why, and the exact command that adopts the file — so the two
            # conflicts read as one rule rather than two behaviours.
            #
            # `== "ask"` and not a bare truth test: only a typed `ask` is a
            # hardening this may refuse a file for. A typed `auto` is an
            # opinion about the same key, but refusing a loosening on its
            # behalf would pin a session to a mode its human never asked for.
            self._emit_notice(
                "keeping tool approvals: ask — set with /approvals in this session; "
                "config.yml now says auto, /approvals auto adopts it",
                "info",
                headline="Approvals unchanged",
            )
            return
        self._auto_approve = wanted_auto
        # The FILE now owns the value in force, so a mode the human typed here
        # earlier no longer describes this gate. Clearing keeps the invariant on
        # `_explicit_approvals_mode` exact (review round 2, R6): without it, a
        # session tightened off a typed `auto` would still be carrying that
        # `auto`, and the keep notice's "set with /approvals in this session"
        # would be describing a choice the file, not the human, had made.
        self._explicit_approvals_mode = None
        self._notify()
        self._emit_notice(
            (
                "tool approvals: auto — config.yml changed; every tool runs without asking"
                if wanted_auto
                else "tool approvals: ask — config.yml changed; write and command tools "
                "prompt again"
            ),
            "warning" if wanted_auto else "info",
            headline=f"Approvals: {mode}",
        )

    def _emit_notice(self, text: str, kind: str, headline: str = "") -> None:
        """Fire-and-forget a ``NoticeEvent`` through the session's emit seam.

        Same shape as :meth:`_grant_notice`: the channel every attached
        terminal and the phone already listen on. Held on
        ``_mcp_reload_tasks`` because it is the same class of sub-second
        best-effort work, and ``dispose`` cancels that set.

        ``headline`` is the SHORT glance a boot toast shows instead of a blind
        cell cut of ``text`` (design round 1, D3; see ``NoticeEvent.headline``).
        Passed by callers that know the state; "" keeps the existing fallback.
        """
        from local_operator.harness.types import NoticeEvent

        emit = getattr(self._session, "_emit", None)
        if not callable(emit):
            logger.info("notice: %s", text)
            return
        event_kind = kind if kind in ("info", "warning", "error") else "info"
        typed_emit = cast("Callable[[Any], Awaitable[Any]]", emit)

        async def _emit_notice() -> None:
            try:
                await typed_emit(NoticeEvent(text=text, kind=event_kind, headline=headline))
            except Exception:  # noqa: BLE001 — a failed notice must not kill the loop
                logger.debug("runtime notice failed", exc_info=True)

        task = self._loop.create_task(_emit_notice())
        self._mcp_reload_tasks.add(task)
        task.add_done_callback(self._mcp_reload_tasks.discard)

    def _register_secret_session(self) -> None:
        """Register THIS process as the session the broker notifies (§6).

        **The notice has to reach the process that owns the output filter and
        the transcript**, and in the attached architecture that is this
        runtime: ``Session.variables`` here is the store the bash and eval
        redactors read, while the TUI holds only an ``AttachedSession`` facade
        with no store at all. Registering from the viewer instead made the
        notice unanswerable by construction, so the broker (correctly) denied
        every descendant retrieval in every attached session.

        Best-effort by contract, exactly like the TUI's own registration: the
        store is an optional capability and never a boot dependency (§13), so a
        session starts and runs normally with no broker to reach.

        Runs on the runtime's loop, so the one blocking step — starting a
        broker that is not up yet — is a bounded, one-time stall
        (``STARTUP_TIMEOUT_S``, 5 s) before the runtime serves anyone. That is
        the cost the TUI already pays at its own boot, and it is paid only when
        a store exists at this handle's root AND no daemon is already
        listening; a warm daemon makes the poll return on its first check.

        The base is the handle's own ``self._config_dir``, not a re-read of
        ``config_dir()``: the spawn sites declare the root they built the
        session from, so the store-existence check inside
        ``register_variable_store_session`` and the registration itself cannot
        disagree about WHICH store this session owns (MINOR-3, NIT-2).
        """
        from local_operator.secrets.session import register_variable_store_session

        try:
            self._secret_registration = register_variable_store_session(
                self._session,
                session_id=getattr(self._session, "session_id", None),
                base=self._config_dir,
            )
        except Exception:  # noqa: BLE001 — the store is optional; boot must not fail on it
            logger.debug("secret session registration failed", exc_info=True)
            self._secret_registration = None

    def register_secret_redaction(self, value: str) -> None:
        """Add ONE value the §6 notice asked this process to scrub.

        The owner side of ``AttachedSession.register_secret_redaction``: the
        viewer forwards a value here when ITS registration, not this runtime's,
        is the one that answered the broker's notice — which happens when this
        runtime registered nothing because no store existed at its base at boot
        (see :meth:`_register_secret_session`). The value has to reach the store
        the bash and eval redactors read, which is ``self._session.variables``,
        so that is the only thing this method touches.

        It writes the value into the redaction set and nothing else: never a
        credential, never an announcement, never a log line, never an audit row
        or an event. It RAISES when there is no store, so the viewer's sink
        declines to acknowledge and the broker denies the child rather than
        serving a value nothing can scrub — the fail-closed direction §6 requires.
        """
        variables = getattr(self._session, "variables", None)
        if variables is None:
            raise RuntimeError("this runtime has no variable store to redact through")
        variables.register_redaction(value)

    def close_secret_registration(self) -> None:
        """Deregister this process, so its descendants stop being authorized.

        Idempotent and non-raising. The socket closing on process exit is the
        backstop; this is the plan (§2.1), and it is what covers a runtime that
        outlives the session it served — an ``exec`` run whose control surface
        closes while the process lives on to be reused.
        """
        registration = self._secret_registration
        self._secret_registration = None
        if registration is not None:
            try:
                registration.close()
            except Exception:  # noqa: BLE001 — teardown must not fail on a deregistration
                logger.debug("secret session deregistration failed", exc_info=True)

    async def dispose(self) -> None:
        """Dispose the underlying session (release the claim, flush, abort).

        The child's clean-exit path calls this rather than reaching through
        to ``self._session`` so the ordering (deny gates first) stays in one
        place and hosts cannot forget the claim release."""
        self._disposing = True
        # The dispose rung of EVERY exit that is not a viewer-driven retirement:
        # SIGTERM/SIGINT in ``amain``, the reaper's ``_clean_exit``, and a host
        # that disposes in place. Recorded BEFORE the abort below so the turn's
        # end event carries it (``_classify_cut_off`` reads it from the emitted
        # event), and suppressed when a deliberate stop was already noted for
        # this turn — the graceful ``stop`` op reaches here too, and relabelling
        # a user's own cancel as an error is the worse mistake.
        session = getattr(self, "_session", None)
        note = getattr(session, "note_cut_off", None)
        if callable(note):
            note(self._retiring_cause or "runtime-shutdown")
        # Revoke the broker registration along with the session: descendants of
        # a session that is going away must not stay authorized behind it
        # (§2.1). Bounded and non-raising, so it cannot delay or break teardown.
        self.close_secret_registration()
        if self._goal_loop is not None:
            await self._goal_loop.cancel()
        if self._unsubscribe_config_watch is not None:
            self._unsubscribe_config_watch()
            self._unsubscribe_config_watch = None
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
        # A grant parked on a browser round trip would otherwise outlive the
        # session it authenticates: it holds the loopback redirect port and a
        # reference to a disposed session, waiting up to ten minutes for a
        # human who is no longer there.
        #
        # Cancelled rather than awaited, because nobody is going to complete a
        # login for a session that is going away. This does NOT suppress the
        # notice — cancelling is what CAUSES ``_settle`` to emit one, and it
        # does land (review F7 corrected an earlier comment here that claimed
        # the opposite). That is deliberate: a `reauth` cancelled after its
        # delete has destroyed a credential, and the user has to be told.
        for grant in list(self._mcp_grant_tasks):
            if not grant.done():
                grant.cancel()
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
        if self._goal_loop is not None and self._goal_loop.running:
            return True
        if any(not task.done() for task in self._mcp_grant_tasks | self._mcp_reload_tasks):
            return True
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

    def is_conversationally_active(self) -> bool:
        """True while the CONVERSATION itself is moving — the spinner's signal.

        Deliberately narrower than :meth:`is_busy`, and the difference is the
        whole point. The two answer different questions:

        * :meth:`is_busy` asks *may this runtime exit?* and must be maximally
          inclusive, because exiting under live work destroys it. A detached
          background job, a running subagent, a retained background task all
          forbid an exit and are all counted there.
        * This asks *is this conversation working right now?*, which is what an
          animated indicator claims to a user. Somebody reading a spinner
          expects tokens to be moving and a reply to be coming.

        Publishing the residency answer as the activity bit made every session
        that had ever backgrounded a job look permanently busy. Measured on the
        reporter's host: 8 of 8 live sessions published ``busy=True``, one of
        them idle for 25.6 minutes with a completed final turn — the runtime
        was correctly resident (a `bash` job was still running) and the sidebar
        was incorrectly claiming it was working. A spinner that is always on is
        not a status; the user's report ("still shows that it's active … even
        though the task is done") is exactly that indicator having become
        meaningless.

        The terms kept here are the ones a user would call "it is working on my
        conversation": a live provider stream, a compaction, a held turn lock,
        a queued or draining prompt, a running goal loop, and a parked gate.
        The gate is included because a parked approval IS a running turn — the
        tool slot is held mid-flight — and the surfaces then draw it with the
        needs-you marker, which outranks the spinner in ``row_state_mark``.

        The terms dropped are the work-RETENTION ones: background jobs,
        subagents, MCP grant/reload tasks and retained background tasks. Each
        is real work the runtime must stay alive for, and none of it means the
        conversation is mid-reply — a `background=true` job exists precisely to
        outlive its turn, so animating a row for it contradicts the flag's
        purpose. Those sessions still render as live (the idle glyph), and
        `lop sessions` still reports them resident.

        Fails CLOSED (True) on an unreadable probe, matching :meth:`is_busy`:
        a spurious spinner is a cosmetic fault, while a missed one would hide a
        session that genuinely is working.
        """
        if self._disposing:
            return False
        session = self._session
        if self._goal_loop is not None and self._goal_loop.running:
            return True
        try:
            if getattr(session, "is_streaming", False):
                return True
            if getattr(session, "_compacting", False):
                return True
            lock = getattr(session, "_turn_lock", None)
            if lock is not None and lock.locked():
                return True
        except Exception:  # noqa: BLE001 — an unreadable session is assumed active
            return True
        if self._prompt_queue or (
            self._prompt_drain_task is not None and not self._prompt_drain_task.done()
        ):
            return True
        if self._pending_futures:
            # A gate parked on a person belongs to a turn that has not ended.
            return True
        return False

    def is_pristine(self) -> bool:
        """True when nothing has ever happened in this session.

        The retirement predicate for an EAGERLY started runtime
        (``retire_if_pristine`` in ``server.py``). A viewer now engages a
        runtime the moment the TUI mounts, so the band can show the model, the
        MCP roster and the context reading without waiting for a keystroke.
        The cost of that is a process the user may never use: opening a
        terminal, reading the band and pressing ``ctrl+d`` must not leave a
        session behind, and neither must ``/resume`` onto a different one.

        "Pristine" is deliberately much stricter than "idle". :meth:`is_busy`
        answers *may this exit later*, which a runtime carrying a whole
        finished conversation satisfies the moment its last turn lands. This
        answers *did this session ever exist as far as the user is concerned*,
        and a single durable row anywhere is enough to say yes. Getting that
        backwards deletes a conversation, so every probe below fails CLOSED:
        anything unreadable reports "not pristine" and the runtime lives on to
        be reaped by the ordinary residency drain instead.

        Wakes are checked because a scheduled wake is the one piece of session
        state that is durable, invisible in the transcript, and worth more than
        the process holding it — retiring a runtime whose scheduler is armed
        would silently drop the schedule the user asked for. Both the live
        scheduler AND the on-disk wake index are consulted: the scheduler is
        the truth while it is armed, but it reports "no wakes" once disposed or
        absent on a reduced host, and the index row is what a cold resume
        re-arms from — so an index row alone is enough to say "not pristine"
        (review round 1, MINOR-4).
        """
        if self.is_busy():
            return False
        session = self._session
        try:
            transcript = getattr(session, "_transcript", None)
            if transcript is None:
                return False
            # The durable row count, not the model-facing window: compaction
            # shrinks what the model sees and must never make a real
            # conversation look like a fresh one.
            if transcript.entries():
                return False
            # A transcript file that exists at all means a write happened, even
            # if every row was since compacted away.
            directory = Path(getattr(transcript, "directory", "") or "")
            if (directory / TRANSCRIPT_FILENAME).exists():
                return False
            # The attachment sidecar is durable state the user asked for with
            # no transcript row to show for it: a routed `/team <name>` or
            # `/agent <name>` (a cold viewer can do that since #624) journals
            # the roster onto `attachment.json` and prints "team X is ready" —
            # then `ctrl+d`. Retiring here discarded the attachment the
            # receipt had just confirmed and stranded a sidecar-only
            # directory nothing lists (review round 2, R7). The sidecar is
            # what a resume restores the team from, so it is exactly as
            # durable as a transcript row for this question.
            from local_operator.resume import ATTACHMENT_SIDECAR_NAME

            if (directory / ATTACHMENT_SIDECAR_NAME).exists():
                return False
        except Exception:  # noqa: BLE001 — an unreadable transcript is not pristine
            logger.debug("pristine probe: transcript unreadable", exc_info=True)
            return False
        try:
            if self.next_wake_due_at() is not None:
                return False
        except Exception:  # noqa: BLE001
            logger.debug("pristine probe: wake scheduler unreadable", exc_info=True)
            return False
        try:
            from local_operator.paths import config_dir
            from local_operator.wakes.store import read_entry

            entry = read_entry(config_dir(), str(getattr(session, "session_id", "") or ""))
            if entry and (entry.get("schedules") or []):
                return False
        except Exception:  # noqa: BLE001
            logger.debug("pristine probe: wake index unreadable", exc_info=True)
            return False
        try:
            if session.history():
                return False
        except Exception:  # noqa: BLE001
            logger.debug("pristine probe: history unreadable", exc_info=True)
            return False
        return True

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        """Commit this runtime to retiring, iff it is idle RIGHT NOW.

        Sets ``_retiring_cause`` in the SAME synchronous step that asks
        :meth:`may_refresh`, and from that instant the admission paths
        (:meth:`prompt`, :meth:`receive_peer_message`) REFUSE rather than queue.
        That is the whole point: both retire paths sample the predicate and then
        act across an ``await`` — the reaper's stagger plus ``announce_retiring``
        (which drains each viewer's writer), and ``dispose`` is async — so the
        "idle" claim was only ever true at ONE instant, and a ``prompt`` or a
        ``peer_message`` arriving in the gap opened a turn the dispose then
        aborted (design §5.1). The latch makes the claim true by construction
        rather than by timing.

        ``cause`` names the retirement for the refusal and the log; the session
        is told as well, so a turn aborted while retiring is labelled with the
        retirement rather than a generic shutdown.
        """
        try:
            reason = str(self.may_refresh() or "")
        except Exception:  # noqa: BLE001 — uncertainty keeps the runtime
            reason = "busy probe failed"
        if reason:
            return False
        self._retiring_cause = cause or "retiring"
        session = getattr(self, "_session", None)
        note = getattr(session, "note_cut_off", None)
        if callable(note):
            note(self._retiring_cause, detail)
        return True

    def _retiring_refusal(self) -> str:
        """The refusal an admission gets once this runtime has committed to leaving."""
        return (
            f"the session runtime is retiring ({self._retiring_cause}); the message "
            "was not admitted — send it again and the next engage runs the new build"
        )

    def may_refresh(self) -> str:
        """Why this runtime must NOT retire for a newer build right now, or
        ``""`` when it may.

        ONE predicate shared by the reaper's self-refresh branch
        (``process._should_refresh``) and the viewer-driven ``refresh_if_idle``
        op, so the two paths can never disagree about what "idle" means. Two
        terms, deliberately the first two of ``process._should_exit`` and NOT
        the third:

        * ``is_busy()`` False — the same authority the reaper uses, never the
          record's derived ``busy`` bit. Covers a live turn, a parked gate, a
          running goal loop, live subagents and background jobs.
        * no wake due inside the warm window — a wake about to fire would be
          paid twice (this runtime retires, the supervisor spawns a successor
          for the wake), and ``WARM_WINDOW_S`` exists to avoid exactly that.

        An attached viewer does NOT hold. That is the operator's rule and the
        point of the whole mechanism: the viewer re-engages a fresh runtime on
        its own when it reads ``retiring``, and holding for it is precisely
        what kept a five-hour-stale runtime resident (design §1.1). Pristine
        runtimes are not exempt either — a pristine stale runtime is the
        cheapest possible refresh.

        Returns the REASON rather than a bool because the viewer paints a
        notice off the ``kept: <reason>`` answer and the log line names it.
        """
        try:
            if self.is_busy():
                return "busy"
        except Exception:  # noqa: BLE001 — uncertainty keeps the runtime
            return "busy probe failed"
        # Term 2 is the reaper's own helper, not a re-derivation: one place
        # decides what "inside the warm window" means. Lazy import — the
        # process module imports this one inside ``amain``, never at load.
        from local_operator.session.runtime.process import _wake_within_window

        if _wake_within_window(self):
            return "wake due within the warm window"
        return ""

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

        DELIBERATE STOP vs PLANNED RETIREMENT, told apart HERE because both
        reach this one rung. A retirement is driven by the runtime itself and
        has already latched the handle (``begin_retire``) before calling this;
        an un-latched call is therefore the user's own ``/stop`` or
        ``lop stop``, and it must be recorded as such or the taxonomy (which
        now requires positive evidence for ``interrupted``) would have to guess.
        """
        self._note_deliberate_stop()
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

    def _note_deliberate_stop(self) -> None:
        """Record that the USER ended this turn, before anything tears it down.

        ONE place for the three deliberate rungs of this handle (``stop``,
        ``abort``, ``cancel``), because the verdict has to be written while the
        act is still knowable: ``Session.dispose()`` notes ``disposed``
        unconditionally, and an un-noted dispose therefore publishes the user's
        own cancel as ``kind=error`` / ``cause=disposed`` (review round 1,
        BLOCKER-1). ``abort`` is the phone's stop button and ``cancel`` its
        supervised sibling; both are a person or a supervisor saying "stop",
        and each was one teardown away from being reported as a failure.

        Refused while a retirement is latched, matching ``request_stop``: the
        runtime is already ending that turn for its own reason (a build flip)
        and the cut-off verdict for it belongs to the retire path, which
        recorded it when it latched. Non-raising by contract — it runs inside a
        stop, and a stop must not fail because a host session has no say in its
        own taxonomy.
        """
        if self._retiring_cause:
            return
        note = getattr(self._session, "note_deliberate_stop", None)
        if callable(note):
            note()

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
    # for a capability it did not advertise. ``AttachedSession`` asks for it
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

    async def refresh_attention(self) -> dict[str, Any]:
        state = await self._session.refresh_attention()
        self._projection.attention = state
        return state

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        state = await self._session.acknowledge_attention(token)
        self._projection.attention = state
        return state

    @property
    def frontend_state_seed(self) -> Any:
        """Canonical state seed for full-TUI attach clients."""
        return self._session.frontend_state

    async def subscribe_frontend(
        self, on_update: Callable[[Any], None], *, display_window: bool = False
    ) -> Any:
        """Snapshot and subscribe atomically, on the loop that publishes.

        ``Session.subscribe_frontend`` refreshes through the publishing path
        and returns the snapshot with its sequence number, which is what lets
        every client's exact-``+1`` gap check detect transport loss. Awaited
        rather than wrapped because the caller is already on this loop.
        """
        return self._session.subscribe_frontend(on_update, display_window=display_window)

    async def record_shell(self, command: str, result: Any) -> None:
        await self._session.record_shell(command, result)

    async def history_page(self, before: str, anchor: str = "") -> dict[str, Any]:
        return self._session.history_page(before, anchor)

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
        # Seed the state (and with it the child roster) ONCE at attach. Until
        # the next event arrives this push is all a freshly attached phone
        # renders, and a settled turn never sends another: without this an
        # already-finished child stays unroutable (``session_id=None``) for as
        # long as the session is quiet. Seeding cannot clobber the identity
        # fields the projection already carries, but not because of
        # ``set_state``'s None-skipping — ``set_subagent_details`` assigns every
        # roster field unconditionally, ``row.session_id`` included, and bumps
        # the version. It is safe because both sides read the SAME registry: the
        # fold's rows and the seed's nodes each describe one child from
        # ``SubagentComms``, so the republish writes the values the row already
        # held.
        self._refresh_state()
        return unsubscribe

    async def prompt(
        self,
        text: str,
        images: list[dict[str, str]] | list["ImageContent"] | None = None,
        command_id: str | None = None,
        *,
        wait_complete: bool = False,
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
        # Bounded BEFORE the reservation, deliberately: the bound is a thread
        # hop, and awaiting between reserving a producer identity and queueing
        # the command would open a suspension point inside that window, where
        # every path out is a reject. Decoding first keeps the reserve-to-queue
        # span await-free. The cost of bounding an image for a prompt that is
        # then rejected is a duplicate submission's worth of CPU, which is the
        # cheaper side of that trade.
        #
        # ALREADY-DECODED blocks pass straight through, and that is not a
        # convenience: ``_admit_without_waiting_for_the_turn`` budgets a few
        # LOOP TURNS for this method's synchronous prelude to raise its
        # reportable refusals, and a thread hop does not resolve inside that
        # budget (measured: not done at 2 sleep(0)s, done by 20) no matter how
        # warm the pool is, because ``to_thread`` always yields at least once.
        # A caller that has already bounded therefore keeps the prelude
        # await-free, which is the premise ``_ADMISSION_PRELUDE_TURNS``
        # documents. Without this, an image-carrying admission reached the
        # budget with its prelude unrun and its refusals were logged to a
        # detached callback instead of reaching the user -- the silent drop
        # that method exists to repair.
        blocks = (
            cast(list["ImageContent"], images)
            if _already_bounded(images)
            else await _image_blocks_async(cast(list[dict[str, str]] | None, images))
        )
        if not self._command_reservations.reserve(command_id, kind="prompt"):
            return "already admitted"
        # Restore intentionally keeps missing attachments readable. Admission is
        # different: executing a plain assistant under a stored profile label
        # would silently change the task. Check the owner's resolved state, not
        # just HTTP's earlier preflight (the profile can disappear during boot).
        from local_operator.session.errors import (
            AttachmentUnavailable,
            ProfileRegistryUnavailable,
        )

        registry = getattr(self._session, "agent_registry", None)
        complete = getattr(registry, "require_complete_metadata", None)
        try:
            if complete is not None and (
                getattr(self._session, "active_agent", "")
                or getattr(self._session, "active_team", None)
                or getattr(self._session, "_unresolved_agent", "")
                or getattr(self._session, "_unresolved_team", "")
            ):
                complete()
        except ProfileRegistryUnavailable:
            self._command_reservations.reject(command_id)
            raise
        unresolved = getattr(self._session, "_unresolved_agent", "") or getattr(
            self._session, "_unresolved_team", ""
        )
        team = getattr(self._session, "active_team", None)
        if team is not None:
            resolve = getattr(self._session, "_resolve_profile_or_specialist", None)
            if resolve is not None and resolve(team.manager)[0] is None:
                unresolved = team.manager
        if unresolved:
            self._command_reservations.reject(command_id)
            raise AttachmentUnavailable()
        if self._disposing:
            self._command_reservations.reject(command_id)
            raise RuntimeError("session is closing; prompt was not admitted")
        if self._retiring_cause:
            # Refused, not queued: a turn admitted here is aborted one await
            # later by the dispose that is already on its way, after the
            # provider has been paid for whatever it managed to stream.
            self._command_reservations.reject(command_id)
            raise RuntimeError(self._retiring_refusal())
        if len(self._prompt_queue) >= MAX_QUEUED_PROMPTS:
            self._command_reservations.reject(command_id)
            raise RuntimeError(
                f"prompt queue is full ({MAX_QUEUED_PROMPTS}); wait for an admitted turn to start"
            )
        self._maybe_name_conversation(text)
        admitted: asyncio.Future[None] = self._loop.create_future()
        completed = self._loop.create_future() if wait_complete else None
        command = _PromptCommand(command_id, text, blocks, admitted, completed)
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
        if completed is not None and not await asyncio.shield(completed):
            raise RuntimeError("The admitted loop turn did not complete")
        if legacy_prompt and position > 1:
            return f"prompt queued ({position})"
        return "prompt admitted"

    def _loop_driver(self):
        from local_operator.session.goal_loop import GoalLoop

        if self._goal_loop is None:

            async def prompt(text: str) -> None:
                self._loop_command_id = str(uuid.uuid4())
                try:
                    await self.prompt(text, command_id=self._loop_command_id, wait_complete=True)
                finally:
                    self._loop_command_id = None

            async def judge(text: str) -> str:
                from local_operator.harness.types import Message

                return await self._session.complete_aside([Message.user(text)])

            def changed(state: dict[str, Any]) -> None:
                store = getattr(self._session, "_frontend_state_store", None)
                if store is not None:
                    store.mutate(loop=state)
                self._notify()

            async def checkpoint() -> None:
                # A headless session need not have a frontend subscriber. Its
                # loop still owns durable progress, including the terminal
                # boundary that occurs AFTER the last turn-end checkpoint.
                store = getattr(self._session, "_frontend_state_store", None)
                transcript = getattr(self._session, "_transcript", None)
                if store is not None and transcript is not None:
                    await store.checkpoint(transcript)

            self._goal_loop = GoalLoop(prompt, judge, self._cancel_loop_turn, changed, checkpoint)
            store = getattr(self._session, "_frontend_state_store", None)
            if store is not None and store.state.loop:
                self._goal_loop.state = dict(store.state.loop)
        return self._goal_loop

    async def run_headless_prompt(self, text: str) -> bool:
        """Submit through the owner queue so live viewers cannot race exec.

        Returns whether the turn completed, rather than raising on failure. A
        provider error arrives as an EVENT, so the renderer already reports the
        provider's own message and owns the exit code; raising on top of that
        would replace a real diagnosis with a generic queue message. Returning
        the outcome still lets the caller fail the run when the turn raised
        without ever emitting an error event. The goal loop takes the raising
        path (:meth:`prompt` directly) because it must stop iterating.
        """
        self._last_prompt_failure = ""
        try:
            await self.prompt(text, command_id=str(uuid.uuid4()), wait_complete=True)
            return True
        except RuntimeError:
            logger.debug("headless turn did not complete", exc_info=True)
            return False

    @property
    def last_prompt_failure(self) -> str:
        """Why the last admitted turn failed, or "" — see the drain's handler."""
        return self._last_prompt_failure

    async def run_headless_loop(self, *, count: int | None, goal: str | None) -> bool:
        """Await the same owner-local driver used by /loop, not another runner."""
        driver = self._loop_driver()
        driver.start(
            str(count) if count is not None else "", self._session.goal, goal_override=goal
        )
        assert driver.task is not None
        try:
            await driver.task
        finally:
            if driver.running:
                await driver.cancel()
        if driver.state.get("status") == "cancelled":
            raise asyncio.CancelledError
        return driver.state.get("status") in {"completed", "achieved"}

    async def cancel_headless_loop(self) -> None:
        if self._goal_loop is not None:
            await self._goal_loop.cancel()

    def _cancel_loop_turn(self) -> None:
        # Another frontend may have queued a manual turn before this iteration.
        # Cancelling automation must not abort that unrelated turn or leave its
        # own queued iteration behind to run after the cancelled driver exits.
        for command in list(self._prompt_queue):
            if command.command_id != self._loop_command_id:
                continue
            if command.command_id == self._active_prompt_command_id:
                self._session.abort()
            else:
                self._prompt_queue.remove(command)
                self._prompt_commands.pop(command.command_id, None)
                self._command_reservations.reject(command.command_id)
                if not command.admitted.done():
                    command.admitted.set_result(None)
                if command.completed is not None and not command.completed.done():
                    command.completed.set_result(False)
            return

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
            # Mirror the stand-in to the analytics ledger at PROVISIONAL rank,
            # exactly as the TUI band does. A mobile-started session goes
            # through this path INSTEAD of the TUI's, so without this a phone
            # session whose naming call failed is the same unreadable 12-hex row
            # in ``/analytics``. Rank-gated, so the real title still overwrites
            # it the moment ``set_conversation_name`` runs below.
            try:
                from local_operator.analytics import get_recorder
                from local_operator.analytics.store import SESSION_NAME_RANK_PROVISIONAL

                session_id = str(getattr(self._session, "session_id", "") or "")
                if session_id:
                    get_recorder().note_session_name(
                        session_id, label, rank=SESSION_NAME_RANK_PROVISIONAL
                    )
            except Exception:  # noqa: BLE001 — analytics is best-effort
                logger.debug("analytics: provisional name mirror failed", exc_info=True)
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
            self._active_prompt_command_id = command.command_id
            succeeded = False
            emitted_failure = False

            def observe_end(event: AgentEvent) -> None:
                nonlocal emitted_failure
                from local_operator.harness.types import AgentEndEvent

                if isinstance(event, AgentEndEvent) and (event.error or event.aborted):
                    emitted_failure = True

            # Session.prompt reports provider errors as events, not exceptions.
            # Queue completion must reflect that terminal result; otherwise a
            # goal loop submits its next turn after the model already failed.
            unsubscribe_outcome = self._session.subscribe(observe_end)
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
                succeeded = not emitted_failure
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
                # Kept for a headless caller, which has no front end to read the
                # projection: a turn that RAISES emits no error event, so this
                # string is the only description of what went wrong.
                self._last_prompt_failure = str(exc)
                logger.exception("mobile prompt failed after admission")
            except BaseException:
                # KeyboardInterrupt/SystemExit retain their process semantics;
                # the finally block still removes exactly the failed admission.
                logger.critical("mobile prompt drain terminated", exc_info=True)
                raise
            finally:
                unsubscribe_outcome()
                if command.completed is not None and not command.completed.done():
                    command.completed.set_result(succeeded)
                self._active_prompt_command_id = None
                self._prompt_queue.popleft()
                self._prompt_commands.pop(command.command_id, None)

    async def steer(
        self,
        text: str,
        images: list[dict[str, str]] | list["ImageContent"] | None = None,
        command_id: str | None = None,
    ) -> str:
        self._check_loop_thread()
        command_id = command_id or str(uuid.uuid4())
        # Bounded BEFORE the reservation, deliberately: the bound is a thread
        # hop, and awaiting between reserving a producer identity and handing
        # the steer to the session would open a suspension point inside that
        # window. Decoding first keeps the reserve-to-mutate span await-free.
        # Already-decoded blocks pass through for the reason ``prompt``
        # documents: an in-process caller that bounded already must not be
        # made to yield again.
        blocks = (
            cast(list["ImageContent"], images)
            if _already_bounded(images)
            else await _image_blocks_async(cast(list[dict[str, str]] | None, images))
        )
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
            self._session.steer(text, blocks, **fields)
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
        # A retiring runtime must not START a turn it will abort one await
        # later. The QUIET record-only delivery (``mailbox``, no wake) is
        # deliberately still admitted: it opens no turn, and refusing it would
        # drop a durable note the sender was promised it had delivered.
        if self._retiring_cause and (wake or mode != "mailbox"):
            raise RuntimeError(self._retiring_refusal())
        detail = await self._session.receive_peer_message(
            text, mode=mode, wake=wake, sender=sender or {}
        )
        self._fold.note_peer_message(text, sender=sender or {})
        self._notify()
        return detail

    async def abort(self) -> str:
        """Stop this session's turn AND its children, and say what was stopped.

        THE CONTROL OP HAS NO SECOND RUNG. The keyboard's Esc ladder can afford
        a narrow first press because a second press within
        ``DOUBLE_STOP_WINDOW_S`` is right there, offered on screen, and stops
        the children. Nothing on this path has that: the mobile relay, a
        supervisor and ``lop`` peers all send ``abort`` once into a session
        they cannot see, and there is no gesture that escalates. Reusing the
        keyboard's narrow semantics here therefore gave callers a stop that
        could NEVER reach a runaway — the operator sent ``abort``, was acked
        ``stopping``, and watched 39 paid provider calls land in the next two
        seconds while their children kept reporting in (QA Q-1). Escalating is
        what makes "a user must always be able to halt a runaway" true on the
        surface that has no ladder to climb.

        THE RECEIPT MUST NOT OVERSTATE, AND IT IS COUNTED AFTER THE FACT.
        It previously returned the literal ``"stopping"`` whatever survived,
        which is how the operator was told the problem was handled while the
        meter ran. Reporting ``cancel_subagents()``'s return value instead
        would only move the lie: that number is ``len(running)`` sampled at
        DISPATCH time, before any child has actually gone, and per-child
        cancellation is fire-and-forget with failures swallowed by design. A
        child that refuses to die was therefore counted as stopped — measured
        with two of three cancels raising, the receipt still said "stopped 3
        subagents" while two kept running and spending (review round 1,
        MAJOR-1).

        So the count is taken from the session's own live predicate once the
        cancellations have had a chance to land, and anything still standing is
        named as still running. Backgrounded ``bash`` jobs deliberately outlive
        a stop (``background=true`` exists so a build survives the turn that
        started it), so they are named too rather than implied stopped.
        """
        self._check_loop_thread()
        # The phone's stop button is the user's own act, so the verdict is
        # recorded before the turn is cut (see `_note_deliberate_stop`): the
        # turn ends aborted either way, and what this decides is whether that
        # abort reads as the user's stop or as a failure.
        self._note_deliberate_stop()
        if self._goal_loop is not None:
            await self._goal_loop.cancel()
        # THE PARENT FIRST, THEN THE CHILDREN. A child settling hands its
        # result back to the parent, and a parent still accepting work would
        # open a turn on it — so stopping the parent first is what makes the
        # children's teardown quiet instead of one last round of arrivals.
        self._session.abort("stopped from mobile")
        before = self._running_children()
        self._cancel_children("stopped from mobile")
        remaining = await self._settled_children(before)
        return self._abort_receipt(stopped=max(before - remaining, 0), still_running=remaining)

    def _cancel_children(self, reason: str) -> int:
        """Cancel this session's subagents, tolerating a host that has none.

        ``getattr``-probed like every other optional capability in this file: a
        reduced handle or a test double need not implement the subagent
        protocol, and a stop must not fail because the thing it was asked to
        stop does not exist.

        The return is the DISPATCH count and must not be reported as an
        outcome — see :meth:`abort`. Callers wanting the truth ask
        :meth:`_running_children` after the cancellations have settled.
        """
        cancel = getattr(self._session, "cancel_subagents", None)
        if not callable(cancel):
            return 0
        try:
            return int(cast(int, cancel(reason)))
        except Exception:  # noqa: BLE001 — a stop must never fail on its children
            logger.warning("cancelling subagents during abort failed", exc_info=True)
            return 0

    def _running_children(self) -> int:
        """Children still running RIGHT NOW, via the session's own predicate.

        ``running_subagents`` is the one predicate the ladder and its counts
        share, so the receipt cannot disagree with what a later Esc would
        offer. Probed and non-raising for the same reasons as
        :meth:`_cancel_children`.
        """
        counter = getattr(self._session, "running_subagents", None)
        if not callable(counter):
            return 0
        try:
            return int(cast(int, counter()))
        except Exception:  # noqa: BLE001 — a stop must never fail on its count
            logger.warning("counting subagents during abort failed", exc_info=True)
            return 0

    async def _settled_children(self, before: int) -> int:
        """Wait briefly for the cancellations to land, then count what is left.

        Cancellation is one fire-and-forget task per child, so the roster is
        still full the instant after dispatch and an immediate count would
        report every child as surviving — the mirror image of the overstatement
        this exists to fix. Polling rather than awaiting the tasks because the
        handle does not own them (the session tracks them for ``dispose``), and
        the loop exits the moment the roster drains, so the healthy case costs
        one tick rather than the whole budget.

        BOUNDED, and short. This runs inside a kill switch whose caller is a
        phone or a supervisor waiting on the ack: a child wedged in an
        uninterruptible await must delay the receipt by a beat, never hold it.
        Whatever has not gone by then is reported as still running, which is
        the honest answer and the one that tells the user to escalate.
        """
        if before <= 0:
            return 0
        deadline = time.monotonic() + _ABORT_SETTLE_BUDGET_S
        remaining = self._running_children()
        while remaining > 0 and time.monotonic() < deadline:
            await asyncio.sleep(_ABORT_SETTLE_POLL_S)
            remaining = self._running_children()
        return remaining

    def _abort_receipt(self, *, stopped: int, still_running: int) -> str:
        """What the abort actually did, including what it deliberately left.

        Survivors are named only when there ARE any: unconditional, it is noise
        on the overwhelmingly common stop that had nothing else running.
        """
        parts = ["stopping this turn"]
        if stopped:
            parts.append(f"stopped {stopped} subagent{'s' if stopped != 1 else ''}")
        if still_running:
            # The whole point of the change: a child that would not die is the
            # one fact the user must have, and it names the stronger lever
            # rather than leaving them to discover the meter still running.
            parts.append(
                f"{still_running} subagent{'s' if still_running != 1 else ''} "
                "did NOT stop — lop stop ends the process"
            )
        spared = self._background_bash_jobs()
        if spared:
            plural = "s" if spared != 1 else ""
            parts.append(f"{spared} background job{plural} still running — jobs cancel to stop")
        return "; ".join(parts)

    def _background_bash_jobs(self) -> int:
        """Backgrounded ``bash`` jobs, which a stop never touches.

        Deliberately spared (see ``Session.cancel_subagents``) and genuinely
        surprising to someone who just asked for everything to stop, so the
        receipt names them. Never raises: this runs inside a kill switch.
        """
        try:
            jobs = self._session.jobs.list()
        except Exception:  # noqa: BLE001 — an unreadable ledger must not break a stop
            logger.warning("listing background jobs during abort failed", exc_info=True)
            return 0
        return sum(
            1
            for job in jobs
            if getattr(job, "type", "") == "bash" and getattr(job, "status", "") == "running"
        )

    async def cancel_gracefully(self, reason: str = "cancelled by supervisor") -> str:
        """Stop at the next post-tool boundary, leaving in-flight work intact.

        The optional capability behind the ``cancel`` op's default mode (see
        the SessionHandle contract). Where :meth:`abort` fires the turn's
        AbortSignal and cancels the running tool task, this only SETS a sticky
        request the harness loop reads once every call in the batch has
        produced a paired result — so a ``git push`` or a merge-request write
        that is already on the wire completes, and the turn then ends as
        aborted with that work in the transcript.

        Returns immediately, and the receipt says so: the boundary may be one
        long tool away, and a caller that needs the process GONE by a deadline
        wants the stop ladder (``lop stop``), not this. Reporting "cancelled"
        here would claim a completion this cannot observe.

        Probed with getattr on a session too: an older ``SessionProtocol``
        implementation (or a test double) that predates
        ``request_graceful_cancel`` gets a clear error rather than a silent
        no-op that would leave a supervisor believing its cancel landed.
        """
        self._check_loop_thread()
        request = getattr(self._session, "request_graceful_cancel", None)
        if not callable(request):
            raise ValueError("this session cannot cancel at a tool boundary")
        # A supervisor's cancel is deliberate too, and it ends the turn the same
        # way (`harness/loop.py` ends it aborted) — so it records the same
        # verdict as the user's own stop rather than letting a later teardown
        # publish it as a cut-off failure (review round 1, BLOCKER-1).
        self._note_deliberate_stop()
        request(reason)
        return "cancelling at the next tool boundary"

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
            # Split rather than falling through to one sentence: joining an
            # EMPTY ladder produced "accepts no rungs, not 'turbo'", a double
            # negative that reads as though the value were nearly right, and
            # "rung" is internal vocabulary no user-facing surface uses.
            if not spec.reasoning_efforts:
                raise ValueError(f"{spec.model_id} has no reasoning-effort levels; drop --effort")
            ladder = ", ".join(spec.reasoning_efforts)
            raise ValueError(f"{spec.model_id} accepts {ladder} \u2014 not '{effort}'")
        self._session.set_model(spec.model_copy(update={"reasoning_effort": effort}))
        self._refresh_state()
        return f"effort: {effort}"

    async def slash(self, command: str, args: str) -> str:
        """Session-level slash commands — the ones with meaning off-terminal.
        TUI chrome (/help tables, /usage panels) is the phone UI's own job."""
        self._check_loop_thread()
        if command == "goal":
            return await self.slash_images(command, args)
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
        if self._watching_surfaces() or self._desktop_notification_available():
            # A visible terminal/phone or a notification-capable desktop can
            # reach a person. This is not the interactivity probe: background
            # delivery earns a parked wait, never an assertion somebody is here.
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

    def _desktop_notification_available(self) -> bool:
        reader = getattr(self._registrant, "notification_surfaces", None)
        if callable(reader):
            try:
                return "desktop" in cast("frozenset[str]", reader())
            except Exception:  # noqa: BLE001 — an unknown lease restores OS fallback
                logger.debug("could not read desktop notification reachability", exc_info=True)
        return False

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

    async def _record_gate_timeout(
        self, tool: str, description: str, kind: str = "approval"
    ) -> None:
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
                        # The gate KIND: an unanswered `ask` was not "denied",
                        # and describing it in the approval gate's vocabulary
                        # told the user something that did not happen (D12).
                        "kind": kind,
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
        if self._desktop_notification_available():
            # A background window is not interactive, but its main process
            # owns notification delivery while leased. Falling through would
            # post both Electron and detached-runtime OS toasts.
            return
        try:
            from local_operator.tui.notify import (
                APP_NAME,
                CONTEXTS,
                detached_notify,
                sanitize_text,
                session_names_in_notifications,
            )

            # TITLE IS THE SESSION NAME ONLY. " needs you" used to be appended
            # AFTER the 80-char cap, so a long model-written name produced a
            # 105-char title and the OS clipped exactly the two words that
            # explained the banner (round 3, D11). The state category rides
            # the subtitle, which is a field of its own and cannot be clipped
            # away by the name — the same place cmux and the in-band notifier
            # already put it.
            #
            # `sanitize_text` for the same reason every other path does it:
            # the name is model-written and reaches argv (D16).
            #
            # `display.notification_session_name` gates this leg too. The flag
            # exists to keep a model-written name off a screen other people can
            # see, and a detached runtime's gate toast reaches the same lock
            # screen as every other banner — a flag that governed only some of
            # them would make its own settings copy false (review round 1, M2).
            name = APP_NAME
            if session_names_in_notifications():
                name = sanitize_text(getattr(self._session, "conversation_name", "") or "lop")
            # The body names the ACTION when there is one, and otherwise falls
            # back to the shared vocabulary — an `ask` with no text used to
            # render as the bare word "question" with no hint it was a
            # question rather than an approval.
            #
            # Composed by `notifications.compose.gate_body` rather than inline,
            # so this leg and every other gate surface cannot drift: the rule it
            # carries (never `f"{title}: {detail}"`, because a tool's
            # `describe_approval` already leads with its own action word and the
            # title IS the tool name — round 4, Q3) is one that was fixed here
            # once and would have to be re-fixed in each new surface otherwise.
            from local_operator.notifications import gate_body

            detached_notify(
                name,
                gate_body(kind, title, detail),
                session_id=self._session_id_for_resume(),
                subtitle=CONTEXTS.get(kind, ""),
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
            payload = front.to_json() if front is not None else None
            if payload is not None:
                # The card gains the session's name so a DESKTOP banner for it
                # can be triaged: "Waiting for approval" with three sessions
                # open names none of them (design round 1, D3). Stamped here
                # rather than inside `PendingRequest` because the name is a
                # property of the SESSION, not of the question, and because the
                # privacy gate belongs on the publication boundary where every
                # other notification fact is decided.
                payload["session_name"] = self._notifiable_session_name()
            store.mutate(pending_gate=payload)
        except Exception:  # noqa: BLE001 — a card is never worth failing a gate
            logger.debug("could not publish the pending gate", exc_info=True)

    def _notifiable_session_name(self) -> str:
        """This conversation's name, or ``""`` when banners may not carry it.

        One helper for both gate-publication sites, because the privacy rule
        must not be able to hold on one and not the other — that asymmetry is
        exactly the defect ``notify.py``'s own flag documentation records
        (review round 1, M2: a flag that governed only some banners made its
        settings copy false).

        Sanitised on the way out for the same reason every other name-bearing
        path sanitises: it is model-written and reaches argv and an AppleScript
        literal on the surfaces that render it.
        """
        try:
            from local_operator.tui.notify import (
                sanitize_text,
                session_names_in_notifications,
            )

            if not session_names_in_notifications():
                return ""
            return sanitize_text(getattr(self._session, "conversation_name", "") or "")
        except Exception:  # noqa: BLE001 — a name is chrome; the gate is not
            logger.debug("could not resolve the gate's session name", exc_info=True)
            return ""

    async def fork_snapshot(self, message: str) -> dict[str, Any]:
        """Snapshot THIS authenticated owner, never a client-supplied path/id."""
        busy = self.is_busy()
        result = await self._session.fork_snapshot(message)
        # Jobs and parked gates also count as original work, even between turns.
        result["busy"] = busy
        return result

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
        complete = self._session.complete_aside
        store = getattr(self._session, "_frontend_state_store", None)
        if store is not None and "on_usage" in inspect.signature(complete).parameters:
            # A remote caller receives text, not a billable usage callback. The
            # authoritative owner must charge the request here; otherwise a
            # hidden goal judge (and other remote asides) silently costs zero.
            return await complete(
                messages, on_usage=lambda usage: store.accrue_usage(self._session, usage)
            )
        return await complete(messages)

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

        Legacy/mobile callers cannot consume action receipts themselves, so
        the owner completes them through normal admission, including images.
        This must share the authoritative path or `/goal` would set metadata
        without starting work only when invoked from the phone.
        """
        result = await self.run_slash_authoritative(command, args, images)
        return str(result.get("text") or f"ran /{command}")

    async def credential_op(self, action: str, key: str, value: str) -> dict[str, Any]:
        """Run one ``/credential`` verb against the session's variable store.

        The runtime's half of a capability the base protocol declares for every
        session: the agent's ``bash`` commands run in THIS process and read
        ``credential_env()`` from this store, so the verb executes here and a
        front end holding its own copy would advertise a key no executing tool
        could read. The front end keeps the masked paste (the user is sitting
        there); only the resulting value crosses, over the dedicated op.

        The verb table itself is shared with the in-process shape
        (:func:`local_operator.session.credential_ops.run_credential_verb`) so
        the two implementations cannot drift — a verb added for one shape is a
        verb added for both.

        Returns plain data rather than a ``SlashResult`` because the caller
        needs the FACTS (did it replace, what was removed) to build its own
        notice, and because the store verb's receipt has to name the key that
        was actually normalized and stored, not the one that was typed.

        The value is never logged, never journalled, and never returned — only
        the key name and the outcome cross back.
        """
        from local_operator.session.credential_ops import run_credential_verb

        return await run_credential_verb(
            getattr(self._session, "variables", None),
            self._journal_credential,
            action,
            key,
            value,
        )

    def _journal_credential(
        self, key: str, *, action: str = "stored", replaced: bool = False
    ) -> None:
        """Best-effort announcement; a failed one must not fail the store."""
        journal = getattr(self._session, "journal_credential_change", None)
        if not callable(journal):
            return
        try:
            journal(key, action=action, replaced=replaced)
        except Exception:  # noqa: BLE001 — the credential is already stored
            logger.warning("could not announce credential change", exc_info=True)

    def cancel_subagents_count(self) -> int:
        """Cancel every running subagent and return the REAL count.

        Esc's second job. `AttachedSession.cancel_subagents` swallows a failure
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
        *,
        locality: str = "local",
        consumers: Iterable[str] | None = None,
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

        ``locality`` is the invoking CLIENT's declared position (see
        ``ClientLocality``), defaulting to ``local`` because every client that
        exists today reaches this runtime over loopback. Only ``/mcp``'s grant
        verbs read it, to decide whether opening a browser here would put the
        tab in front of the person who typed the command.

        ``consumers`` is which action-carrying receipts the invoking client
        renders itself (``SLASH_ACTION_RECEIPTS``); ``None`` means it declared
        nothing, which is what every client built before the field looks like
        on the wire. See :meth:`_complete_unconsumed_action`.
        """
        from local_operator.session.frontend_state import SlashResult

        result = await self._slash_result(command, args, SlashResult, locality)
        result = await self._complete_unconsumed_action(result, images, consumers)
        return result.model_dump(mode="json")

    async def _complete_unconsumed_action(
        self,
        result: Any,
        images: list[dict[str, str]] | None,
        consumers: Iterable[str] | None,
    ) -> Any:
        """Run an attach receipt's request here when the client will not.

        THE INCIDENT THIS REPAIRS. ``/team <name> <request>`` on a viewer
        attaches the team on this runtime and returns a ``team_attached``
        receipt carrying the request; since #624 the VIEWER is expected to
        submit that request as a user turn. A viewer older than #624 prints
        the receipt text and has no consumer for ``data["request"]`` — so the
        team was attached, "sending to <team>. <manager> is coordinating."
        was printed, and the request was dropped with no user row, no turn and
        no error. Skew makes that reachable at any time: the on-disk install
        is replaced under long-lived TUIs several times a day, and the runtime
        a stale TUI spawns is built from the NEW install.

        The rule is ``type not in declared``, never "declared is None": a
        client that declared the type submits the request itself and the
        runtime must NEVER also run it (that would be a double turn), while
        both ``None`` (absent field, older viewer) and ``[]`` (declared, but
        consumes nothing) mean the request has no other home and is admitted
        here.

        Admission goes through the same ``_PromptCommand`` path the ``prompt``
        op uses, so the durable append resolves ``admitted``, the drain emits
        the user ``MessageStartEvent``, and an old viewer — which already
        subscribes with ``events=True`` — paints the user row from that event
        through its existing echo path. No renderer change is required on the
        old viewer, which is the point: the fix reaches TUIs that are already
        running and cannot be updated in place.

        KNOWN DEGRADATION: the viewer expands collapsed pastes into the
        request text at submit time, and those payloads live in the VIEWER's
        composer — they cannot cross the wire retroactively. A request
        admitted here that cites ``<[Paste #1, 240 lines]>`` reaches the
        manager as the chip label rather than the body. Images are unaffected
        (they are already on the wire in the ``slash_result`` frame and are
        passed into the admission unchanged). That is strictly better than the
        silent drop, and the skew notice the viewer paints names ``/reload``
        as the way back to full fidelity.
        """
        if getattr(result, "kind", None) != "notice":
            return result
        data = getattr(result, "data", None) or {}
        receipt_type = data.get("type")
        # The shared predicate, so this host and the TUI-hosted one cannot
        # drift: not an action receipt, or one the client declared it renders
        # (admitting that too would double-submit the user's command).
        if not runtime_must_complete(receipt_type, consumers):
            return result
        request = str(data.get("request") or "")
        if not request:
            # ``/agent clear`` returns ``agent_attached`` with an empty
            # request: a detach is a receipt with no action behind it.
            return result
        try:
            # Mirrors the viewer's own submit split (``app.py::_submit_prompt``):
            # a turn already running is STEERED, because ``prompt`` rejects a
            # concurrent call outright and the text would be thrown away. The
            # steer is delivered at the engine's next tool/message boundary,
            # which is the existing mid-turn channel rather than a new queue.
            #
            # ``steer`` is awaited directly because it cannot park: everything
            # it does is synchronous (reserve, hand the text to the session,
            # note the echo) and it returns on its first step. ``prompt``
            # emphatically CAN park — see below.
            if self._session.is_streaming:
                await self.steer(request, images=images, command_id=str(uuid.uuid4()))
            else:
                await self._admit_without_waiting_for_the_turn(request, images)
        except Exception as exc:  # noqa: BLE001 — the attach happened; say what did not
            # The attach ALREADY landed and stays; only the turn failed to
            # start (session closing, queue full, a rejected prompt). Reporting
            # that as a warning is the whole difference from the original
            # defect: the user learns the request did not run and can resend,
            # instead of watching nothing happen.
            logger.debug("completing an unconsumed slash action failed", exc_info=True)
            return result.model_copy(
                update={
                    "text": f"{result.text} — but the request was not sent: {exc}",
                    "style": "warning",
                }
            )
        return result

    async def _admit_without_waiting_for_the_turn(
        self, request: str, images: list[dict[str, str]] | None
    ) -> None:
        """Admit ``request`` as a turn, reporting only the IMMEDIATE refusals.

        ``prompt`` resolves its receipt on the durable transcript append, which
        the drain performs only when it reaches that command — so awaiting it
        holds this coroutine for the whole of any turn queued ahead. That would
        be paid on the ``slash_result`` REQUEST/RESPONSE, which is a different
        thing from the ``prompt`` op's own fire-and-forget wait: the reply frame
        would be withheld for the duration of the earlier turn, the viewer's
        ``ACK_TIMEOUT_S`` (15 s) would elapse on anything longer, and the old
        viewer would print a transport error for a request that WAS admitted —
        a worse failure than the silent drop this method exists to repair. The
        per-connection reader is strictly serial, so every later op from that
        viewer would queue behind the park as well (review round 1, R1-1).

        The ``is_streaming`` branch above does not cover it: the flag is still
        False during the drain's pre-streaming prelude (lock acquire, record
        and journal flushes), so a prompt sent moments before this lands in
        exactly that window.

        So the admission is DISPATCHED and this returns as soon as its outcome
        can no longer be reported synchronously. The refusals worth reporting —
        a closing session, a full queue, a rejected reservation — all raise in
        ``prompt``'s synchronous prelude, before it ever awaits, so they surface
        here and still become the warning receipt. Anything after that point is
        a turn that is genuinely running and whose result belongs in the
        transcript, not in this receipt.

        Bounded in LOOP TURNS rather than seconds deliberately: a wall-clock
        budget here would be a bet on machine load, whereas "the task has had
        its synchronous prelude" is a fact about scheduling that holds under any
        contention (AGENTS.md, "Wait on the event, never on the clock").
        """
        # Bounded HERE and handed on DECODED, so ``prompt``'s prelude stays
        # await-free and the loop-turn budget below measures what it claims to.
        #
        # The budget gives ``prompt`` a few turns to raise its reportable
        # refusals, on the documented premise that it raises them before its
        # first await. Bounding images inside ``prompt`` broke that premise: a
        # thread hop does not resolve within the budget however warm the pool
        # is, because ``to_thread`` always yields at least once. An
        # image-carrying admission then reached the budget with its prelude
        # unrun, was judged "genuinely queued behind a running turn", and its
        # refusals went to the detached log instead of the warning receipt --
        # the silent drop this method exists to repair, back for requests with
        # attachments.
        blocks = await _image_blocks_async(images)
        task = asyncio.ensure_future(
            self.prompt(request, images=blocks, command_id=str(uuid.uuid4()))
        )
        # One turn is enough today — nothing before ``prompt``'s first suspension
        # point yields — but a couple of extra turns costs nothing and keeps this
        # correct if a future edit puts another await in that prelude.
        for _ in range(_ADMISSION_PRELUDE_TURNS):
            if task.done():
                break
            await asyncio.sleep(0)
        if task.done():
            # Re-raises the synchronous refusal into the caller's warning path.
            # A cancelled task is not a refusal to report: the session is going
            # away and the receipt is the least of it.
            if not task.cancelled():
                task.result()
            return
        # Still parked on the durable append, i.e. genuinely queued behind a
        # running turn. Let it finish on its own; the user row appears through
        # the event relay exactly as it would for any other admitted prompt.
        task.add_done_callback(_log_detached_admission)

    async def _slash_result(
        self, command: str, args: str, SlashResult: Any, locality: str = "local"
    ) -> Any:
        """Dispatch one routed slash command. Mirrors ``OperatorApp._slash_result``.

        Only the commands a viewer ROUTES reach here; process- and
        terminal-local ones (``/quit``, ``/resume``, pickers) never leave the
        viewer. Anything not handled falls through to an honest notice rather
        than the transport's ``unknown op``, because a user typing a command
        this runtime does not implement needs to know what to do instead.

        The word is resolved to its registry PRIMARY name first, exactly as the
        app-hosted twin does: the branches below match literals, so an ALIAS off
        the wire (``/title``, ``/models``, ``/recall``) would otherwise fall
        past every one of them and collect the terminal-only refusal for a
        command this runtime implements.
        """
        from local_operator.slash_commands import primary_slash_name

        command = primary_slash_name(command)
        session = self._session
        if command == "desktop_mcp":
            from local_operator.mcp.config import MCPConfigWriteError
            from local_operator.mcp.desktop import MCPControl, MCPDesktop

            if self._desktop_mcp is None:
                self._desktop_mcp = MCPDesktop(session, self._mcp_grant_tasks, self._desktop_cwd)
            try:
                data = await self._desktop_mcp.execute(MCPControl.model_validate_json(args))
            except (ValueError, MCPConfigWriteError):
                # Refusals are protocol data, not socket outages. Config errors
                # can quote credentials, so only a bounded code crosses back.
                return SlashResult(kind="error", data={"code": "mcp_control_refused"})
            return SlashResult(kind="block", data=data)
        if command == "fork":
            from local_operator.fork import fork_session
            from local_operator.paths import config_dir

            if getattr(session, "_compacting", False):
                raise ValueError("Wait for compaction to finish before forking")
            if getattr(session, "is_streaming", False):
                if session.has_pending_fork():
                    raise ValueError("A fork is already waiting for a safe boundary")
                settled: asyncio.Future[str] = self._loop.create_future()

                def complete(fork_id: str, error: str) -> None:
                    if not settled.done():
                        if error:
                            settled.set_exception(RuntimeError("The fork could not be created"))
                        else:
                            settled.set_result(fork_id)

                session.request_fork(config_dir(), on_complete=complete)
                try:
                    fork_id = await settled
                except BaseException:
                    session.cancel_fork()
                    raise
            else:
                fork_id = await asyncio.to_thread(fork_session, config_dir(), session.session_id)
            return SlashResult(kind="block", data={"type": "forked", "session_id": fork_id})
        if command == "context":
            return self._context_slash(session, SlashResult)
        if command == "team":
            return self._team_slash(session, args, SlashResult)
        if command == "agent":
            return self._agent_slash(session, args, SlashResult)
        if command == "mcp":
            return await self._mcp_slash(session, args, SlashResult, locality)
        if command == "model":
            return await self._model_slash(session, args, SlashResult)
        if command == "goal":
            return self._goal_slash(session, args, SlashResult)
        if command == "rename":
            return await self._rename_slash(session, args, SlashResult)
        if command == "effort":
            return await self._effort_slash(session, args, SlashResult)
        if command == "fast":
            return self._fast_slash(session, args, SlashResult)
        if command == "approvals":
            return self._approvals_slash(session, args, SlashResult)
        if command == "compact":
            return self._compact_slash(session, SlashResult)
        if command == "loop":
            driver = self._loop_driver()
            if args.lower() in ("stop", "cancel", "abort"):
                await driver.cancel()
            elif args.lower() != "status":
                if driver.running:
                    return SlashResult(
                        kind="error", text="A loop is already running", data={"code": "loop_busy"}
                    )
                try:
                    driver.start(args.strip(), str(getattr(session, "goal", "")))
                except ValueError as error:
                    return SlashResult(kind="error", text=str(error), data={"code": "loop_invalid"})
            return SlashResult(kind="block", data={"type": "loop", **driver.state})
        # NEVER TELL AN ATTACHED USER TO REATTACH. Every session on this
        # release is detached, and the viewer routes these before its own
        # local handling — so a user sitting at a terminal was told to take an
        # action they had already taken and could not take again, which reads
        # as a bug rather than a limitation (round 4, R2/U13). `/mcp`,
        # `/team` and `/agent` are a fair "not here": they read registry and
        # profile data that has no session-side home. Say that instead, and
        # name where the command does work.
        return SlashResult(
            kind="notice",
            text=f"/{command} reads this machine's configuration, which the session "
            f"process does not hold — run it from a terminal on the machine you "
            f"want to configure",
            style="warning",
        )

    def _goal_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        arg = arg.strip()
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
                    f"goal set: shortened to the {MAX_GOAL_CHARS}-character cap. "
                    "Sending the full request."
                ),
                style="warning",
                data={"type": "goal_set", "stored": stored, "request": arg.strip()},
            )
        return SlashResult(
            kind="notice",
            text="goal set",
            style="info",
            data={"type": "goal_set", "stored": stored, "request": arg.strip()},
        )

    async def _rename_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """``/title`` on a detached runtime: report, set, or refresh.

        Async only for the refresh branch, which spends a provider call and is
        awaited rather than detached: this method's return value IS the receipt
        the invoking terminal or phone renders, so answering before the call
        settled would report a title that had not been decided.
        """
        from local_operator.session import naming

        try:
            is_refresh, name = naming.parse_title_arg(arg or "")
        except ValueError as error:
            return SlashResult(kind="notice", text=str(error), style="warning")
        if is_refresh:
            return await self._title_refresh_slash(session, SlashResult)
        if not name:
            current = getattr(session, "conversation_name", "") or ""
            text = (
                f"name: {current} — /title <words>, or /title refresh"
                if current
                else "no name set — /title <words> to set one"
            )
            return SlashResult(kind="notice", text=text, style="info")
        setter = getattr(session, "set_conversation_name", None)
        if not callable(setter):
            return SlashResult(kind="notice", text="session is still starting…", style="warning")
        stored = setter(name)
        self._publish_name()
        return SlashResult(kind="notice", text=f"renamed to {stored or name}", style="info")

    async def _title_refresh_slash(self, session: Any, SlashResult: Any) -> Any:
        """``/title refresh`` on a detached runtime.

        Shares :func:`naming.refresh_title` and the release-on-success rule with
        the TUI's own handler, so a session owned by a runtime and one owned by
        an app cannot disagree about what a refresh does to ``user_set``.

        No generation stamp here, and that is not an omission. The TUI's twin
        needs one because :meth:`OperatorApp._name_conversation_worker` stores
        its answer whenever the generation still matches, so a call dispatched
        before a refresh will happily overwrite the refreshed title. This
        runtime's :meth:`_name_conversation_worker` instead re-reads
        ``conversation_name`` AFTER its await and returns when anything is
        already set, so a refresh that landed first is what the late call sees
        and declines to overwrite. The guard is the name check, not a counter —
        and it is why there is no generation concept on this path to stamp.
        """
        from local_operator.session import naming

        setter = getattr(session, "set_conversation_name", None)
        complete_once = getattr(session, "complete_once", None)
        if not callable(setter) or not callable(complete_once):
            return SlashResult(kind="notice", text="session is still starting…", style="warning")
        current = getattr(session, "conversation_name", "") or ""
        # One budget over the history read AND the naming call, and one failure
        # policy for both — shared with the app-hosted twin so the two cannot
        # drift on the deadline the way they once drifted on the receipt.
        result = await naming.routed_refresh(current, session)
        # The second condition is the rename-during-the-call guard: a `/rename`
        # landing while this was in flight outranks an answer decided against a
        # title no longer in force, and storing over it would strip the latch
        # protecting the words the user just typed.
        standing = getattr(session, "conversation_name", "") or ""
        if not result.changed or standing != current:
            return SlashResult(
                kind="notice",
                text=naming.refresh_receipt(result, standing, stored=False),
                style="info",
            )
        state = getattr(session, "conversation_name_state", None)
        release = getattr(state, "release_user_set", None)
        if callable(release):
            # Released only on success, and only here: a refresh that changed
            # nothing must leave a name the user typed exactly as they left it.
            release()
        # `setter` came off a duck-typed handle, so its return is untyped; the
        # stored title is the string the receipt has to quote.
        stored = str(setter(result.title, user_set=False) or result.title)
        self._publish_name()
        return SlashResult(kind="notice", text=naming.refresh_receipt(result, stored), style="info")

    def _publish_name(self) -> None:
        """Push a changed conversation name onto every surface that shows it.

        THREE calls, and dropping any one of them leaves the name visible in a
        different place from where it is true:

        * ``_refresh_state`` rebuilds the projection. This is the one that was
          missing, and its absence was total rather than transient: the
          projection's ``conversation_name`` has exactly one writer, the
          heartbeat republishes the projection's copy every 15 s, and
          ``_republish`` does not carry the name field at all — so a renamed
          session was re-asserted under its OLD name forever, and an attached
          phone kept the stale header for the life of the session. The runtime's
          own naming worker has always made this call; the slash path did not.
        * ``_notify`` pushes that projection to attached clients.
        * ``_republish`` updates the discovery record, so ``lop sessions`` and
          the picker see the change without waiting for the next heartbeat.

        Extracted from ``_rename_slash`` when the refresh branch became a second
        writer: two copies of this would be two chances for one of them to skip
        a step and leave a session listed under a name it no longer has — which
        is precisely what the missing ``_refresh_state`` was, and fixing it in
        this one seam repairs ``/title <words>`` along with the refresh.
        """
        self._refresh_state()
        self._notify()
        republish = getattr(self._registrant, "_republish", None)
        if callable(republish):
            try:
                republish()
            except Exception:  # noqa: BLE001 — a stale name is not worth a failure
                logger.debug("could not republish the renamed record", exc_info=True)

    def _context_slash(self, session: Any, SlashResult: Any) -> Any:
        """The routed ``/context``: the token breakdown, computed HERE.

        `Session.context_breakdown()` is plain session state, so the numbers
        can only be right on the process that holds the session — which is
        the whole reason the command is routed. Returned as a ``block`` whose
        rows the invoking terminal renders locally, identically to the app's
        own handler (`app.py::_context_slash_result`), so the two surfaces
        cannot drift into two different answers.
        """
        from local_operator.session.frontend_state import (
            format_context_tokens,
            format_window,
        )

        breakdown = getattr(session, "context_breakdown", None)
        if not callable(breakdown):
            return SlashResult(kind="notice", text="context breakdown unavailable.", style="info")
        try:
            data = cast("dict[str, int]", breakdown())
        except Exception:  # noqa: BLE001 — a breakdown is never worth an error
            logger.debug("context breakdown failed", exc_info=True)
            return SlashResult(kind="notice", text="context breakdown unavailable.", style="info")

        total = int(data.get("total", 0))
        window = max(int(data.get("context_window", 0)), 1)
        pct = total / window * 100

        def estimated(value: int) -> str:
            return f"~{format_context_tokens(int(value))}"

        rows = [
            ("Instructions", estimated(data.get("instructions", 0))),
            ("Tool inventory", estimated(data.get("tool_inventory", 0))),
            ("Tool schemas", estimated(data.get("tool_schemas", 0))),
            ("Environment", estimated(data.get("environment", 0))),
            ("Skills / MCP / goal", estimated(data.get("knowledge_mcp_goal", 0))),
            ("Messages", estimated(data.get("messages", 0))),
            ("Total", f"{estimated(total)} / {format_window(window)} ({pct:.1f}%)"),
        ]
        if data.get("cache_read"):
            rows.append(("Last cache read (exact)", format_context_tokens(int(data["cache_read"]))))
        return SlashResult(
            kind="block",
            data={"type": "context", "items": rows, "title": "Estimated next request"},
        )

    def _team_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """The routed ``/team``: list, and ATTACH, from the SESSION's registry.

        `Session.team_registry` is session state, so the listing is answered
        here. The ATTACH is answered here too, and that is the whole point:
        attaching stamps the roster and both briefs onto the manager
        (``Session.attach_team``), so it must run where the session state that
        builds the next turn actually lives. A viewer has no such state.

        This used to return ``noop {"type": "team_mutate"}`` for every
        argument-carrying form, on the theory that the invoking terminal would
        host the interaction itself the way bare ``/model`` hosts its picker.
        Nothing ever consumed it: ``_render_authoritative_slash`` returns
        without printing on a ``noop``, so on a viewer — which since 0.46.0 is
        EVERY fresh `lop` — `/team <name> <request>` sent no prompt, wrote no
        transcript row and printed no notice. Total silence, which is what the
        operator reported. (`tests/unit/tui/test_slash_echo.py` measured that
        silence accurately and asserted only that the refusal copy promised no
        false retry, so the behaviour was known and pinned as wording.)

        ``chart`` is deliberately still ``noop``: it opens an org-chart VIEW in
        the invoking terminal, which is local UI the owner cannot paint — the
        same argument bare ``/model`` makes for hosting its own picker. The
        viewer handles that token before it ever routes.

        The turn itself is NOT started here. This returns an ``attached``
        receipt and the viewer submits the request through its ordinary prompt
        path, so the request reaches the model as a real user turn with the
        invoker's own images and paste expansion — the one authority for "what
        did the user actually send" stays in one place.
        """
        registry = getattr(session, "team_registry", None)
        if registry is None or not hasattr(registry, "list_teams"):
            return SlashResult(
                kind="notice",
                text="teams are unavailable in this session. Ask the agent to create one.",
                style="warning",
            )
        if arg:
            return self._team_attach_slash(session, arg, SlashResult)
        try:
            teams = list(registry.list_teams())
        except Exception as exc:  # noqa: BLE001 — a listing is never worth an error
            return SlashResult(kind="notice", text=f"could not list teams: {exc}", style="warning")
        if not teams:
            return SlashResult(
                kind="notice", text="no teams yet. Ask the agent to create one.", style="info"
            )
        # ``member_count()``, matching the TUI's own producer (D2). The old
        # `len(members) + 1` assumed the manager is not on the roster — false
        # for real teams — and collapsed multi-count slots; the plural was also
        # keyed to a different number than the one displayed. A detached
        # runtime and an in-process one must answer the same question with the
        # same number.
        items = [
            (
                team.name,
                f"Led by {team.manager} · {team.member_count()} "
                f"{'member' if team.member_count() == 1 else 'members'}",
                (team.description or "").strip(),
            )
            for team in teams
        ]
        return SlashResult(kind="block", data={"type": "team_list", "items": items})

    def _team_attach_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """``/team <name> [<request>]`` on the owner: resolve, then attach.

        The name grammar MUST match ``app.py::_cmd_team`` token for token, or
        the same command means different things depending on which process ran
        it. That includes the single leading ``=`` escape, which exists so a
        team legitimately named ``chart`` is still reachable — ``=`` cannot
        occur in a validated team name, so stripping one can never shadow a
        real team.

        ``chart`` never arrives here (the viewer keeps it local, see
        ``_team_slash``), so this does not re-implement the reserved word.

        A refusal is a NOTICE and an attach is a ``team_attached`` receipt
        carrying the request; the viewer prints the receipt and submits the
        request. The two are distinguishable by kind so a failed lookup can
        never be rendered as a successful attach.
        """
        name, _, request = arg.partition(" ")
        name = name.strip()
        if name.startswith("="):
            name = name[1:]
        request = request.strip()
        try:
            team = registry_team = session.team_registry.get_team_by_name(name)
        except Exception as exc:  # noqa: BLE001 — a bad registry read is a notice
            return SlashResult(
                kind="notice", text=f"could not load team {name!r}: {exc}", style="warning"
            )
        if registry_team is None:
            return SlashResult(
                kind="notice",
                text=(
                    f"no team named {name!r}. Run /team to list teams, "
                    "or ask the agent to create one."
                ),
                style="warning",
            )
        attach = getattr(session, "attach_team", None)
        if not callable(attach):
            # An owner that cannot attach must REFUSE rather than let the
            # viewer send the request anyway: a turn that runs with no roster
            # and no briefs while the receipt says "<manager> is coordinating"
            # is a wrong persona answering confidently, which is worse than a
            # refusal because nothing on screen says which one you got.
            return SlashResult(
                kind="notice",
                text="this session cannot run a team. /team chart <name> shows a roster",
                style="warning",
            )
        try:
            attach(team)
        except Exception as exc:  # noqa: BLE001 — a failed attach must not kill the turn
            return SlashResult(
                kind="notice", text=f"could not attach team {team.name!r}: {exc}", style="warning"
            )
        # The band and the discovery record both name the attached team, so the
        # projection has to refresh before the viewer paints its receipt.
        self._notify()
        return SlashResult(
            kind="notice",
            text=(
                f"team {team.name} is ready. {team.manager} leads it. "
                f"Send a request with /team {team.name} <message>."
                if not request
                else f"sending to {team.name}. {team.manager} is coordinating."
            ),
            style="info",
            data={
                "type": "team_attached",
                "team": team.name,
                "manager": team.manager,
                "request": request,
            },
        )

    def _agent_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """The routed ``/agent``: list in the invoker, ATTACH here.

        The listing stays ``noop`` on purpose: its rows carry role/specialist
        facts assembled by the frontend's own profile resolver, and a second
        assembly here would be a second source of truth for the same list.

        The mutating forms do NOT stay ``noop``, for the reason spelled out in
        ``_team_slash``: attaching a profile mutates session state (the
        instructions ride the volatile tail) and nothing consumed the
        ``agent_mutate`` receipt, so `/agent <name>` on a viewer was silent.
        """
        if not arg:
            return SlashResult(kind="noop", data={"type": "agent_list", "args": arg})
        name, _, request = arg.partition(" ")
        name = name.strip()
        request = request.strip()
        # ``clear``/``none`` is the DETACH verb, mirroring ``_cmd_agent``: only
        # the bare verb detaches, so ``/agent clear <text>`` stays a (mistyped)
        # attach and reports the unknown name rather than silently detaching.
        if name.lower() in ("clear", "none") and not request:
            detach = getattr(session, "clear_agent_profile", None)
            if not callable(detach):
                return SlashResult(kind="notice", text="nothing to detach", style="info")
            detach()
            self._notify()
            return SlashResult(
                kind="notice",
                text="this session uses its base instructions",
                style="info",
                data={"type": "agent_attached", "agent": "", "request": ""},
            )
        attach = getattr(session, "attach_agent_profile", None)
        if not callable(attach):
            return SlashResult(
                kind="notice", text="this session cannot adopt an agent profile", style="warning"
            )
        try:
            resolved = attach(name)
        except Exception as exc:  # noqa: BLE001 — a failed attach must not kill the turn
            return SlashResult(
                kind="notice", text=f"could not attach agent {name!r}: {exc}", style="warning"
            )
        if not resolved:
            return SlashResult(
                kind="notice",
                text=f"no agent named {name!r}. Run /agent to list agents.",
                style="warning",
            )
        self._notify()
        return SlashResult(
            kind="notice",
            text=f"{resolved} is answering in this session.",
            style="info",
            data={"type": "agent_attached", "agent": resolved, "request": request},
        )

    async def _mcp_slash(
        self,
        session: Any,
        arg: str,
        SlashResult: Any,
        locality: str = "local",
    ) -> Any:
        """The routed ``/mcp``: status from the session's own manager.

        The bare listing is kept LOCAL by the viewer's dispatch (it reads the
        identical rows from its mcp facade), so what reaches here is either
        the empty case or a subcommand.

        A grant (``login``/``logout``/``reauth``) RUNS HERE when the invoking
        client is on this machine, which today it always is: the control
        socket binds ``127.0.0.1`` only. This used to be refused outright with
        "run it from a terminal on that machine", which was self-contradictory
        on a detached session — the terminal routes the verb to the owner, the
        owner told the user to type it in a terminal, and there was no third
        place to type it. The result was that ``/mcp reauth`` could not be run
        at all once a session had detached, which is precisely when an expired
        credential needs it.

        Locality is now the CLIENT's declared property rather than a guess
        made here (see ``ClientLocality``), so a future relay carrying a
        phone's command still gets the refusal, aimed at the case it describes.
        """
        from local_operator.mcp.grants import (
            GRANT_SUBCOMMANDS as _MCP_GRANT_SUBCOMMANDS,
        )
        from local_operator.session.frontend_state import (
            MCP_SUBCOMMANDS as _MCP_SUBCOMMANDS,
        )

        parts = (arg or "").split()
        sub = parts[0].lower() if parts else ""

        # UNKNOWN VERBS ARE REFUSED BY NAME, and this check comes first. Every
        # unrecognised token used to fall through to the server listing at the
        # bottom, which is a PLAUSIBLE answer to `add` — the user asked about
        # servers and got a table of servers — so a typo, or `add` itself, read
        # as "done, here is the current state" while nothing had happened
        # (round 5, U15). The attached path has always validated this way.
        if sub and sub not in _MCP_SUBCOMMANDS:
            return SlashResult(
                kind="notice",
                text=f"unknown mcp subcommand: {parts[0]} — try "
                f"/mcp {'|'.join(_MCP_SUBCOMMANDS)} <name>",
                style="warning",
            )
        # The same fixed-arity refusals the attached path applies, in the same
        # order, so one typed string is answered identically wherever it is
        # typed. Acting on something other than what the user described is the
        # mistake this whole command guards against.
        if sub == "list" and len(parts) > 1:
            return SlashResult(
                kind="notice",
                text=f"/mcp list takes no arguments — got {' '.join(parts[1:])!r}",
                style="warning",
            )
        if sub in _MCP_GRANT_SUBCOMMANDS:
            # Same fixed-arity refusals the attached path applies, so one typed
            # string is answered identically wherever it is typed.
            if len(parts) < 2:
                return SlashResult(kind="notice", text=f"usage: /mcp {sub} <name>", style="warning")
            if len(parts) > 2:
                return SlashResult(
                    kind="notice",
                    text=f"/mcp {sub} takes one server name — got {' '.join(parts[1:])!r}",
                    style="warning",
                )
            from local_operator.mcp.grants import start_grant

            text, kind = await start_grant(
                session,
                sub,
                parts[1],
                browser_is_reachable=locality != "remote",
                notify=self._grant_notice,
                spawn=self._spawn_grant,
            )
            return SlashResult(kind="notice", text=text, style=kind)
        # `add`/`remove` write the GLOBAL mcp.json and reconnect THIS session's
        # manager, so they are genuinely our work — the follower's facade is
        # read-only and its filesystem is not the one this session reads its
        # servers from. Shared with the terminal via `mcp.verbs` rather than
        # reimplemented: the refusals are the substance of these commands.
        if sub in ("add", "remove"):
            from local_operator.mcp.verbs import mcp_add_result, mcp_remove_result

            if sub == "remove":
                if len(parts) < 2:
                    return SlashResult(
                        kind="notice", text=f"usage: /mcp {sub} <name>", style="warning"
                    )
                if len(parts) > 2:
                    return SlashResult(
                        kind="notice",
                        text=f"/mcp {sub} takes one server name — got {' '.join(parts[1:])!r}",
                        style="warning",
                    )
                text, kind = mcp_remove_result(parts[1], self._reconnect_mcp)
            else:
                text, kind = mcp_add_result(parts[1:], self._reconnect_mcp)
            style = "info" if kind == "info" else kind
            return SlashResult(kind="notice", text=text, style=style)
        # What remains is a LISTING (bare, or `list`). The bare form never
        # reaches here — the viewer pulls it back to local because its own
        # facade holds the identical rows — so this answers the explicit
        # `list` and the empty case from the session's own manager.
        manager = getattr(session, "mcp_manager", None)
        servers = getattr(manager, "servers", None) if manager is not None else None
        if not servers:
            return SlashResult(kind="notice", text="no MCP servers configured.", style="info")
        return SlashResult(kind="block", data={"type": "mcp"})

    def _grant_notice(self, text: str, kind: str) -> None:
        """Report a settled MCP grant to every front end watching this session.

        The grant's receipt cannot be its ``result`` frame: the exchange waits
        on a human and the invoking client times out after ``ACK_TIMEOUT_S``.
        A ``NoticeEvent`` is the channel that already reaches every attached
        terminal (and the phone projection), so the outcome lands wherever the
        user is looking rather than only in the terminal that happened to type
        the verb — which for a detached session may well be gone by then.
        """
        from local_operator.harness.types import NoticeEvent

        emit = getattr(self._session, "_emit", None)
        if not callable(emit):
            logger.info("MCP grant: %s", text)
            return
        # ``NoticeEvent`` carries the narrower info/warning/error trio; a
        # grant's "success" maps onto ``info`` rather than inventing a kind
        # the event schema does not define.
        event_kind = kind if kind in ("info", "warning", "error") else "info"
        typed_emit = cast("Callable[[Any], Awaitable[Any]]", emit)

        async def _emit_notice() -> None:
            try:
                await typed_emit(NoticeEvent(text=text, kind=event_kind))
            except Exception:  # noqa: BLE001 — a failed notice must not kill the loop
                logger.debug("MCP grant notice failed", exc_info=True)

        # NOT through ``_spawn_grant``: this is called FROM a grant task, and
        # that helper supersedes the running grant — the notice would cancel
        # the very task reporting it. A notice is also a sub-second local
        # emission with no exclusivity to enforce, so it belongs on the
        # ordinary best-effort holder.
        task = self._loop.create_task(_emit_notice())
        self._mcp_reload_tasks.add(task)
        task.add_done_callback(self._mcp_reload_tasks.discard)

    def _spawn_grant(self, coro: Awaitable[None]) -> None:
        """Run a detached grant on this runtime's own loop, one at a time.

        Held in a set because ``create_task`` keeps only a weak reference, so a
        bare task can be collected mid-flight and the exchange would simply
        stop with no receipt. Its OWN set rather than ``_mcp_reload_tasks``
        (review F4): a config reload is sub-second best-effort work, while a
        grant can sit for ten minutes waiting on a person, and ``dispose``
        must be able to cancel the second without waiting on it.

        **Superseding is the point of the single slot** (review F3). Every
        grant binds the same loopback redirect port, so two concurrent
        exchanges race for it and the loser fails with a bind error that
        describes nothing the user did. The TUI has always serialised these
        through an exclusive ``mcp-login`` worker group; the runtime had no
        equivalent, so a detached session could run two at once. Cancelling the
        previous one reproduces that behaviour: its ``_settle`` reports the
        cancellation through ``notify``, so the superseded grant still gets an
        ending rather than vanishing.
        """
        for previous in list(self._mcp_grant_tasks):
            if not previous.done():
                previous.cancel()
        task = self._loop.create_task(cast("Coroutine[Any, Any, None]", coro))
        self._mcp_grant_tasks.add(task)
        task.add_done_callback(self._mcp_grant_tasks.discard)

    def _reconnect_mcp(self) -> None:
        """Re-read the config and reconnect after ``/mcp add|remove`` wrote it.

        Without this the command is true on disk and invisible in the session:
        the manager holds the configs it discovered at boot. Scheduled on this
        runtime's own loop rather than awaited — the reconnect is a network
        round trip and the receipt is already correct without it, which is the
        same best-effort stance the terminal takes.
        """
        manager = getattr(self._session, "mcp_manager", None)
        reload = getattr(manager, "reload", None)
        if not callable(reload):
            return
        # ``getattr`` on a duck-typed manager yields ``object``; the callable
        # check above is the real guard, so name the awaitable shape for the
        # checker (the same cast `app.py` makes at its own reload site).
        typed_reload = cast("Callable[[], Awaitable[Any]]", reload)

        async def _reload() -> None:
            try:
                await typed_reload()
            except Exception:  # noqa: BLE001 — a failed refresh must not fail the command
                logger.debug("MCP reload after a config change failed", exc_info=True)

        # Fire-and-forget on the loop we are already on. Held in a set so the
        # task is not garbage-collected mid-flight.
        task = self._loop.create_task(_reload())
        self._mcp_reload_tasks.add(task)
        task.add_done_callback(self._mcp_reload_tasks.discard)

    async def _model_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """The routed ``/model <provider>/<id>``: a REAL switch on this session.

        Routed through :meth:`set_model` — the same op the phone's and the
        picker's switches already use — so a typed command and a picked row
        cannot diverge. Refusing this while the picker's identical mutation
        succeeded over the same socket was the sharpest edge of R2/U13.

        The persist half is declined for the machine-locality reason
        `/approvals default` gives: a default belongs to the terminal whose
        launches it governs, not to a runtime that outlives it. ``saved`` is
        NOT declined on those grounds — it only READS that default and switches
        this session, which is this handle's own mutation (QA round 2, Q49).
        """
        target = (arg or "").strip()
        lowered = target.lower()
        if not target:
            # The bare form opens the viewer's own picker; it should never
            # have been routed here, but answering with the current model is
            # more useful than an error.
            return SlashResult(
                kind="notice",
                text=f"model: {getattr(session, 'model_label', '') or 'unknown'}",
                style="info",
            )
        if lowered == "default" or lowered.startswith("default "):
            return SlashResult(
                kind="notice",
                text="/model default persists to the local machine's config — run it "
                "on the terminal whose launches it should govern; /model <p>/<id> "
                "switches the shared session now",
                style="warning",
            )
        if lowered == "saved":
            # ``/model saved`` — adopt the CONFIGURED default (#369). Handled
            # here because this method is what serves a DETACHED runtime's
            # `/model`: `OperatorApp` intercepts `saved` before routing, so a
            # local pane always worked while the phone and any viewer on a
            # runtime-owned session fell through to the `<provider>/<model-id>`
            # usage error — `saved` has no `/` (QA round 2, Q49). That made the
            # keep notice this same change emits (`session.py`, "/model saved
            # adopts it") a dead end on the one surface that prints it from a
            # runtime, and contradicted the `/help` text round 1's U5 added.
            #
            # UNLIKE `/model default`, this is a READ of config, not a write, so
            # the machine-locality reason that declines the persist above does
            # not apply: adopting a value is a switch on this session, which is
            # exactly what this handle owns. The read is direct rather than a
            # cached boot value, matching `OperatorApp._cmd_model_saved`, so a
            # default written during this session is what gets adopted.
            try:
                from local_operator.config import ConfigManager
                from local_operator.paths import config_dir

                manager = ConfigManager(config_dir())
                saved_provider = str(manager.get_config_value("hosting", "") or "").strip().lower()
                saved_model = str(manager.get_config_value("model_name", "") or "").strip()
            except Exception as error:  # noqa: BLE001 — reported, never fatal
                return SlashResult(
                    kind="notice",
                    text=f"could not read the saved default: {error}",
                    style="error",
                )
            if not saved_provider or not saved_model:
                # Honest "there is nothing to go back to", in the app's own
                # words so both surfaces answer an empty config identically.
                return SlashResult(
                    kind="notice",
                    text="no boot default saved yet — /model default <provider>/<model-id> "
                    "sets one",
                    style="warning",
                )
            provider, model_id = saved_provider, saved_model
        else:
            provider, sep, model_id = target.partition("/")
            if not sep or not model_id:
                return SlashResult(
                    kind="notice",
                    text="usage: /model <provider>/<model-id> "
                    "(e.g. openrouter/deepseek/deepseek-chat)",
                    style="warning",
                )
        old_label = getattr(session, "model_label", "")
        try:
            await self.set_model(provider.lower(), model_id)
        except Exception as error:  # noqa: BLE001 — an unknown model is a user error
            return SlashResult(kind="notice", text=f"cannot switch model: {error}", style="error")
        return SlashResult(
            kind="notice",
            text=f"model: {old_label} → {getattr(session, 'model_label', '')} (this session)",
            style="info",
        )

    async def _effort_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """Report the reasoning effort, and CHANGE it — the mutation is ours.

        Round 3 declined the mutation on the grounds that it "reaches into the
        model picker's widget state and the machine's saved default". That was
        wrong, and the reviewer was right to overrule it: :meth:`set_effort`
        on this same handle performs exactly this mutation against this same
        session, and `server.py` already routes an op to it. Only the app's
        extra bookkeeping (the picker's row highlight, the saved default) is
        terminal-local, and none of it is required to change the effort.
        """
        spec = getattr(session, "model", None)
        label = getattr(session, "model_label", "") or "this model"
        if spec is None:
            return SlashResult(kind="notice", text="session is still starting…", style="warning")
        rungs = list(getattr(spec, "reasoning_efforts", []) or [])
        current = getattr(spec, "reasoning_effort", None)
        wanted = arg.strip().lower()
        if not wanted:
            if not rungs:
                return SlashResult(
                    kind="notice", text=f"effort is not adjustable on {label}", style="info"
                )
            ladder = ", ".join(f"[{r}]" if r == current else r for r in rungs)
            return SlashResult(kind="notice", text=f"effort on {label}: {ladder}", style="info")
        try:
            detail = await self.set_effort(wanted)
        except Exception as error:  # noqa: BLE001 — a bad rung is a user error
            return SlashResult(kind="notice", text=str(error), style="warning")
        return SlashResult(kind="notice", text=str(detail), style="info")

    def _fast_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """Report or switch fast mode on the runtime's own spec.

        The same rules as the terminal's ``/fast`` (`OperatorApp._cmd_fast`),
        restated here because the detached runtime builds its requests from
        ITS spec: a phone toggling the dial must reach the spec the next
        provider call is built from, and the app's copy of the rule is not
        loaded in a headless process. Bare toggles, ``on``/``off`` name the
        resulting state, ``status`` only reports. Session-scoped and never
        persisted — fast mode is billed at a premium, so it must not outlive
        the task it was switched on for.
        """
        spec = getattr(session, "model", None)
        label = getattr(session, "model_label", "") or "this model"
        if spec is None or not hasattr(session, "set_model"):
            return SlashResult(kind="notice", text="session is still starting…", style="warning")
        if not getattr(spec, "supports_fast_mode", False):
            return SlashResult(
                kind="notice", text=f"fast mode: not available on {label}", style="info"
            )
        current = bool(getattr(spec, "fast_mode", False))
        wanted = (arg or "").strip().lower()

        def status() -> Any:
            text = (
                f"fast mode: on for {label} — faster output at premium pricing"
                if current
                else f"fast mode: off for {label} — /fast turns it on at premium pricing"
            )
            return SlashResult(kind="notice", text=text, style="info")

        if wanted in ("status", "show"):
            return status()
        if wanted in ("on", "yes", "true", "enable", "enabled"):
            target = True
        elif wanted in ("off", "no", "false", "disable", "disabled"):
            target = False
        elif not wanted:
            target = not current
        else:
            return SlashResult(
                kind="notice",
                text=f"fast mode: {wanted!r} is not one of on, off, status — bare /fast toggles",
                style="warning",
            )
        if target == current:
            return status()
        session.set_model(spec.model_copy(update={"fast_mode": target}))
        if target:
            # An explicit re-ask clears the driver's refusal latch (see
            # `FailoverRouteState.fast_refused`); the terminal does the same.
            forget = getattr(getattr(session, "_stream_fn", None), "forget_fast_refusal", None)
            if callable(forget):
                forget()
        self._notify()
        # Same receipt as the terminal's `_fast_receipt` (design D2/D6): no
        # label, because an aggregator label pushes the line past the notice
        # width and orphans the last word.
        text = (
            "fast mode: off → on — faster output at premium pricing"
            if target
            else "fast mode: on → off — standard speed and pricing"
        )
        return SlashResult(kind="notice", text=text, style="info")

    def _approvals_slash(self, session: Any, arg: str, SlashResult: Any) -> Any:
        """Report or switch the gate the RUNTIME's tools actually consult.

        `self._auto_approve` is the real gate here (see `_install_gates`), so
        unlike the viewer's own widget flag this switch is the one the engine
        honours. The persist half is declined for the same machine-locality
        reason the app gives: a default belongs to the terminal that launches
        sessions, not to a runtime that outlives it.
        """
        argument = (arg or "").strip().lower()
        if not self._gates_installed:
            # Publishing an exec owner did not replace its original gate, so
            # mutating this handle's flag would lie about what tools consult.
            # Gate routing is a launch decision, never a side effect of attach.
            return SlashResult(
                kind="notice",
                text=(
                    "exec uses its original headless approval gate"
                    + (" (--yolo is active)" if self._auto_approve else " (non-TTY requests deny)")
                    + "; launch with --control for supervisor approval controls"
                ),
                style="warning" if argument else "info",
            )
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
            # Compared against the FILE, not against a cached default (UX round
            # 1, U1/U2). This is the surface whose job is "what is in effect and
            # why", and a session that kept its own mode against a config edit
            # has to be able to learn that here — otherwise the one place a user
            # asks reports a matched pair while the two genuinely disagree.
            on_disk = self._configured_approval_mode()
            if on_disk is not None and on_disk != live:
                return SlashResult(
                    kind="notice",
                    text=(
                        f"tool approvals: {live} (this session) — {effect}; "
                        f"config.yml says {on_disk}"
                    ),
                    style="warning" if self._auto_approve else "info",
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
        # A human typed the mode in this session, so a later LOOSENING disk
        # write leaves this gate alone (see :meth:`_on_config_change`). The MODE
        # is recorded, not merely the fact of a choice (review round 2, R6):
        # both directions are worth recording, but only a recorded `ask` is a
        # hardening a file loosening must not revoke.
        self._explicit_approvals_mode = "auto" if wanted_auto else "ask"
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

    def _configured_approval_mode(self) -> str | None:
        """``tool_approval_mode`` as the WATCHER last read it, or ``None``.

        Read off ``existing_watcher().values`` — the last-good snapshot — and
        never from a fresh ``ConfigManager``. Constructing one here would give
        a report path the power to MOVE A MALFORMED FILE ASIDE (the module
        docstring of ``config_watch`` explains why), so asking "what does the
        file say?" could rewrite the user's config as a side effect. ``None``
        when no watcher is running (a headless embed, a test host): the caller
        then reports the live mode alone rather than inventing a comparison.
        """
        try:
            from local_operator.config_watch import existing_watcher
            from local_operator.paths import config_dir

            watcher = existing_watcher(config_dir())
            if watcher is None:
                return None
            mode = str(watcher.values.get("tool_approval_mode", "")).strip().lower()
            return mode if mode in ("ask", "auto") else None
        except Exception:  # noqa: BLE001 — a report must never take down the session
            logger.debug("tool_approval_mode could not be read for the report", exc_info=True)
            return None

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

        async def _run_and_report() -> None:
            """Await the pass so a REFUSAL is not lost.

            A pass that runs narrates itself through the canonical
            `compaction_start`/`compaction_end` events, so this reports only
            what those events never emit: a refusal, and a crash. That is
            exactly the split `app.py::_compact_worker` documents — and it
            matters more here, because the optimistic "compacting context…"
            receipt has already been sent. Without this the user is told a
            pass started and nothing ever contradicts it, which is the one
            outcome that produces no events to render (round 5, U17).
            """
            try:
                outcome = await compact()
            except Exception as exc:  # noqa: BLE001 — a failed pass must not kill the runtime
                logger.debug("manual compaction failed", exc_info=True)
                await self._record_compaction_refusal(f"compaction failed: {exc}")
                return
            if not getattr(outcome, "ran", True):
                # The session's own `detail` is the good copy ("nothing to
                # compact: the whole conversation is ~18 tokens…"); it just
                # never left the runtime before.
                detail = getattr(outcome, "detail", "") or "compaction did not run"
                await self._record_compaction_refusal(str(detail))

        task = self._loop.create_task(_run_and_report())
        # Same retention discipline as the gate tasks above: an un-retained
        # task can be collected mid-flight and the pass would vanish silently.
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return SlashResult(kind="notice", text="compacting context…", style="info")

    async def _record_compaction_refusal(self, detail: str) -> None:
        """Append the row that CORRECTS the optimistic receipt.

        Same shape as the gate-timeout row and for the same reason: a MESSAGE
        entry carrying a `CustomMessage`, so it reaches the model AND every
        viewer that replays the history — a detached session may have no
        terminal attached at the moment the refusal lands, and the user who
        comes back later is the one who most needs to know the pass never ran.
        """
        transcript = getattr(self._session, "transcript", None)
        append = getattr(transcript, "append_message", None)
        if not callable(append):
            return
        try:
            from local_operator.compaction.marker import COMPACTION_REFUSED_TYPE
            from local_operator.harness.types import CustomMessage

            result = append(
                CustomMessage(
                    custom_type=COMPACTION_REFUSED_TYPE,
                    attribution="system",
                    details={"detail": detail},
                )
            )
            if inspect.isawaitable(result):
                await result
            self._notify()
        except Exception:  # noqa: BLE001 — the refusal still stands
            logger.debug("could not record the compaction refusal", exc_info=True)

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
        comms = getattr(self._session, "_subagent_comms", None)
        node = comms.node(job_id) if comms is not None else None
        lookup = getattr(comms, "job", None)
        job = lookup(job_id) if callable(lookup) else None
        if job is None:
            job = self._session.jobs.get(job_id)
        rows = list(getattr(job, "trajectory", None) or []) if job is not None else []
        details: dict[str, Any] = {}
        store = getattr(self._session, "_frontend_state_store", None)
        if (
            offset == 0
            and comms is not None
            and node is not None
            and node.session_id
            and node.session_dir is not None
            and not getattr(node, "live", False)
        ):
            from local_operator.tools.builtin import TODO_STORE, restore_todos

            if node.session_id not in TODO_STORE:
                # A cold owner's roster must not read every child transcript.
                # Hydrate only this selected plan, off-loop, then join the newest
                # canonical snapshot. A live update or resumed identity wins.
                phases = await asyncio.to_thread(_read_child_todo_snapshot, node.session_dir)
                current_node = comms.node(job_id)
                if (
                    phases is not None
                    and current_node is not None
                    and current_node.job_id == node.job_id
                    and current_node.session_id == node.session_id
                ):
                    restore_todos(node.session_id, phases)
        if offset == 0 and store is not None:
            from local_operator.session.frontend_state import job_todos_wire_value

            # Publish the flush before capturing its sequence, with no await in
            # between. A delayed reply can then be joined safely with newer live
            # todo replacements on the follower. Only the first page pays for it.
            store.refresh_jobs(self._session)
            state = store.state
            canonical_id = node.job_id if node is not None else str(getattr(job, "id", job_id))
            row = next((row for row in state.jobs if row.id == canonical_id), None)
            if row is not None:
                details = {
                    "detail_job_id": canonical_id,
                    "detail_session_id": row.session_id,
                    "detail_epoch": state.epoch,
                    "detail_sequence": state.sequence,
                    "todos": job_todos_wire_value(row.todos),
                }
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
            "known": job is not None or node is not None,
            **details,
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

    def _publish_busy_soon(self) -> None:
        """``_publish_busy`` on the next loop iteration; see ``__init__``."""
        try:
            self._loop.call_soon(self._publish_busy)
        except RuntimeError:
            # Loop closed under a disposing session: nothing left to publish.
            return

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

        The authority is :meth:`is_conversationally_active`, NOT ``is_busy()``.
        The record's ``busy`` field is read by exactly one kind of consumer —
        surfaces that render "this session is working" (the sidebar's spinner,
        the picker's marker, `lop sessions`) — and residency is a different
        question that no surface asks. Publishing ``is_busy()`` here meant a
        session holding a background job wore a spinner forever; see
        :meth:`is_conversationally_active` for the measurement.

        This deliberately DOES let the marker and the residency decision
        disagree, which the previous note here forbade. That coupling was the
        defect: a runtime may quite correctly be un-exitable while its
        conversation is finished, and the record has a separate field for each
        fact a reader needs.
        """
        server = self._registrant
        setter = getattr(server, "set_busy", None)
        if not callable(setter):
            return
        try:
            setter(self.is_conversationally_active())
        except Exception:  # noqa: BLE001 — a stale marker is not worth a turn
            logger.debug("could not publish the busy state", exc_info=True)
        self._publish_subagents()

    def _publish_session_started(self) -> None:
        """Flip the record's ``started`` bit once a real turn has run.

        Called from ``Session._run_turn_pipeline`` at the top of every turn;
        the registrant's ``set_record_started`` de-duplicates so only the
        first call publishes. A peer message spooled into an unstarted
        session's mailbox never reaches this — it only becomes ``started``
        when its owner actually runs a turn.
        """
        server = self._registrant
        setter = getattr(server, "set_record_started", None)
        if not callable(setter):
            return
        try:
            setter(True)
        except Exception:  # noqa: BLE001 — a stale flag is not worth a turn
            logger.debug("could not publish the started state", exc_info=True)

    def subagent_counts(self) -> tuple[int | None, int | None]:
        """``(running, queued)`` subagent trajectories, or ``(None, None)``.

        Deliberately NOT folded into :meth:`is_conversationally_active`, which
        drops subagents on purpose (a session whose children are working is not
        itself mid-reply). This is the orthogonal fact ``/info`` needs to answer
        "how many agent trajectories are running across this machine", and it is
        published on the record because a subagent graph is in-process state no
        other session can observe.

        The roster is ONE flat read: ``SubagentComms.nodes()`` already contains
        every nested descendant, so counting is a filter over that list and must
        never have a recursive walk added on top — that would double count every
        node below depth 0. The statuses counted as running mirror the ones
        ``info.collect`` uses, so the record and this session's own tree cannot
        disagree.

        ``(None, None)`` on an unreadable roster rather than ``(0, 0)``: an
        unanswerable probe is not a measurement of zero, and the reader's
        lower-bound caveat depends on being able to tell the two apart.
        """
        session = self._session
        try:
            comms = getattr(session, "subagent_comms", None)
            if comms is None:
                return (None, None)
            nodes = comms.nodes()
        except Exception:  # noqa: BLE001 — an unhealthy session still publishes
            logger.debug("could not read the subagent roster", exc_info=True)
            return (None, None)
        running = queued = 0
        for node in nodes:
            status = str(getattr(node, "status", "") or "")
            if status in RUNNING_SUBAGENT_STATUSES:
                running += 1
            elif status == "queued":
                queued += 1
        return (running, queued)

    def _publish_subagents(self) -> None:
        """Keep the record's subagent counts in step with the roster.

        Driven from ``_publish_busy`` — i.e. from ``_notify`` — because a
        subagent launching or settling IS a session event, so the transition
        publish is sub-second under any real workload while the 15 s heartbeat
        floor bounds a missed publish. ``set_subagents`` de-duplicates, so the
        steady-state cost is one dict walk plus two comparisons per event.
        """
        server = self._registrant
        setter = getattr(server, "set_subagents", None)
        if not callable(setter):
            return
        try:
            running, queued = self.subagent_counts()
            setter(running, queued)
        except Exception:  # noqa: BLE001 — a stale count is not worth a turn
            logger.debug("could not publish the subagent counts", exc_info=True)

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
        # Publish the child roster beside the session state, the way the TUI
        # host does (``mobile/tui_handle.py::_refresh_state``). The folded
        # projection's per-child ``session_id`` is the ONLY route to
        # ``/api/sessions/{sid}/agents/{job}/history``: the event path cannot
        # learn a child's session directory (``SubagentStartEvent`` carries no
        # session id, so those rows are born with ``session_id=None``), and the
        # registry in ``SubagentComms`` is the only place that knows it. Both
        # hosts must therefore agree about the same row, or a runtime-hosted
        # session 404s every child transcript while a TUI-hosted one serves it.
        # The cost is the one the TUI already pays per folded event (one
        # registry walk plus a bounded copy); no child transcript ever leaves
        # with it.
        comms = getattr(self._session, "_subagent_comms", None)
        if comms is not None:
            self._fold.set_subagent_details(comms)

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
    model_selection_override: bool = True,
) -> ServingSessionHandle:
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
    # This is only the BOOT value: ``follow_config`` below keeps the gate on
    # the file for the session's life.
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
        model_selection_override=model_selection_override,
    )
    session = await create_session(
        args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=False,
        cwd=cwd,
    )
    # NOT pinned: this value came from config, so it must keep following
    # config. ``create_session`` has already started the process watcher for
    # this directory (``attach_config_watch``), so ``process_watcher`` returns
    # the running one; the guard mirrors that seam's "boot must not depend on
    # the watcher" degrade.
    handle = ServingSessionHandle(
        session,
        loop,
        cwd=cwd,
        auto_approve=auto_approve,
        approval_pinned=False,
        # Declared so the §6 registration's store-existence check and the
        # registration itself share the config root this session was built
        # from (MINOR-3).
        config_dir=config_directory,
    )
    attach_gate_config_watch(handle, config_directory)
    return handle


def attach_gate_config_watch(handle: ServingSessionHandle, config_directory: Path) -> None:
    """Hang the handle's approval listener on the process watcher, or degrade.

    Shared by the phone/TUI runtime spawn and ``exec --control`` so the two
    cannot drift on how the gate follows config. Degrades to "this gate does
    not follow config" rather than failing the spawn, the same policy as
    ``session_factory.attach_config_watch``.
    """
    try:
        from local_operator.config_watch import process_watcher

        watcher = process_watcher(config_directory)
        watcher.start(asyncio.get_running_loop())
        handle.follow_config(watcher)
    except Exception:  # noqa: BLE001 — the gate keeps its boot value
        logger.warning("config watcher could not be attached to the runtime gate", exc_info=True)
