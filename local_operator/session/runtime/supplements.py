"""The supplement runner: one detached job per eligible turn end (memo §2.1, §2.9).

WHERE IT RUNS. ``ServingSessionHandle._maybe_supplement`` is a sibling of ``_maybe_judge_goal``
on the same runtime event subscription -- installed at boot, with no client attached, so it
sees turns opened by a wake or a resume catch-up as well as typed ones. It is SYNCHRONOUS and
only SCHEDULES (:meth:`SupplementRunner.on_agent_end`); every refusal is a silent return, and
the work happens on a task this class owns.

WHY THE ANSWER ALWAYS GOES FIRST. The subscriber fires from ``_flush_held_end``, i.e. BEFORE
the pipeline's ``_publish_attention_outcome`` and ``on_turn_settled``. That is safe only
because it merely schedules; the job then yields once (``sleep(0)``) and waits on
``Session.wait_turn_settled`` -- the pipeline ``finally`` finishing -- before it touches the
disk or a vendor. So the answer, the attention marker and the banner precede any supplement
work, and a slow or failing vendor can never delay or fail them.

WHAT THE JOB TOUCHES. It never takes ``_turn_lock`` and never reads ``_context``: it works
from the :class:`~local_operator.supplements.trigger.RunProvenance` the session froze, reads
the filesystem with ``stat`` on a worker thread, asks the shared ``ClassificationService``,
and journals through ``Transcript.append_custom`` (serialised on the transcript's own write
lock, the STT-sidecar precedent). A next ``prompt()`` is never delayed by it.

BUSY PREDICATES. The task is held on :attr:`_task`, NOT in the session's ``_background_tasks``,
which ``is_busy`` reads: the reaper, a build refresh or a drain may cut a running job, which is
acceptable because the job is droppable by contract ("updates and shutdown never wait on it").
A cut job leaves at worst a non-terminal row that readers render as ``cancelled - Retry``
(the stale-row rule). :meth:`SupplementRunner.cancel` is called by the handle's ``dispose``.

SUPERSEDE. At most one job per session. A NEW eligible turn cancels the running one; if the
cancelled job's newest journal row is still non-terminal, the next task's
:meth:`SupplementRunner._close_superseded` journals ``cancelled`` / ``error="superseded"``
over it (a row that renders nothing -- no Retry under an answer the user has moved past).
Whether there is a row to close is the JOURNAL's decision, read under the write lock
(round-3 review R3-1); the register only names the anchor it belongs to.

THE GENERATOR TAIL (lane C1b). A graphics "yes" from the decision starts the fork
(``supplements/generator.py``): the job writes a ``decided`` row first (files are known and do
not wait for the model), emits ``supplement_progress`` beats as it runs, stores each accepted
component as a content-addressed blob, and writes ONE terminal row (version + 1) that replaces
the version a reader shows. The four control ops (``supplement_cancel/steer/restart/dismiss``)
are methods here, dispatched by ``session/runtime/server.py`` -- the image-gen receipt rule
(``"already finished"`` for a settled job) and the memo's semantics (§2.7). An op is GATED ON
IDENTITY (round-1 review R1): it can only ever affect the ``(anchor, job)`` it names, and a
steer/restart on an older anchor is queued behind the running job (§2.9, depth 1) instead of
cutting it.

THE WRITE INVARIANT (round-3 review R3-1). Every supplement row for an anchor -- the job's
own decided/terminal appends and every op's settle/supersede/dismiss write -- is written by
:meth:`SupplementRunner._commit_row`, which under ONE lock reads the journal, decides the
version, appends, and publishes the in-memory pointer as a single critical section; the four
ops hold that same lock ACROSS their cut, so no terminal transition can ever decide from a
row staler than the journal. The class this closes (a duplicate version, a ``cancelled`` row
over a delivered ``done``) had appeared three times in different shapes because the
atomicity lived at the call sites; it now lives in that one method. The row and the
live events are SEPARATE on purpose: the row is durable and outside the model's context, the
events are transient and go only to a viewer that negotiated ``supplements-v1``.

EVERYTHING FAILS OPEN. An exception anywhere in the job is logged at DEBUG and swallowed: the
worst outcome of this feature is "no callout".
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Final, Literal, Mapping, Sequence, cast

from local_operator.supplements import generator, policy
from local_operator.supplements.candidates import prefilter
from local_operator.supplements.contract import (
    SUPPLEMENT_CUSTOM_TYPE,
    SupplementDetails,
)
from local_operator.supplements.decision import Decision
from local_operator.supplements.decision import decide as decide_supplement
from local_operator.supplements.evidence import Dataset
from local_operator.supplements.persistence import (
    append_row,
    build_details,
    new_job_id,
    next_version,
    superseded,
)
from local_operator.supplements.prompt import MAX_INSTRUCTION_CHARS
from local_operator.supplements.trigger import final_answer, refusal
from local_operator.supplements.validate import Component

logger = logging.getLogger(__name__)

#: ``Session.classification_seam`` returns the shared ``ClassificationService`` (or ``None``
#: when the layer is off); resolved PER CALL, the monitor gate's convention, so a seam a test
#: swaps in after construction is the one that answers.
SeamResolver = Callable[[], Any]

#: Whether a generator exists to hand a graphics "yes" to. TRUE since lane C1b: the fork in
#: ``supplements/generator.py`` runs the errand, so a graphics question the decision answers
#: "yes" now has something to act on. ``supplements.graphics`` is still OFF by default (memo
#: §6: the flip ships in the window that carries the first renderer), so an operator who has
#: not opted in sees no change at all -- and one who has gets the ROWS and the spend, not a
#: picture, until a surface lane lands.
GENERATOR_AVAILABLE: Final = True

#: How many settled jobs keep their inputs for a later ``restart``/``steer`` (§2.7: both are
#: allowed from ``failed``/``cancelled``/``done``). Bounded because the inputs hold the
#: evidence datasets: a session that ran fifty jobs must not pin fifty of them in memory.
_KEPT_INPUTS: Final = 3

#: The journaled states a version can END in: once one of these is the job's newest
#: JOURNAL row, the job is settled and no control op may cut or re-settle that version
#: (round-2 review R2-3; the non-terminal pair ``decided``/``queued`` is what a stalled reader
#: paints as "cancelled - Retry"). Round-3 review R3-1 made the invariant structural: every
#: decision path reads the journal inside the write lock, so a terminal append still in
#: flight is seen as the settled row it already is -- not as the stale ``decided`` pointer it
#: has not yet superseded.
_TERMINAL_STATES: Final[frozenset[str]] = frozenset({"done", "failed", "cancelled", "skipped"})


def _is_terminal(details: Mapping[str, Any] | None) -> bool:
    """Whether a row is a settled state -- applied to JOURNAL rows in every decision path."""
    return isinstance(details, Mapping) and str(details.get("state", "")) in _TERMINAL_STATES


class _OpsLock:
    """The supplement write lock: task-scoped re-entrant (round-3 review R3-1).

    THE INVARIANT IT CARRIES. Every writer of a supplement row -- the four control ops and
    the job's own decided/terminal appends -- must read the anchor's newest journal row,
    decide the version from it, append, and publish the in-memory pointer as ONE critical
    section; the ops additionally hold it ACROSS their cut (cancel -> await the cut task ->
    settle). The job's terminal append therefore has to take the same lock the ops take, and
    an op's own settle nests inside its hold. A plain ``asyncio.Lock`` deadlocks on that
    nest, and splitting the settle onto a second lock would put half the invariant back
    outside the section that decides it. Re-entrancy is TASK-scoped: only the task already
    holding the lock re-enters; every other task still serialises on the same lock.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth = 0

    async def __aenter__(self) -> "_OpsLock":
        task = asyncio.current_task()
        if task is not None and task is self._owner:
            self._depth += 1
            return self
        await self._lock.acquire()
        self._owner = task
        self._depth = 1
        return self

    async def __aexit__(self, *exc: Any) -> Literal[False]:
        # ``Literal[False]``, not ``bool``: the lock never suppresses an exception, and a
        # plain ``bool`` makes a type checker widen every ``return`` inside an ``async with``
        # body into a fall-off-the-end path (measured: two reportReturnType false positives).
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()
        return False


@dataclass
class _JobInputs:
    """What one job needs to run, and to be re-run by a steer or a restart.

    Held in memory only: nothing here is journaled (the evidence datasets can hold a turn's
    tool output, and the row carries the DIGEST of what was rendered instead -- §2.4's
    "components are blobs" rule). A restart after a process restart therefore has no inputs
    and answers the neutral receipt; a surface that wants Retry across restarts needs the
    evidence in the row, which the memo deliberately does not do.
    """

    anchor: str
    job: str
    decision: Decision
    datasets: tuple[Dataset, ...]
    user_text: str
    answer_text: str
    instruction: str = ""
    version: int = 1
    #: The job's newest row, as the write path published it after its append landed (a plain
    #: mapping of the contract's shapes). ``None`` until the first write. DECISIONS NEVER
    #: READ THIS (round-3 review R3-1): it lags a row that is already durable, so every
    #: settle/cut/dismiss decides from the journal; the ``cancelling`` beats read it as their
    #: payload, and that is the whole of its contract.
    details: dict[str, Any] | None = None


class SupplementRunner:
    """Owns the one supplement task of a runtime handle. See the module docstring."""

    def __init__(
        self,
        session: Any,
        *,
        cwd: str,
        config_dir: Path | None = None,
        seam: SeamResolver | None = None,
    ) -> None:
        self._session = session
        self._cwd = cwd
        self._config_dir = config_dir
        self._seam = seam
        #: THE SETTINGS SNAPSHOT (memo §2.12), read ONCE here, at build, and never
        #: re-read: the section's scope is NEW_SESSIONS because the runner is built per
        #: runtime handle, so an edit lands on the NEXT session -- exactly what the
        #: settings page paints. A per-job re-read (this class's first shape) made an
        #: edit land on the next eligible turn of an EXISTING session: LIVE behaviour
        #: under a NEW_SESSIONS label, with the comments and §2.12 all describing the
        #: snapshot (agent review round 1, R3). The read is synchronous, bounded, and
        #: never raises -- the ``read_monitor_settings`` posture (``Session.__init__``
        #: reads ``values.monitor`` the same way, on the same kind of build path).
        self._settings = self._read_settings()
        self._task: asyncio.Task[None] | None = None
        self._open_row: Mapping[str, Any] | None = None
        #: Jobs this runtime is running right now: the runtime's half of the reader rule
        #: (``contract.reader_disposition(job_live=...)``).
        self._live_jobs: set[str] = set()
        #: The inputs of the newest few jobs, newest last, for ``steer``/``restart`` (§2.7).
        self._inputs: dict[str, _JobInputs] = {}
        #: The ``(anchor, job)`` the task on :attr:`_task` is running, or ``None`` while that
        #: task is the decision half (which has no addressable job yet). Every control op
        #: gates its cut on this (round-1 review R1): an op may only ever affect its own
        #: job -- a cancel for a settled job must not kill a newer one, nor rewrite the
        #: settled row.
        self._running: tuple[str, str] | None = None
        #: The depth-1 queue of memo §2.9: a steer/restart on an OLDER anchor arriving while
        #: a newer job runs waits here (a newer request replaces the queued one) and starts
        #: when the job it waits behind settles. It survives a later supersede on purpose --
        #: the request was explicit, and §2.9 names exactly one eviction rule.
        self._pending: tuple[_JobInputs, str] | None = None
        #: Serialises the four control ops (round-2 review R2-4) AND every writer of a
        #: supplement row (round-3 review R3-1; the invariant is stated on ``_OpsLock`` and
        #: enforced in ``_commit_row``): read-journal -> decide -> append -> publish is one
        #: critical section, so an op can never settle from a row staler than the journal --
        #: a terminal append in flight is atomic with the decision that must see it.
        self._ops_lock = _OpsLock()

    # -- the synchronous trigger ----------------------------------------------------------

    def on_agent_end(self, event: Any, *, goal_loop_running: bool) -> str:
        """Schedule a job for an eligible end. Returns ``""`` when scheduled, else the reason.

        Synchronous and total: it never raises and never awaits. The reason string exists so
        the trigger matrix can pin each refusal rule by name (a guard proven only by the
        all-clear path could be deleted unnoticed).
        """
        try:
            if not policy.enabled():
                return "kill-switch"
            reason = refusal(
                getattr(self._session, "last_run_provenance", None),
                error=bool(getattr(event, "error", None)),
                aborted=bool(getattr(event, "aborted", False)),
                cut_off_cause=str(getattr(event, "cut_off_cause", "") or ""),
                goal_loop_running=goal_loop_running,
            )
            if reason:
                return reason
            provenance = self._session.last_run_provenance
            previous = self._task
            if previous is not None and not previous.done():
                # Prefer the newest answer. The cancelled job's own cleanup journals the
                # superseded row if it had written a non-terminal one.
                previous.cancel()
            # The decision half has no addressable (anchor, job) yet: no op may cut it, and a
            # queued steer waits through it (``_run``'s tail calls the drain).
            self._running = None
            self._task = asyncio.get_running_loop().create_task(self._run(provenance, previous))
            return ""
        except Exception:  # noqa: BLE001 — the event path never fails on this feature
            logger.debug("supplement trigger failed", exc_info=True)
            return "error"

    # -- registry / lifecycle -------------------------------------------------------------

    def is_live(self, job: str) -> bool:
        return job in self._live_jobs

    def _running_under(self, anchor: str, job: str | None = None) -> bool:
        """Whether the live task runs a job of ``anchor`` -- exactly ``job`` when named.

        THE CUT GATE (round-1 review R1): every control op that can cut a task checks this
        first, so an op can only ever affect its own job. ``job=None`` is the dismiss shape,
        which names an anchor and may cut whatever attempt of that anchor is running.
        """
        task = self._task
        if task is None or task.done() or self._running is None:
            return False
        if self._running[0] != anchor:
            return False
        return job is None or self._running[1] == job

    def _graphics_running(self) -> bool:
        """Whether a live task runs an addressable fork job (memo §2.9's "newer job")."""
        task = self._task
        return task is not None and not task.done() and self._running is not None

    def set_open_row(self, details: Mapping[str, Any] | None) -> None:
        """Register (or clear) the job's latest NON-terminal row, for supersede (C1b seam)."""
        self._open_row = details

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def cancel(self) -> None:
        """Cancel and reap the task (dispose). Never raises."""
        task = self._task
        self._task = None
        self._running = None
        self._pending = None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    # -- the job --------------------------------------------------------------------------

    async def _run(self, provenance: Any, previous: asyncio.Task[None] | None) -> None:
        try:
            # First statement: yield to the pipeline ``finally`` (belt and braces beside the
            # settled wait below).
            await asyncio.sleep(0)
            if previous is not None:
                await asyncio.gather(previous, return_exceptions=True)
                await self._close_superseded()
            await self._session.wait_turn_settled(provenance.settled_mark)
            settings = self._settings
            if not settings.active:
                return
            await asyncio.wait_for(self._job(provenance, settings), timeout=settings.timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — fail open: the worst case is no callout
            logger.debug("supplement job failed", exc_info=True)
        # A queued steer/restart (memo §2.9) waits behind the NEWEST work; when this decision
        # tail started no fork, the newer job just settled -- drain. When it did start one,
        # the fork task IS the newer job and the drain belongs to its own end. A cancelled
        # ``_run`` (re-raised above) never drains: it was superseded, and an explicit queued
        # request outlives a turn it was never about.
        if not self._graphics_running():
            self._drain_pending()

    async def _close_superseded(self) -> None:
        """Journal the ``cancelled``/``superseded`` row for a previous task's leftover.

        Round-3 review R3-1: whether there IS a row to close, and which row, is decided from
        the JOURNAL under the write lock -- never from the registered open row's state. A
        terminal write that completed while its task was cancelled inside the append left a
        stale register, and the old shape wrote a spurious ``superseded`` row (version + 1)
        over a delivered one. The register names the anchor to re-read; a terminal or absent
        newest row means there is nothing left to close.
        """
        row, self._open_row = self._open_row, None
        if row is None:
            return
        anchor = str(row.get("anchor", ""))
        if not anchor:
            return

        def build_supersede(base: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
            if base is None or _is_terminal(base):
                return None
            return superseded(base)

        try:
            await self._commit_row(anchor=anchor, build=build_supersede)
        except Exception:  # noqa: BLE001
            logger.debug("could not journal the superseded supplement row", exc_info=True)

    def _read_settings(self) -> policy.SupplementSettings:
        """Read the ``supplements`` section ONCE, at build (``__init__`` keeps the snapshot).

        Never raises, and the failure envelope is why: the old per-job call sat inside the
        job's fail-open ``except``, while THIS one runs in the handle's constructor -- an
        unreadable config must not fail every session started from it. Falls back to the
        built-in defaults with a warning, the ``read_monitor_settings`` posture; the
        defaults keep the feature ON, matching the kill switch's rule that a typo must not
        silently unbuild it.
        """
        try:
            from local_operator.config import ConfigManager
            from local_operator.paths import config_dir

            manager = ConfigManager(self._config_dir or config_dir())
            return policy.SupplementSettings.from_values(
                getattr(manager.get_config(), "values", None)
            )
        except Exception:  # noqa: BLE001 — a bad config must not fail session startup
            logger.warning("values.supplements could not be read; using the built-in defaults")
            return policy.SupplementSettings()

    async def _job(self, provenance: Any, settings: policy.SupplementSettings) -> None:
        transcript = self._session.transcript
        answer = final_answer(provenance.items, persisted=transcript.has_entry)
        if answer is None:
            return
        want_files = settings.files
        want_graphics = settings.graphics and GENERATOR_AVAILABLE
        started = time.perf_counter()
        pre = await asyncio.to_thread(
            prefilter,
            provenance.items,
            answer.text,
            cwd=str(getattr(self._session, "_cwd", "") or self._cwd),
            deny_prefixes=settings.deny_prefixes,
            want_files=want_files,
            want_graphics=want_graphics,
        )
        logger.debug(
            "supplements: prefilter candidates=%d structured=%s skipped=%s rejected=%s "
            "elapsed_ms=%.2f",
            len(pre.candidates),
            pre.evidence.structured,
            pre.skipped,
            pre.rejected,
            pre.elapsed_ms,
        )
        if pre.skipped:
            return
        service = self._seam() if self._seam is not None else None
        decision = await decide_supplement(
            service,
            user_text=provenance.user_text,
            answer_text=answer.text,
            candidates=pre.candidates,
            evidence=pre.evidence,
            want_files=want_files,
            want_graphics=want_graphics,
            max_featured=settings.max_featured,
        )
        logger.debug(
            "supplements: decision vendor=%s featured=%d more=%d graphics=%s total_ms=%.1f",
            decision.vendor,
            len(decision.featured),
            len(decision.more),
            decision.graphics,
            (time.perf_counter() - started) * 1000.0,
        )
        if decision.graphics and want_graphics:
            # Files AND graphics: the generator half runs on its own task so a steer can cut
            # it without touching the decision's work, and so a slow model can never delay
            # the turn that is already over (§2.9).
            self._render(
                answer.id,
                decision,
                pre.evidence.datasets,
                provenance.user_text,
                answer.text,
                settings,
            )
            return
        await self._write(answer.id, decision)

    async def _write(self, anchor: str, decision: Decision) -> None:
        """Journal the decision. A decision with nothing to show writes NOTHING.

        An empty ``done`` row would render as nothing anyway (``reader_disposition``) while
        costing a journal line per turn the vendor declined; the spam-rate measurement reads
        the DEBUG line and the golden set, not the journal.
        """
        if decision.empty and not decision.more:
            return
        job = new_job_id()
        self._live_jobs.add(job)
        try:
            # Files-only: no generator is coming, so the first row is TERMINAL. A ``decided``
            # row would be read cold as "cancelled - Retry" under an answer with nothing to
            # retry (contract.reader_disposition). Through the one write path like every row
            # (round-3 review R3-1), so the version is read from the journal instead of being
            # assumed to be 1 -- a re-used anchor must not collide with rows it already has.
            def build_done(base: Mapping[str, Any] | None) -> Mapping[str, Any]:
                version = int(base.get("version", 0) or 0) + 1 if base is not None else 1
                return dict(
                    build_details(
                        anchor=anchor, job=job, version=version, state="done", decision=decision
                    )
                )

            await self._commit_row(anchor=anchor, build=build_done)
        finally:
            self._live_jobs.discard(job)

    # -- the generator tail (lane C1b) ----------------------------------------------------

    def _render(
        self,
        anchor: str,
        decision: Decision,
        datasets: tuple[Dataset, ...],
        user_text: str,
        answer_text: str,
        settings: policy.SupplementSettings,
    ) -> None:
        """Schedule the fork for a graphics "yes" (memo §2.5).

        A separate task from the decision's, because everything about this half is slower and
        may be cancelled on its own (a steer cancels the IN-FLIGHT attempt and starts
        version + 1); the decision's job is already finished by the time this is reachable.
        """
        inputs = _JobInputs(
            anchor=anchor,
            job=new_job_id(),
            decision=decision,
            datasets=tuple(datasets),
            user_text=user_text,
            answer_text=answer_text,
        )
        self._remember(inputs)
        previous = self._task
        if previous is not None and not previous.done():
            previous.cancel()
        self._running = (inputs.anchor, inputs.job)
        self._task = asyncio.get_running_loop().create_task(
            self._run_graphics(inputs, settings, previous)
        )

    def _remember(self, inputs: _JobInputs) -> None:
        """Keep the newest few jobs' inputs for a later steer/restart (bounded, §2.7)."""
        self._inputs[inputs.anchor] = inputs
        while len(self._inputs) > _KEPT_INPUTS:
            oldest = next(iter(self._inputs))
            if oldest == inputs.anchor:
                break
            self._inputs.pop(oldest, None)

    async def _run_graphics(
        self,
        inputs: _JobInputs,
        settings: policy.SupplementSettings,
        previous: asyncio.Task[None] | None,
    ) -> None:
        """One fork job, from schedule to terminal row -- and the §2.9 drain behind it."""
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        # What has passed validation so far, refilled by ``generator.generate`` as it goes:
        # the ``wait_for`` below cancels the attempt on a hard bound, and this sink is then
        # the only surviving copy of the blocks a cut must keep (memo §2.5, round-1 R4).
        sink: list[Component] = []
        started = time.perf_counter()
        try:
            await asyncio.wait_for(
                self._attempt(inputs, settings, sink, started), timeout=settings.timeout_s
            )
        except asyncio.CancelledError:
            # A cancel is not a failure: the row for it is written by whoever cancelled
            # (``cancel_job``/``steer_job``/``restart_job``), so a dispose or a supersede
            # leaves the ``decided`` row in place and the stale-row rule renders it as
            # "cancelled - Retry" -- the memo's own rule for a cut job (§2.9).
            raise
        except asyncio.TimeoutError:
            await self._finish_bound(inputs, sink, settings)
        except Exception:  # noqa: BLE001 -- fail open: the worst case is no figure
            logger.debug("supplement generator job failed", exc_info=True)
            await self._fail(inputs, "generator:unexpected")
        # The job settled: whatever waited behind it (memo §2.9, depth 1) may start now.
        self._drain_pending()

    async def _attempt(
        self,
        inputs: _JobInputs,
        settings: policy.SupplementSettings,
        sink: list[Component],
        started: float,
    ) -> None:
        """The whole fork run: decided row, generator turns, blobs, terminal row.

        ``sink`` and ``started`` are the run's shared edge with the runner: the sink is
        refilled by the generator with the accepted-so-far blocks (a bound firing inside the
        runner's ``wait_for`` must not lose them -- round-1 review R4), and ``started`` is the
        same clock the ``wait_for`` counts from, so the in-loop between-turns check and the
        outer hard guard agree on the deadline.
        """

        # One version sequence per ANCHOR (QA round 1, Q-2; hardened by round-3 review
        # R3-1): ``_start`` mints for steer/restart, but the render path starts at version 1
        # in memory -- a second job ever started for one anchor would write rows at versions
        # the first job's rows hold, and the runtime reader and ``contract.newest_per_anchor``
        # (different tie-breaks) would then disagree on "newest". The version is read from
        # the journal INSIDE the write's own critical section now, so nothing can land
        # between the read and the append the way a just-published terminal row used to.
        def build_decided(base: Mapping[str, Any] | None) -> Mapping[str, Any]:
            version = int(base.get("version", 0) or 0) + 1 if base is not None else inputs.version
            inputs.version = version
            return dict(
                build_details(
                    anchor=inputs.anchor,
                    job=inputs.job,
                    version=version,
                    state="decided",
                    decision=inputs.decision,
                )
            )

        try:
            # The runtime's half of the reader rule: the job is LIVE from before its first
            # await until this block's ``finally``. C1a left this registry as the C1b seam;
            # round-1 review R2 found the graphics path never added to it.
            self._live_jobs.add(inputs.job)
            details = await self._commit_row(
                anchor=inputs.anchor, build=build_decided, inputs=inputs
            )
            if details is None:  # a declined build: nothing was written, nothing to run
                return
            await self._progress(details, "decided")
            resolved = await asyncio.to_thread(self._resolve_model, settings)
            if resolved is None:
                await self._fail(inputs, "no-model")
                return
            await self._progress(details, "running", stage="generating")
            outcome = await generator.generate(
                complete=self._complete,
                model=resolved,
                datasets=inputs.datasets,
                user_text=inputs.user_text,
                answer_text=inputs.answer_text,
                instruction=inputs.instruction,
                max_turns=settings.max_turns,
                max_output_tokens=settings.max_output_tokens,
                max_cost_usd=settings.max_cost_usd,
                timeout_s=settings.timeout_s,
                price=lambda usage: self._price(usage, resolved),
                on_progress=lambda stage, elapsed: self._progress(
                    details, "running", stage=stage, elapsed_s=elapsed
                ),
                started=started,
                accepted_sink=sink,
            )
            components = await self._store(outcome.components)
            error = outcome.error if not components else ""
            state = "failed" if error else "done"

            def build_final(base: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
                if base is None:
                    # Nothing to replace: the decided write never landed (a cut before its
                    # append), and a terminal row without a version to follow would invent
                    # one. Leaving the absence is the memo's cut-job shape (§2.9).
                    return None
                final: dict[str, Any] = dict(next_version(base, state=state, error=error))
                final["components"] = components
                final["model"] = outcome.model
                final["turns"] = outcome.turns
                final["tokens_in"] = outcome.tokens_in
                final["tokens_out"] = outcome.tokens_out
                final["cost_usd"] = round(outcome.cost_usd, 6)
                if outcome.instruction:
                    final["instruction"] = outcome.instruction[:MAX_INSTRUCTION_CHARS]
                if outcome.detail:
                    # The generator's own notes REACH THE JOURNAL even when blocks survived
                    # and ``error`` is cleared: a bound that fired after acceptance is
                    # reported here, never as ``error`` on a ``done`` row (memo §2.5 as
                    # amended; round-1 R7).
                    final["detail"] = list(outcome.detail)
                return final

            final = await self._commit_row(
                anchor=inputs.anchor, build=build_final, job=inputs.job, inputs=inputs
            )
            if final is None:
                return
            await self._progress(final, state)
        finally:
            self._live_jobs.discard(inputs.job)

    async def _fail(self, inputs: _JobInputs, error: str) -> None:
        """Terminal ``failed`` row for a job that never produced anything (bound or crash).

        The base comes from the JOURNAL like every terminal write (round-3 review R3-1):
        the job's decided row is already durable whenever this is reachable, so the old
        in-memory rebuild of a decided row -- and the version number it invented when the
        pointer was empty -- is gone with the stale-pointer class it belonged to.
        """

        def build_failed(base: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
            return None if base is None else next_version(base, state="failed", error=error)

        try:
            final = await self._commit_row(
                anchor=inputs.anchor, build=build_failed, job=inputs.job, inputs=inputs
            )
            if final is None:
                return
            await self._progress(final, "failed")
        except Exception:  # noqa: BLE001 -- a failed write is a missing callout, not a turn error
            logger.debug("supplement failure row could not be written", exc_info=True)

    async def _finish_bound(
        self,
        inputs: _JobInputs,
        sink: list[Component],
        settings: policy.SupplementSettings,
    ) -> None:
        """A hard bound fired with blocks already accepted: KEEP them (memo §2.5, R4).

        Called from :meth:`_run_graphics`'s ``TimeoutError`` arm -- the ``wait_for`` already
        cancelled the attempt, so the sink holds the only surviving copy of what passed.
        With nothing in the sink the row is ``failed`` / ``bound:time`` as before; with blocks
        the row is ``done`` (the frozen reader paints the settled block only for ``done``)
        and the bound is recorded in its ``detail`` (round-1 review R4/R7).
        """
        components = await self._store(sink)
        if not components:
            await self._fail(inputs, f"bound:{generator.BOUND_TIME}")
            return

        def build_bound(base: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
            if base is None:
                return None
            final: dict[str, Any] = dict(next_version(base, state="done"))
            final["components"] = components
            detail = [str(line) for line in (final.get("detail") or [])]
            detail.append(
                f"bound:{generator.BOUND_TIME} wall clock {settings.timeout_s}s reached mid-job; "
                f"{len(components)} block(s) kept"
            )
            final["detail"] = detail
            return final

        final = await self._commit_row(
            anchor=inputs.anchor, build=build_bound, job=inputs.job, inputs=inputs
        )
        if final is None:
            return
        await self._progress(final, "done")

    def _resolve_model(self, settings: policy.SupplementSettings) -> generator.DesignModel:
        """Gather the live inputs ``generator.resolve_design_model`` needs (memo §2.5).

        Runs on a worker thread (the auth store is SQLite), and every failure degrades to the
        session's own model rather than to no job: a design figure on the conversation's
        model is worse than a good one, and better than nothing.
        """
        from local_operator.model.registry import static_models
        from local_operator.providers.model_access import usable_providers_here

        session_spec = getattr(self._session, "effective_model", None) or getattr(
            self._session, "model", None
        )
        usable: set[str] | None = None
        try:
            usable = usable_providers_here(config_dir=self._config_dir)
        except Exception:  # noqa: BLE001 -- unknowable is the documented ``None``
            logger.debug("supplements: usable providers could not be read", exc_info=True)
        tier: Any = None
        resolve_tier = getattr(self._session, "_resolve_subagent_model", None)
        if callable(resolve_tier):
            try:
                tier = resolve_tier("task", "hi")
            except Exception:  # noqa: BLE001 -- a bad tier is not a failed job
                logger.debug("supplements: hi tier could not be resolved", exc_info=True)
        return generator.resolve_design_model(
            configured=settings.model,
            session_spec=session_spec,
            usable=usable,
            static_models=static_models,
            tier_spec=tier,
        )

    async def _complete(self, request: Any) -> tuple[str, Any]:
        """One fork turn: text plus usage, drained from the session's own stream function.

        The request is built by the generator (isolated, tools-free, ``supplement_render``),
        so this is deliberately the same three lines ``Session._drain_errand`` uses: one
        shape for every errand, and the usage event is what the cost cap is checked against.
        """
        from local_operator.harness.types import StreamTextDelta, StreamUsageEvent

        parts: list[str] = []
        usage: Any = None
        async for event in self._session._stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
            elif isinstance(event, StreamUsageEvent):
                usage = event.usage
        return "".join(parts), usage

    def _price(self, usage: Any, resolved: Any) -> float:
        """What one fork turn cost, through THE money computation (``cost_for_usage``).

        The same function the status band and the ledger writer use, so the cap cannot
        disagree with ``/usage`` about what a job has spent. Never raises: a pricing failure
        must not end a job (the ledger still prices it on its own background thread).
        """
        if usage is None:
            return 0.0
        try:
            from local_operator.model.configure import (
                cost_for_usage,
                resolve_model_info,
            )

            spec = resolved.spec
            info = resolve_model_info(spec.provider, spec.model_id)
            return float(cost_for_usage(spec.provider, info, usage) or 0.0)
        except Exception:  # noqa: BLE001 -- an unpriceable turn is not a failed job
            logger.debug("supplement turn could not be priced", exc_info=True)
            return 0.0

    async def _store(self, components: Sequence[Component]) -> list[dict[str, Any]]:
        """Store each accepted component as a blob and build the row's ``components[]``.

        Content-addressed (memo §2.4): the row carries a 32-hex digest, so a large document
        never enters the transcript payload and sync's existing ``"attachment":"<digest>"``
        byte scan carries it with a moved conversation. A component the store refuses is
        DROPPED, not rendered: ``put_bytes`` returning ``None`` means the bytes are not on
        disk, and a row pointing at a digest that resolves to nothing would paint a
        permanently broken frame.
        """
        if not components:
            return []
        from local_operator.session.attachments import AttachmentStore

        store = AttachmentStore()
        entries: list[dict[str, Any]] = []
        for component in components:
            ref = await asyncio.to_thread(
                store.put_bytes, component.blob.encode("utf-8"), "text/html"
            )
            if ref is None:
                logger.debug("supplement component could not be stored")
                continue
            entries.append(
                {
                    "attachment": ref.digest,
                    "title": component.title,
                    "source": component.source,
                    "mime": "text/html",
                    "height_hint": component.height_hint,
                }
            )
        return entries

    async def _progress(
        self,
        details: Mapping[str, Any],
        state: str,
        *,
        stage: str = "",
        elapsed_s: float = 0.0,
    ) -> None:
        """Emit one ``supplement_progress`` beat (memo §2.7). Never fails the job.

        The event is the LIVE half: ``running``/``cancelling`` are never journaled, and only
        ``decided``/``done`` carry the files/components (the row's own shapes, so a surface
        can swap the indicator for the settled block without a second read).
        """
        try:
            from local_operator.harness.types import SupplementProgressEvent

            emit = getattr(self._session, "_emit", None)
            if not callable(emit):
                return
            await_emit = cast(Callable[[Any], Awaitable[None]], emit)
            event = SupplementProgressEvent(
                anchor=str(details.get("anchor", "")),
                job=str(details.get("job", "")),
                version=int(details.get("version", 1)),
                state=cast("Any", state),
                stage=stage,
                elapsed_s=round(float(elapsed_s), 3),
                files=(
                    [dict(item) for item in (details.get("files") or [])]
                    if state in ("decided", "done")
                    else []
                ),
                components=(
                    [dict(item) for item in (details.get("components") or [])]
                    if state == "done"
                    else []
                ),
                error=str(details.get("error", "") or ""),
            )
            await await_emit(event)
        except Exception:  # noqa: BLE001 -- a missing beat is not a failed job
            logger.debug("supplement progress could not be emitted", exc_info=True)

    # -- the control ops (memo §2.7) -------------------------------------------------------

    def recall(self, anchor: str, job: str) -> _JobInputs | None:
        """The inputs for ``(anchor, job)``, live or recently settled. ``None`` = unknown."""
        inputs = self._inputs.get(anchor)
        if inputs is None or inputs.job != job:
            return None
        return inputs

    # -- the one write path (round-3 review R3-1) -------------------------------------------

    async def _commit_row(
        self,
        *,
        anchor: str,
        build: Callable[[Mapping[str, Any] | None], Mapping[str, Any] | None],
        job: str | None = None,
        inputs: _JobInputs | None = None,
    ) -> dict[str, Any] | None:
        """THE supplement write path: read the journal, decide, append, publish (R3-1).

        Under the one ``_OpsLock`` -- the same lock the control ops hold across a cut -- this
        reads the anchor's newest journal row (``job`` narrows the read to one job's rows),
        lets ``build`` build the row from it (returning ``None`` writes nothing), appends,
        and publishes ``inputs.details``. The read and the append being ONE critical section
        IS the invariant: no other writer can land between a decision and its write, so two
        rows for an anchor can never be built from the same base version, and a terminal
        transition is seen for what it already is by every later decider. That class had
        appeared three times before this shape existed (round-2 R2-3, QA Q-1, round-3 R3-1)
        because the atomicity used to live at the call sites; it now lives here, and a new
        call site inherits it.

        Called both inside an op's hold (its own settle / supersede / dismiss write, via the
        re-entrant lock) and from the job's own task (decided / terminal appends); the helper
        owns the lock either way, so no caller can forget it.
        """
        async with self._ops_lock:
            base = await asyncio.to_thread(self._newest_row, anchor, job=job)
            row = build(base)
            if row is None:
                return None
            # The supersede register moves with the write, inside the section: a NON-terminal
            # row is the leftover a future turn may need to close, a terminal one leaves
            # nothing open. The journal still decides whether a close is owed (R3-1); the
            # register only names the anchor for `_close_superseded`.
            self.set_open_row(None if _is_terminal(row) else row)
            await append_row(self._session.transcript, row)
            if inputs is not None:
                inputs.details = dict(row)
            return dict(row)

    async def _settle(self, inputs: _JobInputs, *, state: str, error: str = "") -> None:
        """Write a terminal row for a job an op just cut, and emit its beat.

        The row is built from the JOURNAL inside the caller's own hold of the write lock
        (round-3 review R3-1): the row a terminal append just made durable is the base,
        never the stale in-memory pointer that used to rebuild the version that row had
        taken (the cancel-over-delivered-``done`` class). A job that never wrote anything
        has no newest row and settles nothing -- the reader paints its absence with the
        stale-row rule (memo §2.5).
        """

        def build_settle(base: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
            return None if base is None else next_version(base, state=state, error=error)

        final = await self._commit_row(
            anchor=inputs.anchor, build=build_settle, job=inputs.job, inputs=inputs
        )
        if final is None:
            return
        await self._progress(final, state)

    async def cancel_job(self, anchor: str, job: str) -> str:
        """``supplement_cancel``: cancel the running attempt, write ``cancelled``.

        Idempotent, and the receipt for a job that is not running is image-gen's neutral
        "already finished" -- a surface that lost a race with the job's own end must not be
        told something happened. The running task is cut ONLY when it runs this very
        ``(anchor, job)`` (round-1 review R1): a cancel for a settled job arriving while a
        NEWER job runs must leave that newer job alone and must not rewrite the settled row.
        Both re-checks read the JOURNAL, not the pointer (round-2 review R2-3/R2-4; round-3
        review R3-1): the job can settle while its terminal append is still in flight --
        the row is already durable and published while ``inputs.details`` still reads
        ``decided`` -- or inside the ``cancelling`` beat, and the bookkeeping can move
        across the reap; either way the cut is no longer this op's to make, and the surface
        gets the neutral receipt without a newer task being touched.
        """
        async with self._ops_lock:
            inputs = self.recall(anchor, job)
            task = self._task
            if inputs is None or task is None or task.done() or self._running != (anchor, job):
                return "already finished"
            await self._progress(inputs.details or {}, "cancelling")
            newest = await asyncio.to_thread(self._newest_row, anchor, job=job)
            if _is_terminal(newest) or self._task is not task:
                return "already finished"
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._task is not task:
                # A newer flow owns the bookkeeping; its own supersede path journals what
                # this cut left non-terminal, so clearing or settling here would clobber it.
                return "already finished"
            self._task = None
            self._running = None
            await self._settle(inputs, state="cancelled")
            # The cut settled the running job: whatever waited behind it (memo §2.9) starts now.
            self._drain_pending()
            return "cancelled"

    async def steer_job(self, anchor: str, job: str, text: str) -> str:
        """``supplement_steer``: cancel any in-flight attempt, start version + 1.

        NOT a turn (memo §2.7): it never touches the session's turn lock, steering queue or
        model context -- it starts a new GENERATOR version with the user's instruction. When
        the named job is NOT the running one, the request is queued behind the running job
        instead of cutting it (memo §2.9; round-1 review R1).
        """
        inputs = self.recall(anchor, job)
        if inputs is None:
            return "already finished"
        instruction = " ".join(str(text or "").split())[:MAX_INSTRUCTION_CHARS]
        async with self._ops_lock:
            await self._cut_and_restart(inputs, instruction=instruction)
        return "steering"

    async def restart_job(self, anchor: str, job: str) -> str:
        """``supplement_restart``: version + 1 with the previous instruction, or none.

        Shares the steer's queue rule: a restart of a job that is not the running one waits
        behind the running job (memo §2.9, depth 1).
        """
        inputs = self.recall(anchor, job)
        if inputs is None:
            return "already finished"
        async with self._ops_lock:
            await self._cut_and_restart(inputs, instruction=inputs.instruction)
        return "restarting"

    async def _cut_and_restart(self, inputs: _JobInputs, *, instruction: str) -> None:
        """Cut the in-flight attempt (if any) and run the next version in its place.

        The ``cancelled`` row is written only when there WAS something to cut. A restart from
        ``done``/``failed`` (the Retry affordance) has no attempt in flight, and a
        ``cancelled`` row in that sequence would paint "Highlights cancelled - Retry" for a
        moment and put a version in the audit trail that describes nothing.

        A request naming a job that is NOT the running one does not cut it: it is queued
        behind the running job, depth 1 (memo §2.9; round-1 review R1) -- a newer request
        replaces the queued one, and the queue drains when the job it waits behind settles.
        Both re-checks read the JOURNAL, not the pointer (round-2 review R2-3/R2-4; round-3
        review R3-1): a job that settled while its terminal append was still in flight, or
        inside the ``cancelling`` beat, is not cut -- its terminal row already exists and
        the next version starts straight away -- and bookkeeping that moved sends the
        request to the queue, exactly as the sequential shape would.
        """
        task = self._task
        if task is not None and not task.done():
            if not self._running_under(inputs.anchor, inputs.job):
                self._pending = (inputs, instruction)
                return
            await self._progress(inputs.details or {}, "cancelling")
            if self._task is not task:
                # The bookkeeping moved while the beat emitted (round-2 review R2-4): the
                # request no longer names the running job, so it queues behind it -- the
                # sequential shape (memo §2.9, depth 1).
                self._pending = (inputs, instruction)
                return
            newest = await asyncio.to_thread(self._newest_row, inputs.anchor, job=inputs.job)
            if not _is_terminal(newest):
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                if self._task is task:
                    self._task = None
                    self._running = None
                    await self._settle(inputs, state="cancelled")
        # This request executes NOW, so it replaces whatever was queued (memo §2.9's one
        # eviction rule).
        self._pending = None
        self._start(inputs, instruction=instruction)

    def _start(self, inputs: _JobInputs, *, instruction: str) -> None:
        """Start the next version of ``inputs`` on a fresh task.

        The ONE place a fork task and its identity are set: ``_running`` must always describe
        the task the runner holds, because every control op's cut is gated on it (round-1
        review R1). The next version is read from the JOURNAL, not counted in memory: a
        cancelled row and any row written by a previous process both occupy version numbers,
        and reusing one would put two different rows at the same version -- a contradiction
        the newest-version-wins rule would silently resolve to whichever came last.
        """
        inputs.version = self._next_free_version(inputs)
        inputs.instruction = instruction
        self._remember(inputs)
        self._running = (inputs.anchor, inputs.job)
        self._task = asyncio.get_running_loop().create_task(
            self._run_graphics(inputs, self._settings, None)
        )

    def _drain_pending(self) -> None:
        """Start the depth-1 queued steer/restart once the job it waited behind has settled.

        Memo §2.9: "a steer/restart on an *older* anchor while a newer job runs is queued
        behind it (depth 1; a newer request replaces the queued one)". Every settle point
        calls this -- a finished fork, a cut by cancel/dismiss, a decision tail that started
        no fork -- while a superseded task never reaches it (its ``CancelledError`` re-raises
        first), so an explicit queued request outlives a supersede by design.
        """
        pending, self._pending = self._pending, None
        if pending is None:
            return
        inputs, instruction = pending
        self._start(inputs, instruction=instruction)

    async def dismiss(self, anchor: str) -> str:
        """``supplement_dismiss``: the operator's "not useful" signal (memo §2.7).

        Writes ``state=skipped, dismissed=true`` on the anchor's newest version -- the spam
        metric reads ``dismissed`` (§5.2) -- and cancels a live job for that anchor first, so
        a dismissed row cannot be overwritten by the job it just hid. The cut is gated on the
        running job being that anchor's (round-1 review R1): dismissing anchor A while a job
        of anchor B runs must never cut B. The version it writes follows the JOURNAL, read
        inside the write lock (round-3 review R3-1): the newest row -- including one a
        terminal append just made durable while its task was still suspended -- is the one a
        surface dismissing from history has to find, and rebuilding a version that row
        already holds is the class this closes.
        """
        async with self._ops_lock:
            task = self._task
            inputs = self._inputs.get(anchor)
            cut = inputs is not None and self._running_under(anchor)
            if cut:
                assert task is not None  # implied by _running_under; named for the type
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self._task = None
                self._running = None
            if self._pending is not None and self._pending[0].anchor == anchor:
                # A queued steer/restart for this anchor would resurrect the row the dismiss just
                # hid -- the same reason the live job is cut.
                self._pending = None
            if cut:
                # The cut settled the running job: whatever waited behind it (memo §2.9) starts.
                self._drain_pending()

            def build_skipped(base: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
                if base is None:
                    return None
                row: dict[str, Any] = dict(next_version(base, state="skipped"))
                row["dismissed"] = True
                return row

            final = await self._commit_row(anchor=anchor, build=build_skipped, inputs=inputs)
            if final is None:
                return "already finished"
            await self._progress(final, "skipped")
            return "dismissed"

    def _next_free_version(self, inputs: _JobInputs) -> int:
        """One past the newest row this anchor already has (memo §2.4's version rule).

        Read from the journal rather than tracked in memory: a restart after a process
        restart, or a steer on a row this runtime never wrote, must not reuse a version
        number that is already on disk.
        """
        newest = self._newest_row(inputs.anchor)
        if newest is None:
            return inputs.version + 1
        return int(newest.get("version", 0) or 0) + 1

    def _newest_row(self, anchor: str, *, job: str | None = None) -> SupplementDetails | None:
        """The newest journaled row for one anchor -- or one ``(anchor, job)`` -- or ``None``.

        A journal walk on a user action or inside a write's critical section, not on a path:
        it is the only way ``dismiss`` can honour a row whose job this process never ran (a
        restart, or a history-page click), and the alternative -- refusing -- would silently
        drop the operator's one explicit "not useful" signal. ``job`` narrows the read to
        ONE job's rows, which is what the control ops' "has THIS job settled?" decision
        needs (round-3 review R3-1): a different job's terminal row must not answer it, and
        a newer job that has not written its first row yet is still live.
        """
        rows: list[dict[str, Any]] = []
        try:
            for entry in self._session.transcript.entries():
                if entry.type != "custom":
                    continue
                payload = entry.payload if isinstance(entry.payload, dict) else {}
                if payload.get("custom_type") != SUPPLEMENT_CUSTOM_TYPE:
                    continue
                details = payload.get("details")
                if not isinstance(details, dict) or str(details.get("anchor", "")) != anchor:
                    continue
                if job is not None and str(details.get("job", "")) != job:
                    continue
                rows.append(details)
        except Exception:  # noqa: BLE001 -- an unreadable journal is a missing callout
            logger.debug("supplement row could not be read for dismiss", exc_info=True)
            return None
        if not rows:
            return None
        newest = rows[0]
        for row in rows[1:]:
            # Ties keep the LATER journal row -- ``contract.newest_per_anchor``'s documented
            # tie-break: the runtime reader and the contract reader must not disagree on
            # "newest" (QA round 1, Q-1).
            if int(row.get("version", 0) or 0) >= int(newest.get("version", 0) or 0):
                newest = row
        return cast(SupplementDetails, newest)
