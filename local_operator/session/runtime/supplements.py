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
cancelled job had already journaled a non-terminal row, :meth:`SupplementRunner.set_open_row`
tells the runner so it can journal ``cancelled`` / ``error="superseded"`` (a row that renders
nothing -- no Retry under an answer the user has moved past). C1a writes terminal rows only,
so nothing registers one; the seam exists for the generator lane (C1b).

EVERYTHING FAILS OPEN. An exception anywhere in the job is logged at DEBUG and swallowed: the
worst outcome of this feature is "no callout".
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Final

from local_operator.supplements import policy
from local_operator.supplements.candidates import prefilter
from local_operator.supplements.contract import SupplementDetails
from local_operator.supplements.decision import Decision
from local_operator.supplements.decision import decide as decide_supplement
from local_operator.supplements.persistence import (
    append_row,
    build_details,
    new_job_id,
    superseded,
)
from local_operator.supplements.trigger import final_answer, refusal

logger = logging.getLogger(__name__)

#: ``Session.classification_seam`` returns the shared ``ClassificationService`` (or ``None``
#: when the layer is off); resolved PER CALL, the monitor gate's convention, so a seam a test
#: swaps in after construction is the one that answers.
SeamResolver = Callable[[], Any]

#: Whether a generator exists to hand a graphics "yes" to. False until lane C1b: asking a paid
#: question whose answer nothing can act on is waste (decision.decide docstring).
GENERATOR_AVAILABLE: Final = False


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
        self._open_row: SupplementDetails | None = None
        #: Jobs this runtime is running right now: the runtime's half of the reader rule
        #: (``contract.reader_disposition(job_live=...)``).
        self._live_jobs: set[str] = set()

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
            self._task = asyncio.get_running_loop().create_task(self._run(provenance, previous))
            return ""
        except Exception:  # noqa: BLE001 — the event path never fails on this feature
            logger.debug("supplement trigger failed", exc_info=True)
            return "error"

    # -- registry / lifecycle -------------------------------------------------------------

    def is_live(self, job: str) -> bool:
        return job in self._live_jobs

    def set_open_row(self, details: SupplementDetails | None) -> None:
        """Register (or clear) the job's latest NON-terminal row, for supersede (C1b seam)."""
        self._open_row = details

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def cancel(self) -> None:
        """Cancel and reap the task (dispose). Never raises."""
        task = self._task
        self._task = None
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

    async def _close_superseded(self) -> None:
        row, self._open_row = self._open_row, None
        if row is None:
            return
        try:
            await append_row(self._session.transcript, superseded(row))
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
            # retry (contract.reader_disposition).
            details = build_details(
                anchor=anchor, job=job, version=1, state="done", decision=decision
            )
            await append_row(self._session.transcript, details)
        finally:
            self._live_jobs.discard(job)
