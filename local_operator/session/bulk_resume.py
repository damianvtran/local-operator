"""Bulk resume: select a set of stored sessions and reopen each, bounded.

WHY THIS MODULE EXISTS (the operator escalation, 2026-10-01): resuming 15+
paused/failed sessions one ``lop exec --resume <id>`` at a time took ~10
minutes and had to be babysat. Nothing about that work was sequential except
the invocation pattern — enumeration is one store scan and the reopen itself
is an independent child process per session — so the fix is enumerate-once
plus N children under a capacity bound, with the per-session outcome reported
as it resolves.

THREE THINGS THIS MODULE DELIBERATELY REUSES RATHER THAN REBUILDS:

* the selection walker — ``info.collect.stored_sessions_by_outcome`` over
  ``resume.recent_sessions``, the same single store walker every listing uses
  (the filter-before-limit rule lives there; this module only slices the
  already-filtered list to report the pre-cap count for the truncated-set
  message, which is the identical operation ``limit`` performs inside it);
* the single-resume PATH itself — each child is the real CLI
  (``python -P -m local_operator.cli exec --background --resume <id> -- <msg>``),
  so every guard, refusal, receipt and ledger row is the single command's, not
  a second implementation that can drift from it;
* the readiness wait — kept intact per child (its receipt is what says
  whether the job went live), with one shared follow-up pass for children
  whose receipt read ``starting``.

CAPACITY, MEASURED (2026-10-01, this host under ~25 sibling sessions and load
35+): a 14-child burst at full parallelism returned DEGRADED receipts (several
children hit their 5 s readiness window and read ``starting``) and a total
wall of 20.6 s, where the same set at bounded parallelism completed in 15.8 s
with accurate receipts — the box's CPU, not the network, is the contended
resource, and each child costs two to three CPython starts. The default is
therefore a small fixed bound rather than the AsyncJobManager's subagent
capacity (15): subagents are coroutines in one process, these are OS
processes. The number is a constant, not a flag — v1 keeps the surface to
set-previews and limits, and a caller who needs a different bound has the
single-session path.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from local_operator.interpreter import python_argv

#: The message a session receives when the caller did not supply one. The
#: phrasing matches the product's own continuation wording (see
#: ``session._CONTINUATION_PROMPT``) so a resumed transcript reads the same as
#: every other automatic continuation, and an operator who wants different
#: words passes their own.
DEFAULT_RESUME_MESSAGE = "Continue the task from where it left off."

#: How many children a bulk resume keeps running at once. See the module
#: docstring for the measurement this number comes from; it is deliberately
#: well under the AsyncJobManager's 15-subagent capacity because each child is
#: a process tree, not a coroutine.
RESUME_BATCH_CONCURRENCY = 6

#: How many sessions a bulk resume acts on when the caller gives no limit.
#: An ACTION needs a safe default the way a LISTING does — ``lop sessions
#: --all`` on a well-used store matches thousands of directories, and "resume
#: everything" must not be one forgotten flag away. The cap keeps the NEWEST
#: members (the selector's own order) and every surface that truncates says
#: so, naming the pre-cap count.
RESUME_DEFAULT_LIMIT = 20

#: Per-child bound on one ``lop exec --background`` launcher, mirroring the
#: sessions tool's ``SESSIONS_LAUNCH_TIMEOUT_S``: a launcher stuck past this
#: is killed (its whole process group) and reported as failed, loudly.
RESUME_CHILD_TIMEOUT_S = 120.0

#: How long the shared follow-up pass waits for children whose receipt read
#: ``starting`` (the worker had not published its running row inside the
#: launcher's own 5 s window — common when a whole batch boots at once).
#: Six times the launcher's window: long enough that a healthy batch resolves
#: in it, short enough that a stalled worker is reported rather than awaited
#: forever. A straggler is reported as ``unresolved`` with the status command
#: to check it — never a silent hang.
RESUME_READY_FOLLOWUP_S = 30.0

#: The launcher's receipt line ("Background job <id>: <status> …"). The same
#: spelling the sessions tool parses; kept here rather than imported from the
#: tool because this module runs in both the CLI and the tool and neither may
#: depend on the other's module (the tool region is ``builtin``'s). Group 3
#: is the line's REMAINDER — where a fast worker failure carries its reason
#: ("… failed (execution receipt) — worker died: boom"), which review round
#: 1's m1 requires the child reader to surface.
_JOB_LINE_RE = re.compile(r"^Background job ([^:\s]+):\s*(\S+)(.*)$", re.MULTILINE)

#: ANSI SGR sequences, stripped from failure details so a one-line report
#: stays one readable line whatever colour the child painted (``lop exec``'s
#: refusals are red).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: Cap on a rendered detail (a log's last line, a stderr tail).
_DETAIL_CHARS = 240


@dataclass(frozen=True)
class ResumeSelection:
    """What a selector matched: the capped set plus the pre-cap count.

    ``matched`` is the count BEFORE the cap so a caller can say "the newest 20
    of 34" — leaving work quietly unmentioned is exactly the failure the
    filter-before-limit rule exists to prevent on the listing side.
    """

    sessions: tuple[tuple[str, float], ...]
    matched: int
    kinds: frozenset[str]


@dataclass
class ResumeOutcome:
    """One session's result: the unit the CLI line and the tool summary render.

    ``status`` is the terminal reading this module could PROVE:
    ``running``/``succeeded`` (the ledger showed the job live or done),
    ``refused``/``failed`` (the launcher itself exited non-zero — its stderr
    is in ``detail``), ``unresolved`` (still ``starting`` after the follow-up
    bound — the job may yet boot; check it with the status command), or
    ``timeout`` (the launcher exceeded its own bound and was killed).
    ``ok`` is the coarse contract the exit codes read: True only for
    ``running``/``succeeded``.
    """

    session_id: str
    name: str = ""
    ok: bool = False
    status: str = ""
    job_id: str = ""
    detail: str = ""
    wall_s: float = 0.0


def live_session_ids(root: Path) -> set[str]:
    """Session ids the registry reports as RUNNING (live or wedged), for exclusion.

    The exclusion set has to draw the line at "a process exists", not at "a
    record exists": a session whose record is stale (its pid is gone, and the
    scan is deleting the record as it answers) is not running — it is a
    death to resume, which is the operator's own scenario for this feature —
    while a live or wedged record names a pid that still exists and must not
    be raced. The child's own lease check refuses anything this set misses,
    so the exclusion narrows what is OFFERED and never weakens the guard.

    Best-effort: an unreadable registry yields no exclusions; the child's
    guard still holds, so the failure direction is a refusal, not a race.
    """
    from local_operator.session.runtime import registry

    try:
        scanned = registry.scan(root)
    except Exception:  # noqa: BLE001 — an exclusion set is a filter, never a gate
        return set()
    #: The states that mean a process EXISTS behind the record — the same two
    #: the sessions tool's own listing treats as live, and the same line
    #: ``info.collect`` draws ("``live`` and ``wedged`` both name a pid that
    #: exists"). ``stale`` is deliberately NOT here: the scan just deleted a
    #: record whose pid is gone, so that session is resumable — which is
    #: exactly the crashed-runtime shape a bulk resume exists to sweep up —
    #: and the child's own lease check remains the backstop for anything the
    #: registry cannot see.
    excluded_states = ("live", "wedged")
    return {
        str(rec.session_id)
        for rec, state in scanned
        if state in excluded_states and getattr(rec, "session_id", None)
    }


def select_resume_candidates(
    root: Path,
    *,
    paused: bool = False,
    failed: bool = False,
    all_sessions: bool = False,
    limit: int | None = None,
    exclude_ids: set[str] | None = None,
) -> ResumeSelection:
    """The sessions a bulk resume should reopen, newest first.

    The vocabulary is the store's own, mapped once here and in the listing
    (``FAILED_OUTCOME_KINDS`` = ``error``; ``PAUSED_OUTCOME_KINDS`` =
    ``{interrupted, retired}``): ``paused`` and ``failed`` union when both are
    given, and either one alone is a complete request (it implies the store
    scope, mirroring the ``lop sessions --paused/--failed`` listing flags).
    ``all_sessions`` WINS over the kind selectors: it selects every stored,
    non-live session, and ``--paused --all`` is all — the widest reading,
    which is the one the CLI header and the tool approval name (review round
    1, m3). It stays behind an explicit flag and a cap because "resume
    everything" is a decision, not a default.

    ``limit`` caps the ANSWER (newest first); ``None`` means
    :data:`RESUME_DEFAULT_LIMIT` — an action with no bound is a footgun, so
    the default is a cap and a caller wanting more says so. The pre-cap count
    rides in ``matched`` so the caller can say when work was left behind.
    ``exclude_ids`` drops live sessions BEFORE the cap (see
    :func:`live_session_ids`).
    """
    from local_operator.info.collect import (
        FAILED_OUTCOME_KINDS,
        PAUSED_OUTCOME_KINDS,
        stored_sessions_by_outcome,
    )
    from local_operator.resume import recent_sessions

    effective_limit = RESUME_DEFAULT_LIMIT if limit is None else limit
    kinds: frozenset[str] = frozenset()
    if paused:
        kinds |= PAUSED_OUTCOME_KINDS
    if failed:
        kinds |= FAILED_OUTCOME_KINDS
    if all_sessions:
        # ALL WINS over the kind selectors — `--paused --all` reads as "the
        # widest of these", and review round 1's m3 was exactly a claim and a
        # behaviour telling two stories (the header said "paused+all" while
        # the selector intersected to paused). One story now: `all` takes the
        # whole store, the header and the approval name `all` alone, and the
        # kind sets are not recorded as a filter that was not applied.
        kinds = frozenset()
        candidates = recent_sessions(root, None)
        if exclude_ids:
            candidates = [row for row in candidates if row[0] not in exclude_ids]
        matched = candidates
    elif kinds:
        # ``limit=None`` here is the acting-on-a-set spelling the selector's
        # docstring names; the slice below is the same operation its own
        # ``limit`` arm performs, moved here only so ``matched`` can report
        # what the cap hid.
        matched = stored_sessions_by_outcome(root, kinds, exclude_ids=exclude_ids, limit=None)
    else:
        return ResumeSelection((), 0, kinds)
    return ResumeSelection(tuple(matched[:effective_limit]), len(matched), kinds)


def resume_child_argv(session_id: str, message: str) -> list[str]:
    """The child invocation AFTER the interpreter prefix (``python_argv``).

    ``exec --background --resume <id> -- <msg>`` is byte-for-byte the order
    ``_sessions_open_argv`` builds for a resume, so a batch child is literally
    the single command with its prompt already attached.

    A MODULE-LEVEL function on purpose: it is the one seam a test can
    substitute to run a stub child (``python_argv(*resume_child_argv(...))``
    is the whole spawn line), which is what lets the bounded-parallelism and
    failure-isolation tests exercise the REAL runner without spawning real
    sessions.
    """
    return [
        "-m",
        "local_operator.cli",
        "exec",
        "--background",
        "--resume",
        session_id,
        "--",
        message,
    ]


def _collapse(text: str) -> str:
    """One line: ANSI stripped, whitespace collapsed, capped."""
    flat = " ".join(_ANSI_RE.sub("", text).split())
    return flat[:_DETAIL_CHARS]


def _log_tail_line(path: str | None) -> str:
    """The last non-empty line of a job log, or ``""``.

    Best-effort (the log may not exist yet, may be mid-write, may be gone) and
    bounded: the tail 8 KiB only, so a multi-hundred-MB transcript-adjacent
    log is never read whole just to render one line.
    """
    if not path:
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            data = handle.read()
    except OSError:
        return ""
    for line in reversed(data.decode("utf-8", "replace").splitlines()):
        if line.strip():
            return _collapse(line)
    return ""


def _kill_group(process: "asyncio.subprocess.Process") -> None:
    """Kill the child and its whole process group, by exact pid.

    ``start_new_session=True`` on the spawn makes the child a group leader, so
    its PID IS ITS PGID and is signalled DIRECTLY — never through
    ``os.getpgid(process.pid)``: once the child has exited and been reaped
    that lookup raises, and the old spelling then degraded to a
    single-process kill that never reached a pipe-holding grandchild
    (review round 1, m4 — reproduced as a survivor after the batch;
    ``clipboard._kill_tree`` records the same failure mode and the same
    remembered-pgid fix). The group is pid-scoped to a tree this module
    created, so the kill can never reach anything else.
    """
    try:
        os.killpg(process.pid, signal.SIGKILL)
        return
    except (OSError, AttributeError):
        # Already reaped, the group vanished between the check and the signal,
        # or the platform has no groups at all (Windows: no ``os.killpg``).
        # The direct kill below is the remaining option.
        pass
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass


async def _run_child(
    session_id: str,
    name: str,
    *,
    message: str,
    env: Mapping[str, str],
    cwd: str | None,
    semaphore: asyncio.Semaphore,
) -> ResumeOutcome:
    """Spawn ONE resume child, bounded, and read its receipt."""
    async with semaphore:
        started = time.perf_counter()
        argv = python_argv(*resume_child_argv(session_id, message))
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(env),
                cwd=cwd,
                start_new_session=True,
            )
        except OSError as exc:
            return ResumeOutcome(
                session_id=session_id,
                name=name,
                ok=False,
                status="failed",
                detail=f"could not start the resume child: {exc}",
                wall_s=time.perf_counter() - started,
            )
        try:
            _out, err = await asyncio.wait_for(process.communicate(), RESUME_CHILD_TIMEOUT_S)
        except (asyncio.TimeoutError, TimeoutError):
            _kill_group(process)
            await process.wait()
            return ResumeOutcome(
                session_id=session_id,
                name=name,
                ok=False,
                status="timeout",
                detail=(
                    f"the launcher did not finish within {int(RESUME_CHILD_TIMEOUT_S)}s "
                    "and was killed; check `lop sessions` before retrying"
                ),
                wall_s=time.perf_counter() - started,
            )
        except asyncio.CancelledError:
            # A cancelled batch must not strand a live child: the group kill
            # is pid-scoped to the tree this child created (see ``_kill_group``),
            # and the wait reaps it before the cancellation propagates.
            _kill_group(process)
            await process.wait()
            raise
        wall = time.perf_counter() - started
        text = err.decode("utf-8", "replace")
        match = _JOB_LINE_RE.search(text)
        job_id = match.group(1) if match else ""
        status = match.group(2) if match else ""
        if process.returncode != 0:
            # rc != 0 is the launcher's own failure, and the receipt it just
            # printed carries the reason. The reason must SURVIVE (review
            # round 1, m1): the old shape took the last stderr line, which is
            # receipt chrome (``Log: <path>``), and the workspace reason was
            # dropped from the only line the operator reads. Parse the shape
            # the launcher actually prints: the job line's em-dash suffix for
            # a run that died, and any non-chrome line as the fallback for a
            # refusal that never created a job. The log path stays attached.
            reason = ""
            if match is not None:
                dashed = re.search("\u2014\\s*(.+)$", match.group(3))
                if dashed:
                    reason = dashed.group(1).strip()
            if not reason:
                for line in reversed(text.splitlines()):
                    stripped = line.strip()
                    if stripped and not stripped.startswith(("Log: ", "Status: ")):
                        reason = stripped
                        break
            log_match = re.search(r"^Log: (.+)$", text, re.MULTILINE)
            log_path = log_match.group(1).strip() if log_match else ""
            detail = (
                _collapse(reason)
                if reason
                else f"`lop exec` exited {process.returncode} without a reason"
            )
            if log_path and log_path not in detail:
                detail = f"{detail} — log: {log_path}"
            return ResumeOutcome(
                session_id=session_id,
                name=name,
                ok=False,
                # A parsed job line means a run existed and died (the
                # launcher's own word for it); no job line means the refusal
                # happened before anything was created, which the loud
                # lease/preflight messages already distinguish.
                status=status if status else "refused",
                job_id=job_id,
                detail=detail,
                wall_s=wall,
            )
        if not job_id:
            # rc 0 with no parseable receipt is a bug to surface, not to paper
            # over — the same posture the sessions tool takes on this shape.
            return ResumeOutcome(
                session_id=session_id,
                name=name,
                ok=False,
                status="failed",
                detail=(
                    "`lop exec` exited 0 but printed no background receipt: "
                    f"{_collapse(text.strip())!r}"
                ),
                wall_s=wall,
            )
        ok = status in ("running", "succeeded")
        return ResumeOutcome(
            session_id=session_id,
            name=name,
            ok=ok,
            status=status or "starting",
            job_id=job_id,
            wall_s=wall,
        )


async def _follow_up(
    pending: list[ResumeOutcome],
    *,
    progress: Callable[[ResumeOutcome], None] | None,
) -> None:
    """Resolve children whose receipt read ``starting``, one shared pass.

    Reads the same append-only ledger the launcher watches, fanned across all
    pending jobs (one initial fold each, then only appended bytes), so the
    cost of waiting on N boots is one watcher rather than N launcher polls.
    A job that goes live becomes ok; a terminal failure becomes a failed
    outcome with the log's last line as its detail; a job still ``starting``
    at the bound becomes ``unresolved`` — reported, never awaited forever.
    """
    from local_operator.exec_mode import (
        JOBS_FILE,
        job_status,
        logs_dir,
        read_job_records_since,
    )

    jobs_path = logs_dir() / JOBS_FILE
    try:
        offset = jobs_path.stat().st_size
    except OSError:
        offset = 0
    try:
        states = {
            outcome.job_id: await asyncio.to_thread(job_status, outcome.job_id, reconcile=False)
            for outcome in pending
        }
    except Exception:  # noqa: BLE001 — a status read must not kill the batch
        states = {outcome.job_id: {} for outcome in pending}
    deadline = time.monotonic() + RESUME_READY_FOLLOWUP_S
    remaining = list(pending)
    # THE INITIAL READ IS APPLIED BEFORE WAITING (review round 1, m2). A child
    # whose job already went live — or already died — while its launcher
    # receipt still read `starting` must resolve NOW; the bound below is only
    # for jobs that genuinely still are starting. The old shape folded the
    # initial states only inside the delta branch, so a child whose ledger row
    # predated the window waited the full bound against a row it was already
    # holding.
    remaining = _apply_states(remaining, states, progress)
    if not remaining:
        return
    while remaining and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        try:
            offset, rows = await asyncio.to_thread(read_job_records_since, jobs_path, offset)
        except Exception:  # noqa: BLE001
            rows = []
        if rows:
            for row in rows:
                job_id = str(row.get("id") or "")
                state = states.get(job_id)
                if state is None:
                    continue
                current = _rank(str(state.get("status") or ""))
                if _rank(str(row.get("status") or "")) >= current:
                    state.update(row)
            remaining = _apply_states(remaining, states, progress)
    for outcome in list(remaining):
        # The bound is spent: one reconciled read per straggler for the most
        # honest final word (a dead owner folds to `interrupted` here).
        try:
            state = await asyncio.to_thread(job_status, outcome.job_id)
        except Exception:  # noqa: BLE001
            state = states.get(outcome.job_id, {})
        states[outcome.job_id] = state
    remaining = _apply_states(remaining, states, progress)
    for outcome in remaining:
        outcome.ok = False
        outcome.status = "unresolved"
        outcome.detail = (
            f"still starting after {int(RESUME_READY_FOLLOWUP_S)}s — check "
            f"`lop exec --status {outcome.job_id}`"
        )
        if progress is not None:
            progress(outcome)


def _rank(status: str) -> int:
    return {
        "starting": 0,
        "running": 1,
        "succeeded": 2,
        "failed": 2,
        "cancelled": 2,
        "interrupted": 2,
    }.get(status, 0)


def _apply_states(
    remaining: list[ResumeOutcome],
    states: Mapping[str, Mapping[str, Any]],
    progress: Callable[[ResumeOutcome], None] | None,
) -> list[ResumeOutcome]:
    """Fold each pending outcome against its ledger state; return the still-pending."""
    still_pending: list[ResumeOutcome] = []
    for outcome in remaining:
        state = states.get(outcome.job_id) or {}
        status = str(state.get("status") or "")
        if status in ("running", "succeeded"):
            outcome.ok = True
            outcome.status = status
            if progress is not None:
                progress(outcome)
            continue
        if status in ("failed", "cancelled", "interrupted"):
            outcome.ok = False
            outcome.status = status
            reason = _log_tail_line(str(state.get("log") or ""))
            outcome.detail = reason or f"job {outcome.job_id} {status}"
            if progress is not None:
                progress(outcome)
            continue
        still_pending.append(outcome)
    return still_pending


async def resume_sessions(
    sessions: Sequence[tuple[str, str]],
    *,
    message: str = DEFAULT_RESUME_MESSAGE,
    env: Mapping[str, str],
    cwd: str | None = None,
    concurrency: int = RESUME_BATCH_CONCURRENCY,
    progress: Callable[[ResumeOutcome], None] | None = None,
) -> list[ResumeOutcome]:
    """Resume ``(session_id, name)`` pairs as bounded concurrent children.

    Returns one :class:`ResumeOutcome` per input pair, in COMPLETION order
    (``progress``, when given, fires per outcome the moment it resolves — the
    streaming contract the CLI's live lines print, review round 1 MAJOR-1:
    a line must appear as its child resolves, not once after the batch). A
    child whose receipt read ``starting`` is held for one shared follow-up
    pass, which fires its progress when it resolves — so every returned
    outcome carries a terminal reading and no line is printed twice.
    """
    if not sessions:
        return []
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(session_id: str, name: str) -> ResumeOutcome:
        outcome = await _run_child(
            session_id,
            name,
            message=message,
            env=env,
            cwd=cwd,
            semaphore=semaphore,
        )
        return outcome

    tasks = [asyncio.create_task(one(sid, name)) for sid, name in sessions]
    results: list[ResumeOutcome] = []
    try:
        for completed in asyncio.as_completed(tasks):
            try:
                item = await completed
            except Exception as exc:  # noqa: BLE001 — one child's bookkeeping
                # A crash inside one child must not cost the batch its report;
                # the exception lost its session, so the row is synthetic.
                synthetic = ResumeOutcome(
                    session_id="?",
                    name="",
                    ok=False,
                    status="failed",
                    detail=f"internal error while resuming a child: {exc!r}",
                )
                results.append(synthetic)
                if progress is not None:
                    progress(synthetic)
                continue
            results.append(item)
            if item.status == "starting" and not item.ok:
                # Held for the shared follow-up pass; its line fires there.
                continue
            if progress is not None:
                progress(item)
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
    pending = [o for o in results if o.status == "starting" and not o.ok]
    if pending:
        await _follow_up(pending, progress=progress)
    return results
