"""``<config>/cloud/``: the immutable request and the mutable task record.

WHY THIS EXISTS. Every cloud task this device starts needs a record that
outlives the window that started it — what was asked for, what it became, and
what stopped it — because the provider's own view is a moving inventory (ECS
ages a STOPPED task out about an hour after it stops) and the person asking
"what happened to my run" is usually asking later. §3.6 names the pattern this
copies and why: ``network/move_queue.py``, for the same reasons and with the
same store discipline.

```
<config>/cloud/requests/<request_id>.json   # the intent, written once, 0600
<config>/cloud/tasks/<request_id>.json      # the ledger task, mutable, 0600
```

THE STORE DISCIPLINE, frozen (§3.6). Atomic write is not concurrency control:
every read-modify-write takes a per-record cross-process lock (``flock``,
through ``wakes.lock.WakeWriteLock`` — the tree's bundled-platform
implementation) and writes temp-file + ``os.replace`` at 0600, in a directory
at 0700. ONE WRITER PER RECORD is the rule the queue states for its own file
("its writers are the source RELAY's transitions and ``--cancel-queued`` —
never the requester's memory"), and it is why this module — not the provider —
is the only thing that writes a task's state.

TWO RECORDS, TWO LIFETIMES. The request is the intent and it is written ONCE:
nothing re-opens it, so a reader can trust that what it says is what was asked
for. The task is the lifecycle. They share a ``request_id``, which is what lets
a prune take both.

BOUNDED, because an unbounded record log is a slow leak: ``MAX_CLOUD_TASKS``
records are retained (the newest, pruned on ``list``) and terminal ones older
than ``TASK_TTL_S`` are swept. A LIVE record is never pruned on age — a task
that is still running is not litter, whatever its ``created_at`` says.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.cloud.provider import CloudRefused, ComputeStatus
from local_operator.cloud.types import (
    ComputeTask,
    PoolRequest,
    TaskState,
    new_request_id,
)
from local_operator.paths import config_dir
from local_operator.wakes.lock import LOCK_WAIT_S, WakeWriteLock

if TYPE_CHECKING:  # pragma: no cover - the seam is structural, so nothing is imported
    from local_operator.cloud.provider import ComputeProvider

logger = logging.getLogger(__name__)

#: Subdirectory of ``<config>/`` holding this device's cloud records.
LEDGER_DIRNAME = "cloud"

#: The two halves. Separate directories rather than one with a suffix, because
#: their lifetimes differ and a reader that wants "the intents" should not have to
#: filter "the tasks" out by name.
REQUESTS_DIRNAME = "requests"
TASKS_DIRNAME = "tasks"

#: How long a TERMINAL record is kept for surfaces to read its outcome.
TASK_TTL_S = 30 * 86400.0

#: How many records are retained at all (the newest, pruned on ``list``).
MAX_CLOUD_TASKS = 64

#: How long a task may sit in ``queued``/``provisioning`` before the ledger calls
#: it ``failed``: a task that never reaches ``running`` inside this bound is not
#: going to, and a record that stays ``provisioning`` forever is a record nobody
#: can act on (§3.4).
PROVISION_DEADLINE_S = 600.0


def cloud_dir(root: Path | str) -> Path:
    """``<root>/cloud`` — the ledger's own directory, beside the mesh's stores."""
    return Path(root) / LEDGER_DIRNAME


def requests_dir(root: Path | str) -> Path:
    return cloud_dir(root) / REQUESTS_DIRNAME


def tasks_dir(root: Path | str) -> Path:
    return cloud_dir(root) / TASKS_DIRNAME


def request_path(root: Path | str, request_id: str) -> Path:
    return requests_dir(root) / f"{request_id}.json"


def task_path(root: Path | str, request_id: str) -> Path:
    return tasks_dir(root) / f"{request_id}.json"


def lock_for(root: Path | str, request_id: str, *, timeout_s: float = LOCK_WAIT_S):
    """The per-record cross-process lock (never held across a blocking call).

    ``wakes.lock.WakeWriteLock`` is the tree's bundled-platform flock and the
    ledger gets its OWN file per record, so two tasks never serialise against
    each other and the wake writer's mutex can never interleave with these.

    THE DIRECTORY IS ENSURED HERE rather than by each writer, for the reason
    ``move_queue.lock_for`` states: ``flock`` opens a file in this directory, so
    a first-ever record would otherwise fail with ENOENT instead of taking the
    The lock's directory is ensured here rather than by each writer, and the
    lock is the right owner of that step: ``flock`` opens a file in this
    directory, so a first-ever record would otherwise fail with ENOENT instead of
    taking the lock. ``0700`` matches the store's directory convention; the
    record's own mode is the 0600 the staged write gives it.
    """
    directory = tasks_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:  # noqa: PERF203 — a directory on a no-chmod fs still works
        pass
    return WakeWriteLock(
        directory,
        name=f"{request_id}.lock",
        timeout_s=timeout_s,
        busy_sentence=(
            # Product words only (the same rule ``move_queue``'s refusal follows):
            # "another process" is machine talk, and a refusal cannot promise that
            # a later retry is the fix.
            "Something else on this device is updating this cloud task. Try again "
            "in a moment."
        ),
    )


def _write_json_raw(path: Path, payload: dict[str, Any]) -> Path:
    """Staged 0600 write + ``os.replace``. The caller holds the record's lock."""
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:  # noqa: PERF203 — see ``lock_for``
        pass
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _read_json(path: Path) -> dict[str, Any] | None:
    """One record as raw JSON, or ``None`` when absent or unreadable.

    Unreadable is treated exactly like absent — a record is a coordination file,
    not a journal, and a reader that raised would take down a status surface for
    one bad byte. The writers keep it whole (staged + ``os.replace``).
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("cloud ledger: unreadable record %s; treating as absent", path)
        return None
    if not isinstance(data, dict):
        logger.warning("cloud ledger: skipping non-object record %s", path)
        return None
    return data


def read_request(root: Path | str, request_id: str) -> PoolRequest | None:
    """The recorded intent, or ``None`` when absent or unreadable."""
    data = _read_json(request_path(root, request_id))
    if data is None:
        return None
    try:
        return PoolRequest.from_json(data)
    except CloudRefused as refusal:
        logger.warning("cloud ledger: unreadable request %s: %s", request_id, refusal.sentence)
        return None


def read_task(root: Path | str, request_id: str) -> ComputeTask | None:
    """The recorded task, or ``None`` when absent or unreadable.

    A record this build cannot parse is reported as absent rather than raised:
    the alternative is a ``status`` verb that dies on one corrupt file instead of
    listing the other sixty-three.
    """
    data = _read_json(task_path(root, request_id))
    if data is None:
        return None
    try:
        return ComputeTask.from_json(data)
    except CloudRefused as refusal:
        logger.warning("cloud ledger: unreadable task %s: %s", request_id, refusal.sentence)
        return None


def state_for_status(status: ComputeStatus) -> tuple[TaskState, str]:
    """The ledger state one provider reading implies, plus the reason to record.

    The ledger's state machine is DRAWN from observations (§3.4: "all observed,
    never inferred"), so the mapping from a provider reading to a state lives
    here, next to the machine, rather than in each provider — a provider that
    decided its own task state would be a second writer.

    A ``stopped`` task with no non-zero exit code folds to ``expired`` with the
    provider's own ``stop_reason`` (``drained`` when it names none): ``expired``
    is this vocabulary's "it ran and it is over", and the doc's own
    ``stop_reason`` list has no ``expired`` member.
    """
    if status.phase == "pending":
        return "provisioning", "not_yet_describable"
    if status.phase == "running":
        return "running", ""
    if status.phase == "absent":
        return "lost", status.stop_reason or "provider_ref_absent"
    if status.exit_code:
        return "failed", status.stop_reason or "failed"
    return "expired", status.stop_reason or "drained"


class CloudLedger:
    """This device's cloud records: the intent, the lifecycle, and the sweeps.

    ``root`` defaults to ``paths.config_dir()``, resolved on EVERY call rather
    than at construction: that is what makes an isolated run isolated (the
    override is read from the environment each time), and it is why an explicit
    ``root`` is the only thing a test has to hand in.
    """

    def __init__(self, root: Path | str | None = None, *, lock_timeout_s: float = LOCK_WAIT_S):
        self._root = root
        #: Injectable so a test can assert a contended lock's refusal in bounded
        #: time instead of waiting out the production bound.
        self.lock_timeout_s = lock_timeout_s

    @property
    def root(self) -> Path:
        return Path(self._root) if self._root is not None else config_dir()

    # -- writing ------------------------------------------------------------

    def create(self, request: PoolRequest) -> ComputeTask:
        """Record an intent and open its task. The intent is written ONCE.

        Idempotent for the SAME ``request_id`` while its task is live: a retried
        request refreshes nothing and reports the task that already exists, which
        is ``move_queue.enqueue``'s rule and the reason a double-submit is not a
        second machine. A ``request_id`` that is already TERMINAL is left alone
        too — the id names one run, and a second run needs its own.
        """
        now = time.time()
        request_id = request.request_id or new_request_id()
        with lock_for(self.root, request_id, timeout_s=self.lock_timeout_s):
            existing = read_task(self.root, request_id)
            if existing is not None:
                return existing
            request = replace(
                request,
                request_id=request_id,
                # A stamped ``requested_at`` is when the INTENT was recorded, which
                # is the only moment this process can witness. An intent that
                # arrived with its own timestamp keeps it.
                requested_at=request.requested_at or now,
            )
            task = ComputeTask(
                request_id=request_id,
                tier=request.tier,
                size=request.size,
                spot=request.spot_ok,
                created_at=now,
                deadline_at=now + request.max_hours * 3600.0,
                budget_usd=request.budget_usd,
                repo_url=request.repo_url,
                repo_sha=request.repo_sha,
                branch=f"lop/{request_id}",
                events=[{"at": now, "phase": "queued", "detail": {}}],
            )
            _write_json_raw(request_path(self.root, request_id), request.to_json())
            _write_json_raw(task_path(self.root, request_id), task.to_json())
            return task

    def transition(
        self, request_id: str, state: TaskState, *, detail: str = "", **extra: Any
    ) -> ComputeTask | None:
        """Drive one record's transition, or ``None`` when it refused to move.

        ``None`` covers both "there is no such record" and "the record would not
        move" (already at ``state``, or terminal — ``ComputeTask.transition``).
        The provider is never consulted here: this advances the LEDGER, and what
        the ledger is told must already have been observed.
        """
        with lock_for(self.root, request_id, timeout_s=self.lock_timeout_s):
            task = read_task(self.root, request_id)
            if task is None:
                return None
            if not task.transition(state, detail=detail, **extra):
                # A re-observation stamps ``last_seen_at`` and nothing else; it is
                # still a change to the record, so it is still written.
                if task.last_seen_at is not None:
                    _write_json_raw(task_path(self.root, request_id), task.to_json())
                return None
            _write_json_raw(task_path(self.root, request_id), task.to_json())
            return task

    def cancel(self, request_id: str) -> tuple[ComputeTask | None, str]:
        """Stop a task that has not started, or say why it is too late.

        Returns ``(task, outcome)`` where outcome is one of:

        * ``"cancelled"`` — the record is now terminal ``cancelled``,
        * ``"too_late"`` — the task asked to stop itself (``draining``); the
          provider owns the outcome now and the record's state says which,
        * ``"already"`` — the record was already terminal; the state says why,
        * ``"absent"`` — there is no record for this id here.

        ``--cancel-queued``'s vocabulary, kept verbatim: a caller that has to
        render a refusal must not also have to learn a second set of words.
        """
        with lock_for(self.root, request_id, timeout_s=self.lock_timeout_s):
            task = read_task(self.root, request_id)
            if task is None:
                return None, "absent"
            if task.state in ("draining",):
                return task, "too_late"
            if task.terminal:
                return task, "already"
            task.transition("cancelled")
            _write_json_raw(task_path(self.root, request_id), task.to_json())
            return task, "cancelled"

    def sweep_terminal(self, *, max_age_s: float = TASK_TTL_S) -> list[str]:
        """Remove terminal records older than ``max_age_s``, and their intents.

        "Older" is measured from when the task STOPPED, not from when it was
        created: a two-hour run whose record ages out by creation time would be
        swept while a surface was still reading it.
        """
        removed: list[str] = []
        for task in self.list_tasks(prune=False):
            if not task.terminal:
                continue
            ended = task.stopped_at or task.created_at
            if time.time() - ended < max_age_s:
                continue
            remove_ids = [task_path(self.root, task.request_id)]
            remove_ids.append(request_path(self.root, task.request_id))
            try:
                for path in remove_ids:
                    path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.debug("cloud ledger: could not prune %s", task.request_id, exc_info=True)
                continue
            removed.append(task.request_id)
        return removed

    # -- reading ------------------------------------------------------------

    def read(self, request_id: str) -> ComputeTask | None:
        """One task record, or ``None`` when absent or unreadable."""
        return read_task(self.root, request_id)

    def payload(self, task: ComputeTask | None) -> dict[str, Any] | None:
        """The wire/receipt projection of a task (a copy; never the live record).

        The projection is deliberately NARROWER than the record: it is what a
        surface shows and what ``--json`` prints, so a field added to the record
        for the ledger's own use does not silently become part of a contract
        another process parses.
        """
        if task is None:
            return None
        return {
            "request_id": task.request_id,
            "tier": task.tier,
            "provider": task.provider,
            "provider_ref": task.provider_ref,
            "state": task.state,
            "size": task.size,
            "spot": task.spot,
            "spot_fallback": task.spot_fallback,
            "branch": task.branch,
            "session_id": task.session_id,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "stopped_at": task.stopped_at,
            "exit_code": task.exit_code,
            "stop_reason": task.stop_reason,
            "budget_usd": task.budget_usd,
            "member_state": task.member_state,
            "events": [dict(event) for event in task.events],
        }

    def list_tasks(self, *, prune: bool = True) -> list[ComputeTask]:
        """Every retained record, newest first. Prunes on the way (``prune=True``)."""
        directory = tasks_dir(self.root)
        try:
            names = sorted(os.listdir(directory))
        except FileNotFoundError:
            return []
        except OSError:
            logger.warning("cloud ledger: cannot list %s", directory)
            return []
        tasks: list[ComputeTask] = []
        for name in names:
            if not name.endswith(".json") or name.startswith("."):
                continue
            task = read_task(self.root, name[: -len(".json")])
            if task is not None:
                tasks.append(task)
        tasks.sort(key=lambda task: (task.created_at, task.request_id), reverse=True)
        if prune:
            tasks = self._prune(tasks)
        return tasks

    def _prune(self, tasks: list[ComputeTask]) -> list[ComputeTask]:
        """Drop TERMINAL records past ``MAX_CLOUD_TASKS``, and return what is left.

        Only terminal records are eligible: a live task is never dropped to stay
        inside a bound, because the bound exists to stop a log growing without
        end, not to forget a machine that is still running.

        The RETAINED list is what this returns, so a caller cannot print a record
        this very call just deleted.
        """
        if len(tasks) <= MAX_CLOUD_TASKS:
            return tasks
        retained: list[ComputeTask] = []
        for index, task in enumerate(tasks):
            if index < MAX_CLOUD_TASKS or not task.terminal:
                retained.append(task)
                continue
            for path in (
                task_path(self.root, task.request_id),
                request_path(self.root, task.request_id),
            ):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.debug("cloud ledger: could not prune %s", task.request_id, exc_info=True)
        return retained

    # -- reconciling --------------------------------------------------------

    async def reconcile(self, provider: ComputeProvider) -> dict[str, str]:
        """Re-read every LIVE record against ``provider`` and drive its transition.

        Returns ``{request_id: what it now reads as}``. This is the queue's
        ``reconcile_on_start`` narrowed to what a cloud task needs: a move needs a
        driver to be RE-ARMED after a crash because the move is the only thing
        that will ever finish it, while a cloud task is observable through its
        provider — so this re-reads and reports instead of re-driving (§9.2).

        A provider that cannot answer leaves its record LIVE and says so in this
        map. Guessing there is the one thing a reconciler must not do: a task
        reported as ended while it is still burning money is worse than a task
        reported as unknown.
        """
        outcomes: dict[str, str] = {}
        for task in self.list_tasks():
            if task.terminal or not task.provider_ref:
                continue
            try:
                status = await provider.status(task.provider_ref)
            except CloudRefused as refusal:
                outcomes[task.request_id] = f"unavailable: {refusal.sentence}"
                continue
            state, reason = state_for_status(status)
            if state == task.state:
                # A reading that changed nothing is still a sighting: the
                # transition stamps ``last_seen_at`` and appends no event.
                self.transition(task.request_id, state)
                outcomes[task.request_id] = state
                continue
            moved = self.transition(
                task.request_id,
                state,
                detail=status.stop_reason,
                reason=reason,
                exit_code=status.exit_code,
            )
            outcomes[task.request_id] = moved.state if moved is not None else state
        return outcomes

    # -- the provisioning deadline -----------------------------------------

    def expire_stalled(self, *, deadline_s: float = PROVISION_DEADLINE_S) -> list[str]:
        """Fail records that never reached ``running`` inside the deadline.

        §3.4's ``queued``/``provisioning`` → ``failed``: no ``running`` within
        ``PROVISION_DEADLINE_S`` means the task is not going to start, and a
        record that says ``provisioning`` forever is one nobody can act on. The
        deadline is measured from ``submitted_at`` when there is one (the task was
        handed to a provider) and from ``created_at`` otherwise, which is the
        honest reading of "how long has this been waiting".
        """
        expired: list[str] = []
        now = time.time()
        for task in self.list_tasks():
            if task.state not in ("queued", "provisioning"):
                continue
            since = task.submitted_at or task.created_at
            if since <= 0.0 or now - since < deadline_s:
                continue
            moved = self.transition(
                task.request_id,
                "failed",
                detail=f"no running state within {deadline_s:g}s",
                reason="provision_deadline",
            )
            if moved is not None:
                expired.append(task.request_id)
        return expired


__all__ = [
    "CloudLedger",
    "LEDGER_DIRNAME",
    "MAX_CLOUD_TASKS",
    "PROVISION_DEADLINE_S",
    "REQUESTS_DIRNAME",
    "TASKS_DIRNAME",
    "TASK_TTL_S",
    "cloud_dir",
    "lock_for",
    "read_request",
    "read_task",
    "request_path",
    "requests_dir",
    "state_for_status",
    "task_path",
    "tasks_dir",
]
