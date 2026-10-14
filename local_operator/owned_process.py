"""Generic managed-process ownership: an opt-in lease over ONE detached owner.

WHY THIS EXISTS (PR #2294 review rounds 1-2, R2). The evaluation caller learned
to refuse releasing an unverified worker, but the STANDALONE entry still ran the
whole episode inside the foreground ``asyncio.run`` loop: the moment that loop
returned, ``Runner`` cancelled every pending task -- including the canonical
cleanup the receipt had just called unverified. A descriptor and a TTL keep the
resource discoverable; neither is an OWNER. Nothing joins the cleanup, and an
operator cannot reconnect to a run whose process has exited.

This module is the minimal, benchmark-agnostic capability that supplies the
missing lifetime: a caller starts ONE process -- its own pinned command line,
an explicit non-secret environment, a unique owner id -- under a managed lease,
and that process OWNS the work. The frontend can wait boundedly for status and
then detach; detaching never stops the owner, and the owner keeps its scopes,
fds, subscribers and canonical cleanup alive until its own run completes. It is
process ownership, not a runtime: no model loop, no tool loop, no remote
session proxy, no instruction channel, no global registry.

WHAT IT DELIBERATELY IS NOT:

* No shell and no string evaluation: ``argv`` is an explicit list; nothing here
  builds a command line from text.
* No force-kill tree claim. POSIX signalling cannot prove an arbitrary process
  TREE stopped (see ``procstate.terminate_process_tree``), so nothing here
  infers descendant death. The guarantee is "the owner survives the frontend",
  not "infinitely resistant children can be force-killed"; a run that NEEDS
  finite whole-tree death must be refused before it starts (the frontend
  does), never papered over with a signal.
* No restart, ever: the Darwin job is ``KeepAlive=false`` and the POSIX child
  is a plain detached process. A crashed owner stays dead -- an episode is
  never silently re-run or re-allocated.
* No secrets in argv, the environment block, the plist, or the status file --
  and no plaintext secret file either. There is NO secret channel here at all
  (round 4, S3: mode 0600 is access control, not encryption, and it bought a
  pre-bind write edge, an ignorable unlink error and a durable artifact for a
  crashed host). The owner resolves its refs from the sealed credential store
  through the isolated config root it inherits; a managed run that lists
  env-only overrides is refused before allocation, and no value of any
  argument is persisted into the plist.

IDENTITY, NOT FLAGS (round-2 acceptance case 2). The owner verifies itself
against the lease record the frontend wrote BEFORE it starts: the recorded
``owner_id``, its own pid, and the pid's birth token must all agree
(``procstate.birth_token``/``same_birth`` -- a reused pid is caught, because a
record naming a different holder of the number is a STALE record, not this
run), and on Darwin launchd itself must report the job running under this pid
(``launchd.job_pid``). A hand-forged ``--owned-host`` flag with no lease, a
foreign lease, or a label launchd does not run as this pid is refused before
any work starts.

BACKENDS. ``launchd`` (Darwin production): a per-owner LaunchAgent
``com.damian.lop-episode.<owner_id>`` -- ``RunAtLoad`` + ``KeepAlive=false`` --
written to the REAL passwd home's ``Library/LaunchAgents`` (``launchd.real_home``
derives the home from the uid, never ``$HOME``, so a redirected home cannot
address or write the operator's jobs and vice versa). ``detached`` (POSIX
fallback used by rigs and tests): ``procstate.detached_popen_kwargs`` plus the
same lease/status contract; it proves process separation and pid identity, not
a supervisor. A caller that requires managed retention on Darwin uses launchd;
on a platform with neither, the caller refuses before allocating (the frontend
maps ``auto`` to exactly that).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from local_operator import procstate

#: Environment variable names the frontend sets for the owned host and the
#: host verifies. Non-secret by construction: the owner id is a nonce, the
#: backend/job label are fixed spellings, and the source pin is a commit hash.
ENV_OWNER_ID = "LOP_OWNED_OWNER_ID"
ENV_BACKEND = "LOP_OWNED_BACKEND"
ENV_JOB_LABEL = "LOP_OWNED_JOB_LABEL"
ENV_SOURCE_PIN = "LOP_OWNED_SOURCE_PIN"
ENV_LEASE_PATH = "LOP_OWNED_LEASE_PATH"

#: How long the owner waits for the frontend to bind its pid into the lease
#: record. The child can boot before the parent has finished writing the
#: post-spawn record, so the owner polls this bound before refusing.
_LEASE_BIND_TIMEOUT_S = 15.0
_LEASE_BIND_POLL_S = 0.05

#: How long start_owned_process waits for the spawned owner to appear under
#: the backend before declaring the start failed (and cleaning its own job).
_OWNER_APPEAR_TIMEOUT_S = 20.0
_OWNER_APPEAR_POLL_S = 0.1

#: Poll interval for lease.wait()'s bounded status reads.
_STATUS_POLL_S = 0.25

#: Files under ``<run_root>/owner/<owner_id>/``. One layout for every writer
#: and reader -- the frontend, the owner, and the tests cannot drift apart
#: about where the lease or the status live.
#:
#: "owner" is an INTERNAL run-root segment, never a session-copy entry: the
#: whole tree lives under a caller-chosen DURABLE run root (``--run-root``),
#: reaches no ``sessions/`` directory, and carries no translated user text.
_OWNER_DIR = "owner"

_OWNER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{3,63}$")

#: Terminal phases: a reader that sees one of these may stop waiting and act
#: on ``outcome_path``. ``provisional`` is deliberately NOT terminal -- it is
#: the retained state a frontend reports as unjoined, not an end.
TERMINAL_PHASES = ("complete", "ended")

PHASE_STARTING = "starting"
PHASE_ADMITTED = "admitted"
PHASE_RUNNING = "running"
PHASE_PROVISIONAL = "provisional"
PHASE_CLOSING = "closing"


class OwnedProcessError(RuntimeError):
    """Base: the managed-process contract could not be honoured."""


class OwnedProcessRefused(OwnedProcessError):
    """A start or admission refusal, with the remedy already in the sentence.

    A refusal is ALWAYS fail-closed and pre-work: the owner was never admitted
    (or the caller never started one), so no episode resource exists yet.
    """


@dataclass(frozen=True)
class OwnerPaths:
    """The one layout of a managed owner's files under its run root."""

    run_root: Path
    owner_id: str

    @property
    def directory(self) -> Path:
        return self.run_root / _OWNER_DIR / self.owner_id

    @property
    def lease_path(self) -> Path:
        return self.directory / "lease.json"

    @property
    def status_path(self) -> Path:
        return self.directory / "status.json"

    @property
    def stdout_path(self) -> Path:
        return self.directory / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.directory / "stderr.log"

    def prepare(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)


def owner_paths(run_root: Path, owner_id: str) -> OwnerPaths:
    return OwnerPaths(run_root=run_root, owner_id=owner_id)


@dataclass(frozen=True)
class OwnedProcessSpec:
    """Everything the launcher pins about the owner it starts.

    ``env`` is the owner's COMPLETE environment block -- safe, non-secret
    values only (see the module docstring). It is passed verbatim on every
    backend; nothing merges the parent's environment into it afterwards.
    """

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    run_root: Path
    owner_id: str
    source_pin: str
    backend: str  # "launchd" | "detached"


@dataclass(frozen=True)
class OwnedProcessStatus:
    """One read of the owner's status file; absent/corrupt reads are None."""

    owner_id: str
    phase: str
    pid: int | None
    birth: str | None
    backend: str | None
    job_label: str | None
    updated_ms: int
    outcome_path: str | None = None
    detail: str | None = None
    receipt: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OwnedProcessLease:
    """A started owner: identify it, read its status, detach -- never stop it.

    ``detach`` is a DISCONNECT, not a stop. There is deliberately no ``stop``
    on this type: releasing an owner is the owner's own close-out (or, for a
    failed START that never admitted, ``start_owned_process``'s own cleanup),
    and an API that could kill a live owner would be the exact foot-gun this
    module exists to remove.
    """

    owner_id: str
    backend: str
    job_label: str | None
    pid: int
    birth_scheme: str | None
    birth: str | None
    source_pin: str
    run_root: Path
    _detached: bool = False

    @property
    def paths(self) -> OwnerPaths:
        return owner_paths(self.run_root, self.owner_id)

    async def wait(self, timeout_s: float) -> OwnedProcessStatus | None:
        """Bounded-wait for a status whose ``owner_id`` matches this lease.

        Returns the LAST status seen (never raises on malformed reads: a
        half-written file is "no status yet", the same reading the writer's
        atomic replace is designed to make rare). ``None`` means no matching
        status appeared inside the bound.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        last: OwnedProcessStatus | None = None
        while True:
            status = read_owner_status(self.paths.status_path)
            if status is not None and status.owner_id == self.owner_id:
                last = status
                if status.phase in TERMINAL_PHASES or status.phase == PHASE_PROVISIONAL:
                    return status
            if time.monotonic() >= deadline:
                return last
            await _sleep(_STATUS_POLL_S)

    def detach(self) -> None:
        """Disconnect from the owner without stopping it (documented no-op).

        Held so callers can SAY the disconnect their exit means, and so a
        future backend that holds a handle has one place to release it.
        """
        self._detached = True


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Write ``payload`` to ``path`` atomically with ``mode``.

    Same-directory temp + ``os.replace``: readers (the frontend polling the
    status, the owner reading the lease) must never observe a half-written
    file, and a same-directory rename is atomic on every platform this runs.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = directory / f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def write_owner_status(
    path: Path,
    *,
    owner_id: str,
    phase: str,
    pid: int | None,
    birth: str | None,
    backend: str | None,
    job_label: str | None,
    outcome_path: str | None = None,
    detail: str | None = None,
    receipt: Mapping[str, Any] | None = None,
) -> None:
    """Write one status snapshot (atomic, 0600, secret-free by construction).

    Callers treat failure as shrug-and-continue: the status file is the
    frontend's reading, never the record of authority -- the RecordSink and
    the lease/descriptor carry that -- so a full disk must not be allowed to
    change what the owner does with its resources.
    """
    payload = {
        "version": 1,
        "owner_id": owner_id,
        "phase": phase,
        "pid": pid,
        "birth": birth,
        "backend": backend,
        "job_label": job_label,
        "updated_ms": int(time.time() * 1000),
        "outcome_path": outcome_path,
        "detail": detail,
        "receipt": dict(receipt or {}),
    }
    _atomic_write_bytes(path, (json.dumps(payload, sort_keys=True) + "\n").encode())


def read_owner_status(path: Path) -> OwnedProcessStatus | None:
    """Parse a status file; ``None`` for absent/corrupt/unreadable/non-dict."""
    try:
        parsed = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    owner_id = parsed.get("owner_id")
    phase = parsed.get("phase")
    if not isinstance(owner_id, str) or not isinstance(phase, str):
        return None
    receipt = parsed.get("receipt")
    pid = parsed.get("pid")
    return OwnedProcessStatus(
        owner_id=owner_id,
        phase=phase,
        pid=pid if isinstance(pid, int) else None,
        birth=parsed.get("birth") if isinstance(parsed.get("birth"), str) else None,
        backend=parsed.get("backend") if isinstance(parsed.get("backend"), str) else None,
        job_label=(parsed.get("job_label") if isinstance(parsed.get("job_label"), str) else None),
        updated_ms=(
            int(parsed["updated_ms"]) if isinstance(parsed.get("updated_ms"), (int, float)) else 0
        ),
        outcome_path=(
            parsed.get("outcome_path") if isinstance(parsed.get("outcome_path"), str) else None
        ),
        detail=parsed.get("detail") if isinstance(parsed.get("detail"), str) else None,
        receipt=receipt if isinstance(receipt, dict) else {},
    )


class OwnerStatusChannel:
    """The owner's write side: a status file plus its own fixed identity.

    The owner constructs one from its verified lease facts and calls
    :meth:`phase` at each lifecycle edge. Every write is best-effort: a failed
    status write is a missing READING for the frontend, never a reason to
    change resource handling (the whole point of the retained join is that it
    does not depend on any writable surface).
    """

    def __init__(
        self,
        path: Path,
        *,
        owner_id: str,
        backend: str,
        job_label: str | None,
        pid: int | None = None,
        birth: str | None = None,
    ) -> None:
        self.path = path
        self.owner_id = owner_id
        self.backend = backend
        self.job_label = job_label
        self.pid = pid if pid is not None else os.getpid()
        self.birth = birth if birth is not None else procstate.self_birth_token()

    def phase(
        self,
        phase: str,
        *,
        outcome_path: str | None = None,
        detail: str | None = None,
        receipt: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            write_owner_status(
                self.path,
                owner_id=self.owner_id,
                phase=phase,
                pid=self.pid,
                birth=self.birth,
                backend=self.backend,
                job_label=self.job_label,
                outcome_path=outcome_path,
                detail=detail,
                receipt=receipt,
            )
        except Exception:  # noqa: BLE001 - see the class docstring
            # Status is a reading, never authority: a failed write may cost the
            # frontend a poll, and must never change what happens to the owned
            # work.
            pass


def _write_lease(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(dict(payload), sort_keys=True) + "\n").encode())


def _read_lease(path: Path) -> dict[str, Any] | None:
    try:
        parsed = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _lease_payload(
    spec: OwnedProcessSpec,
    *,
    job_label: str | None,
    pid: int | None,
    birth_scheme: str | None,
    birth: str | None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "owner_id": spec.owner_id,
        "backend": spec.backend,
        "job_label": job_label,
        "pid": pid,
        "birth_scheme": birth_scheme,
        "birth": birth,
        "source_pin": spec.source_pin,
        "run_root": str(spec.run_root),
        "created_ms": int(time.time() * 1000),
    }


def job_label_for(owner_id: str) -> str:
    """The per-owner LaunchAgent label. Unique per episode by construction."""
    return f"com.damian.lop-episode.{owner_id}"


#: Every ``launchctl`` call is BOUNDED. The startup wait promises a real
#: deadline, and an ``launchctl`` that hangs reopens the unbounded wait under
#: it -- the appear loop's own deadline cannot fire while it is blocked inside
#: one call (R4 review). On expiry the synthetic rc 124 (``timeout(1)``'s own
#: convention) reads as "the supervisor did not answer", which is the same
#: answer the helpers already give a refusal, so no caller mistakes a hung
#: call for a loaded job.
_LAUNCHCTL_TIMEOUT_S = 5.0


def launchctl_runner(*args: str) -> "subprocess.CompletedProcess[str]":
    """The runner shape :mod:`local_operator.launchd`'s helpers expect.

    ``launchd``'s ``_call`` invokes ``runner(*launchctl_args)`` -- NOT a
    ``subprocess.run`` call with kwargs -- so passing ``subprocess.run``
    directly answers ``TypeError: bufsize must be an integer`` (measured:
    bootstrap failed after 10 attempts against a working launchctl). One
    spelling for every caller in this module and the frontend -- and one
    bound (see ``_LAUNCHCTL_TIMEOUT_S``) so the startup deadline is real.
    """
    argv = ["launchctl", *args]
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=_LAUNCHCTL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            argv, 124, "", f"launchctl timed out after {_LAUNCHCTL_TIMEOUT_S:g}s"
        )


class OwnershipGuarantee(str, Enum):
    """Caller requirement; evidence is derived, never supplied as a capability.

    A genuine detached PID/source/birth lease identifies a process but cannot
    retain a production SDK host under a supervisor. Keep that weak generic
    use unchanged; callers needing managed lifetime must ask for the stronger
    proof at this SAME admission boundary, not rely on public CLI flag syntax.
    """

    IDENTIFIED_PROCESS = "identified_process"
    MANAGED_RETENTION = "managed_retention"


def verify_owned_admission(
    *,
    run_root: Path,
    owner_id: str,
    backend: str,
    job_label: str | None,
    actual_source_pin: str,
    runner: Any = launchctl_runner,
    required_guarantee: OwnershipGuarantee = OwnershipGuarantee.IDENTIFIED_PROCESS,
) -> dict[str, Any]:
    """The OWNER-side admission check: identity from the record, not a flag.

    Runs inside the would-be owner, before anything is prepared or allocated.
    Every check is fail-closed with its own sentence; a caller that cannot
    state its identity must not execute an episode. The comparisons, in
    order: the environment's owner id matches the lease record; this process
    IS the recorded holder of its own pid (birth token -- a reused number
    reads as a STALE record, not this run); the backends agree; the recorded
    source pin equals the running tree's pin (the caller computes it -- the
    evaluation layer owns that derivation, this module must not import it);
    and on launchd, launchd itself reports the job running under THIS pid.

    Returns the parsed lease (provenance for the status channel). Raises
    :class:`OwnedProcessRefused` with the remedy named on any mismatch.
    """
    if not isinstance(required_guarantee, OwnershipGuarantee):
        raise OwnedProcessRefused("unknown required ownership guarantee; refusing")
    paths = owner_paths(run_root, owner_id)
    lease = _read_lease(paths.lease_path)
    if lease is None:
        raise OwnedProcessRefused(
            f"no readable lease record at {paths.lease_path}: --owned-host runs only "
            "as the process a managed lease started; start it through the frontend "
            "(scripts/run_episode.py without --owned-host) or use the rig's "
            "--in-process-host mode"
        )
    recorded_owner = lease.get("owner_id")
    if recorded_owner != owner_id:
        raise OwnedProcessRefused(
            f"the lease record names owner {recorded_owner!r}, not {owner_id!r}; "
            "refusing a forged or mismatched owner identity"
        )
    recorded_backend = lease.get("backend")
    if recorded_backend != backend:
        raise OwnedProcessRefused(
            f"the lease record's backend {recorded_backend!r} disagrees with "
            f"{backend!r}; refusing"
        )
    recorded_pin = lease.get("source_pin")
    if recorded_pin != actual_source_pin:
        raise OwnedProcessRefused(
            f"the lease was written for source pin {recorded_pin!r} but this tree is "
            f"{actual_source_pin!r}; rebuild the selector/launcher against the tree "
            "that will run the episode"
        )
    # The bind happens after spawn (the parent cannot know the pid first), so
    # wait for it before comparing. A record that never binds is a failed
    # start, not an admission.
    deadline = time.monotonic() + _LEASE_BIND_TIMEOUT_S
    pid = lease.get("pid")
    while pid is None and time.monotonic() < deadline:
        time.sleep(_LEASE_BIND_POLL_S)
        refreshed = _read_lease(paths.lease_path)
        if refreshed is not None:
            lease = refreshed
            pid = lease.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise OwnedProcessRefused(
            f"the lease record at {paths.lease_path} never bound a pid; the start "
            "failed or the frontend died mid-launch -- refusing"
        )
    if pid != os.getpid():
        raise OwnedProcessRefused(
            f"the lease binds pid {pid}, but this process is {os.getpid()}; refusing"
        )
    birth_scheme = lease.get("birth_scheme")
    birth = lease.get("birth")
    if procstate.same_birth(birth_scheme, birth, pid) is not True:
        # False: the pid is held by a different process than the writer (the
        # number was reused). None: the question could not be asked. Both are
        # doubt, and doubt is not admission.
        raise OwnedProcessRefused(
            f"the lease's birth token does not identify this process at pid {pid} "
            "(a reused pid or an unreadable process table); refusing"
        )
    if backend == "launchd" or required_guarantee is OwnershipGuarantee.MANAGED_RETENTION:
        if not job_label:
            raise OwnedProcessRefused(
                "managed retention requires an actual supervisor registration; "
                "an identified detached process is not sufficient"
            )
        from local_operator import launchd

        home = launchd.real_home()
        if home is None:
            raise OwnedProcessRefused(
                "the real passwd home cannot be read, so the launchd job's identity "
                "is unverifiable; refusing"
            )
        plist_path = home.joinpath("Library", "LaunchAgents", f"{job_label}.plist")
        observed = launchd.observe_job(label=job_label, path=plist_path, run=runner)
        running_pid = observed.pid
        if observed.state is not launchd.JobState.LIVE or running_pid != os.getpid():
            raise OwnedProcessRefused(
                f"launchd does not report job {job_label} running under this pid "
                f"({running_pid!r} != {os.getpid()}); a --owned-host flag alone does "
                "not admit a process"
            )
        if required_guarantee is OwnershipGuarantee.MANAGED_RETENTION:
            if observed.loaded_no_restart is not True:
                # Disk bytes can be restored AFTER bootstrap without changing
                # launchd's loaded KeepAlive/trigger policy. The one supervisor
                # observation must independently certify no automatic restart;
                # missing/unsupported policy is uncertainty, never admission.
                raise OwnedProcessRefused("loaded supervisor no-restart policy is unverified")
            # Match the real owned registration to this lease/source; backend
            # tags and a requested guarantee are NOT evidence. The utility's
            # no-restart contract is KeepAlive=False with no trigger keys.
            # This proves managed retention, not arbitrary split-tree death.
            import plistlib

            try:
                plist = plistlib.loads(plist_path.read_bytes())
                env = plist.get("EnvironmentVariables", {})
                matches = (
                    job_label == job_label_for(owner_id)
                    and plist.get("Label") == job_label
                    and plist.get("KeepAlive") is False
                    and env.get(ENV_OWNER_ID) == owner_id
                    and env.get(ENV_SOURCE_PIN) == actual_source_pin
                    and env.get(ENV_LEASE_PATH) == str(paths.lease_path)
                    and not any(
                        key in plist
                        for key in (
                            "StartInterval",
                            "StartCalendarInterval",
                            "WatchPaths",
                            "QueueDirectories",
                            "StartOnMount",
                            "OtherJobEnabled",
                        )
                    )
                )
            except (OSError, ValueError, TypeError, AttributeError):
                matches = False
            if not matches:
                raise OwnedProcessRefused(
                    "managed retention registration/source/no-restart proof is unverified"
                )
    return lease


def _refuse_spec(spec: OwnedProcessSpec) -> None:
    if not spec.argv:
        raise OwnedProcessRefused("the owner's argv is empty; refusing to start nothing")
    if not _OWNER_ID_RE.match(spec.owner_id):
        raise OwnedProcessRefused(
            f"owner id {spec.owner_id!r} is not a safe nonce "
            "(letters/digits/._- only, 4..64 chars); refusing"
        )
    if spec.backend not in ("launchd", "detached"):
        raise OwnedProcessRefused(
            f"unknown owner backend {spec.backend!r}; supported: launchd, detached"
        )
    if spec.backend == "launchd":
        from local_operator.evaluation.runner.durable_root import refuse_volatile_root

        refuse_volatile_root(spec.run_root, label="owner run root")


def start_owned_process(
    spec: OwnedProcessSpec,
    *,
    startup_timeout_s: float,
) -> OwnedProcessLease:
    """Start ONE owner under a managed lease; verify it is actually running.

    The start is fail-closed and its own cleanup is scoped to EXACTLY what
    this call created: ``_start_launchd`` retires only the plist and the
    registration THIS attempt wrote and loaded, and a collision with another
    run's plist is refused before anything is written and left untouched
    (R3-S1 -- the previous shape let the shared exception handler boot out and
    unlink a job this call never created; measured). The detached branch can
    fail only before a child exists. On success the returned lease is a
    read/detach handle -- see the class docstring for why there is no stop.

    SECRETS NEVER CROSS THIS BOUNDARY: there is no handoff file and no
    secret-bearing field in the spec (R3-S3), so no plaintext artifact exists
    for a failed, interrupted or crashed start to leave behind. The owner
    resolves its own refs from the sealed credential store through the
    config root it inherits.
    """
    _refuse_spec(spec)

    # The agent-shell policy FIRST, before anything is written: starting a
    # session-creating process from inside an agent shell is refused by the
    # CLI's own inventory (one fact, one place -- see sdk._guard_agent_shell,
    # which this mirrors for the launch path). The refusal message is
    # exec_session_refusal's own.
    from local_operator.agent_shell import exec_session_refusal

    refusal = exec_session_refusal()
    if refusal:
        raise OwnedProcessRefused(refusal)

    paths = owner_paths(spec.run_root, spec.owner_id)
    paths.prepare()

    job_label = job_label_for(spec.owner_id) if spec.backend == "launchd" else None
    # PENDING lease first: the owner may boot before the parent knows the pid,
    # and it waits for the bind (verify_owned_admission). An unwritten lease
    # would be indistinguishable from a forged invocation.
    # Exclusive creation is the ownership fence: a repeated owner id may not
    # rewrite another attempt's lease before discovering its plist collision.
    pending = _lease_payload(spec, job_label=job_label, pid=None, birth_scheme=None, birth=None)
    try:
        with paths.lease_path.open("xb") as stream:
            os.chmod(paths.lease_path, 0o600)
            stream.write((json.dumps(pending, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise OwnedProcessRefused("owner lease already exists; choose a fresh owner id") from error

    env = dict(spec.env)
    env[ENV_OWNER_ID] = spec.owner_id
    env[ENV_BACKEND] = spec.backend
    env[ENV_SOURCE_PIN] = spec.source_pin
    env[ENV_LEASE_PATH] = str(paths.lease_path)
    if job_label is not None:
        env[ENV_JOB_LABEL] = job_label

    try:
        if spec.backend == "launchd":
            pid = _start_launchd(
                spec,
                env=env,
                job_label=job_label or "",
                paths=paths,
                appear_timeout_s=startup_timeout_s,
            )
        else:
            pid = _start_detached(spec, env=env, paths=paths)
    except OwnedProcessRefused:
        # ``_start_launchd`` has already retired exactly what it created
        # (its own plist and load); nothing here may touch another run's job
        # (R3-S1).
        raise
    except Exception as error:  # noqa: BLE001 - convert to the module's refusal type
        raise OwnedProcessRefused(
            f"the owner failed to start ({type(error).__name__}: {error}); nothing was "
            "left running"
        ) from error

    birth_scheme = procstate.birth_scheme()
    birth = procstate.birth_token(pid)
    _write_lease(
        paths.lease_path,
        _lease_payload(spec, job_label=job_label, pid=pid, birth_scheme=birth_scheme, birth=birth),
    )
    return OwnedProcessLease(
        owner_id=spec.owner_id,
        backend=spec.backend,
        job_label=job_label,
        pid=pid,
        birth_scheme=birth_scheme,
        birth=birth,
        source_pin=spec.source_pin,
        run_root=spec.run_root,
        _detached=spec.backend == "detached",
    )


def _start_detached(spec: OwnedProcessSpec, *, env: Mapping[str, str], paths: OwnerPaths) -> int:
    """Start the owner as a detached POSIX child; return its pid.

    The child's stdio goes to files under the owner directory (the host's
    outcome JSON on stdout is a debugging surface; the canonical result is the
    outcome FILE in the run root), never to the frontend's terminal: the
    frontend may exit while the owner keeps writing.
    """
    stdout = open(paths.stdout_path, "wb")  # noqa: SIM115 - handed to the child
    stderr = open(paths.stderr_path, "wb")  # noqa: SIM115 - handed to the child
    try:
        process = subprocess.Popen(
            list(spec.argv),
            cwd=str(spec.cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            **procstate.detached_popen_kwargs(),
        )
    finally:
        stdout.close()
        stderr.close()
    return process.pid


def _start_launchd(
    spec: OwnedProcessSpec,
    *,
    env: Mapping[str, str],
    job_label: str,
    paths: OwnerPaths,
    appear_timeout_s: float = _OWNER_APPEAR_TIMEOUT_S,
) -> int:
    """Install + load the per-owner LaunchAgent; return the pid launchd runs.

    The plist is written to the REAL home's LaunchAgents (uid-derived, never
    ``$HOME``) because that is where launchd reads it and what the ownership
    guard addresses; ``$HOME`` in the plist's environment is the EPISODE's
    scratch home, which is a different fact. ``RunAtLoad`` starts the job on
    bootstrap; ``KeepAlive=false`` guarantees a crash is never relaunched.

    FAILURE IS SELF-SCOPED (R3-S1). This call retires ONLY the plist it wrote
    and the load it performed; a collision with an existing plist is refused
    BEFORE anything is written and is never touched -- the previous shape
    left that decision to a shared handler which booted out and unlinked the
    OTHER run's registration (measured: an existing job's pid went away and
    its plist was removed). ``_retire_launchd_attempt`` is the one teardown
    and it is reachable only from here, after the write.
    """
    import plistlib

    from local_operator import launchd

    home = launchd.real_home()
    if home is None:
        raise OwnedProcessRefused(
            "launchd is unavailable (no readable passwd home); use --owner-backend "
            "detached only in rigs/tests, never for a paid run"
        )
    agents = home.joinpath("Library", "LaunchAgents")
    agents.mkdir(parents=True, exist_ok=True)
    plist_path = agents / f"{job_label}.plist"
    if plist_path.exists():
        # A session-unique label colliding with an existing plist means a
        # previous run of THIS owner id leaked; do not stomp it silently --
        # and do not clean it up either: this call did not create it.
        raise OwnedProcessRefused(
            f"{plist_path} already exists for owner {spec.owner_id}; refusing to "
            "overwrite another run's job (boot it out or choose a fresh owner id)"
        )
    existing_job = launchctl_runner("print", f"{launchd.job_domain()}/{job_label}")
    if existing_job.returncode == 0:
        raise OwnedProcessRefused("owner registration already exists; choose a fresh owner id")
    if existing_job.returncode != 113:
        raise OwnedProcessRefused("launchd absence is unverified; no owner start attempted")
    plist = {
        "Label": job_label,
        "ProgramArguments": list(spec.argv),
        "WorkingDirectory": str(spec.cwd),
        "EnvironmentVariables": {key: str(value) for key, value in env.items()},
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardOutPath": str(paths.stdout_path),
        "StandardErrorPath": str(paths.stderr_path),
    }
    created = False
    registered = False
    payload = plistlib.dumps(plist)
    try:
        # Never atomically REPLACE a colliding file: exclusive creation leaves
        # a racing attempt untouched. Only this successfully-created inode is
        # eligible for failed-start retirement, while the lease is still
        # positively fenced PENDING (the owner cannot pass admission yet).
        with plist_path.open("xb") as stream:
            created = True
            os.chmod(plist_path, 0o600)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        loaded = launchctl_runner("bootstrap", launchd.job_domain(), str(plist_path))
        if loaded.returncode != 0:
            # A timeout/error can hide a successful start. Keep this attempt's
            # pending lease/plist UNKNOWN; never boot out a registration we did
            # not positively create (including a racing label collision).
            raise OwnedProcessRefused("owner bootstrap is unverified; pending lease retained")
        registered = True
        # The caller's admission bound is the budget: waiting longer than the
        # frontend will wait only burns the episode's start window.
        appear_budget = min(_OWNER_APPEAR_TIMEOUT_S, max(0.5, appear_timeout_s))
        deadline = time.monotonic() + appear_budget
        while True:
            pid = launchd.job_pid(label=job_label, path=plist_path, run=launchctl_runner)
            if pid is not None:
                return pid
            if time.monotonic() >= deadline:
                break
            time.sleep(_OWNER_APPEAR_POLL_S)
        raise OwnedProcessRefused(
            f"launchd loaded {job_label} but reported no pid within {appear_budget:g}s"
        )
    except BaseException:
        if created and registered:
            pending = _read_lease(paths.lease_path)
            if pending is not None and pending.get("pid") is None:
                _retire_launchd_attempt(
                    job_label=job_label, plist_path=plist_path, expected_bytes=payload
                )
        raise


def _retire_launchd_attempt(*, job_label: str, plist_path: Path, expected_bytes: bytes) -> None:
    """Retire EXACTLY the plist and load THIS attempt created -- nothing else.

    Called only from ``_start_launchd``'s failure paths, after the plist was
    written. The plist is unlinked first (that is the half which guarantees no
    later reload) and the load is then booted out (a job that never loaded
    boots out as a no-op refusal). Never called for a collision: that refusal
    happens before any write, and the other run's job must survive to be
    inspected or reused.
    """
    try:
        if plist_path.read_bytes() != expected_bytes:
            return
        plist_path.unlink()
    except OSError:
        return
    try:
        from local_operator import launchd

        launchd.bootout_job(label=job_label, path=plist_path, runner=launchctl_runner)
    except Exception:  # noqa: BLE001 - a failed start must not raise on cleanup
        pass


def retire_owned_job(*, job_label: str) -> None:
    """The OWNER's own terminal retirement: unregister its job, remove its plist.

    WHY THE OWNER DOES IT (R3-F3): the frontend's post-terminal cleanup runs
    only while the frontend is still alive at the terminal edge; when the
    frontend exited provisionally first, the job stayed registered and the
    plist installed with nothing in the pipeline to reap them (measured:
    stable at t+75 s, in the operator's real LaunchAgents). The owner has just
    sealed its outcome and flushed when this runs, so it is the one actor
    guaranteed to be alive at the moment the job is genuinely finished.

    ORDER MATTERS, and the order is measured on this host: the plist is
    unlinked FIRST -- it is what prevents any later reload -- and the bootout
    follows. ``launchctl bootout`` of one's own job kills the caller as its
    first effect, so code after it may never run (observed: the calling shell
    died inside the call); the unregistration itself still happens, launchd
    answering ``print`` with rc 113 and no plist left behind. Best-effort by
    contract: this runs on the close-out path and must never change an outcome.
    """
    try:
        from local_operator import launchd

        home = launchd.real_home()
        if home is None:
            return
        owner_id = os.environ.get(ENV_OWNER_ID)
        if not owner_id or job_label != job_label_for(owner_id):
            return
        # The same verified lease/source/current process proof as admission,
        # not a caller-supplied label or the mere existence of a plist.
        import plistlib

        plist_path = home.joinpath("Library", "LaunchAgents", f"{job_label}.plist")
        payload = plistlib.loads(plist_path.read_bytes())
        env = payload.get("EnvironmentVariables", {})
        if env.get(ENV_OWNER_ID) != owner_id:
            return
        # Run roots are caller-selected, so the exact lease location travels
        # in a NONSECRET environment field, never guessed from scratch layout.
        lease_path = os.environ.get(ENV_LEASE_PATH)
        if not lease_path:
            return
        lease = _read_lease(Path(lease_path))
        if lease is None or lease.get("owner_id") != owner_id or lease.get("pid") != os.getpid():
            return
        if lease.get("backend") != "launchd" or lease.get("job_label") != job_label:
            return
        if lease.get("source_pin") != os.environ.get(ENV_SOURCE_PIN):
            return
        if env.get(ENV_LEASE_PATH) != lease_path or env.get(ENV_SOURCE_PIN) != lease.get(
            "source_pin"
        ):
            return
        if (
            procstate.same_birth(lease.get("birth_scheme"), lease.get("birth"), os.getpid())
            is not True
        ):
            return
        if launchd.job_pid(label=job_label, path=plist_path, run=launchctl_runner) != os.getpid():
            return
        plist_path.unlink()
        launchd.bootout_job(label=job_label, path=plist_path, runner=launchctl_runner)
    except Exception:  # noqa: BLE001 - a terminal cleanup must not fail the close-out
        pass


def argv_digest(argv: tuple[str, ...]) -> str:
    """Stable provenance digest for a pinned argv (record/audit only)."""
    return hashlib.sha256("\0".join(argv).encode()).hexdigest()


__all__ = [
    "ENV_BACKEND",
    "ENV_JOB_LABEL",
    "ENV_OWNER_ID",
    "ENV_SOURCE_PIN",
    "OwnerPaths",
    "OwnerStatusChannel",
    "OwnedProcessError",
    "OwnedProcessLease",
    "OwnedProcessRefused",
    "OwnedProcessSpec",
    "OwnedProcessStatus",
    "OwnershipGuarantee",
    "PHASE_ADMITTED",
    "PHASE_CLOSING",
    "PHASE_PROVISIONAL",
    "PHASE_RUNNING",
    "PHASE_STARTING",
    "TERMINAL_PHASES",
    "argv_digest",
    "job_label_for",
    "launchctl_runner",
    "owner_paths",
    "read_owner_status",
    "retire_owned_job",
    "start_owned_process",
    "verify_owned_admission",
    "write_owner_status",
]
