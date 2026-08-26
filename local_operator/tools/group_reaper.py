"""Reap orphaned process groups whose owning ``lop`` process died hard.

Every shell command the bash tool runs is spawned ``start_new_session=True``
(`builtin.py`), so each command becomes its own session + process group, and
``execute_bash`` kills that group on every in-process stop path via
``_kill()`` -> ``os.killpg(...)``. Those paths — foreground timeout, real
abort, steering-detach, background-job cleanup, the final reap after a normal
exit — are complete and correct for any death the owning process is alive to
observe.

The one leak they cannot cover is a **HARD death of the owning ``lop``
process**: a SIGKILL when cmux replaces a session, an OOM kill, a crash. No
in-process code runs, so ``_kill()`` never fires. Because ``start_new_session``
already detached the group from the controlling terminal, the group receives no
SIGHUP either; it reparents to launchd/init (pid 1) and runs forever. This was
observed in the wild as a ``pyright`` group still alive ten hours after its
owner died.

This module closes that hole the same way :mod:`local_operator.session.retention`
closes the analogous "hard-killed session left an empty directory behind" hole
one level up: a liveness marker written at spawn, swept at the next startup, and
reaped **only when the OWNER is provably dead**.

The whole point — the invariant every line here defends — is in
:mod:`retention` too, and it is worth restating because a mistake here KILLS a
process rather than removing an empty directory:

    Reap ONLY genuine waste; NEVER a legitimate long-running command whose
    owning session is still alive.

A ten-hour ML trainer and a ten-hour stuck ``pyright`` are **identical** by
runtime, CPU%, and idle time, so those signals are BANNED as inputs to the
decision (see the ``ts`` field, which is diagnostics-only and which no branch in
this module may read). The one reliable signal is **owner liveness**: a group is
waste iff the ``lop`` process that spawned it is dead. That check delegates to
:func:`retention._process_alive` verbatim — the liveness probe is shared truth,
and a second copy could drift from the one retention already trusts.

Deliberately NOT folded into :mod:`retention`: retention's contract is "never
delete a transcript"; this module's contract is "kill a process group". Merging
a killer into the never-delete module would blur the single guarantee that file
exists to make. This module imports ``_process_alive`` from retention and
nothing else.

POSIX-only by nature. The leak is POSIX-specific (``start_new_session``
orphaning); ``os.killpg`` / ``os.getpgid`` do not exist on Windows, and
``_process_alive`` cannot probe there, so every entry point early-returns a
no-op on win32 and nothing is ever registered or reaped.

Out of scope, on purpose: ``exec_mode --background`` workers. Those spawn
``python -m local_operator.exec_worker`` as a SEPARATE detached process (not via
``execute_bash``), so they never get a ledger line — and they should not: they
are deliberate fire-and-forget workers with their own SIGTERM lifecycle and
their own log file. AsyncJobManager-owned background bash jobs, by contrast, DO
go through ``execute_bash`` and so are registered like any command, keyed by the
same owner (the ``lop`` process); they are reaped only when that owner dies,
which is exactly correct — a detached background job outliving its owner is
waste, because the sink its output and completion were destined for died with
the owner.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from local_operator.session.retention import _process_alive

logger = logging.getLogger(__name__)

#: Directory under the config dir holding the per-owner ledgers. 0700 like the
#: rest of the config dir's private state; created lazily on first register.
PROC_GROUPS_DIRNAME = "proc-groups"

#: This module's view of the platform, a module-local copy for the same reason
#: retention keeps one (``_PLATFORM``): a test steers the Windows branch by
#: patching THIS name rather than the global ``sys.platform``, which would tell
#: every other thread in the process it is on Windows for the duration — and
#: this suite runs with threads.
_PLATFORM = sys.platform

#: Whether this platform can spawn/kill process groups at all. ``os.killpg`` and
#: ``os.getpgid`` are POSIX-only, and ``_process_alive`` cannot probe on
#: Windows, so the whole mechanism no-ops there (mirrors retention's
#: ``_LIVENESS_IS_VERIFIABLE``). Named rather than inlined so every entry point
#: reads the same decision.
_REAPING_IS_SUPPORTED = _PLATFORM != "win32"

#: First N chars of the command, stored for the LOG LINE only. Never a decision
#: input. Bounded so a pathological one-line command cannot bloat the ledger.
_CMD_LOG_CHARS = 120


@dataclass(frozen=True)
class ReaperSweepResult:
    """What one hard-death sweep did.

    Returned rather than only logged so a test (and a future benchmark) can
    assert the outcome without scraping log output, exactly as retention's
    ``SweepResult`` is used.
    """

    scanned_ledgers: int = 0
    reaped_groups: int = 0
    skipped_live_owners: int = 0
    errors: int = 0


def _owner_start_token(pid: int) -> str | None:
    """The opaque process-start token for ``pid``, or ``None`` if unreadable.

    ``ps -o lstart=`` prints a process's start time in a fixed, locale-stable
    form under ``LC_ALL=C`` (verified identical on macOS and Linux, e.g.
    ``Wed Aug 26 11:38:17 2026``). It needs no ``/proc`` (macOS has none) and no
    psutil (not installed), which is why it is used here instead of a
    platform-specific probe.

    The token is treated as **opaque**: stored and compared as a raw string,
    NEVER parsed into a datetime. That is a safety property, not laziness — a
    locale quirk or a ``ps`` format surprise can then only make two tokens
    *differ* (which errs toward NOT reaping, the safe direction), never make two
    genuinely-different processes' tokens falsely *match* (which could reap a
    live victim).

    Returns ``None`` when the pid is gone or ``ps`` fails for any reason; every
    caller treats ``None`` as "cannot establish identity" and refuses to reap.
    """
    if pid <= 0:
        return None
    try:
        # LC_ALL=C pins the month/day names so the token is byte-stable across
        # machines and users; text mode + explicit decode keeps it a str.
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            env={**os.environ, "LC_ALL": "C"},
            check=False,
        )
    except (OSError, ValueError) as exc:  # ps missing, or bad argv
        logger.debug("group reaper: ps lstart failed for pid %s: %s", pid, exc)
        return None
    if completed.returncode != 0:
        # Non-zero almost always means "no such process" — the pid is gone.
        return None
    token = completed.stdout.strip()
    return token or None


#: Memoized start token for THIS process. The owner's own start token is
#: immutable for the life of the process, so the ``ps -o lstart=`` fork it costs
#: need only be paid once rather than on every register/unregister/kill of the
#: bash hot path (a session that spawns many short commands otherwise pays a
#: ``ps`` fork/exec per command just to re-derive a constant). ``None`` means
#: "not yet resolved" — a transient ``ps`` failure is NOT cached, so a later
#: call retries rather than poisoning the process with a permanent ``None``.
#:
#: Fork-staleness assumption: this memo is process-global, so a
#: fork-WITHOUT-exec child would inherit the PARENT's token under its own
#: (different) pid and read a stale identity. That is latent only and
#: unreachable on the install path — the reaper installs solely on the
#: interactive TUI / headless entry, which never forks without a following exec
#: (a fork+exec child re-imports this module and starts again at ``None``). If a
#: future entry point ever forks without exec, reset this in the child.
_SELF_START_TOKEN: str | None = None


def _self_start_token(pid: int) -> str | None:
    """The current process's own start token, cached; live probe for any other.

    Only ``os.getpid()``'s token is safe to memoize: it cannot change under a
    fixed pid. Any OTHER pid could be recycled between calls, so it must always
    be probed live (and no foreign value is ever stored here). This is the hot
    path — register/unregister/kill all call it — which is the whole reason the
    self case is cached rather than re-forking ``ps`` each time.
    """
    if pid != os.getpid():
        return _owner_start_token(pid)
    global _SELF_START_TOKEN
    if _SELF_START_TOKEN is None:
        _SELF_START_TOKEN = _owner_start_token(pid)
    return _SELF_START_TOKEN


def _ledger_path(config_dir: Path, owner_pid: int, owner_start: str) -> Path:
    """Path to the ledger for the owner identified by ``(pid, start)``.

    One file per OWNER (the ``lop`` process), keyed by its two-factor identity
    so the soft-death handler and the sweep can both answer "which groups belong
    to owner P?" with a single stat, and a dead owner's whole ledger can be
    removed in one unlink after its groups are handled. The start token is
    sanitised into the filename (spaces/colons -> ``-``) so it is a legal,
    single-path-segment name on every filesystem while staying unique per owner.
    """
    safe_start = "".join(c if c.isalnum() else "-" for c in owner_start)
    return config_dir / PROC_GROUPS_DIRNAME / f"{owner_pid}-{safe_start}.jsonl"


#: Suffix appended to a ledger's name to form its advisory-lock sidecar. A
#: single named constant so the lock CREATOR (``_ledger_lock``) and the lock
#: HUSK reaper (the startup sweep) can never disagree on the name and leave a
#: file one of them cannot see.
_LOCK_SUFFIX = ".lock"


def _lock_path(ledger_path: Path) -> Path:
    """The advisory-lock sidecar path for ``ledger_path``.

    ``_ledger_lock`` creates ``<ledger>.lock`` next to the ledger; the sweep
    removes that husk when the owner is proven dead. Both go through here so the
    name is defined in exactly one place.
    """
    return ledger_path.parent / (ledger_path.name + _LOCK_SUFFIX)


def _owner_pid_from_lock(lock_path: Path) -> int | None:
    """Parse the owner pid out of a ``<pid>-<start>.jsonl.lock`` husk name.

    The sweep needs the owner pid to ask ``_process_alive`` before removing an
    ORPHAN lock (one with no surviving ledger to read the identity from). Only
    the pid is recoverable: the start token was sanitised into the filename
    (non-alphanumerics collapsed to ``-``), so it cannot be reversed — which is
    exactly why the orphan-lock pass reaps ONLY on the pid-gone row of the owner
    truth table (no token needed) and never on the recycle row. ``None`` for any
    name whose leading segment is not an integer, so an unexpected file is left
    alone rather than guessed at.
    """
    prefix = lock_path.name.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def _resolve_config_dir(config_dir: Path | None) -> Path | None:
    """Resolve the config dir, ``None`` when the helper itself fails.

    ``config_dir=None`` means "ask :func:`paths.config_dir`", imported lazily so
    this module stays import-light and so a test that patched the env var is
    honoured on every call (the helper re-reads it each time).
    """
    if config_dir is not None:
        return config_dir
    try:
        from local_operator.paths import config_dir as app_config_dir

        return app_config_dir()
    except Exception as exc:  # noqa: BLE001 — best-effort; never break a command
        logger.debug("group reaper: cannot resolve config dir: %s", exc)
        return None


def register_group(
    pgid: int,
    cmd: str,
    *,
    config_dir: Path | None = None,
    owner_pid: int | None = None,
) -> None:
    """Append this group to the owner's ledger. Best-effort; never raises.

    Called at spawn, immediately after the pgid is known. Records the
    two-factor OWNER identity (this ``lop`` process's pid + start token) that
    the reap gate keys on, and the group leader's start token (the pgid-reuse
    defence — see :func:`sweep_orphan_groups`).

    Best-effort by design, exactly like ``retention.claim_session``: a marker
    that cannot be written (disk full, permission) must NEVER stop a command
    from running. The worst case of a missed marker is the pre-existing
    behaviour — one leaked group on a hard owner death.
    """
    if not _REAPING_IS_SUPPORTED:
        return
    resolved = _resolve_config_dir(config_dir)
    if resolved is None:
        return
    pid = os.getpid() if owner_pid is None else owner_pid
    owner_start = _self_start_token(pid)
    if owner_start is None:
        # Cannot establish our own identity -> a ledger keyed by it would be
        # un-reapable (the sweep needs the start token to disambiguate reuse).
        # Skip rather than write an unusable line.
        logger.debug("group reaper: no start token for self (pid %s); skip register", pid)
        return
    # The leader pid IS the pgid for a group created by start_new_session; its
    # start token is the pgid-reuse defence recorded at the one moment we know
    # the group is genuinely ours.
    grp_leader_start = _owner_start_token(pgid)
    entry = {
        "pgid": pgid,
        "owner_pid": pid,
        "owner_start": owner_start,
        "grp_leader_start": grp_leader_start,
        "cmd": cmd[:_CMD_LOG_CHARS],
        # Diagnostics ONLY. BANNED as a decision input (§2 of the design): a
        # stuck pyright and a live trainer are indistinguishable by age. No
        # branch in this module may read this field.
        "ts": _now(),
    }
    path = _ledger_path(resolved, pid, owner_start)
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Single append write of one JSON line. A POSIX append under PIPE_BUF
        # (this line is well under 512 bytes) is atomic, so parallel bash calls
        # from the same owner interleave cleanly at line boundaries without
        # rewriting the file — which is why the ledger is JSONL, not a
        # read-modify-write document. The SHARED ledger lock lets concurrent
        # appends still run together but blocks while a compacting unregister
        # holds the file EXCLUSIVE, so an append can never land inside (and be
        # lost by) that read-modify-rewrite window. See ``_ledger_lock``.
        with _ledger_lock(path, exclusive=False):
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.debug("group reaper: cannot register pgid %s: %s", pgid, exc)


def unregister_group(
    pgid: int,
    *,
    config_dir: Path | None = None,
    owner_pid: int | None = None,
) -> None:
    """Drop one pgid from THIS owner's ledger after a clean group death.

    Called on the normal-exit reap and the detached-job cleanup once the group
    is confirmed dead. Keeps a long host session's ledger from growing without
    bound and means a clean run leaves nothing for the sweep to consider.

    Best-effort: a missed unregister is harmless — the group is already dead, so
    both the soft-death and sweep paths find the leader gone and simply drop the
    line without killing anything.
    """
    if not _REAPING_IS_SUPPORTED:
        return
    resolved = _resolve_config_dir(config_dir)
    if resolved is None:
        return
    pid = os.getpid() if owner_pid is None else owner_pid
    owner_start = _self_start_token(pid)
    if owner_start is None:
        return
    path = _ledger_path(resolved, pid, owner_start)
    _rewrite_without_pgid(path, pgid)


@contextlib.contextmanager
def _ledger_lock(path: Path, *, exclusive: bool) -> Iterator[None]:
    """Advisory ``flock`` serialising ledger COMPACTION against APPENDS.

    Registration is a lock-free atomic append and unregistration a
    read-filter-truncate-rewrite. Without coordination a register append that
    lands between an unregister's read and its truncating write is silently
    lost, so that group never enters the ledger and a later hard owner-death
    never reaps it. The window is real here because parallel bash + background
    jobs from one owner is a first-class feature.

    The fix is a classic readers/writer lock keyed per owner ledger (each owner
    has its own file, so owners never contend): appends take it SHARED so they
    still proceed concurrently with each other (a POSIX sub-``PIPE_BUF`` append
    is already atomic between appenders and must not be serialised on the hot
    path), while the compacting unregister takes it EXCLUSIVE so no append is
    in flight while it truncates and rewrites.

    Best-effort: if the lock itself cannot be taken (permissions, an exotic FS
    without ``flock``) we DEGRADE to running unlocked rather than block or fail a
    command — the worst case is the pre-existing lost-append behaviour, never a
    wrong kill. POSIX only; reached solely after the ``_REAPING_IS_SUPPORTED``
    guard, so ``fcntl`` (absent on Windows) is imported lazily here.
    """
    fd: int | None = None
    try:
        try:
            import fcntl

            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            lock_path = _lock_path(path)
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        except (OSError, ImportError) as exc:
            # Degrade to unlocked: never block or break a command over the lock.
            logger.debug("group reaper: ledger lock unavailable for %s: %s", path, exc)
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
                fd = None
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError, ImportError):
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)


def _rewrite_without_pgid(path: Path, pgid: int) -> None:
    """Rewrite ``path`` keeping every line whose ``pgid`` differs. Best-effort.

    Reads the whole ledger, filters the one pgid, and writes it back under an
    EXCLUSIVE ledger lock so a concurrent register append cannot be lost between
    this read and this truncating write (see :func:`_ledger_lock`). Torn or
    unparseable lines are KEPT verbatim (never silently dropped by a filter that
    could not read them). If nothing remains, the file is unlinked so the
    owner's ledger disappears once its last group is accounted for.
    """
    with _ledger_lock(path, exclusive=True):
        _rewrite_without_pgid_locked(path, pgid)


def _rewrite_without_pgid_locked(path: Path, pgid: int) -> None:
    """The read-filter-rewrite body of :func:`_rewrite_without_pgid`.

    Split out so the exclusive-lock scope in the caller stays a thin wrapper and
    the filtering logic is testable directly. Callers MUST hold the ledger lock.
    """
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.debug("group reaper: cannot read ledger %s: %s", path, exc)
        return
    kept: list[str] = []
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            # Unparseable (possibly a torn concurrent append): keep it, since
            # dropping a line we could not read would lose a sibling's group.
            kept.append(line)
            continue
        if isinstance(obj, dict) and obj.get("pgid") == pgid:
            continue
        kept.append(line)
    try:
        if kept:
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("group reaper: cannot rewrite ledger %s: %s", path, exc)


def kill_own_groups(
    *,
    config_dir: Path | None = None,
    owner_pid: int | None = None,
) -> None:
    """SOFT-DEATH path: reap every still-live group THIS owner registered.

    Wired at the TUI/headless entry via ``atexit`` + a SIGTERM handler +
    the teardown ``finally``, so a catchable stop (the polite cmux stop, a
    launchd stop, a clean quit) reaps this process's groups precisely and
    instantly instead of waiting for the next process's startup sweep. NOT
    wired on SIGINT: a headless-REPL Ctrl-C is a turn abort that spares live
    ``background=true`` jobs, so reaping there would kill a live owner's groups
    (see ``_install_group_reaper_soft_death`` in ``cli.py``).

    **Scope guarantee**: this reads ONLY the current process's own ledger
    (``<config_dir>/proc-groups/<self_pid>-<self_start>.jsonl``). It never scans
    another owner's ledger, so a sibling ``lop``'s live groups are untouchable by
    construction — this can only ever kill groups THIS process spawned.

    Idempotent: a second call finds the ledger already unlinked and does
    nothing; a ``killpg`` on an already-dead group raises ``ProcessLookupError``,
    which is suppressed. POSIX only.
    """
    if not _REAPING_IS_SUPPORTED:
        return
    resolved = _resolve_config_dir(config_dir)
    if resolved is None:
        return
    pid = os.getpid() if owner_pid is None else owner_pid
    owner_start = _self_start_token(pid)
    if owner_start is None:
        return
    path = _ledger_path(resolved, pid, owner_start)
    try:
        entries = _read_ledger(path)
    except OSError:
        return
    for entry in entries:
        # This IS our own ledger, so the owner check is moot — but the
        # pgid-reuse (victim) gate still applies: between a group's death and
        # this call the pgid may have been recycled onto an unrelated live
        # group, and killing that would murder an innocent. Same guard as the
        # sweep.
        _reap_if_still_ours(entry)
    # Drop the whole ledger: this process is going away, so nothing more will be
    # appended to it, and everything in it has now been handled.
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def sweep_orphan_groups(
    config_dir: Path,
    *,
    self_pid: int | None = None,
) -> ReaperSweepResult:
    """HARD-DEATH path: reap groups whose owning ``lop`` process is provably dead.

    Runs at startup beside the retention sweep, off-loop, best-effort. Scans
    every owner's ledger under ``proc-groups/``; for each whose owner is dead
    (two-factor identity, below), kills every still-live registered group whose
    leader identity still matches, then unlinks the ledger. A LIVE owner's
    ledger is never touched — that is the live-trainer guarantee.

    OWNER identity is two-factor (``owner_pid`` + ``owner_start``), because a
    bare pid is forgeable: after the OS recycles a dead ``lop``'s pid onto an
    unrelated live process, a bare-pid probe reads "alive" and the sweep would
    refuse to reap a genuinely-orphaned group forever. The truth table:

        _process_alive(owner_pid) | start token | verdict
        --------------------------|-------------|--------------------------------
        False (pid gone)          | —           | owner dead        -> reap
        True                      | matches     | owner ALIVE       -> NEVER reap
        True                      | differs     | pid recycled/dead -> reap
        True                      | unreadable  | ambiguous         -> NEVER reap

    Every ambiguous cell resolves to NOT reaping.

    POSIX only; on Windows returns an empty result and reaps nothing (mirrors
    retention). Best-effort: a scan error is counted and swallowed so a sweep
    failure never blocks session start.
    """
    result = ReaperSweepResult()
    if not _REAPING_IS_SUPPORTED:
        return result
    me = os.getpid() if self_pid is None else self_pid
    groups_dir = config_dir / PROC_GROUPS_DIRNAME
    try:
        ledgers = sorted(groups_dir.glob("*.jsonl"))
    except OSError as exc:
        logger.debug("group reaper: cannot list %s: %s", groups_dir, exc)
        return result
    scanned = 0
    reaped = 0
    skipped_live = 0
    errors = 0
    for ledger in ledgers:
        scanned += 1
        try:
            entries = _read_ledger(ledger)
        except OSError as exc:
            # A wholly-unreadable ledger is left in place (err toward keeping
            # evidence), never blindly unlinked.
            logger.debug("group reaper: cannot read ledger %s: %s", ledger, exc)
            errors += 1
            continue
        if not entries:
            # Empty or all-torn ledger with nothing actionable: remove the shell
            # so the directory does not accumulate husks. Safe — no pgid to act
            # on.
            _safe_unlink(ledger)
            continue
        # Owner identity is taken from the FILE's first readable entry: every
        # line in one ledger shares the same owner by construction (it is keyed
        # by owner). If the owner is our own live pid, this is a sibling-safe
        # skip — but that only happens if a stale ledger from a PRIOR process
        # reused our pid, which the start-token check below still resolves
        # correctly.
        owner_pid = entries[0].get("owner_pid")
        owner_start = entries[0].get("owner_start")
        if not isinstance(owner_pid, int) or not isinstance(owner_start, str):
            errors += 1
            continue
        if owner_pid == me:
            # Never reap groups attributed to the sweeping process itself; the
            # soft-death path owns those.
            skipped_live += 1
            continue
        verdict = _owner_is_dead(owner_pid, owner_start)
        if verdict is None:
            # Ambiguous (pid alive, token unreadable) -> never reap.
            skipped_live += 1
            continue
        if not verdict:
            # Owner alive, token matches -> the live-trainer case. Leave the
            # ledger entirely untouched; it belongs to a running session.
            skipped_live += 1
            continue
        # Owner is provably dead: reap each still-ours group, then drop the
        # ledger AND its lock husk. Removing the lock here (rather than via a
        # separate liveness check) ties husk cleanup to the exact moment this
        # sweep has already PROVEN the owner dead — no new liveness logic, and
        # never a live owner's lock, since a live owner's ledger is skipped
        # above before this point is reached.
        for entry in entries:
            if _reap_if_still_ours(entry):
                reaped += 1
        _safe_unlink(ledger)
        _safe_unlink(_lock_path(ledger))
    # Second pass: reap ORPHAN lock husks — ``*.jsonl.lock`` files whose ledger
    # was already unlinked by a clean-exit compaction (unregister/kill_own_groups
    # remove the ledger but leave the lock sidecar behind) or by the loop above.
    # This is the husk source that actually accumulates: the common path is a
    # graceful shutdown, which never leaves a ledger for the loop above to key
    # on. Guarded on the SINGLE unambiguous owner-death signal recoverable from a
    # lock name alone — the pid is gone (``_process_alive`` false) — because the
    # start token cannot be reversed out of the sanitised filename. A lock whose
    # pid is still alive, or whose name is not pid-parseable, is left untouched:
    # every ambiguous case errs toward NOT touching a possibly-live owner's lock.
    errors += _sweep_orphan_locks(groups_dir)
    return ReaperSweepResult(
        scanned_ledgers=scanned,
        reaped_groups=reaped,
        skipped_live_owners=skipped_live,
        errors=errors,
    )


def _sweep_orphan_locks(groups_dir: Path) -> int:
    """Unlink ``*.jsonl.lock`` husks whose owner pid is provably gone.

    Called after the ledger loop so any lock paired with a still-present ledger
    has already been handled (a live owner's ledger — and thus its lock — was
    skipped; a dead owner's ledger and lock were removed together). What remains
    are ORPHAN locks: sidecars left behind when a clean shutdown compacted the
    ledger away but not its lock file. Reap one ONLY when its parsed owner pid is
    no longer a live process; a pid still alive, or a name that does not start
    with an integer, is left in place. Returns the count of errors swallowed so
    the caller can fold them into the sweep result. Best-effort throughout: a
    stray husk is cosmetic, never worth failing a session start over.
    """
    errors = 0
    try:
        locks = sorted(groups_dir.glob("*" + _LOCK_SUFFIX))
    except OSError as exc:
        logger.debug("group reaper: cannot list locks in %s: %s", groups_dir, exc)
        return errors
    for lock in locks:
        owner_pid = _owner_pid_from_lock(lock)
        if owner_pid is None:
            # Unparseable name — not ours to guess at. Leave it.
            continue
        if _process_alive(owner_pid):
            # Owner pid still live: this lock may belong to a running session's
            # ledger that simply has no groups registered right now. Never touch
            # a possibly-live owner's lock.
            continue
        _safe_unlink(lock)
    return errors


def _owner_is_dead(owner_pid: int, owner_start: str) -> bool | None:
    """Is the ledger's owner provably dead? ``None`` when it cannot be decided.

    Implements the owner-side truth table. Returns ``True`` (reap) only when the
    pid is gone OR the pid is alive but its start token differs (pid recycled
    onto a different process). Returns ``False`` (never reap) when the original
    owner is provably still alive. Returns ``None`` (never reap, ambiguous) when
    the pid is alive but its current start token is unreadable.
    """
    if not _process_alive(owner_pid):
        return True  # pid gone -> owner dead -> reap
    current = _owner_start_token(owner_pid)
    if current is None:
        return None  # alive but unreadable -> ambiguous -> never reap
    if current != owner_start:
        return True  # pid recycled onto a different process -> owner dead -> reap
    return False  # same pid, same start -> original owner ALIVE -> never reap


def _reap_if_still_ours(entry: dict[str, object]) -> bool:
    """killpg one registered group, but ONLY if it is still genuinely ours.

    The pgid-reuse (victim) gate: even with a dead owner, the pgid itself may
    have been recycled by the OS onto an unrelated live group between the
    owner's death and now. Killing a recycled pgid would murder an innocent
    process. Defence uses the group-leader start token recorded at spawn (the
    leader pid == the pgid):

        still_ours = _process_alive(pgid) and
                     _owner_start_token(pgid) == grp_leader_start

    - leader gone            -> the group already died; drop, no kill.
    - leader alive, differs  -> pgid recycled onto a new leader; drop, no kill.
    - leader alive, matches  -> genuinely our orphaned group; killpg.

    Returns ``True`` only when a live, still-ours group was actually signalled.
    """
    pgid = entry.get("pgid")
    if not isinstance(pgid, int) or pgid <= 0:
        return False
    grp_leader_start = entry.get("grp_leader_start")
    if not _process_alive(pgid):
        return False  # leader gone: group already dead, nothing to kill
    current_leader = _owner_start_token(pgid)
    if current_leader is None:
        # Alive but the leader's identity is unreadable -> ambiguous -> do NOT
        # kill (err toward not killing an innocent).
        return False
    if current_leader != grp_leader_start:
        # pgid recycled onto a different leader -> the original group is gone,
        # this is an innocent live group. Do NOT kill.
        return False
    try:
        os.killpg(pgid, _sigkill())
    except ProcessLookupError:
        # Died between the liveness check and the kill: idempotent, fine.
        return False
    except OSError as exc:
        logger.debug("group reaper: killpg %s failed: %s", pgid, exc)
        return False
    logger.info(
        "group reaper: reaped orphaned group pgid=%s cmd=%r",
        pgid,
        entry.get("cmd", ""),
    )
    return True


def _read_ledger(path: Path) -> list[dict[str, object]]:
    """Parse a ledger into a list of entry dicts, skipping torn/corrupt lines.

    A torn final line (a concurrent append still mid-write) or any otherwise
    unparseable line is SKIPPED, not fatal — the same posture retention takes
    with a bad claim marker. Raises ``OSError`` only when the file itself cannot
    be read (a missing file is not an error: returns ``[]``).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    entries: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue  # torn or corrupt line: skip, keep processing the rest
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _safe_unlink(path: Path) -> None:
    """Unlink ``path`` tolerating a concurrent sweep that already removed it."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("group reaper: cannot unlink ledger %s: %s", path, exc)


def _sigkill() -> int:
    """``signal.SIGKILL``, imported lazily to keep this module import-light."""
    import signal

    return signal.SIGKILL


def _now() -> float:
    """Wall clock for the diagnostics-only ``ts`` field. Never a decision input."""
    import time

    return time.time()
