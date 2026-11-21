"""Roll the secret broker onto a new build WITHOUT a credential gap.

**Why this exists (measured on node2, 2026-11).** The broker is spawned on
demand and exits on its idle timer — but ``SecretBroker._should_idle_out``
never fires while ANY session is registered, so on a node with a long-lived
serve runtime the broker never idles out. node2 sat on 0.102.2 for 7+ days
while every update reported it "pending … it moves when it idles out", which
was false there. And the only manual route, ``lop secret broker restart``, was
stop-then-start: a window with no broker, every registration dropped, and —
because ``SessionRegistration`` did not reconnect — every live session's
descendants permanently unregistered. In the keyfile tier
``access.retrieve_secret`` then serves a denied descendant through the
UNNOTIFIED local decrypt, so a naive restart was a §6 redaction regression,
not only an availability blip.

**How a roll works** (``roll_broker``; each step names the security
constraint it carries):

1. Take ``broker.lock`` non-blocking — the same lock ``ensure_broker`` spawns
   under — so no third broker appears and a second roll defers (R4).
2. Preflight with no side effects: ``status`` (pre-gate metadata only), the
   generation question, the TIER RULE (roll only when nothing would need the
   passphrase again), the CAPABILITY GATE (every registered session must be
   able to re-register; otherwise defer and NAME the pids — R7), and a
   SETTLING check (a broker still holding sessions from its own roll defers a
   second one until they re-register — QA Q1).
3. Write the ``broker.rolling`` sentinel (old pid + deadline, no secret) so a
   retrieval that meets the hand-off waits instead of falling back (R1).
4. Start the successor BESIDE the live broker on ``broker.next`` from THIS
   interpreter (the new generation, in the update's refresh child). The old
   broker hands it nothing but session pids and start times (R2).
5. Readiness on the temp path: pid, protocol, lock state, install prefix and
   version must all match (R5). Any mismatch kills the successor; the old
   broker never noticed anything.
6. Verify the temp socket (0600, ours, a socket, in the 0700 dir) and that the
   canonical path is still the SAME socket inode answered by the SAME pid
   (R4), and that the old broker has no session the successor does not know
   (S1: one registered after the successor's snapshot aborts the roll). Re-arm
   the sentinel for the successor's hold, then ``rename`` atomically. This is
   the commit point: the old broker stays authoritative until here.
7. Mark the sentinel ``signalled``, then SIGTERM the old broker after
   re-checking its argv and start time (pid-reuse guard). If the roller dies
   before that, the successor does it (CR3). Its ``stop()`` drains in-flight
   requests before closing session channels, so no notify/ack is cut (R3);
   each session sees EOF and registers
   afresh with the successor, which HOLDS (never grants) their descendants'
   requests meanwhile (``broker.ROLL_GRACE_S``).
8. Exactly one ``roll`` audit row: the successor writes it on success; the
   roller writes the ``fail`` row on failure (keyfile tier — a locked
   passphrase store has no chain to append to; the caller's report is the
   record there) (R6).

**Local only (R6).** Nothing on the broker's wire protocol starts a roll; the
successor switches are passed only on a local spawn by a process holding the
lock. Mesh nodes each roll their own broker from their own update child.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

logger = logging.getLogger(__name__)

RollStatus = Literal["rolled", "already", "deferred", "failed"]

#: How long the successor gets to answer on its temp path. The same bound the
#: lazy start gives a fresh broker (``client.STARTUP_TIMEOUT_S``).
READY_TIMEOUT_S = 5.0

#: How long the old broker gets to exit after SIGTERM. Mirrors
#: ``handlers._BROKER_STOP_TIMEOUT_S``: its ``stop()`` joins in-flight workers
#: (≤5 s each by its own bound) before closing session channels.
OLD_EXIT_TIMEOUT_S = 10.0

#: Lifetime of the roll sentinel. Comfortably above the roll's worst case
#: (5 s readiness + 10 s old exit) so a live roll never looks finished, and
#: short enough that a sentinel left by a roller that died stops making
#: retrievals wait within a minute.
SENTINEL_TTL_S = 60.0

_BROKER_MODULE = "local_operator.secrets.brokerd"


@dataclass(frozen=True)
class RollResult:
    """What a roll did. Never an exception: every caller renders it.

    ``code`` is the machine-readable cause behind ``deferred``/``failed``/
    ``already`` (``busy``, ``passphrase``, ``tier-unknown``, ``sessions``,
    ``unreadable``, ``no-broker``, ``current``, ``readiness``, ``switch``,
    ``spawn``). ``stranded`` names the session pids behind a ``sessions``
    deferral, so the report can say exactly who has to restart.
    """

    status: RollStatus
    reason: str = ""
    code: str = ""
    old_pid: int | None = None
    new_pid: int | None = None
    warning: str = ""
    stranded: tuple[int, ...] = field(default_factory=tuple)


def _argv_of(pid: int) -> str:
    """The process's command line, or ``""`` when it cannot be read."""
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return ""
        return raw.replace(b"\0", b" ").decode("utf-8", "replace")
    import subprocess

    try:
        done = subprocess.run(
            ["ps", "-ww", "-o", "args=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


def _reaped(pid: int) -> bool:
    """Has our own child ``pid`` exited? Reaps it if so; ``False`` for a non-child."""
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        from local_operator.secrets.peer import process_info

        return process_info(pid) is None
    except OSError:
        return False
    return done == pid


def _kill_successor(pid: int, temp: Path) -> None:
    """Stop a successor whose hand-off failed, by exact pid, and clear its temp path.

    SIGTERM first so it unlinks its own socket; SIGKILL only if it ignores that.
    The temp socket is removed if anything is left there — the roller holds
    ``broker.lock`` and the canonical socket was never touched, so nothing
    else can own that name.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not _reaped(pid):
        time.sleep(0.05)
    if not _reaped(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not _reaped(pid):
            time.sleep(0.05)
    temp.unlink(missing_ok=True)


def _status_at(path: Path, operation: str = "status") -> dict[str, object]:
    """One pre-gate request on a SPECIFIC socket path (the successor's temp one)."""
    from local_operator.secrets import client

    return client.request(operation, connection=client._connect(path))


def _check_private_socket(path: Path) -> str | None:
    """Why ``path`` is not a 0600 socket of ours in a 0700 dir of ours, else ``None`` (R4)."""
    try:
        info = os.lstat(path)
        parent = os.lstat(path.parent)
    except OSError as exc:
        return f"{path} vanished before the hand-off: {exc}"
    # The roll only runs where `fcntl` exists (POSIX), but the guard is spelled
    # here so the check stands on its own and `xplat_probe`'s static scan can
    # see it: `os.getuid` does not exist on Windows. Refusing is the safe answer.
    if not hasattr(os, "getuid"):
        return "this platform cannot verify socket ownership"
    uid = os.getuid()
    if not stat.S_ISSOCK(info.st_mode):
        return f"{path} is not a socket"
    if info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o600:
        return f"{path} is not a 0600 socket owned by this user"
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != uid:
        return f"{path.parent} is not a directory owned by this user"
    if stat.S_IMODE(parent.st_mode) != 0o700:
        return f"{path.parent} is not mode 0700"
    return None


def _record_failure(base: Path | None, old_pid: int, reason: str) -> None:
    """The ``roll``/``fail`` audit row, written by the roller (keyfile tier only).

    Only the keyfile tier has a key this process may read without the broker;
    a passphrase-tier store's chain is unreachable here, and the caller's
    report is the record. Best-effort: losing the row must not mask the
    failure it describes. No secret name or value is part of ``reason``.
    """
    try:
        from local_operator.secrets.keys import (
            key_of_record_is_plaintext,
            load_master_key,
        )
        from local_operator.secrets.store import SecretStore

        if not key_of_record_is_plaintext(base):
            return
        SecretStore(load_master_key(base), base=base).record_broker_event(
            event="roll", outcome=f"fail:{reason}", pid=old_pid
        )
    except Exception:  # noqa: BLE001 - audit must never mask the failure itself
        logger.debug("could not record the failed roll", exc_info=True)


def _write_sentinel(
    path: Path, old_pid: int, *, ttl: float = SENTINEL_TTL_S, signalled: bool = False
) -> None:
    """Create the 0600 roll sentinel; ``O_NOFOLLOW`` so a planted link is refused.

    Written three times on a successful roll: before anything moves (``ttl`` =
    :data:`SENTINEL_TTL_S`), just before the rename (``ttl`` = the successor's
    hold, security S1) and once more before the SIGTERM with ``signalled``
    set (review CR3: the successor reads it to know the roller did its part).
    ``O_TRUNC`` on one fd, so a reader sees either the old or the new payload
    or, at worst, an empty file it treats as "not rolling" for one poll.
    """
    payload = json.dumps(
        {"old_pid": old_pid, "deadline": time.time() + ttl, "signalled": signalled}
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    handle = os.open(path, flags, 0o600)
    try:
        os.write(handle, payload.encode("utf-8"))
    finally:
        os.close(handle)


def roller_signalled(base: Path | None, old_pid: int) -> bool:
    """Did the roller record that it is signalling ``old_pid``? (review CR3)

    Read by the SUCCESSOR, which retires its predecessor itself only when the
    roller died first. Anything unreadable counts as NOT signalled: the
    successor's other checks (roller gone, predecessor still the pinned broker
    process) still have to hold before it sends anything.
    """
    from local_operator.secrets.protocol import rolling_path

    try:
        payload = json.loads(rolling_path(base).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("old_pid") == old_pid
        and payload.get("signalled") is True
    )


def verified_broker_process(identity: object) -> bool:
    """Is ``identity``'s pid still THAT process, and still a broker? (pid-reuse guard)

    The one check both the roller and a successor make before sending SIGTERM
    to an old broker: same pid AND start time (``same_process_as``), and an
    argv that runs :data:`_BROKER_MODULE`. Anything unreadable is ``False`` —
    not signalling is always the safe failure.
    """
    from local_operator.secrets.peer import process_info

    pid = getattr(identity, "pid", None)
    if not isinstance(pid, int):
        return False
    current = process_info(pid)
    if current is None or not current.same_process_as(identity):  # type: ignore[arg-type]
        return False
    return _BROKER_MODULE in _argv_of(pid)


def _unknown_to_successor(
    old_status: dict[str, object], new_status: dict[str, object]
) -> list[int]:
    """Sessions the old broker has that the successor neither holds nor knows (S1).

    The successor read its predecessor's sessions ONCE, at startup
    (``broker._successor_inputs``), and ``ensure_broker`` takes no lock to
    register, so a session can register with the old broker after that
    snapshot. Unknown to the successor, its descendants would be DENIED (not
    held) after the rename. Compared by pid AND start time.
    """

    def pinned(status: dict[str, object], *keys: str) -> set[tuple[object, object]]:
        found: set[tuple[object, object]] = set()
        for key in keys:
            entries = status.get(key)
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and isinstance(entry.get("pid"), int):
                        found.add((entry["pid"], entry.get("start_time")))
        return found

    from local_operator.secrets.peer import process_info

    old = pinned(old_status, "session_lineage", "expected_lineage")
    new = pinned(new_status, "session_lineage", "expected_lineage")
    if "session_lineage" not in old_status:
        # S7: a broker released before the roll reports only bare ``sessions``
        # pids, and the FIRST roll after an update always starts from one; a
        # lineage-only compare saw nothing there and the re-read was a no-op.
        # Its pids carry no start time (``None``), so they are matched on pid
        # against the successor's lineage, and the start-time check below is
        # skipped for them — a recycled pid can only make this abort a roll the
        # next attempt retries, never let one through.
        known = {pid for pid, _start in new}
        legacy = old_status.get("sessions")
        old = {
            (pid, None)
            for pid in (legacy if isinstance(legacy, list) else [])
            if isinstance(pid, int) and pid not in known
        }
        new = set()
    late = []
    for pid, start in old - new:
        # A session that EXITED after the successor's snapshot is still listed
        # by the old broker until its watcher's next tick (≤1 s); it has no
        # descendants left to protect, so it must not abort the roll (measured:
        # a back-to-back `restart` right after a session closed failed on it).
        current = process_info(pid)  # type: ignore[arg-type]
        if current is None or (start is not None and current.start_time != start):
            continue
        late.append(pid)
    return sorted(late)  # type: ignore[type-var]


def stranded_sessions_reason(pids: tuple[int, ...]) -> str:
    """The deferral sentence for sessions that cannot re-register (79's addition (a)).

    Names every pid and says plainly when the roll will happen. It offers NO
    restart route (security R7): restarting the broker under these sessions is
    exactly what unregisters them into the unnotified fallback.
    """
    listed = ", ".join(str(pid) for pid in pids)
    noun = "session runs" if len(pids) == 1 else "sessions run"
    return (
        f"{len(pids)} registered {noun} a build that cannot re-register with a new "
        f"secret broker (pid {listed}); the broker will roll once these sessions restart"
    )


def roll_broker(
    base: Path | None = None,
    *,
    force: bool = False,
    generation_stale: Callable[[int], bool | None] | None = None,
    spawn: Callable[[Path | None, tuple[str, ...]], int] | None = None,
    ready_timeout: float = READY_TIMEOUT_S,
    before_switch: Callable[[], None] | None = None,
) -> RollResult:
    """Replace the running broker with one from THIS interpreter's build. Never raises.

    ``force`` skips the "is the running broker on an older generation" question
    (``lop secret broker restart`` asks for a roll regardless of build); every
    SAFETY rule still applies. ``generation_stale``, ``spawn`` and
    ``before_switch`` are seams for tests; production passes none of them.
    """
    try:
        return _roll(
            base,
            force=force,
            generation_stale=generation_stale,
            spawn=spawn,
            ready_timeout=ready_timeout,
            before_switch=before_switch,
        )
    except Exception as exc:  # noqa: BLE001 - a roll reports; it never raises into an update
        logger.debug("broker roll failed unexpectedly", exc_info=True)
        return RollResult("failed", f"unexpected error: {exc}", code="error")


def _roll(
    base: Path | None,
    *,
    force: bool,
    generation_stale: Callable[[int], bool | None] | None,
    spawn: Callable[[Path | None, tuple[str, ...]], int] | None,
    ready_timeout: float,
    before_switch: Callable[[], None] | None,
) -> RollResult:
    try:
        import fcntl
    except ImportError:
        return RollResult("already", "this platform runs no secret broker", code="no-broker")
    from local_operator.secrets import client
    from local_operator.secrets.errors import BrokerIncompatible, SecretStoreError
    from local_operator.secrets.protocol import (
        PROTOCOL_VERSION,
        lock_path,
        rolling_path,
        socket_path,
        successor_socket_path,
    )

    canonical = socket_path(base)
    if not canonical.parent.is_dir():
        return RollResult("already", "no secret broker is running", code="no-broker")
    handle = os.open(lock_path(base), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return RollResult(
                "deferred",
                "another secret broker start or roll is in progress",
                code="busy",
            )
        return _roll_locked(
            base,
            canonical=canonical,
            temp=successor_socket_path(base),
            sentinel=rolling_path(base),
            protocol=PROTOCOL_VERSION,
            force=force,
            generation_stale=generation_stale,
            spawn=spawn or client._spawn_broker,
            ready_timeout=ready_timeout,
            before_switch=before_switch,
            incompatible=BrokerIncompatible,
            store_error=SecretStoreError,
        )
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


def _roll_locked(
    base: Path | None,
    *,
    canonical: Path,
    temp: Path,
    sentinel: Path,
    protocol: int,
    force: bool,
    generation_stale: Callable[[int], bool | None] | None,
    spawn: Callable[[Path | None, tuple[str, ...]], int],
    ready_timeout: float,
    before_switch: Callable[[], None] | None,
    incompatible: type[Exception],
    store_error: type[Exception],
) -> RollResult:
    from local_operator.secrets import client
    from local_operator.secrets.broker import _install_version
    from local_operator.secrets.keys import key_mode, key_of_record_is_plaintext
    from local_operator.secrets.peer import process_info

    # --- 2. preflight: no side effects -------------------------------------
    try:
        status = client.request("status", base)
    except incompatible as exc:
        return RollResult(
            "deferred",
            # CR4: a real remedy, since `restart` refuses this case too.
            f"the running secret broker speaks another protocol, so its sessions cannot be read: "
            f"{exc}. Stop it with `lop secret broker stop` once the sessions registered with it "
            "have restarted; the next secret use starts a current broker",
            code="unreadable",
        )
    except store_error:
        return RollResult("already", "no secret broker is running", code="no-broker")
    old_pid = status.get("pid")
    if not isinstance(old_pid, int):
        return RollResult("deferred", "the running broker did not report a pid", code="unreadable")
    old_identity = process_info(old_pid)
    if old_identity is None:
        return RollResult("already", "no secret broker is running", code="no-broker")
    try:
        canonical_info = os.lstat(canonical)
    except OSError:
        return RollResult("already", "no secret broker is running", code="no-broker")
    if not stat.S_ISSOCK(canonical_info.st_mode):
        return RollResult(
            "deferred", f"{canonical} is not a socket; not touching it", code="unreadable"
        )

    def stale_of(pid: int) -> bool | None:
        if generation_stale is not None:
            return generation_stale(pid)
        from local_operator import update

        return update.generation_staleness_of_process(pid)

    if not force and stale_of(old_pid) is not True:
        return RollResult(
            "already", "the secret broker runs the current build", code="current", old_pid=old_pid
        )

    # The TIER RULE: roll only when nothing would need the passphrase again. A
    # locked broker holds no key and no grants, so nothing is lost; a keyfile
    # store (by the VALIDATING predicate — a planted master.key must not flip
    # this, review R4-1) lets the successor load its own key from disk. An
    # unlocked passphrase-tier broker holds the only unwrapped key: replacing it
    # would demand the passphrase again, which an update must never prompt for.
    locked = status.get("locked") is True
    if not locked:
        try:
            plaintext = key_of_record_is_plaintext(base)
        except Exception:  # noqa: BLE001 - fail closed: treat as hardened
            plaintext = False
        if not plaintext:
            if key_mode(base) == "passphrase":
                return RollResult(
                    "deferred",
                    "the secret store is hardened and unlocked; a new broker would start "
                    "locked and ask for the passphrase again, so the broker is left running "
                    "until it is locked or exits",
                    code="passphrase",
                    old_pid=old_pid,
                )
            return RollResult(
                "deferred",
                "could not confirm that the store's key of record is on disk, so the "
                "broker is left running rather than risk a credential gap",
                code="tier-unknown",
                old_pid=old_pid,
            )

    # The CAPABILITY GATE: a session that cannot re-register would stay
    # unregistered for life after the roll, and its descendants would meet the
    # unnotified fallback. A broker that reports caps answers for itself; for an
    # older broker a session counts as capable only if its process provably
    # runs the current generation (which carries the re-registration code).
    # QA Q1: a broker that is itself a fresh successor still HOLDS sessions
    # that have not re-registered with it yet. Rolling again now would hand the
    # next successor a snapshot without them; wait for them instead. Bounded by
    # the broker's own hold (`settling` ends with ROLL_GRACE_S), so a session
    # that never comes back delays the next roll by seconds, not forever.
    held = status.get("expected_lineage")
    if status.get("settling") is True and isinstance(held, list) and held:
        return RollResult(
            "deferred",
            "the secret broker was just replaced and its sessions are still re-registering; "
            "try again in a few seconds",
            code="settling",
            old_pid=old_pid,
        )
    lineage = status.get("session_lineage")
    sessions: list[tuple[int, tuple[object, ...]]]
    if isinstance(lineage, list):
        sessions = [
            (pid, tuple(entry.get("caps") or ()))
            for entry in [*lineage, *(held if isinstance(held, list) else [])]
            if isinstance(entry, dict) and isinstance(pid := entry.get("pid"), int)
        ]
    else:
        sessions = [(pid, ()) for pid in status.get("sessions") or () if isinstance(pid, int)]
    stranded = tuple(
        sorted(pid for pid, caps in sessions if "reroll" not in caps and stale_of(pid) is not False)
    )
    if stranded:
        return RollResult(
            "deferred",
            stranded_sessions_reason(stranded),
            code="sessions",
            old_pid=old_pid,
            stranded=stranded,
        )

    # --- 3. sentinel, 4. successor, 5. readiness ----------------------------
    _write_sentinel(sentinel, old_pid)
    result: RollResult | None = None
    try:
        result = _hand_off(
            base,
            canonical=canonical,
            sentinel=sentinel,
            canonical_inode=canonical_info.st_ino,
            temp=temp,
            protocol=protocol,
            old_pid=old_pid,
            old_identity=old_identity,
            locked=locked,
            version=_install_version(),
            spawn=spawn,
            ready_timeout=ready_timeout,
            before_switch=before_switch,
            store_error=store_error,
        )
        return result
    finally:
        # SECURITY S1: on success the sentinel is NOT removed. `_hand_off`
        # re-wrote it at the rename with deadline = rename + ROLL_GRACE_S, the
        # window in which sessions are still re-registering; deleting it here
        # (as soon as the old broker exited) ended the client's retry before
        # they had, and a denial then reached the unnotified fallback. It
        # expires by its own deadline (`client.rolling_in_progress`). Any
        # other outcome removes it: nothing moved, so nothing needs waiting out.
        if result is None or result.status != "rolled":
            sentinel.unlink(missing_ok=True)


def _hand_off(
    base: Path | None,
    *,
    canonical: Path,
    canonical_inode: int,
    sentinel: Path,
    temp: Path,
    protocol: int,
    old_pid: int,
    old_identity: object,
    locked: bool,
    version: str,
    spawn: Callable[[Path | None, tuple[str, ...]], int],
    ready_timeout: float,
    before_switch: Callable[[], None] | None,
    store_error: type[Exception],
) -> RollResult:
    from local_operator.secrets import client
    from local_operator.secrets.broker import ROLL_GRACE_S

    def fail(new_pid: int | None, reason: str, code: str) -> RollResult:
        if new_pid is not None:
            _kill_successor(new_pid, temp)
        _record_failure(base, old_pid, reason)
        return RollResult("failed", reason, code=code, old_pid=old_pid, new_pid=new_pid)

    try:
        new_pid = spawn(base, ("--bind", str(temp), "--successor-of", str(old_pid)))
    except Exception as exc:  # noqa: BLE001 - reported, old broker untouched
        return fail(None, f"could not start a new broker: {exc}", "spawn")

    deadline = time.monotonic() + ready_timeout
    last = "it never answered"
    ready = False
    while time.monotonic() < deadline:
        if _reaped(new_pid):
            # QA minor: an early exit is its own cause (commonly: the canonical
            # socket was no longer answered by the old broker), not slowness.
            reason = "the new broker exited before answering on its temporary socket"
            return fail(new_pid, reason, "readiness")
        try:
            ping = _status_at(temp, "ping")
            fresh = _status_at(temp, "status")
        except (store_error, OSError) as exc:
            last = str(exc)
            time.sleep(0.05)
            continue
        # R5: the successor must be the process WE started, speak our protocol,
        # hold the same lock state, and run the same install as this roller.
        mismatches = []
        if ping.get("pid") != new_pid or fresh.get("pid") != new_pid:
            mismatches.append(f"pid {ping.get('pid')} is not the spawned {new_pid}")
        if fresh.get("protocol") != protocol:
            mismatches.append(f"protocol {fresh.get('protocol')!r} is not {protocol}")
        if (fresh.get("locked") is True) != locked:
            mismatches.append(f"locked={fresh.get('locked')!r}, expected {locked}")
        if fresh.get("prefix") != sys.prefix:
            mismatches.append(f"install {fresh.get('prefix')!r} is not {sys.prefix!r}")
        if fresh.get("version") != version:
            mismatches.append(f"version {fresh.get('version')!r} is not {version!r}")
        if mismatches:
            reason = "the new broker failed readiness: " + "; ".join(mismatches)
            return fail(new_pid, reason, "readiness")
        ready = True
        break
    if not ready:
        reason = f"the new broker was not ready within {ready_timeout:g}s: {last}"
        return fail(new_pid, reason, "readiness")

    if before_switch is not None:
        before_switch()

    # --- 6. verify, then the atomic switch (the commit point) ---------------
    problem = _check_private_socket(temp)
    if problem is None:
        try:
            info = os.lstat(canonical)
            if not stat.S_ISSOCK(info.st_mode) or info.st_ino != canonical_inode:
                problem = f"{canonical} changed during the roll"
        except OSError as exc:
            problem = f"{canonical} vanished during the roll: {exc}"
    if problem is None:
        try:
            old_status = client.request("status", base)
            if old_status.get("pid") != old_pid:
                problem = "a different broker answers the canonical socket"
            else:
                # SECURITY S1, second half: re-read the old broker's sessions
                # right before the commit point and abort if any is unknown to
                # the successor (it registered after the successor's one-time
                # snapshot). Aborting is free: the old broker never noticed
                # anything, and the next roll snapshots it.
                late = _unknown_to_successor(old_status, _status_at(temp, "status"))
                if late:
                    problem = (
                        "session(s) registered with the old broker during the roll (pid "
                        + ", ".join(str(pid) for pid in late)
                        + "); the next roll will include them"
                    )
        except (store_error, OSError) as exc:
            problem = f"the old broker stopped answering during the roll: {exc}"
    if problem is not None:
        return fail(new_pid, problem, "switch")
    # S1: from the rename the successor holds for ROLL_GRACE_S; the client keeps
    # retrying a denial or a silence for exactly as long as this sentinel lives.
    _write_sentinel(sentinel, old_pid, ttl=ROLL_GRACE_S)
    try:
        os.rename(temp, canonical)
    except OSError as exc:
        return fail(new_pid, f"could not move the new broker into place: {exc}", "switch")

    # --- 7. the old broker exits ------------------------------------------
    # From here the successor is authoritative and the old broker is not
    # resurrectable, so a problem is a WARNING on a rolled result, never a
    # kill -9 and never a rollback.
    from local_operator.secrets.peer import process_info

    warning = ""
    current = process_info(old_pid)
    if current is None:
        pass
    elif not verified_broker_process(old_identity):
        warning = f"pid {old_pid} is no longer the old broker; nothing was signalled"
    else:
        # CR3: record the intent BEFORE the signal, so a successor that sees
        # this roller gone never sends a second SIGTERM into the old broker's
        # draining stop(). Dying between this write and the kill leaves the
        # same orphan as before CR3, over a window of one syscall.
        try:
            _write_sentinel(sentinel, old_pid, ttl=ROLL_GRACE_S, signalled=True)
        except OSError:
            pass
        try:
            os.kill(old_pid, signal.SIGTERM)
        except OSError as exc:
            warning = f"could not signal the old broker (pid {old_pid}): {exc}"
        else:
            exit_deadline = time.monotonic() + OLD_EXIT_TIMEOUT_S
            while time.monotonic() < exit_deadline:
                now = process_info(old_pid)
                if (
                    now is None
                    or not now.same_process_as(old_identity)  # type: ignore[arg-type]
                    or _reaped(old_pid)
                ):
                    # S6: re-arm the sentinel from the moment the old broker is
                    # GONE, not from the rename. A session that registered with
                    # it in the re-read→rename gap (or sat in its listen
                    # backlog) is unknown to the successor and only re-registers
                    # after this exit, i.e. after the old broker's drain; with
                    # the window counted from the rename, a slow drain let the
                    # sentinel expire first and its descendants' denials stood.
                    try:
                        _write_sentinel(sentinel, old_pid, ttl=ROLL_GRACE_S, signalled=True)
                    except OSError:
                        pass
                    break
                time.sleep(0.05)
            else:
                warning = (
                    f"the old broker (pid {old_pid}) did not exit within "
                    f"{OLD_EXIT_TIMEOUT_S:g}s; it no longer receives new connections"
                )
    return RollResult(
        "rolled",
        f"secret broker moved from pid {old_pid} to pid {new_pid}",
        code="rolled",
        old_pid=old_pid,
        new_pid=new_pid,
        warning=warning,
    )


__all__ = ["RollResult", "roll_broker", "stranded_sessions_reason"]
