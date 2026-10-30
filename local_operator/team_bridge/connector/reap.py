"""Reap a connector from an older build that is still holding this lock.

THE DEFECT THIS ANSWERS (measured 2026-10-30). A connector from generation
``20261023T142851Z-0.86.1`` (pid 48311, ppid 1, seven days old, idle, no sockets)
held ``<config>/team_bridge/connector.lock``, so the supervised job
``com.local-operator.bridge-connector`` was refused ``ConnectorLockBusy`` 2,826
times and exited 75 every time. Nothing inbound was admitted for days. The
process was frozen — it answered ``ps`` and nothing else — so no graceful exit
was ever coming, and ``flock`` is only released when the holder dies.

WHO MAY ACT, and the direction is REFUSAL. Every proof below must hold; the
first one that does not is logged as a closed reason and the caller exits 75
exactly as it does today. A wrong reap stops someone's working receiver, so the
asymmetry is deliberate: a missed reap costs another supervised start, a wrong
one costs a live process.

1. **The waiter is provably current.** ``generation_staleness_of_process`` on our
   own pid must be ``False`` — both sides read and equal. ``True`` means we are
   ourselves superseded, ``None`` means we could not read our own argv or the
   pointer; neither case may judge another process.
2. **Exactly one holder, and it is not us.** Linux parses ``/proc/locks`` for the
   inode's ``FLOCK`` rows; macOS asks ``lsof -F p`` (measured 0.55 s on this
   host). No ``lsof``, a timeout, a non-zero exit that still printed something,
   unparseable output, zero holders or more than one all refuse.
3. **Same account, same store, and it is a connector.** Same uid as the waiter,
   ``-m local_operator.team_bridge.connector`` in its argv, and its
   ``--config-dir`` resolving to the SAME directory as ours. Another user's lock
   is not ours to touch, and neither is another config dir's — the singleton is
   per store, and the label is per user.
4. **The holder's build is provably older.** ``generation_staleness_of_process``
   on the holder must be ``True``: the generation was read AND differs from
   ``current``. ``None`` (no generation in its argv — a pip/pipx install, or an
   unreadable pointer) and ``False`` both refuse. A live current-generation
   connector is NEVER signalled, whatever it is doing.
5. **The identity is re-checked immediately before every signal.** The holder's
   start time is recorded with the proofs and compared again before SIGTERM and
   before any SIGKILL, because pids are reused and the process we proved about
   may not be the process we would signal.

WHAT IS DELIBERATELY NOT A PROOF: a socket check. A healthy connector idles in
its own backoff with no sockets open (the loop's ≤60 s pause), so "no sockets"
would select live connectors — architect decision D1, and the reason admission
liveness is never consulted here.

THE ACTION is a stop, not an eviction: SIGTERM first, which is what launchd sends
on a bootout and what the connector's own graceful handler expects
(``STOP_GRACE_S + STOP_CANCEL_S`` ≈ 10 s). SIGKILL follows ONLY when the holder is
still alive after ``STOP_WAIT_S``, the pid and start time still match, AND its
cumulative CPU time grew by less than ``CPU_FLAT_S`` since the SIGTERM sample —
i.e. it is wedged, which is the state the incident's holder was in. A holder that
is busy burning CPU is not wedged and is given its graceful window instead.

AT MOST ONE ATTEMPT PER PROCESS START: the only caller is the ``ConnectorLockBusy``
arm of ``__main__._main_async``, immediately before it exits 75.

Every probe forks (``ps``, ``lsof``), so every probe runs in the default executor
and never on the event loop: the waiter's own SIGTERM must stay responsive while
it waits. The seams at the bottom (``_read_process``, ``_lock_holder_pids``,
``_signal``, ``_pause``, ``_clock``) are module functions so a unit test can
replace them and needs no real signalling and no real holder.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from local_operator.team_bridge.connector.lock import CONNECTOR_LOCK_NAME

logger = logging.getLogger("local_operator.team_bridge.connector")

#: The module the holder's argv must name, matched as a whole word: an argv that
#: says ``-m local_operator.team_bridge.connector_v2`` is not this connector.
_MODULE_PATTERN = re.compile(r"(?:^|\s)-m\s+local_operator\.team_bridge\.connector(?=\s|$)")

#: How long the holder is given to exit after SIGTERM before it is a candidate
#: for SIGKILL. STOP_GRACE_S + STOP_CANCEL_S is ≈10 s; the extra margin covers a
#: SQLite busy wait inside the in-flight iteration.
STOP_WAIT_S = 20.0

#: The poll between liveness reads while waiting, and the liveness read's own
#: bound is ``PROBE_TIMEOUT_S``.
POLL_S = 0.25

#: Cumulative CPU growth below which a holder that ignored SIGTERM is treated as
#: WEDGED rather than busy. The incident holder was frozen; a connector doing
#: work either exits or accrues far more than this in twenty seconds.
CPU_FLAT_S = 0.1

#: Every probe is bounded: this runs in a process that must not hang on an
#: unresponsive kernel table.
PROBE_TIMEOUT_S = 5.0

#: The closed reasons a reap refuses, in the order the proofs are asked. Named
#: rather than sentence-shaped because each one is a log FIELD (content-free,
#: greppable) and the count of ways to be wrong is what the tests parametrize.
Refusal = Literal[
    "platform",
    "waiter_generation_unproven",
    "holders_unknown",
    "holder_count_not_one",
    "holder_unreadable",
    "different_uid",
    "not_a_connector",
    "different_config",
    "holder_generation_unproven",
    "holder_identity_moved",
    "signal_refused",
]


@dataclass(frozen=True)
class ReapOutcome:
    """What one attempt did. ``reaped`` is the only kind the caller retries on."""

    kind: Literal["reaped", "refused"]
    reason: str = ""
    pid: int = 0


@dataclass(frozen=True)
class _Process:
    """The holder's identity at one read: what ``ps`` answers, parsed.

    ``start`` is the opaque ``lstart`` token, compared byte-for-byte and never
    parsed: its only job is to tell a reused pid from the process that was
    proved about.
    """

    pid: int
    uid: int
    start: str
    cpu_s: float
    argv: str


# -- the probes (seams) --------------------------------------------------------


def _run(argv: list[str]) -> subprocess.CompletedProcess[str] | None:
    """One bounded subprocess, or ``None``. ``LC_ALL=C`` pins ``lstart``'s words."""

    try:
        return subprocess.run(  # noqa: S603 — fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=PROBE_TIMEOUT_S,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _read_process(pid: int) -> _Process | None:
    """``ps``'s answer for one pid, or ``None`` when it cannot be read.

    ONE call carries every field the proofs need, so a verification costs one
    fork rather than four. ``-ww`` is load-bearing for the reason
    ``update._process_argv`` documents: a piped ``ps`` truncates at the column
    width on Linux and the generation id lands past the cut, which would read a
    generation that is not there.

    ``ps`` prints ``lstart`` as five whitespace-separated words, which is why the
    parse is positional rather than a regex: uid, then those five, then ``time``,
    and the remainder is argv exactly as the process composed it.
    """

    result = _run(
        [
            "ps",
            "-ww",
            "-o",
            "uid=",
            "-o",
            "lstart=",
            "-o",
            "time=",
            "-o",
            "args=",
            "-p",
            str(pid),
        ]
    )
    if result is None or result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    fields = text.split(None, 7)
    if len(fields) < 8:
        return None
    try:
        uid = int(fields[0])
    except ValueError:
        return None
    start = " ".join(fields[1:6])
    cpu_s = _cpu_seconds(fields[6])
    if cpu_s is None:
        return None
    return _Process(pid=pid, uid=uid, start=start, cpu_s=cpu_s, argv=fields[7])


def _cpu_seconds(token: str) -> float | None:
    """``ps -o time=`` (``MM:SS.ss`` or ``HH:MM:SS``) as seconds."""

    parts = token.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if not 2 <= len(numbers) <= 3:
        return None
    total = 0.0
    for number in numbers:
        total = total * 60.0 + number
    return total


def _proc_locks_holder_pids(path: Path) -> list[int] | None:
    """The pids holding a ``FLOCK`` on ``path``, from Linux's ``/proc/locks``.

    ``None`` when the table cannot be read or the file's identity cannot be
    taken; an empty list is a real answer ("nobody holds a lock").
    """

    try:
        sample = path.stat()
    except OSError:
        return None
    try:
        table = Path("/proc/locks").read_text()
    except OSError:
        return None
    wanted = (os.major(sample.st_dev), os.minor(sample.st_dev), sample.st_ino)
    pids: list[int] = []
    for line in table.splitlines():
        fields = line.split()
        # ``1: FLOCK  ADVISORY  WRITE 48311 00:1b:123456 0 EOF``
        if len(fields) < 6 or fields[1] != "FLOCK":
            continue
        try:
            major, minor, inode = fields[5].split(":")
            identity = (int(major, 16), int(minor, 16), int(inode))
            pid = int(fields[4])
        except ValueError:
            continue
        if identity == wanted and pid not in pids:
            pids.append(pid)
    return pids


def _lsof_holder_pids(path: Path) -> list[int] | None:
    """The pids ``lsof`` reports holding ``path`` open, or ``None``.

    ANY output with a non-zero exit is refused rather than parsed: ``lsof``
    answers 1 for "no matches" but also for several real errors, and this probe
    cannot tell those apart (the installer's own reader takes the same view).
    """

    binary = "/usr/sbin/lsof" if Path("/usr/sbin/lsof").exists() else "lsof"
    result = _run([binary, "-F", "p", "--", str(path)])
    if result is None:
        return None
    if result.returncode != 0:
        # ``lsof`` answers 1 for "no matches" AND for several real errors, so the
        # only non-zero answer this probe keeps is the empty one: that is "nobody
        # holds it", which the caller refuses on as zero holders. Anything else
        # is an unanswered question.
        return [] if result.returncode == 1 and not result.stdout.strip() else None
    pids: list[int] = []
    for line in result.stdout.splitlines():
        if not line.startswith("p"):
            continue
        try:
            pid = int(line[1:])
        except ValueError:
            return None
        if pid not in pids:
            pids.append(pid)
    return pids


def _lock_holder_pids(path: Path) -> list[int] | None:
    """Holder discovery, platform-shaped, with Linux's ``lsof`` fallback."""

    if os.path.exists("/proc/locks"):
        pids = _proc_locks_holder_pids(path)
        if pids is not None:
            return pids
    return _lsof_holder_pids(path)


def _is_posix() -> bool:
    """The platform gate, as a seam: Windows has no flock and no generations.

    A seam rather than ``os.name`` inline for the reason
    ``connector/install._is_windows`` is one — patching ``os.name`` process-wide
    makes ``pathlib`` build ``WindowsPath`` objects, which cannot be instantiated
    off Windows.
    """

    return os.name == "posix"


def _signal(pid: int, number: int) -> bool:
    """Send one signal to ``pid``; ``False`` when it is already gone."""

    try:
        os.kill(pid, number)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _clock() -> float:
    return time.monotonic()


async def _pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


# -- the proofs ----------------------------------------------------------------


def _waiter_is_current() -> bool:
    """Proof 1: ``True`` only when our own generation was read AND matches."""

    from local_operator.update import generation_staleness_of_process

    return generation_staleness_of_process(os.getpid()) is False


def _holder_is_stale(pid: int) -> bool:
    """Proof 4: ``True`` only when the holder's generation was read AND differs."""

    from local_operator.update import generation_staleness_of_process

    return generation_staleness_of_process(pid) is True


def _holder_generation(pid: int) -> str:
    """The holder's generation name, for the log line only; ``""`` when unread."""

    from local_operator.update import generation_of_process

    generation = generation_of_process(pid)
    return generation.name if generation is not None else ""


def _is_connector(argv: str) -> bool:
    return _MODULE_PATTERN.search(argv) is not None


def _option(argv: str, flag: str) -> str | None:
    """The value after ``flag`` in a ``ps`` argv line, or ``None``.

    Whitespace splitting is exact for every spelling this installer writes; a
    path containing a space degrades to "cannot parse", which refuses rather than
    guesses. ``--flag=value`` is accepted as well as ``--flag value``.
    """

    tokens = argv.split()
    for index, token in enumerate(tokens):
        if token == flag:
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            return following if following and not following.startswith("-") else None
        if token.startswith(f"{flag}="):
            return token.partition("=")[2] or None
    return None


def _same_config(argv: str, config_dir: Path) -> bool:
    """Proof 3's second half: the holder serves the SAME store as the waiter."""

    named = _option(argv, "--config-dir")
    if not named:
        return False
    try:
        return Path(named).resolve() == Path(config_dir).resolve()
    except (OSError, ValueError):
        return False


def _still_ours(pid: int, recorded: _Process) -> bool:
    """Proof 5: pid, start time and argv all unchanged since the proofs ran."""

    now = _read_process(pid)
    if now is None:
        return False
    return now.start == recorded.start and now.argv == recorded.argv


def _refuse(reason: str, *, pid: int = 0) -> ReapOutcome:
    """Refuse, and say which proof failed — content-free, one line, once."""

    if pid:
        logger.warning("bridge connector reap refused: proof=%s pid=%d", reason, pid)
    else:
        logger.warning("bridge connector reap refused: proof=%s", reason)
    return ReapOutcome(kind="refused", reason=reason, pid=pid)


# -- the attempt ---------------------------------------------------------------


async def _in_executor(function: Callable[..., object], *args: object) -> object:
    """Run a forking probe off the loop: the waiter's SIGTERM must stay live."""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, function, *args)


async def _await_gone(pid: int, recorded: _Process, timeout_s: float) -> bool:
    """Wait (bounded) for the holder to die; ``True`` when it did."""

    deadline = _clock() + timeout_s
    while _clock() < deadline:
        now = await _in_executor(_read_process, pid)
        if now is None:
            return True
        assert isinstance(now, _Process)
        if now.start != recorded.start:
            # The pid was REUSED: the process we proved about is gone, and the
            # lock went with it. Signalling the new owner would be the whole
            # hazard proof 5 exists for.
            return True
        await _pause(POLL_S)
    return False


async def reap_stale_holder(config_dir: Path) -> ReapOutcome:
    """One attempt to stop a stale connector that holds this store's lock.

    Returns ``reaped`` only after a signal was delivered to a process the proofs
    selected; every other path is ``refused`` with the proof that failed. The
    caller retries its acquire on ``reaped`` and exits 75 on everything else,
    which is exactly today's behaviour for a busy lock.
    """

    if not _is_posix():
        # Windows has no flock and no generation layout; the probe there is the
        # installer's own, with a zero deadline and no reaping behind it.
        return _refuse("platform")

    # THE PROOFS RUN IN THE ORDER THE DESIGN NAMES THEM, because each later one
    # is only meaningful about a process the earlier ones have identified.
    if not await _in_executor(_waiter_is_current):
        return _refuse("waiter_generation_unproven")

    lock_path = Path(config_dir) / "team_bridge" / CONNECTOR_LOCK_NAME
    pids = await _in_executor(_lock_holder_pids, lock_path)
    if not isinstance(pids, list):
        return _refuse("holders_unknown")
    # A pid of zero (or a negative) is not a process ``/proc/locks`` or ``lsof``
    # can have reported as a holder; counting it would let the count check pass
    # on a reading that identified nobody.
    others = sorted(pid for pid in pids if pid != os.getpid() and pid > 0)
    if len(others) != 1:
        return _refuse("holder_count_not_one")
    pid = others[0]

    recorded = await _in_executor(_read_process, pid)
    if not isinstance(recorded, _Process):
        return _refuse("holder_unreadable", pid=pid)
    if recorded.uid != os.getuid():
        return _refuse("different_uid", pid=pid)
    if not _is_connector(recorded.argv):
        return _refuse("not_a_connector", pid=pid)
    if not _same_config(recorded.argv, config_dir):
        return _refuse("different_config", pid=pid)
    if not await _in_executor(_holder_is_stale, pid):
        return _refuse("holder_generation_unproven", pid=pid)

    generation = await _in_executor(_holder_generation, pid)
    # Proof 5, first application: the identity recorded above must still be the
    # live one — `ps` answered twice and a reuse can land between the two.
    if not await _in_executor(_still_ours, pid, recorded):
        return _refuse("holder_identity_moved", pid=pid)

    if not _signal(pid, signal.SIGTERM):
        # Either it died between the check and the call — nothing was signalled,
        # the kernel has already dropped the lock, and the caller's retry is the
        # repair — or the signal was refused, which is NOT a reap and must not be
        # reported as one.
        if await _in_executor(_read_process, pid) is None:
            return ReapOutcome(kind="reaped", reason="already_gone", pid=pid)
        return _refuse("signal_refused", pid=pid)
    logger.warning(
        "bridge connector reaped stale holder: pid=%d generation=%s "
        "evidence=same_uid,same_config,older_generation,sigterm",
        pid,
        generation,
    )

    if await _await_gone(pid, recorded, STOP_WAIT_S):
        return ReapOutcome(kind="reaped", reason="sigterm", pid=pid)

    # THE WEDGED CASE. Alive after the graceful window, and it must still BE the
    # process that was proved about, twice over: identity, then CPU.
    now = await _in_executor(_read_process, pid)
    if not isinstance(now, _Process) or now.start != recorded.start:
        return ReapOutcome(kind="reaped", reason="sigterm", pid=pid)
    if now.cpu_s - recorded.cpu_s >= CPU_FLAT_S:
        # Busy, not wedged: it consumed real CPU inside the window, so it is
        # making progress and gets the same benefit of the doubt a fresh
        # connector would. One SIGTERM per attempt is the contract.
        logger.warning(
            "bridge connector reap left a busy holder: pid=%d generation=%s evidence=sigterm",
            pid,
            generation,
        )
        return ReapOutcome(kind="reaped", reason="sigterm_busy", pid=pid)
    if not await _in_executor(_still_ours, pid, recorded):
        return ReapOutcome(kind="reaped", reason="sigterm", pid=pid)

    _signal(pid, signal.SIGKILL)
    logger.warning(
        "bridge connector reaped stale holder: pid=%d generation=%s "
        "evidence=same_uid,same_config,older_generation,sigkill",
        pid,
        generation,
    )
    return ReapOutcome(kind="reaped", reason="sigkill", pid=pid)
