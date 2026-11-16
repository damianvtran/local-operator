"""A machine-wide lease that lets ONE heavy build or test run at a time per host.

WHY THIS EXISTS
---------------
On 2026-11-16 at 02:06Z, lop-remote-node-2 (15.7 GB RAM) hit a global OOM:
several sessions ran parallel esbuild, node-22 and chrome builds at once, and the
kernel killed seven processes, among them a node-22 at 1.47 GB RSS. The fleet
rule "one heavy job at a time, memory-capped" existed only as guidance, and
agents running in parallel sessions do not coordinate on guidance.

The per-command memory guard (:mod:`local_operator.memory_guard`) could not
prevent it: it sizes and polices ONE command's process group, so every build
stayed under its own ceiling while their SUM exhausted the machine. What was
missing is a node-wide concurrency limit, and this module is that limit.

HOW IT WORKS
------------
A cross-process counting semaphore made of N slot files,
``<config_dir>/run/heavy-jobs/slot-<i>.lock``. Taking a slot is a non-blocking
``flock(LOCK_EX)`` on one of them; the lock is held by an open file descriptor
in the harness process for the life of the command. Constraints this relies on:

* ``flock`` locks belong to the OPEN FILE DESCRIPTION, not the process, so two
  sessions inside one harness process (subagents share the pid) still exclude
  each other: each ``open`` makes its own description.
* The kernel drops the lock when the descriptor closes, including when the
  holder dies by SIGKILL or OOM. A crashed harness therefore can never wedge
  the queue, which is why the lock, not the JSON record, is the truth.
* ``flock`` exists on Linux and macOS. On Windows ``fcntl`` is absent and the
  lease is DISABLED (:func:`supported` is False): there is no fleet of
  Windows build nodes to protect, and a half-working lease would be worse than
  none.
* ``flock`` is not reliable on network filesystems; the config dir is local on
  every supported install.

Each holder writes ``slot-<i>.json`` beside its lock (session id, harness pid,
command pgid, command preview, start time) so a waiter can say WHO it is
queued behind. The record is advisory: a record whose ``pid`` is dead is
stale and ignored, because only the flock can prove a slot is held.

USING AND DISABLING IT
----------------------
The ``bash`` tool classifies each command with :func:`is_heavy` and, for a
heavy one, acquires a slot before spawning (see ``execute_bash``). A
non-heavy command never touches the lease. The kill switch is the config key
``bash.heavy_jobs.enabled`` (LIVE, read per heavy command; default on). Two
node tuning knobs are environment overrides: ``LOCAL_OPERATOR_HEAVY_JOB_SLOTS``
(0 = auto, see :func:`auto_slot_count`) and ``LOCAL_OPERATOR_HEAVY_JOB_WAIT_S``
(how long a heavy command queues before it is refused, default 600).
``lop services status`` prints the current holders.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

try:  # POSIX only; Windows has no flock and runs without the lease.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows hosts
    fcntl = None  # type: ignore[assignment]

#: Where the one config key lives under ``values``; spelled once and pinned to
#: the ``settings_io`` row by a test, like ``memory_guard.BASH_MEMORY_*_PATH``.
HEAVY_JOBS_ENABLED_PATH: tuple[str, ...] = ("bash", "heavy_jobs", "enabled")

#: On by default: the failure it prevents (a node-wide OOM killing other
#: sessions' work) is far worse than the one it risks (a build waits its turn).
HEAVY_JOBS_ENABLED_DEFAULT = True

#: The slot count and the queue wait are ENVIRONMENT overrides, not config keys,
#: following ``memory_guard.ADMISSION_FLOOR_ENV``. Why: every config key must be a
#: localized ``/settings`` row (AGENTS.md "Adding a configuration key"), and these
#: two are node-operator tuning knobs nobody needs to discover in a settings
#: screen — the on/off switch is the one the operator does. Constraint: the
#: environment is the HARNESS process's, read per heavy command; set it where
#: lop is launched (shell profile, systemd unit) so every session on the node
#: agrees, because two processes that disagree on the slot count share only the
#: slots both of them count.
HEAVY_JOBS_SLOTS_ENV = "LOCAL_OPERATOR_HEAVY_JOB_SLOTS"
HEAVY_JOBS_WAIT_SECONDS_ENV = "LOCAL_OPERATOR_HEAVY_JOB_WAIT_S"
#: ``0`` (or unset) means "size from the host" (:func:`auto_slot_count`).
HEAVY_JOBS_SLOTS_DEFAULT = 0
#: Ten minutes: long enough to outlast a typical build ahead in the queue,
#: short enough that a wedged holder surfaces as a refusal, not a silent hang.
HEAVY_JOBS_WAIT_SECONDS_DEFAULT = 600

#: A caller asking for this much memory (``memory_mb``) is declaring a heavy
#: job whatever the command looks like.
HEAVY_MEMORY_MB_THRESHOLD = 1024

#: Hosts at or below this much RAM get exactly one slot. node2's 15.7 GB OOM'd
#: with three builds in parallel; one is the only count proven safe there.
_SINGLE_SLOT_MAX_TOTAL_MB = 16 * 1024
#: Above that, one slot per this much AVAILABLE memory: a node-22 build peaked
#: at ~1.5 GB RSS and chrome/esbuild siblings several more, so 8 GB per slot
#: leaves headroom for everything else on the box.
_MB_PER_SLOT = 8 * 1024
_MAX_SLOTS = 4

#: How often a waiter retries; flock(LOCK_NB) is a syscall, so this is cheap.
_POLL_SECONDS = 0.25
#: Command preview length in records and messages: enough to recognise a build.
PREVIEW_CHARS = 80


@dataclass(frozen=True)
class HeavyClass:
    """Why a command counts as heavy, as a short human-readable label."""

    reason: str


@dataclass(frozen=True)
class Holder:
    """One live slot holder, as read from its JSON record."""

    slot: int
    pid: int
    session_id: str
    command: str
    started_at: float
    pgid: int | None = None

    def who(self) -> str:
        """The name a waiter is told: the session when known, else the pid."""
        return self.session_id or f"pid {self.pid}"


def supported() -> bool:
    """Whether this platform can host the lease at all (POSIX flock)."""
    return fcntl is not None


# -- classification ---------------------------------------------------------

#: Splits a command line into simple commands. ``&`` is included so that
#: ``build & other`` is two segments; redirections like ``2>&1`` survive because
#: shlex keeps ``2>&1`` punctuation together with its neighbours only as ``>&``
#: and that is harmless (it never matches a tool name).
_SEPARATORS = {"&&", "||", ";", "|", "&", "\n", "(", ")", ";;", "|&"}
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: Wrappers that run "the rest of the line" as the real command. Each maps to
#: the number of positional arguments it consumes before the wrapped command
#: (``timeout 600 cmd`` eats one); option tokens are skipped separately.
_PREFIX_WRAPPERS: dict[str, int] = {
    "timeout": 1,
    "nice": 0,
    "nohup": 0,
    "setsid": 0,
    "time": 0,
    "command": 0,
    "exec": 0,
    "env": 0,
    "stdbuf": 0,
    "ionice": 0,
    "caffeinate": 0,
    "npx": 0,
    "bunx": 0,
}
#: Options of those wrappers that take a separate value (``nice -n 10``).
_WRAPPER_VALUE_OPTS = {"-n", "-s", "-k", "--signal", "--kill-after", "-u", "-c", "-C"}

#: Package-manager scripts that run builds, test runners or type checks. These
#: are the scripts every JS repo on the fleet maps to tsc/vite/vitest/esbuild.
_PM_HEAVY_SCRIPT = re.compile(r"^(build|test|typecheck|type-check)(:.*)?$")
_PM_VALUE_OPTS = {"-C", "--dir", "--filter", "-F", "--prefix", "--cwd", "-w", "--workspace"}
_VERSION_FLAGS = {"--version", "-v", "-V", "--help", "-h"}


def _segments(command: str) -> list[list[str]]:
    """Split ``command`` into simple-command token lists; [] when unparseable."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError:
        # An unbalanced quote: fall back to a plain whitespace split rather
        # than give up, so a heavy build with a stray quote still queues.
        tokens = command.split()
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


def _strip_wrappers(tokens: list[str]) -> list[str]:
    """Drop env assignments and runner prefixes to reach the real program."""
    i = 0
    while i < len(tokens):
        token = tokens[i]
        base = os.path.basename(token)
        if _ENV_ASSIGNMENT.match(token):
            i += 1
            continue
        if base == "systemd-run":
            # Everything up to `--` (or the first non-option) is systemd-run's.
            rest = tokens[i + 1 :]
            if "--" in rest:
                i = i + 1 + rest.index("--") + 1
                continue
            i += 1
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 1
            continue
        if base in ("uv", "poetry", "pdm", "hatch", "pipenv") and tokens[i + 1 : i + 2] == ["run"]:
            i += 2
            while i < len(tokens) and tokens[i].startswith("-"):
                # `uv run --with x pytest`: value-taking options are common here.
                if tokens[i] in ("--with", "--python", "-p", "--project", "--directory", "--env"):
                    i += 1
                i += 1
            continue
        if base in ("uvx", "pipx") or (
            base in ("pnpm", "yarn") and tokens[i + 1 : i + 2] == ["exec"]
        ):
            i += 1 if base in ("uvx", "pipx") else 2
            while i < len(tokens) and tokens[i].startswith("-"):
                i += 1
            if base == "pipx" and i < len(tokens) and tokens[i] == "run":
                i += 1
            continue
        if base in _PREFIX_WRAPPERS:
            i += 1
            positionals = _PREFIX_WRAPPERS[base]
            while i < len(tokens):
                if tokens[i].startswith("-") and tokens[i] != "-":
                    i += 2 if tokens[i] in _WRAPPER_VALUE_OPTS else 1
                    continue
                if base == "env" and _ENV_ASSIGNMENT.match(tokens[i]):
                    i += 1
                    continue
                if positionals > 0:
                    positionals -= 1
                    i += 1
                    continue
                break
            continue
        break
    return tokens[i:]


def _pm_script(args: list[str]) -> str | None:
    """The script name a package-manager invocation runs, if any."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in _PM_VALUE_OPTS else 1
    if i >= len(args):
        return None
    if args[i] in ("run", "run-script"):
        i += 1
        while i < len(args) and args[i].startswith("-"):
            i += 2 if args[i] in _PM_VALUE_OPTS else 1
        return args[i] if i < len(args) else None
    # `pnpm build` / `yarn build` run the script directly; `npm build` is a
    # different (obsolete) command, but `npm test` is the test script.
    return args[i]


def _classify_segment(tokens: list[str]) -> str | None:
    tokens = _strip_wrappers(tokens)
    if not tokens:
        return None
    program = os.path.basename(tokens[0])
    args = tokens[1:]
    if any(arg in _VERSION_FLAGS for arg in args[:1]):
        # `tsc --version`, `vitest -h`: an informational call is not a build.
        return None
    if program in ("pnpm", "npm", "yarn", "bun"):
        script = _pm_script(args)
        if script is not None and _PM_HEAVY_SCRIPT.match(script):
            if program == "npm" and script.startswith("build") and "run" not in args:
                return None
            return f"{program} {script}"
        return None
    if program in ("vite", "electron-vite"):
        # Builds only. `vite`/`electron-vite dev` is a long-lived dev server:
        # leasing it would pin the slot for hours while using little memory.
        return f"{program} build" if "build" in args else None
    if program in ("tsc", "vue-tsc"):
        return program
    if program in ("esbuild", "vitest", "jest", "webpack", "rollup", "turbo", "nx"):
        if program in ("turbo", "nx") and not any(_PM_HEAVY_SCRIPT.match(a) for a in args):
            return None
        return program
    if program in ("node", "bun", "deno") and "--test" in args:
        return f"{program} --test"
    if program in ("cargo", "go") and args[:1] and args[0] in ("build", "test"):
        return f"{program} {args[0]}"
    if program == "make" and any(a.startswith("test") for a in args if not a.startswith("-")):
        return "make test"
    pytest_args: list[str] | None = None
    if program in ("pytest", "py.test"):
        pytest_args = args
    elif re.fullmatch(r"python(\d+(\.\d+)?)?", program) and args[:2] == ["-m", "pytest"]:
        pytest_args = args[2:]
    if pytest_args is not None:
        # Only a PARALLEL pytest is heavy: `-n 8` forks eight interpreters,
        # each importing the whole package. A single targeted file is not.
        for index, arg in enumerate(pytest_args):
            if arg in ("-n", "--numprocesses"):
                value = pytest_args[index + 1] if index + 1 < len(pytest_args) else ""
            elif match := re.fullmatch(r"-n(\S+)|--numprocesses=(\S+)", arg):
                value = match.group(1) or match.group(2)
            else:
                continue
            # `-n 0` is xdist's "run in-process": one interpreter, not heavy.
            if value != "0":
                return "pytest -n"
        return None
    return None


def is_heavy(command: str, memory_mb: float | None = None) -> HeavyClass | None:
    """Classify a ``bash`` command as heavy (lease it) or not (run it now).

    Deliberately conservative: a false positive only queues a command behind a
    build, while a false negative is what the lease exists to prevent, yet a
    lease that queues `ls` would be disabled by the first operator it annoys.
    So only program families that measurably spend gigabytes are matched:
    bundlers (esbuild, vite/electron-vite build, webpack, rollup), the
    TypeScript compiler, JS test runners (vitest, jest, node --test), package
    manager build/test/typecheck scripts, parallel pytest (``-n``),
    ``make test*`` and ``cargo``/``go`` build|test. Wrappers (``cd x &&``,
    env assignments, ``timeout N``, ``nice``, ``systemd-run … --``,
    ``uv run``, ``npx``) are seen through. A caller-declared ``memory_mb`` at
    or above :data:`HEAVY_MEMORY_MB_THRESHOLD` is heavy regardless.
    """
    if memory_mb is not None and memory_mb >= HEAVY_MEMORY_MB_THRESHOLD:
        return HeavyClass(f"memory_mb={int(memory_mb)}")
    for segment in _segments(command):
        reason = _classify_segment(segment)
        if reason is not None:
            return HeavyClass(reason)
    return None


# -- slot sizing ------------------------------------------------------------


def auto_slot_count(total_mb: int | None, available_mb: int | None) -> int:
    """Slots for ``slots: 0``: 1 on ≤16 GB hosts, else one per 8 GB available, max 4.

    Unmeasurable memory means one slot: the safe answer, never a guess upward.
    Available (not total) memory above the small-host line, because a large
    host already crowded by other work should not admit four builds.
    """
    if total_mb is None or total_mb <= _SINGLE_SLOT_MAX_TOTAL_MB or available_mb is None:
        return 1
    return max(1, min(_MAX_SLOTS, available_mb // _MB_PER_SLOT))


def resolve_slot_count(configured: Any) -> int:
    """The slot count for a configured value; non-positive or bad means auto."""
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        return configured
    from local_operator import memory_guard

    return auto_slot_count(
        memory_guard._total_memory_mb(),
        memory_guard._available_memory_mb(memory_guard._default_runner),
    )


@dataclass(frozen=True)
class Settings:
    enabled: bool
    slots: int
    wait_seconds: float


def _env_number(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def configured_settings() -> Settings:
    """Read the switch LIVE (fresh ConfigManager) and the two env overrides.

    A broken ``config.yml`` or a malformed override must never stop a command,
    so any read failure or wrongly typed value falls back to the default.
    """
    enabled: Any = HEAVY_JOBS_ENABLED_DEFAULT
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        config = ConfigManager(config_dir())
        enabled = config.get_nested_value(HEAVY_JOBS_ENABLED_PATH, enabled)
    except Exception:  # noqa: BLE001 - config trouble must never block a command
        pass
    if not isinstance(enabled, bool):
        enabled = HEAVY_JOBS_ENABLED_DEFAULT
    slots = int(_env_number(HEAVY_JOBS_SLOTS_ENV, HEAVY_JOBS_SLOTS_DEFAULT))
    wait = _env_number(HEAVY_JOBS_WAIT_SECONDS_ENV, HEAVY_JOBS_WAIT_SECONDS_DEFAULT)
    return Settings(enabled=enabled, slots=resolve_slot_count(slots), wait_seconds=wait)


def lease_dir() -> Path:
    """``<config_dir>/run/heavy-jobs``: shared by every session of this config."""
    from local_operator.paths import config_dir

    return config_dir() / "run" / "heavy-jobs"


# -- the lease ----------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    from local_operator.procstate import pid_liveness

    # Doubt counts as alive: naming a holder that just died is harmless, while
    # hiding a live one would misreport who is using the machine.
    return pid_liveness(pid) is not False


class Lease:
    """One held slot. Release exactly once; closing the fd drops the flock."""

    def __init__(self, slot: int, fd: int, record_path: Path, record: dict[str, Any]):
        self.slot = slot
        self._fd: int | None = fd
        self._record_path = record_path
        self._record = record

    @property
    def held(self) -> bool:
        return self._fd is not None

    def note_pgid(self, pgid: int | None) -> None:
        """Record the spawned group once known, so status can name the command."""
        if self._fd is None or pgid is None:
            return
        self._record["pgid"] = pgid
        _write_record(self._record_path, self._record)

    def __del__(self) -> None:
        # Safety net only: every caller releases explicitly. A Lease dropped on
        # an unexpected path must not keep the machine's slot until exit.
        self.release()

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        # Remove the record BEFORE dropping the lock, so the next holder's
        # fresh record can never be deleted by our late unlink.
        try:
            self._record_path.unlink()
        except OSError:
            pass
        try:
            os.close(fd)  # closing the only descriptor releases the flock
        except OSError:
            pass


def _write_record(path: Path, record: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(record)}.tmp")
    try:
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # The record is advisory; a failed write only costs a waiter the name.
        try:
            tmp.unlink()
        except OSError:
            pass


def try_acquire(
    directory: Path,
    slots: int,
    *,
    session_id: str = "",
    command: str = "",
    pid: int | None = None,
) -> Lease | None:
    """Take a free slot without waiting, or return None when all are held."""
    if fcntl is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    for slot in range(max(1, slots)):
        lock_path = directory / f"slot-{slot}.lock"
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        record = {
            "slot": slot,
            "pid": pid if pid is not None else os.getpid(),
            "session_id": session_id,
            "command": command[:PREVIEW_CHARS],
            "started_at": time.time(),
            "pgid": None,
        }
        record_path = directory / f"slot-{slot}.json"
        _write_record(record_path, record)
        return Lease(slot, fd, record_path, record)
    return None


def read_holders(directory: Path, slots: int | None = None) -> list[Holder]:
    """Live holders from the slot records; stale (dead-pid) records are skipped.

    Reads JSON only and never probes the flock, so it is cheap enough for a
    status command and cannot disturb a holder.
    """
    holders: list[Holder] = []
    try:
        paths = sorted(directory.glob("slot-*.json"))
    except OSError:
        return holders
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            slot = int(raw["slot"])
            pid = int(raw["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if slots is not None and slot >= slots:
            continue
        if not _pid_alive(pid):
            continue
        pgid = raw.get("pgid")
        holders.append(
            Holder(
                slot=slot,
                pid=pid,
                session_id=str(raw.get("session_id") or ""),
                command=str(raw.get("command") or ""),
                started_at=float(raw.get("started_at") or 0.0),
                pgid=pgid if isinstance(pgid, int) else None,
            )
        )
    return holders


def describe_holders(holders: Iterable[Holder]) -> str:
    """``<session or pid N> (<command>)`` for each holder, comma-joined."""
    parts = [f"{h.who()} ({h.command})" if h.command else h.who() for h in holders]
    return ", ".join(parts) if parts else "an unidentified holder"


@dataclass(frozen=True)
class Refusal:
    """The wait ran out; ``message`` is the tool error to return."""

    message: str
    holders: tuple[Holder, ...]


def refusal_message(holders: Iterable[Holder], waited: float, reason: str) -> str:
    return (
        f"HEAVY JOB QUEUE FULL: this command ({reason}) waited {int(waited)}s for the "
        f"machine-wide heavy-job slot and did not get it.\n"
        f"Held by: {describe_holders(holders)}.\n"
        "Only one heavy build/test runs per machine at a time, so parallel builds "
        "cannot OOM the host. Retry once the holder finishes (check with "
        "`lop services status`), or raise LOCAL_OPERATOR_HEAVY_JOB_WAIT_S. To turn the "
        "queue off on this machine: set bash.heavy_jobs.enabled to false."
    )


def queued_line(holders: Iterable[Holder], waited: float) -> str:
    return f"queued behind {describe_holders(holders)}, waited {int(waited)}s"


async def acquire(
    directory: Path,
    slots: int,
    *,
    wait_seconds: float,
    reason: str,
    session_id: str = "",
    command: str = "",
    on_wait: Callable[[str], None] | None = None,
    aborted: Callable[[], bool] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> Lease | Refusal | None:
    """Wait up to ``wait_seconds`` for a slot.

    Returns the :class:`Lease`, a :class:`Refusal` when the wait ran out, or
    ``None`` when ``aborted()`` turned true (the caller handles its own abort).
    ``on_wait`` receives the visible "queued behind …" line about once a second.
    """
    started = time.monotonic()
    last_note = -1.0
    while True:
        lease = try_acquire(directory, slots, session_id=session_id, command=command)
        if lease is not None:
            return lease
        waited = time.monotonic() - started
        if aborted is not None and aborted():
            return None
        if waited >= wait_seconds:
            holders = tuple(read_holders(directory, slots))
            return Refusal(refusal_message(holders, waited, reason), holders)
        if on_wait is not None and waited - last_note >= 1.0:
            last_note = waited
            on_wait(queued_line(read_holders(directory, slots), waited))
        await sleep(min(_POLL_SECONDS, max(0.01, wait_seconds - waited)))


def status_line(directory: Path, slots: int, now: float | None = None) -> str:
    """The ``lop services status`` row: holders with age, or the free count."""
    holders = read_holders(directory, slots)
    if not holders:
        return f"heavy jobs: 0/{slots} slots in use"
    now = time.time() if now is None else now
    parts = []
    for holder in holders:
        minutes = max(0, int((now - holder.started_at) // 60))
        who = f"{holder.session_id}/{holder.pid}" if holder.session_id else f"pid {holder.pid}"
        parts.append(f"{who} {holder.command} ({minutes}m)")
    return f"heavy jobs: {len(holders)}/{slots} slots held: " + "; ".join(parts)
