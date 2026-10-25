"""Capture shard memory state at the moment an ENOMEM test failure surfaces.

WHY THIS EXISTS
---------------
On 2026-10-24 ``test (3.13, 3)`` failed with eight unrelated subprocess-spawning
cells raising ``OSError: [Errno 12] Cannot allocate memory`` inside the same
second (``tests/unit/providers/test_login_cancel_cli.py::test_ctrl_c_really_exits_the_command``
for both providers, plus six in ``tests/unit/secrets/test_cli.py``). The same
shape had already hit ``test (3.13, 3)``, ``test (3.13, 1)`` and
``test (3.12, 1)`` on earlier main runs.

Errno 12 is the kernel refusing an allocation, and that is the whole of what
the traceback says. It cannot distinguish a runner that was genuinely short at
that instant from a leak earlier in the shard that consumed the memory before
these cells ran. The runner is gone by the time anyone reads the log, so the
evidence has to be written at the failure.

This module records that evidence from inside the xdist worker, before the test
process unwinds: host memory and swap, load average, the failing worker's own
resource and kernel ceilings, the shard's process tree and per-process RSS, the
cgroup v2 high-water mark (and the v1 ceiling where that hierarchy is in use),
and the failing node id.

WHAT THE FIRST EIGHT DUMPS SETTLED
----------------------------------
The dump fired again on main run 40140983105, shard ``test (3.13, 3)``, head
``a96e13e8d8``: eight dumps, all readable. Every one showed ``MemAvailable``
11.9-12.1 GiB, ``MemFree`` 9.1 GiB, ``SwapFree`` 3.0 GiB, runner load
3.66/3.12/1.60, no cgroup v2 memory limits, and a seven-process shard tree at
about 3.0 GB RSS. The runner was not short of memory. That falsifies the
capacity explanation -- it is not merely unproven -- and the worker-count
mitigation drafted against it was withdrawn for that reason, not deferred.

The one remaining class that produces ``ENOMEM`` with 12 GiB free is a
per-process, kernel or commit ceiling. So the dump now also records, from the
failing worker itself: ``RLIMIT_AS``/``DATA``/``NPROC``/``STACK``/``NOFILE`` soft
and hard, the worker's mapping count (``/proc/<pid>/maps`` lines, never the
lines), ``VmPeak``/``VmSize``/``VmData``/``VmStk``/``VmRSS``, ``Threads``, the
open descriptor count and the per-UID process count, ``vm.max_map_count`` and
the ``map_count / max_map_count`` ratio, the namespace task total from
``/proc/loadavg``, the commit accounting and overcommit policy
(``CommitLimit``/``Committed_AS``, ``vm.overcommit_memory``/``ratio``), and the
cgroup v1 memory ceiling beside the v2 files.

Commit accounting is the strongest remaining capacity-shaped explanation: with
``vm.overcommit_memory=2`` the kernel refuses ``mmap``/``brk`` once
``Committed_AS`` reaches ``CommitLimit``, and ``MemAvailable`` can still read
12 GiB while that refusal happens.

WHAT THIS PROBE IS FOR, AND ITS FALSIFIER
-----------------------------------------
If the next failure shows every limit permissive and ``map_count`` far below
``vm.max_map_count``, the ceiling theory is wrong as well, and the hunt moves to
the spawning call itself -- the code asking the kernel for an allocation, not
the budget the kernel reports. The dump is evidence for that decision, not a
claim that any number is currently out of range.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not retry, reduce a worker count, change a timeout or touch a test.
It never reads a command line: this dump is uploaded as a public artifact, an
argument can be a credential, and the process tree only needs pids, ppids, RSS
and names (the root comes from the worker's parent pid). It does not emit map
lines either, only their count: a mapping line carries filesystem paths, and
this artifact is public.

Nothing is written on a healthy run: the hook is one ``isinstance`` check per
test report and the directory is created only when an ENOMEM failure actually
occurs. It never fails a run either -- every read is bounded and wrapped, and
an unusable dump directory disables the instrument instead of raising inside a
pytest hook. That is the same rule ``tests/shard_stall_watchdog.py`` follows,
and for the same reason: a diagnostic that can turn a green run red is switched
off the first time it is wrong.

WHERE THE FILE LANDS
--------------------
One file per failure under ``$TMPDIR/lo-shard-memory/run-<pid>/``, or under
``LOCAL_OPERATOR_SHARD_MEMORY_DIR`` when that names a directory (the CI shard
job sets it per shard and attempt). The path is deliberately outside the
checkout: a test run must never write into the repo tree, and the CI upload
step reads it with ``always()`` so a failed job still carries the file.
"""

from __future__ import annotations

import errno
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

#: The error that motivated this module. Named rather than spelled ``12`` at
#: every use so the trigger and the report cannot drift apart.
ENOMEM = errno.ENOMEM

#: The ``/proc/meminfo`` keys the capacity question needs, in report order.
#: ``MemAvailable`` is the kernel's estimate of what a new allocation can get
#: without swapping; ``Cached`` is page cache that can be reclaimed, so a low
#: ``MemFree`` with a healthy ``Cached`` is not the same reading as a shortage.
MEMINFO_KEYS = ("MemAvailable", "MemFree", "SwapFree", "Cached")

#: cgroup v2 memory files. ``memory.peak`` is the high-water mark since the
#: container started -- the number a worker-count decision needs after the fact,
#: because ``memory.current`` has usually fallen again by the time it is read.
CGROUP_FILES = ("memory.max", "memory.current", "memory.peak")

#: The worker's resource limits that can refuse an allocation while the host
#: still has free memory, mapped from the label ``/proc/<pid>/limits`` prints.
#: ``NOFILE`` fails with ``EMFILE`` rather than ``ENOMEM``, but it is one row on
#: the same read and an exhausted descriptor table is the other classic
#: per-process ceiling, so it is recorded beside the rest.
LIMIT_ROWS = (
    ("RLIMIT_AS", "Max address space"),
    ("RLIMIT_DATA", "Max data size"),
    ("RLIMIT_NPROC", "Max processes"),
    ("RLIMIT_STACK", "Max stack size"),
    ("RLIMIT_NOFILE", "Max open files"),
)

#: ``/proc/<pid>/status`` keys for the address-space, stack and thread
#: readings. ``VmPeak`` is the high-water mark a ceiling reading is compared
#: against; ``VmSize``/``VmData``/``VmStk``/``VmRSS`` are the same instant's
#: current values. ``VmData`` is the one ``RLIMIT_DATA`` actually governs and
#: ``VmStk`` the one ``RLIMIT_STACK`` governs, so each limit has a paired usage
#: rather than only the address-space total.
WORKER_STATUS_KEYS = ("VmPeak", "VmSize", "VmData", "VmStk", "VmRSS", "Threads")

#: Relative to a ``/proc`` root, so the fake tree in the unit tests reads the
#: same path shape the kernel exposes.
MAX_MAP_COUNT_PATH = Path("sys") / "vm" / "max_map_count"

#: ``/proc/meminfo`` keys and ``/proc/sys/vm`` files of the commit accounting.
#: With ``overcommit_memory=2`` the kernel refuses an ``mmap``/``brk`` once
#: ``Committed_AS`` reaches ``CommitLimit``, while ``MemAvailable`` can still be
#: high -- the exact shape the eight dumps measured. ``ratio`` only participates
#: when the policy is strict (2); it is still recorded for every policy.
COMMIT_MEMINFO_KEYS = ("CommitLimit", "Committed_AS")
OVERCOMMIT_SYSCTLS = ("overcommit_memory", "overcommit_ratio")
OVERCOMMIT_POLICIES = {
    "0": "heuristic overcommit (the kernel guesses)",
    "1": "always overcommit (no commit limit is enforced)",
    "2": "strict (refuses once Committed_AS reaches CommitLimit)",
}

#: cgroup v1 exposes its memory controller one directory below the mount root.
#: A v1 host has nothing at the v2 paths, so reading both is what keeps a v1
#: ceiling from being invisible.
CGROUP_V1_DIR = "memory"
CGROUP_V1_FILES = ("memory.limit_in_bytes", "memory.usage_in_bytes")
CGROUP_V1_STAT_KEYS = ("hierarchical_memory_limit", "total_rss")

#: cgroup v1 writes this near-``LONG_MAX`` page count when a group has no
#: ceiling. Rendering it as ``unlimited`` matches how the v2 block renders
#: ``max``; a reader must not have to recognize a sentinel to see "no limit".
CGROUP_V1_UNLIMITED = 9223372036854771712

#: Base for local runs. CI overrides the whole directory through
#: :data:`ENV_RUN_DIR`; ``TMPDIR`` is the platform-correct scratch root and is
#: outside the repository on every host.
DUMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "lo-shard-memory"

#: Exported by the controller so every xdist worker writes into this run's
#: directory. Workers are spawned after ``pytest_configure`` and inherit the
#: controller's environment, which is what makes one directory per run possible
#: without a lock shared between concurrent suites.
ENV_RUN_DIR = "LOCAL_OPERATOR_SHARD_MEMORY_DIR"

#: Bound on the process table. A shard's tree is normally two to ten processes;
#: a pathological tree (a runaway spawner) is exactly when the hook fires, so
#: the list is capped rather than allowed to grow with the failure. The true
#: count is always reported even when rows are omitted.
MAX_PROCESSES = 40

#: Bound on the exception text. The failure message is useful detail; an
#: unbounded one is a denial-of-evidence in reverse.
MAX_MESSAGE_CHARS = 500


#: Where the kernel and cgroup expose their state. Parameters on the builder so
#: the unit tests can hand it a fake tree instead of depending on the host.
PROC_ROOT = Path("/proc")
CGROUP_ROOT = Path("/sys/fs/cgroup")

#: This process's run directory, resolved once by :func:`install`. ``None``
#: until then and on a process that never installed; :func:`run_dir` still
#: returns a usable path in that case.
_RUN_DIR: Path | None = None


class ProcInfo(NamedTuple):
    """One row of the process table this module reports."""

    pid: int
    ppid: int
    name: str
    rss_kb: int | None


def run_dir() -> Path:
    """The directory dumps from THIS run belong in.

    The pid fallback is for a process that never installed (a direct import in
    a test, an unusual gateway): it cannot collide with a concurrent suite,
    which a single shared directory would.
    """
    if _RUN_DIR is not None:
        return _RUN_DIR
    explicit = os.environ.get(ENV_RUN_DIR, "").strip()
    if explicit:
        return Path(explicit)
    return DUMP_ROOT / f"run-{os.getpid()}"


def install() -> None:
    """Resolve this run's dump directory and export it to future workers.

    Deliberately does NOT create the directory. The instrument is installed on
    every pytest process in the repository -- the CI ``-n0`` stages included --
    and a healthy run must leave no trace at all. :func:`write_dump` creates the
    directory when there is something to put in it.

    Never raises: a hook that fails during ``pytest_configure`` would fail the
    run before a single test ran, which is worse than having no diagnostic.
    """
    global _RUN_DIR
    try:
        explicit = os.environ.get(ENV_RUN_DIR, "").strip()
        _RUN_DIR = Path(explicit) if explicit else DUMP_ROOT / f"run-{os.getpid()}"
        os.environ[ENV_RUN_DIR] = str(_RUN_DIR)
    except Exception:  # a diagnostic must never be the reason a run fails
        _RUN_DIR = None


def note_failure(item: object, call: object) -> None:
    """``pytest_runtest_makereport`` body: dump only on an ENOMEM failure.

    Kept as a plain function so the root conftest owns the one-line hook
    registration and this module's trigger is unit-testable without pytest
    machinery. Every attribute is read defensively: this runs in a worker whose
    failure is already being reported, and replacing that report with a
    diagnostic traceback would lose both.
    """
    try:
        excinfo = getattr(call, "excinfo", None)
        if excinfo is None:
            return
        error = getattr(excinfo, "value", None)
        if not isinstance(error, OSError) or error.errno != ENOMEM:
            return
        config = getattr(item, "config", None)
        write_dump(
            nodeid=str(getattr(item, "nodeid", "<unknown nodeid>")),
            error=error,
            config=config,
            root_pid=_controller_pid_hint(config),
        )
    except Exception:  # the hook must never raise
        return


def write_dump(
    *,
    nodeid: str,
    error: BaseException,
    config: Any = None,
    dump_dir: Path | None = None,
    now: datetime | None = None,
    proc_root: Path = PROC_ROOT,
    cgroup_root: Path = CGROUP_ROOT,
    pid: int | None = None,
    processes: list[ProcInfo] | None = None,
    root_pid: int | None = None,
) -> Path | None:
    """Write one dump and return its path, or ``None`` if it could not be written.

    ``processes`` and ``root_pid`` exist for the unit tests: the cap and the
    tree walk are exercised with a synthetic table instead of a hundred fake
    ``/proc`` entries. Production callers leave them unset; the hook supplies
    the controller hint and :func:`read_processes` reads the kernel.
    """
    try:
        current_pid = os.getpid() if pid is None else pid
        moment = now or datetime.now(timezone.utc)
        text = build_report(
            nodeid=nodeid,
            error=error,
            config=config,
            now=moment,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
            pid=current_pid,
            processes=processes,
            root_pid=root_pid,
        )
        target = (run_dir() if dump_dir is None else Path(dump_dir)) / _dump_name(
            config, current_pid, moment
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        try:
            # The controller's log is where someone first sees the failure; a
            # pointer there saves them hunting the artifact. Never fatal.
            sys.stderr.write(f"[shard memory] ENOMEM dump written to {target}\n")
            sys.stderr.flush()
        except Exception:
            pass
        return target
    except Exception:  # an unwritable directory disables the instrument
        return None


def _controller_pid_hint(config: Any) -> int | None:
    """The xdist controller's pid when this process is one of its workers.

    xdist starts each worker with a direct ``Popen`` from the controller, so
    the worker's parent IS the controller. Only a worker gets the hint: in a
    plain ``-n0`` run the parent is a shell, and rooting there would pull that
    shell's siblings into the tree. The pid-1 guard covers a reparented worker
    whose real controller is gone -- rooting at init would report the machine.
    """
    try:
        if not isinstance(getattr(config, "workerinput", None), dict):
            return None
        parent = os.getppid()
        return parent if parent > 1 else None
    except Exception:
        return None


def _dump_name(config: Any, pid: int, now: datetime) -> str:
    worker = _worker_identity(config)[0]
    safe_worker = re.sub(r"[^A-Za-z0-9_.-]", "_", worker) or "unknown"
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    return f"enomem-{safe_worker}-pid{pid}-{stamp}.txt"


def _worker_identity(config: Any) -> tuple[str, str, str]:
    """Identify the xdist worker and the count ``-n`` resolved to.

    The worker's ``workerinput`` carries xdist's own resolved ``workercount``
    (``len(nodemanager.specs)``), which is the authoritative number: on the
    controller ``-n auto`` is resolved before workers spawn, and the worker
    never sees the literal string "auto". The controller fallback is for a
    dump produced outside a worker (a direct hook call), where
    ``config.option.numprocesses`` is the best available answer.
    """
    if config is None:
        return ("unknown", "unknown", "no pytest config")
    workerinput = getattr(config, "workerinput", None)
    if isinstance(workerinput, dict):
        worker = str(workerinput.get("workerid", "unknown"))
        count = workerinput.get("workercount")
        if count is not None:
            return (worker, str(count), "workerinput.workercount")
        return (worker, "unknown", "workerinput has no workercount")
    option = getattr(config, "option", None)
    raw = getattr(option, "numprocesses", None)
    return (
        "controller (not an xdist worker)",
        "unknown" if raw is None else str(raw),
        "config.option.numprocesses",
    )


def build_report(
    *,
    nodeid: str,
    error: BaseException,
    config: Any = None,
    now: datetime | None = None,
    proc_root: Path = PROC_ROOT,
    cgroup_root: Path = CGROUP_ROOT,
    pid: int | None = None,
    processes: list[ProcInfo] | None = None,
    root_pid: int | None = None,
) -> str:
    """Render the whole dump as text. Pure enough to unit-test with fake trees."""
    moment = now or datetime.now(timezone.utc)
    current_pid = os.getpid() if pid is None else pid

    lines: list[str] = ["[shard memory] ENOMEM dump"]
    lines.append(f"timestamp: {moment.isoformat()} (epoch {moment.timestamp():.3f})")
    lines.extend(_error_lines(nodeid, error))
    lines.extend(_identity_lines(config, current_pid))
    lines.extend(_ceiling_lines(proc_root, current_pid))
    lines.extend(_meminfo_lines(proc_root))
    lines.extend(_commit_lines(proc_root))
    lines.extend(_loadavg_lines(proc_root))
    lines.extend(_process_lines(proc_root, current_pid, processes, root_pid))
    lines.extend(_cgroup_lines(proc_root, current_pid, cgroup_root))
    lines.extend(_cgroup_v1_lines(proc_root, current_pid, cgroup_root))
    return "\n".join(lines) + "\n"


def _error_lines(nodeid: str, error: BaseException) -> list[str]:
    raw_errno = getattr(error, "errno", None)
    name = errno.errorcode.get(raw_errno, "unknown") if isinstance(raw_errno, int) else "unknown"
    message = _one_line(str(error), MAX_MESSAGE_CHARS)
    return [
        "",
        "-- failure --",
        f"failing test: {nodeid}",
        f"errno: {raw_errno} ({name})",
        f"exception: {type(error).__name__}",
        f"message: {message}",
    ]


def _identity_lines(config: Any, pid: int) -> list[str]:
    worker, count, source = _worker_identity(config)
    return [
        "",
        "-- pytest identity --",
        f"xdist worker: {worker}",
        f"configured worker count: {count} (source: {source})",
        f"pytest pid: {pid}",
    ]


def _meminfo_lines(proc_root: Path) -> list[str]:
    path = proc_root / "meminfo"
    raw, read_error = _read_text(path)
    lines = ["", f"-- {path} --"]
    if raw is None:
        lines.append(f"(unavailable: {read_error})")
        return lines
    values: dict[str, int] = {}
    for line in raw.splitlines():
        key, separator, rest = line.partition(":")
        if not separator:
            continue
        parts = rest.split()
        if key in MEMINFO_KEYS and parts:
            try:
                values[key] = int(parts[0])
            except ValueError:
                continue
    for key in MEMINFO_KEYS:
        if key in values:
            lines.append(f"{key}: {_format_kb(values[key])}")
        else:
            lines.append(f"{key}: <missing>")
    return lines


def _loadavg_lines(proc_root: Path) -> list[str]:
    path = proc_root / "loadavg"
    raw, read_error = _read_text(path)
    lines = ["", f"-- {path} --"]
    if raw is None:
        lines.append(f"(unavailable: {read_error})")
        return lines
    lines.append(f"raw: {_one_line(raw, 200)}")
    parts = raw.split()
    if len(parts) >= 3:
        lines.append(
            f"load average: 1min={parts[0]} 5min={parts[1]} 15min={parts[2]}"
            + (f" runnable/total={parts[3]}" if len(parts) > 3 else "")
        )
    return lines


def _ceiling_lines(proc_root: Path, pid: int) -> list[str]:
    """The failing worker's own ceilings: rlimits, address space, task counts.

    ``pid`` is the worker that caught the ``OSError`` -- in production the
    process this hook runs in, whose ``/proc/self`` these files are. Any other
    process would describe a neighbour, and the ceiling this probe exists to
    test is the failing worker's. Every read is best-effort: a missing file is
    stated, never raised, and ``maps`` is counted rather than copied because a
    mapping line carries filesystem paths into a public artifact.
    """
    worker = proc_root / str(pid)
    lines = ["", f"-- resource limits: failing worker pid {pid} ({worker / 'limits'}) --"]
    limits_raw, limits_error = _read_text(worker / "limits")
    if limits_raw is None:
        lines.append(f"(unavailable: {limits_error})")
    else:
        rows = _parse_limit_rows(limits_raw)
        for rlimit, label in LIMIT_ROWS:
            row = rows.get(label)
            if row is None:
                lines.append(f"{rlimit} ({label}): unavailable: row absent")
                continue
            soft, hard, units = row
            lines.append(f"{rlimit} ({label}): soft={soft} hard={hard} units={units or 'unknown'}")

    lines.extend(["", f"-- address space and threads: failing worker pid {pid} ({worker}) --"])
    status_raw, status_error = _read_text(worker / "status")
    if status_raw is None:
        lines.append(f"(unavailable: {status_error})")
    else:
        values = _status_values(status_raw)
        for key in WORKER_STATUS_KEYS:
            value = values.get(key)
            if value is None:
                lines.append(f"{key}: <missing>")
            elif key == "Threads":
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {_format_kb(value)}")

    lines.extend(["", f"-- mapping and task ceilings: failing worker pid {pid} --"])
    map_count, maps_error = _count_lines(worker / "maps")
    if map_count is None:
        lines.append(f"map_count: unavailable: {maps_error}")
    else:
        lines.append(f"map_count: {map_count} ({worker / 'maps'} lines; contents not emitted)")
    max_path = proc_root / MAX_MAP_COUNT_PATH
    max_raw, max_error = _read_text(max_path)
    max_map_count: int | None = None
    if max_raw is None:
        lines.append(f"max_map_count: unavailable: {max_error}")
    else:
        try:
            max_map_count = int(max_raw.strip())
        except ValueError:
            lines.append(f"max_map_count: {_one_line(max_raw, 80)} (unparsed)")
        else:
            lines.append(f"max_map_count: {max_map_count} ({max_path})")
    if map_count is None or max_map_count is None or max_map_count <= 0:
        lines.append("map_count / max_map_count: unavailable (a reading is missing)")
    else:
        ratio = map_count / max_map_count
        lines.append(
            f"map_count / max_map_count: {map_count}/{max_map_count} = "
            f"{ratio:.4f} ({ratio * 100:.2f}%)"
        )
    tasks = _namespace_tasks(proc_root)
    if tasks is None:
        lines.append("namespace tasks: unavailable (no parsable /proc/loadavg)")
    else:
        runnable, total = tasks
        lines.append(f"namespace tasks: total={total} runnable={runnable} (/proc/loadavg field 4)")

    fd_count, fd_error = _count_entries(worker / "fd")
    if fd_count is None:
        lines.append(f"open file descriptors: unavailable: {fd_error}")
    else:
        # Reading /proc/self/fd opens a descriptor on that directory, which
        # then appears in its own listing; say so rather than silently
        # reporting one more open file than the worker holds.
        caveat = " (includes the descriptor this read opened)" if pid == os.getpid() else ""
        lines.append(f"open file descriptors: {fd_count} ({worker / 'fd'} entries){caveat}")

    uid, uid_count, uid_error = _uid_processes(proc_root, pid)
    if uid is None:
        lines.append(f"processes owned by uid: unavailable: {uid_error}")
    elif uid_count is None:
        lines.append(f"processes owned by uid {uid}: unavailable: {uid_error}")
    else:
        lines.append(
            f"processes owned by uid {uid}: {uid_count} "
            f"(/proc/<pid>/status Uid; all {proc_root} entries scanned)"
        )
    return lines


def _parse_limit_rows(raw: str) -> dict[str, tuple[str, str, str]]:
    """Parse ``/proc/<pid>/limits`` into ``label -> (soft, hard, units)``.

    Matched by the kernel's own labels rather than by column offsets: the file
    is column-aligned today, but a value wider than its column collapses the
    padding, and splitting the whole line on whitespace would then read part of
    the label as the soft limit. The labels in :data:`LIMIT_ROWS` contain no
    runs of whitespace, so a prefix match is stable in both layouts.
    """
    labels = {label for _, label in LIMIT_ROWS}
    rows: dict[str, tuple[str, str, str]] = {}
    for line in raw.splitlines():
        for label in labels:
            if not line.startswith(label):
                continue
            fields = line[len(label) :].split()
            if len(fields) >= 2:
                rows[label] = (fields[0], fields[1], fields[2] if len(fields) > 2 else "")
            break
    return rows


def _status_values(raw: str) -> dict[str, int]:
    """The :data:`WORKER_STATUS_KEYS` values present in a status file."""
    values: dict[str, int] = {}
    for line in raw.splitlines():
        key, separator, rest = line.partition(":")
        if not separator or key not in WORKER_STATUS_KEYS:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            values[key] = int(parts[0])
        except ValueError:
            continue
    return values


def _count_lines(path: Path) -> tuple[int | None, str | None]:
    """Count a file's lines without holding it or its contents.

    ``/proc/<pid>/maps`` emits one mapping per line and each line carries
    filesystem paths, so only the count is reported. Iterating the handle keeps
    the peak at one line even when the mapping table is at the ceiling that
    made this dump run; the whole loop is wrapped like every other read.
    """
    try:
        count = 0
        with path.open("rb") as handle:
            for _ in handle:
                count += 1
        return count, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _namespace_tasks(proc_root: Path) -> tuple[int, int] | None:
    """The ``runnable/total`` task counts in field 4 of ``/proc/loadavg``."""
    raw, _ = _read_text(proc_root / "loadavg")
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 4 or "/" not in parts[3]:
        return None
    runnable, _, total = parts[3].partition("/")
    try:
        return int(runnable), int(total)
    except ValueError:
        return None


def _commit_lines(proc_root: Path) -> list[str]:
    """Commit accounting and the overcommit policy, which bound allocations.

    ``overcommit_memory=2`` is the one policy that can refuse an allocation
    while ``MemAvailable`` is high: the kernel checks ``Committed_AS`` against
    ``CommitLimit`` instead of free pages. ``CommitLimit`` is ``0 kB`` on
    kernels that do not track it, so the ratio is stated unavailable rather
    than divided by zero.
    """
    lines = ["", f"-- commit and overcommit ({proc_root}) --"]
    values: dict[str, int] = {}
    meminfo_raw, meminfo_error = _read_text(proc_root / "meminfo")
    if meminfo_raw is None:
        lines.append(f"commit readings unavailable: {meminfo_error}")
    else:
        for line in meminfo_raw.splitlines():
            key, separator, rest = line.partition(":")
            if not separator or key not in COMMIT_MEMINFO_KEYS:
                continue
            parts = rest.split()
            if not parts:
                continue
            try:
                values[key] = int(parts[0])
            except ValueError:
                continue
        for key in COMMIT_MEMINFO_KEYS:
            if key in values:
                lines.append(f"{key}: {_format_kb(values[key])}")
            else:
                lines.append(f"{key}: <missing>")

    limit = values.get("CommitLimit")
    committed = values.get("Committed_AS")
    if limit is None or committed is None or limit <= 0:
        lines.append("Committed_AS / CommitLimit: unavailable (a reading is missing)")
    else:
        ratio = committed / limit
        lines.append(
            f"Committed_AS / CommitLimit: {committed}/{limit} = "
            f"{ratio:.4f} ({ratio * 100:.2f}%)"
        )

    policy: str | None = None
    for name in OVERCOMMIT_SYSCTLS:
        path = proc_root / "sys" / "vm" / name
        raw, read_error = _read_text(path)
        if raw is None:
            lines.append(f"vm.{name}: unavailable: {read_error}")
            continue
        value = raw.strip()
        try:
            int(value)
        except ValueError:
            lines.append(f"vm.{name}: {_one_line(value, 80)} (unparsed)")
            continue
        lines.append(f"vm.{name}: {value} ({path})")
        if name == "overcommit_memory":
            policy = value
    if policy is not None:
        description = OVERCOMMIT_POLICIES.get(policy, "unknown policy value")
        lines.append(f"overcommit policy: {description}")
    return lines


def _count_entries(path: Path) -> tuple[int | None, str | None]:
    """Count directory entries without reading what they point at.

    ``/proc/<pid>/fd`` holds one symlink per open descriptor, and each target
    is a path; only the count is reported because this artifact is public.
    """
    try:
        count = 0
        with os.scandir(path) as entries:
            for _entry in entries:
                count += 1
        return count, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _real_uid(proc_root: Path, pid: int) -> tuple[int | None, str | None]:
    """The worker's real UID, the one ``RLIMIT_NPROC`` is charged against."""
    raw, error = _read_text(proc_root / str(pid) / "status")
    if raw is None:
        return None, error
    for line in raw.splitlines():
        key, separator, rest = line.partition(":")
        if not separator or key != "Uid":
            continue
        parts = rest.split()
        if not parts:
            return None, "Uid is empty"
        try:
            return int(parts[0]), None
        except ValueError:
            return None, "Uid is not an integer"
    return None, "Uid is absent from status"


def _uid_processes(proc_root: Path, pid: int) -> tuple[int | None, int | None, str | None]:
    """How many processes the worker's real UID owns, across the namespace.

    ``RLIMIT_NPROC`` is charged per real UID, so the namespace task total from
    ``/proc/loadavg`` is not the number it is compared with. One bounded scan
    of the pid directories: the same read :func:`read_processes` performs for
    the process tree, and no field beyond the ``Uid`` line is parsed.
    """
    uid, uid_error = _real_uid(proc_root, pid)
    if uid is None:
        return None, None, uid_error
    try:
        entries = list(proc_root.iterdir())
    except Exception as exc:
        return uid, None, f"{type(exc).__name__}: {exc}"
    count = 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        raw, _error = _read_text(entry / "status")
        if raw is None:
            continue
        for line in raw.splitlines():
            key, separator, rest = line.partition(":")
            if not separator or key != "Uid":
                continue
            parts = rest.split()
            if parts and parts[0].isdigit() and int(parts[0]) == uid:
                count += 1
            break
    return uid, count, None


def _process_lines(
    proc_root: Path,
    pid: int,
    processes: list[ProcInfo] | None,
    root_pid: int | None,
) -> list[str]:
    if processes is None:
        table, read_error = read_processes(proc_root)
    else:
        table, read_error = list(processes), None
    lines = ["", "-- shard process tree --"]
    if read_error is not None:
        lines.append(f"process table unavailable: {read_error}")
        return lines
    tree, root = _process_tree(table, pid, root_pid)
    lines.append(f"tree root pid: {root}")
    lines.append(f"processes in tree: {len(tree)}")
    ordered = sorted(tree, key=lambda proc: ((proc.rss_kb or 0), proc.pid), reverse=True)
    if len(ordered) > MAX_PROCESSES:
        # Pinned rows keep the two processes the question is about even when the
        # tree is dominated by heavier children. The cap is on rows, not weight.
        pinned = [proc for proc in ordered if proc.pid in (pid, root)]
        rest = [proc for proc in ordered if proc.pid not in (pid, root)]
        shown = (pinned + rest)[:MAX_PROCESSES]
        lines.append(
            f"showing: {len(shown)} of {len(ordered)} (cap {MAX_PROCESSES}; "
            f"{len(ordered) - len(shown)} omitted below the RSS cut)"
        )
    else:
        shown = ordered
        lines.append(f"showing: {len(shown)} of {len(ordered)} (cap {MAX_PROCESSES})")
    lines.append(f"{'pid':>7} {'ppid':>7} {'rss_kb':>10} name")
    for proc in shown:
        rss = "-" if proc.rss_kb is None else str(proc.rss_kb)
        lines.append(f"{proc.pid:>7} {proc.ppid:>7} {rss:>10} {proc.name[:40]}")
    return lines


def _cgroup_relpath(proc_root: Path, pid: int, hierarchy: str) -> tuple[str | None, str | None]:
    """The memory-controller path from ``/proc/<pid>/cgroup``.

    ``0::<path>`` is the unified (v2) entry; ``<id>:memory:<path>`` is the v1
    entry whose controller list names ``memory``. The path is relative to that
    hierarchy's mount point -- the caller joins it under the root it knows --
    so a container or systemd scope resolves to the group that actually
    constrains this worker rather than always to the mount root.
    """
    source = proc_root / str(pid) / "cgroup"
    raw, read_error = _read_text(source)
    if raw is None:
        return None, f"{source} unreadable: {read_error}"
    for line in raw.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy_id, controllers, path = fields
        if hierarchy == "v2":
            if hierarchy_id == "0" and controllers == "":
                return path or "/", None
        elif "memory" in controllers.split(","):
            return path or "/", None
    return None, f"no {hierarchy} memory entry in {source}"


def _cgroup_memory_dir(
    proc_root: Path, pid: int, root_dir: Path, hierarchy: str
) -> tuple[Path, str, str | None]:
    """The most specific existing memory-cgroup directory, and why it was read.

    The process's own cgroup path carries the ceiling a container or scope
    sets; preferring it is what makes a nested slice visible. The provenance
    string says which directory was chosen, because a silent fallback to the
    mount root would hide that the nested one was missing. The third value is
    the resolution error for the caller to state, or ``None``.
    """
    rel, rel_error = _cgroup_relpath(proc_root, pid, hierarchy)
    if rel is not None:
        nested = root_dir / rel.lstrip("/")
        if nested.is_dir():
            return nested, f"nested {hierarchy} group {rel}", None
        return root_dir, f"mount root (nested {hierarchy} group {rel} not present)", None
    return root_dir, "mount root", rel_error


def _cgroup_lines(proc_root: Path, pid: int, cgroup_root: Path) -> list[str]:
    """The cgroup v2 memory files, from this process's own group when present.

    Every file is reported individually: a failed read is ``unavailable:
    <reason>``, and only a missing hierarchy is reported as an absence.
    Collapsing a failed read into "no files" would turn the exact case this
    probe exists to catch -- a ceiling that exists but cannot be read -- into a
    false negative.
    """
    base, provenance, resolution_error = _cgroup_memory_dir(proc_root, pid, cgroup_root, "v2")
    lines = ["", f"-- cgroup v2 memory ({base}) --"]
    source = f"cgroup source: {provenance}"
    if resolution_error is not None:
        source += f"; {resolution_error}"
    lines.append(source)
    if not cgroup_root.exists():
        lines.append(f"(no cgroup v2 hierarchy at {cgroup_root}; cgroup v1 or none)")
        return lines
    for name in CGROUP_FILES:
        path = base / name
        raw, read_error = _read_text(path)
        if raw is None:
            lines.append(f"{name}: unavailable: {read_error}")
            continue
        value = raw.strip()
        if name == "memory.max" and value == "max":
            lines.append(f"{name}: max (unlimited)")
            continue
        try:
            byte_count = int(value)
        except ValueError:
            lines.append(f"{name}: {_one_line(value, 80)} (unparsed)")
            continue
        lines.append(f"{name}: {byte_count} bytes ({_format_mib(byte_count)})")
    return lines


def _cgroup_v1_lines(proc_root: Path, pid: int, cgroup_root: Path) -> list[str]:
    """The cgroup v1 memory ceiling, beside the v2 files that v1 lacks.

    A v1 host has nothing at the v2 paths, so without this block a v1 ceiling
    is invisible. The process's own v1 memory path is preferred when present.
    ``memory.stat``'s ``hierarchical_memory_limit`` and ``total_rss`` are the
    effective ceiling and current charge of a nested group. A missing or
    unreadable file is stated per file, never folded into "no files": a
    permission error on the limit must not read as an absent controller.
    """
    root_dir = cgroup_root / CGROUP_V1_DIR
    base, provenance, resolution_error = _cgroup_memory_dir(proc_root, pid, root_dir, "v1")
    lines = ["", f"-- cgroup v1 memory ({base}) --"]
    source = f"cgroup source: {provenance}"
    if resolution_error is not None:
        source += f"; {resolution_error}"
    lines.append(source)
    if not base.exists():
        lines.append(f"(no cgroup v1 memory controller at {base}; v2 or none)")
        return lines
    for name in CGROUP_V1_FILES:
        path = base / name
        raw, read_error = _read_text(path)
        if raw is None:
            lines.append(f"{name}: unavailable: {read_error}")
            continue
        value = raw.strip()
        try:
            byte_count = int(value)
        except ValueError:
            lines.append(f"{name}: {_one_line(value, 80)} (unparsed)")
            continue
        if name == "memory.limit_in_bytes" and byte_count == CGROUP_V1_UNLIMITED:
            lines.append(f"{name}: unlimited")
            continue
        lines.append(f"{name}: {byte_count} bytes ({_format_mib(byte_count)})")
    stat_raw, stat_error = _read_text(base / "memory.stat")
    if stat_raw is None:
        lines.append(f"memory.stat: unavailable: {stat_error}")
        return lines
    stats: dict[str, int] = {}
    for line in stat_raw.splitlines():
        key, _, rest = line.partition(" ")
        if key not in CGROUP_V1_STAT_KEYS or not rest.strip():
            continue
        try:
            stats[key] = int(rest.strip())
        except ValueError:
            continue
    for key in CGROUP_V1_STAT_KEYS:
        if key not in stats:
            lines.append(f"memory.stat {key}: <missing>")
        elif key == "hierarchical_memory_limit" and stats[key] == CGROUP_V1_UNLIMITED:
            lines.append(f"memory.stat {key}: unlimited")
        else:
            lines.append(f"memory.stat {key}: {stats[key]} bytes ({_format_mib(stats[key])})")
    return lines


def read_processes(proc_root: Path) -> tuple[list[ProcInfo], str | None]:
    """Read the process table, skipping entries that vanish mid-scan.

    Processes exit while ``/proc`` is being read; that is normal and is not an
    error to report. Only an unreadable ``/proc`` itself returns an error.
    """
    try:
        entries = sorted(proc_root.iterdir(), key=lambda entry: entry.name)
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    table: list[ProcInfo] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        info = _read_process(entry, int(entry.name))
        if info is not None:
            table.append(info)
    return table, None


def _read_process(entry: Path, pid: int) -> ProcInfo | None:
    stat_text, _ = _read_text(entry / "stat")
    if stat_text is None:
        return None
    # ``comm`` may contain spaces and parentheses, so the ppid field is located
    # from the LAST ')' rather than by splitting the whole line.
    left = stat_text.find("(")
    right = stat_text.rfind(")")
    if left < 0 or right <= left:
        return None
    name = stat_text[left + 1 : right]
    rest = stat_text[right + 1 :].split()
    if len(rest) < 2:
        return None
    try:
        ppid = int(rest[1])
    except ValueError:
        return None

    status_text, _ = _read_text(entry / "status")
    rss_kb: int | None = None
    if status_text is not None:
        for line in status_text.splitlines():
            if line.startswith("Name:"):
                name = line.partition(":")[2].strip() or name
            elif line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rss_kb = int(parts[1])
                    except ValueError:
                        rss_kb = None

    # Command lines are never read: this dump is a public artifact, and an
    # argument can be a credential. The tree root comes from the parent pid
    # xdist gives the worker, not from matching text in another process.
    return ProcInfo(pid=pid, ppid=ppid, name=name, rss_kb=rss_kb)


def _process_tree(
    table: list[ProcInfo], pid: int, root_pid: int | None
) -> tuple[list[ProcInfo], int]:
    """Return the shard's tree and the pid it was rooted at.

    The root is the worker's parent -- the xdist controller, which starts each
    worker directly -- when that relationship is visible in the table. Nothing
    about a command line is consulted, so no argument text can influence or
    enter the report. Descendants of that root are the whole shard: every xdist
    worker, the controller and whatever children a test spawned. A missing or
    inconsistent hint falls back to this process, never to the whole table's
    parent chain.
    """
    by_pid = {process.pid: process for process in table}
    root = pid
    current = by_pid.get(pid)
    if (
        root_pid is not None
        and root_pid in by_pid
        and current is not None
        and current.ppid == root_pid
    ):
        root = root_pid

    if root not in by_pid:
        # The worker itself vanished from the table (a fake table keyed
        # elsewhere, or a race). Report every row rather than nothing: an
        # over-broad table is still evidence, an empty one is not.
        return sorted(table, key=lambda proc: proc.pid), root

    children: dict[int, list[ProcInfo]] = {}
    for process in table:
        children.setdefault(process.ppid, []).append(process)
    tree: list[ProcInfo] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        current_pid = stack.pop()
        if current_pid in seen:
            continue
        seen.add(current_pid)
        process = by_pid.get(current_pid)
        if process is None:
            continue
        tree.append(process)
        stack.extend(child.pid for child in children.get(current_pid, []))
    return tree, root


def _read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _format_kb(kilobytes: int) -> str:
    return f"{kilobytes} kB ({_format_mib(kilobytes * 1024)})"


def _format_mib(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.1f} MiB"


def _one_line(value: str, limit: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 16].rstrip() + " ... [truncated]"
