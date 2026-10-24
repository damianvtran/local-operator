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
process unwinds: host memory and swap, load average, the shard's process tree
and per-process RSS, the cgroup v2 high-water mark, and the failing node id.
The next measurement -- whether to bound workers or hunt a leak -- needs exactly
these numbers, and ``memory.peak`` is the one that outlives the moment.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not retry, reduce a worker count, change a timeout or touch a test.
Nothing is written on a healthy run: the hook is one ``isinstance`` check per
test report and the directory is created only when an ENOMEM failure actually
occurs. It never fails a run either -- every read is bounded and wrapped, and an
unusable dump directory disables the instrument instead of raising inside a
pytest hook. That is the same rule ``tests/shard_stall_watchdog.py`` follows,
and for the same reason: a diagnostic that can turn a green run red is switched
off the first time it is wrong.

THE READING THIS SUPPORTS, STATED UP FRONT
-----------------------------------------
If the same cells fail again in a run that shows plenty of ``MemAvailable`` and
no cgroup pressure at the failure, this capacity explanation is WRONG and the
response changes to a real-defect hunt. The dump is evidence for that decision,
not a claim that any number is currently out of range.

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
    #: Whether this process's command line names pytest. Read and discarded at
    #: scan time so the tree can be rooted without ever putting the text in the
    #: record -- or in the dump, where an uploaded artifact would expose it.
    is_pytest: bool


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
        write_dump(
            nodeid=str(getattr(item, "nodeid", "<unknown nodeid>")),
            error=error,
            config=getattr(item, "config", None),
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
) -> Path | None:
    """Write one dump and return its path, or ``None`` if it could not be written.

    ``processes`` exists for the unit tests: the cap and the tree walk are
    exercised with a synthetic table instead of a hundred fake ``/proc``
    entries. Production callers leave it unset and :func:`read_processes`
    reads the kernel.
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
) -> str:
    """Render the whole dump as text. Pure enough to unit-test with fake trees."""
    moment = now or datetime.now(timezone.utc)
    current_pid = os.getpid() if pid is None else pid

    lines: list[str] = ["[shard memory] ENOMEM dump"]
    lines.append(f"timestamp: {moment.isoformat()} (epoch {moment.timestamp():.3f})")
    lines.extend(_error_lines(nodeid, error))
    lines.extend(_identity_lines(config, current_pid))
    lines.extend(_meminfo_lines(proc_root))
    lines.extend(_loadavg_lines(proc_root))
    lines.extend(_process_lines(proc_root, current_pid, processes))
    lines.extend(_cgroup_lines(cgroup_root))
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


def _process_lines(proc_root: Path, pid: int, processes: list[ProcInfo] | None) -> list[str]:
    if processes is None:
        table, read_error = read_processes(proc_root)
    else:
        table, read_error = list(processes), None
    lines = ["", "-- shard process tree --"]
    if read_error is not None:
        lines.append(f"process table unavailable: {read_error}")
        return lines
    tree, root = _process_tree(table, pid)
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


def _cgroup_lines(cgroup_root: Path) -> list[str]:
    lines = ["", f"-- cgroup v2 memory ({cgroup_root}) --"]
    found = False
    for name in CGROUP_FILES:
        path = cgroup_root / name
        raw, read_error = _read_text(path)
        if raw is None:
            continue
        found = True
        value = raw.strip()
        if name == "memory.max" and value == "max":
            lines.append(f"{name}: max (unlimited)")
            continue
        try:
            byte_count = int(value)
        except ValueError:
            lines.append(f"{name}: {_one_line(value, 80)} (unparsed; {read_error or 'raw'})")
            continue
        lines.append(f"{name}: {byte_count} bytes ({_format_mib(byte_count)})")
    if not found:
        lines.append("(no cgroup v2 memory.max/memory.current/memory.peak; cgroup v1 or none)")
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

    # The command line is READ ONLY to answer one boolean question -- does this
    # process name pytest? -- and the bytes are discarded. It can contain
    # credentials (a token passed as an argument), and this dump is uploaded as
    # a public artifact; a redaction filter would have to anticipate every way
    # a secret can be spelled, while not carrying the text cannot be wrong.
    cmdline_bytes, _ = _read_bytes(entry / "cmdline")
    is_pytest = cmdline_bytes is not None and b"pytest" in cmdline_bytes.lower()
    return ProcInfo(pid=pid, ppid=ppid, name=name, rss_kb=rss_kb, is_pytest=is_pytest)


def _process_tree(table: list[ProcInfo], pid: int) -> tuple[list[ProcInfo], int]:
    """Return the shard's tree and the pid it was rooted at.

    The root is the highest ancestor whose command line named pytest (the
    controller), reached from this worker. The text itself is never stored;
    only the boolean survives into :class:`ProcInfo`. Descendants of that root
    are the whole shard:
    every xdist worker, the controller and whatever children a test spawned.
    Climbing stops at the first ancestor that does not name pytest, so a suite
    started from an agent shell does not drag that shell's siblings in.
    """
    by_pid = {proc.pid: proc for proc in table}
    root = pid
    for _ in range(16):  # bounded: an insane parent chain must not hang the dump
        current = by_pid.get(root)
        if current is None:
            break
        parent = by_pid.get(current.ppid)
        if parent is None or not parent.is_pytest:
            break
        root = parent.pid

    if root not in by_pid:
        # The worker itself vanished from the table (a fake table keyed
        # elsewhere, or a race). Report every row rather than nothing: an
        # over-broad table is still evidence, an empty one is not.
        return sorted(table, key=lambda proc: proc.pid), root

    children: dict[int, list[ProcInfo]] = {}
    for proc in table:
        children.setdefault(proc.ppid, []).append(proc)
    tree: list[ProcInfo] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        current_pid = stack.pop()
        if current_pid in seen:
            continue
        seen.add(current_pid)
        proc = by_pid.get(current_pid)
        if proc is None:
            continue
        tree.append(proc)
        stack.extend(child.pid for child in children.get(current_pid, []))
    return tree, root


def _read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _read_bytes(path: Path) -> tuple[bytes | None, str | None]:
    try:
        return path.read_bytes(), None
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
