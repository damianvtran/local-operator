"""Guards for :mod:`tests.shard_memory_dump`, the ENOMEM shard snapshot.

The snapshot has one job: answer, after a CI shard dies of ``errno 12``, whether
the runner was short at that instant or whether something earlier in the shard
had already consumed the memory. Every way of failing at that job is silent --
a missing field, a truncated process list that omits the controller, a hook that
raises and replaces the real failure with a diagnostic traceback -- so the
assertions below are written against the reading the next decision needs, not
against the dump's formatting.

The fake ``/proc`` tree is deliberate: these tests must not assert on the
host's current memory, its process table or its cgroup layout, because the CI
runner and a developer laptop have none of those in common.
"""

from __future__ import annotations

import errno
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests import shard_memory_dump as memory

#: A fixed instant keeps the filename and the timestamp line deterministic.
MOMENT = datetime(2026, 10, 24, 12, 0, 0, 123456, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give every test its own run directory and DUMP_ROOT.

    ``DUMP_ROOT`` is computed from ``TMPDIR`` at import, so a test that let the
    real one through would write into the directory a concurrent suite is using.
    The module caches the resolved run directory, so both the cache and the
    exported variable are reset here as well.
    """
    monkeypatch.setattr(memory, "DUMP_ROOT", tmp_path / "dumps")
    monkeypatch.setattr(memory, "_RUN_DIR", None)
    monkeypatch.delenv(memory.ENV_RUN_DIR, raising=False)


class _Option:
    def __init__(self, numprocesses: object) -> None:
        self.numprocesses = numprocesses


class _Config:
    def __init__(self, workerinput: dict[str, object] | None, numprocesses: object = None) -> None:
        self.workerinput = workerinput
        self.option = _Option(numprocesses)


class _ExceptionInfo:
    def __init__(self, value: BaseException) -> None:
        self.value = value


class _Call:
    def __init__(self, value: BaseException | None) -> None:
        self.excinfo = None if value is None else _ExceptionInfo(value)


class _Item:
    def __init__(self, nodeid: str, config: _Config | None = None) -> None:
        self.nodeid = nodeid
        self.config = config


def _write_process(
    root: Path,
    pid: int,
    ppid: int,
    name: str,
    rss_kb: int | None,
    cmdline: str,
) -> None:
    entry = root / str(pid)
    entry.mkdir()
    # Only the comm and ppid fields matter to the parser; the rest of the
    # kernel's stat line is filler, present so the shape matches /proc.
    (entry / "stat").write_text(f"{pid} ({name}) S {ppid} 100 100 0 -1 4194560 0 0\n")
    status = f"Name:\t{name}\n"
    if rss_kb is not None:
        status += f"VmRSS:\t   {rss_kb} kB\n"
    (entry / "status").write_text(status)
    (entry / "cmdline").write_bytes(cmdline.replace(" ", "\x00").encode() + b"\x00")


@pytest.fixture
def fake_proc(tmp_path: Path) -> Path:
    """A fake /proc: one pytest controller, a worker, two children, one outsider."""
    root = tmp_path / "proc"
    root.mkdir()
    (root / "meminfo").write_text(
        "MemTotal:       16384000 kB\n"
        "MemFree:         1234567 kB\n"
        "MemAvailable:    2345678 kB\n"
        "Cached:          3456789 kB\n"
        "SwapFree:         456789 kB\n"
    )
    (root / "loadavg").write_text("0.52 0.58 0.59 2/456 7890\n")
    _write_process(root, 100, 1, "python", 400_000, "python -m pytest tests/unit -n 3")
    _write_process(
        root,
        101,
        100,
        "python",
        262_000,
        "python -u -c import sys; exec(eval(sys.stdin.readline()))",
    )
    _write_process(root, 102, 101, "Local Operator", 80_000, "/opt/lo/bin/Local Operator --run")
    _write_process(root, 103, 101, "sleep", None, "sleep 30")
    _write_process(root, 999, 1, "unrelated", 5_000, "/usr/bin/other --daemon")
    return root


def test_the_dump_records_every_field_the_next_measurement_needs(
    fake_proc: Path, tmp_path: Path
) -> None:
    """One assertion per required reading, with units where the reading has one.

    Mutation that must fail this: drop any key from ``MEMINFO_KEYS``, report kB
    without the MiB conversion, or omit the worker count.
    """
    text = memory.build_report(
        nodeid="tests/unit/secrets/test_cli.py::test_keyring_write",
        error=OSError(errno.ENOMEM, "Cannot allocate memory"),
        config=_Config({"workerid": "gw3", "workercount": 3}),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
    )

    # Timestamp: the reading is only evidence with the instant attached.
    assert "timestamp: 2026-10-24T12:00:00.123456+00:00" in text
    assert "epoch 1792843200.123" in text

    # Every memory key, as the kernel's kB and as MiB. Exact MiB values also pin
    # the conversion factor: a kB value read as bytes would be ~1000x off.
    assert "MemAvailable: 2345678 kB (2290.7 MiB)" in text
    assert "MemFree: 1234567 kB (1205.6 MiB)" in text
    assert "SwapFree: 456789 kB (446.1 MiB)" in text
    assert "Cached: 3456789 kB (3375.8 MiB)" in text

    # Load, both raw and split so a reader does not have to re-parse it.
    assert "raw: 0.52 0.58 0.59 2/456 7890" in text
    assert "load average: 1min=0.52 5min=0.58 15min=0.59 runnable/total=2/456" in text

    # Worker identity and the resolved -n count, which is the number a capacity
    # decision would change. "auto" never appears here: xdist resolved it.
    assert "xdist worker: gw3" in text
    assert "configured worker count: 3 (source: workerinput.workercount)" in text
    assert "pytest pid: 101" in text

    # The shard's tree: controller (100), worker (101), and both children.
    assert "tree root pid: 100" in text
    assert "processes in tree: 4" in text
    for pid in ("100", "101", "102", "103"):
        assert re.search(rf"^\s*{pid}\s", text, re.MULTILINE), f"pid {pid} is absent"
    assert "262000" in text  # the worker's RSS, in kB
    assert "Local Operator" in text
    # A process outside the shard must not be dragged in by "all of /proc".
    assert "unrelated" not in text

    # The failure itself, including the errno as a name.
    assert "failing test: tests/unit/secrets/test_cli.py::test_keyring_write" in text
    assert "errno: 12 (ENOMEM)" in text
    assert "exception: OSError" in text
    assert "message: [Errno 12] Cannot allocate memory" in text

    # cgroup v2 absent is stated, not silently blank.
    assert "no cgroup v2 memory.max/memory.current/memory.peak" in text


def test_no_command_line_or_argv_text_reaches_the_dump(
    fake_proc: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credential-bearing command lines must never be carried into the artifact.

    A command line can contain a token passed as an argument, and this artifact
    is uploaded for anyone to read; redacting it would mean anticipating every
    spelling of every secret. The dump therefore carries only pid, ppid, RSS,
    name and a boolean derived from the command line. This builds the marker
    into every per-process ``cmdline`` file (the real source), and into both
    argv surfaces a naive implementation might reach for, then asserts neither
    marker appears anywhere in the rendered dump.
    """
    line_marker = "LEAKMARKER-ARGV-31337"
    argv_marker = "LEAKMARKER-AUX-90210"
    for entry in fake_proc.iterdir():
        if entry.is_dir() and (entry / "cmdline").exists():
            (entry / "cmdline").write_bytes(
                f"--label {line_marker} pytest".replace(" ", "\x00").encode() + b"\x00"
            )
    monkeypatch.setattr(memory.sys, "argv", ["pytest", "--label", argv_marker])
    config = _Config(
        {
            "workerid": "gw3",
            "workercount": 3,
            "mainargv": ["pytest", "--label", argv_marker],
        }
    )

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        config=config,
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
    )

    assert line_marker not in text
    assert argv_marker not in text
    # Removing the text did not remove the reading: the tree still roots at the
    # process whose command line named pytest.
    assert "tree root pid: 100" in text


def test_cgroup_v2_peak_is_included_when_the_files_exist(tmp_path: Path) -> None:
    """``memory.peak`` is the reading that survives the moment; it must be read.

    Mutation that must fail this: read only ``memory.current``, the value that
    has already fallen back by the time a reader opens the artifact.
    """
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text("1073741824\n")
    (cgroup / "memory.current").write_text("536870912\n")
    (cgroup / "memory.peak").write_text("805306368\n")

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=tmp_path / "no-proc",
        cgroup_root=cgroup,
        pid=101,
        processes=[],
    )

    assert "memory.max: 1073741824 bytes (1024.0 MiB)" in text
    assert "memory.current: 536870912 bytes (512.0 MiB)" in text
    assert "memory.peak: 805306368 bytes (768.0 MiB)" in text


def test_the_process_list_is_capped_and_reports_the_omitted_count(tmp_path: Path) -> None:
    """The cap bounds the file; the true count and the worker rows stay.

    The failing case is a runaway spawner, so a list that grows with the failure
    is worst exactly when it matters. The controller and the reporting worker are
    pinned above the RSS cut because they are the two rows the question is about.
    """
    table = [memory.ProcInfo(pid=1000, ppid=1, name="python", rss_kb=500_000, is_pytest=True)]
    table.extend(
        memory.ProcInfo(
            pid=1001 + index,
            ppid=1000,
            name="python",
            rss_kb=1_000 + index,
            is_pytest=False,
        )
        for index in range(99)
    )

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=tmp_path / "no-proc",
        cgroup_root=tmp_path / "no-cgroup",
        pid=1001,
        processes=table,
    )

    assert "processes in tree: 100" in text
    assert f"showing: {memory.MAX_PROCESSES} of 100 (cap {memory.MAX_PROCESSES};" in text
    assert f"showing: {memory.MAX_PROCESSES} of 100" in text
    assert "omitted below the RSS cut" in text
    rows = [line for line in text.splitlines() if re.match(r"^\s+\d+\s+\d+\s+", line)]
    assert len(rows) == memory.MAX_PROCESSES
    assert re.search(r"^\s*1000\s", text, re.MULTILINE), "the controller was capped out"
    assert re.search(r"^\s*1001\s", text, re.MULTILINE), "the worker was capped out"


def test_missing_and_unreadable_files_are_reported_not_raised(tmp_path: Path) -> None:
    """Every reader is best-effort: a partially readable /proc still produces a file.

    Mutation that must fail this: let ``Path.read_text`` propagate from
    ``_read_meminfo`` or ``_read_process`` -- on a runner at its memory ceiling,
    the dump is the last thing that should raise.
    """
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").mkdir()  # IsADirectoryError, not FileNotFoundError
    (proc / "loadavg").mkdir()
    entry = proc / "101"
    entry.mkdir()
    (entry / "stat").write_text("101 (python) S 100 100 100 0 0\n")
    (entry / "status").mkdir()
    (entry / "cmdline").mkdir()

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=proc,
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
    )

    assert "unavailable: IsADirectoryError" in text
    # The process is still represented by its stat line, with no RSS.
    assert "processes in tree: 1" in text
    assert re.search(r"^\s*101\s+100\s+-", text, re.MULTILINE)
    assert "no cgroup v2" in text


def test_a_worker_missing_from_the_table_still_reports_the_rows_it_has(
    tmp_path: Path,
) -> None:
    """A race that drops the worker is not a reason to return an empty table."""
    table = [
        memory.ProcInfo(pid=200, ppid=1, name="python", rss_kb=10, is_pytest=True),
        memory.ProcInfo(pid=201, ppid=200, name="python", rss_kb=20, is_pytest=False),
    ]
    text = memory.build_report(
        nodeid="n",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=tmp_path / "no-proc",
        cgroup_root=tmp_path / "no-cgroup",
        pid=999,
        processes=table,
    )
    assert "processes in tree: 2" in text
    assert re.search(r"^\s*200\s", text, re.MULTILINE)
    assert re.search(r"^\s*201\s", text, re.MULTILINE)


def test_the_hook_records_only_enomem(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other OSErrors and ordinary failures must not produce a memory artifact.

    ``errno == 12`` is the entire trigger. A hook that wrote on every OSError
    would turn every permission error and every broken pipe into an artifact,
    burying the one reading this exists to collect.
    """
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(memory, "write_dump", lambda **kwargs: calls.append(kwargs))
    item = _Item("tests/unit/x.py::test_y")

    memory.note_failure(item, _Call(OSError(errno.EAGAIN, "again")))
    memory.note_failure(item, _Call(ValueError("not an OSError")))
    memory.note_failure(item, _Call(None))
    assert calls == []

    memory.note_failure(item, _Call(OSError(errno.ENOMEM, "cannot allocate memory")))
    assert len(calls) == 1
    assert calls[0]["nodeid"] == "tests/unit/x.py::test_y"
    assert getattr(calls[0]["error"], "errno") == errno.ENOMEM


def test_the_hook_never_raises_when_the_writer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unwritable dump directory disables the instrument, never fails a run.

    The hook runs while pytest is building the report for a test that has
    already died; a diagnostic traceback here would replace the failure it was
    sent to explain.
    """

    def _boom(**kwargs: object) -> None:
        raise MemoryError("the dump builder could not allocate either")

    monkeypatch.setattr(memory, "write_dump", _boom)
    memory.note_failure(_Item("n"), _Call(OSError(errno.ENOMEM, "boom")))


def test_write_dump_returns_none_when_the_directory_is_unusable(tmp_path: Path) -> None:
    """A file where the directory should be is a disabled instrument, not a raise."""
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    result = memory.write_dump(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        dump_dir=blocker,
        now=MOMENT,
        proc_root=tmp_path / "no-proc",
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
        processes=[],
    )
    assert result is None


def test_write_dump_names_the_worker_and_lands_outside_the_repo(
    fake_proc: Path, tmp_path: Path
) -> None:
    """The artifact filename identifies the worker and pid; the path is scratch.

    Mutation that must fail this: default the dump directory to the checkout
    (``Path.cwd()``) -- a test run would then dirty the repository, which is the
    reason the path derives from ``TMPDIR`` instead.
    """
    dump_dir = tmp_path / "dumps"
    path = memory.write_dump(
        nodeid="tests/unit/secrets/test_cli.py::test_keyring_write",
        error=OSError(errno.ENOMEM, "Cannot allocate memory"),
        config=_Config({"workerid": "gw3", "workercount": 3}),
        dump_dir=dump_dir,
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
    )

    assert path is not None
    assert path.parent == dump_dir
    assert path.name == "enomem-gw3-pid101-20261024T120000.123456Z.txt"
    written = path.read_text()
    assert "failing test: tests/unit/secrets/test_cli.py::test_keyring_write" in written
    assert "MemAvailable: 2345678 kB" in written


def test_install_exports_one_directory_and_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers inherit one run directory; a healthy run leaves no directory behind.

    The controller's ``pytest_configure`` runs before xdist spawns workers, so
    exporting the variable there is what puts every worker's file in the same
    place. Creating the directory here would litter ``TMPDIR`` on every suite in
    the repo -- this conftest is loaded by all of them.
    """
    memory.install()

    expected = memory.DUMP_ROOT / f"run-{os.getpid()}"
    assert memory.run_dir() == expected
    assert os.environ[memory.ENV_RUN_DIR] == str(expected)
    assert not expected.exists()


def test_the_controller_fallback_reports_numprocesses(tmp_path: Path) -> None:
    """Outside a worker, ``-n``'s resolved count still reaches the dump."""
    text = memory.build_report(
        nodeid="n",
        error=OSError(errno.ENOMEM, "boom"),
        config=_Config(None, numprocesses=4),
        now=MOMENT,
        proc_root=tmp_path / "no-proc",
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
        processes=[],
    )
    assert "xdist worker: controller (not an xdist worker)" in text
    assert "configured worker count: 4 (source: config.option.numprocesses)" in text
