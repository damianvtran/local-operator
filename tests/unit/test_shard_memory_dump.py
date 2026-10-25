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
runner and a developer laptop have none of those in common. The worker's own
resource and kernel ceilings are asserted the same way -- the fixture gives the
controller different values, so a dump that reads the wrong process cannot pass.
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
    uid: int = 4242,
) -> None:
    entry = root / str(pid)
    entry.mkdir()
    # Only the comm and ppid fields matter to the parser; the rest of the
    # kernel's stat line is filler, present so the shape matches /proc.
    (entry / "stat").write_text(f"{pid} ({name}) S {ppid} 100 100 0 -1 4194560 0 0\n")
    status = f"Name:\t{name}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
    if rss_kb is not None:
        status += f"VmRSS:\t   {rss_kb} kB\n"
    (entry / "status").write_text(status)
    (entry / "cmdline").write_bytes(cmdline.replace(" ", "\x00").encode() + b"\x00")


#: The worker's limits. The controller's differ on purpose: a dump that reads
#: the wrong process is indistinguishable from a correct one unless the fixture
#: makes the two disagree on a value the real one does not also carry.
WORKER_LIMITS = {
    "Max address space": ("unlimited", "unlimited", "bytes"),
    "Max data size": ("unlimited", "unlimited", "bytes"),
    "Max processes": ("63488", "63488", "processes"),
    "Max stack size": ("8388608", "unlimited", "bytes"),
    "Max open files": ("1048576", "1048576", "files"),
}
CONTROLLER_LIMITS = {
    "Max address space": ("unlimited", "unlimited", "bytes"),
    "Max data size": ("unlimited", "unlimited", "bytes"),
    "Max processes": ("4096", "4096", "processes"),
    "Max stack size": ("8388608", "unlimited", "bytes"),
    "Max open files": ("4096", "4096", "files"),
}


def _write_limits(root: Path, pid: int, rows: dict[str, tuple[str, str, str]]) -> None:
    lines = ["Limit                     Soft Limit           Hard Limit           Units"]
    for label, (soft, hard, units) in rows.items():
        lines.append(f"{label:<26}{soft:<21}{hard:<21}{units}")
    (root / str(pid) / "limits").write_text("\n".join(lines) + "\n")


def _write_maps(root: Path, pid: int, lines: list[str]) -> None:
    (root / str(pid) / "maps").write_text("\n".join(lines) + "\n")


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
        "CommitLimit:     1048576 kB\n"
        "Committed_AS:    2097152 kB\n"
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
    _write_process(root, 999, 1, "unrelated", 5_000, "/usr/bin/other --daemon", uid=0)

    # Ceilings for the failing worker. 3277 mappings against the older 65530
    # default gives a 5% ratio, so the arithmetic is visible in the rendering.
    (root / "sys" / "vm").mkdir(parents=True)
    (root / "sys" / "vm" / "max_map_count").write_text("65530\n")
    (root / "sys" / "vm" / "overcommit_memory").write_text("2\n")
    (root / "sys" / "vm" / "overcommit_ratio").write_text("50\n")
    _write_limits(root, 101, WORKER_LIMITS)
    _write_limits(root, 100, CONTROLLER_LIMITS)
    _write_maps(
        root,
        101,
        [
            f"7f{index:014x}-7f{index + 1:014x} r-xp 00000000 08:01 42 /usr/lib/libx.so"
            for index in range(3277)
        ],
    )
    (root / "101" / "fd").mkdir()
    for descriptor in range(3):
        (root / "101" / "fd" / str(descriptor)).write_text("")
    # The process's own cgroup path: a nested group, so the resolver has a
    # path to prefer (the tests that exercise nesting create it) and the
    # fallback tests have something concrete to report as not present.
    (root / "101" / "cgroup").write_text("0::/nested/group\n4:memory:/nested/group\n")
    with (root / "101" / "status").open("a") as status_file:
        status_file.write(
            "VmPeak:\t 3145728 kB\nVmSize:\t 2097152 kB\n"
            "VmData:\t 1048576 kB\nVmStk:\t 132 kB\nThreads:\t 17\n"
        )

    # A v1 hierarchy has its ceiling below memory/, where the v2 reads above
    # find nothing; both are recorded so neither layout can hide a ceiling.
    cgroup_v1 = tmp_path / "cgroup" / "memory"
    cgroup_v1.mkdir(parents=True)
    (cgroup_v1 / "memory.limit_in_bytes").write_text("524288000\n")
    (cgroup_v1 / "memory.usage_in_bytes").write_text("104857600\n")
    (cgroup_v1 / "memory.stat").write_text(
        "cache 0\nhierarchical_memory_limit 524288000\ntotal_rss 104857600\n"
    )
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
        root_pid=100,
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
    assert "no cgroup v2 hierarchy" in text


def test_the_worker_ceilings_are_read_from_the_failing_worker(
    fake_proc: Path, tmp_path: Path
) -> None:
    """Every ceiling reading, taken from the worker that caught the error.

    Mutation that must fail this: read ``/proc/self`` instead of the worker's
    pid (the controller's limits differ here), print a numeric sentinel for an
    unlimited limit, or emit ``maps`` contents rather than their line count.
    """
    text = memory.build_report(
        nodeid="tests/unit/secrets/test_cli.py::test_keyring_write",
        error=OSError(errno.ENOMEM, "Cannot allocate memory"),
        config=_Config({"workerid": "gw3", "workercount": 3}),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
        root_pid=100,
    )

    # Soft and hard, with ``unlimited`` rendered as the word the kernel prints.
    assert "RLIMIT_AS (Max address space): soft=unlimited hard=unlimited units=bytes" in text
    assert "RLIMIT_DATA (Max data size): soft=unlimited hard=unlimited units=bytes" in text
    assert "RLIMIT_NPROC (Max processes): soft=63488 hard=63488 units=processes" in text
    assert "RLIMIT_STACK (Max stack size): soft=8388608 hard=unlimited units=bytes" in text
    assert "RLIMIT_NOFILE (Max open files): soft=1048576 hard=1048576 units=files" in text
    # The controller's descriptor limit is 4096; only the worker's row may appear.
    assert "soft=4096" not in text

    # Address space and thread count at the failure, from the worker's status.
    assert "VmPeak: 3145728 kB (3072.0 MiB)" in text
    assert "VmSize: 2097152 kB (2048.0 MiB)" in text
    assert "VmData: 1048576 kB (1024.0 MiB)" in text
    assert "VmStk: 132 kB (0.1 MiB)" in text
    assert "VmRSS: 262000 kB (255.9 MiB)" in text
    assert "Threads: 17" in text

    # Mapping headroom: the count only, then the ratio to the kernel ceiling.
    assert "map_count: 3277 (" in text
    assert "max_map_count: 65530 (" in text
    assert "map_count / max_map_count: 3277/65530 = 0.0500 (5.00%)" in text

    # The namespace task total, the same 4/788 figure /proc/loadavg carries.
    assert "namespace tasks: total=456 runnable=2 (/proc/loadavg field 4)" in text

    # Descriptor and per-UID task ceilings: both must describe the worker.
    assert "open file descriptors: 3 (" in text
    assert "processes owned by uid 4242: 4 (" in text


def test_commit_overcommit_and_cgroup_v1_ceilings_are_recorded(
    fake_proc: Path, tmp_path: Path
) -> None:
    """The strongest capacity-shaped hypothesis, plus the v1 hierarchy.

    Mutation that must fail this: drop the commit block, divide by a zero
    ``CommitLimit``, or leave the v1 files unread (the v2 block alone finds
    nothing on a v1 host).
    """
    text = memory.build_report(
        nodeid="tests/unit/secrets/test_cli.py::test_keyring_write",
        error=OSError(errno.ENOMEM, "Cannot allocate memory"),
        config=_Config({"workerid": "gw3", "workercount": 3}),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "cgroup",
        pid=101,
        root_pid=100,
    )

    assert "CommitLimit: 1048576 kB (1024.0 MiB)" in text
    assert "Committed_AS: 2097152 kB (2048.0 MiB)" in text
    assert "Committed_AS / CommitLimit: 2097152/1048576 = 2.0000 (200.00%)" in text
    assert "vm.overcommit_memory: 2 (" in text
    assert "vm.overcommit_ratio: 50 (" in text
    assert "overcommit policy: strict" in text

    assert "memory.limit_in_bytes: 524288000 bytes (500.0 MiB)" in text
    assert "memory.usage_in_bytes: 104857600 bytes (100.0 MiB)" in text
    assert "memory.stat hierarchical_memory_limit: 524288000 bytes (500.0 MiB)" in text
    assert "memory.stat total_rss: 104857600 bytes (100.0 MiB)" in text
    assert "cgroup source: mount root (nested v1 group /nested/group not present)" in text


def test_a_zero_commit_limit_does_not_produce_a_ratio(tmp_path: Path) -> None:
    """Overcommit policy 1 reports ``CommitLimit: 0 kB``; do not divide by it."""
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("CommitLimit: 0 kB\nCommitted_AS: 1024 kB\n")

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "Cannot allocate memory"),
        proc_root=proc,
        cgroup_root=tmp_path / "no-cgroup",
        pid=1,
    )

    assert "CommitLimit: 0 kB (0.0 MiB)" in text
    assert "Committed_AS / CommitLimit: unavailable (a reading is missing)" in text


def test_no_command_line_or_argv_text_reaches_the_dump(
    fake_proc: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Credential-bearing command lines must never be carried into the artifact.

    A command line can contain a token passed as an argument, and this artifact
    is uploaded for anyone to read; redacting it would mean anticipating every
    spelling of every secret. The dump therefore carries only pid, ppid, RSS
    and name, and never opens ``/proc/<pid>/cmdline`` at all. This writes the
    marker into every per-process cmdline file and into both argv surfaces a
    naive implementation might reach for, then asserts neither marker appears
    anywhere in the rendered dump.
    """
    line_marker = "LEAKMARKER-ARGV-31337"
    argv_marker = "LEAKMARKER-AUX-90210"
    map_marker = "LEAKMARKER-MAP-PATH-77777"
    for entry in fake_proc.iterdir():
        if entry.is_dir() and (entry / "cmdline").exists():
            (entry / "cmdline").write_bytes(
                f"--label {line_marker} pytest".replace(" ", "\x00").encode() + b"\x00"
            )
    (fake_proc / "101" / "maps").write_text(
        f"7f0000000000-7f0000001000 r-xp 00000000 08:01 42 /opt/{map_marker}/lib.so\n"
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
        root_pid=100,
    )

    assert line_marker not in text
    assert argv_marker not in text
    assert map_marker not in text
    assert "/opt/" not in text
    # Removing the text did not remove the reading: the tree still roots at the
    # controller from the worker's parent pid, and the map table is counted.
    assert "tree root pid: 100" in text
    assert "map_count: 1 (" in text


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
    # The process's cgroup file is unreadable here, and the block says it
    # fell back to the mount root.
    assert "cgroup source: mount root;" in text


def test_one_unreadable_cgroup_v1_file_is_rendered_as_unavailable(
    fake_proc: Path, tmp_path: Path
) -> None:
    """A failed read on the limit must not read as an absent controller.

    Mutation that must fail this: ``continue`` on a failed read (the shipped
    round-1 behaviour), which printed the usage line and never mentioned the
    limit.
    """
    memory_dir = tmp_path / "cgroup-one" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "memory.limit_in_bytes").mkdir()
    (memory_dir / "memory.usage_in_bytes").write_text("123\n")
    (memory_dir / "memory.stat").write_text("hierarchical_memory_limit 456\ntotal_rss 123\n")

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "cgroup-one",
        pid=101,
    )

    assert "memory.limit_in_bytes: unavailable: IsADirectoryError" in text
    assert "memory.usage_in_bytes: 123 bytes (0.0 MiB)" in text
    assert "no cgroup v1 memory controller" not in text


def test_all_unreadable_cgroup_v1_files_are_not_reported_as_absent(
    fake_proc: Path, tmp_path: Path
) -> None:
    """Every file failing is still four stated reads, not a missing controller."""
    memory_dir = tmp_path / "cgroup-all" / "memory"
    memory_dir.mkdir(parents=True)
    for name in ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.stat"):
        (memory_dir / name).mkdir()

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "cgroup-all",
        pid=101,
    )

    assert "memory.limit_in_bytes: unavailable: IsADirectoryError" in text
    assert "memory.usage_in_bytes: unavailable: IsADirectoryError" in text
    assert "memory.stat: unavailable: IsADirectoryError" in text
    assert "no cgroup v1 memory controller" not in text


def test_a_file_where_the_cgroup_v1_directory_belongs_is_not_absence(
    fake_proc: Path, tmp_path: Path
) -> None:
    """``memory`` occupied by a regular file is an error, not a missing controller."""
    root = tmp_path / "cgroup-file"
    root.mkdir()
    (root / "memory").write_text("")

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=root,
        pid=101,
    )

    assert "memory.limit_in_bytes: unavailable: NotADirectoryError" in text
    assert "memory.usage_in_bytes: unavailable: NotADirectoryError" in text
    assert "memory.stat: unavailable: NotADirectoryError" in text
    assert "no cgroup v1 memory controller" not in text


def test_a_missing_cgroup_v1_controller_is_reported_as_absent(
    fake_proc: Path, tmp_path: Path
) -> None:
    """Only a controller directory that does not exist is an absence."""
    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "cgroup-absent",
        pid=101,
    )

    assert "(no cgroup v1 memory controller at" in text
    assert "unavailable" not in text.split("-- cgroup v1 memory", 1)[1].split("--", 1)[0]


def test_cgroup_v2_read_errors_are_reported_per_file(tmp_path: Path) -> None:
    """The v2 block states each failed read instead of dropping it."""
    root = tmp_path / "cgroup-v2-broken"
    root.mkdir()
    (root / "memory.max").mkdir()
    (root / "memory.current").write_text("100\n")
    (root / "memory.peak").mkdir()

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=tmp_path / "no-proc",
        cgroup_root=root,
        pid=101,
    )

    assert "memory.max: unavailable: IsADirectoryError" in text
    assert "memory.current: 100 bytes (0.0 MiB)" in text
    assert "memory.peak: unavailable: IsADirectoryError" in text
    assert "no cgroup v2 hierarchy" not in text


def test_a_nested_cgroup_v2_ceiling_is_preferred_over_the_mount_root(
    fake_proc: Path, tmp_path: Path
) -> None:
    """A container or scope ceiling is lower than the root's; read the nested one."""
    root = tmp_path / "cgroup-v2-nested"
    nested = root / "nested" / "group"
    nested.mkdir(parents=True)
    (root / "memory.max").write_text("max\n")
    (nested / "memory.max").write_text("536870912\n")
    (nested / "memory.current").write_text("1048576\n")
    (nested / "memory.peak").write_text("2097152\n")

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=root,
        pid=101,
    )

    assert f"-- cgroup v2 memory ({nested}) --" in text
    assert "cgroup source: nested v2 group /nested/group" in text
    assert "memory.max: 536870912 bytes (512.0 MiB)" in text
    assert "memory.current: 1048576 bytes (1.0 MiB)" in text
    assert "memory.peak: 2097152 bytes (2.0 MiB)" in text


def test_a_nested_cgroup_v1_ceiling_is_preferred_over_the_mount_root(
    fake_proc: Path, tmp_path: Path
) -> None:
    """The v1 memory controller resolves the same nested path."""
    root = tmp_path / "cgroup-v1-nested"
    nested = root / "memory" / "nested" / "group"
    nested.mkdir(parents=True)
    (root / "memory" / "memory.limit_in_bytes").write_text("555\n")
    (nested / "memory.limit_in_bytes").write_text("524288000\n")
    (nested / "memory.usage_in_bytes").write_text("104857600\n")
    (nested / "memory.stat").write_text(
        "hierarchical_memory_limit 524288000\ntotal_rss 104857600\n"
    )

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=root,
        pid=101,
    )

    assert f"-- cgroup v1 memory ({nested}) --" in text
    assert "cgroup source: nested v1 group /nested/group" in text
    assert "memory.limit_in_bytes: 524288000 bytes (500.0 MiB)" in text
    assert "memory.limit_in_bytes: 555 bytes" not in text


def test_the_v1_unlimited_sentinel_renders_as_unlimited(fake_proc: Path, tmp_path: Path) -> None:
    """The near-LONG_MAX page count is a no-limit value, like v2's ``max``."""
    memory_dir = tmp_path / "cgroup-unlimited" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (memory_dir / "memory.usage_in_bytes").write_text("1234\n")
    (memory_dir / "memory.stat").write_text(
        "hierarchical_memory_limit 9223372036854771712\ntotal_rss 1234\n"
    )

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=fake_proc,
        cgroup_root=tmp_path / "cgroup-unlimited",
        pid=101,
    )

    assert "memory.limit_in_bytes: unlimited" in text
    assert "memory.stat hierarchical_memory_limit: unlimited" in text
    assert "9223372036854771712" not in text


def test_the_process_list_is_capped_and_reports_the_omitted_count(tmp_path: Path) -> None:
    """The cap bounds the file; the true count and the worker rows stay.

    The failing case is a runaway spawner, so a list that grows with the failure
    is worst exactly when it matters. The controller and the reporting worker are
    pinned above the RSS cut because they are the two rows the question is about.
    """
    table = [memory.ProcInfo(pid=1000, ppid=1, name="python", rss_kb=500_000)]
    table.extend(
        memory.ProcInfo(
            pid=1001 + index,
            ppid=1000,
            name="python",
            rss_kb=1_000 + index,
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
        root_pid=1000,
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
    (entry / "limits").mkdir()
    (entry / "maps").mkdir()

    text = memory.build_report(
        nodeid="tests/unit/x.py::test_y",
        error=OSError(errno.ENOMEM, "boom"),
        now=MOMENT,
        proc_root=proc,
        cgroup_root=tmp_path / "no-cgroup",
        pid=101,
    )

    assert "unavailable: IsADirectoryError" in text
    assert "map_count: unavailable: IsADirectoryError" in text
    assert "max_map_count: unavailable: FileNotFoundError" in text
    assert "map_count / max_map_count: unavailable (a reading is missing)" in text
    assert "open file descriptors: unavailable: FileNotFoundError" in text
    assert "processes owned by uid: unavailable: IsADirectoryError" in text
    assert "commit readings unavailable: IsADirectoryError" in text
    assert "Committed_AS / CommitLimit: unavailable (a reading is missing)" in text
    assert "vm.overcommit_memory: unavailable: FileNotFoundError" in text
    assert "(no cgroup v1 memory controller at" in text
    assert "memory.stat: unavailable: IsADirectoryError" not in text
    # The process is still represented by its stat line, with no RSS.
    assert "processes in tree: 1" in text
    assert re.search(r"^\s*101\s+100\s+-", text, re.MULTILINE)
    assert "no cgroup v2" in text


def test_a_worker_missing_from_the_table_still_reports_the_rows_it_has(
    tmp_path: Path,
) -> None:
    """A race that drops the worker is not a reason to return an empty table."""
    table = [
        memory.ProcInfo(pid=200, ppid=1, name="python", rss_kb=10),
        memory.ProcInfo(pid=201, ppid=200, name="python", rss_kb=20),
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


def test_the_controller_hint_is_taken_only_from_an_xdist_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plain ``-n0`` run's parent is a shell; rooting there reports siblings.

    Mutation that must fail this: use ``os.getppid()`` without checking
    ``workerinput`` -- a serialised local run would dump the shell's whole
    child tree, which is not this shard. A reparented worker (parent pid 1)
    must also fall back rather than root at init.
    """
    monkeypatch.setattr(memory.os, "getppid", lambda: 4242)
    assert memory._controller_pid_hint(_Config({"workerid": "gw3"})) == 4242
    assert memory._controller_pid_hint(_Config(None)) is None
    monkeypatch.setattr(memory.os, "getppid", lambda: 1)
    assert memory._controller_pid_hint(_Config({"workerid": "gw3"})) is None


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
