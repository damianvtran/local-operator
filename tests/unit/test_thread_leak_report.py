"""Guards for the root conftest's REPORT-ONLY thread-leak reporter.

The reporter exists to name the tests that leave threads behind in the CI shard
that dies of ``OSError: [Errno 12]`` (see the section header in ``conftest.py``
for the measured evidence). It is reporting only, so the properties worth
pinning are not "does it fail the run" but:

* a test whose threads are still there after a short grace is named with BOTH
  deltas -- ``at_boundary`` and ``after_grace`` -- because a thread alive at the
  teardown report is not a thread the test kept: ``RelayServer.stop()`` closes
  its links and lets its daemon loops exit on their next tick;
* a test whose threads vanish during that grace is wind-down, not a leak, and
  gets an aggregate count instead of a per-test line;
* a test that inherits a large count and grows nothing is NOT reported, because
  an absolute reading would blame every later test in a shard for its
  predecessor's threads and name none of the real leakers;
* a clean test neither enumerates threads nor waits (the per-test cost is two
  ``active_count()`` reads and nothing else);
* the per-test lines are capped while the counters keep counting, and the
  summary still reports the true after-grace total;
* an internal error is swallowed and noted once, never raised into a test.

Every count below is synthetic: the reporter's own reads go through
``threading.active_count``/``threading.enumerate``, and those are monkeypatched,
so these tests neither create threads nor depend on what the suite happens to be
running beside them. The grace is injected as a recording no-op wait for the
same reason: no test here pays real wall time.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import pathlib
import re
import subprocess
import sys
import threading
import types
from typing import Any

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFTEST = REPO / "conftest.py"


def _load_hook_module() -> types.ModuleType:
    """Import the ROOT ``conftest.py`` under a private name.

    Loaded fresh under a distinct name for the same reason
    ``test_xdist_worker_budget.py`` does it: pytest has already imported the real
    one as a plugin, and monkeypatching the live object's globals would change
    the behaviour of the session running these tests.
    """
    spec = importlib.util.spec_from_file_location("_thread_leak_under_test", CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook_module() -> types.ModuleType:
    return _load_hook_module()


class _FakeThread:
    """Just the two attributes the reporter reads: never a stack or locals."""

    __slots__ = ("daemon", "name")

    def __init__(self, name: str, *, daemon: bool = False) -> None:
        self.name = name
        self.daemon = daemon


def _pin_counts(monkeypatch: pytest.MonkeyPatch, *values: int) -> None:
    """Pin the reporter's view of the live count, extra reads repeating the last.

    Repeating rather than exhausting matters beyond neatness: the live guard of
    the session running these tests calls the same function, and a
    ``StopIteration`` escaping from a monkeypatched probe would stand that guard
    down for the rest of the worker.
    """
    pending = list(values)

    def fake_active_count() -> int:
        if len(pending) > 1:
            return pending.pop(0)
        return pending[0]

    monkeypatch.setattr(threading, "active_count", fake_active_count)


def _pin_threads(monkeypatch: pytest.MonkeyPatch, *threads: _FakeThread) -> None:
    monkeypatch.setattr(threading, "enumerate", lambda: list(threads))


def _report(
    hook_module: types.ModuleType, lines: list[str], *, waits: list[float] | None = None
) -> Any:
    """A reporter whose grace records the wait instead of sleeping it."""
    return hook_module._ThreadLeakReport(
        "gw7",
        sink=lines.append,
        grace_seconds=0.25,
        wait=(waits if waits is not None else []).append,
    )


def test_a_thread_still_alive_after_the_grace_is_the_leak(hook_module, monkeypatch) -> None:
    """The line an operator greps: both readings, both deltas, the surviving name."""
    lines: list[str] = []
    waits: list[float] = []
    _pin_counts(monkeypatch, 5, 8, 7)
    # Two different populations: names sampled BEFORE the grace would say "pre",
    # so the assertion below is what pins "sampled after".
    pre_grace = [_FakeThread(f"pre{index}") for index in range(8)]
    post_grace = [_FakeThread(f"post{index}", daemon=True) for index in range(7)]
    monkeypatch.setattr(threading, "enumerate", lambda: list(post_grace if waits else pre_grace))
    report = _report(hook_module, lines, waits=waits)

    report.start("tests/unit/test_x.py::test_leaks")
    report.finish("tests/unit/test_x.py::test_leaks")

    assert waits == [0.25], "the grace is paid exactly once, by the flagged test"
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[thread-leak] gw7:")
    assert "tests/unit/test_x.py::test_leaks" in line
    assert "threads_at_start=5" in line
    assert "threads_at_end=8" in line
    assert "at_boundary=+3" in line
    assert "after_grace=+2" in line
    # Two threads still there after the grace: the last two names, both daemons.
    assert "names=['post5', 'post6']" in line
    assert "daemon=2/2" in line
    assert report.tests_reported == 1
    assert report.total_leaked == 2
    assert report.boundary_tests == 1
    assert report.boundary_threads == 3
    assert report.wind_down_tests == 0


def test_a_thread_gone_by_the_grace_is_wind_down_not_a_leak(hook_module, monkeypatch) -> None:
    """RelayServer.stop() does not join its loops; they exit on their next tick.

    A boundary reading calls that a leak. The grace is what separates it from a
    thread the test kept, and the wind-down aggregate is where it is recorded.
    """
    lines: list[str] = []
    waits: list[float] = []
    _pin_counts(monkeypatch, 5, 6, 5)
    monkeypatch.setattr(
        threading,
        "enumerate",
        lambda: pytest.fail("a wind-down must not sample names"),
    )
    report = _report(hook_module, lines, waits=waits)

    report.start("tests/unit/test_x.py::test_relay_stop")
    report.finish("tests/unit/test_x.py::test_relay_stop")

    assert waits == [0.25]
    assert lines == [], "a wind-down must not get a per-test leak line"
    assert report.tests_reported == 0
    assert report.total_leaked == 0
    assert report.boundary_tests == 1
    assert report.boundary_threads == 1
    assert report.wind_down_tests == 1
    assert report.wind_down_threads == 1

    summary = report.summary()
    assert len(summary) == 1
    assert "wind-down: 1 test(s) had 1 thread(s) gone by the grace" in summary[0]
    assert "at_boundary total 1 thread(s) over 1 test(s)" in summary[0]


def test_a_clean_test_is_silent_never_enumerates_and_never_waits(hook_module, monkeypatch) -> None:
    """Quiet by default, and cheap: no enumeration and no grace without a delta."""
    lines: list[str] = []
    waits: list[float] = []
    reads: list[int] = []

    def fake_active_count() -> int:
        reads.append(1)
        return 5

    monkeypatch.setattr(threading, "active_count", fake_active_count)
    monkeypatch.setattr(
        threading,
        "enumerate",
        lambda: pytest.fail("enumerate() must not be called on a clean test"),
    )
    report = _report(hook_module, lines, waits=waits)

    report.start("tests/unit/test_x.py::test_clean")
    report.finish("tests/unit/test_x.py::test_clean")

    assert len(reads) == 2, "a clean test reads the count twice and no more"
    assert waits == [], "a clean test must not pay the wind-down grace"
    assert lines == []
    assert report.summary() == []
    assert report.tests_reported == 0
    assert report.peak == 5


def test_inherited_threads_are_not_blamed_on_the_next_test(hook_module, monkeypatch) -> None:
    """The delta, not the absolute count.

    A shard's worker inherits every thread an earlier test left alive. A test
    that starts from 400 and ends at 400 grew nothing and must be silent; one
    that starts from 400 and ends at 401 is a +1 leaker, not a 401-thread one.
    """
    lines: list[str] = []
    waits: list[float] = []
    report = _report(hook_module, lines, waits=waits)
    _pin_threads(monkeypatch, _FakeThread("inherited", daemon=True))

    _pin_counts(monkeypatch, 400, 400)
    report.start("tests/unit/test_x.py::test_inherits_quietly")
    report.finish("tests/unit/test_x.py::test_inherits_quietly")
    assert lines == []
    assert report.summary() == []
    assert report.peak == 400
    assert waits == []

    _pin_counts(monkeypatch, 400, 401, 401)
    report.start("tests/unit/test_x.py::test_inherits_and_leaks_one")
    report.finish("tests/unit/test_x.py::test_inherits_and_leaks_one")

    assert waits == [0.25]
    assert len(lines) == 1
    assert "at_boundary=+1" in lines[0], "the absolute count leaked into the delta"
    assert "after_grace=+1" in lines[0]
    assert "threads_at_start=400" in lines[0]
    assert "threads_at_end=401" in lines[0]
    assert report.total_leaked == 1
    assert report.peak == 401


def test_per_test_lines_are_capped_and_the_summary_still_counts(hook_module, monkeypatch) -> None:
    """A shard-wide leak must not write one line per test, and must not under-count."""
    lines: list[str] = []
    waits: list[float] = []
    counter = itertools.count()
    monkeypatch.setattr(threading, "active_count", lambda: next(counter))
    _pin_threads(monkeypatch, _FakeThread("pool", daemon=True))
    report = _report(hook_module, lines, waits=waits)
    leaking = 45

    for index in range(leaking):
        nodeid = f"tests/unit/test_x.py::test_leak_{index}"
        report.start(nodeid)
        report.finish(nodeid)

    # Three reads per test out of one counter: start=n, end=n+1, settled=n+2.
    cap = hook_module._PER_TEST_LINE_CAP
    assert len(lines) == cap + 1, "one line per leak plus a single cap notice"
    assert f"capped at {cap}" in lines[-1]
    assert report.lines_emitted == cap
    assert report.suppressed == leaking - cap
    assert report.tests_reported == leaking
    assert report.total_leaked == leaking * 2
    assert report.boundary_threads == leaking
    assert len(waits) == leaking, "only flagged tests pay the grace"

    summary = report.summary()
    assert len(summary) == 1
    assert f"{leaking} test(s) still held {leaking * 2} thread(s)" in summary[0]
    assert "after a 250ms grace" in summary[0]
    assert f"peak {leaking * 3 - 1} live" in summary[0]
    assert summary[0].count("::") == hook_module._WORST_CAP, "the worst list must be capped too"
    assert f"{leaking - cap} further line(s) suppressed" in summary[0]


def test_the_summary_names_the_worst_post_grace_deltas_first(hook_module, monkeypatch) -> None:
    """Worst N by after_grace, so the first line of triage is the loudest leaker."""
    lines: list[str] = []
    report = _report(hook_module, lines)
    _pin_threads(monkeypatch, *[_FakeThread(f"t{index}") for index in range(8)])

    for delta in (1, 7, 3):
        _pin_counts(monkeypatch, 10, 10 + delta, 10 + delta)
        report.start(f"tests/unit/test_x.py::test_{delta}")
        report.finish(f"tests/unit/test_x.py::test_{delta}")

    summary = report.summary()
    assert len(summary) == 1
    assert summary[0].index("+7") < summary[0].index("+3") < summary[0].index("+1")
    assert "3 test(s) still held 11 thread(s)" in summary[0]


def test_the_grace_is_configurable_and_clamped(hook_module, monkeypatch) -> None:
    """Default 250 ms, overridable, and a typo cannot turn the shard into sleeps."""
    monkeypatch.delenv(hook_module._GRACE_ENV, raising=False)
    assert hook_module._thread_leak_grace_seconds() == 0.25
    monkeypatch.setenv(hook_module._GRACE_ENV, "500")
    assert hook_module._thread_leak_grace_seconds() == 0.5
    monkeypatch.setenv(hook_module._GRACE_ENV, "not-a-number")
    assert hook_module._thread_leak_grace_seconds() == 0.25
    monkeypatch.setenv(hook_module._GRACE_ENV, "-50")
    assert hook_module._thread_leak_grace_seconds() == 0.0
    monkeypatch.setenv(hook_module._GRACE_ENV, "999999")
    assert hook_module._thread_leak_grace_seconds() == hook_module._MAX_GRACE_MS / 1000


def test_an_internal_error_is_swallowed_and_noted_once(hook_module, monkeypatch, capsys) -> None:
    """REPORT-ONLY: a broken probe cannot become a broken or noisy run."""
    _pin_threads(monkeypatch, _FakeThread("pool"))
    report = _report(hook_module, [])
    monkeypatch.setattr(hook_module, "_THREAD_LEAK", report)
    monkeypatch.setattr(hook_module, "_THREAD_LEAK_FAILED", False)

    def boom() -> int:
        raise OSError("probe exploded")

    monkeypatch.setattr(threading, "active_count", boom)

    hook_module._thread_leak_start("tests/unit/test_x.py::test_one")
    hook_module._thread_leak_start("tests/unit/test_x.py::test_two")

    captured = capsys.readouterr()
    assert captured.err.count("stood down") == 1, "the note must appear exactly once"
    assert hook_module._THREAD_LEAK_FAILED is True
    # Once stood down, the hooks return without touching the probe again.
    hook_module._thread_leak_finish("tests/unit/test_x.py::test_one")
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("value", "enabled"),
    (
        ("", True),
        (None, True),
        ("1", True),
        ("true", True),
        ("0", False),
        ("off", False),
        ("NO", False),
    ),
)
def test_the_kill_switch_is_off_only_when_it_says_so(
    hook_module, monkeypatch, value, enabled
) -> None:
    """Default ON -- a diagnostic that has to be armed is absent when a shard dies."""
    if value is None:
        monkeypatch.delenv(hook_module._LEAK_ENV, raising=False)
    else:
        monkeypatch.setenv(hook_module._LEAK_ENV, value)
    assert hook_module._thread_leak_enabled() is enabled


def test_only_the_process_that_runs_tests_arms_the_guard(hook_module, monkeypatch) -> None:
    """The xdist controller sees every report but runs no test, so it must not arm.

    A controller-side counter would read its own thread table -- where a relay
    thread appearing mid-run looks exactly like a leak -- and label it with a
    process that never executed a test.
    """
    monkeypatch.setenv(hook_module._LEAK_ENV, "1")

    def config(**kwargs: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            workerinput=kwargs.get("workerinput"),
            option=types.SimpleNamespace(numprocesses=kwargs.get("numprocesses")),
        )

    monkeypatch.setattr(hook_module, "_THREAD_LEAK", None)
    hook_module._install_thread_leak_report(config(numprocesses="auto"))
    assert hook_module._THREAD_LEAK is None, "the controller armed a reporter"

    monkeypatch.setattr(hook_module, "_THREAD_LEAK", None)
    hook_module._install_thread_leak_report(config(workerinput={"workerid": "gw3"}))
    assert hook_module._THREAD_LEAK is not None
    assert hook_module._THREAD_LEAK.worker == "gw3"

    monkeypatch.setattr(hook_module, "_THREAD_LEAK", None)
    hook_module._install_thread_leak_report(config(numprocesses=0))
    assert hook_module._THREAD_LEAK is not None
    assert hook_module._THREAD_LEAK.worker == "main"

    monkeypatch.setenv(hook_module._LEAK_ENV, "0")
    monkeypatch.setattr(hook_module, "_THREAD_LEAK", None)
    hook_module._install_thread_leak_report(config(numprocesses=0))
    assert hook_module._THREAD_LEAK is None, "the kill-switch did not stop arming"


def test_the_real_hooks_report_a_leak_and_not_its_inheritor(tmp_path: pathlib.Path) -> None:
    """The WIRING, not the reporter: a real pytest run must carry the line.

    Every other case here calls ``_ThreadLeakReport`` directly, so deleting the
    ``_thread_leak_start``/``_thread_leak_finish`` calls from
    ``pytest_runtest_logstart``/``pytest_runtest_logreport`` would leave them all
    green while the guard silently stopped being called. This runs a child
    pytest over a temporary file and asserts on what a CI log would carry.

    The child is pointed at the root conftest with ``-p conftest`` (from
    ``cwd=REPO``, so the module resolves): a test file outside the repo rootdir
    gets no conftest chain of its own. ``-n 1`` keeps a worker in the picture,
    because the summary's worker label is part of what is being asserted.
    """
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import threading\n"
        "import time\n"
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.fixture\n"
        "def leaky():\n"
        "    threading.Thread(\n"
        "        target=time.sleep, args=(30,), name='probe-leak', daemon=True\n"
        "    ).start()\n"
        "    yield\n"
        "\n"
        "\n"
        "def test_a_leaks(leaky):\n"
        "    pass\n"
        "\n"
        "\n"
        "def test_b_inherits_but_leaks_nothing():\n"
        "    pass\n"
    )
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator"),
        "PATH": os.environ.get("PATH", ""),
        "TERM": "xterm-256color",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(probe),
            "-p",
            "conftest",
            "-q",
            "-n",
            "1",
            "-p",
            "no:randomly",
            "--basetemp",
            str(tmp_path / "child-pytest"),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert "2 passed" in output, output
    # Exactly two lines: one leak, one summary. A duplicated conftest import or
    # a second registration would print more, and the inheritor would print one.
    assert output.count("[thread-leak]") == 2, output
    assert "test_b_inherits_but_leaks_nothing" not in output, "the inheritor was blamed"

    leak = re.search(
        r"\[thread-leak\] (gw\d+): test_probe\.py::test_a_leaks "
        r"threads_at_start=(\d+) threads_at_end=(\d+) at_boundary=\+(\d+) "
        r"after_grace=\+(\d+) daemon=\d+/\d+ names=\[.*probe-leak.*\]",
        output,
    )
    assert leak is not None, output
    assert leak.group(1) == "gw0"
    assert int(leak.group(4)) == 1, "the boundary delta is not the one the fixture created"
    assert int(leak.group(5)) == 1, "a 30 s thread outlives the grace, so it is the leak"
    assert int(leak.group(3)) - int(leak.group(2)) == 1

    assert re.search(
        r"\[thread-leak\] gw0: 1 test\(s\) still held 1 thread\(s\) " r"after a 250ms grace",
        output,
    ), output
