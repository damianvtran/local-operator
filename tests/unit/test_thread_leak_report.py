"""Guards for the root conftest's REPORT-ONLY thread-leak reporter.

The reporter exists to name the tests that leave threads behind in the CI shard
that dies of ``OSError: [Errno 12]`` (see the section header in ``conftest.py``
for the measured evidence). It is reporting only, so the properties worth
pinning are not "does it fail the run" but:

* a test that grows the live thread count is named with its DELTA;
* a test that inherits a large count and grows nothing is NOT reported, because
  an absolute reading would blame every later test in a shard for its
  predecessor's threads and name none of the real leakers;
* a clean test does not even enumerate threads (the per-test cost is two
  ``active_count()`` reads and nothing else);
* the per-test lines are capped while the counters keep counting, and the
  summary still reports the true total;
* an internal error is swallowed and noted once, never raised into a test.

Every count below is synthetic: the reporter's own reads go through
``threading.active_count``/``threading.enumerate``, and those are monkeypatched,
so these tests neither create threads nor depend on what the suite happens to be
running beside them.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
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


def _report(hook_module: types.ModuleType, lines: list[str]) -> Any:
    return hook_module._ThreadLeakReport("gw7", sink=lines.append)


def test_a_leaking_test_is_named_with_its_delta(hook_module, monkeypatch) -> None:
    """The one line an operator has to be able to grep out of a CI log."""
    lines: list[str] = []
    _pin_counts(monkeypatch, 5, 8)
    _pin_threads(
        monkeypatch,
        *[_FakeThread(f"t{index}", daemon=index >= 5) for index in range(8)],
    )
    report = _report(hook_module, lines)

    report.start("tests/unit/test_x.py::test_leaks")
    report.finish("tests/unit/test_x.py::test_leaks")

    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[thread-leak] gw7:")
    assert "tests/unit/test_x.py::test_leaks" in line
    assert "before=5" in line
    assert "after=8" in line
    assert "delta=+3" in line
    # The tail of the live table: three threads by delta, five names at most.
    assert "names=['t5', 't6', 't7']" in line
    assert "daemon=3/3" in line
    assert report.tests_reported == 1
    assert report.total_leaked == 3


def test_a_clean_test_is_silent_and_never_enumerates(hook_module, monkeypatch) -> None:
    """Quiet by default, and cheap: no enumeration unless a delta was found."""
    lines: list[str] = []
    _pin_counts(monkeypatch, 5, 5)
    monkeypatch.setattr(
        threading,
        "enumerate",
        lambda: pytest.fail("enumerate() must not be called on a clean test"),
    )
    report = _report(hook_module, lines)

    report.start("tests/unit/test_x.py::test_clean")
    report.finish("tests/unit/test_x.py::test_clean")

    assert lines == []
    assert report.summary() is None
    assert report.tests_reported == 0
    assert report.peak == 5


def test_inherited_threads_are_not_blamed_on_the_next_test(hook_module, monkeypatch) -> None:
    """The delta, not the absolute count.

    A shard's worker inherits every thread an earlier test left alive. A test
    that starts from 400 and ends at 400 grew nothing and must be silent; one
    that starts from 400 and ends at 401 is a +1 leaker, not a 401-thread one.
    """
    lines: list[str] = []
    report = _report(hook_module, lines)
    _pin_threads(monkeypatch, _FakeThread("inherited", daemon=True))

    _pin_counts(monkeypatch, 400, 400)
    report.start("tests/unit/test_x.py::test_inherits_quietly")
    report.finish("tests/unit/test_x.py::test_inherits_quietly")
    assert lines == []
    assert report.summary() is None
    assert report.peak == 400

    _pin_counts(monkeypatch, 400, 401)
    report.start("tests/unit/test_x.py::test_inherits_and_leaks_one")
    report.finish("tests/unit/test_x.py::test_inherits_and_leaks_one")

    assert len(lines) == 1
    assert "delta=+1" in lines[0], "the absolute count leaked into the delta"
    assert "before=400" in lines[0]
    assert "after=401" in lines[0]
    assert report.total_leaked == 1
    assert report.peak == 401


def test_per_test_lines_are_capped_and_the_summary_still_counts(hook_module, monkeypatch) -> None:
    """A shard-wide leak must not write one line per test, and must not under-count."""
    lines: list[str] = []
    counter = itertools.count()
    monkeypatch.setattr(threading, "active_count", lambda: next(counter))
    _pin_threads(monkeypatch, _FakeThread("pool", daemon=True))
    report = _report(hook_module, lines)
    leaking = 45

    for index in range(leaking):
        nodeid = f"tests/unit/test_x.py::test_leak_{index}"
        report.start(nodeid)
        report.finish(nodeid)

    cap = hook_module._PER_TEST_LINE_CAP
    assert len(lines) == cap + 1, "one line per leak plus a single cap notice"
    assert f"capped at {cap}" in lines[-1]
    assert report.lines_emitted == cap
    assert report.suppressed == leaking - cap
    assert report.tests_reported == leaking
    assert report.total_leaked == leaking

    summary = report.summary()
    assert summary is not None
    assert f"{leaking} test(s) left {leaking} thread(s) behind" in summary
    assert f"peak {leaking * 2 - 1} live" in summary
    assert summary.count("::") == hook_module._WORST_CAP, "the worst list must be capped too"
    assert f"{leaking - cap} further line(s) suppressed" in summary


def test_the_summary_names_the_worst_deltas_first(hook_module, monkeypatch) -> None:
    """Worst N by delta, so the first line of triage is the loudest leaker."""
    lines: list[str] = []
    report = _report(hook_module, lines)
    _pin_threads(monkeypatch, *[_FakeThread(f"t{index}") for index in range(8)])

    for delta in (1, 7, 3):
        _pin_counts(monkeypatch, 10, 10 + delta)
        report.start(f"tests/unit/test_x.py::test_{delta}")
        report.finish(f"tests/unit/test_x.py::test_{delta}")

    summary = report.summary()
    assert summary is not None
    assert summary.index("+7") < summary.index("+3") < summary.index("+1")
    assert "3 test(s) left 11 thread(s) behind" in summary


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
