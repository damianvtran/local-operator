"""Guards for the root conftest's ADVISORY ``[fd-leak]`` reporter.

The reporter names tests that leave descriptors open (the cause of PR #4472's
``filedescriptor out of range in select()`` failures). It never fails a run, so
what is worth pinning is: deltas rather than absolutes are blamed, per-file
totals net out, the summary is bounded, and an internal error stands it down
instead of reaching a test. Counts are synthetic (an injected counter), so these
tests neither open descriptors nor depend on the worker's real fd table.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def hook_module() -> types.ModuleType:
    # A private copy, as test_thread_leak_report.py does: patching the live
    # conftest would change the reporter observing this very session.
    spec = importlib.util.spec_from_file_location("_fd_growth_under_test", REPO / "conftest.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counter(*values: int):
    pending = list(values)
    return lambda: pending.pop(0)


def test_growth_is_blamed_on_the_test_and_its_file(hook_module) -> None:
    lines: list[str] = []
    # a: 1000 -> 1000 (inherits a big table, grows nothing); b: 1000 -> 1007;
    # c (same file as b): 1007 -> 1005.
    report = hook_module._FdGrowthReport(
        "gw0", _counter(1000, 1000, 1000, 1007, 1007, 1005), sink=lines.append
    )
    for nodeid in ("t/a.py::a", "t/b.py::b", "t/b.py::c"):
        report.start(nodeid)
        report.finish(nodeid)

    summary = report.summary()
    assert summary[0].startswith("[fd-leak] gw0: 3 test(s); open fds first=1000 last=1005")
    assert "net=+5" in summary[0] and "1 test(s) ended with more fds" in summary[0]
    assert "[fd-leak] gw0: file t/b.py net=+5" in summary
    assert "[fd-leak] gw0: test t/b.py::b +7" in summary
    assert not any("t/a.py" in line for line in summary), "an inherited count is not a leak"
    assert lines == [], "nothing is printed per test"


def test_a_run_where_nothing_grew_prints_nothing(hook_module) -> None:
    report = hook_module._FdGrowthReport("main", _counter(20, 20, 20, 19))
    for nodeid in ("t/a.py::a", "t/a.py::b"):
        report.start(nodeid)
        report.finish(nodeid)
    assert report.summary() == []


def test_summary_is_capped(hook_module) -> None:
    values = [value for index in range(100) for value in (0, index + 1)]
    report = hook_module._FdGrowthReport("main", _counter(*values), worst_cap=3)
    for index in range(100):
        report.start(f"t/f{index}.py::x")
        report.finish(f"t/f{index}.py::x")
    summary = report.summary()
    tests = [line for line in summary if " test t/" in line]
    files = [line for line in summary if " file t/" in line]
    assert [line.rsplit(" ", 1)[1] for line in tests] == ["+100", "+99", "+98"]
    assert len(files) == 3


def test_an_internal_error_stands_the_reporter_down(hook_module) -> None:
    def broken() -> int:
        raise OSError("no fd table")

    hook_module._FD_GROWTH = hook_module._FdGrowthReport("main", broken)
    hook_module._fd_growth_start("t/x.py::x")  # must not raise
    assert hook_module._FD_GROWTH is None
    hook_module._fd_growth_finish("t/x.py::x")
    hook_module._fd_growth_summary()


def test_the_real_counter_sees_a_new_descriptor(hook_module) -> None:
    import os

    before = hook_module._count_open_fds()
    fd = os.open(os.devnull, os.O_RDONLY)
    try:
        assert hook_module._count_open_fds() == before + 1
    finally:
        os.close(fd)


def test_the_off_switch(hook_module, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_FD_LEAK_REPORT", "0")
    hook_module._FD_GROWTH = None
    config = types.SimpleNamespace(option=types.SimpleNamespace(numprocesses=0))
    hook_module._install_fd_growth_report(config)
    assert hook_module._FD_GROWTH is None
    monkeypatch.delenv("LOCAL_OPERATOR_FD_LEAK_REPORT")
    hook_module._install_fd_growth_report(config)
    assert hook_module._FD_GROWTH is not None
