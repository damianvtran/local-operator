"""Guards for :mod:`tests.shard_stall_watchdog`, the shard-stall reporter.

These exist because the reporter has exactly one job -- name the test a CI shard
is stuck on -- and every way of failing at that job is silent. A watchdog that
reports "0 still in flight" during a real stall, or stays inert because a
workflow env var was renamed, or floods the log with stale stacks, all look like
a healthy run. Each assertion below is written against one of those, and the
xdist report-phase rule is mutation-tested: flipping ``when == "teardown"`` to
"any report completes" makes
``test_a_setup_report_does_not_count_as_completion`` fail.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests import shard_stall_watchdog as watchdog


@pytest.fixture(autouse=True)
def _isolate_module_state(monkeypatch, tmp_path: Path):
    """Every test owns the module's process-wide state and its dump directory.

    The dump path is a module-level constant computed from ``TMPDIR`` at import,
    so a test that let the real one through would write into the shared temp dir
    that a concurrent suite is also using -- and the CI workflow's own reporting
    step reads that same directory.
    """
    monkeypatch.setattr(watchdog, "DUMP_DIR", tmp_path / "lo-shard-stall")
    monkeypatch.setattr(watchdog, "_CONTROLLER", None)
    monkeypatch.setattr(watchdog, "_WORKER", None)
    yield


class _Sink:
    """Captures reports instead of racing the terminal."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> None:
        self.texts.append(text)


class _Report:
    """The two attributes ``note_report`` reads off a pytest TestReport."""

    def __init__(self, nodeid: str, when: str) -> None:
        self.nodeid = nodeid
        self.when = when


class _FakeConfig:
    def __init__(self, worker: bool) -> None:
        if worker:
            self.workerinput = {"workerid": "gw0"}


def test_inert_without_the_env_var(monkeypatch) -> None:
    """Unset means inert, so a developer run and the ``-n0`` e2e stage are untouched.

    Inert has to mean *no thread and no directory*, not merely a disabled
    report: this hook runs in every pytest process in the repo, including the
    e2e stage that carries its own tighter bound.
    """
    monkeypatch.delenv(watchdog.ENV_SECONDS, raising=False)
    assert watchdog.enabled_seconds() is None
    watchdog.install(_FakeConfig(worker=False))
    assert watchdog._CONTROLLER is None


def test_a_malformed_bound_disables_rather_than_shrinks(monkeypatch) -> None:
    """A typo in the workflow must not become a shorter bound.

    ``float("4m")`` raising would take out ``pytest_configure`` and fail the
    whole suite; silently treating it as 0 would make every run report a stall.
    Neither is acceptable, so the only safe reading of an unparseable value is
    "not configured".
    """
    monkeypatch.setenv(watchdog.ENV_SECONDS, "4m")
    assert watchdog.enabled_seconds() is None
    monkeypatch.setenv(watchdog.ENV_SECONDS, "-5")
    assert watchdog.enabled_seconds() is None
    monkeypatch.setenv(watchdog.ENV_SECONDS, "240")
    assert watchdog.enabled_seconds() == 240.0


def test_a_setup_report_does_not_count_as_completion(monkeypatch) -> None:
    """The bug this test pins: xdist reports ``setup`` BEFORE the call phase.

    Counting any report as completion empties the in-flight set for the whole
    duration of the test body, so a real stall reports "0 still in flight" --
    measured while building this module, and it read as a healthy quiet run
    rather than as a broken instrument.
    """
    sink = _Sink()
    controller = watchdog._install_controller_for_test(0.01, sink)
    nodeid = "tests/unit/tui/test_x.py::test_a_long_one"

    watchdog.note_start(nodeid)
    watchdog.note_report(_Report(nodeid, "setup"))
    controller._last_progress = time.monotonic() - 5.0

    assert controller.report_if_stalled() is True
    assert nodeid in sink.texts[0], "the in-flight test must be named by node id"


def test_the_report_names_the_test_and_its_elapsed_time(monkeypatch) -> None:
    """The deliverable: a node id and how long it has been stuck."""
    sink = _Sink()
    controller = watchdog._install_controller_for_test(0.01, sink)
    under_test = "tests/unit/providers/test_oauth_flows.py::test_something"
    watchdog.note_start(under_test)
    watchdog.note_report(_Report("tests/unit/other.py::test_fast", "teardown"))
    controller._in_flight[under_test] = time.monotonic() - 42.0
    controller._last_progress = time.monotonic() - 60.0

    assert controller.report_if_stalled() is True
    report = sink.texts[0]
    assert "SHARD STALL" in report
    assert under_test in report
    assert "42s" in report or "4" in report


def test_teardown_completes_a_test_and_resets_the_clock() -> None:
    """A finished test is progress: it leaves the in-flight set and the clock."""
    sink = _Sink()
    controller = watchdog._install_controller_for_test(0.01, sink)
    watchdog.note_start("tests/unit/a.py::test_one")
    # Rewound BEFORE the teardown report, which is the only way to show that the
    # report itself is what resets it -- rewinding afterwards would make the
    # assertion below true no matter what note() did.
    controller._last_progress = time.monotonic() - 5.0
    watchdog.note_report(_Report("tests/unit/a.py::test_one", "teardown"))

    assert controller.report_if_stalled() is False
    assert controller._in_flight == {}


def test_a_stall_with_nothing_in_flight_is_still_reported() -> None:
    """The one case where the controller is the last process alive.

    Silence from a worker it believes is running something is a *different*
    diagnosis from a named stuck test, and suppressing it would hide the only
    case that has no other witness.
    """
    sink = _Sink()
    controller = watchdog._install_controller_for_test(0.01, sink)
    controller._last_progress = time.monotonic() - 5.0

    assert controller.report_if_stalled() is True
    assert "0 still in flight" in sink.texts[0]


def test_reports_are_rate_limited() -> None:
    """A stalled shard must not bury its own log.

    The cap can arrive mid-stall, so the reporter repeats -- but every repeat
    costs a stack excerpt, and an unthrottled one would push the first (and
    most useful) report out of a cancelled job's retained output.
    """
    sink = _Sink()
    controller = watchdog._install_controller_for_test(0.01, sink)
    watchdog.note_start("tests/unit/a.py::test_one")
    controller._last_progress = time.monotonic() - 5.0

    assert controller.report_if_stalled() is True
    assert controller.report_if_stalled() is False


def test_only_one_snapshot_is_printed_and_unfired_dumps_are_skipped(tmp_path) -> None:
    """Stack excerpts come from a real dump, and only from one that fired.

    ``repeat=True`` appends a whole new dump every interval, so a naive reader
    would print the same stacks until the log cap; and a file whose timer was
    armed but never fired is a header only, which must not be reported as a
    timeout. The excerpt must also carry the INNERMOST frames, which is where
    the stuck call is, so a test frame has to be asserted rather than assumed.
    """
    fired = watchdog._dump_path("worker-1")
    fired.write_text(
        f"[shard stall] test_x exceeded 4s\n{watchdog.FIRED_MARKER}0:04)!\n"
        "Thread 0x1 (most recent call first):\n"
        '  File "/repo/tests/unit/tui/test_x.py", line 12 in test_x\n'
        "\n"
        f"{watchdog.FIRED_MARKER}0:04)!\n"
        "Thread 0x1 (most recent call first):\n"
        '  File "/repo/tests/unit/tui/test_x.py", line 12 in test_x\n'
    )
    unfired = watchdog._dump_path("worker-2")
    unfired.write_text("[shard stall] test_y exceeded 4s\n")

    sink = _Sink()
    controller = watchdog._install_controller_for_test(0.01, sink)
    controller._last_progress = time.monotonic() - 10.0
    watchdog.note_start("tests/unit/tui/test_x.py::test_x")
    controller.report_if_stalled()
    report = sink.texts[0]

    assert report.count("most recent call first") == 1, "printed more than one snapshot"
    assert "test_x.py" in report, "the innermost frames are the point of the dump"
    assert "worker-2" not in report, "a dump that never fired was reported as a timeout"


def test_cleanup_keeps_fired_dumps_and_removes_armed_ones() -> None:
    """A fired dump is evidence; a header from an armed timer is noise.

    Keeping the noise would leave every healthy run's temp dir looking like a
    hang report, which is precisely how a diagnostic stops being read -- the
    same discipline ``tests.e2e.watchdog`` applies to its own dump files.
    """
    fired = watchdog._dump_path("worker-fired")
    fired.write_text(f"header\n{watchdog.FIRED_MARKER}0:04)!\nstacks\n")
    armed = watchdog._dump_path("worker-armed")
    armed.write_text("header only\n")

    controller = watchdog._install_controller_for_test(60.0, _Sink())
    controller.cleanup()

    assert fired.exists()
    assert not armed.exists()


def test_the_workflow_enables_the_watchdog() -> None:
    """The instrument is useless if its env var is renamed or dropped.

    Asserted against ``ci.yml`` rather than against this module, because the
    failure mode is a workflow edit -- the module cannot notice that nothing
    sets the variable, and an inert watchdog is indistinguishable from a suite
    that never stalls.
    """
    import yaml

    repo = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((repo / ".github" / "workflows" / "ci.yml").read_text())["jobs"]
    env = jobs["test"].get("env") or {}
    assert env.get(watchdog.ENV_SECONDS), f"the test job does not set {watchdog.ENV_SECONDS}"
    assert float(env[watchdog.ENV_SECONDS]) > 0
