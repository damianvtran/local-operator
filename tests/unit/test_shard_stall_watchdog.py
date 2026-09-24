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


def test_inert_on_ci_without_the_env_var(monkeypatch) -> None:
    """A CI job that did not ask for the shard bound stays exactly as inert as before.

    That is what keeps the ``-n0`` e2e stage (one process, its own tighter bound)
    out of this module's worker branch: only CI's explicit value can enable it
    there. Inert means *no thread and no directory*, not merely a disabled report
    -- this hook runs in every pytest process in the repo.
    """
    monkeypatch.delenv(watchdog.ENV_SECONDS, raising=False)
    monkeypatch.delenv(watchdog.LOCAL_ENV_SECONDS, raising=False)
    assert watchdog.enabled_seconds() is None
    assert watchdog.local_seconds(on_ci=True) is None


def test_an_explicit_local_bound_is_honoured_on_ci_too(monkeypatch) -> None:
    """The off-switch arm must not swallow a value the operator typed in.

    ``local_seconds`` refuses to INVENT a bound on CI; it does not refuse one it was
    handed. Worth a guard because the two arms live next to each other and a
    later edit that returns early on ``on_ci`` -- a plausible reading of "CI keeps
    the shard bound only" -- would silently drop the bound on the one machine where
    a stall costs a cancelled job.
    """
    monkeypatch.delenv(watchdog.ENV_SECONDS, raising=False)
    monkeypatch.setenv(watchdog.LOCAL_ENV_SECONDS, "120")
    assert watchdog.local_seconds(on_ci=True) == 120.0
    assert watchdog.local_seconds(on_ci=False) == 120.0


def test_the_local_default_reports_without_any_env_var(monkeypatch, capsys) -> None:
    """The point of the local default: silence is what cost a local run its evening.

        Measured 2026-09-24: a whole-tree local run under load average 80-200 reached
        69% and then its log stopped -- because a wall-clock ``timeout`` killed the
    group, not because a test hung, and the log tail (a ``PluggyTeardownRaisedWarning``
        and ``OSError: cannot send (already closed?)`` from orphaned workers) could not
        say which. Nothing in the run could have told the reader, so the reader watched
        it by hand for four hours. This guard is the fix: absent an explicit value, a
        local run instruments itself.
    """
    monkeypatch.delenv(watchdog.ENV_SECONDS, raising=False)
    monkeypatch.delenv(watchdog.LOCAL_ENV_SECONDS, raising=False)

    assert watchdog.local_seconds(on_ci=False) == watchdog.LOCAL_DEFAULT_SECONDS
    watchdog.install(_FakeConfig(worker=False), on_ci=False)
    assert watchdog._CONTROLLER is not None
    assert watchdog._CONTROLLER.seconds == watchdog.LOCAL_DEFAULT_SECONDS
    # One line, naming the switch that turns it off: an instrument that reports on
    # a healthy run has to be silenceable by the person reading it.
    notice = capsys.readouterr().err
    assert watchdog.LOCAL_ENV_SECONDS in notice and "=0" in notice


def test_the_local_default_is_silenceable_and_overridable(monkeypatch) -> None:
    """A debugger session needs it OFF, and a bound someone typed in must win.

    ``0``/``false``/``no``/``off`` is the off switch (any case, the repo's usual
    flag reading, and the same set the root conftest's worker-cap flag uses).
    An explicit value wins over the default in both directions, because the only
    reason to set it is that the default is wrong for this host or this session.
    """
    monkeypatch.delenv(watchdog.ENV_SECONDS, raising=False)
    for off in ("0", "false", "NO", "off"):
        monkeypatch.setenv(watchdog.LOCAL_ENV_SECONDS, off)
        assert watchdog.local_seconds(on_ci=False) is None, off
    monkeypatch.setenv(watchdog.LOCAL_ENV_SECONDS, "45")
    assert watchdog.local_seconds(on_ci=False) == 45.0


def test_a_malformed_local_bound_falls_back_to_the_default(monkeypatch) -> None:
    """A typo must not be read as "no reporting" -- that is the failure mode here.

    Falling back to the default is safe in a way a shorter bound would not be:
    the default only prints, and it is already sized above any legitimate item, so
    the worst a typo can do is report at the bound nobody chose. Disabling instead
    would turn a five-character mistake into the silence this module exists to
    remove -- and that silence is indistinguishable from a healthy run.
    """
    monkeypatch.delenv(watchdog.ENV_SECONDS, raising=False)
    for typo in ("4m", "-5", "inf", "nan"):
        monkeypatch.setenv(watchdog.LOCAL_ENV_SECONDS, typo)
        assert watchdog.local_seconds(on_ci=False) == watchdog.LOCAL_DEFAULT_SECONDS, typo


def test_the_off_switch_spellings_match_the_worker_cap_flag() -> None:
    """One reading of ``0``/``false``/``no``/``off`` per module, and they must agree.

    Two modules now decide whether an env flag is off (this one and the root
    conftest's worker cap), and they are read by the same person in the same
    session: `PYTEST_QUIET_WORKER_CAP=0` silences one and
    `LOCAL_OPERATOR_LOCAL_STALL_SECONDS=0` silences the other. A second reading
    that treated `off` as ON would be a defect nobody notices until the value they
    typed failed to take effect -- so the sets are pinned equal here rather than
    kept equal by a comment.
    """
    import importlib.util

    # The root conftest under a private name: pytest has already imported the real
    # one as a plugin, and it must not be monkeypatched (see
    # `tests/unit/test_xdist_worker_budget.py::_load_hook_module`).
    spec = importlib.util.spec_from_file_location(
        "_conftest_flags_under_test", Path(__file__).resolve().parents[2] / "conftest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._FALSY_ENV_VALUES == watchdog.FALSY_ENV_VALUES


def test_the_hard_bound_is_off_unless_asked_for(monkeypatch) -> None:
    """Report-only is the default everywhere, including CI.

    A fired hard bound kills the process that hits it -- a worker carrying
    unrelated tests -- so it cannot be the default; ``exit=True`` only ever comes
    from an explicit request (see the module docstring).
    """
    monkeypatch.delenv(watchdog.TEST_TIMEOUT_ENV, raising=False)
    assert watchdog.per_test_bound() is None
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv(watchdog.TEST_TIMEOUT_ENV, off)
        assert watchdog.per_test_bound() is None, off
    monkeypatch.setenv(watchdog.TEST_TIMEOUT_ENV, "4m")
    assert watchdog.per_test_bound() is None, "a typo must not start killing workers"


def test_the_hard_bound_is_sized_from_the_manifest_and_floored(monkeypatch) -> None:
    """The bound comes from ``tests/durations.json``, never from taste or a clock.

    Two directions, because both are the sizing claim: a heavy file gets slack over
    its own measured total (so a legitimately slow test under fleet load cannot
    trip it), and a file the manifest does not mention gets the floor rather than
    an exemption -- an unbounded new file is exactly where a hang would hide.
    """
    monkeypatch.setenv(watchdog.TEST_TIMEOUT_ENV, "1")
    assert watchdog.per_test_bound() == (None, True)

    heavy = "tests/unit/tui/test_settings_view.py::test_one"
    measured = watchdog._manifest_seconds(heavy)
    assert measured is not None and measured > watchdog.BOUND_FLOOR_S
    assert watchdog.sized_bound_seconds(heavy) == pytest.approx(measured * watchdog.BOUND_SLACK)
    assert watchdog.sized_bound_seconds(heavy) > SLOWEST_LEGITIMATE_ITEM_S

    unknown = "tests/unit/tui/test_a_file_the_manifest_does_not_mention.py::test_one"
    assert watchdog._manifest_seconds(unknown) is None
    assert watchdog.sized_bound_seconds(unknown) == watchdog.BOUND_FLOOR_S


#: The worst legitimate single item AGENTS.md records for a local run, in seconds
#: (top of the Environment section: "the worst legitimate test measured in a full
#: local run is 81s"). It is here as a number because the sizing claim above needs
#: one: a bound under this would fail healthy runs.
SLOWEST_LEGITIMATE_ITEM_S = 81.0


def test_a_typed_bound_wins_over_the_sized_one(monkeypatch) -> None:
    """The escape hatch, for someone who has already seen the sized bound misfire."""
    monkeypatch.setenv(watchdog.TEST_TIMEOUT_ENV, "12.5")
    assert watchdog.per_test_bound() == (12.5, True)


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
    # Deterministic: the in-flight entry was rewound 42.0s at :118, so the
    # elapsed column reads exactly "42s". The earlier `or "4" in report`
    # passed on any digit 4 anywhere in the text and asserted nothing.
    assert "42s" in report


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


class _FakeFaulthandler:
    """Stands in for :mod:`faulthandler` without touching the real module.

    The C timer is the whole point of the worker half, and no unit test can make
    a real one fire, so it is spied on instead: the assertions are that the timer
    was armed for the running test, at the bound captured at install, and
    cancelled only by that test's teardown.
    """

    def __init__(self) -> None:
        self.armed: list[tuple[float, dict[str, object]]] = []
        self.cancels = 0

    def dump_traceback_later(self, seconds: float, **kwargs) -> None:
        self.armed.append((seconds, kwargs))

    def cancel_dump_traceback_later(self) -> None:
        self.cancels += 1


def _install_worker(monkeypatch, bound: str = "4") -> _FakeFaulthandler:
    """Install the worker branch and return the C-timer spy it will call."""
    monkeypatch.setenv(watchdog.ENV_SECONDS, bound)
    fake = _FakeFaulthandler()
    monkeypatch.setattr(watchdog, "faulthandler", fake)
    watchdog.install(_FakeConfig(worker=True))
    return fake


def test_the_worker_arms_the_c_timer_and_writes_the_header(monkeypatch) -> None:
    """The worker half produces the stacks; a no-op ``arm`` must fail a test.

    ``arm`` is the only thing that starts the C timer, and the round-1 review
    measured that turning it into an early ``return`` left all ten guards green
    while no dump could ever be produced -- the reporter would name the test and
    hand the reader no stacks at all.
    """
    fake = _install_worker(monkeypatch)
    nodeid = "tests/unit/tui/test_x.py::test_a_long_one"

    watchdog.note_start(nodeid)

    worker = watchdog._WORKER
    assert worker is not None and worker._armed
    assert fake.armed == [(4.0, {"file": worker._handle, "repeat": True, "exit": False})]
    assert f"{watchdog.ARM_MARKER}{nodeid} exceeded 4s" in worker._path.read_text(encoding="utf-8")


def test_the_worker_timer_is_cancelled_only_by_teardown(monkeypatch) -> None:
    """A call-phase report must not cancel the worker's timer.

    This is the worker-side twin of the setup-phase defect the controller guard
    already pins: cancelling on any report kills the timer for the rest of the
    test body, so a stall there produces no dump. Two round-1 mutants survived
    without it -- worker ``note_report`` disarming on any ``when``, and
    ``disarm`` becoming a no-op -- and the assertions below kill both.
    """
    fake = _install_worker(monkeypatch)
    nodeid = "tests/unit/tui/test_x.py::test_a_long_one"
    watchdog.note_start(nodeid)
    worker = watchdog._WORKER
    assert worker is not None and worker._armed

    for when in ("setup", "call"):
        watchdog.note_report(_Report(nodeid, when))
        assert worker._armed, f"a {when!r} report must not cancel the worker timer"
        assert fake.cancels == 0

    watchdog.note_report(_Report(nodeid, "teardown"))
    assert worker._armed is False
    assert fake.cancels == 1


def test_the_worker_keeps_the_bound_it_was_installed_with(monkeypatch) -> None:
    """Unsetting the variable after install must not raise inside the hook.

    ``note_start`` used to re-read the bound and pass it on unguarded, so an
    unset value reached ``dump_traceback_later(None)`` -> ``TypeError`` inside
    ``pytest_runtest_logstart`` -> ``INTERNALERROR``. The bound is captured once,
    at install, where it is known to be usable.
    """
    fake = _install_worker(monkeypatch)
    monkeypatch.delenv(watchdog.ENV_SECONDS, raising=False)

    watchdog.note_start("tests/unit/a.py::test_one")

    assert fake.armed and fake.armed[0][0] == 4.0


def test_the_worker_re_arms_per_item_and_sizes_the_sized_bound(monkeypatch) -> None:
    """One item is the instrument's unit, so a new test gets a new countdown.

    Arming ONCE per process (what this did before the hard bound existed) means the
    countdown an item inherits started while some earlier test was running: the
    first fast test to follow a slow one would be reported as a stall, and a test
    that parks early in its file would be waited out only after the previous test's
    bound had already elapsed. Two items, two arms, and the stale timer cancelled
    -- the assertion that both happen is what stops a later edit from making
    ``arm`` a first-call-only no-op again.

    The sized bound comes from ``tests/durations.json`` (see
    :func:`sized_bound_seconds`), so the two arms are asserted at each file's own
    value rather than at one number the test would have to hardcode twice.
    """
    monkeypatch.setenv(watchdog.TEST_TIMEOUT_ENV, "1")
    fake = _install_worker(monkeypatch, bound="4")
    heavy = "tests/unit/tui/test_settings_view.py::test_one"
    light = "tests/unit/analytics/test_model.py::test_one"

    watchdog.note_start(heavy)
    watchdog.note_start(light)

    worker = watchdog._WORKER
    assert worker is not None
    assert [armed for armed, _ in fake.armed] == [
        watchdog.sized_bound_seconds(heavy),
        watchdog.sized_bound_seconds(light),
    ]
    assert fake.cancels == 1, "the previous item's timer must not be left running"


def test_the_hard_bound_kills_the_process_instead_of_repeating(monkeypatch) -> None:
    """``exit=True, repeat=False`` is the opt-in; the default repeats and never exits.

    A repeating timer that exits would kill the worker at the first snapshot with
    its stacks half-written, and an exiting timer that repeated would be reported
    twice for one park. The two modes are therefore mutually exclusive by
    construction, and this guard is what says so -- the whole difference between
    "a report" and "a failure" lives in these two keywords.
    """
    monkeypatch.setenv(watchdog.TEST_TIMEOUT_ENV, "600")
    fake = _install_worker(monkeypatch, bound="4")

    watchdog.note_start("tests/unit/a.py::test_one")

    worker = watchdog._WORKER
    assert worker is not None
    assert fake.armed == [(600.0, {"file": worker._handle, "repeat": False, "exit": True})]


def test_an_unusable_dump_directory_disables_the_worker(monkeypatch, tmp_path) -> None:
    """An unusable ``TMPDIR`` must disable the instrument, not fail the shard.

    ``install`` runs from ``pytest_configure``, where an unguarded ``mkdir`` was
    measured to raise ``NotADirectoryError`` (``TMPDIR=/dev/null/x``) and
    ``FileExistsError`` (a file occupying the path) -- each an ``INTERNALERROR``
    that failed the whole shard with ``no tests ran``. That is the one way this
    diagnostic could turn a green run red, which its safety argument forbids.
    """
    occupied = tmp_path / "occupied"
    occupied.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setattr(watchdog, "DUMP_DIR", occupied / "lo-shard-stall")
    fake = _install_worker(monkeypatch)

    worker = watchdog._WORKER
    assert worker is not None
    watchdog.note_start("tests/unit/a.py::test_one")  # must not raise either
    assert worker._armed is False
    assert worker._disabled
    assert fake.armed == []


def test_the_report_mode_prints_fired_stacks_and_labels_unfired_headers(tmp_path) -> None:
    """The workflow step's output: evidence, plus honestly labelled noise.

    This is the surface a cancelled job leaves behind, so it carries the module's
    own FIRED_MARKER discipline rather than a raw ``cat``. An arm-time header is
    a claim about a test that STARTED; a cancelled shard's files held thousands
    of them (3088 measured), burying the one real timeout.
    """
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    (dumps / "worker-1.log").write_text(
        f"{watchdog.ARM_MARKER}tests/unit/a.py::test_fast exceeded 240s; every thread follows.\n"
        f"{watchdog.FIRED_MARKER}0:04:00)!\n"
        "Thread 0x1 (most recent call first):\n"
        '  File "/repo/tests/unit/a.py", line 12 in test_fast\n',
        encoding="utf-8",
    )
    (dumps / "worker-2.log").write_text(
        "".join(
            f"{watchdog.ARM_MARKER}tests/unit/a.py::test_{i} exceeded 240s; every thread follows.\n"
            for i in range(3000)
        ),
        encoding="utf-8",
    )

    report = watchdog.report_dumps(dumps)

    assert "fired" in report
    assert "line 12 in test_fast" in report
    # A fired file names its test too: the stacks say what was parked, but the
    # node id is what a reader of a cancelled run needs first.
    assert "in flight when the timer fired: tests/unit/a.py::test_fast" in report
    assert "never fired" in report
    assert "3000 tests started" in report
    assert "in flight at cancel, not fired: tests/unit/a.py::test_2999" in report
    # The defect: no arm-time header may be presented as a fired timeout.
    assert "exceeded 240s" not in report
    assert report.count(watchdog.FIRED_MARKER) == 1


def test_the_report_mode_does_not_read_a_marker_out_of_a_node_id(tmp_path) -> None:
    """A header containing the literal marker must not read as a fired dump.

    The filter matches the marker at the start of a line. Matching it anywhere
    would classify an arm-time header as a snapshot whenever a node id contained
    that text -- the one direction in which this filter can lie, and the report
    is the surface a cancelled job is judged by.
    """
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    nodeid = f"tests/unit/a.py::test_weird[{watchdog.FIRED_MARKER}0:04:00)]"
    (dumps / "worker-7.log").write_text(
        f"{watchdog.ARM_MARKER}{nodeid} exceeded 240s; every thread follows.\n",
        encoding="utf-8",
    )

    report = watchdog.report_dumps(dumps)

    assert "(fired," not in report, "an arm-time header was read as a snapshot"
    assert "never fired" in report
    assert f"in flight at cancel, not fired: {nodeid}" in report


def test_a_bound_no_timer_can_hold_disables_the_instrument(monkeypatch) -> None:
    """A parseable-but-oversized bound must disable, not kill the shard.

    ``float()`` accepts ``inf`` and ``nan``, and every finite value at or above
    ``2**63`` nanoseconds is beyond the ``time_t`` ``faulthandler`` converts to,
    where the C timer raises ``OverflowError`` from inside the first test's hook
    -- an ``INTERNALERROR`` that fails the shard with ``no tests ran``. That is
    the failure class this module's safety argument is written about, so the
    value has to be refused before it reaches the timer.
    """
    for raw in ("1e18", "1e10", "2e10", "inf", "1e400", "nan", "9223372036.854776"):
        monkeypatch.setenv(watchdog.ENV_SECONDS, raw)
        assert watchdog.enabled_seconds() is None, raw
    # The largest value the timer actually accepts is still honoured.
    monkeypatch.setenv(watchdog.ENV_SECONDS, "9223372035.854776")
    assert watchdog.enabled_seconds() == 9223372035.854776


def test_the_worker_survives_a_bound_the_timer_rejects(monkeypatch) -> None:
    """``arm`` swallows ``OverflowError`` too, belt and braces.

    ``enabled_seconds`` refuses the value at parse time, but a bound can reach
    the timer from a direct construction -- and this is the one call whose
    failure mode is the whole shard, so the guard names every exception the C
    timer is known to raise rather than trusting the caller.
    """

    class _OverflowingFaulthandler(_FakeFaulthandler):
        def dump_traceback_later(self, seconds, **kwargs):
            raise OverflowError("timestamp out of range for platform time_t")

    _install_worker(monkeypatch)
    monkeypatch.setattr(watchdog, "faulthandler", _OverflowingFaulthandler())

    watchdog.note_start("tests/unit/a.py::test_one")  # must not raise

    worker = watchdog._WORKER
    assert worker is not None and worker._disabled and not worker._armed


def test_the_report_mode_is_quiet_on_a_missing_directory(tmp_path, capsys) -> None:
    """The step runs on every shard, including the ones with nothing to say.

    The old step's ``for f in "$dir"/*.log`` ran once on the literal pattern when
    no file matched and ``bash -e`` exited 1, and the directory does not exist at
    all on a shard that never armed a timer. Neither may fail the step.
    """
    assert watchdog.main([str(tmp_path / "does-not-exist")]) == 0
    assert "no shard stall report" in capsys.readouterr().out


def test_the_workflow_reports_through_the_module_not_a_raw_cat() -> None:
    """The step must not re-implement the fired/unfired rule in YAML.

    Pinned against ``ci.yml`` because the failure mode is a workflow edit: a
    ``cat`` of ``*.log`` is exactly what printed 3088 false "exceeded 240s"
    claims on this PR's own cancelled shard while the one real timeout was lost
    among them.
    """
    import yaml

    repo = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((repo / ".github" / "workflows" / "ci.yml").read_text())["jobs"]
    step = next(s for s in jobs["test"]["steps"] if s.get("name") == "Print the shard stall report")
    assert "tests.shard_stall_watchdog" in step["run"]
    assert "cat " not in step["run"]
    assert "*.log" not in step["run"]


def test_no_ci_job_runs_both_watchdogs_in_one_process() -> None:
    """The two C-timer instruments never share a process in CI.

    ``faulthandler``'s timer is process-global, so a ``tests.e2e.watchdog.bounded``
    block inside a test body displaces an armed worker timer and leaves that test
    with no stacks. That is latent only because the shard job (which sets
    ENV_SECONDS, and deselects ``e2e`` via addopts) is the only job that arms a
    worker, while the e2e stage runs ``-n0`` with no variable. Pin the invariant
    rather than leaving it to prose.
    """
    import tomllib

    import yaml

    repo = Path(__file__).resolve().parents[2]
    jobs = yaml.safe_load((repo / ".github" / "workflows" / "ci.yml").read_text())["jobs"]
    config = tomllib.loads((repo / "pyproject.toml").read_text())
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]

    # Both halves of the invariant: the shard job is the one that arms a worker
    # timer, and it must keep deselecting the e2e stage whose own C timer would
    # displace ours. Without the second assertion, dropping `-m "not e2e"` from
    # addopts would falsify the documented claim while this pin stayed green.
    assert (jobs["test"].get("env") or {}).get(
        watchdog.ENV_SECONDS
    ), "the test job must arm the watchdog"
    assert "not e2e" in addopts, "the shard job must keep deselecting the e2e stage"
    for name, job in jobs.items():
        if name == "test":
            continue
        assert not (job.get("env") or {}).get(
            watchdog.ENV_SECONDS
        ), f"{name} arms the shard watchdog; it must not also run the e2e C timer"
