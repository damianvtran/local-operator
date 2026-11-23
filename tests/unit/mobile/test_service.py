"""The first-run refusal has to name a command that works on THIS host (A23).

``lop mobile serve`` with no password exits 2 and tells the operator what to do
about it. That sentence used to be the same one line everywhere — "Run `lop
mobile install`" — which is a command that cannot work on a host with no user
service supervisor at all (a container, OpenRC/Alpine, a Linux without systemd):
``mobile.install`` refuses there with ``supervisors.no_supervisor_error`` and
names the foreground command instead. The store in the sentence was
macOS-shaped for the same reason: "the Keychain" is only one of the three
stores ``mobile.auth`` now chooses between, and ``store_description()`` is the
single place that knows which.

The last test pins a different seam: the refusal path returns 2 *before* the
daemon is constructed, so nothing in this file (or the propagation suite, which
calls the method directly) executes ``amain``'s display-settings wiring.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from local_operator import supervisors
from local_operator.mobile import log_collapse, service

_LoggerState = tuple[int, bool, bool, list[Any], list[logging.Handler]]


@contextlib.contextmanager
def _process_logging_restored() -> Iterator[None]:
    """Put every logger back exactly as it was once ``amain`` has run.

    WHY THIS EXISTS. ``amain`` is a daemon entry point and configures the
    PROCESS's logging the way a daemon should, permanently: it replaces the
    root handlers (``configure_console_logging``), lowers wire-client levels
    (``quiet_wire_clients``), raises the push worker to INFO, and attaches a
    ``log_collapse.RepeatCollapse`` filter to the ``asyncio`` logger. Every
    test here runs it inside the pytest worker, so without this all of that
    outlived the test and leaked into whatever the worker ran next.

    THE MEASURED LEAK (merge-gate shard 3 on #4169 run 44281659052 and #4211
    run 44348031838, both interpreters): the collapse filter admits one
    ``Task was destroyed but it is pending!`` per 60 s and drops the rest at
    the LOGGER, before propagation, so caplog never sees them. On a worker
    that ran this file, a destroyed-task line from one test silenced the next
    one's, and ``test_session_plane``'s vacuity control ("the log capture
    cannot see asyncio's complaint") failed on PRs that touched neither area.

    A full snapshot of every logger rather than the four ``amain`` touches
    today, so a fifth daemon-only logging tweak cannot reopen the same leak.
    """
    manager = logging.Logger.manager

    def state(lg: logging.Logger) -> _LoggerState:
        return (lg.level, lg.propagate, lg.disabled, list(lg.filters), list(lg.handlers))

    saved: dict[str, _LoggerState] = {"": state(logging.root)}
    for name, lg in list(manager.loggerDict.items()):
        if isinstance(lg, logging.Logger):
            saved[name] = state(lg)
    try:
        yield
    finally:
        for name, lg in list(manager.loggerDict.items()):
            # A logger first created during the run had no prior state: the
            # Logger defaults are what "as it was" means for it.
            if isinstance(lg, logging.Logger) and name not in saved:
                saved[name] = (logging.NOTSET, True, False, [], [])
        for name, (level, propagate, disabled, filters, handlers) in saved.items():
            lg = logging.root if name == "" else logging.getLogger(name)
            lg.setLevel(level)
            lg.propagate = propagate
            lg.disabled = disabled
            lg.filters[:] = filters
            lg.handlers[:] = handlers


@pytest.fixture(autouse=True)
def _amain_logging_does_not_outlive_the_test() -> Iterator[None]:
    with _process_logging_restored():
        yield


def test_amains_logging_setup_is_undone_for_the_next_test(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression for the shard-3 vacuity-control failure (see the helper above).

    The control comes first: inside the guard ``amain`` really DOES install the
    collapse filter, so the restored state afterwards is not vacuously clean.
    Then two identical asyncio ERROR records must BOTH reach a handler — the
    exact pair the leaked filter used to cut to one.
    """
    asyncio_logger = logging.getLogger("asyncio")
    before = list(asyncio_logger.filters)
    with _process_logging_restored():
        _first_run_message(monkeypatch, capsys, supervisor=None, store="s")
        assert any(
            isinstance(f, log_collapse.RepeatCollapse) for f in asyncio_logger.filters
        ), "amain no longer installs the collapse filter; this test pins nothing"
    assert asyncio_logger.filters == before

    seen: list[logging.LogRecord] = []
    probe = logging.Handler()
    probe.emit = seen.append  # type: ignore[method-assign]
    asyncio_logger.addHandler(probe)
    try:
        for _ in range(2):
            asyncio_logger.error("Task was destroyed but it is pending!")
    finally:
        asyncio_logger.removeHandler(probe)
    assert len(seen) == 2, "a repeated asyncio complaint is still being collapsed"


def _first_run_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    supervisor: str | None,
    store: str,
) -> str:
    """Run ``amain`` with no password and return what it printed."""
    monkeypatch.setattr(service, "load_password", lambda: None)
    monkeypatch.setattr(service, "store_description", lambda: store)
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisor)

    assert asyncio.run(service.amain()) == 2, "an unauthenticated daemon must not bind"
    return capsys.readouterr().out


def test_a_host_with_a_supervisor_is_told_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _first_run_message(monkeypatch, capsys, supervisor="launchctl", store="the test store")

    assert "`lop mobile install`" in out
    assert "the test store" in out
    assert "LOP_MOBILE_PASSWORD" in out


def test_a_host_without_a_supervisor_is_not_sent_to_the_installer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _first_run_message(monkeypatch, capsys, supervisor=None, store="a 0600 file (/tmp/pw)")

    assert "`lop mobile install` cannot run here" in out
    assert "`lop mobile password`" in out
    assert "`lop mobile serve`" in out
    assert "a 0600 file (/tmp/pw)" in out


def test_the_store_in_the_message_is_the_platforms_own(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read from ``auth`` rather than spelled here, so it cannot drift again."""
    calls: list[int] = []
    real = service.store_description

    def counted() -> str:
        calls.append(1)
        return real()

    monkeypatch.setattr(service, "load_password", lambda: None)
    monkeypatch.setattr(service, "store_description", counted)
    monkeypatch.setattr(supervisors, "supervisor", lambda: "launchctl")

    asyncio.run(service.amain())

    assert calls == [1]
    assert real() in capsys.readouterr().out


def test_amain_wires_the_daemon_into_the_display_settings_follower() -> None:
    """The daemon boot must subscribe the cross-process ``display.*`` follower.

    ``service.amain``'s ``daemon.watch_display_settings()`` is the ONE seam no
    other test executes: the propagation tests call the method directly, and
    every test above returns 2 at the no-password refusal *before* the daemon
    is constructed. Deleting the call re-opens the daemon half of design
    review round 1's D1 with the suite green (agent review round 2, M1), so pin
    it the way this repo pins entry-point wiring — against ``amain``'s own
    source.
    """
    import inspect

    source = inspect.getsource(service.amain)
    assert "daemon.watch_display_settings()" in source, (
        "service.amain must call daemon.watch_display_settings(): without it a "
        "running relay keeps the display value from its first read until a "
        "restart, which is the D1 defect the follower exists to fix."
    )


def test_both_push_lane_paths_read_the_listing_count_of_the_daemons_own_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2026-11-14 defect's wiring: ``amain`` built the lane with no
    ``unread_count``, so the worker fell back to the lifetime census ("47502
    conversations need you"). Both construction paths — the lane built at start
    and the late-tunnel factory — must hand the worker the reader of the SAME
    table the daemon serves the badge from."""
    from local_operator.mobile import push_transport

    lanes: list[object] = []

    class FakeLane:
        def __init__(self, unread_count: object) -> None:
            self.unread_count = unread_count
            self.transport = type("T", (), {"computer": "computer-handle"})()

    def build(config_dir: object, *, unread_count: object = None) -> FakeLane:
        lane = FakeLane(unread_count)
        lanes.append(lane)
        return lane

    captured: dict[str, object] = {}

    class Stop(Exception):
        pass

    def daemon(**kwargs: object) -> None:
        captured.update(kwargs)
        raise Stop

    monkeypatch.setattr(service, "load_password", lambda: "pw123")
    monkeypatch.setattr(push_transport, "build_push_lane", build)
    monkeypatch.setattr(service, "MobileDaemon", daemon)
    monkeypatch.delenv("LO_MOBILE_NO_DIAL", raising=False)

    with pytest.raises(Stop):
        asyncio.run(service.amain())

    table = captured["table"]
    assert isinstance(table, service.SessionTable)
    start = captured["push_lane"]
    assert isinstance(start, FakeLane)
    assert start.unread_count == table.listing_unread_count
    factory = captured["push_lane_factory"]
    assert callable(factory)
    late = factory()
    assert isinstance(late, FakeLane)
    assert late.unread_count == table.listing_unread_count
    assert len(lanes) == 2
