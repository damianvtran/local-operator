"""A supervised connector stops itself when the install moved past its own build.

WHY THIS EXISTS. The connector is always-on (``launchd`` ``KeepAlive``, a systemd
``Restart=always``), so it outlives every update that moves the generation
pointer underneath it: the promotion happens, the pointer moves, and the process
that is already running keeps serving the build it was exec'd from. Measured
2026-10-04, that is how a daemon ends up importing from a pruned tree days after
a released fix. Measured 2026-10-30 in the connector's own terms: pid 48311
(generation 20261023T142851Z-0.86.1) held
``<config-dir>/team_bridge/connector.lock``, so 2,826 supervised starts of
``com.local-operator.bridge-connector`` were refused with exit 75 and nothing
inbound was admitted for days.

WHAT IS PINNED, each cell because the opposite is a real failure:

* a SUPERVISED connector on a superseded build sets the same ``stop`` the signal
  handlers set, so the ordinary graceful path runs and the process exits 0 —
  under a supervisor that is a restart onto ``current``, not a crash;
* a FOREGROUND connector only warns, once, and keeps running: exiting there is a
  connector that silently vanished;
* ``False`` (both sides read, they match) and ``None`` (not comparable) never
  act. ``None`` is "no generation in this argv, or an unreadable pointer" — never
  "older";
* the uptime gate is checked BEFORE the probe forks, which is why there is no
  check before the lock: a connector started during a flip must not read that
  flip's own pointer move as staleness and restart-loop.

Time is shrunk through the module's constants, never by sleeping: the interval
becomes a few ``asyncio.sleep`` ticks and ``_PROCESS_STARTED_S`` is moved, so a
five-minute gate is one patched value.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import pytest

from local_operator.team_bridge.connector import __main__ as main_mod

LOG = "local_operator.team_bridge.connector"


def _fast(
    monkeypatch: pytest.MonkeyPatch,
    *,
    min_uptime: float = 0.0,
    uptime: float = 3_600.0,
    is_stale: bool | None = True,
) -> list[str]:
    """Open the timing gates and replace the probe; return its call log.

    The probe's own answer logic lives in
    ``update.generation_staleness_of_process`` and is pinned there. What these
    cells are about is what the WATCHER does with an answer, so the seam is
    replaced and every read is counted.
    """

    monkeypatch.setattr(main_mod, "SELF_CHECK_INTERVAL_S", 0.01)
    monkeypatch.setattr(main_mod, "SELF_CHECK_MIN_UPTIME_S", min_uptime)
    monkeypatch.setattr(main_mod, "_PROCESS_STARTED_S", time.monotonic() - uptime)
    reads: list[str] = []

    def probe() -> bool | None:
        reads.append("probe")
        return is_stale

    monkeypatch.setattr(main_mod, "_own_generation_is_stale", probe)
    return reads


def _run_watch(ticks: int = 3) -> asyncio.Event:
    """Run the watcher for a few ticks, cancel it, return the event it held."""

    async def drive() -> asyncio.Event:
        stop = asyncio.Event()
        task = asyncio.create_task(main_mod._watch_own_generation(stop))
        for _ in range(ticks):
            await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return stop

    return asyncio.run(drive())


def test_a_supervised_stale_connector_stops_through_the_graceful_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The whole point: exit 0 after the ordinary stop sequence, one reason line.

    ``_main_async`` is replaced by the only behaviour the real one has that this
    cell needs — it returns when ``stop`` is set — so what is asserted is
    ``_supervise``'s sequencing (stop -> grace -> 0), not the receive loop, which
    has its own file.
    """

    _fast(monkeypatch)
    monkeypatch.setattr(main_mod, "_is_supervised", lambda: True)

    async def fake_main(args: argparse.Namespace, stop: asyncio.Event | None = None) -> int:
        assert stop is not None
        await stop.wait()
        return 0

    monkeypatch.setattr(main_mod, "_main_async", fake_main)
    args = argparse.Namespace(config_dir="/nonexistent")
    with caplog.at_level(logging.INFO, logger=LOG):
        assert asyncio.run(main_mod._supervise(args)) == 0
    lines = [record.getMessage() for record in caplog.records]
    assert "bridge connector stopping: stage=stop code=stale_generation" in lines, lines
    # The self-check already said why; the signal line would name the wrong cause.
    assert "bridge connector stopping: stage=stop code=signal" not in lines, lines


def test_a_foreground_connector_warns_once_and_keeps_running(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No supervisor, no exit: one warning for the whole run, ``stop`` stays clear.

    Six ticks against a one-warning assertion is the point: a line per minute in
    an always-on process's log is noise the operator learns to skip.
    """

    _fast(monkeypatch)
    monkeypatch.setattr(main_mod, "_is_supervised", lambda: False)
    with caplog.at_level(logging.WARNING, logger=LOG):
        stop = _run_watch(ticks=6)
    assert not stop.is_set()
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == LOG
    ]
    assert warnings == ["bridge connector stale generation: stage=run code=stale_generation"]


@pytest.mark.parametrize("answer", [False, None])
def test_an_unproven_or_current_build_never_stops_the_connector(
    answer: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conclusion ``False`` and ``None`` are the same non-event, and only ``True``
    may act. ``None`` covers "no generation in this argv" (a pip/pipx install) and
    "a pointer that could not be read" — neither is evidence of staleness."""

    reads = _fast(monkeypatch, is_stale=answer)
    monkeypatch.setattr(main_mod, "_is_supervised", lambda: True)
    stop = _run_watch()
    assert reads, "the watcher never probed"
    assert not stop.is_set()


def test_a_connector_younger_than_the_uptime_gate_never_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE FLAP GUARD. A process started during a flip must not act on the move
    that flip caused, so uptime is checked before the probe forks at all."""

    reads = _fast(monkeypatch, min_uptime=300.0, uptime=1.0)
    monkeypatch.setattr(main_mod, "_is_supervised", lambda: True)
    stop = _run_watch(ticks=4)
    assert reads == []
    assert not stop.is_set()


@pytest.mark.parametrize(
    ("environ", "windows", "expected"),
    [
        ({"XPC_SERVICE_NAME": "com.local-operator.bridge-connector"}, False, True),
        ({"XPC_SERVICE_NAME": "0"}, False, False),
        ({}, False, False),
        ({"INVOCATION_ID": "9d1c"}, False, True),
        ({"XPC_SERVICE_NAME": "com.local-operator.bridge-connector"}, True, False),
    ],
)
def test_the_supervisor_marker_is_the_supervisors_own(
    environ: dict[str, str], windows: bool, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``XPC_SERVICE_NAME=0`` is what macOS puts in an interactive shell, so "the
    variable is set" is not the question; and Windows has no generation layout at
    all, so it must never answer yes.

    ``os.name`` is NOT patched here (``pathlib`` builds ``WindowsPath`` from it
    and that cannot be instantiated off Windows); the platform answer comes
    through the one seam the package already has for it.
    """

    for name in ("XPC_SERVICE_NAME", "INVOCATION_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("local_operator.procstate.is_windows", lambda: windows)
    assert main_mod._is_supervised() is expected
