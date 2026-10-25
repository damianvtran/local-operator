"""The relay's five lifecycle loops: joined at stop, and startable only once.

WHY THIS FILE EXISTS. ``RelayServer.stop()`` set ``_stop``, closed its listening
sockets and teardown of links, then returned — while every one of the five daemon
loops it started was still running. ``self._threads`` was append-only: nothing
joined it and nothing ever cleared it, so each relay generation (and each of the
hundreds of relay generations the network suite builds) left five threads behind
for the rest of the process. The suite's leak instrument measured the consequence:
200 tests leaked to a peak of 662 live threads, network tests contributing ~8-22
each, and a per-thread virtual reservation of 45.6-46.1 MiB — about 241 threads
exhaust the runner's 11,063 MiB CommitLimit. That is the ``OSError: [Errno 12]``
on the 3.13 shard, and the tests it names are victims rather than causes.

``start()`` had the matching hole: no guard at all, so a second call appended
five more loops beside the first five and re-ran the once-only start work.

THE CELLS ARE THE CONTRACT, in four readings:

* a real relay's five threads are all dead by the time ``stop()`` returns, with
  the whole stop inside its bound;
* one loop that ignores its stop produces exactly ONE warning, keeps its handle,
  and does not stop the other four from being joined;
* a second ``start()`` raises instead of starting a second fleet (this cell fails
  on the old behaviour, which silently returned);
* a repeated ``stop()`` re-joins and re-checks a survivor but does not warn twice,
  and a survivor that finally dies clears its entry.

THE FAKE-THREAD CELLS follow the #2913 pattern: ``join(timeout)`` returns nothing
whether or not the thread stopped, so the surviving branch is answered directly by
a stand-in whose ``is_alive()`` says True instead of wedging a real thread and
racing a sleep.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import relay
from tests.unit.network.test_relay_e2e import serve_shaped_relay


class _LoopThatWillNotStop:
    """A lifecycle loop whose bounded join returns while it keeps running.

    ``_join_loop_threads`` can only learn whether a join succeeded by asking
    ``is_alive()``, so the failing branch is answered directly rather than by
    holding a real thread for the bound.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.join_timeouts: list[float | None] = []
        self.alive = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return self.alive


class _LoopThatStops:
    """The clean arm: the join returns and the thread is already gone."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.join_timeouts: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)

    def is_alive(self) -> bool:
        return False


def _relay_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == relay.logger.name and record.levelno == logging.WARNING
    ]


def test_stop_joins_all_five_loops_and_clears_their_entries(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A real relay: five loops running, five dead when ``stop()`` returns.

    THE CLOCK IS THE DISCRIMINATOR against the old behaviour, not a sleep-and-hope:
    the heartbeat loop is replaced with one that answers ``_stop`` but takes 0.3 s
    to finish, so ``stop()`` returning sooner — as it used to — would leave that
    thread alive and fail here deterministically. The total bound is asserted too:
    a clean stop must not approach it.
    """
    server = serve_shaped_relay(root, monkeypatch)

    def heartbeat_that_takes_the_bound_to_finish() -> None:
        server._stop.wait(5.0)  # noqa: SLF001 — the lifecycle flag under test
        time.sleep(0.3)

    monkeypatch.setattr(server, "_heartbeat_loop", heartbeat_that_takes_the_bound_to_finish)
    server.start()
    threads = list(server._threads)  # noqa: SLF001 — the handles under test
    assert [thread.name for thread in threads] == [
        "mesh-accept",
        "mesh-control",
        "mesh-heartbeat",
        "mesh-membership",
        "mesh-redial",
    ], "the five lifecycle loops were not the five that started"

    with caplog.at_level(logging.WARNING, logger=relay.logger.name):
        started = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - started

    alive = [thread.name for thread in threads if thread.is_alive()]
    assert not alive, f"stop() returned with lifecycle loops still running: {alive}"
    assert (
        server._threads == []
    ), "a stopped loop must clear its entry, not stay in the append-only list"  # noqa: SLF001
    assert elapsed >= 0.3, (
        f"a clean stop took {elapsed:.2f}s; it returned before the delayed loop finished, "
        "so the join did not wait for it"
    )
    assert elapsed < relay.RELAY_LOOP_STOP_TOTAL_S, (
        f"a clean stop took {elapsed:.2f}s, past the {relay.RELAY_LOOP_STOP_TOTAL_S:.1f}s " "bound"
    )
    assert _relay_warnings(caplog) == [], "a clean stop must not warn about survivors"


def test_a_loop_that_ignores_its_stop_is_named_once_and_does_not_block_the_rest(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One survivor: exactly one warning, its handle retained, the others joined.

    The four clean stand-ins must each be joined even though the survivor's join
    timed out first, and the warning must name the loop and the timeout so an
    operator can tell which one survived and how long it was given.
    """
    server = relay.RelayServer(root=root)
    survivor = _LoopThatWillNotStop("mesh-heartbeat")
    clean = [
        _LoopThatStops("mesh-accept"),
        _LoopThatStops("mesh-control"),
        _LoopThatStops("mesh-membership"),
        _LoopThatStops("mesh-redial"),
    ]
    server._threads = cast(Any, [survivor, *clean])  # noqa: SLF001 — the handles under test
    try:
        with caplog.at_level(logging.WARNING, logger=relay.logger.name):
            started = time.monotonic()
            server.stop()
            elapsed = time.monotonic() - started

        assert all(
            loop.join_timeouts for loop in clean
        ), "a wedged loop stopped the join from reaching the other four"
        warnings = _relay_warnings(caplog)
        assert len(warnings) == 1, [record.getMessage() for record in warnings]
        message = warnings[0].getMessage()
        assert "mesh-heartbeat" in message, message
        assert f"{relay.RELAY_LOOP_STOP_JOIN_S:.1f}s" in message, message
        assert (
            survivor in server._threads
        ), "a survivor must stay reachable, or nothing can name it again"  # noqa: SLF001
        assert "mesh-accept" not in server._threads  # noqa: SLF001 — dead entries cleared
        assert (
            elapsed < relay.RELAY_LOOP_STOP_JOIN_S + 1.0
        ), f"the survivor held stop() for {elapsed:.2f}s"
    finally:
        server._threads.clear()  # noqa: SLF001 — do not leave a standing stand-in


def test_a_second_start_is_refused_instead_of_starting_a_second_fleet(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old behaviour appended five more loops; the new one raises.

    Refused both while running and after a stop: a stopped relay's sockets are
    closed and its ``_stop`` is set for good, so "started again" would be a lie
    whichever way it were spelled.
    """
    server = serve_shaped_relay(root, monkeypatch)
    server.start()
    try:
        with pytest.raises(RuntimeError, match="only be called once"):
            server.start()
        started_accepts = [
            thread for thread in server._threads if thread.name == "mesh-accept"  # noqa: SLF001
        ]
        assert (
            len(started_accepts) == 1
        ), f"the second start started another accept loop: {len(started_accepts)}"
    finally:
        server.stop()

    with pytest.raises(RuntimeError, match="a stopped relay cannot be restarted"):
        server.start()


def test_a_repeated_stop_re_joins_but_does_not_warn_twice(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The latch from #2913, on the loop fleet: one warning per survivor episode.

    The repeated stop must still join and re-check — that re-check is the only path
    that clears the handle once the survivor dies — while naming it only once. A
    dead survivor then clears its entry and its latch.
    """
    server = relay.RelayServer(root=root)
    survivor = _LoopThatWillNotStop("mesh-redial")
    server._threads = cast(Any, [survivor])  # noqa: SLF001 — the survivor under test
    try:
        with caplog.at_level(logging.WARNING, logger=relay.logger.name):
            server.stop()
            assert len(_relay_warnings(caplog)) == 1

            joins_after_first = len(survivor.join_timeouts)
            server.stop()
            assert (
                len(survivor.join_timeouts) > joins_after_first
            ), "a repeated stop must re-join the survivor to re-check is_alive()"
            assert len(_relay_warnings(caplog)) == 1, "the same episode was warned twice"
            assert survivor in server._threads  # noqa: SLF001

            survivor.alive = False
            server.stop()
            assert len(_relay_warnings(caplog)) == 1, "a dead survivor must stay silent"
            assert server._threads == []  # noqa: SLF001 — the entry cleared
    finally:
        server._threads.clear()  # noqa: SLF001 — do not leave a standing stand-in
