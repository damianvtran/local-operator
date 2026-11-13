"""A silent peer must not be handed MORE work than a healthy one.

WHY THIS FILE EXISTS (2026-11-13, lop-remote-node-2). The node went completely
silent for 120 s at a time under session load. This Mac's relay then made it worse:
it pinged each silent link twice a second (frames_out 273/194/218 against frames_in
43/17/9), scanned every network file on disk per ping, let several surfaces dial the
same peer at once (two links to it idled out 5 s apart), and a writer killed by an
unexpected exception would have left a half-dead link for the peer to idle out.
Each rule is pinned here with reduced stand-ins and a fake clock, so no socket or
real wait is involved.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import redial, relay, wire

# ---------------------------------------------------------------------------
# Keepalive cadence and the writer's exception path
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, start: float) -> None:
        self.now = start

    def time(self) -> float:
        return self.now


class _ClockedWake:
    """The writer's ``_wake``: each wait advances the fake clock by its timeout and
    closes the link once ``stop_after_s`` of silence has elapsed."""

    def __init__(self, clock: _FakeClock, closed: threading.Event, stop_after_s: float) -> None:
        self._clock = clock
        self._closed = closed
        self._stop_at = clock.now + stop_after_s

    def wait(self, timeout: float | None = None) -> bool:
        self._clock.now += float(timeout or 0.0)
        if self._clock.now >= self._stop_at:
            self._closed.set()
        return False

    def set(self) -> None:
        pass

    def clear(self) -> None:
        pass


def _writer_link(server: Any) -> Any:
    link: Any = object.__new__(relay.PeerLink)
    link.server = server
    link.settings = relay.NetworkSettings(port=4097)
    link.link_id = "l_silent"
    link.device_id = "d_silent"
    link._closed = threading.Event()
    link._flushing = threading.Event()
    link._flushed = threading.Event()
    link._reliable = queue.Queue()
    link._droppable = {}
    link._droppable_lock = threading.Lock()
    return link


def test_a_silent_peer_gets_one_ping_per_keepalive_not_two_per_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock(1_000_000.0)
    monkeypatch.setattr(relay, "time", SimpleNamespace(time=clock.time))

    def no_disk_scan() -> Any:
        raise AssertionError("the keepalive must not rebuild the peer record per ping")

    link = _writer_link(SimpleNamespace(peer_record=no_disk_scan))
    link.last_frame_at = clock.now  # the peer's last frame: silence starts now
    link.last_ping_sent_at = 0.0
    link._wake = _ClockedWake(clock, link._closed, stop_after_s=link.settings.link_idle_s)
    written: list[dict[str, Any]] = []
    link._write = written.append

    link._write_loop()

    pings = [frame for frame in written if wire.is_keepalive(frame)]
    # 120 s of silence at a 30 s keepalive: pings near 30, 60 and 90 s. The old rule
    # pinged on every 0.5 s wake after the first 30 s, ~180 of them.
    assert 3 <= len(pings) <= 4, len(pings)
    assert all(ping["req"] == os.getpid() for ping in pings)


def test_a_writer_exception_outside_the_network_set_logs_and_closes_the_link(
    caplog: pytest.LogCaptureFixture,
) -> None:
    link = _writer_link(SimpleNamespace())
    link.last_frame_at = 0.0
    link.last_ping_sent_at = 0.0
    link._wake = threading.Event()
    closes: list[tuple[str, float]] = []

    def close(reason: str = "we-closed", *, flush_s: float = 0.0) -> None:
        closes.append((reason, flush_s))
        link._closed.set()

    def seal_fails(frame: dict[str, Any]) -> None:
        raise ValueError("Out of range float values are not JSON compliant")

    link.close = close
    link._write = seal_fails
    link._reliable.put({"op": "net_stream", "value": float("nan")})

    with caplog.at_level(logging.WARNING, logger=relay.logger.name):
        link._write_loop()  # returns: the thread does not die with the exception

    assert closes == [("error", 0.0)]
    rows = [r for r in caplog.records if "closed by its writer" in r.getMessage()]
    assert len(rows) == 1 and "ValueError" in rows[0].getMessage()
    assert rows[0].exc_info is not None, "an unexpected writer death carries its traceback"


# ---------------------------------------------------------------------------
# Single-flight dials
# ---------------------------------------------------------------------------


def _dial_server() -> Any:
    server: Any = object.__new__(relay.RelayServer)
    server._dial_flights = {}
    server._dial_flights_lock = threading.Lock()
    return server


def test_concurrent_ensure_link_calls_make_exactly_one_dial() -> None:
    server = _dial_server()
    table: dict[str, Any] = {}
    link = SimpleNamespace(link_id="l_one")
    dials: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def link_for(device_id: str, *, network_id: str | None = None) -> Any:
        return table.get(device_id)

    def dial(device_id: str, **_kw: Any) -> Any:
        dials.append(device_id)
        started.set()
        assert release.wait(5)
        table[device_id] = link  # admitted: callers arriving from now see it
        return link, "", []

    server._link_for = link_for
    server._dial_link_with_probe = dial
    results: list[Any] = []

    def caller() -> None:
        results.append(server._ensure_link_with_probe("d_peer", network_id="n_x"))

    threads = [threading.Thread(target=caller) for _ in range(8)]
    threads[0].start()
    assert started.wait(5)
    for thread in threads[1:]:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()

    assert dials == ["d_peer"]
    assert len(results) == 8 and all(r[0] is link and r[1] == "" for r in results)
    assert server._dial_flights == {}


def test_a_waiter_answers_within_its_own_deadline_without_dialling() -> None:
    server = _dial_server()
    dials: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def dial(device_id: str, **_kw: Any) -> Any:
        dials.append(device_id)
        started.set()
        assert release.wait(5)
        return None, "refused", []

    server._link_for = lambda device_id, **_kw: None
    server._dial_link_with_probe = dial
    leader = threading.Thread(target=lambda: server._ensure_link_with_probe("d_peer"))
    leader.start()
    try:
        assert started.wait(5)
        link, reason, attempts = server._ensure_link_with_probe("d_peer", probe_timeout_s=0.05)
        assert (link, reason, attempts) == (None, relay.NOT_ATTEMPTED_REASON, [])
        assert dials == ["d_peer"]
    finally:
        release.set()
        leader.join(5)


def test_an_unbounded_waiter_does_not_inherit_a_listing_probes_capped_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _dial_server()
    waiting = threading.Event()
    real_flight = relay._DialFlight

    class _ObservedDone(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            waiting.set()  # a caller has joined the flight rather than dialling
            return super().wait(timeout)

    monkeypatch.setattr(
        relay, "_DialFlight", lambda bounded: real_flight(bounded=bounded, done=_ObservedDone())
    )
    calls: list[float | None] = []
    started = threading.Event()
    release = threading.Event()
    link = SimpleNamespace(link_id="l_full_budget")

    def dial(device_id: str, *, probe_timeout_s: float | None = None, **_kw: Any) -> Any:
        calls.append(probe_timeout_s)
        if probe_timeout_s is not None:  # the listing probe: capped connects
            started.set()
            assert release.wait(5)
            return None, "timeout", []
        return link, "", []

    server._link_for = lambda device_id, **_kw: None
    server._dial_link_with_probe = dial
    probe = threading.Thread(
        target=lambda: server._ensure_link_with_probe("d_peer", probe_timeout_s=5.0)
    )
    probe.start()
    assert started.wait(5)
    got: list[Any] = []
    op = threading.Thread(target=lambda: got.append(server._ensure_link_with_probe("d_peer")))
    op.start()
    assert waiting.wait(5), "the session op must join the in-flight probe first"
    release.set()
    probe.join(5)
    op.join(5)

    assert got and got[0][0] is link
    assert len(calls) == 2 and calls[1] is None


# ---------------------------------------------------------------------------
# The dialler's duplicate-link fence
# ---------------------------------------------------------------------------


class _Link:
    def __init__(self, link_id: str, device_id: str = "d_peer", network_id: str = "n_x") -> None:
        self.link_id = link_id
        self.device_id = device_id
        self.network_id = network_id
        self.alive = True
        self.closing = False
        self.link_up_hooks = False
        self.sent: list[dict[str, Any]] = []
        self.closed: list[str] = []
        self.started = False

    def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    def close(self, reason: str = "we-closed", **_kw: Any) -> None:
        self.closed.append(reason)
        self.closing = True

    def start(self) -> None:
        self.started = True


def _admit_server() -> Any:
    server: Any = object.__new__(relay.RelayServer)
    server._links_lock = threading.RLock()
    server.links = {}
    server._stop = threading.Event()
    server.redial = redial.RedialTracker()
    server._refusal_reason = lambda network: ""
    server.note_link_activity = lambda link, *, force=False: None
    return server


def test_a_second_outbound_link_to_one_peer_leaves_only_one_live_link() -> None:
    server = _admit_server()
    first, second = _Link("l_first"), _Link("l_second")
    assert server._admit_link(first, outbound=True) == ""
    assert server._admit_link(second, outbound=True) == ""

    # Newest wins, as at the listener's fence (which has already evicted ITS copy
    # of the older link for this same handshake): the older one is told and closed.
    assert first.closed == ["replaced"]
    assert first.sent == [{"op": "net_bye", "reason": "replaced"}]
    assert second.started and second.closed == []
    live = [link for link in server.links.values() if link.alive and not link.closing]
    assert live == [second]


def test_the_outbound_fence_is_per_network_and_inbound_admission_is_not_fenced_here() -> None:
    server = _admit_server()
    outbound = _Link("l_out")
    other_network = _Link("l_other", network_id="n_y")
    assert server._admit_link(outbound, outbound=True) == ""
    assert server._admit_link(other_network, outbound=True) == ""
    # An inbound admission is fenced by ``register_link`` (identity claims), not here.
    inbound = _Link("l_inbound")
    assert server._admit_link(inbound) == ""

    assert outbound.closed == [] and other_network.closed == [] and inbound.closed == []
    assert set(server.links) == {"l_out", "l_other", "l_inbound"}
