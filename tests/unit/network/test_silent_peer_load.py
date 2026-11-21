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
    monkeypatch.setattr(relay, "time", SimpleNamespace(time=clock.time, monotonic=clock.time))

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
    # This cell tests a seal failure, not an already-expired idle deadline.
    link.last_frame_at = relay.time.time()
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


def _writer_after_clock_gap(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wall_gap: float,
    monotonic_gap: float,
    queued_after_wake: bool = False,
    written: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run the actual writer across one synthetic sleep, never sleeping the host."""
    wall, monotonic = _FakeClock(1_000_000.0), _FakeClock(100.0)
    monkeypatch.setattr(relay, "time", SimpleNamespace(time=wall.time, monotonic=monotonic.time))
    rows: list[dict[str, Any]] = []
    server: Any = SimpleNamespace(
        audit=SimpleNamespace(
            record=lambda event: rows.append({"event": event.event, "detail": event.detail})
        ),
        identity=SimpleNamespace(device_id="d_local"),
        _links_lock=threading.RLock(),
    )
    server.audit_idle = lambda link: relay.RelayServer.audit_idle(server, link)
    server.audit_host_resume = lambda **kwargs: relay.RelayServer.audit_host_resume(
        server, **kwargs
    )
    link = _writer_link(server)
    link.network_id = "n_sleep"
    link.epoch = 1
    server.settings = link.settings
    link.last_frame_at = wall.now
    link.last_ping_sent_at = wall.now
    link._write = lambda frame: written.append(frame) if written is not None else None
    closed: list[str] = []

    def close(reason: str, **_kwargs: Any) -> None:
        closed.append(reason)
        link._closed.set()

    class Wake:
        waits = 0

        def wait(self, timeout: float | None = None) -> bool:
            self.waits += 1
            if self.waits == 1:
                wall.now += wall_gap
                monotonic.now += monotonic_gap
                if queued_after_wake:
                    link._reliable.put({"op": "net_ping", "req": 1})
            else:
                # Finite negative control: the unfixed writer must return too.
                link._closed.set()
            return False

        def clear(self) -> None:
            pass

    link.close = close
    link._wake = Wake()
    link._write_loop()
    if closed:
        # A reader timeout can race the writer's wall-idle check. It must not
        # turn one dead link into two idle incidents.
        server.audit_idle(link)
    return rows, closed


def test_a_wall_idle_deadline_closes_on_the_first_tick_after_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, closed = _writer_after_clock_gap(monkeypatch, wall_gap=121.0, monotonic_gap=0.0)
    assert closed == ["timeout"], "a sleep-stale link survived its first wake tick"
    assert len([row for row in rows if row["event"] == "link_idle"]) == 1


def test_the_first_wake_tick_does_not_drain_work_into_a_sleep_stale_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[dict[str, Any]] = []
    _, closed = _writer_after_clock_gap(
        monkeypatch,
        wall_gap=121.0,
        monotonic_gap=0.0,
        queued_after_wake=True,
        written=written,
    )
    assert closed == ["timeout"]
    assert written == [], "queued work was sent before the wall-stale link closed"


def test_a_host_resume_is_audited_once_without_exposing_clock_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, closed = _writer_after_clock_gap(monkeypatch, wall_gap=31.0, monotonic_gap=0.0)
    resumed = [row for row in rows if row["event"] == "host_resumed"]
    assert len(resumed) == 1, "the writer did not explain the host's wall-clock gap"
    assert resumed[0]["detail"] == {"gap_s": 31.0}
    assert closed == [], "resume alone must not close a link inside its idle deadline"


def test_equal_wall_and_monotonic_progress_neither_resumes_nor_closes_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, closed = _writer_after_clock_gap(monkeypatch, wall_gap=31.0, monotonic_gap=31.0)
    assert rows == []
    assert closed == []


def test_all_waking_links_share_one_host_resume_row_but_a_later_gap_is_new() -> None:
    rows: list[Any] = []
    server: Any = SimpleNamespace(
        _links_lock=threading.RLock(),
        settings=relay.NetworkSettings(port=4097),
        identity=SimpleNamespace(device_id="d_local"),
        audit=SimpleNamespace(record=rows.append),
    )
    # Two writer ticks observe the same host discontinuity, with normal sampling
    # skew. A later discontinuity is not suppressed by a fixed audit time window.
    relay.RelayServer.audit_host_resume(server, gap_s=31.0, clock_offset=1_000.0)
    relay.RelayServer.audit_host_resume(server, gap_s=31.0, clock_offset=1_000.001)
    assert len(rows) == 1
    relay.RelayServer.audit_host_resume(server, gap_s=31.0, clock_offset=1_031.0)
    assert len(rows) == 2
    assert all(row.event == "host_resumed" and row.detail == {"gap_s": 31.0} for row in rows)


# ---------------------------------------------------------------------------
# Single-flight dials: one silent peer never holds up another
# ---------------------------------------------------------------------------
#
# The single-flight contract itself (N callers, one dial; waiter deadlines; failed
# and raising leaders) is pinned by ``test_ensure_link_single_flight.py`` (#3885).
# This file pins the silent-peer half: a dial stuck on a peer that has gone quiet
# must never delay a dial to any OTHER peer.


def test_a_dial_stuck_on_a_silent_peer_never_blocks_a_dial_to_another_peer() -> None:
    server: Any = object.__new__(relay.RelayServer)
    silent_started = threading.Event()
    release_silent = threading.Event()
    healthy = SimpleNamespace(link_id="l_healthy")

    def dial(device_id: str, **_kw: Any) -> Any:
        if device_id == "d_silent":
            silent_started.set()
            assert release_silent.wait(10)  # the handshake that never answers
            return None, "timeout", []
        return healthy, "", []

    server._link_for = lambda device_id, **_kw: None
    server._dial_for_link = dial
    stuck = threading.Thread(target=lambda: server._ensure_link_with_probe("d_silent"))
    stuck.start()
    try:
        assert silent_started.wait(5)
        got: list[Any] = []
        other = threading.Thread(
            target=lambda: got.append(server._ensure_link_with_probe("d_healthy"))
        )
        other.start()
        other.join(2)
        # Answered while the silent peer's dial is STILL in flight.
        assert not other.is_alive(), "a dial to a healthy peer waited on a silent one"
        assert got == [(healthy, "", [])]
        assert stuck.is_alive()
    finally:
        release_silent.set()
        stuck.join(5)


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
        self.outbound = False
        # A stand-in for an AUTHENTICATED link: ``_admit_link`` refuses any link
        # whose peer was not verified (mesh-listener-auth), and this rig's subject
        # is the duplicate fence, not authentication.
        self.peer_verified = True

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


def test_a_simultaneous_cross_dial_keeps_the_link_the_peer_dialled() -> None:
    """A dials B while B dials A: the outbound fence must not replace an INBOUND link.

    The older link here is the one the PEER dialled. Its far end (the peer, as
    dialler) is not about to evict it — the peer's own outbound fence is busy
    replacing ITS copy of our link — so closing it here too would leave the pair
    with no link at all. Inbound duplicates stay ``register_link``'s business.
    """
    server = _admit_server()
    inbound = _Link("l_peer_dialled")
    assert server._admit_link(inbound) == ""
    ours = _Link("l_we_dialled")
    assert server._admit_link(ours, outbound=True) == ""

    assert inbound.closed == [] and inbound.sent == []
    assert ours.outbound and not inbound.outbound
    assert set(server.links) == {"l_peer_dialled", "l_we_dialled"}
