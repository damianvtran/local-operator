"""Relay resilience: redial schedule, last-seen truth, refusal coalescing, link restore.

WHY THIS FILE EXISTS (2026-10-21, cloud-node-1). The node's link dropped and the
desktop sidebar said ``Link down · unknown`` at once: ``last_seen_at`` was written
only on rotation frames, nothing redialled a lost link on its own schedule, the
audit grew a row per repeated refusal (5.3 MB), and a late stream push after a
viewer left was refused 26 times in 50 ms. Each rule is pinned here; the last cells
drive two REAL relays on loopback through a drop and a restore.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import redial, relay, store, wire
from tests.unit.network.test_readiness_link import (
    _bind_and_start,
    _set_peer_endpoints,
    _settle_the_peer_link,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


class _Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# -- the schedule ------------------------------------------------------------


def test_backoff_is_exponential_capped_and_jittered() -> None:
    lows = [redial.backoff_s(n, rand=lambda: 0.0) for n in range(12)]
    highs = [redial.backoff_s(n, rand=lambda: 1.0) for n in range(12)]
    mids = [redial.backoff_s(n, rand=lambda: 0.5) for n in range(12)]
    assert mids[:4] == [1.0, 2.0, 4.0, 8.0]
    assert max(mids) == wire.RECONNECT_MAX_S, "capped at the declared ceiling"
    assert all(low == pytest.approx(mid * 0.75) for low, mid in zip(lows, mids))
    assert all(high == pytest.approx(mid * 1.25) for high, mid in zip(highs, mids))
    # The spread survives at the cap: 60 s ± 25 %, never lock-step at exactly 60.
    assert lows[-1] == pytest.approx(45.0) and highs[-1] == pytest.approx(75.0)


def test_a_lost_link_reads_reconnecting_until_grace_and_attempts_both_pass() -> None:
    clock = _Clock()
    tracker = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    tracker.lost("d_peer", cause="timeout", last_frame_at=990.0)
    snap = tracker.snapshot("d_peer")
    assert snap["state"] == redial.RECONNECTING
    assert snap["last_frame_at"] == 990.0 and snap["last_error"] == "timeout"
    assert snap["next_retry_at"] == pytest.approx(1_001.0)
    # Many fast failures inside the grace window do NOT escalate (a relay restart
    # refuses for a second or two).
    for _ in range(5):
        clock.now += 2
        tracker.failed("d_peer", cause="connect_failed:ConnectionRefusedError")
    assert tracker.snapshot("d_peer")["state"] == redial.RECONNECTING
    # Time alone does not escalate either...
    quiet = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    quiet.lost("d_quiet", cause="timeout")
    clock.now += redial.GRACE_S + 1
    assert quiet.snapshot("d_quiet")["state"] == redial.RECONNECTING
    # ...but both together do.
    assert tracker.snapshot("d_peer")["state"] == redial.DOWN
    tracker.connected("d_peer")
    assert tracker.snapshot("d_peer") == {"state": redial.CONNECTED, "attempts": 0}


def test_due_respects_the_backoff_and_a_seed_is_due_at_once() -> None:
    clock = _Clock()
    tracker = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    tracker.seed("d_seed")
    tracker.lost("d_lost", cause="timeout")
    assert tracker.due() == ["d_seed"]
    assert tracker.snapshot("d_seed")["state"] == redial.CONNECTED, "no loss to describe"
    clock.now += 1.0
    assert set(tracker.due()) == {"d_seed", "d_lost"}
    tracker.failed("d_lost", cause="timeout")
    assert "d_lost" not in tracker.due(), "a failure pushes the next retry out"
    # A failed seed becomes an ordinary loss on the capped schedule.
    tracker.failed("d_seed", cause="timeout")
    assert tracker.snapshot("d_seed")["state"] == redial.RECONNECTING
    clock.now += 1_000
    assert set(tracker.due()) == {"d_seed", "d_lost"}


def test_a_second_close_does_not_reset_the_grace_clock() -> None:
    clock = _Clock()
    tracker = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    tracker.lost("d_peer", cause="timeout", last_frame_at=990.0)
    clock.now += 50
    tracker.lost("d_peer", cause="error", last_frame_at=995.0)
    snap = tracker.snapshot("d_peer")
    assert snap["lost_at"] == 1_000.0 and snap["last_frame_at"] == 995.0


# -- audit coalescing ----------------------------------------------------------


def test_identical_refusals_collapse_into_one_row_per_window_with_a_count() -> None:
    clock = _Clock()
    rows = redial.RefusalCoalescer(window_s=300.0, clock=clock)
    key = ("peer_closed_silently", "n_1", "d_old", "99.79.190.164")
    assert rows.admit(key) == (True, 0)
    for n in range(1, 50):
        clock.now += 3
        assert rows.admit(key) == (False, n)
    clock.now += 300
    assert rows.admit(key) == (True, 49), "the next row carries the folded count"
    assert rows.admit(("unknown_network", "n_2", "", "192.168.0.155")) == (True, 0)


def test_the_audit_whitelist_keeps_the_repeats_field() -> None:
    assert "repeats" in audit_mod.DETAIL_KEYS["handshake_refused"]


# -- the late-push storm -------------------------------------------------------


def test_a_late_push_for_an_ended_stream_is_dropped_without_a_refusal() -> None:
    """26 ``authorisation_refused`` rows in 50 ms came from exactly this path."""
    dispatched: list[dict[str, Any]] = []
    sent: list[dict[str, Any]] = []
    server = SimpleNamespace(
        deliver_reply=lambda link, frame: False,
        route_stream_push=lambda link, frame: False,
        route_stream_closed=lambda link, frame: False,
        dispatch=lambda link, frame: dispatched.append(frame) or {"op": "error"},
    )
    # A duck-typed link: only the attributes ``_handle`` reads on this path.
    link: Any = cast(Any, SimpleNamespace(server=server, late_stream_frames=0, send=sent.append))
    for _ in range(26):
        relay.PeerLink._handle(link, {"op": "net_stream", "action": "push", "stream": "s1"})
    relay.PeerLink._handle(link, {"op": "net_stream", "action": "closed", "stream": "s1"})
    assert dispatched == [] and sent == []
    assert link.late_stream_frames == 27
    # A REQUEST (open/send/close) still reaches the authorising dispatch.
    relay.PeerLink._handle(link, {"op": "net_stream", "action": "close", "stream": "s1"})
    assert len(dispatched) == 1


# -- endpoints -----------------------------------------------------------------


def test_an_observed_endpoint_is_a_last_resort_dial_candidate() -> None:
    member: Any = cast(
        Any,
        SimpleNamespace(
            endpoints=["99.79.190.164:4097", "172.31.22.23:4097"],
            observed_endpoint="3.96.223.236:4097",
        ),
    )
    assert relay._dial_candidates(member) == [
        "99.79.190.164:4097",
        "172.31.22.23:4097",
        "3.96.223.236:4097",
    ]
    member.observed_endpoint = "172.31.22.23:4097"
    assert relay._dial_candidates(member) == ["99.79.190.164:4097", "172.31.22.23:4097"]


def test_newest_seen_picks_the_freshest_evidence() -> None:
    assert relay._newest_seen(None, None, None) is None
    assert relay._newest_seen(None, 10.0, 12.5) == 12.5
    assert relay._newest_seen(20.0, 10.0, None) == 20.0
    assert relay._newest_seen(True, 0, -1) is None


# -- two real relays -----------------------------------------------------------


def _wait(predicate: Any, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


def _row(server: relay.RelayServer, device_id: str) -> dict[str, Any]:
    rows = [row for row in server.peer_status() if row["device_id"] == device_id]
    assert rows, "the peer was not in the table"
    return rows[0]


def test_link_activity_persists_last_seen_throttled(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair
    record, _h, _p = _pair(pair, monkeypatch)
    device_b = server_b.identity.device_id

    def seen() -> float | None:
        member = store.load(record.network_id, server_a.root).member(device_b)
        assert member is not None
        return member.last_seen_at

    link: Any = SimpleNamespace(
        network_id=record.network_id,
        device_id=device_b,
        last_frame_at=time.time() + 5_000.0,
        seen_persisted_at=0.0,
    )
    first = link.last_frame_at
    server_a.note_link_activity(link)
    assert seen() == round(first, 3)
    link.last_frame_at = first + 10.0
    server_a.note_link_activity(link)  # inside the 30 s throttle: not written
    assert seen() == round(first, 3)
    server_a.note_link_activity(link, force=True)  # the close path always writes
    assert seen() == round(first + 10.0, 3)


def test_a_dropped_link_reports_reconnecting_with_a_real_last_seen_then_recovers(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drop → reconnecting (real last_seen, not null) → peer back → link restored.

    The incident's shape on loopback: a live link, the peer's relay stops, the
    listing must say ``reconnecting`` with the last frame's time, and when the
    peer's relay serves again on the same address the REDIAL LOOP restores the link
    with no surface touched.
    """
    pair: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair
    record, _h, _p = _pair(pair, monkeypatch)
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])
    device_b = server_b.identity.device_id
    assert _row(server_a, device_b)["reachable"] is True
    persisted = store.load(record.network_id, server_a.root).member(device_b)
    assert persisted is not None and persisted.last_seen_at is not None

    port_b = int(live.rsplit(":", 1)[1])
    server_b.stop()
    assert _wait(lambda: server_a._link_for(device_b) is None)
    row = _row(server_a, device_b)
    assert row["reachable"] is False
    assert row["redial"]["state"] == redial.RECONNECTING, row["redial"]
    assert isinstance(row["last_seen_at"], float), "never 'unknown' for a peer seen this run"
    assert time.time() - row["last_seen_at"] < 60
    assert row["redial"]["attempts"] >= 1, "the listing's failed probe counts"

    revived = relay.RelayServer(
        root=server_b.root,
        identity=server_b.identity,
        settings=relay.replace(server_b.settings, port=port_b),
        audit=audit_mod.AuditLog(server_b.root),
    )
    # ONLY A's REDIAL LOOP MAY RESTORE THE LINK. B's outbound dials are disabled,
    # and so is every dial on A from any thread but ``mesh-redial`` — the
    # credentials/definitions tick (``definitions.py``, ~15 s) also dials peers, and
    # without this it restored the link on its own, so the cell passed with the loop
    # removed (measured). With the filter, removing ``_redial_loop`` from
    # ``start()`` makes this cell time out.
    monkeypatch.setattr(revived, "dial", lambda *a, **k: (None, "disabled_in_test"))
    real_dial = server_a.dial

    def redial_thread_only(*args: Any, **kwargs: Any) -> Any:
        if threading.current_thread().name != "mesh-redial":
            return None, "disabled_in_test"
        return real_dial(*args, **kwargs)

    monkeypatch.setattr(server_a, "dial", redial_thread_only)
    revived.bind()
    revived.bind_control()
    revived.start()
    try:
        assert _wait(lambda: server_a._link_for(device_b) is not None, timeout_s=30.0)
        # B ACCEPTED it: the listener's ``link_opened`` row names A as the actor.
        revived.audit.flush()
        opened = [
            row
            for row in revived.audit.tail(50, network_id=record.network_id)
            if row.get("event") == "link_opened"
        ]
        assert opened and opened[-1]["actor"] == server_a.identity.device_id, opened
        assert opened[-1]["detail"].get("role") == "listener", opened
        row = _row(server_a, device_b)
        assert row["reachable"] is True
        assert row["redial"] == {"state": redial.CONNECTED, "attempts": 0}
    finally:
        revived.stop()


def test_backoff_remains_finite_after_massive_attempt_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for attempt in (1024, 1100, 10**100):
        assert redial.backoff_s(attempt, rand=lambda: 0.5) == wire.RECONNECT_MAX_S
    clock = _Clock()
    tracker = redial.RedialTracker(clock=clock, rand=lambda: 0.5)
    tracker.lost("d_long", cause="timeout")
    # Continue well beyond the formerly fatal 1024th failed retry.
    for _ in range(1100):
        clock.now += wire.RECONNECT_MAX_S
        tracker.failed("d_long", cause="timeout")
    assert tracker.snapshot("d_long")["attempts"] == 1100
    clock.now += wire.RECONNECT_MAX_S
    assert tracker.due() == ["d_long"]
    # The real loop's failed() call used to escape with OverflowError, killing
    # its thread. Drive two ticks after saturation and prove both are served.
    server: Any = object.__new__(relay.RelayServer)
    server.root = None
    server.redial = tracker
    ticks: list[int] = []

    def tick(_delay: float) -> bool:
        ticks.append(1)
        clock.now += wire.RECONNECT_MAX_S
        return len(ticks) > 2

    server._stop = SimpleNamespace(wait=tick, is_set=lambda: False)
    server._link_for = lambda device: None
    server._members_named = lambda device: []
    monkeypatch.setattr(store, "list_networks", lambda root: [])
    server._redial_loop()
    assert tracker.snapshot("d_long")["attempts"] == 1102
    assert len(ticks) == 3


def test_reader_close_during_admission_keeps_its_newer_loss_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual admission/close methods, with a reader released by start().

    Pause admission's persistence after its ownership section so the reader can
    close before admission returns. The old connected reset was outside that
    section and ran AFTER close, erasing its loss.
    """
    server: Any = object.__new__(relay.RelayServer)
    server._links_lock = threading.RLock()
    server._streams_lock = threading.Lock()
    server.links = {}
    server._streams = {}
    server._stop = threading.Event()
    server.redial = redial.RedialTracker()
    server._refusal_reason = lambda network: ""
    server.fail_replies = lambda link: 0
    server.identity_use = SimpleNamespace(released=lambda *args: None)
    server.audit = SimpleNamespace(record=lambda event: None)
    closed = threading.Event()
    failures: list[BaseException] = []
    reader: list[threading.Thread] = []
    connected = server.redial.connected

    def connected_after_close_when_unlocked(device: str) -> None:
        # Reproduce the old unlocked gap deterministically: let the actual reader
        # close finish before the unlocked reset. Under the fix this transition
        # owns the RLock, so the reader cannot interleave and no barrier is needed.
        if not server._links_lock._is_owned():
            assert closed.wait(5), "reader did not reach the unlocked admission gap"
        connected(device)

    monkeypatch.setattr(server.redial, "connected", connected_after_close_when_unlocked)

    def persist(link: Any, *, force: bool = False) -> None:
        if threading.current_thread().name != "synthetic-reader":
            assert closed.wait(5), "the reader did not close"

    server.note_link_activity = persist
    link: Any = SimpleNamespace(
        link_id="l_synthetic",
        device_id="d_synthetic",
        network_id="n_synthetic",
        last_frame_at=time.time(),
        alive=False,
        epoch=1,
        frames_in=0,
        frames_out=0,
    )

    def close_reader() -> None:
        try:
            server.link_closed(link, "timeout")
        except BaseException as exc:
            failures.append(exc)
        finally:
            closed.set()

    def start() -> None:
        thread = threading.Thread(target=close_reader, name="synthetic-reader")
        reader.append(thread)
        thread.start()

    link.start = start
    assert server._admit_link(link) == ""
    reader[0].join(5)
    assert not reader[0].is_alive() and not failures, failures
    assert not server.links
    assert server.redial.snapshot(link.device_id)["state"] == redial.RECONNECTING
    assert server.redial.tracked() == [link.device_id]


def test_refusal_coalescing_keeps_proof_and_mode_transitions(tmp_path: Any) -> None:
    server: Any = object.__new__(relay.RelayServer)
    server.audit = audit_mod.AuditLog(tmp_path)
    server._refusal_rows = redial.RefusalCoalescer()
    refusal = relay.MeshRefusal("malformed_frame", "invalid frame")
    for verified, mode in ((False, "member"), (True, "member"), (True, "invite")):
        # Identical repeats fold; a materially different proof/mode does not.
        for _ in range(2):
            server._audit_handshake_refusal(
                refusal,
                "n_synthetic",
                "192.0.2.1:1234",
                mode,
                declared_device="d_synthetic",
                verified=verified,
            )
    server.audit.flush()
    rows = [r for r in server.audit.tail(20) if r["event"] == "handshake_refused"]
    assert len(rows) == 3
    assert [r["actor"] for r in rows] == ["unknown", "d_synthetic", "d_synthetic"]
    assert [r["detail"]["mode"] for r in rows] == ["member", "member", "invite"]


def test_worker_live_lookup_cannot_erase_a_newer_link_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual worker and close methods, paused at a stale live-link lookup.

    A due snapshot can outlive admission of a replacement. When that replacement
    closes after the worker's lookup, only admission/close own the transitions;
    the worker must leave the newly recorded loss scheduled for the next tick.
    """
    server: Any = object.__new__(relay.RelayServer)
    server.root = None
    server._links_lock = threading.RLock()
    server._streams_lock = threading.Lock()
    server._streams = {}
    server.redial = redial.RedialTracker()
    server.redial.seed("d_synthetic")
    link: Any = SimpleNamespace(
        link_id="l_synthetic",
        device_id="d_synthetic",
        network_id="n_synthetic",
        alive=True,
        closing=False,
        last_frame_at=time.time(),
        epoch=1,
        frames_in=0,
        frames_out=0,
    )
    server.links = {link.link_id: link}
    server.note_link_activity = lambda *args, **kwargs: None
    server.fail_replies = lambda link: 0
    server.identity_use = SimpleNamespace(released=lambda *args: None)
    server.audit = SimpleNamespace(record=lambda event: None)
    ticks = iter((False, True))
    server._stop = SimpleNamespace(wait=lambda delay: next(ticks), is_set=lambda: False)
    monkeypatch.setattr(store, "list_networks", lambda root: [])
    looked_up = threading.Event()
    closed = threading.Event()
    failures: list[BaseException] = []
    lookup = server._link_for

    def paused_lookup(device: str) -> Any:
        held = lookup(device)
        if held is not None:
            looked_up.set()
            assert closed.wait(5), "reader did not close after live lookup"
        return held

    server._link_for = paused_lookup

    def worker() -> None:
        try:
            server._redial_loop()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=worker, name="synthetic-redial-worker")
    thread.start()
    try:
        assert looked_up.wait(5), "worker never observed the replacement"
        # The real close removes ownership and records loss while the worker
        # holds a stale returned link. Close's own lookup sees an empty table.
        server.link_closed(link, "timeout")
    finally:
        closed.set()
        thread.join(5)
    assert not thread.is_alive() and not failures, failures
    assert not server.links
    assert server.redial.snapshot(link.device_id)["state"] == redial.RECONNECTING
    assert server.redial.tracked() == [link.device_id]
