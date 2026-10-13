"""The PRODUCTION availability lane on a real pair: deferred, bounded, responsive.

This module began as a pre-lane dispatcher PROTOTYPE: a test-local copy of the
deferred continuation, standing in for a lane that did not exist yet. The lane
is production now (``RelayServer.dispatch`` -> ``_dispatch_source_availability``
-> one coalesced flight -> ``_answer_source_availability``), and the copy would
SHADOW it — a passing run would have proven nothing about the shipped code
(agent review round 1, M1: the copy also pre-empted the module's own negative
control). This module was rewritten to drive the real thing: frames on a real
paired link, admitted by the production dispatcher, answered by the production
flight after its fresh rechecks.

The prototype's negative controls described the PRE-lane world and are retired
with it: a ``None`` from an inline handler answering an empty ACK, and an
inline handler holding the reader. What replaces them are controls against the
PRODUCTION dispatcher — the intercept is scoped to ``source_read=availability``
(a page read is still served inline), and an availability frame without the
complete declaration is refused BY NAME, never deferred.

Cap, busy, once-answer and the post-wait rechecks are pinned at the relay level
in ``tests/unit/network/test_peer_source_contract.py``; THIS module keeps the
end-to-end claims that need a real pair and a live viewer: four held reads join
ONE scan while a ping, a prompt and a live event still flow; the release
answers every waiter once; a caller that gives up changes nothing on the owner
(no cancel protocol — the observer settles at its own bound); a revoked grant
and a closed link each settle exactly once; and the deliberately stopped owner
keeps its stop marker with no runtime started.
"""

import concurrent.futures
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay, store
from local_operator.session import transcript_index as ti
from local_operator.wakes import store as wake_store
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixture registration
    devices,
)
from tests.unit.network.test_session_plane import (
    SESSION,
    _dial_to,
    _listen,
    _pair,
    _seed_journal,
    _serve,
    _stop_all,
    _StreamClient,
    _warm,
)


def _until(predicate: Any, bound: float = 8) -> bool:
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _availability_frame(req: int, session_id: str = SESSION, **extra: Any) -> dict[str, Any]:
    """One availability ask as a granted peer sends it on the wire."""
    return {
        "op": "net_session_history",
        "req": req,
        "locality": "remote",
        "session_id": session_id,
        "source_read": "availability",
        **extra,
    }


def _settle_and_reset(root: Path, session_id: str = SESSION) -> None:
    """Settle the previous cell's SHIELDED refresh, then start the next COLD.

    A held cell's refresh task is shielded (a timed-out waiter must never
    cancel the shared scan), so it can still be IN FLIGHT when the next cell
    submits. ``start_refresh`` then JOINS that running task instead of starting
    the new scan the cell holds, the wave never reaches the scanner, and the
    cell's ``entered`` assertion races — measured as a CI shard failure on
    exactly that (2026-10-13, ``test (3.12, 4)``). Waiting for residency proves
    the previous scan finished; the reset makes the next one cold, which is
    also what the availability lane needs: with a resident index there is
    nothing left to wait for.
    """
    assert _until(
        lambda: ti.fresh_resident(root, session_id) is not None, bound=15
    ), "the previous cell's refresh never settled"
    ti._reset_for_tests()


def test_real_pair_four_held_reads_prompt_live_and_bounded_cleanup(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair = request.getfixturevalue("devices")
    a, b, _, _ = pair
    _seed_journal(b.root, SESSION, ["", ""])
    wake_store.write_entry(
        b.root,
        SESSION,
        cwd="",
        schedules=[{"id": "parked-proof", "state": "scheduled"}],
        preserve={"stopped_at": 1234},
    )
    live_id = "feedface0002"
    _seed_journal(b.root, live_id, [""])
    record, _, _ = _pair(pair, monkeypatch, role="drive")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    served = _serve(monkeypatch, b.root)
    _warm(b.root, live_id)
    client = _StreamClient(a.root)
    assert client.open_stream(b.identity.device_id, live_id, events=True)["op"] == "ack"
    welcome = client.recv()
    assert welcome is not None and welcome["op"] in {"projection", "welcome"}

    # THE HOLD IS ARMED, not always-on: the controls below must run through
    # real scans (they are about the dispatcher, not the scanner), and only
    # the held cells may park the scan.
    armed, entered, release = threading.Event(), threading.Event(), threading.Event()
    refresh = ti.refresh_index
    scan_calls: list[str] = []
    scan_threads: list[str] = []

    def held_scan(root: Path | str, session_id: str) -> Any:
        if armed.is_set() and Path(root) == b.root and session_id == SESSION:
            scan_calls.append(session_id)
            scan_threads.append(threading.current_thread().name)
            entered.set()
            assert release.wait(12), "proof scanner hold exceeded its own bound"
        return refresh(root, session_id)

    monkeypatch.setattr(ti, "refresh_index", held_scan)
    # The held cells need room to hold: the shipped bound (4 s) is a BOUND, not
    # a promise, and this module asserts the release path answers READY — so
    # the wait is widened here rather than raced. The timeout cell below sets
    # it back down to prove the bounded-unavailable answer.
    monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_WAIT_S", 30.0)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    private: list[dict[str, Any]] = []
    matrix: dict[str, Any] = {}
    owner_link = next(iter(b.links.values()))
    try:
        # -- Discriminating controls against the PRODUCTION dispatcher --------
        # (a) The intercept is SCOPED: a page read is still served inline by
        # the handler, not parked on the lane.
        page = link.request(
            {
                "op": "net_session_history",
                "req": 90,
                "locality": "remote",
                "session_id": SESSION,
                "limit": 2,
            },
            timeout=5,
        )
        assert page is not None and page["op"] == "ack", page
        assert isinstance(page.get("detail"), dict) and "entries" in page["detail"], page
        private.append(page)
        matrix["page_read_still_inline"] = True

        # (b) An availability frame WITHOUT the complete declaration is refused
        # BY NAME at admission — the gate, not a wait.
        undeclared = link.request(_availability_frame(91), timeout=3)
        assert undeclared is not None and undeclared["op"] == "error", undeclared
        assert "requires peer_journal_source" in str(undeclared.get("message")), undeclared
        private.append(undeclared)
        matrix["undeclared_availability_refused"] = undeclared["message"]

        # (c) A session this device does not hold is refused, never deferred.
        unknown = link.request(
            _availability_frame(92, session_id="fffffffffff0", peer_journal_source=True),
            timeout=3,
        )
        assert unknown is not None and unknown["op"] == "error", unknown
        private.append(unknown)
        matrix["unknown_session_refused"] = unknown.get("message")

        # -- Four held reads on the production lane ---------------------------
        # Cold first: the controls above ran real scans, and a fresh task is
        # what the hold needs to see (see ``_settle_and_reset``).
        ti._reset_for_tests()
        armed.set()
        futures = [
            pool.submit(
                link.request, _availability_frame(200 + i, peer_journal_source=True), timeout=12
            )
            for i in range(4)
        ]
        assert _until(lambda: entered.is_set(), bound=10), "the lane never started the scan"
        assert _until(lambda: b._source_wait_total == 4, bound=6), b._source_wait_total
        assert scan_calls == [SESSION], "four observers must join one existing refresh"
        assert b._slow_pool is None, "availability consumed shared slow workers"
        matrix["joined_scan_calls"] = len(scan_calls)
        matrix["source_workers_for_four_observers"] = len(set(scan_threads))
        matrix["shared_slow_pool_created"] = False

        # Responsiveness WHILE HELD: a ping, a prompt and a live event all flow.
        ping_at = time.monotonic()
        during = link.request({"op": "ping", "req": 210, "locality": "remote"}, timeout=2)
        responsive_ping_seconds = time.monotonic() - ping_at
        assert during is not None and during["op"] == "ack", during
        private.append(during)
        prompt_at = time.monotonic()
        prompt_request = {
            "op": "prompt",
            "req": 3,
            "command_id": str(uuid.uuid4()),
            "text": "fixture prompt",
        }
        client.send(prompt_request)
        assert _until(lambda: len(served[live_id].handle.prompts()) == 1, bound=6)
        prompt_seconds = time.monotonic() - prompt_at
        live_at = time.monotonic()
        emit_live = served[live_id].handle._event_handler
        assert emit_live is not None
        emit_live({"type": "text_delta", "text": "proof live event"})
        frames = []
        for _ in range(12):
            frame = client.recv(timeout=2)
            assert frame is not None
            frames.append(frame)
            if frame.get("op") == "event" and (frame.get("data") or {}).get("type") == "text_delta":
                break
        else:
            raise AssertionError("live event not delivered while availability held")
        live_seconds = time.monotonic() - live_at
        private.extend(frames)
        assert not release.is_set()
        assert b._source_wait_total == 4

        # -- Release: every waiter answered ONCE, ready, then drained ---------
        release.set()
        raw_replies = [f.result(12) for f in futures]
        assert all(
            r is not None
            and r["op"] == "ack"
            and r["detail"]["source_available"] is True
            and r["detail"]["journal_order_state"] == "ready"
            and r["detail"]["peer_journal_source"] is True
            for r in raw_replies
        ), raw_replies
        replies = [r for r in raw_replies if r is not None]
        private.extend(replies)
        assert _until(lambda: b._source_wait_total == 0, bound=8), b._source_wait_total
        matrix["ready_responses"] = len(replies)
        _settle_and_reset(b.root)

        # -- A caller that gives up changes nothing on the owner --------------
        # No cancellation protocol exists by design: the observer settles at
        # its own (here: shortened) bound, and the lane claims no instant
        # cleanup. The caller's own timeout is the client-side bound.
        monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_WAIT_S", 0.3)
        release.clear()
        entered.clear()
        timed_at = time.monotonic()
        timed = link.request(_availability_frame(300, peer_journal_source=True), timeout=0.05)
        assert timed is None
        assert _until(lambda: entered.is_set(), bound=10)
        assert _until(lambda: b._source_wait_total == 0, bound=8), b._source_wait_total
        matrix["timeout_observer_seconds"] = time.monotonic() - timed_at
        matrix["caller_timeout_cleanup"] = (
            "owner observer persists until its own bound; "
            "no cancel protocol, no instant-cleanup claim"
        )
        release.set()
        _settle_and_reset(b.root)

        # -- A revoked grant settles once, at the answer ----------------------
        monkeypatch.setattr(relay, "SOURCE_AVAILABILITY_WAIT_S", 30.0)
        release.clear()
        entered.clear()
        revoked = pool.submit(
            link.request, _availability_frame(400, peer_journal_source=True), timeout=12
        )
        assert _until(lambda: entered.is_set(), bound=10)
        assert _until(lambda: b._source_wait_total == 1, bound=6)
        local = store.load(record.network_id, b.root)
        assert local is not None
        next(m for m in local.members if m.device_id == a.identity.device_id).capabilities = []
        store.save(local, b.root)
        release.set()
        result = revoked.result(12)
        assert result is not None and result["op"] == "error", result
        private.append(result)
        assert _until(lambda: b._source_wait_total == 0, bound=8)
        matrix["revoke_after_admission"] = "one refusal; no source facts; observer removed"
        next(m for m in local.members if m.device_id == a.identity.device_id).capabilities = [
            "view"
        ]
        store.save(local, b.root)
        _settle_and_reset(b.root)

        # -- A closed link receives no late answer ----------------------------
        release.clear()
        entered.clear()
        closed = pool.submit(
            link.request, _availability_frame(500, peer_journal_source=True), timeout=12
        )
        assert _until(lambda: entered.is_set(), bound=10)
        assert _until(lambda: b._source_wait_total == 1, bound=6)
        link_close_at = time.monotonic()
        link.close()
        assert closed.result(6) is None
        assert _until(lambda: not owner_link.alive, bound=8), "the owner side never saw the close"
        release.set()
        assert _until(lambda: b._source_wait_total == 0, bound=8), b._source_wait_total
        matrix["link_close_observer_seconds"] = time.monotonic() - link_close_at
        matrix["link_close_cleanup"] = (
            "requester stops; owner observer removed at the release; zero late answers "
            "(the answer path returns early on a closed link)"
        )

        # -- The deliberately stopped owner is untouched ----------------------
        states = [row["state"] for row in b.local_session_rows() if row["session_id"] == SESSION]
        assert states == ["stored"]
        stop_entry = wake_store.read_entry(b.root, SESSION)
        assert stop_entry is not None and stop_entry["stopped_at"] == 1234
        matrix["deliberate_stop_marker_preserved"] = True
        matrix["stored_owner_runtime"] = "no runtime/lease started for availability target"
        matrix["latency_seconds"] = {
            "deferred_ping": responsive_ping_seconds,
            "fixture_prompt_delivery": prompt_seconds,
            "fixture_live_event_delivery": live_seconds,
        }
        matrix["producer_scope"] = (
            "production relay dispatcher/availability lane over a real paired link; "
            "live producer is FakeHandle._event_handler injection, "
            "NOT Session production admission or native E2E"
        )
        matrix["bounds_seconds"] = {
            "held_cells_availability": 30.0,
            "timeout_cell_availability": 0.3,
            "shared_refresh_test_hold": 12,
            "outer_process_group": 90,
        }
    finally:
        release.set()
        armed.set()
        pool.shutdown(wait=True)
        client.close()
        link.close()
        _stop_all(served)
    output = os.environ.get("PEER_PROOF_OUTPUT")
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        body = out / "paired-responses.private.json"
        body.write_text(json.dumps(private))
        body.chmod(0o600)
        (out / "matrix-B.json").write_text(json.dumps(matrix, indent=2))
