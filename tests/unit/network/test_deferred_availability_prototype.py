"""Test-only dispatcher prototype; no complete capability or runtime changes."""

import asyncio
import concurrent.futures
import json
import os
import threading
import time
import uuid
from pathlib import Path

import pytest

from local_operator.network import store, wire
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


def _until(predicate, bound=5):
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _install(owner, monkeypatch, *, deadline=2):
    original = owner.dispatch
    pending, answers = {}, []
    guard = threading.Lock()

    async def wait_and_answer(link, frame):
        req = frame["req"]
        key = (link.link_id, req)
        try:
            task, _ = ti.start_refresh(owner.root, frame["session_id"])
            try:
                await asyncio.wait_for(asyncio.shield(task), deadline)
                detail = {
                    "source_available": ti.fresh_resident(owner.root, frame["session_id"])
                    is not None
                }
                reply = {"op": "ack", "req": req, "detail": detail}
            except TimeoutError:
                reply = wire.refusal_frame(req, "availability bound expired")
            # Final current-link and unchanged view authority checks are after
            # the await, not inherited from admission. No new grant is needed.
            with owner._links_lock:
                current = owner.links.get(link.link_id) is link and link.alive
            if not current:
                return
            try:
                owner.authorizer.check(link.context, frame)
                if not (owner.root / "sessions" / frame["session_id"]).is_dir():
                    raise ValueError("session no longer owned")
            except Exception:
                reply = wire.refusal_frame(req, "availability authority no longer valid")
            with guard:
                if key not in pending:
                    return
                pending.pop(key)
                answers.append((req, reply["op"]))
            link.send(reply)
        finally:
            with guard:
                pending.pop(key, None)

    def dispatch(link, frame):
        if frame.get("op") != "net_session_history" or frame.get("source_read") != "availability":
            return original(link, frame)
        try:
            owner.authorizer.check(link.context, frame)
            if not (owner.root / "sessions" / frame["session_id"]).is_dir():
                return wire.refusal_frame(frame.get("req"), "unknown session")
        except Exception:
            return wire.refusal_frame(frame.get("req"), "availability denied")
        key = (link.link_id, frame["req"])
        with guard:
            if key in pending:
                # A duplicate wire request must not create a second answer to
                # the original request ID while its first observer is pending.
                return None
            pending[key] = True
        with owner._page_loop_lock:
            loop = owner._page_loop_locked()
            asyncio.run_coroutine_threadsafe(wait_and_answer(link, dict(frame)), loop)
        # This None is at DISPATCH, not _run_handler: reader sends nothing now.
        # No result() wait and no shared slow worker is consumed.
        return None

    monkeypatch.setattr(owner, "dispatch", dispatch)
    return pending, answers


def test_real_pair_four_held_reads_prompt_live_and_bounded_cleanup(
    request: pytest.FixtureRequest, monkeypatch
):
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
    assert client.recv()["op"] in {"projection", "welcome"}
    entered, release = threading.Event(), threading.Event()
    refresh = ti.refresh_index
    scan_calls = []
    scan_threads = []

    def held_scan(root, session_id):
        if Path(root) == b.root and session_id == SESSION:
            scan_calls.append(session_id)
            scan_threads.append(threading.current_thread().name)
            entered.set()
            assert release.wait(8), "proof scanner hold exceeded its own bound"
        return refresh(root, session_id)

    monkeypatch.setattr(ti, "refresh_index", held_scan)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    private, matrix = [], {}
    original = b._handlers["net_session_history"]
    try:
        # Negative control: None from today's inline handler is an EMPTY ACK,
        # not a deferred protocol. Capture actual response on the paired link.
        monkeypatch.setitem(b._handlers, "net_session_history", lambda *args: None)
        none_reply = link.request(
            {"op": "net_session_history", "req": 100, "locality": "remote", "session_id": SESSION}
        )
        assert none_reply == {"op": "ack", "req": 100, "detail": ""}
        private.append(none_reply)
        monkeypatch.setitem(b._handlers, "net_session_history", original)

        # Negative control: today's normal handler runs INLINE and blocks the
        # reader. The artificial hold represents a source-availability wait.
        def inline_hold(owner_link, frame):
            if frame.get("source_read") == "availability":
                entered.set()
                assert release.wait(4)
                return {"source_available": False}
            return original(owner_link, frame)

        monkeypatch.setitem(b._handlers, "net_session_history", inline_hold)
        before = pool.submit(
            link.request,
            {
                "op": "net_session_history",
                "req": 101,
                "locality": "remote",
                "session_id": SESSION,
                "source_read": "availability",
            },
            timeout=5,
        )
        assert entered.wait(2)
        inline_ping_at = time.monotonic()
        ping = link.request({"op": "ping", "req": 102, "locality": "remote"}, timeout=0.15)
        inline_ping_seconds = time.monotonic() - inline_ping_at
        assert ping is None, "negative control did not actually hold reader"
        release.set()
        private.append(before.result(5))
        monkeypatch.setitem(b._handlers, "net_session_history", original)
        entered.clear()
        release.clear()
        pending, answers = _install(b, monkeypatch, deadline=6)
        futures = [
            pool.submit(
                link.request,
                {
                    "op": "net_session_history",
                    "req": 200 + i,
                    "locality": "remote",
                    "session_id": SESSION,
                    "source_read": "availability",
                },
                timeout=8,
            )
            for i in range(4)
        ]
        assert entered.wait(2)
        assert _until(lambda: len(pending) == 4)
        assert scan_calls == [SESSION], "four observers must join one existing refresh"
        assert b._slow_pool is None, "availability consumed shared slow workers"
        assert link.send(
            {
                "op": "net_session_history",
                "req": 200,
                "locality": "remote",
                "session_id": SESSION,
                "source_read": "availability",
            }
        )
        responsive_ping_at = time.monotonic()
        during = link.request({"op": "ping", "req": 210, "locality": "remote"}, timeout=1)
        responsive_ping_seconds = time.monotonic() - responsive_ping_at
        assert during is not None and during["op"] == "ack"
        private.append(during)
        prompt_at = time.monotonic()
        prompt_request = {
            "op": "prompt",
            "req": 3,
            "command_id": str(uuid.uuid4()),
            "text": "fixture prompt",
        }
        client.send(prompt_request)
        assert _until(lambda: len(served[live_id].handle.prompts()) == 1)
        prompt_seconds = time.monotonic() - prompt_at
        live_at = time.monotonic()
        served[live_id].handle._event_handler({"type": "text_delta", "text": "proof live event"})
        frames = []
        for _ in range(12):
            frame = client.recv(timeout=1)
            assert frame is not None
            frames.append(frame)
            if frame.get("op") == "event" and (frame.get("data") or {}).get("type") == "text_delta":
                break
        else:
            raise AssertionError("live event not delivered while availability held")
        live_seconds = time.monotonic() - live_at
        private.extend(frames)
        assert len(pending) == 4 and not release.is_set()
        release.set()
        replies = [f.result(4) for f in futures]
        private.extend(replies)
        assert all(r["op"] == "ack" and r["detail"]["source_available"] for r in replies)
        assert _until(lambda: not pending)
        assert len(answers) == 4
        matrix = {
            "inline_none": none_reply,
            "inline_reader_ping": "TIMEOUT while hold active",
            "held_observers": 4,
            "joined_scan_calls": len(scan_calls),
            "source_workers_for_four_observers": len(set(scan_threads)),
            "source_thread_names": list(set(scan_threads)),
            "shared_slow_pool_created": False,
            "prompt_calls_before_release": 1,
            "live_event_before_release": True,
            "ready_responses": len(replies),
            "once_response_count": len(answers),
            "remaining_observers": len(pending),
        }
        # Timeout at the CALLER has no cancellation frame. The owner observer
        # remains until its advertised bound, rather than claiming instant cleanup.
        ti._reset_for_tests()
        release.clear()
        entered.clear()
        pending2, answers2 = _install(b, monkeypatch, deadline=0.3)
        timed_at = time.monotonic()
        timed = link.request(
            {
                "op": "net_session_history",
                "req": 300,
                "locality": "remote",
                "session_id": SESSION,
                "source_read": "availability",
            },
            timeout=0.05,
        )
        assert timed is None and entered.wait(1)
        assert len(pending2) == 1
        assert _until(lambda: not pending2)
        assert answers2 == [(300, "error")]
        matrix["timeout_observer_seconds"] = time.monotonic() - timed_at
        release.set()
        matrix["caller_timeout_cleanup"] = (
            "owner observer persists until 0.3s owner bound; no cancel protocol"
        )
        assert _until(lambda: ti.fresh_resident(b.root, SESSION) is not None)
        ti._reset_for_tests()
        release.clear()
        entered.clear()
        pending3, answers3 = _install(b, monkeypatch, deadline=1)
        revoked = pool.submit(
            link.request,
            {
                "op": "net_session_history",
                "req": 400,
                "locality": "remote",
                "session_id": SESSION,
                "source_read": "availability",
            },
            timeout=2,
        )
        assert entered.wait(1) and _until(lambda: len(pending3) == 1)
        local = store.load(record.network_id, b.root)
        next(m for m in local.members if m.device_id == a.identity.device_id).capabilities = []
        store.save(local, b.root)
        release.set()
        result = revoked.result(2)
        assert result["op"] == "error"
        assert _until(lambda: not pending3) and answers3 == [(400, "error")]
        private.append(result)
        matrix["revoke_after_admission"] = "one refusal; no source facts; observer removed"
        assert _until(lambda: ti.fresh_resident(b.root, SESSION) is not None)
        # There is no requester-cancel wire operation today. Closing its real
        # link ends the requester, but owner observation persists to its bound.
        next(m for m in local.members if m.device_id == a.identity.device_id).capabilities = [
            "view"
        ]
        store.save(local, b.root)
        ti._reset_for_tests()
        release.clear()
        entered.clear()
        pending4, answers4 = _install(b, monkeypatch, deadline=0.3)
        closed = pool.submit(
            link.request,
            {
                "op": "net_session_history",
                "req": 500,
                "locality": "remote",
                "session_id": SESSION,
                "source_read": "availability",
            },
            timeout=2,
        )
        assert entered.wait(1) and _until(lambda: len(pending4) == 1)
        link_close_at = time.monotonic()
        link.close()
        assert closed.result(2) is None
        assert _until(lambda: not pending4)
        assert answers4 == [], "closed link must not receive a late source answer"
        matrix["link_close_observer_seconds"] = time.monotonic() - link_close_at
        release.set()
        matrix["link_close_cleanup"] = (
            "requester stops; owner observer removed at 0.3s bound; zero late answers"
        )
        states = [row["state"] for row in b.local_session_rows() if row["session_id"] == SESSION]
        assert states == ["stored"]
        assert wake_store.read_entry(b.root, SESSION)["stopped_at"] == 1234
        matrix["deliberate_stop_marker_preserved"] = True
        matrix["stored_owner_runtime"] = "no runtime/lease started for availability target"
        matrix["latency_seconds"] = {
            "inline_ping_timeout": inline_ping_seconds,
            "deferred_ping": responsive_ping_seconds,
            "fixture_prompt_delivery": prompt_seconds,
            "fixture_live_event_delivery": live_seconds,
        }
        matrix["producer_scope"] = (
            "actual relay/API and RuntimeServer socket transport into fixture FakeHandle; "
            "live producer is FakeHandle._event_handler injection, "
            "NOT Session production admission or native E2E"
        )
        matrix["bounds_seconds"] = {
            "main_availability": 6,
            "timeout_and_link_close": 0.3,
            "shared_refresh_test_hold": 8,
            "inline_test_hold": 4,
            "outer_process_group": 90,
        }
    finally:
        release.set()
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
        requests = [
            {
                "op": "net_session_history",
                "req": 200 + i,
                "locality": "remote",
                "session_id": SESSION,
                "source_read": "availability",
            }
            for i in range(4)
        ]
        requests.append(prompt_request)
        request_body = out / "paired-requests.private.json"
        request_body.write_text(json.dumps(requests))
        request_body.chmod(0o600)
        (out / "matrix-B.json").write_text(json.dumps(matrix, indent=2))
