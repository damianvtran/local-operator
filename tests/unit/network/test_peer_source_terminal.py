"""Failed source waves are terminal availability, not unsupported ordering.

The slow-owner cell uses the product bridge/control/paired TCP path and the real
index scanner, held only at its disk-work seam. Race cells deliberately control
the asynchronous completion seam: they prove latch ownership, not transport.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from local_operator.network import projection
from local_operator.resume import SessionRow
from local_operator.server.utils.desktop_sessions import DesktopSessionBridge
from local_operator.session import transcript_index as ti
from tests.unit.network.test_peer_source_contract import _bridge
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixture registration
    devices,
)
from tests.unit.network.test_session_plane import (
    SESSION,
    _dial_to,
    _listen,
    _pair,
    _seed_journal,
)


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("devices")


@pytest.mark.asyncio
async def test_failed_wave_real_pair_terminal_retry_and_legacy(
    peer_pair: Any, monkeypatch: Any
) -> None:
    a, b, _, _ = peer_pair
    ids = _seed_journal(b.root, SESSION, [""])
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    gate = threading.Event()
    scanner = ti._scan_full
    waits: list[bool] = []
    answer_availability = b._answer_source_availability

    def slow(path: Any, naming: Any) -> Any:
        if str(b.root) in str(path):
            assert gate.wait(12), "test scanner backstop expired"
        return scanner(path, naming)

    def observe(owner_link: Any, frame: Any, *, ready: bool) -> None:
        waits.append(ready)
        answer_availability(owner_link, frame, ready=ready)

    monkeypatch.setattr(ti, "_scan_full", slow)
    monkeypatch.setattr(b, "_answer_source_availability", observe)
    row = SessionRow(
        id=SESSION,
        mtime=0,
        name="",
        live_state="idle",
        locality="remote",
        owner_device=b.identity.device_id,
        owner_device_name="owner",
        reachable=True,
        unreachable_reason="",
    )
    bridge = DesktopSessionBridge(a.root, SESSION, "", remote_row=row)
    subscriber = bridge.subscribe(journal_order=True, peer_journal_source=True)

    async def typed(purpose: str = "metadata") -> dict[str, Any]:
        return await bridge.history(
            limit=2,
            journal_order=True,
            order_ids=[ids[0]],
            peer_journal_source=True,
            source_read=purpose,
            expected_journal_generation="unproven" if purpose == "payload" else None,
        )

    try:
        legacy_before = await bridge.history(limit=2)
        cold = await typed()
        assert cold["journal_order_state"] == "building"
        assert cold["peer_journal_source"] is True
        task = bridge._source_wait_task
        assert task is not None
        await asyncio.wait_for(task, 7)
        assert waits == [False]
        frame = subscriber.queue.get_nowait()
        assert frame is not None
        event = frame[0]
        assert event["payload"] == {
            "type": "journal_order_ready",
            "reconcile": True,
            "peer_source": True,
        }
        for purpose in ("metadata", "payload"):
            retry = await typed(purpose)
            assert retry == {**cold, "journal_order_state": "unavailable"}
        page = await bridge.history(limit=2, journal_order=True, peer_journal_source=True)
        assert page["journal_order_state"] == "unavailable"
        assert page["peer_journal_source"] is True
        assert "reconcile" not in page and "journal_generation" not in page
        # A fresh declaration on the SAME bridge cannot reset a failed wave.
        bridge.subscribe(journal_order=True, peer_journal_source=True)
        assert (await typed())["journal_order_state"] == "unavailable"
        assert bridge._source_wait_task is task and waits == [False]
        assert subscriber.queue.empty()
        legacy_after = await bridge.history(limit=2)
        assert legacy_after == legacy_before
        for key in ("journal_order_state", "peer_journal_source", "reconcile"):
            assert key not in legacy_after
        gate.set()
        # Join the existing owner scan rather than invent another refresh/read.
        future = asyncio.run_coroutine_threadsafe(_join_index(b.root), b._page_loop)
        await asyncio.wait_for(asyncio.wrap_future(future), 4)
        assert ti.fresh_resident(b.root, SESSION) is not None
        assert subscriber.queue.empty(), "index completion must not fabricate a bridge wake"
        ready = await typed()
        assert ready["journal_order_state"] == "ready"
        assert ready["journal_ordinals"] == {ids[0]: 0}
        assert bridge._source_wait_failed is False and waits == [False]
        missing = await bridge.history(
            journal_order=True,
            peer_journal_source=True,
            source_read="metadata",
            order_ids=["absent"],
        )
        assert missing["journal_missing_ids"] == ["absent"]
        assert "journal_ordinals" not in missing
    finally:
        gate.set()
        await bridge.close()
        link.close()


async def _join_index(root: Any) -> None:
    task, _ = ti.start_refresh(root, SESSION)
    await task


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["close", "owner", "client", "ready", "unsupported"])
@pytest.mark.parametrize("outcome", ["ready", "unavailable"])
async def test_old_flight_cannot_write_current_wave(
    tmp_path: Any,
    monkeypatch: Any,
    change: str,
    outcome: str,
) -> None:
    bridge, row = _bridge(tmp_path)
    subscriber = bridge.subscribe(journal_order=True, peer_journal_source=True)
    started = threading.Event()
    finish = threading.Event()

    def availability(*args: Any, **kwargs: Any) -> str:
        started.set()
        assert finish.wait(4)
        return outcome

    monkeypatch.setattr(projection, "peer_source_availability", availability)
    bridge._note_source_availability("building")
    task = bridge._source_wait_task
    assert task is not None
    assert await asyncio.to_thread(started.wait, 2)
    if change == "close":
        # Do not cancel: the fence itself must work even if cancellation loses.
        bridge._closing = True
    elif change == "owner":
        bridge.remote_row = row._replace(owner_device="new-owner")
    elif change == "client":
        new_client = object()
        monkeypatch.setattr(bridge, "_owner_connection", lambda: new_client)
    else:
        bridge._note_source_availability(change)
    finish.set()
    await asyncio.gather(task, return_exceptions=True)
    assert not bridge._source_wait_failed
    assert subscriber.queue.empty()
    await bridge.close()


def test_normalization_preserves_facts_and_source_change_drops_failure(tmp_path: Any) -> None:
    bridge, row = _bridge(tmp_path)
    witness = bridge._source_witness()
    bridge._source_wait_failed = True
    answer = {
        "entries": [{"id": "retained", "body": "not reattested"}],
        "has_more": True,
        "cursor_missing": False,
        "has_newer": True,
        "peer_journal_source": True,
        "journal_order_state": "building",
        "journal_generation": "proven",
        "journal_ordinals": {"retained": 7},
    }
    assert bridge._source_answer(answer, witness) == {
        **answer,
        "journal_order_state": "unavailable",
    }
    assert answer["journal_order_state"] == "building"
    bridge.remote_row = row._replace(owner_device="new-owner")
    assert not bridge._source_witness_current(witness)
    assert not bridge._source_wait_failed


@pytest.mark.asyncio
async def test_real_revocation_during_availability_cannot_establish_failed_wave(
    peer_pair: Any,
    monkeypatch: Any,
) -> None:
    from local_operator.network import store as network_store

    a, b, _, _ = peer_pair
    ids = _seed_journal(b.root, SESSION, [""])
    record, _, _ = _pair(peer_pair, monkeypatch, role="read")
    host, port = _listen(b)
    link = _dial_to(a, record, host, port)
    gate = threading.Event()
    registered = threading.Event()
    scanner = ti._scan_full
    dispatch = b._dispatch_source_availability

    def slow(path: Any, naming: Any) -> Any:
        if str(b.root) in str(path):
            assert gate.wait(8)
        return scanner(path, naming)

    def observe(owner_link: Any, frame: Any) -> Any:
        result = dispatch(owner_link, frame)
        registered.set()
        return result

    monkeypatch.setattr(ti, "_scan_full", slow)
    monkeypatch.setattr(b, "_dispatch_source_availability", observe)
    row = SessionRow(
        id=SESSION,
        mtime=0,
        name="",
        live_state="idle",
        locality="remote",
        owner_device=b.identity.device_id,
        owner_device_name="owner",
        reachable=True,
        unreachable_reason="",
    )
    bridge = DesktopSessionBridge(a.root, SESSION, "", remote_row=row)
    subscriber = bridge.subscribe(journal_order=True, peer_journal_source=True)
    wrong = DesktopSessionBridge(a.root, "ffffffffffe1", "", remote_row=row)
    try:
        wrong_answer = await wrong.history(
            journal_order=True,
            peer_journal_source=True,
            source_read="metadata",
            order_ids=ids,
        )
        assert wrong_answer["journal_order_state"] == "unsupported"
        assert "journal_generation" not in wrong_answer
        assert "peer_journal_source" not in wrong_answer
        cold = await bridge.history(
            journal_order=True,
            peer_journal_source=True,
            source_read="metadata",
            order_ids=ids,
        )
        assert cold["journal_order_state"] == "building"
        task = bridge._source_wait_task
        assert task is not None
        assert await asyncio.to_thread(registered.wait, 2)
        membership = network_store.load(record.network_id, b.root)
        assert membership is not None
        member = next(m for m in membership.members if m.device_id == a.identity.device_id)
        member.capabilities = []
        network_store.save(membership, b.root)
        gate.set()
        await asyncio.wait_for(task, 5)
        assert bridge._source_wait_failed is False
        assert subscriber.queue.empty()
        denied = await bridge.history(
            journal_order=True,
            peer_journal_source=True,
            source_read="metadata",
            order_ids=ids,
        )
        assert denied == wrong_answer
        page = await bridge.history(journal_order=True, peer_journal_source=True)
        assert page["journal_order_state"] == "unsupported"
        assert "peer_journal_source" not in page and "journal_generation" not in page
    finally:
        gate.set()
        await wrong.close()
        await bridge.close()
        link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["typed", "page"])
@pytest.mark.parametrize("change", ["owner", "client", "close", "ready"])
async def test_read_response_must_keep_its_source_wave(
    tmp_path: Any,
    monkeypatch: Any,
    route: str,
    change: str,
) -> None:
    bridge, row = _bridge(tmp_path)
    started = threading.Event()
    finish = threading.Event()
    answer = {
        "entries": [],
        "has_more": False,
        "cursor_missing": False,
        "has_newer": None,
        "peer_journal_source": True,
        "journal_order_state": "ready",
        "journal_generation": "old",
        "journal_ordinals": {"x": 0},
    }

    def read(*args: Any, **kwargs: Any) -> dict[str, Any]:
        started.set()
        assert finish.wait(4)
        return answer

    monkeypatch.setattr(projection, "peer_source_read", read)
    monkeypatch.setattr(projection, "peer_stored_history_page", read)
    if route == "typed":
        request = bridge.history(
            journal_order=True,
            peer_journal_source=True,
            source_read="metadata",
            order_ids=["x"],
        )
    else:
        request = bridge.history(journal_order=True, peer_journal_source=True)
    task = asyncio.create_task(request)
    assert await asyncio.to_thread(started.wait, 2)
    if change == "owner":
        bridge.remote_row = row._replace(owner_device="new-owner")
    elif change == "client":
        client = object()
        monkeypatch.setattr(bridge, "_owner_connection", lambda: client)
    elif change == "close":
        bridge._closing = True
    else:
        bridge._note_source_availability("ready")
        # A subsequent wave's failure must survive the OLD ready response.
        bridge._source_wait_failed = True
    finish.set()
    response = await task
    assert response["journal_order_state"] == "unsupported"
    assert "journal_generation" not in response and "journal_ordinals" not in response
    assert bridge._source_wait_failed is (change == "ready")
    await bridge.close()


@pytest.mark.asyncio
async def test_warm_wire_window_keeps_body_and_failed_availability(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from types import SimpleNamespace

    from local_operator.harness.types import Message

    bridge, _ = _bridge(tmp_path)
    row = Message.user("warm body remains")
    remote = SimpleNamespace(
        is_cold=False,
        history=lambda: [row],
        history_before_token=None,
        history_entry_times=lambda: {row.id: 123.0},
    )
    bridge.remote = remote
    bridge._source_witness()
    bridge._source_wait_failed = True
    monkeypatch.setattr(
        projection,
        "peer_stored_history_page",
        lambda *a, **k: {
            "entries": [],
            "has_more": False,
            "cursor_missing": False,
            "has_newer": None,
            "journal_order_state": "building",
            "peer_journal_source": True,
        },
    )
    before = await bridge.history()
    after = await bridge.history(journal_order=True, peer_journal_source=True)
    assert after == {**before, "journal_order_state": "unavailable"}
    assert bridge._source_wait_task is None
    # This facade is a read-only double, so don't dispatch attach disposal on it.
    bridge.remote = None
    await bridge.close()


@pytest.mark.asyncio
async def test_actual_disposal_cancels_flight_without_late_latch(
    tmp_path: Any, monkeypatch: Any
) -> None:
    bridge, _ = _bridge(tmp_path)
    started = threading.Event()
    finish = threading.Event()
    returned = threading.Event()

    def wait(*args: Any, **kwargs: Any) -> str:
        started.set()
        assert finish.wait(4)
        returned.set()
        return "unavailable"

    monkeypatch.setattr(projection, "peer_source_availability", wait)
    bridge._note_source_availability("building")
    task = bridge._source_wait_task
    assert task is not None
    assert await asyncio.to_thread(started.wait, 2)
    try:
        await bridge.close()
    finally:
        finish.set()
    await asyncio.gather(task, return_exceptions=True)
    assert await asyncio.to_thread(returned.wait, 2)
    assert not bridge._source_wait_failed
    assert bridge._closing and not bridge.subscribers
