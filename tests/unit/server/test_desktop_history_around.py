"""The anchored history read over the REAL route (design §D4/§D9).

The reader's own contract is pinned in ``tests/unit/session/test_transcript.py``
(the forward oracle) and the cache's in ``test_page_cache.py``; this file is the
half a client actually meets: the query parameters, the statuses for a request
that names no window or an out-of-range bound, the reconciliation answer for an
absent journal, and the peer-session branch that v1 answers unsupported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.resume import SessionRow
from local_operator.server.routes import desktop_sessions
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
)

TOKEN = "synthetic-desktop-token"


@pytest.fixture(autouse=True)
def _desktop_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every row runs against a synthetic HOME, not the operator's."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


def _seed_journal(directory: Path, rows: int) -> list[str]:
    """``rows`` message rows in journal order; returns their ids, oldest first."""
    directory.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    lines: list[str] = []
    for index in range(rows):
        entry = TranscriptEntry(
            f"{index:012x}",
            float(index),
            ENTRY_MESSAGE,
            {"role": "user", "content": f"message {index}"},
        )
        ids.append(entry.id)
        lines.append(entry.to_json())
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids


class _Harness:
    """One pool, one real router app, one session with a seeded journal."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.app = FastAPI()
        self.app.include_router(desktop_sessions.router)
        self.pool = DesktopSessions(root)
        self.app.state.desktop_sessions = self.pool
        # The route ladder resolves some arms through app state; an isolated
        # manager keeps those off the operator's own config.
        self.app.state.config_manager = ConfigManager(config_dir=root)
        self.session_id = ""
        self.ids: list[str] = []
        self.client: AsyncClient | None = None

    async def __aenter__(self) -> "_Harness":
        self.session_id = await self.pool.create(str(self.root))
        self.ids = _seed_journal(self.root / "sessions" / self.session_id, rows=8)
        self.client = AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self.client is not None:
            await self.client.aclose()


@pytest.mark.asyncio
async def test_history_around_returns_the_window_and_both_edges(tmp_path: Path) -> None:
    """The anchored read, end to end: rows oldest→newest, flags on the wire."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}"

        response = await harness.client.get(
            f"{base}/history",
            params={"around_id": harness.ids[3], "before": 1, "after": 2},
        )
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        assert [row["id"] for row in page["entries"]] == harness.ids[2:6]
        assert page["has_more"] is True
        assert page["has_newer"] is True
        assert page["cursor_missing"] is False

        at_tail = await harness.client.get(
            f"{base}/history",
            params={"around_id": harness.ids[-1], "before": 1, "after": 1},
        )
        assert at_tail.status_code == 200, at_tail.text
        tail_page = at_tail.json()["result"]
        assert [row["id"] for row in tail_page["entries"]] == harness.ids[-2:]
        assert tail_page["has_newer"] is False
        assert tail_page["has_more"] is True


@pytest.mark.asyncio
async def test_history_around_defaults_centre_the_window_on_the_id(tmp_path: Path) -> None:
    """An around request without counts still answers a ``limit``-sized page."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(
            f"/v1/desktop/sessions/{harness.session_id}/history",
            params={"around_id": harness.ids[4], "limit": 4},
        )
        assert response.status_code == 200, response.text
        assert [row["id"] for row in response.json()["result"]["entries"]] == harness.ids[2:6]


@pytest.mark.asyncio
async def test_history_around_unknown_session_is_the_shared_404(tmp_path: Path) -> None:
    """THE 404 BELONGS TO THE LOOKUP: an id nobody holds is unknown, not empty."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(
            "/v1/desktop/sessions/ffffffffffff/history", params={"around_id": "deadbeef"}
        )
        assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_history_around_rejects_malformed_requests(tmp_path: Path) -> None:
    """A request that describes no window fails the same way at every door.

    The combinations go through the reader's own validator (409, the reader's
    sentence — the same refusal the sidecar tests pin), while the numeric
    bounds are the wire's (422 from the query declaration).
    """
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        base = f"/v1/desktop/sessions/{harness.session_id}"

        both = await harness.client.get(
            f"{base}/history",
            params={"around_id": harness.ids[1], "before_id": harness.ids[0]},
        )
        assert both.status_code == 409, both.text
        assert "choose at most one" in both.json()["detail"]

        counts = await harness.client.get(f"{base}/history", params={"before": 2})
        assert counts.status_code == 409, counts.text
        assert "only meaningful with around_id" in counts.json()["detail"]

        negative = await harness.client.get(
            f"{base}/history", params={"around_id": harness.ids[1], "before": -1}
        )
        assert negative.status_code == 422, negative.text

        over = await harness.client.get(
            f"{base}/history", params={"around_id": harness.ids[1], "after": 501}
        )
        assert over.status_code == 422, over.text


@pytest.mark.asyncio
async def test_history_around_on_an_absent_journal_reconciles(tmp_path: Path) -> None:
    """A session with no journal answers the empty reconciled page, not a 404."""
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        other = await harness.pool.create(str(tmp_path))
        response = await harness.client.get(
            f"/v1/desktop/sessions/{other}/history", params={"around_id": "deadbeef"}
        )
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        assert page["entries"] == []
        assert page["cursor_missing"] is True
        assert page["has_newer"] is None


@pytest.mark.asyncio
async def test_history_around_on_a_peer_session_is_an_unsupported_empty_page(
    tmp_path: Path,
) -> None:
    """V1 cannot anchor a peer's wire window; the answer is the empty reconciled page.

    Pinned at the facade rather than the route because a real peer bridge needs
    the mesh: the branch under test is the facade's ``remote_row`` switch. The
    plain remote read beside it is the one that CHANGED with D5-core (design
    ``docs/design/mesh-cold-read-stored-history.md``): a cold peer page is now
    read from the OWNER's stored journal, and this bridge has no relay and no
    owner at all, so the page is UNSERVABLE -- and an unservable page says so
    (``cursor_missing: True``) rather than answering the empty triple a
    conversation with no rows produces. That distinction is the point of the
    change: an empty page on a peer now means the owner has no rows. What a
    served cold page looks like is pinned over two real relays in
    ``tests/unit/network/test_remote_viewer.py``.
    """
    row = SessionRow(
        id="ffffffffffff",
        mtime=1789400000.0,
        name="peer chat",
        live_state="idle",
        locality="remote",
        owner_device="peer-device",
        owner_device_name="build-box",
        reachable=True,
        unreachable_reason="",
    )
    bridge = DesktopSessionBridge(tmp_path, row.id, "", remote_row=row)

    anchored = await bridge.history(around_id="deadbeef", before=2, after=2)
    assert anchored == {
        "entries": [],
        "has_more": False,
        "cursor_missing": True,
        "has_newer": None,
    }

    plain = await bridge.history(limit=10)
    assert plain == {
        "entries": [],
        "has_more": False,
        # CHANGED BY DESIGN (D5-core): the reader could not fetch the owner's
        # stored page (no relay on this device at all), so the page is marked
        # untrustworthy rather than published as "this conversation is empty".
        "cursor_missing": True,
        "has_newer": None,
    }

    # Peer parity (remediation round 1, R1-1): a malformed request must refuse
    # the SAME way here as on a local session — the remote branch is reached
    # only by requests that pass the shared validator first.
    with pytest.raises(ValueError, match="only meaningful with around_id"):
        await bridge.history(before=2)
    with pytest.raises(ValueError, match="choose at most one"):
        await bridge.history(around_id="deadbeef", before_id="deadbeef")


@pytest.mark.asyncio
async def test_history_before_id_keeps_has_newer_null(tmp_path: Path) -> None:
    """The unchanged reads carry the null on the wire: an unasked question
    is not answered False (remediation round 1, R1-3).

    The anchored read is the only one whose contract bounds its newer side, so
    a paging read must serialize ``has_newer: null`` — a client that branches
    on the key must see "not asked", never a claim.
    """
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(
            f"/v1/desktop/sessions/{harness.session_id}/history",
            params={"before_id": harness.ids[4], "limit": 2},
        )
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        assert [row["id"] for row in page["entries"]] == harness.ids[2:4]
        assert page["has_newer"] is None
        assert page["cursor_missing"] is False


def _seed_order_journal(directory: Path, ids: list[str]) -> None:
    rows = [
        TranscriptEntry(
            id_,
            100.0 - at,
            ENTRY_MESSAGE,
            {
                "kind": "message",
                "role": "user",
                "content": [{"type": "text", "text": f"synthetic order row {at}"}],
                "tool_calls": [],
            },
        ).to_json()
        for at, id_ in enumerate(ids)
    ]
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_journal_order_is_opt_in_and_lookup_is_session_scoped(tmp_path: Path):
    from local_operator.session import transcript_index as ti

    ti._reset_for_tests()
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        _seed_order_journal(tmp_path / "sessions" / harness.session_id, harness.ids)
        base = f"/v1/desktop/sessions/{harness.session_id}"
        legacy = (await harness.client.get(base + "/history", params={"limit": 2})).json()["result"]
        assert not any(key.startswith("journal_") for key in legacy)
        assert all("journal_seq" not in row for row in legacy["entries"])
        task, _ = ti.start_refresh(tmp_path, harness.session_id)
        await task
        response = await harness.client.get(
            base + "/history",
            params=[
                ("journal_order", "1"),
                ("limit", "2"),
                ("order_ids", harness.ids[0]),
                ("order_ids", "not-in-this-session"),
            ],
        )
        assert response.status_code == 200, response.text
        page = response.json()["result"]
        assert page["journal_order_state"] == "ready"
        assert [row["journal_seq"] for row in page["entries"]] == [6, 7]
        assert page["journal_ordinals"] == {harness.ids[0]: 0}
        assert page["journal_missing_ids"] == ["not-in-this-session"]
        generation = page["journal_generation"]
        older = (
            await harness.client.get(
                base + "/history",
                params={"journal_order": 1, "before_id": harness.ids[6], "limit": 2},
            )
        ).json()["result"]
        assert [row["journal_seq"] for row in older["entries"]] == [4, 5]
        assert older["journal_generation"] == generation
        around = (
            await harness.client.get(
                base + "/history",
                params={"journal_order": 1, "around_id": harness.ids[3], "before": 1, "after": 1},
            )
        ).json()["result"]
        assert [row["journal_seq"] for row in around["entries"]] == [2, 3, 4]
        assert around["journal_generation"] == generation
        snapshot_response = await harness.client.get(base, params={"journal_order": 1})
        assert snapshot_response.status_code == 200, snapshot_response.text
        snapshot = snapshot_response.json()["result"]["payload"]["history"]
        assert snapshot["journal_generation"] == generation
        assert [row["journal_seq"] for row in snapshot["entries"]] == list(range(8))
        # New optional fields must not leak to an old strict viewer after warmup.
        old_again = (await harness.client.get(base + "/history", params={"limit": 2})).json()[
            "result"
        ]
        assert old_again == legacy
        await harness.pool.close()
    ti._reset_for_tests()


@pytest.mark.asyncio
async def test_journal_order_cold_index_has_no_fabricated_authority(tmp_path: Path):
    from local_operator.session import transcript_index as ti

    ti._reset_for_tests()
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        _seed_order_journal(tmp_path / "sessions" / harness.session_id, harness.ids)
        base = f"/v1/desktop/sessions/{harness.session_id}/history"
        first = await harness.client.get(base, params={"journal_order": 1})
        assert first.status_code == 200, first.text
        page = first.json()["result"]
        assert page["journal_order_state"] == "building"
        assert "journal_generation" not in page
        assert all("journal_seq" not in row for row in page["entries"])
        task, _ = ti.start_refresh(tmp_path, harness.session_id)
        await task
        second = (await harness.client.get(base, params={"journal_order": 1})).json()["result"]
        assert second["journal_order_state"] == "ready"
        assert second["journal_generation"]
        assert [row["journal_seq"] for row in second["entries"]] == list(range(8))
        await harness.pool.close()
    ti._reset_for_tests()


@pytest.mark.asyncio
async def test_cold_order_ids_correct_quiet_cached_echo_without_another_read(tmp_path, monkeypatch):
    import asyncio

    from local_operator.session import transcript_index as ti

    ti._reset_for_tests()
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        ids = [f"{index:012x}" for index in range(205)]
        _seed_order_journal(tmp_path / "sessions" / harness.session_id, ids)
        # Hold only the existing index scheduler, so the real router must return
        # a cold page. Releasing it is not a new page, commit, token or user action.
        release = asyncio.Event()
        original_refresh = ti.start_refresh

        async def delayed_refresh():
            await release.wait()
            actual, _ = original_refresh(tmp_path, harness.session_id)
            return await actual

        gated = asyncio.create_task(delayed_refresh())
        monkeypatch.setattr(ti, "start_refresh", lambda root, sid: (gated, False))
        async with harness.pool.session(harness.session_id, read=True, allow_draft=True) as bridge:
            sub = bridge.subscribe(journal_order=True)
            try:
                response = await harness.client.get(
                    f"/v1/desktop/sessions/{harness.session_id}/history",
                    params=[("journal_order", "1"), ("limit", "2"), ("order_ids", ids[0])],
                )
                assert response.status_code == 200, response.text
                page = response.json()["result"]
                assert page["journal_order_state"] == "building"
                assert "journal_generation" not in page
                assert ids[0] not in {row["id"] for row in page["entries"]}
                assert all("journal_seq" not in row for row in page["entries"])
                release.set()
                await asyncio.wait_for(gated, timeout=2)
                queued = await asyncio.wait_for(sub.queue.get(), timeout=2)
                assert queued is not None
                frame, _ = queued
                assert frame["payload"]["type"] == "journal_order_ready", frame
                correction = frame["payload"]
                assert correction["journal_generation"]
                assert {row["id"]: row["journal_seq"] for row in correction["rows"]} == {
                    ids[0]: 0,
                    ids[-2]: 203,
                    ids[-1]: 204,
                }
                assert sub.queue.empty()
            finally:
                release.set()
                await gated
                bridge.subscribers.pop(sub.id, None)
        await harness.pool.close()
    ti._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        [("journal_order", "2")],
        [("journal_order", "1"), ("order_ids", "")],
        [("journal_order", "1"), ("order_ids", "a" * 129)],
        [("journal_order", "1")] + [("order_ids", "x") for _ in range(513)],
    ],
)
async def test_journal_order_query_bounds_are_validation_errors(tmp_path: Path, params):
    async with _Harness(tmp_path) as harness:
        assert harness.client is not None
        response = await harness.client.get(
            f"/v1/desktop/sessions/{harness.session_id}/history", params=params
        )
        assert response.status_code == 422, response.text
        await harness.pool.close()


@pytest.mark.asyncio
async def test_order_ready_events_are_opt_in_for_live_and_replay(tmp_path: Path, monkeypatch):
    from local_operator.session.runtime.types import JOURNAL_ORDER_EVENT

    bridge = DesktopSessionBridge(tmp_path, "events-order-fixture", str(tmp_path))
    legacy = bridge.subscribe()
    current = bridge.subscribe(journal_order=True)
    bridge.publish(
        "event",
        {
            "type": JOURNAL_ORDER_EVENT,
            "journal_generation": "synthetic-domain",
            "rows": [{"id": "a", "journal_seq": 0}],
        },
    )
    assert legacy.queue.empty()
    assert current.queue.qsize() == 1

    async def snapshot(**kwargs):
        return {
            "session_id": bridge.session_id,
            "epoch": bridge.epoch,
            "seq": bridge.sequence,
            "type": "snapshot",
            "payload": {
                "frontend": {},
                "history": {"entries": [], "has_more": False, "cursor_missing": False},
            },
        }

    monkeypatch.setattr(bridge, "snapshot", snapshot)
    late_legacy = bridge.subscribe()
    stream = bridge.events(late_legacy, epoch=bridge.epoch, after_seq=0)
    try:
        # Consume handshake/snapshot. The unknown order event must not replay.
        first = await anext(stream)
        second = await anext(stream)
        assert first["type"] == "open"
        assert second["type"] == "snapshot"
        assert late_legacy.queue.empty()
        import asyncio

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(stream), timeout=0.05)
    finally:
        await stream.aclose()
        await bridge.close()


@pytest.mark.asyncio
async def test_peer_order_never_uses_receiver_index(tmp_path: Path, monkeypatch):
    row = SessionRow(
        id="fffffffffff1",
        mtime=1789400000.0,
        name="peer order",
        live_state="idle",
        locality="remote",
        owner_device="synthetic-peer",
        owner_device_name="peer-box",
        reachable=True,
        unreachable_reason="",
    )
    bridge = DesktopSessionBridge(tmp_path, row.id, str(tmp_path), remote_row=row)

    async def legacy_peer(**kwargs):
        assert kwargs["journal_order"] is True
        assert kwargs["order_ids"] == ["receiver-row"]
        return {
            "entries": [],
            "has_more": False,
            "cursor_missing": False,
            "journal_order_state": "unsupported",
        }

    monkeypatch.setattr(bridge, "_remote_history", legacy_peer)
    unknown = await bridge.history(journal_order=True, order_ids=["receiver-row"])
    assert unknown["journal_order_state"] == "unsupported"
    assert "journal_generation" not in unknown
    assert "journal_ordinals" not in unknown

    async def current_peer(**kwargs):
        return {
            "entries": [],
            "has_more": False,
            "cursor_missing": False,
            "journal_order_state": "ready",
            "journal_generation": "owner-domain",
            "journal_ordinals": {"owner-row": 7},
        }

    monkeypatch.setattr(bridge, "_remote_history", current_peer)
    known = await bridge.history(journal_order=True)
    assert known["journal_generation"] == "owner-domain"
    assert known["journal_ordinals"] == {"owner-row": 7}
    await bridge.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_in_replay", [False, True])
async def test_off_page_order_correction_survives_snapshot_cutover(
    tmp_path, monkeypatch, already_in_replay
):
    import asyncio

    from local_operator.session import transcript_index as ti
    from local_operator.session.runtime.types import JOURNAL_ORDER_EVENT

    ti._reset_for_tests()
    async with _Harness(tmp_path) as harness:
        ids = [f"{index:012x}" for index in range(205)]
        _seed_order_journal(tmp_path / "sessions" / harness.session_id, ids)
        task, _ = ti.start_refresh(tmp_path, harness.session_id)
        await task
        index = ti.fresh_resident(tmp_path, harness.session_id)
        assert index is not None
        async with harness.pool.session(harness.session_id, read=True, allow_draft=True) as bridge:
            sub = bridge.subscribe(journal_order=True)
            legacy = bridge.subscribe()
            payload = {
                "type": JOURNAL_ORDER_EVENT,
                "journal_generation": index.journal_generation,
                "rows": [{"id": ids[0], "journal_seq": 0, "role": "user"}],
            }
            if already_in_replay:
                bridge.publish("event", payload)

            async def attention_glance():
                # Only this decorative wait is controlled; snapshot/history and
                # the pre-open cutover are production. The old ID is off-page.
                if not already_in_replay:
                    bridge.publish("event", payload)
                await asyncio.sleep(0)

            monkeypatch.setattr(
                bridge, "_shared_attention_refresh", lambda: asyncio.create_task(attention_glance())
            )
            stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0, journal_order=True)
            try:
                assert (await anext(stream))["type"] == "open"
                received = []
                while True:
                    frame = await asyncio.wait_for(anext(stream), timeout=2)
                    if frame["type"] == "snapshot":
                        snapshot = frame["payload"]["history"]
                        break
                    received.append(frame)
                assert len(snapshot["entries"]) == 100
                assert ids[0] not in {row["id"] for row in snapshot["entries"]}
                assert ids[0] not in snapshot.get("journal_ordinals", {})
                assert snapshot["journal_generation"] == index.journal_generation
                if not already_in_replay:
                    received.append(await asyncio.wait_for(anext(stream), timeout=2))
                corrections = [
                    frame
                    for frame in received
                    if frame.get("payload", {}).get("type") == JOURNAL_ORDER_EVENT
                ]
                assert len(corrections) == 1
                assert corrections[0]["payload"] == payload
                assert legacy.queue.empty()
                # Ordinary watch/attention frames are independent of this
                # metadata proof. Only a duplicate of the same published
                # correction (not any subsequent frame) would violate cutover.
                while True:
                    try:
                        extra = await asyncio.wait_for(anext(stream), timeout=0.05)
                    except asyncio.TimeoutError:
                        break
                    assert extra["seq"] != corrections[0]["seq"], extra
            finally:
                await stream.aclose()
        await harness.pool.close()
    ti._reset_for_tests()


def test_source_route_defaults_are_plain_python_values():
    import inspect

    handlers = [
        value
        for _, value in inspect.getmembers(desktop_sessions, inspect.isfunction)
        if "journal_order" in inspect.signature(value).parameters
    ]
    assert len(handlers) == 3
    for handler in handlers:
        parameters = inspect.signature(handler).parameters
        assert parameters["journal_order"].default == 0
        assert parameters["order_ids"].default is None
