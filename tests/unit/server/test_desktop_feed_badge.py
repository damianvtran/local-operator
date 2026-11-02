"""Native attention startup levels reuse existing feed authority, not banners.

These are focused source-contract tests. They do not execute Electron or
prove native paint, and use only synthetic roots/status/receipt identities.
"""
from pathlib import Path
from typing import Any

import pytest

from local_operator.server.utils.desktop_feed import DesktopFeed


def test_badge_seed_reuses_primed_status_without_rederiving_catalogue(tmp_path: Path, monkeypatch: Any) -> None:
    feed = DesktopFeed(tmp_path, bridged=lambda: set())
    states = {"session/synthetic": {"unseen": True, "completion_token": "token", "revision": [1, 0]}}
    monkeypatch.setattr(feed, "_snapshot", lambda: (states, 4, 5))
    feed._status_seen = {"synthetic": ("approval", "Waiting"), "not-in-authority": ("complete", "Done")}
    feed._status_revision = {"synthetic": 7}
    feed._notifications_seen = (8, 2, 1, 0)
    attention, catalogue, authoring, statuses, inbox = feed._badge_open_snapshot()
    assert attention is states
    assert (catalogue, authoring) == (4, 5)
    assert statuses == {"session/synthetic": {"code": "approval", "revision": 7}}
    assert inbox == {"revision": 8, "unread_count": 2, "waiting_count": 1, "waiting_unread_count": 0}


def test_badge_seed_preserves_unknown_inbox_and_missing_status(tmp_path: Path, monkeypatch: Any) -> None:
    feed = DesktopFeed(tmp_path, bridged=lambda: set())
    monkeypatch.setattr(feed, "_snapshot", lambda: ({"session/synthetic": {}}, 0, 0))
    assert feed._badge_open_snapshot()[3:] == ({}, None)


def test_approval_projection_reuses_requested_authority_and_expiry_gate(tmp_path: Path, monkeypatch: Any) -> None:
    from local_operator.network import approvals
    from local_operator.server.utils import desktop_feed

    feed = DesktopFeed(tmp_path, bridged=lambda: set())
    calls = []
    monkeypatch.setattr(desktop_feed.time, "time", lambda: 100.0)
    monkeypatch.setattr(approvals, "list_records", lambda **kwargs: calls.append(kwargs) or [
        {"state": "requested", "expires_at": 110.0}, {"state": "approved", "expires_at": 120.0}
    ])
    token, pending, expiry = feed._approval_snapshot()
    assert pending is True and expiry == 110.0
    feed._approvals_token, feed._approvals_pending, feed._approvals_expiry = token, pending, expiry
    assert feed._approval_snapshot() == (token, True, 110.0)
    assert len(calls) == 1, "quiet directory/expiry state must not rescan records"
    monkeypatch.setattr(desktop_feed.time, "time", lambda: 111.0)
    monkeypatch.setattr(approvals, "list_records", lambda **kwargs: [])
    assert feed._approval_snapshot()[1:] == (False, 0.0)


@pytest.mark.asyncio
async def test_tick_approval_edge_survives_another_subscriber_open(tmp_path: Path, monkeypatch: Any) -> None:
    from types import SimpleNamespace

    feed = DesktopFeed(tmp_path, bridged=lambda: set())
    feed._authoring_token = 1
    feed._approvals_pending = False
    monkeypatch.setattr(feed, "_badge_open_snapshot", lambda: ({}, 0, 0, {}, None))
    monkeypatch.setattr(feed, "_approval_snapshot", lambda: ((None, None), True, 0.0))
    frames = []
    monkeypatch.setattr(feed, "_publish", lambda kind, payload: frames.append((kind, payload)))
    opening_b = await feed._open_frame(SimpleNamespace(id="subscriber-b"))
    assert opening_b["payload"]["approvals_pending"] is True
    assert feed._approvals_pending is False, "B must not consume A's pending ticker edge"
    await feed._maybe_emit_approvals()
    assert frames == [("approvals", {"pending": True})]
    assert feed._approvals_pending is True


@pytest.mark.asyncio
async def test_open_cursor_precedes_edges_queued_during_snapshot(tmp_path: Path, monkeypatch: Any) -> None:
    from types import SimpleNamespace

    feed = DesktopFeed(tmp_path, bridged=lambda: set())
    feed._authoring_token = 1
    queued = []
    def snapshot() -> Any:
        queued.append(feed._frame("notifications", {"revision": 2, "unread_count": 0,
            "waiting_count": 0, "waiting_unread_count": 0}))
        return {}, 0, 0, {}, {"revision": 1, "unread_count": 1, "waiting_count": 0, "waiting_unread_count": 0}
    monkeypatch.setattr(feed, "_badge_open_snapshot", snapshot)
    monkeypatch.setattr(feed, "_approval_snapshot", lambda: ((None, None), False, 0.0))
    opening = await feed._open_frame(SimpleNamespace(id="subscriber-b"))
    assert opening["seq"] < queued[0]["seq"]
    assert opening["payload"]["notifications"]["unread_count"] == 1
    assert queued[0]["payload"]["unread_count"] == 0


@pytest.mark.asyncio
async def test_approval_failures_back_off_on_existing_probe_clock(tmp_path: Path, monkeypatch: Any) -> None:
    from local_operator.server.utils import desktop_feed

    feed = DesktopFeed(tmp_path, bridged=lambda: set())
    clock = [0.0]
    calls = []
    monkeypatch.setattr(desktop_feed.time, "monotonic", lambda: clock[0])
    def broken() -> Any:
        calls.append(clock[0])
        raise OSError("synthetic optional store failure")
    monkeypatch.setattr(feed, "_approval_snapshot", broken)
    await feed._maybe_emit_approvals()
    clock[0] = 0.1
    await feed._maybe_emit_approvals()
    assert calls == [0.0]
    clock[0] = desktop_feed.CATALOGUE_PROBE_INTERVAL_S
    await feed._maybe_emit_approvals()
    assert len(calls) == 2
    assert feed._approvals_pending is None
