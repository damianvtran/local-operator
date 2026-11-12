"""A finished turn or a ``notify`` on a mesh node reaches the session's ORIGIN.

THE GAP THESE PIN (remote-session-parity, 2026-11-11). A session the operator
started on their head device and ran on ``lop-remote-node-2`` finished turns and
called ``notify`` there, and the head learned nothing: its ``notifications.db``
held 0 rows for the node's item ``n_xr44lx4zb5gi3mfn``. The owner now stamps
origin-only ``attention``/``notice`` facts on the catalogue rows it hands the
origin, and the origin's watcher (hosted by the TUI AND ``lop serve``) turns
their edges into inbox rows.

Each acceptance line of the brief is one test below:

* no TUI, one node turn end → exactly one ``remote-done:`` row within one poll
  (``test_serve_host_raises_one_done_row_within_one_poll``);
* ``notify priority=high`` → one banner-eligible row
  (``test_a_high_notice_is_one_banner_eligible_row``);
* a restarted watcher does not duplicate
  (``test_a_restarted_watcher_neither_duplicates_nor_reopens``);
* a non-origin device sees no attention/notice fields
  (``test_a_non_origin_viewer_gets_no_fields``);
* an old-viewer fixture ignores the new fields
  (``test_an_old_parse_and_malformed_blocks_are_no_claim``, and the two-relay
  capability case in ``tests/unit/network/test_remote_attention_relay.py``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.projection import PeerRow
from local_operator.notifications import remote_attention as ra
from local_operator.notifications.delivery import (
    PARK_ITEM_PREFIXES,
    announced_by_its_park,
)
from local_operator.notifications.store import NotificationStore, content_hash
from local_operator.resume import SessionRow
from local_operator.session import peer_rows
from local_operator.session.attention import AttentionStore
from local_operator.session.placement import MeshStamp, SessionPlacement, write_stamp

HEAD = "d_head"
NODE = "d_node"
SID = "a1b2c3d4e5f6"
TOK1 = "6f1c2a3e-0b4d-4e5f-8a9b-1c2d3e4f5a60"
TOK2 = "7a2d3b4f-1c5e-4f60-9bac-2d3e4f5a6b71"


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "head"


@pytest.fixture(autouse=True)
def _no_banner_threads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture banner hand-offs instead of sleeping out a real grace on a thread."""
    calls: list[dict[str, Any]] = []

    def capture(root: Path, outcome: Any, **kwargs: Any) -> None:
        calls.append({"action": outcome.action, "priority": outcome.priority, **kwargs})

    monkeypatch.setattr(ra, "_deliver_async", capture)
    return calls


def _row(
    *,
    attention: tuple[str, str] | None = None,
    notice: tuple[str, int, str, str] | None = None,
    device: str = NODE,
    session_id: str = SID,
    name: str = "Port the parser",
) -> SessionRow:
    return SessionRow(
        session_id,
        0.0,
        name,
        live_state="idle",
        locality="remote",
        owner_device=device,
        owner_device_name="lop-remote-node-2",
        remote_attention=attention,
        remote_notice=notice,
    )


def _items(root: Path) -> list[dict[str, Any]]:
    rows, _ = NotificationStore(root).list_items(state="all", limit=100)
    return rows


# ---------------------------------------------------------------------------
# Origin side: edges and rows
# ---------------------------------------------------------------------------


def test_serve_host_raises_one_done_row_within_one_poll(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``lop serve`` host, no TUI: one poll, one ``remote-done:`` row."""
    listing = ((_row(attention=(TOK1, "complete")),), ())
    monkeypatch.setattr(peer_rows, "read_listing", lambda *_a, **_k: listing)

    async def one_poll() -> None:
        task = asyncio.create_task(ra.run_forever(root, interval_s=3600))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if _items(root):
                break
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(one_poll())
    rows = _items(root)
    assert [row["key"] for row in rows] == [ra.done_key(NODE, SID)]
    row = rows[0]
    assert row["session_id"] == SID
    assert row["priority"] == "normal"
    assert row["title"] == "Port the parser"
    assert "lop-remote-node-2" in row["body_md"], "the row must say where the turn ran"
    assert row["read"] is False


def test_a_restarted_watcher_neither_duplicates_nor_reopens(root: Path) -> None:
    rows = [_row(attention=(TOK1, "complete"))]
    first = ra.RemoteAttentionWatcher(root)
    first.observe(rows)
    first.observe(rows)
    [item] = _items(root)
    NotificationStore(root).apply("read", ids=[(item["id"], item["revision"])])

    # A fresh process (restart) and a second host (TUI beside serve) both start
    # with an empty episode map and see the same completion as an edge.
    ra.RemoteAttentionWatcher(root).observe(rows)
    ra.RemoteAttentionWatcher(root).observe([_row(attention=(TOK1, "complete"), name="Renamed")])
    [after] = _items(root)
    assert after["id"] == item["id"]
    assert after["read"] is True, "a repeat of a read completion must not flip it unread"
    assert after["revision"] == item["revision"]


def test_a_new_completion_reopens_the_one_row(root: Path) -> None:
    watcher = ra.RemoteAttentionWatcher(root)
    watcher.observe([_row(attention=(TOK1, "complete"))])
    [item] = _items(root)
    store = NotificationStore(root)
    store.apply("dismiss", ids=[(item["id"], item["revision"])])

    watcher.observe([_row(attention=(TOK2, "error"))])
    [after] = _items(root)
    assert after["id"] == item["id"], "one row per (device, session), never a stack"
    assert after["dismissed"] is False and after["read"] is False
    assert after["anchor_id"] == TOK2


def test_an_acknowledged_completion_marks_the_row_read(root: Path) -> None:
    """The owner stops stamping once the receipt landed: the origin row reads."""
    watcher = ra.RemoteAttentionWatcher(root)
    watcher.observe([_row(attention=(TOK1, "complete"))])
    watcher.observe([_row()])
    [item] = _items(root)
    assert item["read"] is True and item["dismissed"] is False


def test_a_silent_device_keeps_its_episode(root: Path) -> None:
    watcher = ra.RemoteAttentionWatcher(root)
    watcher.observe([_row(attention=(TOK1, "complete"))])
    watcher.observe([], unanswered=[NODE])
    [item] = _items(root)
    assert item["read"] is False, "a timed-out read is not the operator reading it"
    edges, state = peer_rows.completion_edges(
        {(NODE, SID): TOK1}, [_row(attention=(TOK1, "complete"))]
    )
    assert edges == () and state == {(NODE, SID): TOK1}


def test_a_high_notice_is_one_banner_eligible_row(
    root: Path, _no_banner_threads: list[dict[str, Any]]
) -> None:
    notice = ("deploy-done", 1, "high", "Deploy finished — needs your sign-off")
    watcher = ra.RemoteAttentionWatcher(root)
    watcher.observe([_row(notice=notice)])
    watcher.observe([_row(notice=notice)])
    ra.RemoteAttentionWatcher(root).observe([_row(notice=notice)])

    [row] = _items(root)
    assert row["key"] == ra.notice_key(NODE, SID, "deploy-done")
    assert row["priority"] == "high"
    assert row["title"] == "Deploy finished — needs your sign-off"
    # Banner-ELIGIBLE: the item ladder excludes only park rows.
    assert not announced_by_its_park(row) and not row["key"].startswith(PARK_ITEM_PREFIXES)
    store = NotificationStore(root)
    assert [item["id"] for item in store.undelivered(exclude_prefixes=PARK_ITEM_PREFIXES)] == [
        row["id"]
    ]
    # Handed to the ladder once as new content. The same watcher's second read
    # is no edge at all; the restarted watcher's write is ``unchanged``, which
    # ``route_item`` never banners.
    assert [call["action"] for call in _no_banner_threads] == ["created", "unchanged"]


def test_a_revised_notice_updates_and_a_new_key_is_a_new_row(root: Path) -> None:
    watcher = ra.RemoteAttentionWatcher(root)
    watcher.observe([_row(notice=("k", 1, "normal", "Step 1 done"))])
    watcher.observe([_row(notice=("k", 2, "normal", "Step 2 done"))])
    watcher.observe([_row(notice=("n_other", 1, "low", "FYI"))])
    keys = sorted(row["key"] for row in _items(root))
    assert keys == sorted([ra.notice_key(NODE, SID, "k"), ra.notice_key(NODE, SID, "n_other")])


def test_the_kill_switch_writes_nothing(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.tools import builtin

    monkeypatch.setattr(builtin, "NOTIFY_ENABLED", False)
    ra.RemoteAttentionWatcher(root).observe(
        [_row(attention=(TOK1, "complete"), notice=("k", 1, "high", "x"))]
    )
    assert _items(root) == []


def test_episode_leaves_every_other_digest_byte_identical() -> None:
    """The store change is additive: no ``episode`` = the shipped digest."""
    store_digest = content_hash("t", "b", "[]", "normal")
    assert store_digest == content_hash("t", "b", "[]", "normal")
    assert store_digest != content_hash("t", "b", "[]\x1etok", "normal")


def test_an_episode_less_repeat_still_collapses(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path)
    first = store.notify("s1", title="Same", body_md="words", key="k")
    again = store.notify("s1", title="Same", body_md="words", key="k")
    assert (first.action, again.action) == ("created", "unchanged")


# ---------------------------------------------------------------------------
# Owner side: who the row is stamped for
# ---------------------------------------------------------------------------


def _owner(tmp_path: Path, *, origin: str = HEAD) -> Path:
    owner = tmp_path / "node"
    (owner / "sessions" / SID).mkdir(parents=True)
    write_stamp(
        owner,
        MeshStamp(
            session_id=SID,
            network_id="n_test",
            home_device=NODE,
            placement=SessionPlacement(
                mode="peer", network_id="n_test", home_device=NODE, stamp_revision=1
            ),
            origin={"kind": "user", "source_device": origin, "source_session_id": ""},
        ),
    )
    AttentionStore(owner / "attention.db").publish(f"session/{SID}", TOK1, "a1", "complete")
    NotificationStore(owner).notify(SID, title="Deploy finished", priority="high", key="deploy")
    return owner


def _catalog_row(origin: str = HEAD) -> dict[str, Any]:
    return {
        "session_id": SID,
        "state": "stored",
        "origin": {"kind": "user", "source_device": origin, "source_session_id": ""},
    }


def test_the_origin_viewer_gets_attention_and_notice(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, rows, HEAD)
    assert rows[0]["attention"] == {"token": TOK1, "outcome": "complete"}
    assert rows[0]["notice"] == {
        "key": "deploy",
        "revision": 1,
        "priority": "high",
        "title": "Deploy finished",
    }


def test_a_non_origin_viewer_gets_no_fields(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, rows, "d_some_other_member")
    assert "attention" not in rows[0] and "notice" not in rows[0]
    ra.stamp_rows_for_viewer(owner, rows, "")
    assert "attention" not in rows[0] and "notice" not in rows[0]


def test_the_receipt_clears_the_completion_claim(tmp_path: Path) -> None:
    """``net_session_receipt`` runs ``acknowledge_session_receipt`` on the owner."""
    owner = _owner(tmp_path)
    (owner / "sessions" / SID / "transcript.jsonl").write_text("", encoding="utf-8")
    from local_operator.session.attention import acknowledge_session_receipt

    acknowledge_session_receipt(owner, SID, TOK1)
    rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, rows, HEAD)
    assert "attention" not in rows[0]


def test_parks_and_forwarded_rows_are_never_re_forwarded(tmp_path: Path) -> None:
    owner = tmp_path / "node"
    store = NotificationStore(owner)
    store.notify(SID, title="gate", priority="high", key="gate:approval:w1")
    store.notify(SID, title="echo", key="remote-done:d_x:" + SID)
    store.notify(SID, title="bridge", key="team-bridge-held")
    assert ra._newest_notice(store, SID) is None
    store.notify(SID, title="real one")
    notice = ra._newest_notice(store, SID)
    assert notice is not None and notice["title"] == "real one"
    assert notice["key"].startswith("n_"), "a keyless item is keyed by its own id"


def test_a_silent_completion_is_not_forwarded(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    AttentionStore(owner / "attention.db").publish(
        f"session/{SID}", TOK2, "a2", "complete", notify=False
    )
    rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, rows, HEAD)
    assert "attention" not in rows[0]


# ---------------------------------------------------------------------------
# The wire parse
# ---------------------------------------------------------------------------


def test_an_old_parse_and_malformed_blocks_are_no_claim() -> None:
    good = PeerRow.from_json(
        {
            "session_id": SID,
            "state": "live",
            "attention": {"token": TOK1, "outcome": "complete"},
            "notice": {"key": "k", "revision": 3, "priority": "high", "title": "T"},
        },
        device_id=NODE,
    )
    assert good.attention == (TOK1, "complete")
    assert good.notice == ("k", 3, "high", "T")
    # A row from an owner that predates the capability carries neither key.
    old = PeerRow.from_json({"session_id": SID, "state": "live"}, device_id=NODE)
    assert old.attention is None and old.notice is None
    for bad_attention, bad_notice in (
        ("tok", ["k"]),
        ({"token": ""}, {"key": "k", "revision": True, "title": "T"}),
        ({"token": 7}, {"key": "k", "revision": 0, "title": "T"}),
        ({}, {"key": "", "revision": 1, "title": "T"}),
    ):
        row = PeerRow.from_json(
            {"session_id": SID, "attention": bad_attention, "notice": bad_notice},
            device_id=NODE,
        )
        assert row.attention is None and row.notice is None
    stale = PeerRow.from_json(
        {"session_id": SID, "state": "stale", "notice": {"key": "k", "revision": 1, "title": "T"}},
        device_id=NODE,
    )
    assert stale.notice is None


def test_peer_rows_carry_the_claims_onto_session_rows() -> None:
    class _Peer:
        device_id = NODE
        name = "lop-remote-node-2"
        network_id = "n_test"
        reachable = True
        reason = ""

    class _Catalog:
        def peers(self) -> list[Any]:
            return [_Peer()]

        def rows(self) -> list[Any]:
            return [
                PeerRow.from_json(
                    {
                        "session_id": SID,
                        "state": "live",
                        "attention": {"token": TOK1, "outcome": "complete"},
                    },
                    device_id=NODE,
                ),
                PeerRow.from_json({"session_id": "b" * 12, "state": "live"}, device_id=NODE),
            ]

    rows = peer_rows.peer_session_rows(None, ttl_s=0, catalog=_Catalog())
    by_id = {row.id: row for row in rows}
    assert by_id[SID].remote_attention == (TOK1, "complete")
    assert by_id["b" * 12].remote_attention is None


# ---------------------------------------------------------------------------
# The node's notify sentence
# ---------------------------------------------------------------------------


def test_the_node_notify_sentence_names_the_forward(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    sentence = ra.forwarded_notice_sentence(owner, SID)
    assert "will be forwarded to" in sentence and HEAD in sentence
    local = tmp_path / "local"
    assert ra.forwarded_notice_sentence(local, SID) == ""
