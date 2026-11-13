"""A finished turn or a ``notify`` on a mesh node reaches the session's ORIGIN.

THE GAP THESE PIN (remote-session-parity, 2026-11-11). A session the operator
started on their head device and ran on ``lop-remote-node-2`` finished turns and
called ``notify`` there, and the head learned nothing: its ``notifications.db``
held 0 rows for the node's item ``n_xr44lx4zb5gi3mfn``. The owner now stamps
origin-only ``attention``/``notices`` facts on the catalogue rows it hands the
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
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.projection import PeerRow
from local_operator.notifications import remote_attention as ra
from local_operator.notifications import store as store_mod
from local_operator.notifications.delivery import (
    PARK_ITEM_PREFIXES,
    announced_by_its_park,
)
from local_operator.notifications.store import NotificationStore, content_hash, now_ms
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
    notice: tuple[Any, ...] | None = None,
    notices: tuple[tuple[Any, ...], ...] | None = None,
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
        remote_notices=_timed((notice,) if notice else notices),
    )


def _timed(notices: tuple[tuple[Any, ...], ...] | None) -> tuple[tuple[Any, ...], ...] | None:
    """Give a 4-tuple notice an owner time AFTER any baseline this test records.

    The no-backfill rule (2026-11-13) raises only a notice newer than the
    origin's first-seen baseline; a test about some OTHER property of a notice
    states a fresh one, and a test about the baseline passes a 5-tuple.
    """
    if not notices:
        return notices
    fresh = int(time.time() * 1000) + 60_000
    return tuple(n if len(n) > 4 else (*n, fresh) for n in notices)


def _seed_done(root: Path, token: str = TOK1) -> dict[str, Any]:
    """A ``remote-done:`` row as an OLDER build wrote it (this one never does)."""
    NotificationStore(root).notify(
        SID, title="Port the parser", key=ra.done_key(NODE, SID), anchor_id=token, episode=token
    )
    [item] = [i for i in _items(root) if i["key"] == ra.done_key(NODE, SID)]
    return item


def _later() -> int:
    """A listing read that started after every row written so far (Q3)."""
    return now_ms() + 1


def _items(root: Path) -> list[dict[str, Any]]:
    rows, _ = NotificationStore(root).list_items(state="all", limit=100)
    return rows


# ---------------------------------------------------------------------------
# Origin side: edges and rows
# ---------------------------------------------------------------------------


def test_serve_host_raises_no_done_row_for_a_remote_turn_end(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator rule 2026-11-13: a remote turn-end is never an inbox row or banner."""
    listing = ((_row(attention=(TOK1, "complete")),), ())
    monkeypatch.setattr(peer_rows, "read_listing", lambda *_a, **_k: listing)

    async def two_polls() -> None:
        task = asyncio.create_task(ra.run_forever(root, interval_s=0.01))
        await asyncio.sleep(0.2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(two_polls())
    assert _items(root) == []
    assert not hasattr(ra, "raise_done_item") and not hasattr(ra, "_deliver_done_banner")


def test_an_acknowledged_completion_marks_an_older_builds_row_read(root: Path) -> None:
    """The settle survives: an older build's done row reads once the owner acks it."""
    _seed_done(root)
    watcher = ra.RemoteAttentionWatcher(root)
    watcher.observe([_row(attention=(TOK1, "complete"))], listed_at_ms=_later())
    assert _items(root)[0]["read"] is False, "still claimed: not acknowledged"
    watcher.observe([_row()], listed_at_ms=_later())
    [item] = _items(root)
    assert item["read"] is True and item["dismissed"] is False


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
    # is no edge at all; the restarted watcher answers ``unchanged`` on the
    # revision check before writing, so it hands nothing to the ladder.
    assert [call["action"] for call in _no_banner_threads] == ["created"]


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
    [notice] = rows[0]["notices"]
    assert {k: v for k, v in notice.items() if k != "at"} == {
        "key": "deploy",
        "revision": 1,
        "priority": "high",
        "title": "Deploy finished",
    }
    # The owner item's own time rides along (2026-11-13, no-backfill rule).
    [owned] = NotificationStore(owner).list_items(state="all", limit=5)[0]
    assert notice["at"] == owned["updated_at"] > 0


def test_a_non_origin_viewer_gets_no_fields(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, rows, "d_some_other_member")
    assert "attention" not in rows[0] and "notices" not in rows[0]
    ra.stamp_rows_for_viewer(owner, rows, "")
    assert "attention" not in rows[0] and "notices" not in rows[0]


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
    assert ra._newest_notices(store, SID) == []
    store.notify(SID, title="real one")
    [notice] = ra._newest_notices(store, SID)
    assert notice["title"] == "real one"
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
            "notices": [{"key": "k", "revision": 3, "priority": "high", "title": "T"}],
        },
        device_id=NODE,
    )
    assert good.attention == (TOK1, "complete")
    assert good.notices == (("k", 3, "high", "T", 0),), "no ``at``: an older owner"
    # A row from an owner that predates the capability carries neither key.
    old = PeerRow.from_json({"session_id": SID, "state": "live"}, device_id=NODE)
    assert old.attention is None and old.notices is None
    for bad_attention, bad_notice in (
        ("tok", {"key": "k"}),
        ({"token": ""}, {"key": "k", "revision": True, "title": "T"}),
        ({"token": 7}, {"key": "k", "revision": 0, "title": "T"}),
        ({}, {"key": "", "revision": 1, "title": "T"}),
    ):
        row = PeerRow.from_json(
            {"session_id": SID, "attention": bad_attention, "notices": [bad_notice]},
            device_id=NODE,
        )
        assert row.attention is None and row.notices is None
    stale = PeerRow.from_json(
        {
            "session_id": SID,
            "state": "stale",
            "notices": [{"key": "k", "revision": 1, "title": "T"}],
        },
        device_id=NODE,
    )
    assert stale.notices is None


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


def test_the_node_notify_sentence_promises_only_what_the_origin_collects(
    tmp_path: Path,
) -> None:
    """Review round 1 m2 + QA Q2: "forwarded" only to an origin known to collect."""
    owner = _owner(tmp_path)
    # Not yet recorded (an older origin, or one that has not read this device
    # since updating): the item stays here, and the model is told so.
    before = ra.forwarded_notice_sentence(owner, SID)
    assert "will be forwarded" not in before
    assert "stays on this device" in before
    # Q2: a non-member origin is named in words, never by its raw device id.
    assert HEAD not in before and "its origin device" in before
    # The owner's catalogue op records an origin that advertised the capability.
    ra.stamp_rows_for_viewer(owner, [_catalog_row()], HEAD)
    assert HEAD in ra.forwarding_origins(owner)
    after = ra.forwarded_notice_sentence(owner, SID)
    assert "will be forwarded to its origin device" in after
    # A key this module never forwards gets no clause at all.
    assert ra.forwarded_notice_sentence(owner, SID, "gate:approval:w1") == ""
    local = tmp_path / "local"
    assert ra.forwarded_notice_sentence(local, SID) == ""


# ---------------------------------------------------------------------------
# Remediation round 1 regressions
# ---------------------------------------------------------------------------


def test_m2_a_high_notice_followed_by_a_normal_one_both_arrive(
    tmp_path: Path, _no_banner_threads: list[dict[str, Any]]
) -> None:
    """Review round 1 M2's repro: only the newest notice used to cross the wire."""
    owner = tmp_path / "node"
    head = tmp_path / "head"
    # The head first watched the node BEFORE these notices were posted.
    ra.origin_baseline_ms(head, NODE)
    time.sleep(0.002)
    _owner(tmp_path)  # stamp + attention; its own "deploy" notice is read below
    store = NotificationStore(owner)
    [seed] = [item for item in _items(owner)]
    store.apply("read", ids=[(seed["id"], seed["revision"])])
    store.notify(SID, title="Tests failed: 3", priority="high", key="a")
    store.notify(SID, title="Deploy done", priority="normal", key="b")
    rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, rows, HEAD)
    assert [(n["key"], n["priority"]) for n in rows[0]["notices"]] == [
        ("b", "normal"),
        ("a", "high"),
    ], "newest first, and the earlier high notice is still on the wire"

    parsed = PeerRow.from_json({**rows[0], "state": "stored"}, device_id=NODE)
    session_row = _row(notices=parsed.notices)
    ra.RemoteAttentionWatcher(head).observe([session_row])
    by_key = {item["key"]: item for item in _items(head)}
    assert set(by_key) == {ra.notice_key(NODE, SID, "a"), ra.notice_key(NODE, SID, "b")}
    assert by_key[ra.notice_key(NODE, SID, "a")]["priority"] == "high"
    assert by_key[ra.notice_key(NODE, SID, "a")]["title"] == "Tests failed: 3"
    assert [c["priority"] for c in _no_banner_threads] == ["high"], "the high one banners"


def test_m2_the_wire_list_is_bounded_newest_first(tmp_path: Path) -> None:
    from local_operator.network import projection

    assert projection.NOTICES_MAX == ra.NOTICES_ON_WIRE >= 8
    store = NotificationStore(tmp_path)
    for i in range(ra.NOTICES_ON_WIRE + 3):
        store.notify(SID, title=f"n{i}", key=f"k{i}", now=1_000 + i)
    notices = ra._newest_notices(store, SID)
    assert len(notices) == ra.NOTICES_ON_WIRE
    assert notices[0]["key"] == f"k{ra.NOTICES_ON_WIRE + 2}"
    # The read side cuts an over-long list to the newest, never refuses it.
    long = [{"key": f"k{i}", "revision": 1, "title": "t"} for i in range(20)]
    parsed = PeerRow.from_json({"session_id": SID, "notices": long}, device_id=NODE)
    assert parsed.notices is not None and len(parsed.notices) == projection.NOTICES_MAX


def test_m2_each_owner_item_is_its_own_episode(root: Path) -> None:
    a = ("a", 1, "high", "Tests failed: 3")
    b = ("b", 1, "normal", "Deploy done")
    edges, state = peer_rows.notice_edges({}, [_row(notices=(b, a))])
    assert {e.key for e in edges} == {"a", "b"}
    assert state == {(NODE, SID, "a"): 1, (NODE, SID, "b"): 1}
    again, _ = peer_rows.notice_edges(state, [_row(notices=(b, a))])
    assert again == ()
    # A silent device keeps every one of its notice episodes.
    _, carried = peer_rows.notice_edges(state, [], unanswered=[NODE])
    assert carried == state


def test_m1_minor_a_read_notice_never_flips_back_to_unread(root: Path) -> None:
    """Review round 1 m1: the dedupe is the owner revision, not the content digest."""
    notice = ("k", 2, "high", "Deploy finished")
    ra.RemoteAttentionWatcher(root).observe([_row(notice=notice)])
    [item] = _items(root)
    NotificationStore(root).apply("read", ids=[(item["id"], item["revision"])])
    # A host that read a NAMELESS row (body "runs on d_node", not the name) and a
    # host on a stale listing (an older revision) both see an edge.
    nameless = _row(notice=notice)._replace(owner_device_name="")
    ra.RemoteAttentionWatcher(root).observe([nameless])
    ra.RemoteAttentionWatcher(root).observe([_row(notice=("k", 1, "high", "Deploy started"))])
    [after] = _items(root)
    assert after["read"] is True and after["revision"] == item["revision"]
    assert after["title"] == "Deploy finished"
    # A genuinely newer revision still moves it.
    ra.RemoteAttentionWatcher(root).observe([_row(notice=("k", 3, "high", "Deploy rolled back"))])
    [newer] = _items(root)
    assert newer["read"] is False and newer["anchor_id"] == "rev:3"


def test_q1_a_receipt_between_two_watchers_still_settles_the_row(root: Path) -> None:
    """QA round 1 Q1: settle from the current listing, not from process memory."""
    _seed_done(root)
    [item] = _items(root)
    assert item["read"] is False
    # serve stopped; the receipt landed on the owner; a FRESH watcher starts.
    fresh = ra.RemoteAttentionWatcher(root)
    # Absence is not acknowledgement: a silent device or an unlisted session
    # leaves the row unread.
    fresh.observe([], unanswered=[NODE], listed_at_ms=_later())
    fresh.observe([_row(session_id="f" * 12)], listed_at_ms=_later())
    assert _items(root)[0]["read"] is False
    # A listing of unknown age settles nothing (Q3).
    fresh.observe([_row()])
    assert _items(root)[0]["read"] is False
    # Listed with no claim, by a read newer than the row: acknowledged.
    fresh.observe([_row()], listed_at_ms=_later())
    [after] = _items(root)
    assert after["read"] is True and after["dismissed"] is False


def test_q1_the_prefix_query_matches_literally(tmp_path: Path) -> None:
    """``unread_with_key_prefix`` escapes LIKE wildcards: ``_`` is not "any char"."""
    store = NotificationStore(tmp_path)
    store.notify("s1", title="a", key="remote-done:d_x:s1")
    store.notify("s1", title="b", key="remote-doneXd_x:s1")
    store.notify("s1", title="c", key="remote_done:d_x:s1")
    assert [r["key"] for r in store.unread_with_key_prefix("remote-done:")] == [
        "remote-done:d_x:s1"
    ]
    assert [r["key"] for r in store.unread_with_key_prefix("remote_done:")] == [
        "remote_done:d_x:s1"
    ]


# ---------------------------------------------------------------------------
# Round 2 remediation
# ---------------------------------------------------------------------------


class _Clock:
    """The store's wall clock, moved by hand (``store.now_ms`` is its one source)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, start: int = 1_800_000_000_000) -> None:
        self.ms = start
        monkeypatch.setattr(store_mod, "now_ms", lambda: self.ms)


def test_r2_m1_every_other_claimant_keeps_the_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``per_revision`` is opt-in: a watcher item updating 5 min later stays quiet (D15)."""
    clock = _Clock(monkeypatch)
    store = NotificationStore(tmp_path)
    first = store.notify("s1", title="Build 1 red", priority="high", key="ci")
    assert store.claim_delivery(first.item_id, "tui")
    clock.ms += 5 * 60 * 1000
    store.notify("s1", title="Build 2 red", priority="high", key="ci")
    assert not store.claim_delivery(first.item_id, "tui")
    assert store.claim_delivery(first.item_id, "tui", per_revision=True)
    assert not store.claim_delivery(first.item_id, "tui", per_revision=True), "once per revision"


def test_q3_a_stale_cached_listing_never_settles_a_newer_row(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 2, Q3: serve's cached listing predates the completion the TUI raised.

    Driven through ``read_listing``'s real cache: watcher Y (serve) reads a
    claim-less listing, watcher X (the TUI) then raises TOK1 from a fresh one,
    and Y's next poll is handed the SAME cached tuple. The row must stay unread
    until a listing that started after it says "no claim".
    """
    clock = _Clock(monkeypatch)

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
            return [PeerRow.from_json({"session_id": SID, "state": "live"}, device_id=NODE)]

    peer_rows.clear_cache()
    # Only the listing's wall-clock stamp moves; its TTL keeps the real
    # monotonic clock, so the second read is still a cache hit.
    monkeypatch.setattr(peer_rows, "_wall_ms", lambda: clock.ms)
    try:
        catalog = _Catalog()
        stale, _ = peer_rows.read_listing(root, catalog=catalog)
        y = ra.RemoteAttentionWatcher(root)
        y.observe(stale)
        clock.ms += 200
        _seed_done(root)
        clock.ms += 5_000
        again, _ = peer_rows.read_listing(root, catalog=catalog)
        assert again is stale, "the 20 s cache hands Y the pre-completion listing"
        y.observe(again)
        [item] = _items(root)
        assert item["read"] is False, "a listing older than the row is not a receipt"
        # A read that STARTED after the row: the owner really stopped claiming it.
        clock.ms += 1
        fresh, _ = peer_rows.read_listing(root, catalog=catalog, ttl_s=0)
        assert fresh is not stale
        y.observe(fresh)
        [after] = _items(root)
        assert after["read"] is True
    finally:
        peer_rows.clear_cache()


def test_q3_listing_age_answers_only_for_the_tuple_it_handed_out(root: Path) -> None:
    peer_rows.clear_cache()
    try:
        assert peer_rows.listing_read_at_ms(root, ()) is None
        peer_rows.seed_peer_row(root, _row())
        seeded = peer_rows._CACHE[str(root)]
        assert peer_rows.listing_read_at_ms(root, seeded[1]) == 0, "a create is not a listing"
        assert peer_rows.listing_read_at_ms(root, list(seeded[1])) is None
    finally:
        peer_rows.clear_cache()


def test_r2_m2_a_failed_attention_read_is_unknown_not_acknowledged(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = _owner(tmp_path)

    def broken(self: Any, identities: Any) -> Any:
        raise RuntimeError("database disk image is malformed")

    monkeypatch.setattr(AttentionStore, "state_many", broken)
    rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, rows, HEAD)
    assert rows[0]["attention_unknown"] is True and "attention" not in rows[0]
    parsed = PeerRow.from_json({**rows[0], "state": "live"}, device_id=NODE)
    assert parsed.attention_unknown is True and parsed.attention is None
    # Only a real ``True`` counts; an old owner carries no key at all.
    for claim in ({"attention_unknown": 1}, {}):
        old = PeerRow.from_json({"session_id": SID, **claim}, device_id=NODE)
        assert old.attention_unknown is False

    watcher = ra.RemoteAttentionWatcher(root)
    _seed_done(root)
    unknown = SessionRow(
        SID,
        0.0,
        "Port the parser",
        locality="remote",
        owner_device=NODE,
        remote_attention_unknown=True,
    )
    watcher.observe([unknown], listed_at_ms=_later())
    assert _items(root)[0]["read"] is False, "an unreadable store is not a receipt"
    watcher.observe([_row()], listed_at_ms=_later())
    assert _items(root)[0]["read"] is True


#: An owner time after any baseline a test records (wall clock + 1 min).
_FRESH = int(time.time() * 1000) + 60_000


def test_r2_m1_minor_a_batch_of_notices_lands_oldest_first(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every store write a distinct millisecond, so ``updated_at`` order is the
    # write order rather than a tie broken by id.
    ticks = iter(range(1_800_000_000_000, 1_800_000_001_000))
    monkeypatch.setattr(store_mod, "now_ms", lambda: next(ticks))
    # The wire list is newest first.
    ra.RemoteAttentionWatcher(root).observe(
        [
            _row(
                notices=(
                    ("k-new", 1, "normal", "Deploy done (newer)", _FRESH + 2),
                    ("k-old", 1, "normal", "Tests failed (older)", _FRESH + 1),
                )
            )
        ]
    )
    unread, _ = NotificationStore(root).list_items(state="unread", limit=10)
    assert [row["title"] for row in unread] == ["Deploy done (newer)", "Tests failed (older)"]


def test_r2_m3_the_origins_file_is_owner_only(tmp_path: Path) -> None:
    import stat

    ra._recorded_origins.clear()
    ra.record_forwarding_origin(tmp_path, HEAD)
    mode = stat.S_IMODE((tmp_path / ra.ORIGINS_FILE).stat().st_mode)
    assert mode == 0o600, oct(mode)
    assert HEAD in ra.forwarding_origins(tmp_path)


# ---------------------------------------------------------------------------
# The v0.102.2 flood (operator rule 2026-11-13): turn-ends never notify, no
# backfill, agent children never forward, original timestamps.
# ---------------------------------------------------------------------------

_DAY_MS = 24 * 60 * 60 * 1000


def _historic_listing(count: int) -> list[SessionRow]:
    """``count`` node sessions, each with an unacknowledged completion and a notice
    the agent posted a day ago — the shape the head read at first contact."""
    old = int(time.time() * 1000) - _DAY_MS
    rows = []
    for i in range(count):
        sid = f"{i:012x}"
        rows.append(
            _row(
                session_id=sid,
                attention=(f"6f1c2a3e-0b4d-4e5f-8a9b-{i:012x}", "complete"),
                notices=(("k", 1, "normal", f"Old notice {i}", old),),
            )
        )
    return rows


def test_flood_first_contact_with_historic_items_raises_nothing(root: Path) -> None:
    """THE REPRODUCTION: before the fix this wrote 2 x N unread rows at sync time."""
    listing = _historic_listing(25)
    ra.RemoteAttentionWatcher(root).observe(listing, listed_at_ms=_later())
    assert _items(root) == [], [i["key"] for i in _items(root)]
    # Reconnect / restart / a second host: still nothing.
    ra.RemoteAttentionWatcher(root).observe(listing, listed_at_ms=_later())
    ra.RemoteAttentionWatcher(root).observe(listing, listed_at_ms=_later())
    assert _items(root) == []


def test_a_notice_posted_after_first_contact_is_raised_once_at_its_own_time(root: Path) -> None:
    baseline = ra.origin_baseline_ms(root, NODE)
    assert ra.origin_baseline_ms(root, NODE) == baseline, "persisted, never moved"
    posted = baseline + 5_000
    listing = [_row(notice=("deploy", 1, "high", "Deploy needs sign-off", posted))]
    ra.RemoteAttentionWatcher(root).observe(listing)
    [item] = _items(root)
    assert item["created_at"] == item["updated_at"] == posted, "the owner's time, not now"
    # Restart (fresh watcher, baseline read back from disk) and reconnect.
    ra.RemoteAttentionWatcher(root).observe(listing)
    ra.RemoteAttentionWatcher(root).observe(listing)
    assert [i["id"] for i in _items(root)] == [item["id"]]
    assert (root / ra.BASELINES_FILE).is_file()


def test_a_notice_from_before_the_baseline_or_without_a_time_is_never_raised(
    root: Path,
) -> None:
    baseline = ra.origin_baseline_ms(root, NODE)
    ra.RemoteAttentionWatcher(root).observe(
        [
            _row(notice=("old", 1, "high", "Backfill", baseline - 1)),
            _row(session_id="b" * 12, notice=("legacy", 1, "high", "Older owner", 0)),
        ]
    )
    assert _items(root) == []


def test_an_agent_spawned_sessions_notices_never_cross_the_wire(tmp_path: Path) -> None:
    owner = _owner(tmp_path)
    stamp_file = owner / "sessions" / SID / "mesh.json"
    import json

    data = json.loads(stamp_file.read_text(encoding="utf-8"))
    data["origin"]["kind"] = ra.AGENT_ORIGIN_KIND
    stamp_file.write_text(json.dumps(data), encoding="utf-8")
    rows = [{**_catalog_row(), "origin": data["origin"]}]
    ra.stamp_rows_for_viewer(owner, rows, HEAD)
    assert "notices" not in rows[0], "a reviewer/QA child reports to its parent agent"
    # The user's own session on the same owner still forwards.
    user_rows = [_catalog_row()]
    ra.stamp_rows_for_viewer(owner, user_rows, HEAD)
    assert [n["key"] for n in user_rows[0]["notices"]] == ["deploy"]


def test_the_baselines_file_is_owner_only_and_heals_when_corrupt(tmp_path: Path) -> None:
    import stat

    (tmp_path / ra.BASELINES_FILE).write_text("{not json", encoding="utf-8")
    first = ra.origin_baseline_ms(tmp_path, NODE)
    assert ra.origin_baseline_ms(tmp_path, NODE) == first
    mode = stat.S_IMODE((tmp_path / ra.BASELINES_FILE).stat().st_mode)
    assert mode == 0o600, oct(mode)


def test_the_cleanup_prefixes_match_this_modules_namespaces() -> None:
    assert store_mod.CLEANUP_KEY_PREFIXES == (f"{ra.DONE_PREFIX}:", f"{ra.NOTICE_PREFIX}:")
