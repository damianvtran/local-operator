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


#: The real banner leg, kept before the autouse fixture below replaces it, so
#: the claim tests can drive it synchronously with a fake notifier.
_REAL_DONE_BANNER = ra._deliver_done_banner


@pytest.fixture(autouse=True)
def done_banners(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture done-banner hand-offs: no daemon thread, no real notifier."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(ra, "_deliver_done_banner", lambda root, **kw: calls.append(kw))
    return calls


def _row(
    *,
    attention: tuple[str, str] | None = None,
    notice: tuple[str, int, str, str] | None = None,
    notices: tuple[tuple[str, int, str, str], ...] | None = None,
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
        remote_notices=(notice,) if notice else notices,
    )


def _later() -> int:
    """A listing read that started after every row written so far (Q3)."""
    return now_ms() + 1


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
    watcher.observe([_row()], listed_at_ms=_later())
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
    assert rows[0]["notices"] == [
        {"key": "deploy", "revision": 1, "priority": "high", "title": "Deploy finished"}
    ]


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
    assert good.notices == (("k", 3, "high", "T"),)
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

    head = tmp_path / "head"
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
    ra.RemoteAttentionWatcher(root).observe([_row(attention=(TOK1, "complete"))])
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


def test_m1_a_new_completion_hands_one_banner_per_token(
    root: Path, done_banners: list[dict[str, Any]]
) -> None:
    watcher = ra.RemoteAttentionWatcher(root)
    watcher.observe([_row(attention=(TOK1, "error"))])
    watcher.observe([_row(attention=(TOK1, "error"))])
    ra.RemoteAttentionWatcher(root).observe([_row(attention=(TOK1, "error"))])
    [item] = _items(root)
    assert len(done_banners) == 1, done_banners
    call = done_banners[0]
    assert call["item_id"] == item["id"] and call["revision"] == item["revision"]
    assert call["session_id"] == SID and call["title"] == "Port the parser"
    from local_operator.tui import notify as notify_copy

    assert call["body"] == notify_copy.BODIES["error"]
    assert call["subtitle"] == notify_copy.CONTEXTS.get("error", "")
    watcher.observe([_row(attention=(TOK2, "complete"))])
    assert len(done_banners) == 2, "a NEW completion banners again"


def _banner_row(root: Path) -> dict[str, Any]:
    ra.raise_done_item(root, session_id=SID, device_id=NODE, token=TOK1, name="N", deliver=False)
    [item] = _items(root)
    return item


def test_m1_the_done_banner_is_claimed_once_across_hosts(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.notifications import delivery

    monkeypatch.setattr(delivery, "gates_refuse", lambda: "")
    item = _banner_row(root)
    sent: list[tuple[str, str, dict[str, Any]]] = []

    def notifier(title: str, body: str, **kw: Any) -> bool:
        sent.append((title, body, kw))
        return True

    kwargs = dict(
        item_id=item["id"],
        revision=item["revision"],
        session_id=SID,
        title="N",
        body="Done",
    )
    # The TUI's watcher and serve's both reach the same revision.
    _REAL_DONE_BANNER(root, **kwargs, notify=notifier, background=False)
    _REAL_DONE_BANNER(root, **kwargs, notify=notifier, background=False)
    assert len(sent) == 1
    assert sent[0][2]["session_id"] == SID, "clickable: the click opens the peer session"
    row = NotificationStore(root).get(item["id"])
    assert row is not None and row["delivered_by"] == ra.DONE_BANNER_CLAIMANT


def test_m1_a_notifier_that_started_nothing_gives_the_claim_back(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.notifications import delivery

    item = _banner_row(root)
    kwargs = dict(
        item_id=item["id"],
        revision=item["revision"],
        session_id=SID,
        title="N",
        body="Done",
    )
    # Gated (rig guard / switch off): refused BEFORE the claim.
    monkeypatch.setattr(delivery, "gates_refuse", lambda: "notifications disabled")
    _REAL_DONE_BANNER(root, **kwargs, notify=lambda *a, **k: True, background=False)
    stored = NotificationStore(root).get(item["id"])
    assert stored is not None
    assert stored["delivered_rev"] == 0
    monkeypatch.setattr(delivery, "gates_refuse", lambda: "")
    _REAL_DONE_BANNER(root, **kwargs, notify=lambda *a, **k: False, background=False)
    stored = NotificationStore(root).get(item["id"])
    assert stored is not None
    assert stored["delivered_rev"] == 0
    # A read row is never bannered (the store's claim refuses it).
    NotificationStore(root).apply("read", ids=[(item["id"], item["revision"])])
    sent: list[str] = []
    _REAL_DONE_BANNER(
        root,
        **kwargs,
        notify=lambda t, b, **k: bool(sent.append(t)) or True,
        background=False,
    )
    assert sent == []


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


def test_r2_m1_two_completions_five_minutes_apart_banner_twice(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-M1 / Q4: per completion token, no cooldown — the local leg's rule.

    The REAL banner leg and the real store claim run (synchronously); only the
    OS notifier is a recorder, because a test may not raise a real banner.
    """
    from local_operator.notifications import delivery

    monkeypatch.setattr(delivery, "gates_refuse", lambda: "")
    sent: list[dict[str, Any]] = []

    def notifier(title: str, body: str, **kw: Any) -> bool:
        sent.append({"title": title, **kw})
        return True

    monkeypatch.setattr(
        ra,
        "_deliver_done_banner",
        lambda root, **kw: _REAL_DONE_BANNER(root, **kw, notify=notifier, background=False),
    )
    clock = _Clock(monkeypatch)
    tui, serve = ra.RemoteAttentionWatcher(root), ra.RemoteAttentionWatcher(root)
    tui.observe([_row(attention=(TOK1, "complete"))])
    serve.observe([_row(attention=(TOK1, "complete"))])
    assert len(sent) == 1, "one banner per token across both hosts"
    clock.ms += 5 * 60 * 1000
    tui.observe([_row(attention=(TOK2, "complete"))])
    serve.observe([_row(attention=(TOK2, "complete"))])
    assert len(sent) == 2, "a second completion 5 min later banners again"
    [row] = _items(root)
    assert row["delivered_rev"] == row["revision"] == 2
    assert row["delivered_by"] == ra.DONE_BANNER_CLAIMANT


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
        ra.RemoteAttentionWatcher(root).observe([_row(attention=(TOK1, "complete"))])
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
    watcher.observe([_row(attention=(TOK1, "complete"))])
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
                    ("k-new", 1, "normal", "Deploy done (newer)"),
                    ("k-old", 1, "normal", "Tests failed (older)"),
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
