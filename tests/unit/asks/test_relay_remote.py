"""Relaying a NODE session's forwarded card from the head (E2E 44ab2361d833).

``ask(relay='<node session>/<ask>')`` on the operator's Mac was refused with
"names no ask in the original session's log" although the card's
``remote-notice:<node>:<session>:remote-ask:<ask>`` row sat in the Mac's inbox:
the relay read only local ask logs. The fix re-surfaces THAT ONE ROW (no relay
row, no copy — ``AskQueue._relay_remote`` says why), so these cells pin:

* a forwarded, attested, live card is re-raised unread and the receipt says the
  answer goes to the node session, never here;
* the refusals: unknown id, a card the node already settled (retired row), a
  card the operator dismissed, a row from a device that is no active member;
* nothing is written to the relaying session's ask log, the card's words are
  unchanged, a relay of a still-unread card writes nothing (``pending``), and a
  relay after the operator read it re-raises again (the reason is in its cell);
* a card the operator READ is still retired once the node stops listing it, so
  its relay is refused as expired (R1-1), and an unreadable inbox is reported
  as such, not as an unknown id (R1-4);
* a LOCAL origin still takes the pointer path (the whole of ``test_relay.py``
  is the byte-for-byte proof; one cell here shows a local hit never consults
  the inbox).

The two-relay loopback (forward over the wire → relay → node answer → row
retires) is ``tests/unit/network/test_remote_attention_relay.py``.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.asks import render, store
from local_operator.notifications import remote_attention as ra
from local_operator.notifications.store import NotificationStore
from tests.unit.asks.test_relay import ORIGIN, _origin_ask, _pair, _relay

HEAD = "d_head"
NODE = "d_node2"
STRANGER = "d_stranger"
NODE_SESSION = "44ab2361d833"
ASK = "a-4452"
TITLE = "Allow the browser on example.com?"


@pytest.fixture(autouse=True)
def banners(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture banner hand-offs instead of a real grace thread."""
    calls: list[dict[str, Any]] = []

    def capture(root: Path, outcome: Any, **kwargs: Any) -> None:
        calls.append({"action": outcome.action, "revision": outcome.revision, **kwargs})

    monkeypatch.setattr(ra, "_deliver_async", capture)
    return calls


def _mesh(root: Path, *, members: tuple[str, ...] = (NODE,), removed: tuple[str, ...] = ()):
    from local_operator.network import store as net_store
    from local_operator.network.types import MemberRecord, NetworkRecord

    record = NetworkRecord(network_id="n_test", name="mesh", self_device_id=HEAD, trust="active")
    record.members = [MemberRecord(device_id=HEAD, name="mac")] + [
        MemberRecord(device_id=member, name=f"{member}-box") for member in members
    ]
    record.removed_ids.extend(removed)
    net_store.save(record, root)


def _forward(root: Path, *, device: str = NODE, ask: str = ASK) -> dict[str, Any]:
    """The row the watcher writes for a forwarded ask (its own writer)."""
    action = ra.raise_notice_item(
        root,
        session_id=NODE_SESSION,
        device_id=device,
        key=f"{ra.ASK_NOTICE_PREFIX}{ask}",
        revision=1,
        priority="high",
        title=TITLE,
        device_name=f"{device}-box",
        deliver=False,
    )
    assert action == "created"
    return _row(root, device=device, ask=ask)


def _row(root: Path, *, device: str = NODE, ask: str = ASK) -> dict[str, Any]:
    item = NotificationStore(root).get_by_key(
        NODE_SESSION, ra.notice_key(device, NODE_SESSION, f"{ra.ASK_NOTICE_PREFIX}{ask}")
    )
    assert item is not None
    return item


def _watch_log(root: Path) -> list[dict[str, Any]]:
    from tests.unit.asks.test_relay import WATCH

    return store.read_events(store.session_dir(root, WATCH))


def test_a_forwarded_card_is_resurfaced_and_the_receipt_says_where_the_answer_goes(
    tmp_path: Path, banners: list[dict[str, Any]]
) -> None:
    _origin, watch, _o, _w = _pair(tmp_path)
    _mesh(tmp_path)
    before = _forward(tmp_path)
    NotificationStore(tmp_path).apply("read", ids=[(before["id"], before["revision"])], via="test")
    assert _row(tmp_path)["read"]

    result = _relay(watch, ASK, origin=NODE_SESSION)

    assert result["ok"], result
    text = result["text"]
    assert f"card {ASK} from session {NODE_SESSION} on {NODE}-box" in text
    assert "ORIGINAL session" in text and "will NOT receive it" in text
    assert "cannot withdraw it" in text and "NOT CONSENT" in text
    assert result["details"]["status"] == "resurfaced"
    assert "ask_id" not in result["details"], "nothing was queued here"
    assert result["details"]["relay"] == {
        "session_id": NODE_SESSION,
        "ask_id": ASK,
        "device": f"{NODE}-box",
        "remote": True,
    }
    after = _row(tmp_path)
    # The ONE card, unread again, one new revision — and not reworded.
    assert after["id"] == before["id"] and not after["read"]
    assert after["revision"] == before["revision"] + 1
    for column in ("title", "body_md", "priority", "key", "anchor_id", "session_id"):
        assert after[column] == before[column], column
    assert [(c["action"], c["revision"]) for c in banners] == [("updated", after["revision"])]
    # Pointer-not-copy, taken one step further: no relay row in THIS log.
    assert _watch_log(tmp_path) == []
    assert watch.records() == []


def test_the_watchers_next_listing_does_not_flip_a_resurfaced_row(tmp_path: Path) -> None:
    """The owner anchor is kept, so the watcher reads the row as current."""
    _origin, watch, _o, _w = _pair(tmp_path)
    _mesh(tmp_path)
    item = _forward(tmp_path)
    # Read first, so the relay really writes a new revision (an unread row is
    # left untouched — R1-2).
    NotificationStore(tmp_path).apply("read", ids=[(item["id"], item["revision"])], via="test")
    assert _relay(watch, ASK, origin=NODE_SESSION)["details"]["status"] == "resurfaced"
    resurfaced = _row(tmp_path)
    assert resurfaced["revision"] == item["revision"] + 1
    again = ra.raise_notice_item(
        tmp_path,
        session_id=NODE_SESSION,
        device_id=NODE,
        key=f"{ra.ASK_NOTICE_PREFIX}{ASK}",
        revision=1,
        priority="high",
        title=TITLE,
        device_name=f"{NODE}-box",
        deliver=False,
    )
    assert again == "unchanged"
    assert _row(tmp_path)["revision"] == resurfaced["revision"]


def test_an_unknown_id_is_refused_with_the_existing_sentence(tmp_path: Path) -> None:
    _origin, watch, _o, _w = _pair(tmp_path)
    # No network at all: nothing could have been forwarded.
    assert _relay(watch, ASK, origin=NODE_SESSION) == {
        "ok": False,
        "error": render.RELAY_ORIGIN_UNKNOWN,
    }
    # A mesh and a forwarded card, but a different ask id.
    _mesh(tmp_path)
    _forward(tmp_path)
    assert _relay(watch, "a-0000", origin=NODE_SESSION) == {
        "ok": False,
        "error": render.RELAY_ORIGIN_UNKNOWN,
    }
    assert _watch_log(tmp_path) == []


def test_a_card_the_node_settled_is_refused_as_expired_or_answered(
    tmp_path: Path, banners: list[dict[str, Any]]
) -> None:
    _origin, watch, _o, _w = _pair(tmp_path)
    _mesh(tmp_path)
    item = _forward(tmp_path)
    # What ``settle_unlisted_ask_items`` writes when the node stops listing it.
    NotificationStore(tmp_path).apply(
        "dismiss", ids=[(item["id"], item["revision"])], via="remote-attention"
    )

    result = _relay(watch, ASK, origin=NODE_SESSION)

    assert result == {
        "ok": False,
        "error": render.RELAY_REMOTE_SETTLED.format(device=f"{NODE}-box"),
    }
    assert "expired or was already answered" in result["error"]
    assert _row(tmp_path)["revision"] == item["revision"] and banners == []


def test_a_card_the_operator_dismissed_is_never_re_raised(
    tmp_path: Path, banners: list[dict[str, Any]]
) -> None:
    _origin, watch, _o, _w = _pair(tmp_path)
    _mesh(tmp_path)
    item = _forward(tmp_path)
    NotificationStore(tmp_path).apply("dismiss", ids=[(item["id"], item["revision"])], via="tui")

    result = _relay(watch, ASK, origin=NODE_SESSION)

    assert result == {
        "ok": False,
        "error": render.RELAY_REMOTE_PUT_AWAY.format(device=f"{NODE}-box"),
    }
    row = _row(tmp_path)
    assert row["dismissed"] and row["revision"] == item["revision"] and banners == []


@pytest.mark.parametrize("why", ["not_a_member", "removed", "untrusted"])
def test_a_row_from_a_device_that_is_no_active_member_is_not_resolvable(
    tmp_path: Path, why: str, banners: list[dict[str, Any]]
) -> None:
    from local_operator.network import store as net_store

    _origin, watch, _o, _w = _pair(tmp_path)
    if why == "not_a_member":
        _mesh(tmp_path)
        item = _forward(tmp_path, device=STRANGER)
    elif why == "removed":
        _mesh(tmp_path, members=(NODE,), removed=(NODE,))
        item = _forward(tmp_path)
    else:
        _mesh(tmp_path)
        record = net_store.load("n_test", tmp_path)
        record.trust = "untrusted"
        net_store.save(record, tmp_path)
        item = _forward(tmp_path)

    assert _relay(watch, ASK, origin=NODE_SESSION) == {
        "ok": False,
        "error": render.RELAY_ORIGIN_UNKNOWN,
    }
    row = _row(tmp_path, device=item["key"].split(":")[1])
    assert row["revision"] == item["revision"] and banners == []


def test_relaying_a_card_already_unread_writes_nothing_and_says_it_is_pending(
    tmp_path: Path, banners: list[dict[str, Any]]
) -> None:
    """PR #4082 review round 1, R1-2: a repeat relay must not churn the row.

    A repeat relay is ALLOWED, unlike a local repeat (``RELAY_ALREADY_HELD``):
    no card is minted, and refusing would need state this session deliberately
    does not keep. But while the row is still unread there is nothing to raise —
    a write would only bump the revision, re-sort the feed and grow
    ``item_history`` on every call of a relay loop — so it is a no-op whose
    receipt still says the card is pending in the operator's inbox. Once the
    operator reads it, the next relay re-raises it again (one new revision, the
    store's per-row banner cooldown bounding the banners).
    """
    _origin, watch, _o, _w = _pair(tmp_path)
    _mesh(tmp_path)
    item = _forward(tmp_path)
    assert not item["read"]

    for _ in range(3):
        result = _relay(watch, ASK, origin=NODE_SESSION)
        assert result["ok"], result
        assert result["details"]["status"] == "pending"
        text = result["text"]
        assert f"Card {ASK} from session {NODE_SESSION} on {NODE}-box" in text
        assert "pending, unread, in the operator's inbox" in text
        assert "ORIGINAL session" in text and "will NOT receive it" in text
        assert "NOT CONSENT" in text
        assert "ask_id" not in result["details"]
    row = _row(tmp_path)
    assert row["revision"] == item["revision"] and row["seq"] == item["seq"], "no write"
    assert banners == []

    # Read → the next relay re-raises the same row once more.
    NotificationStore(tmp_path).apply("read", ids=[(row["id"], row["revision"])], via="test")
    again = _relay(watch, ASK, origin=NODE_SESSION)
    assert again["ok"] and again["details"]["status"] == "resurfaced"
    row = _row(tmp_path)
    assert row["id"] == item["id"] and row["revision"] == item["revision"] + 1
    assert not row["read"]
    assert len(NotificationStore(tmp_path).find_by_key(row["key"])) == 1
    assert [c["revision"] for c in banners] == [item["revision"] + 1]
    assert _watch_log(tmp_path) == []


def _listing(*, notices: tuple[tuple[Any, ...], ...] = ()) -> SimpleNamespace:
    """The node's session row as the head's watcher reads it (no claim = answered)."""
    return SimpleNamespace(
        owner_device=NODE, id=NODE_SESSION, remote_asks_unknown=False, remote_notices=notices
    )


def test_a_read_card_the_node_answered_is_retired_and_its_relay_refused(
    tmp_path: Path, banners: list[dict[str, Any]]
) -> None:
    """PR #4082 review round 1, R1-1 (the reviewer's repro, as a test).

    The settle used to retire UNREAD forwarded ask rows only, so a card the
    operator had already read stayed undismissed after the node answered it, and
    a relay re-raised the answered card as live (unread again, banner 1). An
    ask's absence from the node's listing is its answer whatever the row's read
    state, so the read row is retired and the relay lands in the expired refusal.
    """
    _origin, watch, _o, _w = _pair(tmp_path)
    _mesh(tmp_path)
    item = _forward(tmp_path)
    store_ = NotificationStore(tmp_path)
    store_.apply("read", ids=[(item["id"], item["revision"])], via="test")
    assert _row(tmp_path)["read"]

    # The node answered: its next (newer) listing no longer carries the ask.
    settled = ra.settle_unlisted_ask_items(
        tmp_path, [_listing()], listed_at_ms=int(time.time() * 1000) + 1_000
    )

    assert settled == 1
    row = _row(tmp_path)
    assert row["dismissed_at"] and row["dismissed_via"] == "remote-attention"
    result = _relay(watch, ASK, origin=NODE_SESSION)
    assert result == {
        "ok": False,
        "error": render.RELAY_REMOTE_SETTLED.format(device=f"{NODE}-box"),
    }
    assert "expired or was already answered" in result["error"]
    after = _row(tmp_path)
    assert after["revision"] == row["revision"] and after["read"], "not re-raised"
    assert banners == []


def test_the_read_row_settle_leaves_notices_and_operator_dismissals_alone(
    tmp_path: Path,
) -> None:
    """R1-1's neighbours: widening the settle to READ rows must not reach a
    forwarded NOTICE (never retired by absence — its owner's list just aged it
    out) nor overwrite a dismissal the operator made (it stays ``put_away``)."""
    _mesh(tmp_path)
    store_ = NotificationStore(tmp_path)
    notice = ra.raise_notice_item(
        tmp_path,
        session_id=NODE_SESSION,
        device_id=NODE,
        key="deploy",
        revision=1,
        priority="high",
        title="Deploy needs sign-off",
        device_name=f"{NODE}-box",
        deliver=False,
    )
    assert notice == "created"
    notice_row = store_.get_by_key(NODE_SESSION, ra.notice_key(NODE, NODE_SESSION, "deploy"))
    assert notice_row is not None
    store_.apply("read", ids=[(notice_row["id"], notice_row["revision"])], via="test")
    ask_row = _forward(tmp_path)
    store_.apply("dismiss", ids=[(ask_row["id"], ask_row["revision"])], via="tui")

    settled = ra.settle_unlisted_ask_items(
        tmp_path, [_listing()], listed_at_ms=int(time.time() * 1000) + 1_000
    )

    assert settled == 0
    kept = store_.get_by_key(NODE_SESSION, notice_row["key"])
    assert kept is not None and kept["read"] and not kept["dismissed_at"]
    assert kept["revision"] == notice_row["revision"]
    assert _row(tmp_path)["dismissed_via"] == "tui"


def test_an_unreadable_inbox_is_not_reported_as_an_unknown_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    banners: list[dict[str, Any]],
) -> None:
    """PR #4082 review round 1, R1-4: a store failure used to answer "names no
    ask in the original session's log", teaching the model its id was wrong.
    It now says the inbox could not be read and to retry; the failure is still
    logged."""
    import logging

    _origin, watch, _o, _w = _pair(tmp_path)
    _mesh(tmp_path)
    _forward(tmp_path)

    def busy(*_a: Any, **_k: Any) -> Any:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(NotificationStore, "get_by_key", busy)
    with caplog.at_level(logging.DEBUG, logger=ra.logger.name):
        result = _relay(watch, ASK, origin=NODE_SESSION)

    assert result == {"ok": False, "error": render.RELAY_REMOTE_UNREADABLE}
    assert result["error"] != render.RELAY_ORIGIN_UNKNOWN
    assert "could not be read" in result["error"] and "try the relay again" in result["error"]
    assert any("resurface failed" in r.getMessage() and r.exc_info for r in caplog.records)
    assert banners == [] and _watch_log(tmp_path) == []


def test_a_local_origin_never_consults_the_inbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local relays are the pointer path, untouched: the remote branch is a miss-only arm."""
    origin, watch, _o, _w = _pair(tmp_path)
    ask_id = _origin_ask(origin)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("a local relay consulted the inbox")

    monkeypatch.setattr(ra, "resurface_forwarded_ask", boom)
    result = _relay(watch, ask_id, origin=ORIGIN)
    assert result["ok"] and result["details"]["relay"]["origin_status"] == store.STATUS_OPEN
