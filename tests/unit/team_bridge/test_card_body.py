"""Display-only Team Bridge bodies: the single writer, the resolver, the excerpt gate.

Drives the REAL stores (sqlite on disk) and the real transcript reader; only the
hub wire is a double, and that double FAILS if the screener's classification
fetch (``conversation_body`` / ``/messages/{id}/body``) is ever reached — the
display path must never show unscreened bytes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
from local_operator.harness.types import CustomMessage
from local_operator.team_bridge import card_projection
from local_operator.team_bridge.card_body import resolve_card_body
from local_operator.team_bridge.card_projection import (
    EXCERPT_MAX_BYTES,
    EXCERPT_MAX_CHARS,
    EXCERPTS_PER_CARD,
    PENDING_CARD,
    attach_history_excerpts,
    bounded_excerpt,
    model_card_text,
)
from local_operator.team_bridge.client import BridgeRefusal, BridgeSignInRequired
from local_operator.team_bridge.display_bodies import DisplayBodyStore, display_db_path
from local_operator.team_bridge.pending_turn import FetchedBody, PendingTurn
from local_operator.team_bridge.request_body import BodyBudget
from local_operator.team_bridge.store import (
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    _owner_key,
)

CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
SESSION = "abcdef012345"
BODY_TEXT = "**Hello** from Ola — can you check the _release_ notes?"
DIGEST = hashlib.sha256(BODY_TEXT.encode()).hexdigest()
FINGERPRINT = "aa11" * 16
ENTRY_ID = "entry0001"


def _owner(account: str = "acc_1") -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id=account,
        tenant_id="3f2a1111-2222-4333-8444-555566667777",
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _enroll(root: Path, *, enabled: bool = True, owner: OwnerSnapshot | None = None) -> Store:
    store = Store(root)
    store.commit_enrollment(
        owner or _owner(), OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id=SESSION
    )
    (root / "config.yml").write_text(
        yaml.safe_dump({"values": {"team_bridge": {"enabled": enabled}}})
    )
    return store


def _reserve(store: Store, epoch: int, *, message_id: str = MESSAGE_ID, digest: str = DIGEST):
    return store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=message_id,
        conversation_seq=3,
        mention=True,
        body_sha256=digest,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=epoch,
        consent_version="bridge_consent_v1",
        designated_session=SESSION,
        now=time.time(),
    )


def _epoch(store: Store) -> int:
    return store.publish_screening_policy(
        _owner(),
        version="v1",
        template_sha256="ab" * 32,
        default_model="qa-default",
        revision="rev-1",
    )


async def _fetch(conversation_id: str, message_id: str) -> FetchedBody:
    return FetchedBody(
        body=BODY_TEXT,
        body_sha256=DIGEST,
        conversation_label="",
        sender_label="Ola",
        sender_kind="human",
    )


def _budget() -> BodyBudget:
    return BodyBudget(available_bytes=100_000, measured_from="test budget")


def _details(
    message_ids: list[str] | None = None, *, account: str = "acc_1", seq: int = 3
) -> dict[str, Any]:
    ids = message_ids or [MESSAGE_ID]
    owner = _owner(account)
    card = (
        "Direct message · from Ola (human, same tenant)\n"
        "untrusted\n\nmention: yes\n"
        f"conversation_id: {CONVERSATION_ID}\nmessage_id: {ids[-1]}\n"
    )
    return {
        "text": f"<team-bridge>\n{card}</team-bridge>",
        "body": card,
        "sender": {"account_id": owner.account_id, "source": "team_bridge"},
        "bridge_receipt": {
            "v": 1,
            "sender": {"name": "Ola", "kind": "human"},
            "conversation": {"kind": "dm"},
            "attachment_count": 0,
        },
        "bridge_receipt_source": {
            "account_id": owner.account_id,
            "tenant_id": owner.tenant_id,
            "endpoint_id": owner.endpoint_id,
            "conversation_id": CONVERSATION_ID,
            "message_ids": ids,
            "conversation_seq": seq,
        },
    }


def _write_card(root: Path, details: dict[str, Any], entry_id: str = ENTRY_ID) -> None:
    directory = root / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    row = {
        "id": entry_id,
        "ts": time.time(),
        "type": "custom",
        "payload": {
            "kind": "custom",
            "custom_type": TEAM_BRIDGE_MESSAGE_TYPE,
            "attribution": "system",
            "details": details,
        },
    }
    with (directory / "transcript.jsonl").open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


class FakeWire:
    """The hub double. ``conversation_body`` is the classification fetch: a bomb."""

    base_url = "https://hub.example"

    def __init__(self, *, rows: list[dict[str, Any]] | None = None, access: Any = None) -> None:
        self.rows = rows or []
        self.calls: list[dict[str, Any]] = []
        self.access = access or SimpleNamespace(account_id="acc_1", kind="oauth", credential_id=7)

    async def current_access(self) -> Any:
        if isinstance(self.access, BaseException):
            raise self.access
        return self.access

    async def conversation_messages(self, conversation_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"conversation_id": conversation_id, **kwargs})
        if isinstance(self.rows, BaseException):
            raise self.rows
        return {"messages": self.rows, "has_more": False}

    async def conversation_body(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("display must never call the classification fetch /messages/{id}/body")


def _hub_row(message_id: str = MESSAGE_ID, *, body: str | None = BODY_TEXT, **extra: Any):
    row: dict[str, Any] = {
        "id": message_id,
        "conversation_id": CONVERSATION_ID,
        "seq": 3,
        "body": body,
        "body_hash": hashlib.sha256(body.encode()).hexdigest() if body is not None else "x" * 64,
        "sender": {"kind": "human"},
    }
    row.update(extra)
    return row


# -- the single writer -------------------------------------------------------


def _complete_turn(tmp_path: Path) -> tuple[Store, PendingTurn, str]:
    store = _enroll(tmp_path)
    assert _reserve(store, _epoch(store))
    turn = PendingTurn(_owner(), SESSION, config_dir=tmp_path)
    assert asyncio.run(turn.open(budget=_budget(), fetch=_fetch)) is True
    return store, turn, _owner_key(_owner())


def test_a_completed_settlement_records_the_admitted_body_privately(tmp_path: Path) -> None:
    _store, turn, key = _complete_turn(tmp_path)
    assert turn.for_call() is not None
    assert turn.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False) == "completed"

    rows = DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID])
    assert rows[MESSAGE_ID]["body"] == BODY_TEXT
    assert rows[MESSAGE_ID]["source"] == "turn"
    assert rows[MESSAGE_ID]["sender_kind"] == "human"
    mode = stat.S_IMODE(os.stat(display_db_path(tmp_path)).st_mode)
    assert mode == 0o600


@pytest.mark.parametrize(
    "settle",
    [
        # never appended: a success verdict alone must not write
        {"append": False, "error": None, "stop_reason": "stop", "aborted": False},
        {"append": True, "error": "boom", "stop_reason": "error", "aborted": False},
        {"append": True, "error": None, "stop_reason": "stop", "aborted": True},
    ],
)
def test_no_other_settlement_records_a_body(tmp_path: Path, settle: dict[str, Any]) -> None:
    _store, turn, key = _complete_turn(tmp_path)
    if settle.pop("append"):
        assert turn.for_call() is not None
    assert turn.settle(entry_id="e1", **settle) != "completed"
    assert DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID]) == {}
    assert not display_db_path(tmp_path).exists()


def test_the_store_refuses_bytes_that_do_not_match_their_digest(tmp_path: Path) -> None:
    display = DisplayBodyStore(tmp_path)
    assert not display.record(
        "k",
        message_id=MESSAGE_ID,
        conversation_id=CONVERSATION_ID,
        body="tampered",
        body_sha256=DIGEST,
        sender_kind="human",
        source="turn",
    )
    assert display.bodies("k", [MESSAGE_ID]) == {}


def test_ttl_and_total_size_bounds_evict_oldest_first(tmp_path: Path, monkeypatch) -> None:
    from local_operator.team_bridge import display_bodies

    display = DisplayBodyStore(tmp_path)

    def put(mid: str, body: str, at: float) -> None:
        assert display.record(
            "k",
            message_id=mid,
            conversation_id=CONVERSATION_ID,
            body=body,
            body_sha256=hashlib.sha256(body.encode()).hexdigest(),
            sender_kind="agent",
            source="hub",
            now=at,
        )

    now = time.time()
    put("bcmsg_old", "x", now - display_bodies.TTL_SECONDS - 10)
    put("bcmsg_new", "y", now)
    assert set(display.bodies("k", ["bcmsg_old", "bcmsg_new"])) == {"bcmsg_new"}

    monkeypatch.setattr(display_bodies, "MAX_TOTAL_BYTES", 10)
    put("bcmsg_a", "a" * 6, now + 1)
    put("bcmsg_b", "b" * 6, now + 2)
    assert set(display.bodies("k", ["bcmsg_new", "bcmsg_a", "bcmsg_b"])) == {"bcmsg_b"}


def test_an_enrollment_under_another_account_purges_display_bodies(tmp_path: Path) -> None:
    _store, turn, key = _complete_turn(tmp_path)
    assert turn.for_call() is not None
    turn.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False)
    assert DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID])
    # Same owner re-commit (a heal / generation move) keeps them.
    import dataclasses

    Store(tmp_path).commit_enrollment(
        dataclasses.replace(_owner(), enrollment_generation=2),
        OwnerReceipt("bridge_consent_v1", "ref-1"),
        cos_session_id=SESSION,
    )
    assert DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID])

    # A real account switch: the old owner's rows are gone.
    store = Store(tmp_path)
    import sqlite3

    from local_operator.team_bridge.store import state_db_path

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("DELETE FROM enrollments")
    store.commit_enrollment(_owner("acc_2"), OwnerReceipt("bridge_consent_v1", "ref-2"))
    assert DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID]) == {}


# -- the resolver ------------------------------------------------------------


def _resolve(root: Path, wire: FakeWire, entry_id: str = ENTRY_ID) -> dict[str, Any]:
    return asyncio.run(resolve_card_body(root, SESSION, entry_id, wire=wire))


def test_a_local_body_is_served_without_any_hub_read(tmp_path: Path) -> None:
    _store, turn, _key = _complete_turn(tmp_path)
    assert turn.for_call() is not None
    turn.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False)
    _write_card(tmp_path, _details())
    wire = FakeWire()

    answer = _resolve(tmp_path, wire)

    assert answer == {
        "v": 1,
        "items": [
            {
                "message_id": MESSAGE_ID,
                "conversation_id": CONVERSATION_ID,
                "state": "admitted",
                "body": BODY_TEXT,
                "source": "local",
            }
        ],
        "more": 0,
    }
    assert wire.calls == []


def test_backfill_uses_the_admitted_history_read_and_caches_the_verified_body(
    tmp_path: Path,
) -> None:
    _enroll(tmp_path)
    _write_card(tmp_path, _details())
    wire = FakeWire(rows=[_hub_row()])

    answer = _resolve(tmp_path, wire)

    assert answer["items"][0]["state"] == "admitted"
    assert answer["items"][0]["source"] == "hub"
    (call,) = wire.calls
    assert call["endpoint_id"] == "ep_1" and call["before_seq"] == 4 and call["limit"] == 1
    # Cached for display only — never as a turn-sourced (excerpt-eligible) row.
    row = DisplayBodyStore(tmp_path).bodies(_owner_key(_owner()), [MESSAGE_ID])[MESSAGE_ID]
    assert row["source"] == "hub"
    # Second view: local, no second hub read.
    assert _resolve(tmp_path, wire)["items"][0]["source"] == "local"
    assert len(wire.calls) == 1


def test_a_hub_body_that_does_not_match_its_digest_is_never_shown(tmp_path: Path) -> None:
    _enroll(tmp_path)
    _write_card(tmp_path, _details())
    row = _hub_row()
    row["body"] = "swapped"
    answer = _resolve(tmp_path, FakeWire(rows=[row]))
    assert answer["items"][0]["state"] == "unavailable"
    assert "body" not in answer["items"][0]


@pytest.mark.parametrize(
    ("body_state", "expected"),
    [
        ({"state": "rejected", "reason": "secret category"}, "refused"),
        ({"state": "held", "hold_code": "model_hold"}, "held"),
        ({"state": "not_admitted"}, "not_admitted"),
        (None, "not_admitted"),
    ],
)
def test_hub_null_bodies_map_to_closed_states_without_detail(
    tmp_path: Path, body_state: Any, expected: str
) -> None:
    _enroll(tmp_path)
    _write_card(tmp_path, _details())
    row = _hub_row(body=None, body_state=body_state)
    (item,) = _resolve(tmp_path, FakeWire(rows=[row]))["items"]
    assert item["state"] == expected
    assert set(item) == {"message_id", "conversation_id", "state"}


def test_a_local_hold_reads_held_with_class_and_age_only(tmp_path: Path, monkeypatch) -> None:
    _enroll(tmp_path)
    _write_card(tmp_path, _details())
    monkeypatch.setattr(
        Store,
        "hold_states_readonly",
        lambda self, owner, ids: {
            MESSAGE_ID: {"hold_class": "awaiting_review", "created_at": time.time() - 30}
        },
    )
    wire = FakeWire()
    (item,) = _resolve(tmp_path, wire)["items"]
    assert item["state"] == "held"
    assert item["hold"]["class"] == "awaiting_review" and item["hold"]["held_for_s"] >= 30
    assert wire.calls == []


def test_a_non_terminal_reservation_reads_pending(tmp_path: Path) -> None:
    store = _enroll(tmp_path)
    assert _reserve(store, _epoch(store))
    _write_card(tmp_path, _details())
    wire = FakeWire()
    (item,) = _resolve(tmp_path, wire)["items"]
    assert item["state"] == "pending"
    assert wire.calls == []


def test_a_card_from_another_account_answers_other_account_and_reads_nothing(
    tmp_path: Path,
) -> None:
    _enroll(tmp_path)
    _write_card(tmp_path, _details(account="acc_other"))
    wire = FakeWire(rows=[_hub_row()])
    (item,) = _resolve(tmp_path, wire)["items"]
    assert item["state"] == "unavailable" and item["reason"] == "other_account"
    assert wire.calls == []


def test_signing_in_as_another_account_purges_and_refuses(tmp_path: Path) -> None:
    _store, turn, key = _complete_turn(tmp_path)
    assert turn.for_call() is not None
    turn.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False)
    _write_card(tmp_path, _details())
    wire = FakeWire(access=SimpleNamespace(account_id="acc_2", kind="oauth", credential_id=9))
    (item,) = _resolve(tmp_path, wire)["items"]
    assert item["reason"] == "other_account"
    assert DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID]) == {}


def test_signing_out_purges_local_bodies(tmp_path: Path) -> None:
    _store, turn, key = _complete_turn(tmp_path)
    assert turn.for_call() is not None
    turn.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False)
    _write_card(tmp_path, _details())
    (item,) = _resolve(tmp_path, FakeWire(access=BridgeSignInRequired("sign in")))["items"]
    assert item == {
        "message_id": MESSAGE_ID,
        "conversation_id": CONVERSATION_ID,
        "state": "unavailable",
        "reason": "signed_out",
    }
    assert DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID]) == {}


def test_an_unreachable_hub_still_serves_local_bodies(tmp_path: Path) -> None:
    _store, turn, _key = _complete_turn(tmp_path)
    assert turn.for_call() is not None
    turn.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False)
    _write_card(tmp_path, _details([MESSAGE_ID, "bcmsg_abcdef02"]))
    down = BridgeRefusal("bridge_unreachable", "down")
    items = _resolve(tmp_path, FakeWire(access=down))["items"]
    assert [item["state"] for item in items] == ["admitted", "unavailable"]
    assert items[1]["reason"] == "offline"


@pytest.mark.parametrize(("status", "reason"), [(403, "not_member"), (None, "offline")])
def test_hub_refusals_map_to_closed_reasons(tmp_path: Path, status, reason) -> None:
    _enroll(tmp_path)
    _write_card(tmp_path, _details())
    wire = FakeWire()
    refusal = BridgeRefusal("forbidden", "server text never shown", http_status=status)
    wire.rows = refusal  # type: ignore[assignment]
    (item,) = _resolve(tmp_path, wire)["items"]
    assert item["reason"] == reason


def test_turning_the_bridge_off_purges_and_backfills_without_caching(tmp_path: Path) -> None:
    _store, turn, key = _complete_turn(tmp_path)
    assert turn.for_call() is not None
    turn.settle(entry_id="e1", error=None, stop_reason="stop", aborted=False)
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"values": {"team_bridge": {"enabled": False}}})
    )
    _write_card(tmp_path, _details())
    wire = FakeWire(rows=[_hub_row()])
    (item,) = _resolve(tmp_path, wire)["items"]
    assert item["source"] == "hub"
    assert DisplayBodyStore(tmp_path).bodies(key, [MESSAGE_ID]) == {}


def test_a_coalesced_card_resolves_twenty_and_counts_the_rest(tmp_path: Path) -> None:
    _enroll(tmp_path)
    ids = [f"bcmsg_{index:08d}" for index in range(25)]
    _write_card(tmp_path, _details(ids, seq=40))
    wire = FakeWire(rows=[_hub_row(mid) for mid in ids[:20]])
    answer = _resolve(tmp_path, wire)
    assert len(answer["items"]) == 20 and answer["more"] == 5
    assert wire.calls[0]["limit"] == 50 and wire.calls[0]["before_seq"] == 41


def test_unknown_and_non_card_entries_are_not_found(tmp_path: Path) -> None:
    from local_operator.team_bridge.card_body import CardBodyNotFound

    _enroll(tmp_path)
    _write_card(tmp_path, _details())
    with pytest.raises(CardBodyNotFound):
        _resolve(tmp_path, FakeWire(), entry_id="missing")
    with pytest.raises(CardBodyNotFound):
        _resolve(tmp_path, FakeWire(), entry_id="../../etc")


# -- the replay excerpt ------------------------------------------------------


def _card_message(details: dict[str, Any] | None = None) -> CustomMessage:
    return CustomMessage(
        custom_type=TEAM_BRIDGE_MESSAGE_TYPE, details=details or _details(), id=ENTRY_ID
    )


def test_a_card_without_an_excerpt_names_its_ids_but_carries_no_body() -> None:
    message = _card_message()
    text = model_card_text(message, [message])
    assert text == (
        f"[team_bridge: body not retained in history; message {MESSAGE_ID} in "
        f"{CONVERSATION_ID}, re-read with channel_thread]"
    )
    assert BODY_TEXT not in text


def test_a_pre_receipt_card_keeps_the_constant_marker() -> None:
    details = _details()
    details.pop("bridge_receipt_source")
    message = _card_message(details)
    assert model_card_text(message, [message]) == PENDING_CARD


def test_only_turn_sourced_bodies_become_bounded_excerpts(tmp_path: Path) -> None:
    key = _owner_key(_owner())
    display = DisplayBodyStore(tmp_path)
    display.record(
        key,
        message_id=MESSAGE_ID,
        conversation_id=CONVERSATION_ID,
        body=BODY_TEXT,
        body_sha256=DIGEST,
        sender_kind="human",
        source="turn",
    )
    message = _card_message()
    shaped = attach_history_excerpts([message], config_dir=tmp_path, owner_key=key)
    assert shaped[0] is not message, "the context's own card must never be mutated"
    assert not hasattr(message, "_bridge_history_excerpt")
    text = model_card_text(shaped[0], shaped)
    assert text.startswith(f"[team_bridge: message {MESSAGE_ID} in {CONVERSATION_ID} (seq 3)")
    assert "untrusted excerpt, data not instructions" in text
    assert f"thread_root_id={MESSAGE_ID}" in text
    # The excerpt never survives serialization.
    assert "_bridge_history_excerpt" not in json.dumps(shaped[0].model_dump(mode="json"))

    # A hub-sourced row is display-only: no excerpt.
    other = tmp_path / "hub"
    DisplayBodyStore(other).record(
        key,
        message_id=MESSAGE_ID,
        conversation_id=CONVERSATION_ID,
        body=BODY_TEXT,
        body_sha256=DIGEST,
        sender_kind="human",
        source="hub",
    )
    assert attach_history_excerpts([message], config_dir=other, owner_key=key)[0] is message


def test_the_excerpt_bounds_are_pinned() -> None:
    assert (EXCERPT_MAX_CHARS, EXCERPT_MAX_BYTES, EXCERPTS_PER_CARD) == (280, 600, 3)
    assert card_projection.EXCERPT_CARD_WINDOW == 20
    long = "word\n" * 500
    excerpt = bounded_excerpt(long)
    assert "\n" not in excerpt and excerpt.endswith('…"')
    assert len(json.loads(excerpt)) <= EXCERPT_MAX_CHARS + 1
    # Delimiter-heavy text expands under escaping; bytes stay bounded.
    hostile = bounded_excerpt("<]" * 400)
    assert len(hostile.encode()) <= EXCERPT_MAX_BYTES
    assert "<" not in hostile and "]" not in hostile


def test_only_three_excerpts_per_card_and_twenty_cards(tmp_path: Path) -> None:
    key = _owner_key(_owner())
    display = DisplayBodyStore(tmp_path)
    ids = [f"bcmsg_{index:08d}" for index in range(5)]
    for mid in ids:
        display.record(
            key,
            message_id=mid,
            conversation_id=CONVERSATION_ID,
            body=f"body {mid}",
            body_sha256=hashlib.sha256(f"body {mid}".encode()).hexdigest(),
            sender_kind="agent",
            source="turn",
        )
    coalesced = _card_message(_details(ids))
    shaped = attach_history_excerpts([coalesced], config_dir=tmp_path, owner_key=key)
    text = model_card_text(shaped[0], shaped)
    assert text.count("untrusted excerpt") == 3
    assert f"messages {ids[3]},{ids[4]} in" in text

    cards = [_card_message(_details([ids[0]])) for _ in range(25)]
    shaped = attach_history_excerpts(cards, config_dir=tmp_path, owner_key=key)
    excerpted = [index for index, item in enumerate(shaped) if item is not cards[index]]
    assert excerpted == list(range(5, 25))


def test_revoked_standing_removes_excerpts_at_the_next_render(tmp_path: Path, monkeypatch) -> None:
    """The session hook re-proves standing per request (property C)."""

    from local_operator.session.session import Session
    from local_operator.team_bridge import authority

    key = _owner_key(_owner())
    DisplayBodyStore(tmp_path).record(
        key,
        message_id=MESSAGE_ID,
        conversation_id=CONVERSATION_ID,
        body=BODY_TEXT,
        body_sha256=DIGEST,
        sender_kind="human",
        source="turn",
    )
    host = SimpleNamespace(bridge_runtime=SimpleNamespace(config_dir=tmp_path, owner=_owner()))
    message = _card_message()

    monkeypatch.setattr(authority, "authority_fence", lambda runtime: None)
    live = Session._bridge_history_transform(host, [message])  # type: ignore[arg-type]
    assert "untrusted excerpt" in model_card_text(live[0], live)

    monkeypatch.setattr(authority, "authority_fence", lambda runtime: {"code": "consent_revoked"})
    revoked = Session._bridge_history_transform(host, [message])  # type: ignore[arg-type]
    assert revoked[0] is message
    assert BODY_TEXT not in model_card_text(revoked[0], revoked)


def test_the_phone_relay_route_answers_through_the_same_resolver(monkeypatch) -> None:
    """``/api/sessions/{id}/bridge-cards/{entry}/body``: gated, private, same answer."""

    from starlette.testclient import TestClient

    from local_operator.mobile.daemon import MobileDaemon, build_app
    from local_operator.paths import config_dir
    from local_operator.team_bridge import read_context

    root = config_dir()
    _enroll(root)
    _write_card(root, _details())
    DisplayBodyStore(root).record(
        _owner_key(_owner()),
        message_id=MESSAGE_ID,
        conversation_id=CONVERSATION_ID,
        body=BODY_TEXT,
        body_sha256=DIGEST,
        sender_kind="human",
        source="turn",
    )
    wire = FakeWire()
    monkeypatch.setattr(read_context, "wire_for", lambda _config: wire)
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    path = f"/api/sessions/{SESSION}/bridge-cards/{ENTRY_ID}/body"
    assert client.get(path).status_code in (302, 303, 401)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    (item,) = response.json()["items"]
    assert item == {
        "message_id": MESSAGE_ID,
        "conversation_id": CONVERSATION_ID,
        "state": "admitted",
        "body": BODY_TEXT,
        "source": "local",
    }
    assert client.get(f"/api/sessions/{SESSION}/bridge-cards/missing/body").status_code == 404
    assert client.get("/api/sessions/ffffffffffff/bridge-cards/x/body").status_code == 404
    assert wire.calls == []
