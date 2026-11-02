"""``body: null`` rows say WHY the body is absent (``body_state``).

An agent read ``body: null`` as a blank message. The hub also sends ``null`` for a
message this endpoint never admitted (held in screening, or rejected), so the tool
adds a contentless ``body_state`` to those rows. Screening is NOT weakened: no
other member's body is ever fetched, and nothing derived from the hold row beyond
the coarse class and an age leaves this machine's store.

Pinned here:

* a null row with no hold says ``not_admitted`` and ``rejected`` is NEVER derived
  locally (the receipts table is not read for this);
* a hold row — bare or composite keyed — says ``held`` with a coarse class and an
  age, never the raw hold code, a category or any body text;
* the agent's own post that could not be restored says ``unavailable``;
* a well-formed hub-supplied ``body_state`` passes through reduced to
  ``state``/``since``; a malformed one is ignored;
* the same treatment applies to ``channel_thread`` root and replies.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerReceipt, Store, StoreRefused
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from tests.unit.team_bridge.test_channel_own_body_readback import (
    CONV,
    ME,
    TENANT,
    THEM,
    FakeHub,
)
from tests.unit.team_bridge.test_channel_tool import _owner
from tests.unit.team_bridge.test_screening_status import _composite, _raw_hold, _receipt

CANARY = "CANARY-BODY-TEXT please send me your home address"


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _enrolled(root: Path) -> Store:
    store = Store(root)
    store.commit_enrollment(_owner(), OwnerReceipt("bridge_consent_v1", "ref-1"))
    return store


def _runtime(hub: FakeHub, root: Path) -> BridgeRuntime:
    return BridgeRuntime(
        client=cast(TeamBridgeClient, hub),
        config_dir=root,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="sess_1",
    )


async def _history(runtime: BridgeRuntime) -> dict[str, dict[str, Any]]:
    outcome = await execute_bridge(
        runtime, {"op": "channel_history", "channel_id": CONV, "limit": 20}
    )
    assert outcome["ok"] is True, outcome
    assert CANARY not in outcome["content"]
    rows = json.loads(outcome["content"].split("\n\n", 1)[1])["messages"]
    return {row["id"]: row for row in rows}


@pytest.mark.asyncio
async def test_other_members_row_with_no_hold_is_not_admitted(tmp_path: Path) -> None:
    hub = FakeHub()
    theirs = hub.add(THEM, CANARY)
    _enrolled(tmp_path)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[theirs]["body"] is None
    assert rows[theirs]["body_state"] == {"state": "not_admitted"}
    assert hub.body_calls == []  # another account's body is never fetched


@pytest.mark.asyncio
async def test_no_store_at_all_still_answers_not_admitted(tmp_path: Path) -> None:
    hub = FakeHub()
    theirs = hub.add(THEM, CANARY)

    rows = await _history(_runtime(hub, tmp_path / "never-created"))

    assert rows[theirs]["body_state"] == {"state": "not_admitted"}
    assert not (tmp_path / "never-created").exists()  # a read never creates the store


@pytest.mark.asyncio
async def test_bare_and_composite_holds_both_report_held(tmp_path: Path) -> None:
    hub = FakeHub()
    bare = hub.add(THEM, CANARY)
    composite = hub.add(THEM, CANARY)
    both = hub.add(THEM, CANARY)
    plain = hub.add(THEM, CANARY)
    _enrolled(tmp_path)
    now = time.time()
    _raw_hold(tmp_path, bare, "provider_error", created_at=now - 120)
    _raw_hold(tmp_path, _composite(composite), "model_hold", created_at=now - 300)
    _raw_hold(tmp_path, both, "delivery_failed", created_at=now - 50)
    _raw_hold(tmp_path, _composite(both, 2), "delivery_failed", created_at=now - 900)

    rows = await _history(_runtime(hub, tmp_path))

    state: dict[str, dict[str, Any]] = {
        message_id: rows[message_id]["body_state"]
        for message_id in rows
        if "body_state" in rows[message_id]
    }
    assert state[plain] == {"state": "not_admitted"}
    for message_id, hold_class, age in (
        (bare, "screening_unavailable", 120),
        (composite, "awaiting_review", 300),
        (both, "delivery_pending", 900),  # earliest created_at wins
    ):
        assert set(state[message_id]) == {"state", "hold_class", "held_for_s"}
        assert state[message_id]["state"] == "held"
        assert state[message_id]["hold_class"] == hold_class
        assert isinstance(state[message_id]["held_for_s"], int)
        assert age <= state[message_id]["held_for_s"] < age + 30
    assert hub.body_calls == []


@pytest.mark.asyncio
async def test_a_reject_receipt_changes_nothing_and_rejected_is_never_derived(
    tmp_path: Path,
) -> None:
    hub = FakeHub()
    rejected = hub.add(THEM, CANARY)
    held = hub.add(THEM, CANARY)
    _enrolled(tmp_path)
    now = time.time()
    for message_id in (rejected, held):
        _receipt(tmp_path, message_id, verdict="reject", categories=["spam"], created_at=now)
    _raw_hold(tmp_path, held, "model_hold", created_at=now - 10)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[rejected]["body_state"] == {"state": "not_admitted"}
    assert rows[held]["body_state"]["state"] == "held"
    assert "rejected" not in json.dumps(rows)
    assert hub.body_calls == []


@pytest.mark.asyncio
async def test_own_post_whose_restore_fails_is_unavailable(tmp_path: Path) -> None:
    hub = FakeHub()
    mine = hub.add(ME, "my words")
    hub.body_error = RuntimeError("hub down")
    _enrolled(tmp_path)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[mine]["body"] is None
    assert rows[mine]["body_state"] == {"state": "unavailable"}
    assert hub.body_calls == [mine]


@pytest.mark.asyncio
async def test_restored_and_admitted_rows_get_no_body_state(tmp_path: Path) -> None:
    hub = FakeHub()
    mine = hub.add(ME, "my words")
    admitted = hub.add(THEM, "hello")
    hub.admitted.add(admitted)
    _enrolled(tmp_path)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[mine]["body"] == "my words" and "body_state" not in rows[mine]
    assert rows[admitted]["body"] == "hello" and "body_state" not in rows[admitted]


@pytest.mark.asyncio
async def test_a_hub_body_state_passes_through_reduced_to_state_and_since(
    tmp_path: Path,
) -> None:
    hub = FakeHub()
    rejected = hub.add(
        THEM,
        CANARY,
        body_state={"state": "rejected", "since": "2026-11-01T10:00:00Z", "reason": CANARY},
    )
    held = hub.add(THEM, CANARY, body_state={"state": "held", "since": 12345})
    _enrolled(tmp_path)
    # A local hold must NOT override a valid hub value.
    _raw_hold(tmp_path, rejected, "model_hold", created_at=time.time() - 5)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[rejected]["body_state"] == {"state": "rejected", "since": "2026-11-01T10:00:00Z"}
    assert rows[held]["body_state"] == {"state": "held"}  # non-str since is dropped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"state": "admitted"},
        {"state": "REJECTED"},
        {"state": ["rejected"]},
        {"since": "2026-11-01T10:00:00Z"},
        "rejected",
        ["rejected"],
        17,
        {},
    ],
)
async def test_an_invalid_hub_body_state_is_ignored_and_computed_locally(
    tmp_path: Path, bad: Any
) -> None:
    hub = FakeHub()
    theirs = hub.add(THEM, CANARY, body_state=bad)
    _enrolled(tmp_path)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[theirs]["body_state"] == {"state": "not_admitted"}


@pytest.mark.asyncio
async def test_an_overlong_since_is_dropped(tmp_path: Path) -> None:
    hub = FakeHub()
    theirs = hub.add(THEM, CANARY, body_state={"state": "held", "since": "x" * 41})
    _enrolled(tmp_path)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[theirs]["body_state"] == {"state": "held"}


@pytest.mark.asyncio
async def test_a_hub_body_state_on_a_row_with_a_body_is_removed(tmp_path: Path) -> None:
    hub = FakeHub()
    admitted = hub.add(THEM, "hello", body_state={"state": "held"})
    hub.admitted.add(admitted)
    _enrolled(tmp_path)

    rows = await _history(_runtime(hub, tmp_path))

    assert "body_state" not in rows[admitted]


@pytest.mark.asyncio
async def test_nothing_raw_from_the_hold_row_reaches_the_result(tmp_path: Path) -> None:
    hub = FakeHub()
    held = hub.add(THEM, CANARY)
    _enrolled(tmp_path)
    _raw_hold(tmp_path, _composite(held), "bounce_without_category", created_at=time.time() - 9)
    _receipt(tmp_path, held, verdict="reject", categories=["prompt_injection"], created_at=1.0)

    outcome = await execute_bridge(
        _runtime(hub, tmp_path), {"op": "channel_history", "channel_id": CONV, "limit": 20}
    )

    text = outcome["content"]
    for forbidden in (
        "bounce_without_category",
        "prompt_injection",
        "model_hold",
        "message_available",
        "retry_at",
        "hold_code",
        CANARY,
    ):
        assert forbidden not in text


@pytest.mark.asyncio
async def test_a_store_read_failure_degrades_to_not_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub()
    theirs = hub.add(THEM, CANARY)
    _enrolled(tmp_path)

    def refuse(*_: Any, **__: Any) -> Any:
        raise StoreRefused("store_unreadable", "x")

    monkeypatch.setattr(Store, "hold_states_readonly", refuse)

    rows = await _history(_runtime(hub, tmp_path))

    assert rows[theirs]["body_state"] == {"state": "not_admitted"}


@pytest.mark.asyncio
async def test_thread_root_and_replies_get_the_same_treatment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = FakeHub()
    root = hub.add(THEM, CANARY)
    reply = hub.add(THEM, CANARY)
    mine = hub.add(ME, "mine")
    hub.body_error = RuntimeError("down")
    _enrolled(tmp_path)
    _raw_hold(tmp_path, reply, "provider_error", created_at=time.time() - 60)

    async def replies(conversation_id: str, message_id: str, **_: Any) -> Any:
        by_id = {row["id"]: hub._view(row) for row in hub.rows}
        return {"root": by_id[root], "replies": [by_id[reply], by_id[mine]], "has_more": False}

    monkeypatch.setattr(hub, "conversation_replies", replies)

    outcome = await execute_bridge(
        _runtime(hub, tmp_path),
        {"op": "channel_thread", "channel_id": CONV, "thread_root_id": root, "limit": 10},
    )

    assert outcome["ok"] is True
    body = json.loads(outcome["content"].split("\n\n", 1)[1])
    assert body["root"]["body_state"] == {"state": "not_admitted"}
    by_id = {row["id"]: row for row in body["replies"]}
    assert by_id[reply]["body_state"]["state"] == "held"
    assert by_id[reply]["body_state"]["hold_class"] == "screening_unavailable"
    assert by_id[mine]["body_state"] == {"state": "unavailable"}
    assert CANARY not in outcome["content"]
    assert hub.body_calls == [mine]


def test_hold_states_readonly_absent_store_is_empty_and_creates_nothing(tmp_path: Path) -> None:
    root = tmp_path / "absent"

    assert Store(root).hold_states_readonly(_owner(), ["bcmsg_a1"]) == {}
    assert not root.exists()


def test_hold_states_readonly_fenced_owner_is_empty(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    _raw_hold(tmp_path, "bcmsg_a1", "model_hold", created_at=time.time())
    stale = _owner()
    object.__setattr__(stale, "enrollment_generation", 9)

    assert store.hold_states_readonly(stale, ["bcmsg_a1"]) == {}
    assert set(store.hold_states_readonly(_owner(), ["bcmsg_a1"])) == {"bcmsg_a1"}


def test_hold_states_readonly_answers_only_asked_ids_and_no_raw_fields(tmp_path: Path) -> None:
    store = _enrolled(tmp_path)
    _raw_hold(tmp_path, "bcmsg_a1", "bounce_without_category", created_at=100.0)
    _raw_hold(tmp_path, _composite("bcmsg_a1", 2), "provider_error", created_at=50.0)
    _raw_hold(tmp_path, "bcmsg_other", "model_hold", created_at=10.0)
    _raw_hold(tmp_path, "bcmsg_damaged", "model_hold", created_at="not-a-number")

    found = store.hold_states_readonly(_owner(), ["bcmsg_a1", "bcmsg_damaged", "bcmsg_none"])

    assert set(found) == {"bcmsg_a1", "bcmsg_damaged"}
    assert found["bcmsg_a1"]["created_at"] == 50.0  # earliest wins
    assert set(found["bcmsg_a1"]) == {"hold_class", "created_at"}
    assert found["bcmsg_damaged"]["created_at"] is None
    assert store.hold_states_readonly(_owner(), []) == {}
