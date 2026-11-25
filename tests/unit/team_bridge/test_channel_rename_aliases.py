"""Channel RENAME aliases, client-side (operator need, 2026-11-25).

When a hub channel is renamed (``aws-ai-build-day`` -> ``pergamon``) agents keep
addressing it by the old name. The hub deliberately keeps no rename history, so
this machine remembers (tenant, id, name) sightings from conversation listings
and admitted inbound display names, and ``_resolve_channel_name`` consults them
ONLY after a live exact match failed, always re-checking the aliased id against
the live list. These tests pin that order and its refusals at the tool boundary
(only the wire is faked; the store is the real SQLite file under the cwd).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.team_bridge import tool as tb_tool
from local_operator.team_bridge.connector import conversation as tb_conversation
from local_operator.team_bridge.store import Store
from local_operator.team_bridge.tool import execute_bridge
from tests.unit.team_bridge.test_channel_names import (  # noqa: F401 - open_gates is a fixture
    FakeHub,
    _gated_runtime,
    open_gates,
)

OLD = "aws-ai-build-day"
NEW = "pergamon"
ROOM = "bcv_a0000000000000000000000000000001"
OTHER = "bcv_b0000000000000000000000000000002"
THIRD = "bcv_c0000000000000000000000000000003"


def _channel(cid: str, name: str, **extra: Any) -> dict[str, Any]:
    return {"id": cid, "kind": "channel", "name": name, "visibility": "tenant", **extra}


def _store() -> Store:
    # The gated runtimes use config_dir="." and open_gates chdirs to tmp_path.
    return Store(Path("."))


def _seen(tenant: str, cid: str, name: str, at: float) -> None:
    _store().record_channel_names(tenant, [(cid, name)], now=at)


async def _history(hub: FakeHub, name: str, tenant: str = "tnt_1") -> dict[str, Any]:
    runtime = _gated_runtime(hub)
    runtime.tenant_id = tenant
    return await execute_bridge(runtime, {"op": "channel_history", "channel_id": name})


pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("open_gates")]


async def test_an_old_name_resolves_through_the_alias_with_the_new_name() -> None:
    _seen("tnt_1", ROOM, OLD, 1_000.0)
    _seen("tnt_1", ROOM, NEW, 2_000.0)
    hub = FakeHub([_channel(ROOM, NEW)])

    outcome = await _history(hub, f"#{OLD}")

    assert outcome["ok"] is True, outcome
    assert outcome["resolved_channel_id"] == ROOM
    assert outcome["resolved_via_alias"] is True
    assert outcome["current_name"] == NEW
    assert outcome["channel_name"] == NEW
    assert outcome["renamed_at"] == "1970-01-01T00:33:20Z"
    assert f"use '{NEW}'" in outcome["alias_note"]
    reads = [c["conversation_id"] for c in hub.calls if c["method"] == "conversation_messages"]
    assert reads == [ROOM]


async def test_the_listing_itself_records_the_rename() -> None:
    """No manual seeding: two lookups across a rename teach the alias."""

    before = FakeHub([_channel(ROOM, OLD)])
    assert (await _history(before, OLD))["ok"] is True
    after = FakeHub([_channel(ROOM, NEW)])

    outcome = await _history(after, OLD)

    assert outcome["ok"] is True, outcome
    assert outcome["resolved_via_alias"] is True and outcome["current_name"] == NEW
    rows = _store().channel_alias_rows_readonly("tnt_1")
    assert {(r["conversation_id"], r["name_key"]) for r in rows} == {(ROOM, OLD), (ROOM, NEW)}
    # Names and ids only: no body, sender or label column exists.
    assert set(rows[0]) == {
        "conversation_id",
        "name_key",
        "display_name",
        "first_seen",
        "last_seen",
    }


async def test_a_live_exact_name_wins_over_an_alias() -> None:
    _seen("tnt_1", ROOM, OLD, 1.0)
    hub = FakeHub([_channel(ROOM, NEW), _channel(OTHER, OLD)])

    outcome = await _history(hub, OLD)

    assert outcome["resolved_channel_id"] == OTHER
    assert "resolved_via_alias" not in outcome
    assert outcome["channel_name"] == OLD


async def test_an_old_name_reused_by_a_different_live_channel_resolves_to_the_live_one() -> None:
    _seen("tnt_1", ROOM, OLD, 1.0)
    _seen("tnt_1", ROOM, NEW, 2.0)
    # Someone created a fresh room with the freed name.
    hub = FakeHub([_channel(ROOM, NEW), _channel(THIRD, OLD.upper())])

    outcome = await _history(hub, OLD)

    assert outcome["resolved_channel_id"] == THIRD
    assert "resolved_via_alias" not in outcome


async def test_an_alias_held_by_two_live_rooms_is_ambiguous() -> None:
    _seen("tnt_1", ROOM, OLD, 1.0)
    _seen("tnt_1", OTHER, OLD, 1.5)
    hub = FakeHub([_channel(ROOM, NEW), _channel(OTHER, "ops")])

    outcome = await _history(hub, OLD)

    assert outcome["ok"] is False and outcome["code"] == "channel_ambiguous"
    assert ROOM in outcome["error"] and OTHER in outcome["error"]
    assert all(c["method"] != "conversation_messages" for c in hub.calls)


@pytest.mark.parametrize(
    "rooms",
    [
        [],  # closed/archived: absent from the open list
        [{"id": ROOM, "kind": "channel", "name": NEW, "visibility": "private"}],  # now private
    ],
    ids=["closed", "invisible"],
)
async def test_an_aliased_room_no_longer_open_and_visible_is_not_found_with_the_rename(
    rooms: list[dict[str, Any]],
) -> None:
    _seen("tnt_1", ROOM, OLD, 1.0)
    _seen("tnt_1", ROOM, NEW, 2.0)
    hub = FakeHub(rooms)

    outcome = await _history(hub, OLD)

    assert outcome["ok"] is False and outcome["code"] == "channel_not_found"
    assert "was renamed or closed" in outcome["error"]
    # Never a silent redirect, and never the hidden room's id or new name.
    assert ROOM not in outcome["error"] and NEW not in outcome["error"]
    assert all(c["method"] != "conversation_messages" for c in hub.calls)


async def test_an_alias_from_another_tenant_is_never_used() -> None:
    _seen("tnt_other", ROOM, OLD, 1.0)
    _seen("tnt_other", ROOM, NEW, 2.0)
    hub = FakeHub([_channel(ROOM, NEW)])

    outcome = await _history(hub, OLD, tenant="tnt_1")

    assert outcome["code"] == "channel_not_found"
    assert "renamed" not in outcome["error"]


async def test_channel_post_and_history_results_carry_channel_name() -> None:
    hub = FakeHub([_channel(ROOM, NEW)])
    runtime = _gated_runtime(hub)

    posted = await execute_bridge(
        runtime, {"op": "channel_post", "channel_id": NEW, "body": "hi", "new_post": True}
    )
    by_id = await execute_bridge(runtime, {"op": "channel_history", "channel_id": ROOM})
    unknown = await execute_bridge(runtime, {"op": "channel_history", "channel_id": OTHER})

    assert posted["ok"] is True, posted
    assert posted["channel_name"] == NEW
    # An id-addressed read names the room from what this machine last saw.
    assert by_id["channel_name"] == NEW
    # Never seen: present, and honestly unknown rather than guessed.
    assert "channel_name" in unknown and unknown["channel_name"] is None


async def test_not_found_suggests_near_matches_and_keeps_the_truncation_caveat() -> None:
    _seen("tnt_1", ROOM, OLD, 1.0)
    hub = FakeHub([_channel(ROOM, NEW), _channel(OTHER, "qa-coordination")])

    typo = await _history(hub, "qa-cordination")
    via_old = await _history(hub, "aws-ai-buildday")

    assert typo["code"] == "channel_not_found"
    assert "Did you mean: 'qa-coordination'?" in typo["error"]
    # An old name is suggested as the room's CURRENT name (what resolves).
    assert f"Did you mean: '{NEW}'?" in via_old["error"]

    filler = [_channel(f"bcv_f{i:031x}", f"room-{i}") for i in range(30)]
    deep = FakeHub(filler, page_size=5)
    missed = await _history(deep, "room-99")
    assert "in the first 500 conversations" in missed["error"]


async def test_inbound_display_names_feed_only_channel_aliases() -> None:
    owner = SimpleNamespace(tenant_id="tnt_1")

    def meta(cid: str, kind: str | None, label: str) -> Any:
        return SimpleNamespace(
            conversation_id=cid, conversation_kind=kind, conversation_label=label
        )

    tb_conversation._remember_channel_name(Path("."), owner, meta(ROOM, "channel", NEW))
    tb_conversation._remember_channel_name(Path("."), owner, meta(OTHER, "dm", "Alice"))
    tb_conversation._remember_channel_name(Path("."), owner, meta(THIRD, None, "legacy"))

    rows = _store().channel_alias_rows_readonly("tnt_1")
    assert [(r["conversation_id"], r["display_name"]) for r in rows] == [(ROOM, NEW)]


async def test_an_unwritable_alias_store_never_fails_the_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self: Store, *a: Any, **k: Any) -> int:
        raise RuntimeError("busy")

    monkeypatch.setattr(Store, "record_channel_names", boom)
    outcome = await _history(FakeHub([_channel(ROOM, NEW)]), NEW)
    assert outcome["ok"] is True and outcome["resolved_channel_id"] == ROOM
    assert tb_tool._channel_alias_rows(_gated_runtime(FakeHub())) == []
