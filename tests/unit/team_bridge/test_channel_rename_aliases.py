"""Channel RENAME aliases, client-side (operator need, 2026-11-25).

When a hub channel is renamed (``aws-ai-build-day`` -> ``pergamon``) agents keep
addressing it by the old name. The hub deliberately keeps no rename history, so
this machine remembers (tenant, id, name) sightings from TENANT-VISIBLE
conversation listings only, and ``_resolve_channel_name`` consults them ONLY
after a live exact match failed on an untruncated listing, always re-checking
the aliased id against the live list. A post addressed by an old name is
refused (``channel_renamed``) rather than sent. These tests pin that order and
its refusals at the tool boundary (only the wire is faked; the store is the
real SQLite file under the cwd).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import tool as tb_tool
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
    # Q2: the note names the old room without the caller's ``#`` sigil.
    assert outcome["alias_note"].startswith(f"'{OLD}' was renamed to '{NEW}'")
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


async def test_an_unwritable_alias_store_never_fails_the_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self: Store, *a: Any, **k: Any) -> int:
        raise RuntimeError("busy")

    monkeypatch.setattr(Store, "record_channel_names", boom)
    outcome = await _history(FakeHub([_channel(ROOM, NEW)]), NEW)
    assert outcome["ok"] is True and outcome["resolved_channel_id"] == ROOM
    assert tb_tool._channel_alias_rows(_gated_runtime(FakeHub())) == []


def _sends(hub: FakeHub) -> list[Any]:
    return [c for c in hub.calls if c["method"] == "conversation_send"]


async def test_channel_post_by_an_old_name_is_refused_and_sends_nothing() -> None:
    """S-1: squat a name, rename away, then a post by that name must not land.

    Reproduces the security probe: one ordinary lookup records the attacker's
    room under the squatted name, the room is renamed, and a later
    ``channel_post channel_id=finance`` used to SEND to it through the alias.
    """

    evil = "bcv_e0000000000000000000000000000009"
    squat = FakeHub([_channel(evil, "finance")])
    assert (await _history(squat, "finance"))["resolved_channel_id"] == evil
    renamed = FakeHub([_channel(evil, "lunch")])
    runtime = _gated_runtime(renamed)

    outcome = await execute_bridge(
        runtime,
        {"op": "channel_post", "channel_id": "finance", "body": "Q4 numbers", "new_post": True},
    )

    assert outcome["ok"] is False and outcome["code"] == "channel_renamed", outcome
    assert outcome["conversation_id"] == evil
    assert outcome["current_name"] == "lunch"
    assert outcome["renamed_at"]
    assert "re-send" in outcome["error"] and f"channel_id={evil}" in outcome["error"]
    assert _sends(renamed) == []
    # A READ by the old name still resolves (it sends nothing anywhere).
    read = await _history(renamed, "finance")
    assert read["ok"] is True and read["resolved_via_alias"] is True


@pytest.mark.parametrize(
    "closed_at",
    [50.0, 0.5],
    ids=["closed-holder-is-newer", "closed-holder-is-older"],
)
async def test_an_alias_once_held_by_a_closed_room_never_redirects(closed_at: float) -> None:
    """R1-1: the room most recently called ``standup`` closed; never redirect.

    ROOM was ``standup`` then ``daily``; OTHER also held ``standup`` and is now
    closed. Whichever held it last, the caller cannot be told which room it
    meant, so the lookup refuses without naming ROOM or ``daily``.
    """

    _seen("tnt_1", ROOM, "standup", 1.0)
    _seen("tnt_1", ROOM, "daily", 2.0)
    _seen("tnt_1", OTHER, "standup", closed_at)
    hub = FakeHub([_channel(ROOM, "daily")])

    outcome = await _history(hub, "standup")

    assert outcome["ok"] is False and outcome["code"] == "channel_not_found", outcome
    assert "was renamed or closed" in outcome["error"]
    assert ROOM not in outcome["error"] and "daily" not in outcome["error"]
    assert all(c["method"] != "conversation_messages" for c in hub.calls)


async def test_a_truncated_listing_never_resolves_through_an_alias() -> None:
    """S-3: the live exact holder may sit past the page bound; do not guess."""

    _seen("tnt_1", ROOM, OLD, 1.0)
    _seen("tnt_1", ROOM, NEW, 2.0)
    filler = [_channel(f"bcv_f{i:031x}", f"room-{i}") for i in range(30)]
    # ROOM is on the first page; the room that now holds OLD is past the cut.
    hub = FakeHub([_channel(ROOM, NEW), *filler, _channel(THIRD, OLD)], page_size=5)

    outcome = await _history(hub, OLD)

    assert outcome["ok"] is False and outcome["code"] == "channel_not_found", outcome
    assert "in the first 500 conversations" in outcome["error"]
    # The aliased room may be live past the cut too: no "renamed or closed" claim.
    assert "renamed" not in outcome["error"]
    assert all(c["method"] != "conversation_messages" for c in hub.calls)


async def test_renamed_at_is_the_return_to_the_current_name() -> None:
    """R1-2: b -> a -> b stamps the SECOND sighting of b, after a was last seen."""

    _seen("tnt_1", ROOM, "b", 1.0)
    _seen("tnt_1", ROOM, "a", 100.0)
    _seen("tnt_1", ROOM, "b", 200.0)
    # A plain refresh of the current name keeps the stamp of the return.
    _seen("tnt_1", ROOM, "b", 300.0)
    hub = FakeHub([_channel(ROOM, "b")])

    outcome = await _history(hub, "a")

    assert outcome["ok"] is True and outcome["resolved_via_alias"] is True, outcome
    assert outcome["renamed_at"] == "1970-01-01T00:03:20Z"


async def test_an_old_alias_row_beyond_the_scan_bound_still_resolves() -> None:
    """R1-4: an old name is the OLDEST row by last_seen; a lookup must still see it."""

    _seen("tnt_1", ROOM, OLD, 1.0)
    _seen("tnt_1", ROOM, NEW, 2.0)
    filler = [(f"bcv_f{i:031x}", f"room-{i}") for i in range(Store.CHANNEL_ALIAS_SCAN_MAX + 50)]
    _store().record_channel_names("tnt_1", filler, now=10.0)
    hub = FakeHub([_channel(ROOM, NEW)])

    outcome = await _history(hub, OLD)

    assert outcome["ok"] is True and outcome["resolved_channel_id"] == ROOM, outcome
    assert outcome["resolved_via_alias"] is True
