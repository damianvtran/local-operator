"""Rooms by NAME, and no duplicate rooms (operator incident, 2026-11-06).

A task session that could not call ``channels`` (``not_designated``) and was
refused channel_post/channel_history for lacking a ``bcv_`` id created a second
``qa-coordination`` channel. These tests pin both fixes at the tool boundary:

* ``channel_create`` with a name an open, visible room of that kind already
  holds stores nothing and answers ``channel_exists`` with the first room's id —
  from the local precheck, and from the hub's 409 ``conversation_name_taken``
  when the precheck cannot see the room;
* a NON-designated session posts to and reads ``qa-coordination`` by name,
  through the REAL fences (only the wire is faked), while ``channels`` still
  refuses it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import tool as tb_tool
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from local_operator.team_bridge.host import (
    admit_bridge_runtime,
    bridge_runtime_for_session,
)
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from tests.unit.team_bridge.test_channel_tool import RecordingClient

QA_ID = "bcv_c719ca33307b45e189cf11b419fdc8c1"

OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_a",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://hub.example/v1",
    enrollment_generation=1,
)


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


class FakeHub(RecordingClient):
    """A RecordingClient whose conversation list is the rooms it has stored.

    ``enforce_unique`` makes create behave like the hub's 409 rule, so a test
    can prove the TOOL maps it; ``list_fails`` breaks the precheck listing.
    """

    def __init__(
        self,
        rooms: list[dict[str, Any]] | None = None,
        *,
        enforce_unique: bool = False,
        list_fails: bool = False,
        page_size: int | None = None,
    ) -> None:
        super().__init__()
        self.rooms: list[dict[str, Any]] = list(rooms or [])
        self.enforce_unique = enforce_unique
        self.list_fails = list_fails
        self.page_size = page_size

    async def conversations(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "conversations", **kwargs})
        if self.list_fails:
            raise RuntimeError("listing unavailable")
        if self.page_size is None:
            return {"conversations": list(self.rooms), "next_cursor": None}
        start = int(kwargs.get("cursor") or 0)
        end = start + self.page_size
        more = end < len(self.rooms)
        return {"conversations": self.rooms[start:end], "next_cursor": str(end) if more else None}

    async def create_conversation(self, body: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "create_conversation", "body": dict(body)})
        name = str(body.get("name") or "")
        if self.enforce_unique and name:
            for room in self.rooms:
                if room["kind"] == body["kind"] and room["name"].strip().lower() == (
                    name.strip().lower()
                ):
                    raise BridgeRefusal(
                        "conversation_name_taken",
                        "A conversation with this name already exists.",
                        http_status=409,
                        details={
                            "field": "name",
                            "rule": "taken",
                            "conversation_id": room["id"],
                            "name": room["name"],
                        },
                    )
        room = {
            "id": f"bcv_{len(self.rooms) + 1:032x}",
            "kind": body["kind"],
            "name": name,
            "visibility": body.get("visibility", "private"),
            "member": True,
        }
        self.rooms.append(room)
        return room


def _gated_runtime(client: Any) -> BridgeRuntime:
    return BridgeRuntime(
        client=cast(TeamBridgeClient, client),
        config_dir=".",
        tenant_id="tnt_1",
        owner=OwnerSnapshot(
            credential_id=1,
            account_id="acc_1",
            tenant_id="tnt_1",
            endpoint_id="bep_1",
            canonical_base_url="https://hub.example",
            enrollment_generation=1,
        ),
        session_id="sess_1",
    )


@pytest.fixture
def open_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.team_bridge import host as host_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)


def _create(name: str, kind: str = "channel", visibility: str = "tenant") -> dict[str, Any]:
    return {
        "op": "channel_create",
        "kind": kind,
        "name": name,
        "visibility": visibility,
        "members": [],
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_a_second_create_with_the_same_name_makes_no_room_and_returns_the_first() -> None:
    hub = FakeHub()
    runtime = _gated_runtime(hub)

    first = await execute_bridge(runtime, _create("qa-coordination"))
    second = await execute_bridge(runtime, _create("  QA-Coordination "))

    assert first["ok"] is True
    assert len(hub.rooms) == 1
    first_id = first["conversation_id"]
    assert second["ok"] is False
    assert second["code"] == "channel_exists"
    assert second["conversation_id"] == first_id
    assert f"channel_post channel_id={first_id}" in second["error"]
    assert "do not create another" in second["error"]
    assert [c["method"] for c in hub.calls].count("create_conversation") == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_groups_refuse_a_duplicate_name_too_but_not_across_kinds() -> None:
    hub = FakeHub()
    runtime = _gated_runtime(hub)

    group = await execute_bridge(runtime, _create("release", kind="group", visibility="private"))
    again = await execute_bridge(runtime, _create("Release", kind="group", visibility="private"))
    channel = await execute_bridge(runtime, _create("release"))

    assert again["code"] == "channel_exists"
    assert again["conversation_id"] == group["conversation_id"]
    assert "a group named 'release'" in again["error"]
    # A channel and a group are different namespaces on the hub.
    assert channel["ok"] is True
    assert len(hub.rooms) == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
@pytest.mark.parametrize("list_fails", [True, False])
async def test_the_hub_409_maps_to_channel_exists_when_the_precheck_misses(
    list_fails: bool,
) -> None:
    existing = {
        "id": QA_ID,
        "kind": "channel",
        "name": "qa-coordination",
        "visibility": "tenant",
    }
    hub = FakeHub([existing], enforce_unique=True, list_fails=list_fails)
    if not list_fails:
        # The precheck reads a list that does not (yet) show the room — a race.
        hub.conversations = _empty_listing(hub)  # type: ignore[method-assign]
    outcome = await execute_bridge(_gated_runtime(hub), _create("qa-coordination"))

    assert outcome == {
        "ok": False,
        "code": "channel_exists",
        "conversation_id": QA_ID,
        "error": (
            f"a channel named 'qa-coordination' already exists ({QA_ID}) — post there "
            f"with channel_post channel_id={QA_ID}; do not create another"
        ),
    }
    assert len(hub.rooms) == 1


def _empty_listing(hub: FakeHub) -> Any:
    async def listing(**kwargs: Any) -> dict[str, Any]:
        hub.calls.append({"method": "conversations", **kwargs})
        return {"conversations": [], "next_cursor": None}

    return listing


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_an_unrelated_create_failure_stays_create_failed() -> None:
    hub = FakeHub()

    async def boom(body: Any, **kwargs: Any) -> dict[str, Any]:
        raise BridgeRefusal("invalid_request", "bad", http_status=400, details={"field": "kind"})

    hub.create_conversation = boom  # type: ignore[method-assign]
    outcome = await execute_bridge(_gated_runtime(hub), _create("fresh"))
    assert outcome["code"] == "create_failed"


# -- a NON-designated session, real fences -------------------------------------


@pytest.fixture
def task_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BridgeRuntime:
    """An enrolled owner whose designated session is ``sess_cos``; this is NOT it."""

    monkeypatch.chdir(tmp_path)
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["team_bridge.enabled"], True)
    Store(tmp_path).commit_enrollment(
        OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="sess_cos"
    )
    runtime = bridge_runtime_for_session(config_manager=manager, session_id="qa-lane-task")
    assert runtime is not None
    return runtime


async def _admit(runtime: BridgeRuntime, monkeypatch: pytest.MonkeyPatch) -> None:
    async def status() -> dict[str, Any]:
        return {
            "api_version": "1",
            "tenants": [
                {
                    "tenant_id": OWNER.tenant_id,
                    "eligible": True,
                    "endpoint": {"id": OWNER.endpoint_id, "state": "active"},
                }
            ],
        }

    monkeypatch.setattr(runtime.client, "status", status)
    assert await admit_bridge_runtime(runtime) == "admitted"


def _qa_rooms() -> list[dict[str, Any]]:
    return [
        {"id": "bcv_dm0000000000000000000000000001", "kind": "dm", "name": "qa-coordination"},
        {
            "id": "bcv_private000000000000000000000001",
            "kind": "channel",
            "name": "qa-coordination",
            "visibility": "private",
            "member": True,
        },
        {
            "id": QA_ID,
            "kind": "channel",
            "name": "qa-coordination",
            "discoverable": True,
            "member": False,
            "member_count": 9,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["qa-coordination", "#qa-coordination", " QA-Coordination "])
async def test_a_non_designated_session_posts_to_qa_coordination_by_name(
    task_session: BridgeRuntime, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    await _admit(task_session, monkeypatch)
    hub = FakeHub(_qa_rooms())
    task_session.client = cast(TeamBridgeClient, hub)

    listing = await execute_bridge(task_session, {"op": "channels"})
    assert listing["ok"] is False and listing["code"] == "not_designated"

    outcome = await execute_bridge(
        task_session,
        {"op": "channel_post", "channel_id": name, "body": "QA window open", "new_post": True},
    )

    assert outcome["ok"] is True, outcome
    sends = [c for c in hub.calls if c["method"] == "conversation_send"]
    assert [c["conversation_id"] for c in sends] == [QA_ID]
    assert outcome["resolved_channel_id"] == QA_ID
    assert outcome["resolved_from_name"] == name.strip()
    # ``channels`` refused before any wire read; the lookup is the only listing.
    assert [c["method"] for c in hub.calls].count("conversations") == 1


@pytest.mark.asyncio
async def test_a_non_designated_session_reads_qa_coordination_by_name(
    task_session: BridgeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _admit(task_session, monkeypatch)
    hub = FakeHub(_qa_rooms())
    task_session.client = cast(TeamBridgeClient, hub)

    by_channel_id = await execute_bridge(
        task_session, {"op": "channel_history", "channel_id": "#qa-coordination"}
    )
    by_name = await execute_bridge(
        task_session, {"op": "channel_history", "name": "qa-coordination"}
    )

    for outcome in (by_channel_id, by_name):
        assert outcome["ok"] is True, outcome
        assert outcome["resolved_channel_id"] == QA_ID
    reads = [c["conversation_id"] for c in hub.calls if c["method"] == "conversation_messages"]
    assert reads == [QA_ID, QA_ID]


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_an_id_needs_no_lookup_and_carries_no_resolution_keys() -> None:
    hub = FakeHub(_qa_rooms())
    outcome = await execute_bridge(
        _gated_runtime(hub), {"op": "channel_history", "channel_id": QA_ID}
    )
    assert outcome["ok"] is True
    assert "resolved_channel_id" not in outcome
    assert all(c["method"] != "conversations" for c in hub.calls)


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_a_private_channel_with_the_name_is_not_resolved() -> None:
    hub = FakeHub(_qa_rooms()[:2])
    outcome = await execute_bridge(
        _gated_runtime(hub),
        {"op": "channel_post", "channel_id": "qa-coordination", "body": "x", "new_post": True},
    )
    assert outcome["ok"] is False
    assert outcome["code"] == "channel_not_found"
    assert "Do not create one" in outcome["error"]
    assert "bcv_private" not in outcome["error"]
    assert all(c["method"] != "conversation_send" for c in hub.calls)


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_two_tenant_visible_channels_with_the_name_are_ambiguous() -> None:
    twin = {
        "id": "bcv_twin0000000000000000000000000001",
        "kind": "channel",
        "name": "QA-coordination",
        "visibility": "tenant",
    }
    hub = FakeHub([*_qa_rooms(), twin])
    outcome = await execute_bridge(
        _gated_runtime(hub), {"op": "channel_history", "channel_id": "qa-coordination"}
    )
    assert outcome["code"] == "channel_ambiguous"
    assert QA_ID in outcome["error"] and twin["id"] in outcome["error"]
    assert all(c["method"] != "conversation_messages" for c in hub.calls)


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_the_lookup_pages_but_stays_bounded() -> None:
    filler = [
        {"id": f"bcv_f{i:031x}", "kind": "channel", "name": f"room-{i}", "visibility": "tenant"}
        for i in range(12)
    ]
    hub = FakeHub([*filler, _qa_rooms()[2]], page_size=5)
    found = await execute_bridge(
        _gated_runtime(hub), {"op": "channel_history", "channel_id": "qa-coordination"}
    )
    assert found["resolved_channel_id"] == QA_ID
    assert [c.get("cursor") for c in hub.calls if c["method"] == "conversations"] == [
        None,
        "5",
        "10",
    ]

    deep = FakeHub([*(filler * 3), _qa_rooms()[2]], page_size=5)
    missed = await execute_bridge(
        _gated_runtime(deep), {"op": "channel_history", "channel_id": "qa-coordination"}
    )
    assert missed["code"] == "channel_not_found"
    assert [c["method"] for c in deep.calls].count("conversations") == 5
    # The bound stopped the read with rooms unread: say how far it went.
    assert "was found in the first 500 conversations" in missed["error"]

    # A list that ENDED without the name is a plain "not found".
    short = FakeHub(filler, page_size=5)
    absent = await execute_bridge(
        _gated_runtime(short), {"op": "channel_history", "channel_id": "qa-coordination"}
    )
    assert absent["code"] == "channel_not_found"
    assert "in the first" not in absent["error"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_neither_id_nor_name_names_both_forms() -> None:
    outcome = await execute_bridge(_gated_runtime(FakeHub()), {"op": "channel_history"})
    assert outcome["error"] == "channel_history needs a conversation id (bcv_…) or a channel name"
    content = json.dumps(outcome)
    assert "not_found" not in content


@pytest.mark.asyncio
@pytest.mark.usefixtures("open_gates")
async def test_the_create_precheck_uses_simple_case_folding_like_the_hub() -> None:
    # Go's EqualFold("Straße", "STRASSE") is false, so the hub accepts this
    # create; full folding (str.casefold) would wrongly refuse it locally.
    hub = FakeHub([{"id": QA_ID, "kind": "channel", "name": "STRASSE", "visibility": "tenant"}])
    runtime = _gated_runtime(hub)

    distinct = await execute_bridge(runtime, _create("Straße"))
    same = await execute_bridge(runtime, _create("strasse"))

    assert distinct["ok"] is True, distinct
    assert same["code"] == "channel_exists"
    assert same["conversation_id"] == QA_ID


@pytest.mark.asyncio
async def test_a_resolved_name_is_written_back_into_args(
    task_session: BridgeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After resolution every reader of ``args`` sees the id, never the name."""

    await _admit(task_session, monkeypatch)
    hub = FakeHub(_qa_rooms())
    task_session.client = cast(TeamBridgeClient, hub)
    seen: list[str] = []
    real = tb_tool._validate_reply_destination

    def recording(runtime: BridgeRuntime, args: dict[str, Any]) -> dict[str, Any] | None:
        seen.append(args.get("channel_id", ""))
        return real(runtime, args)

    monkeypatch.setattr(tb_tool, "_validate_reply_destination", recording)
    outcome = await execute_bridge(
        task_session,
        {
            "op": "channel_post",
            "channel_id": "#qa-coordination",
            "body": "QA window open",
            "new_post": True,
        },
    )

    assert outcome["ok"] is True, outcome
    # The pre-admission check necessarily sees the name; every later one, the id.
    assert seen[0] == "#qa-coordination"
    assert len(seen) > 1
    assert set(seen[1:]) == {QA_ID}
