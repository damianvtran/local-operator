"""Moving a fact between scopes is one atomic, lineage-preserving mutation.

The defect these tests pin: a fact placed in the wrong scope had no way back.
Forget destroys it; edit cannot change its scope (an entity's scope is immutable
in the op protocol). The move verb is the only operation that relieves one
scope's quota without losing the fact, its provenance or its timestamps.
"""

import json

import pytest

from local_operator.harness.types import ToolContext
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore
from local_operator.tools.memory_tools import execute_memory


def fact(body="Release evidence lives at docs/release.md", **kwargs):
    return dict(title="Release evidence", body=body, tags=["release", "evidence"], **kwargs)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("session-a", "team-a", "agent-a"),
        tmp_path / "sessions" / "session-a",
        origin="device-a",
    )


def reference(scope, item):
    return f"{scope}/{item['id']}"


def test_single_move_preserves_the_fact_and_its_provenance(store):
    original = store.put(
        "global",
        fact(created_at=1700000000.0, verified_at=1700000100.0, pinned=True),
    )
    before = store.show("global", original["id"])
    moved = store.move(reference("global", original), "session")
    assert moved["action"] == "moved"
    assert len(moved["moved"]) == 1
    entry = moved["moved"][0]
    assert entry["from"] == reference("global", original)
    assert entry["to"].startswith("session/")
    _, new_id = entry["to"].split("/", 1)
    after = store.show("session", new_id)
    # Byte-identical: text, provenance, created_at, verified_at and pinned flag.
    assert after["fact"] == before["fact"]
    assert after["fact"]["created_at"] == 1700000000.0
    # Lineage: the destination's first revision descends from the source head, so
    # this reads as a move, not as a fresh unrelated fact.
    assert after["moved_from"] == {"scope": "global", "id": original["id"]}
    # The source is gone from the live store, and its quota is relieved.
    assert store.show("global", original["id"])["status"] == "forgotten"
    assert store.inventory("global")["active_tokens"] == 0
    assert store.inventory("session")["active_tokens"] > 0


def test_batch_move_is_all_or_nothing_and_relieves_the_source(store):
    first = store.put("global", fact("Fact one lives at docs/one.md"))
    second = store.put("global", fact("Fact two lives at docs/two.md"))
    assert store.inventory("global")["active_tokens"] > 0
    moved = store.move([reference("global", first), reference("global", second)], "session")
    assert len(moved["moved"]) == 2
    assert store.inventory("global")["active_tokens"] == 0
    assert len(store.inventory("session")["items"]) == 2
    assert {row["scope"] for row in store.inventory("session")["items"]} == {"session"}


def test_a_late_failure_rolls_an_entire_batch_back(store, monkeypatch):
    first = store.put("global", fact("Fact one lives at docs/one.md"))
    second = store.put("global", fact("Fact two lives at docs/two.md"))
    calls = {"n": 0}
    real = MemoryStore._quota

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 1:
            raise MemoryError("synthetic late failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(MemoryStore, "_quota", staticmethod(explode))
    with pytest.raises(MemoryError, match="synthetic late failure"):
        store.move([reference("global", first), reference("global", second)], "session")
    assert store.show("global", first["id"])["status"] == "active"
    assert store.show("global", second["id"])["status"] == "active"
    assert store.inventory("session")["items"] == []
    assert store.inventory("global")["active_tokens"] > 0


def test_destination_quota_refusal_names_the_numbers_and_changes_nothing(store, monkeypatch):
    import local_operator.memory as memory

    monkeypatch.setattr(memory, "OWNER_TOKENS", 300)
    kept = store.put("team", fact("Existing team fact " + "word " * 150))
    incoming = store.put("session", fact("Incoming session fact " + "word " * 150))
    with pytest.raises(MemoryError) as refused:
        store.move(reference("session", incoming), "team")
    message = str(refused.value)
    assert "team" in message and "current" in message and "needs" in message
    assert "cap" in message and "300" in message
    assert store.show("session", incoming["id"])["status"] == "active"
    assert store.show("team", kept["id"])["status"] == "active"
    assert len(store.inventory("team")["items"]) == 1


def test_an_identical_destination_fact_is_refused_not_silently_merged(store):
    body = "Shared release evidence lives at docs/shared.md"
    source = store.put("global", fact(body))
    destination = store.put("session", fact(body))
    with pytest.raises(MemoryError, match="identical"):
        store.move(reference("global", source), "session")
    assert store.show("global", source["id"])["status"] == "active"
    assert len(store.inventory("session")["items"]) == 1
    assert store.show("session", destination["id"])["status"] == "active"


def test_unknown_and_ambiguous_ids_are_refused(store, monkeypatch):
    import local_operator.memory as memory

    with pytest.raises(MemoryError, match="not found"):
        store.move("global/00000000-dead-beef-0000-000000000000", "session")
    import itertools

    counter = itertools.count(1)
    monkeypatch.setattr(memory, "_id", lambda: f"abcdefab-0000-7000-8000-{next(counter):012d}")
    store.put("global", fact("First ambiguous fact"))
    store.put("global", fact("Second ambiguous fact"))
    with pytest.raises(MemoryError, match="matches 2"):
        store.move("abcdefab", "session")
    assert len(store.inventory("global")["items"]) == 2


def test_moving_to_the_current_scope_and_repeating_is_a_clear_refusal(store):
    item = store.put("session", fact())
    with pytest.raises(MemoryError, match="already"):
        store.move(reference("session", item), "session")
    moved = store.move(reference("session", item), "team")
    assert moved["moved"][0]["to"].startswith("team/")
    # A repeat finds a forgotten source and refuses; it never creates a second copy.
    with pytest.raises(MemoryError, match="forgotten|not found"):
        store.move(reference("session", item), "team")
    assert len(store.inventory("team")["items"]) == 1


def test_a_conflicted_entity_is_refused_rather_than_losing_a_sibling(store):
    import hashlib

    from local_operator.memory import _dump

    item = store.put("global", fact("Original release evidence"))
    with store.connection() as db:
        base = store._ops(db, item["id"])[0]
    forks = []
    for origin, n in (("device-b", 1), ("device-c", 2)):
        sibling = {**base["fact"], "body": f"A concurrent sibling fact {n}"}
        forks.append(
            {
                "v": 1,
                "op_id": f"01a10000-0000-7000-8000-00000000000{n}",
                "entity_id": item["id"],
                "origin_device": origin,
                "seq": 1,
                "hlc": [base["hlc"][0] + n, 0, origin],
                "parents": [base["op_id"]],
                "kind": "put",
                "scope": "global",
                "owner_key": "global",
                "fact": sibling,
                "body_sha": hashlib.sha256(_dump(sibling).encode()).hexdigest(),
            }
        )
    store.apply_ops(forks, origins={"device-b", "device-c"}, contiguous=False)
    with pytest.raises(MemoryError, match="concurrent revisions"):
        store.move(reference("global", item), "session")
    assert store.show("global", item["id"])["status"] in ("active", "candidate")
    assert store.inventory("session")["items"] == []
    assert len(store.show("global", item["id"])["conflicts"]) == 1


def test_move_records_ops_and_propagates_to_a_peer_like_any_mutation(store, tmp_path):
    from local_operator.network.memory import batch

    item = store.put("global", fact("Mesh move proof lives at docs/mesh.md"))
    store.move(reference("global", item), "session")
    payload = batch(
        store,
        {},
        scopes=["global", "session", "team", "agent"],
        origins={"device-a"},
    )
    peer = MemoryStore(
        tmp_path / "peer",
        MemoryIdentity("session-a", "team-a", "agent-a"),
        origin="device-b",
    )
    peer.apply_ops(payload["ops"], origins={"device-a"})
    # No divergence and no duplication: the peer holds the fact once, at the
    # destination, and the source is forgotten there too.
    assert (
        len([row for row in peer.inventory()["items"] if row["title"] == "Release evidence"]) == 1
    )
    destination = [row for row in peer.inventory("session")["items"]]
    assert len(destination) == 1
    assert peer.show("session", destination[0]["id"])["moved_from"]["scope"] == "global"
    assert peer.show("global", item["id"])["status"] == "forgotten"


def test_the_edit_tool_exposes_move_with_its_revision_guard(tmp_path):
    context = ToolContext(
        session_id="session-a",
        team_id="team-a",
        agent_id="agent-a",
        memory_config_dir=str(tmp_path / "cfg"),
        session_dir=str(tmp_path / "sessions" / "session-a"),
    )
    created = execute_memory_sync(
        "write", {"path": "memory://global", "content": json.dumps(fact())}, context
    )
    assert not created.is_error
    created_value = json.loads(str(getattr(created.content[0], "text", "")))
    stale = execute_memory_sync(
        "edit",
        {
            "path": "memory://global/" + created_value["id"],
            "old_text": "stale-revision",
            "new_text": json.dumps({"move": "session"}),
        },
        context,
    )
    assert stale.is_error
    moved = execute_memory_sync(
        "edit",
        {
            "path": "memory://global/" + created_value["id"],
            "old_text": created_value["version"],
            "new_text": json.dumps({"move": "session"}),
        },
        context,
    )
    assert not moved.is_error
    value = json.loads(str(getattr(moved.content[0], "text", "")))
    assert value["moved"][0]["from"] == "global/" + created_value["id"]
    assert value["moved"][0]["to"].startswith("session/")


def test_cli_move_verb_moves_a_batch_and_prints_a_paste_ready_receipt(store):
    from local_operator.memory_cli import render_human, run

    first = store.put("global", fact("Fact one lives at docs/one.md"))
    second = store.put("global", fact("Fact two lives at docs/two.md"))
    text, failed = run(
        f"move {reference('global', first)} {reference('global', second)} --to session",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
        prefix="/memory",
    )
    assert not failed, text
    receipt = render_human(text, failed, "/memory")
    assert "Moved 2 fact(s) to session." in receipt
    assert "→" in receipt and "session/" in receipt
    value = json.loads(text)
    assert value["destination"] == "session"
    assert len(value["moved"]) == 2


def execute_memory_sync(name, args, context):
    import asyncio

    return asyncio.run(execute_memory("call", name, args, context))
