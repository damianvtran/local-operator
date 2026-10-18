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


def test_move_into_global_applies_the_same_scope_fit_guard_as_put(store):
    work = store.put(
        "session",
        {
            "title": "Release cut",
            "body": "Merge PR #2463 after CI goes green; next step: tag v1.2.3",
        },
    )
    with pytest.raises(MemoryError) as refused:
        store.move(reference("session", work), "global")
    # Same value-free refusal a global put gets; no work text is echoed.
    assert "memory://session" in str(refused.value)
    assert "#2463" not in str(refused.value)
    assert store.show("session", work["id"])["status"] == "active"
    assert store.inventory("global")["items"] == []


def test_the_edit_tool_refuses_a_move_that_bypasses_scope_fit(tmp_path):
    context = ToolContext(
        session_id="session-a",
        team_id="team-a",
        agent_id="agent-a",
        memory_config_dir=str(tmp_path / "cfg"),
        session_dir=str(tmp_path / "sessions" / "session-a"),
    )
    created = execute_memory_sync(
        "write",
        {
            "path": "memory://session",
            "content": json.dumps(
                {
                    "title": "Release cut",
                    "body": "Merge PR #2463; next step: tag v1.2.3",
                }
            ),
        },
        context,
    )
    assert not created.is_error
    item = json.loads(str(getattr(created.content[0], "text", "")))
    moved = execute_memory_sync(
        "edit",
        {
            "path": "memory://session/" + item["id"],
            "old_text": item["version"],
            "new_text": json.dumps({"move": "global"}),
        },
        context,
    )
    assert moved.is_error
    assert "memory://session" in str(getattr(moved.content[0], "text", ""))


def test_a_peer_correction_the_mover_never_saw_is_refused_not_erased(tmp_path):
    from local_operator.network.memory import batch

    scopes = ["global", "session", "team", "agent"]
    mover = MemoryStore(
        tmp_path / "a", MemoryIdentity("session-a", "team-a", "agent-a"), origin="device-a"
    )
    peer = MemoryStore(
        tmp_path / "b", MemoryIdentity("session-a", "team-a", "agent-a"), origin="device-b"
    )
    item = mover.put("global", fact("Shared fact about the kestrel runbook"))
    first = batch(mover, {}, scopes=scopes, origins={"device-a"})
    peer.apply_ops(first["ops"], origins={"device-a"})
    # The peer corrects the fact; the mover has not synced that correction.
    head = peer.show("global", item["id"])
    peer.put(
        "global",
        fact("IMPORTANT correction made on device B"),
        entity=item["id"],
        expected_version=head["version"],
    )
    mover.move(reference("global", item), "session")
    second = batch(mover, first["after"], scopes=scopes, origins={"device-a"})
    with pytest.raises(MemoryError, match="revision"):
        peer.apply_ops(second["ops"], origins={"device-a"})
    # The correction survives, is still searchable, and no destination copy appeared.
    assert peer.show("global", item["id"])["fact"]["body"].startswith("IMPORTANT")
    assert peer.search("IMPORTANT")
    assert peer.inventory("session")["items"] == []
    # Resolution: keep what the correction adds, forget the source, sync again.
    current = peer.show("global", item["id"])
    peer.put("global", {}, entity=item["id"], expected_version=current["version"], forget=True)
    peer.apply_ops(second["ops"], origins={"device-a"})
    assert peer.show("global", item["id"])["status"] == "forgotten"
    destination = peer.inventory("session")["items"]
    assert len(destination) == 1
    assert peer.show("session", destination[0]["id"])["moved_from"] == {
        "scope": "global",
        "id": item["id"],
    }


def test_a_move_whose_destination_scope_is_filtered_still_replicates(tmp_path):
    from local_operator.network.memory import batch

    mover = MemoryStore(
        tmp_path / "a", MemoryIdentity("session-a", "team-a", "agent-a"), origin="device-a"
    )
    peer = MemoryStore(
        tmp_path / "b", MemoryIdentity("session-a", "team-a", "agent-a"), origin="device-b"
    )
    item = mover.put("global", fact("Fact moving to a filtered destination"))
    first = batch(mover, {}, scopes=["global"], origins={"device-a"})
    peer.apply_ops(first["ops"], origins={"device-a"})
    mover.move(reference("global", item), "team")
    # This peer syncs global only: the tombstone travels, the destination put does not.
    second = batch(mover, first["after"], scopes=["global"], origins={"device-a"})
    assert [op["kind"] for op in second["ops"]] == ["tombstone"]
    # The filtered-out destination op is a sequence SKIP; the wire receiver applies
    # batches with contiguous=False, exactly as network.memory.receive does.
    peer.apply_ops(second["ops"], origins={"device-a"}, contiguous=False)
    assert peer.show("global", item["id"])["status"] == "forgotten"


def test_moved_from_survives_a_correction_and_is_printed_by_show(store):
    from local_operator.memory_cli import render_human, run

    original = store.put("global", fact("Fact that will be moved"))
    moved = store.move(reference("global", original), "session")
    _, dest_id = moved["moved"][0]["to"].split("/", 1)
    head = store.show("session", dest_id)
    store.put(
        "session",
        fact("Fact that was moved and then corrected"),
        entity=dest_id,
        expected_version=head["version"],
    )
    assert store.show("session", dest_id)["moved_from"] == {
        "scope": "global",
        "id": original["id"],
    }
    text, failed = run(
        f"show session/{dest_id}",
        config=store.root.parent,
        identity=MemoryIdentity("session-a", "team-a", "agent-a"),
        session_dir=store.session_dir,
    )
    assert not failed, text
    assert "Moved from: global/" + original["id"] in render_human(text, failed)


def test_move_out_resets_the_source_warning_state(store, monkeypatch):
    import local_operator.memory as memory

    monkeypatch.setattr(memory, "OWNER_TOKENS", 1000)
    references = []
    while store.inventory("session")["active_tokens"] < 900:
        item = store.put("session", fact(f"Work checkpoint {len(references)}"))
        references.append(reference("session", item))
        if len(references) > 20:
            break
    # Relief out of the warned scope must clear its stored crossing level.
    store.move(references, "team")
    assert store.inventory("session")["active_tokens"] == 0
    # Need the new destination ids, not the old session ids.
    destinations = []
    for row in store.inventory("team", limit=50)["items"]:
        destinations.append(f"team/{row['id']}")
    assert len(destinations) == len(references)
    result = store.move(destinations, "session")
    assert "warning" in result
    assert "90" in result["warning"]


def test_move_refuses_when_the_source_would_exceed_its_history_cap(store, monkeypatch):
    import local_operator.memory as memory

    monkeypatch.setattr(memory, "MAX_TOMBSTONES", 2)
    items = [store.put("session", fact(f"Fact number {n}")) for n in range(3)]
    with pytest.raises(MemoryError, match="history cap"):
        store.move([reference("session", item) for item in items], "team")
    assert len(store.inventory("session")["items"]) == 3
    assert store.inventory("team")["items"] == []


def test_move_honours_a_scope_restriction_on_the_target(store):
    item = store.put("session", fact("Session-only fact"))
    with pytest.raises(MemoryError, match="not found"):
        store.move(item["id"], "team", scope="global")
    assert store.show("session", item["id"])["status"] == "active"
    assert store.inventory("team")["items"] == []


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
