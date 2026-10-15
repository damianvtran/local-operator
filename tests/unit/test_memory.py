"""Synthetic stores only: prove persistence, negative residue and wire algebra."""

import asyncio
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest

from local_operator.harness.types import ToolContext
from local_operator.memory import (
    SESSION_EXPORT_NAME,
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    _dump,
    parse_url,
)
from local_operator.tools.memory_tools import execute_memory


def fact(body="Release evidence lives at docs/release.md", **kwargs):
    return dict(title="Release evidence", body=body, tags=["release", "evidence", "docs"], **kwargs)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("session-a", "team-a", "agent-a"),
        tmp_path / "sessions" / "session-a",
        origin="device-a",
    )


def test_store_recall_restart_and_optimistic_correction(store):
    first = store.put("session", fact())
    reopened = MemoryStore(store.root.parent, store.identity, store.session_dir, origin="device-a")
    assert reopened.search("releases")
    assert reopened.show("session", first["id"])["fact"]["body"] == fact()["body"]
    with pytest.raises(MemoryError, match="revision"):
        reopened.put(
            "session",
            fact("Corrected release location"),
            entity=first["id"],
            expected_version="stale",
        )
    after = reopened.put(
        "session",
        fact("Corrected release location"),
        entity=first["id"],
        expected_version=first["version"],
    )
    assert after["version"] != first["version"]
    assert reopened.show("session", first["id"])["fact"]["body"] == "Corrected release location"


@pytest.mark.parametrize("scope", ["session", "team", "agent", "global"])
def test_scope_filter_on_every_path(store, scope):
    entry = store.put(scope, fact())
    other = MemoryStore(store.root.parent, MemoryIdentity("other", "other", "other"))
    if scope == "global":
        assert other.search("release")
        assert other.show(scope, entry["id"])
    else:
        assert other.search("release") == []
        assert other.inventory(scope)["items"] == []
        with pytest.raises(MemoryError):
            other.show(scope, entry["id"])
        with pytest.raises(MemoryError):
            other.put(scope, fact(), entity=entry["id"], expected_version=entry["version"])
    with pytest.raises(MemoryError):
        store.show("global" if scope != "global" else "session", entry["id"])


def test_reserved_main_cannot_agent_scope(tmp_path):
    store = MemoryStore(tmp_path, MemoryIdentity("s", "", "main"))
    for action in (
        lambda: store.inventory("agent"),
        lambda: store.put("agent", fact()),
        lambda: store.show("agent", "x"),
    ):
        with pytest.raises(MemoryError, match="session or global"):
            action()
    assert not store.path.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"body": "SSN 123-45-6789"},
        {"title": "passport number AB1234567"},
        {"tags": ["licence number A1234567"]},
        {"source": {"session": "s", "kind": "tool", "entry": "card number 4111 1111 1111 1111"}},
        {"aliases": ["mongodb://reader:synthetic-password@db.invalid/test"]},
        {"body": "Private file ~/.ssh/id_rsa"},
    ],
)
def test_sensitive_entire_field_rejection_leaves_no_residue(tmp_path, payload):
    store = MemoryStore(tmp_path, MemoryIdentity("s", "t", "a"))
    item = fact()
    item.update(payload)
    with pytest.raises(MemoryError):
        store.put("session", item)
    assert not store.root.exists()


def test_secret_name_and_safe_contact_allowed(store):
    item = store.put(
        "global", fact("Contact alice@example.invalid; passport stored as secret PASSPORT_NO")
    )
    assert store.show("global", item["id"])["fact"]["body"].endswith("PASSPORT_NO")


def test_no_whole_bank_and_bounded_recall(store):
    for n in range(25):
        store.put("global", fact(f"Release evidence number {n} is at docs/notes-{n}.md"))
    inventory = store.inventory()
    assert len(inventory["items"]) == 20
    assert all("body" not in row for row in inventory["items"])
    assert store.search("release", tokens=1) == []
    with pytest.raises(MemoryError):
        store.search("release", tokens=2001)
    with pytest.raises(MemoryError, match="256"):
        store.put("global", fact("very long sentence " * 300))


def test_history_quota_refuses_without_eviction(store, monkeypatch):
    import local_operator.memory as memory

    first = store.put("session", fact())
    monkeypatch.setattr(memory, "MAX_OWNER_OPS", 1)
    with pytest.raises(MemoryError, match="quota"):
        store.put("session", fact("Another explicit fact"))
    assert store.inventory("session")["ops"] == 1
    assert store.show("session", first["id"])["fact"]["body"] == fact()["body"]


def test_concurrent_correction_exactly_one_wins(store):
    first = store.put("team", fact())

    def update(n):
        try:
            return store.put(
                "team",
                fact(f"Release corrected {n}"),
                entity=first["id"],
                expected_version=first["version"],
            )
        except MemoryError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(x is not None for x in pool.map(update, (1, 2))) == 1


def test_portable_session_rebuild_and_promotion(store, tmp_path):
    first = store.put("session", fact())
    store.put("team", fact("Team-only release evidence"))
    exported = (store.session_dir / SESSION_EXPORT_NAME).read_text()
    assert "Team-only" not in exported
    target = MemoryStore(tmp_path / "second", store.identity, store.session_dir, origin="device-b")
    target.import_session()
    assert target.show("session", first["id"])["fact"]["body"] == fact()["body"]
    target.import_session()
    assert target.inventory()["ops"] == 1
    promoted_dir = tmp_path / "promoted-session"
    promoted_dir.mkdir()
    shutil.copyfile(store.session_dir / SESSION_EXPORT_NAME, promoted_dir / SESSION_EXPORT_NAME)
    promoted = MemoryStore(
        tmp_path / "third", MemoryIdentity("new-session", "team-a", "agent-a"), promoted_dir
    )
    promoted.import_session()
    assert len(promoted.search("release", "session")) == 1
    assert promoted.inventory("session")["items"][0]["id"] != first["id"]
    promoted.import_session()
    assert promoted.inventory()["ops"] == 1


def test_tombstone_removes_all_bodies_and_search(store, tmp_path):
    first = store.put("session", fact("Unique forgotten body zebra"))
    changed = store.put(
        "session",
        fact("Second forgotten body zebra"),
        entity=first["id"],
        expected_version=first["version"],
    )
    store.put("session", {}, entity=first["id"], expected_version=changed["version"], forget=True)
    assert not store.search("zebra")
    assert "forgotten body" not in (store.session_dir / SESSION_EXPORT_NAME).read_text()
    with store.connection() as db:
        ops = store._ops(db, first["id"])
        assert all(op["fact"] is None for op in ops)
    assert store.show("session", first["id"])["status"] == "forgotten"
    clone = MemoryStore(tmp_path / "other", store.identity, store.session_dir)
    clone.import_session()
    assert not clone.search("zebra")
    for path in store.root.glob("memory.db*"):
        assert b"forgotten body" not in path.read_bytes()


def test_causal_conflict_retained_and_tombstone_order_independent(store, tmp_path):
    first = store.put("global", fact())
    with store.connection() as db:
        put = store._ops(db, first["id"])[0]
    second = dict(
        put,
        op_id="other-op",
        origin_device="device-b",
        seq=1,
        hlc=[put["hlc"][0] + 1, 0, "device-b"],
        parents=[],
    )
    second["fact"] = dict(put["fact"], body="Independent release correction")
    second["body_sha"] = hashlib.sha256(_dump(second["fact"]).encode()).hexdigest()
    store.apply_ops([second])
    view = store.show("global", first["id"])
    assert len(view["conflicts"]) == 1
    assert view["fact"]["body"] == "Independent release correction"
    tomb = dict(
        put,
        op_id="tomb-op",
        origin_device="device-c",
        seq=1,
        hlc=[put["hlc"][0] - 1, 0, "device-c"],
        parents=[],
        kind="tombstone",
        fact=None,
    )
    for order in ([put, second, tomb], [tomb, second, put]):
        clone = MemoryStore(tmp_path / str(order[0]["op_id"]), store.identity)
        for op in order:
            clone.apply_ops([op])
        assert clone.show("global", first["id"])["status"] == "forgotten"
        assert not clone.search("release")
        assert clone.apply_ops(order) == 0


def test_invalid_batch_is_atomic_and_no_sensitive_residue(tmp_path, store):
    first = store.put("global", fact())
    with store.connection() as db:
        op = store._ops(db, first["id"])[0]
    bad = dict(op, op_id="bad", fact=dict(op["fact"], title="passport number AB1234567"))
    target = MemoryStore(tmp_path / "rejected", store.identity)
    with pytest.raises(MemoryError):
        target.apply_ops([op, bad])
    assert not target.root.exists()
    with pytest.raises(MemoryError, match="origin"):
        target.apply_ops([op], origins={"device-b"})
    assert not target.root.exists()


def test_exact_dedup_preserves_near_duplicate_contradictions(store):
    first = store.put("session", fact("Deploy Mercury on Friday", subject="Mercury"))
    assert (
        store.put("session", fact("Deploy Mercury on Friday", subject="Mercury"))["id"]
        == first["id"]
    )
    other = store.put("session", fact("Do not deploy Mercury on Friday", subject="Mercury"))
    assert other["id"] != first["id"]
    assert len(store.search("Mercury Friday")) == 2


def test_only_explicit_same_scope_replacement_suppresses_old_recall(store):
    first = store.put("session", fact("Mercury launches Friday", subject="Mercury"))
    other = store.put("session", fact("Mercury launches Monday", subject="Mercury"))
    assert len(store.search("Mercury launches")) == 2
    latest = store.put(
        "session", fact("Mercury launches Tuesday", subject="Mercury", replaces=first["id"])
    )
    ids = {row["id"] for row in store.search("Mercury launches")}
    assert first["id"] not in ids and other["id"] in ids and latest["id"] in ids
    assert store.show("session", first["id"])["fact"]["body"].endswith("Friday")
    with pytest.raises(MemoryError, match="same authorized scope"):
        store.put("global", fact("Mercury launches Wednesday", replaces=other["id"]))


def test_related_is_one_hop_scope_bound_and_newest_first(store):
    first = store.put("session", fact("Source docs/shared.md", created_at=100))
    second = store.put("session", fact("Decision docs/shared.md and docs/hop.md", created_at=200))
    third = store.put("session", fact("Neighbor docs/shared.md", created_at=300))
    store.put("global", fact("Private foreign association docs/shared.md", created_at=400))
    store.put("session", fact("Two-hop association docs/hop.md", created_at=500))
    related = store.related("session", first["id"])
    assert [row["id"] for row in related] == [third["id"], second["id"]]
    assert "related" not in store.show("session", first["id"])
    with pytest.raises(MemoryError):
        store.related("global", first["id"])


def test_actual_read_write_edit_routes(store):
    store.root.parent.mkdir(parents=True, exist_ok=True)
    (store.root.parent / "config.yml").write_text(json.dumps({"values": {"memory": {"enabled": True}}}))
    from local_operator.tools.builtin import execute_edit, execute_read, execute_write

    context = ToolContext(
        session_id="session-a",
        team_id="team-a",
        agent_id="agent-a",
        memory_config_dir=str(store.root.parent),
        session_dir=str(store.session_dir),
    )

    async def run():
        created = await execute_write(
            "w", {"path": "memory://session", "content": json.dumps(fact())}, context=context
        )
        assert not created.is_error, created
        item = json.loads(created.content[0].text)
        shown = await execute_read("r", {"path": "memory://session/" + item["id"]}, context=context)
        assert not shown.is_error, shown
        assert fact()["body"] in shown.content[0].text
        changed = await execute_edit(
            "e",
            {
                "path": "memory://session/" + item["id"],
                "old_text": item["version"],
                "new_text": json.dumps(fact("Corrected through actual edit tool")),
            },
            context=context,
        )
        assert not changed.is_error, changed
        rejected = await execute_memory(
            "bad",
            "write",
            {"path": "memory://agent", "content": json.dumps(fact("SSN 123-45-6789"))},
            context,
        )
        assert rejected.is_error
        assert "123-45" not in rejected.content[0].text

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "memory://other/x",
        "memory://session/../x",
        "memory://session/a/b",
        "memory://global/x#y",
        "memory://agent?owner=other",
    ],
)
def test_no_cross_owner_addressing(url):
    with pytest.raises((MemoryError, ValueError)):
        parse_url(url)
