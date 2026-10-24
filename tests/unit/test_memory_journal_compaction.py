"""Bounded journal compaction: the two ways a plausible implementation is wrong.

The pass keeps ONE row per entity (all heads) plus every tombstone plus the ops
a cross-entity causal edge pins. `_reindex` rebuilds `current`/`recall`/
`associations` from an entity's ops and materialises `_heads(ops)[0]`, so that
retention is state-preserving by construction.

TWO SIMPLIFICATIONS PASS EVERY NAIVE TEST AND ARE STILL DEFECTS, so each has a
test that fails loudly:

* "keep heads[0]" keeps `current` correct and deletes conflict visibility.
  `test_compaction_keeps_every_head_of_a_concurrent_edit` builds a genuine
  two-head entity and asserts BOTH survive; a one-head rule drops one.
* "drop tombstones, they are 42% of the journal" makes a peer that received the
  put but not the tombstone hold a fact nothing will ever clear.
  `test_compaction_keeps_the_tombstone_of_a_forgotten_entity` asserts the
  tombstone survives and the puts do not.

The rest pin the pass's contract: materialised state does not move, the preview
deletes nothing, a stale fingerprint refuses the whole pass, a second pass is a
no-op, cross-entity pins keep `moved_from`/`replaces` working, and the portable
session export is rewritten so an import cannot resurrect what was dropped.
"""

import hashlib
import json
import shutil
import sqlite3
import threading
import time
from typing import Any

import pytest

from local_operator.memory import (
    SESSION_EXPORT_NAME,
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    _dump,
)
from local_operator.memory_cli import render_human, run


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


def _rows(store: MemoryStore) -> dict[str, list[tuple[Any, ...]]]:
    """Every row a compaction could move, read through a side connection."""
    db = sqlite3.connect(f"{store.path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return {
            "ops": [
                tuple(row)
                for row in db.execute(
                    "SELECT op_id,entity_id,origin,seq,scope,owner,kind,data FROM ops "
                    "ORDER BY op_id"
                )
            ],
            "current": [
                tuple(row)
                for row in db.execute(
                    "SELECT entity_id,scope,owner,op_id,fact,tokens FROM current ORDER BY entity_id"
                )
            ],
            "recall": [
                tuple(row)
                for row in db.execute(
                    "SELECT entity_id,title,tags,body FROM recall ORDER BY entity_id"
                )
            ],
            "associations": [
                tuple(row)
                for row in db.execute(
                    "SELECT entity,scope,owner,locator FROM associations ORDER BY entity,locator"
                )
            ],
        }
    finally:
        db.close()


def _all_ops(store: MemoryStore) -> list[dict[str, Any]]:
    with store.connection() as db:
        return [json.loads(row[0]) for row in db.execute("SELECT data FROM ops")]


def _schema_objects(store: MemoryStore, name: str) -> int:
    """Count a schema object through a READ-ONLY connection (``connection()`` would create it)."""
    db = sqlite3.connect(f"{store.path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return int(
            db.execute("SELECT count(*) FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
        )
    finally:
        db.close()


def _compacted(store: MemoryStore) -> dict[str, Any]:
    """Preview, then apply the previewed set — the only two-step the API allows."""
    plan = store.journal_plan()
    return store.compact(expect=plan["fingerprint"])


def _sibling(store: MemoryStore, entity: str, op_id: str = "sibling-op") -> dict[str, Any]:
    """A concurrent edit: a second head for ``entity``, parented on its first op."""
    with store.connection() as db:
        first = store._ops(db, entity)[0]
    concurrent = dict(
        first,
        op_id=op_id,
        origin_device="device-b",
        seq=1,
        hlc=[first["hlc"][0] + 1, 0, "device-b"],
        parents=[first["op_id"]],
    )
    concurrent["fact"] = dict(first["fact"], body="Independent release correction")
    concurrent["body_sha"] = hashlib.sha256(_dump(concurrent["fact"]).encode()).hexdigest()
    store.apply_ops([concurrent])
    return concurrent


def test_compaction_keeps_every_head_of_a_concurrent_edit(store):
    """Keep the head SET. Dropping a sibling keeps `current` and loses the conflict."""
    first = store.put("global", fact())
    edited = store.put(
        "global",
        fact("Corrected release location"),
        entity=first["id"],
        expected_version=first["version"],
    )
    sibling = _sibling(store, first["id"])
    before = store.show("global", first["id"])
    assert len(before["conflicts"]) == 1
    with store.connection() as db:
        before_heads = {op["op_id"] for op in store._heads(store._ops(db, first["id"]))}
    assert before_heads == {edited["version"], sibling["op_id"]}

    receipt = _compacted(store)

    # The superseded first revision is the ONLY thing dropped here.
    assert receipt["ops"]["dropped"] == 1
    with store.connection() as db:
        survived = {op["op_id"] for op in store._ops(db, first["id"])}
        after_heads = {op["op_id"] for op in store._heads(store._ops(db, first["id"]))}
    assert survived == {edited["version"], sibling["op_id"]}
    assert after_heads == before_heads, "compaction changed the head set"
    after = store.show("global", first["id"])
    assert len(after["conflicts"]) == 1, "compaction hid a visible concurrent edit"
    assert after["fact"] == before["fact"]


def test_compaction_keeps_the_tombstone_of_a_forgotten_entity(store):
    """A tombstone is the only instruction that clears a peer's `current` row."""
    first = store.put("session", fact())
    edited = store.put(
        "session",
        fact("Second body"),
        entity=first["id"],
        expected_version=first["version"],
    )
    store.put(
        "session", fact(), entity=first["id"], expected_version=edited["version"], forget=True
    )
    receipt = _compacted(store)

    assert receipt["dropped"]["under_forgotten"] == 2
    assert receipt["retained"]["tombstones"] == 1
    with store.connection() as db:
        remaining = store._ops(db, first["id"])
    assert [op["kind"] for op in remaining] == ["tombstone"]
    assert store.show("session", first["id"])["status"] == "forgotten"
    with pytest.raises(MemoryError, match="forgotten"):
        store.put(
            "session",
            fact(),
            entity=first["id"],
            expected_version=remaining[0]["op_id"],
        )


def test_compaction_does_not_move_any_materialised_row(store):
    """`current`/`recall`/`associations` are rebuilt from ops; they must not shift."""
    live = store.put("global", fact("Deploy mercury on friday at docs/runbook.md"))
    store.put(
        "global",
        fact("Deploy mercury on saturday at docs/runbook.md"),
        entity=live["id"],
        expected_version=live["version"],
    )
    dead = store.put("session", fact("Temporary staging note"))
    store.put("session", fact(), entity=dead["id"], expected_version=dead["version"], forget=True)
    before = _rows(store)

    receipt = _compacted(store)

    after = _rows(store)
    assert receipt["ops"]["dropped"] > 0
    for table in ("current", "recall", "associations"):
        assert after[table] == before[table], f"{table} changed under compaction"


def test_compaction_keeps_a_moved_fact_s_source_for_moved_from(store):
    """`moved_from` resolves through a CROSS-entity parent op, so it must be pinned."""
    original = store.put("session", fact("Moved release note"))
    moved = store.move(f"session/{original['id']}", "global")
    destination = moved["moved"][0]["to"]
    _, moved_id = destination.split("/", 1)
    assert store.show("global", moved_id)["moved_from"] == {
        "scope": "session",
        "id": original["id"],
    }

    receipt = _compacted(store)

    # The source put is under a tombstoned entity and would be dropped as bytes
    # without the pin; `_moved_from` reads its row, not just the edge.
    assert receipt["retained"]["pinned"] >= 2
    with store.connection() as db:
        assert store._ops(db, original["id"])[0]["kind"] == "put"
    assert store.show("global", moved_id)["moved_from"] == {
        "scope": "session",
        "id": original["id"],
    }


def test_compaction_keeps_a_replacement_edge_that_hides_the_replaced_fact(store):
    """`_NOT_REPLACED_SQL` INNER JOINs the parent op; dropping it un-hides the old fact."""
    old = store.put("global", dict(fact("Release evidence was at docs/old.md"), tags=["rel"]))
    new = store.put(
        "global",
        dict(fact("Release evidence is at docs/new.md"), tags=["rel"], replaces=old["id"]),
    )
    assert [row["id"] for row in store.tagged("rel")] == [new["id"]]

    receipt = _compacted(store)

    assert receipt["retained"]["pinned"] >= 1
    assert [row["id"] for row in store.tagged("rel")] == [new["id"]]
    assert store.search("old") == []


def test_compaction_prunes_dangling_causal_edges_and_reindex_regrows_them_inertly(store):
    """An edge row is inert once an endpoint is gone; the prune is what reclaims it.

    `_reindex` names every parent of a surviving op, so a later rebuild re-adds an
    edge to a dropped parent. That row is inert by construction (the only reader
    INNER JOINs both endpoints), and suppressing it would mean rewriting op
    bodies, which the wire format forbids. Pinned here so the regrowth is a known
    ceiling rather than a surprise.
    """
    first = store.put("global", fact())
    store.put("global", fact("Edited"), entity=first["id"], expected_version=first["version"])
    _compacted(store)

    with store.connection() as db:
        count = db.execute("SELECT count(*) FROM causal_edges").fetchone()[0]
        assert count == 0
        store._reindex(db, first["id"])
        assert db.execute("SELECT count(*) FROM causal_edges").fetchone()[0] == 1
        dangling = db.execute(
            "SELECT count(*) FROM causal_edges WHERE op_id NOT IN (SELECT op_id FROM ops) "
            "OR parent NOT IN (SELECT op_id FROM ops)"
        ).fetchone()[0]
        assert dangling == 1, "the regrown edge is inert (its parent is gone)"


def test_compaction_rewrites_the_session_export_so_nothing_resurrects(store):
    """A stale export re-applies dropped ops on the next session write (bytes back)."""
    first = store.put("session", fact())
    store.put("session", fact("Edited body"), entity=first["id"], expected_version=first["version"])
    export = store.session_dir / SESSION_EXPORT_NAME
    assert len(export.read_text().splitlines()) == 2

    _compacted(store)

    assert len(export.read_text().splitlines()) == 1
    before = store.inventory()["ops"]
    store.import_session()
    assert store.inventory()["ops"] == before


def test_compaction_preview_is_read_only_and_a_repeat_pass_is_a_no_op(store):
    first = store.put("global", fact())
    store.put("global", fact("Edited"), entity=first["id"], expected_version=first["version"])
    before = store.inventory()

    plan = store.journal_plan()
    assert plan["read_only"] is True
    assert plan["ops"]["dropped"] == 1
    assert plan["storage"]["bytes_after_estimate"] <= plan["storage"]["bytes_before"]
    untouched = store.inventory()
    assert untouched["ops"] == before["ops"]
    assert untouched["storage_bytes"] == before["storage_bytes"]

    with pytest.raises(MemoryError, match="fingerprint"):
        store.compact(expect="deadbeefcafe")
    assert store.inventory()["ops"] == before["ops"], "a refused pass deleted rows"

    store.compact(expect=plan["fingerprint"])
    assert store.inventory()["ops"] == plan["ops"]["kept"]
    second = store.journal_plan()
    assert second["ops"]["dropped"] == 0
    again = store.compact(expect=second["fingerprint"])
    assert again["vacuumed"] is False
    assert again["storage"]["reclaimed_bytes"] == 0
    assert store.inventory()["ops"] == plan["ops"]["kept"]


def test_compaction_preview_does_not_create_a_store(tmp_path):
    absent = MemoryStore(tmp_path / "cfg", MemoryIdentity("s", "", ""), tmp_path / "sessions" / "s")

    plan = absent.journal_plan()

    assert plan["exists"] is False
    assert plan["ops"] == {"before": 0, "kept": 0, "dropped": 0}
    assert not absent.path.exists()
    with pytest.raises(MemoryError):
        absent.compact(expect=plan["fingerprint"])


def test_cli_compact_previews_then_applies_the_previewed_fingerprint(tmp_path):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("cli-session", "", "")
    store = MemoryStore(root, identity, root / "sessions" / "cli-session", origin="cli-device")
    first = store.put("global", fact())
    store.put("global", fact("Edited"), entity=first["id"], expected_version=first["version"])

    preview, failed = run("compact", config=root, identity=identity, prefix="lop memory")
    plan = json.loads(preview)
    assert failed is False
    assert plan["action"] == "compact-preview" and plan["read_only"] is True
    assert plan["ops"]["dropped"] == 1
    assert f"--yes --expect {plan['fingerprint']}" in plan["confirm"]
    text = render_human(preview, failed, "lop memory")
    assert "Nothing was deleted" in text and "Would delete 1 operations" in text

    receipt, failed = run(
        f"compact --yes --expect {plan['fingerprint']}",
        config=root,
        identity=identity,
        prefix="lop memory",
    )
    applied = json.loads(receipt)
    assert failed is False
    assert applied["action"] == "compact" and applied["read_only"] is False
    assert applied["storage"]["bytes_after"] <= applied["storage"]["bytes_before"]
    assert "Deleted 1 operations" in render_human(receipt, failed, "lop memory")


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("compact --yes --expect deadbeefcafe", "fingerprint"),
        ("compact --yes", "needs the fingerprint from its preview"),
        ("compact --expect deadbeefcafe", "pass it with --yes"),
        ("compact --scope global", "does not narrow it"),
        ("compact some-id", "never targets a row"),
        ("list --expect deadbeefcafe", "--expect confirms a preview"),
        ("compact --force", "filter-based `forget` only"),
    ],
)
def test_cli_compact_refuses_a_confirmation_it_cannot_honour(tmp_path, command, message):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("cli-session", "", "")
    store = MemoryStore(root, identity, root / "sessions" / "cli-session", origin="cli-device")
    item = store.put("global", fact())
    store.put("global", fact("Edited"), entity=item["id"], expected_version=item["version"])
    before = store.inventory()["ops"]

    out, failed = run(command, config=root, identity=identity, prefix="lop memory")

    assert failed is True
    assert message in out
    assert store.inventory()["ops"] == before, f"`{command}` changed the journal"


def _live_ops(store: MemoryStore) -> int:
    """Ops count on a raw read-only connection (WAL readers do not block on a writer)."""
    db = sqlite3.connect(f"{store.path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return int(db.execute("SELECT count(*) FROM ops").fetchone()[0])
    finally:
        db.close()


def test_compaction_waits_for_a_concurrent_writer_and_deletes_nothing_mid_push(store):
    """The pass serialises against a push, rather than half-applying beside one.

    This is the reason the plan is recomputed INSIDE the ``BEGIN IMMEDIATE``
    transaction: a peer's push is the other writer, and it either lands wholly
    before the plan (and can invalidate the previewed fingerprint) or wholly
    after the deletes. ``busy_timeout=5000`` from ``connection()`` is what makes
    the pass wait instead of failing on the first lock refusal.
    """
    first = store.put("global", fact())
    store.put("global", fact("Edited"), entity=first["id"], expected_version=first["version"])
    plan = store.journal_plan()
    before = _live_ops(store)
    assert before == plan["ops"]["before"]

    holder = sqlite3.connect(store.path)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT INTO meta VALUES ('concurrent-writer','1')")
    errors: list[BaseException] = []

    def apply() -> None:
        try:
            store.compact(expect=plan["fingerprint"])
        except BaseException as exc:  # noqa: BLE001 - re-raised by the assertion below
            errors.append(exc)

    worker = threading.Thread(target=apply)
    worker.start()
    try:
        time.sleep(0.5)
        assert worker.is_alive(), "compaction did not wait for the writer"
        assert errors == []
        assert _live_ops(store) == before, "compaction deleted rows while a writer held the lock"
    finally:
        holder.commit()
        holder.close()
        worker.join(timeout=30)
    assert not worker.is_alive()
    assert errors == []
    assert _live_ops(store) == plan["ops"]["kept"]


def test_compaction_never_reissues_a_sequence_number_a_peer_already_holds(store):
    """A delete must not lower the counter a peer's cursor is compared against.

    With ``max(seq)+1`` the store re-issued ``(device-a, 1)`` for a NEW operation
    after a peer's tombstone made the seq-1 put deletable. A receiver seeded at
    cursor 1 then evaluates ``seq > 1`` as false and never learns the fact:
    silent, permanent divergence produced by an ordinary forget.

    Three layers are asserted here: allocation persisted the floor at write time
    (``origin-high-water``), the pass keeps each origin's newest op so the floor
    is readable from the journal, and the wire delivers the new op to a peer at
    the old cursor. The floor-alone case (row gone behind the store's back) is
    ``test_sequence_allocation_reads_the_persisted_floor_not_the_table``.
    """
    first = store.put("global", fact())
    with store.connection() as db:
        put = store._ops(db, first["id"])[0]
    assert put["seq"] == 1
    peer_tombstone = dict(
        put,
        op_id="peer-tombstone",
        origin_device="device-b",
        seq=1,
        hlc=[put["hlc"][0] + 1, 0, "device-b"],
        parents=[put["op_id"]],
        kind="tombstone",
        fact=None,
    )
    store.apply_ops([peer_tombstone], origins={"device-b"})
    receipt = _compacted(store)
    assert receipt["retained"]["tombstones"] == 1
    # The put survives BECAUSE it is device-a's newest op (the floor defence).
    assert first["version"] in {op["op_id"] for op in _all_ops(store)}

    fresh = store.put("global", fact("A wholly new fact"))
    with store.connection() as db:
        seq = db.execute("SELECT seq FROM ops WHERE op_id=?", (fresh["version"],)).fetchone()[0]
        floor = db.execute(
            "SELECT value FROM meta WHERE key=?",
            (MemoryStore.SEQ_HIGH_WATER + "device-a",),
        ).fetchone()[0]
    assert seq == 2, "the compacted store re-issued a sequence number"
    assert int(floor) >= 2

    from local_operator.network.memory import batch

    wire = batch(
        store,
        {"device-a": 1},
        ["global", "session", "team", "agent"],
        {"device-a"},
    )
    assert [op["op_id"] for op in wire["ops"]] == [
        fresh["version"]
    ], "a peer at cursor 1 never receives the new fact"


def test_sequence_allocation_reads_the_persisted_floor_not_the_table(store):
    """The meta floor is the guarantee; keeping the newest row is only backup.

    Simulates a store that already lost an origin's newest row (the state a pass
    without the floor leaves behind): allocation must still skip the number that
    row used, or the next operation reuses an identity a peer holds.
    """
    first = store.put("global", fact())
    with store.connection() as db:
        seq = store._ops(db, first["id"])[0]["seq"]
        db.execute("DELETE FROM ops WHERE op_id=?", (first["version"],))
        db.commit()

    second = store.put("global", fact("Written after the row was removed"))
    with store.connection() as db:
        assert store._ops(db, second["id"])[0]["seq"] == seq + 1


def test_compaction_does_not_promote_a_pinned_revision_into_a_phantom_head(store):
    """A pin must drag the chain above it, or `_heads` promotes it to a second head.

    ``replaces`` names the replaced entity's head as it was at write time. Two
    later edits to that entity make the pinned revision a non-head, and its only
    child is itself a non-head — so a retention set of "heads + pins" deletes the
    link between them and `show()` reports a conflict nobody made.
    """
    old = store.put("global", fact("Old release location"))
    replacement = store.put("global", fact("New release location", replaces=old["id"]))
    edit_one = store.put(
        "global",
        fact("Old location, first correction"),
        entity=old["id"],
        expected_version=old["version"],
    )
    store.put(
        "global",
        fact("Old location, second correction"),
        entity=old["id"],
        expected_version=edit_one["version"],
    )
    before = store.show("global", old["id"])
    assert before["conflicts"] == []
    assert before["revisions"] == 3

    receipt = _compacted(store)

    after = store.show("global", old["id"])
    assert after["conflicts"] == [], "compaction promoted a pinned revision into a head"
    assert receipt["retained"]["closure_ops"] == 1
    assert after["revisions"] == 3
    assert after["fact"] == before["fact"]
    # The replacement edge still hides the replaced fact: both endpoints survived.
    assert [row["id"] for row in store.tagged("release")] == [replacement["id"]]


def test_preview_creates_nothing_and_works_on_a_read_only_copy(store, tmp_path):
    """A preview that mutates is a trap on the full store it exists to rescue."""
    first = store.put("global", fact())
    store.put("global", fact("Edited"), entity=first["id"], expected_version=first["version"])
    with store.connection() as db:
        db.execute("DROP TABLE journal_prune")
        db.commit()

    plan = store.journal_plan()

    assert plan["last_pass"] is None
    assert plan["ops"]["dropped"] == 1
    assert _schema_objects(store, "journal_prune") == 0, "the preview created its audit table"

    copy = tmp_path / "read-only" / "memory"
    copy.mkdir(parents=True)
    shutil.copyfile(store.path, copy / "memory.db")
    (copy / "memory.db").chmod(0o444)
    frozen = MemoryStore(copy.parent, MemoryIdentity("session-a", "team-a", "agent-a"), None)
    assert frozen.journal_plan()["ops"]["dropped"] == 1
