"""A peer must not be able to undo journal compaction by pushing the history back.

`apply_ops` dedups by op id, so an op `compact` deleted looks new to it. On the
10-24 store one peer re-sent 22,551 of the 25,192 dropped ops within ~100
minutes. `MemoryStore._compacted_echo` now drops, as accepted no-ops, the two
shapes compaction removes: puts under a stored tombstone, and revisions that
already have a stored descendant.

The filter is only correct if dropping an op cannot change what any reader
sees, so most tests here are about what it must NOT drop:

* a concurrent sibling (no stored descendant) stays a visible conflict;
* an op whose drop would turn a stored parent into a fabricated head;
* an op that is the endpoint of a cross-entity `replaces` edge, which
  `search` reads to hide the replaced fact.

Plus the bookkeeping: a dropped op still raises its origin's sequence floor,
so neither a peer's next contiguous op nor a local allocation is refused or
re-issued.
"""

import hashlib
import json
import logging
from typing import Any

import pytest

from local_operator.memory import (
    SESSION_EXPORT_NAME,
    MemoryIdentity,
    MemoryStore,
    _dump,
)


def fact(body: str = "Release evidence lives at docs/release.md", **kwargs: Any) -> dict[str, Any]:
    return dict(title="Release evidence", body=body, tags=["release", "evidence"], **kwargs)


def _store(tmp_path, name: str, origin: str) -> MemoryStore:
    return MemoryStore(
        tmp_path / name / "cfg",
        MemoryIdentity("session-a", "team-a", "agent-a"),
        tmp_path / name / "sessions" / "session-a",
        origin=origin,
    )


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path, "a", "device-a")


def _all_ops(store: MemoryStore) -> list[dict[str, Any]]:
    with store.connection() as db:
        return [json.loads(row[0]) for row in db.execute("SELECT data FROM ops ORDER BY op_id")]


def _compacted(store: MemoryStore) -> dict[str, Any]:
    plan = store.journal_plan()
    return store.compact(expect=plan["fingerprint"])


def _state(store: MemoryStore) -> dict[str, Any]:
    """Everything a reader can observe: `current` plus each entity's head set or death."""
    with store.connection() as db:
        current = [
            tuple(row)
            for row in db.execute(
                "SELECT entity_id,scope,owner,op_id,fact FROM current ORDER BY entity_id"
            )
        ]
        entities = [row[0] for row in db.execute("SELECT DISTINCT entity_id FROM ops")]
        heads = {}
        for entity in entities:
            ops = store._ops(db, entity)
            if any(op["kind"] == "tombstone" for op in ops):
                heads[entity] = "forgotten"
            else:
                heads[entity] = sorted(op["op_id"] for op in store._heads(ops))
    return {"current": current, "heads": heads}


def _floor(store: MemoryStore, origin: str) -> int:
    with store.connection() as db:
        return MemoryStore._seq_high_water(db, origin)


def _edit(store: MemoryStore, entity: str, version: str, body: str) -> dict[str, Any]:
    return store.put("global", fact(body), entity=entity, expected_version=version)


def _peer_put(base: dict[str, Any], op_id: str, seq: int, parents: list[str], body: str):
    """A put minted by ``device-b`` for ``base``'s entity (a peer's edit)."""
    op = dict(
        base,
        op_id=op_id,
        origin_device="device-b",
        seq=seq,
        hlc=[base["hlc"][0] + seq, 0, "device-b"],
        parents=parents,
    )
    op["fact"] = dict(base["fact"], body=body)
    op["body_sha"] = hashlib.sha256(_dump(op["fact"]).encode()).hexdigest()
    return op


def _replay(store: MemoryStore, history: list[dict[str, Any]]) -> int:
    """Push ``history`` back the way `network/memory.receive` does: non-contiguous batches.

    Puts under a tombstone are already redacted in a journal (``_reindex``), so
    the rows are sent as stored, exactly as `network/memory.batch` reads them.
    """
    ordered = sorted(history, key=lambda op: (op["origin_device"], op["seq"]))
    return sum(
        store.apply_ops(ordered[start : start + 100], contiguous=False)
        for start in range(0, len(ordered), 100)
    )


def test_compacted_history_pushed_back_by_a_peer_is_not_readmitted(store, caplog):
    """THE DEFECT: compact, replay what was dropped, and every op used to come back."""
    live = store.put("global", fact())
    second = _edit(store, live["id"], live["version"], "Second body")
    _edit(store, live["id"], second["version"], "Third body")
    gone = store.put("global", fact("Forgotten body"))
    gone2 = _edit(store, gone["id"], gone["version"], "Forgotten edit")
    store.put(
        "global", fact("x"), entity=gone["id"], expected_version=gone2["version"], forget=True
    )
    history = _all_ops(store)
    receipt = _compacted(store)
    assert receipt["ops"]["dropped"] == 4  # two superseded, two under a tombstone
    kept, state = _all_ops(store), _state(store)

    with caplog.at_level(logging.INFO, logger="local_operator.memory"):
        changed = _replay(store, history)

    assert changed == 0
    assert _all_ops(store) == kept, "a peer's echo undid the compaction"
    assert _state(store) == state
    assert "dropped_compacted=4 (forgotten=2 superseded=2)" in caplog.text
    assert store.journal_plan()["ops"]["dropped"] == 0


def test_a_stale_session_export_does_not_resurrect_compacted_ops(store):
    """`import_session` goes through the same filter: a pre-compaction export is inert."""
    first = store.put("session", fact())
    _edit_session = store.put(
        "session", fact("Edited body"), entity=first["id"], expected_version=first["version"]
    )
    export = store.session_dir / SESSION_EXPORT_NAME
    stale = export.read_text()
    assert len(stale.splitlines()) == 2
    _compacted(store)
    kept = _all_ops(store)
    assert [op["op_id"] for op in kept] == [_edit_session["version"]]

    export.write_text(stale)  # what a crash between delete and export rewrite leaves
    store.import_session()

    assert _all_ops(store) == kept
    assert len(export.read_text().splitlines()) == 1


def test_a_concurrent_peer_edit_is_still_admitted_and_shown_as_a_conflict(store):
    """No stored descendant means it could be a head: it must land and stay visible.

    Parented on a revision compaction already removed, which is the shape most
    likely to be mistaken for an echo.
    """
    first = store.put("global", fact())
    second = _edit(store, first["id"], first["version"], "Local correction")
    with store.connection() as db:
        base = store._ops(db, first["id"])[0]
    _compacted(store)
    assert first["version"] not in {op["op_id"] for op in _all_ops(store)}

    sibling = _peer_put(base, "peer-sibling", 1, [first["version"]], "Peer's correction")
    assert store.apply_ops([sibling], origins={"device-b"}, contiguous=False) == 1

    view = store.show("global", first["id"])
    versions = {view["version"], *(c["version"] for c in view["conflicts"])}
    assert versions == {second["version"], "peer-sibling"}
    assert len(view["conflicts"]) == 1


def test_an_op_whose_drop_would_turn_a_stored_parent_into_a_head_is_admitted(store):
    """Stored descendant AND a stored parent it alone covers: dropping it forks the entity.

    Arrival order across origins makes this real: device-c's child of device-b's
    edit lands before the edit itself. Until the edit arrives the entity shows
    a conflict; dropping it as "superseded" would leave that conflict forever.
    """
    first = store.put("global", fact())
    with store.connection() as db:
        base = store._ops(db, first["id"])[0]
    edit = _peer_put(base, "peer-edit", 1, [first["version"]], "Peer's edit")
    child = dict(
        _peer_put(base, "third-edit", 1, ["peer-edit"], "Third device's follow-up"),
        origin_device="device-c",
        hlc=[base["hlc"][0] + 9, 0, "device-c"],
    )
    store.apply_ops([child], origins={"device-c"}, contiguous=False)
    assert len(store.show("global", first["id"])["conflicts"]) == 1

    assert store.apply_ops([edit], origins={"device-b"}, contiguous=False) == 1

    view = store.show("global", first["id"])
    assert view["version"] == "third-edit"
    assert view["conflicts"] == []


def test_an_op_with_an_unknown_parent_is_admitted_so_that_parent_cannot_fork_later(store):
    """An unknown parent may arrive in a later batch; only this op can bury it."""
    first = store.put("global", fact())
    with store.connection() as db:
        base = store._ops(db, first["id"])[0]
    late = _peer_put(base, "late-parent", 1, [first["version"]], "Late parent")
    middle = _peer_put(base, "middle", 2, ["late-parent"], "Middle")
    tip = _peer_put(base, "tip", 3, ["middle"], "Tip")
    store.apply_ops([tip], origins={"device-b"}, contiguous=False)
    store.apply_ops([middle], origins={"device-b"}, contiguous=False)
    store.apply_ops([late], origins={"device-b"}, contiguous=False)

    view = store.show("global", first["id"])
    assert view["version"] == "tip"
    assert view["conflicts"] == []


def test_the_endpoint_of_a_replacement_edge_is_never_dropped(tmp_path, store):
    """`search` hides a replaced fact by joining the edge's PARENT row; it must exist.

    ``v2`` is a superseded revision of the target with a stored descendant, so
    without the cross-entity check it would be dropped as an echo and the
    replaced fact would reappear in search on this peer only.
    """
    target = store.put("global", fact("Deploy mercury on friday"))
    v2 = _edit(store, target["id"], target["version"], "Deploy mercury on saturday")
    store.put("global", fact("Deploy mercury on sunday", replaces=target["id"]))
    _edit(store, target["id"], v2["version"], "Deploy mercury on monday")
    assert len(store.search("mercury")) == 1
    history = sorted(_all_ops(store), key=lambda op: op["seq"])
    assert [op["op_id"] for op in history][:2] == [target["version"], v2["version"]]

    peer = _store(tmp_path, "b", "device-b")
    # Gap fill: the replacement and the latest edit arrive before the two old revisions.
    peer.apply_ops(history[2:], origins={"device-a"}, contiguous=False)
    peer.apply_ops(history[:2], origins={"device-a"}, contiguous=False)

    assert v2["version"] in {op["op_id"] for op in _all_ops(peer)}
    assert [hit["id"] for hit in peer.search("mercury")] == [
        hit["id"] for hit in store.search("mercury")
    ]


def test_a_dropped_op_still_raises_its_origin_floor_and_leaves_no_gap(store):
    """Accepted, not stored: the seq is evidence of the counter, exactly as for an insert."""
    first = store.put("global", fact())
    with store.connection() as db:
        base = store._ops(db, first["id"])[0]
    store.put("global", fact(), entity=first["id"], expected_version=first["version"], forget=True)
    with store.connection() as db:
        generation = db.execute("SELECT value FROM meta WHERE key='stream-generation'").fetchone()
    # A peer's edit the local forget beat: a put under a stored tombstone, redacted
    # by the sender beside its proof. Seq 1 and 2 arrive, seq 3 is a new fact.
    beaten = [
        dict(_peer_put(base, f"beaten-{seq}", seq, [first["version"]], "x"), fact=None)
        for seq in (1, 2)
    ]
    assert store.apply_ops(beaten, origins={"device-b"}) == 0
    assert "beaten-1" not in {op["op_id"] for op in _all_ops(store)}
    assert _floor(store, "device-b") == 2

    other = store.put("global", fact("Another fact"))
    with store.connection() as db:
        other_base = store._ops(db, other["id"])[0]
    fresh = _peer_put(other_base, "fresh", 3, [other["version"]], "A legitimate edit")
    # contiguous=True: refused as a gap if the dropped seqs had not counted.
    assert store.apply_ops([fresh], origins={"device-b"}) == 1
    assert store.show("global", other["id"])["version"] == "fresh"
    with store.connection() as db:
        assert (
            db.execute("SELECT value FROM meta WHERE key='stream-generation'").fetchone()
            == generation
        ), "a dropped op is not new history; peers must not be made to resend everything"

    # The local allocator is above every floor, including one only a dropped echo raised.
    floor = _floor(store, "device-a")
    with store.connection() as db:
        own = store._ops(db, first["id"])[0]
    echo = dict(own, op_id="echo-above-floor", seq=floor + 5, fact=None)
    store.apply_ops([echo], origins={"device-a"}, contiguous=False)
    assert "echo-above-floor" not in {op["op_id"] for op in _all_ops(store)}
    later = store.put("global", fact("Written after the echo"))
    with store.connection() as db:
        seq = db.execute("SELECT seq FROM ops WHERE op_id=?", (later["version"],)).fetchone()[0]
    assert seq == floor + 6


def _sync(source: MemoryStore, target: MemoryStore) -> None:
    """Everything ``source`` holds, pushed to ``target`` in seq order (a full catch-up)."""
    history = _all_ops(source)
    origins = {op["origin_device"] for op in history}
    ordered = sorted(history, key=lambda op: (op["origin_device"], op["seq"]))
    for start in range(0, len(ordered), 50):
        target.apply_ops(ordered[start : start + 50], origins=origins, contiguous=False)


def test_two_stores_converge_through_edits_forgets_conflicts_and_compaction(tmp_path):
    """Compaction on one side plus the other side's echo must still meet in one state."""
    left = _store(tmp_path, "left", "device-a")
    right = _store(tmp_path, "right", "device-b")
    facts = [left.put("global", fact(f"Fact {index} at docs/{index}.md")) for index in range(6)]
    versions = {item["id"]: item["version"] for item in facts}
    for item in facts:
        versions[item["id"]] = _edit(left, item["id"], item["version"], "Left edit 1")["version"]
    _sync(left, right)
    ids = [item["id"] for item in facts]

    # Concurrent edits of the same heads on both sides: genuine conflicts.
    for entity in ids[:3]:
        _edit(left, entity, versions[entity], "Left concurrent")
        _edit(right, entity, versions[entity], "Right concurrent")
    # A forget on one side racing an edit on the other.
    _edit(right, ids[3], versions[ids[3]], "Right edit the forget beats")
    left.put("global", fact(), entity=ids[3], expected_version=versions[ids[3]], forget=True)
    # A chain of edits only one side made, then compacted away there.
    head = versions[ids[4]]
    for step in range(3):
        head = _edit(left, ids[4], head, f"Left chain {step}")["version"]
    right.put("global", fact(), entity=ids[5], expected_version=versions[ids[5]], forget=True)
    # Right catches up on left's chain BEFORE left compacts. Compacting an
    # intermediate revision a peer never received breaks that peer's ancestry
    # (a fabricated conflict on both sides, with or without this filter), which
    # is a compaction-scheduling property, not an ingress one.
    _sync(left, right)

    held = {op["op_id"] for op in _all_ops(left)}
    _compacted(left)
    dropped = held - {op["op_id"] for op in _all_ops(left)}
    assert len(dropped) >= 12  # the superseded edits and chain plus the forgotten puts
    # Right still holds everything left just dropped: the echo. Right's concurrent
    # edits and its forget arrive in the same stream and must still land.
    _sync(right, left)
    _sync(left, right)
    _sync(right, left)

    assert _state(left) == _state(right)
    state = _state(left)
    assert state["heads"][ids[3]] == "forgotten"
    assert state["heads"][ids[5]] == "forgotten"
    for entity in ids[:3]:
        assert len(state["heads"][entity]) == 2, "a concurrent edit stopped being a conflict"
        assert len(left.show("global", entity)["conflicts"]) == 1
    # And left really stayed compacted: nothing it dropped came back.
    assert dropped.isdisjoint(op["op_id"] for op in _all_ops(left))
