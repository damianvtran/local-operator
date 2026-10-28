"""A peer must not be able to undo journal compaction by pushing the history back.

`apply_ops` dedups by op id, so an op `compact` deleted looks new to it. On the
10-24 store one peer re-sent 22,551 of the 25,192 dropped ops within ~100
minutes; 22,403 of them were puts under a tombstone. `MemoryStore._compacted_echo`
now drops exactly that shape, as an accepted no-op: a put for an entity with a
STORED tombstone.

The filter is only correct if dropping an op cannot change what any reader
sees, now or after a later op. Most tests here are about what it must NOT drop:

* a superseded revision of a LIVE entity: a later `replaces` edge or move
  tombstone may name it by op id (review round 1 on PR #3058, M1 and M2);
* a concurrent sibling stays a visible conflict;
* a dead put that may be one half of a cross-entity edge.

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
    MemoryError,
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
    superseded = {live["version"], second["version"]}

    with caplog.at_level(logging.INFO, logger="local_operator.memory"):
        _replay(store, history)

    held = {op["op_id"] for op in _all_ops(store)}
    assert held == {op["op_id"] for op in kept} | superseded, "a forgotten put came back"
    assert _state(store) == state
    assert "dropped_compacted=2 of" in caplog.text
    # Only the two superseded revisions of the live fact are left for the next pass.
    assert store.journal_plan()["ops"]["dropped"] == 2


def test_a_stale_session_export_does_not_resurrect_compacted_ops(store):
    """`import_session` goes through the same filter: a pre-compaction export is inert."""
    first = store.put("session", fact())
    edited = store.put(
        "session", fact("Edited body"), entity=first["id"], expected_version=first["version"]
    )
    store.put(
        "session", fact(), entity=first["id"], expected_version=edited["version"], forget=True
    )
    export = store.session_dir / SESSION_EXPORT_NAME
    stale = export.read_text()
    assert len(stale.splitlines()) == 3
    _compacted(store)
    kept = _all_ops(store)
    assert [op["kind"] for op in kept] == ["tombstone"]

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


def _push(source: MemoryStore, target: MemoryStore, vector, scopes=("global",)):
    """One real `network.memory.batch()` frame applied the way `receive()` applies it."""
    from local_operator.network.memory import batch

    origins = {"device-a", "device-b", "device-c", "device-z"}
    payload = batch(source, dict(vector), list(scopes), origins)
    target.apply_ops(payload["ops"] + payload["proofs"], origins=origins, contiguous=False)
    return payload["after"]


def _ids(store: MemoryStore) -> set[str]:
    return {op["op_id"] for op in _all_ops(store)}


def test_a_superseded_revision_a_later_replacement_names_is_kept(tmp_path):
    """Review round 1, M1: a `replaces` edge arriving AFTER the revision must still hide it.

    C relays A's ``v1``/``v2`` together with its own ``v3`` (``device-c`` sorts
    before ``device-z``, so ``v3`` lands first). A then replaces the fact while its
    head is still ``v2``. `_NOT_REPLACED_SQL` joins the edge's parent ROW, so if
    ``v2`` had been dropped as superseded, B would keep showing the replaced fact.
    """
    a = _store(tmp_path, "a", "device-z")
    c = _store(tmp_path, "c", "device-c")
    b = _store(tmp_path, "b", "device-b")
    v1 = a.put("global", fact("Deploy mercury on friday"))
    v2 = _edit(a, v1["id"], v1["version"], "Deploy mercury on saturday")
    a.put("global", fact("Unrelated later fact"))
    _push(a, c, {})
    _edit(c, v1["id"], v2["version"], "Deploy mercury on monday")
    _push(c, b, {})
    cursor_ab = _push(a, b, {})
    replacement = a.put("global", fact("Deploy mercury on sunday", replaces=v1["id"]))
    _push(a, b, cursor_ab)
    _push(c, a, {})

    assert {v1["version"], v2["version"]} <= _ids(b)
    assert sorted(h["id"] for h in b.search("mercury")) == [replacement["id"]]
    assert sorted(h["id"] for h in b.search("mercury")) == sorted(
        h["id"] for h in a.search("mercury")
    )


def test_a_move_naming_a_revision_this_peer_superseded_is_still_refused(tmp_path):
    """Review round 1, M2: the unseen-revision guard needs the mover's parent row.

    B syncs only ``global``, so the move's destination never arrives and the
    marker is "a second, unresolvable parent beside a covered one". If ``v2`` had
    been dropped, nothing is covered and the move erases C's correction on B only.
    """
    a = _store(tmp_path, "a", "device-z")
    b = _store(tmp_path, "b", "device-b")
    c = _store(tmp_path, "c", "device-c")
    v1 = a.put("global", fact("v1"))
    v2 = _edit(a, v1["id"], v1["version"], "v2")
    _push(a, c, {})
    v3 = _edit(c, v1["id"], v2["version"], "C's correction v3")
    _push(c, b, {})
    assert v2["version"] in _ids(b)
    a.move(v1["id"], "team", expected_version=v2["version"], scope="global")

    for peer in (b, c):
        with pytest.raises(MemoryError, match="move refused"):
            _push(a, peer, {"device-z": 2})
        assert peer.show("global", v1["id"])["version"] == v3["version"]


def test_a_forgotten_replacement_still_hides_its_target_whatever_the_arrival_order(tmp_path):
    """A dead put whose parent this store cannot place may be half of a `replaces` edge.

    A writes Y, then X replacing Y (X's parent is Y's head), then forgets X. Search
    keeps hiding Y on A: `_NOT_REPLACED_SQL` needs only the child ROW, and a
    forget redacts the put without deleting it. B gets X's tombstone first, then
    X's put before Y's head. The put is redacted, so its `replaces` is
    unreadable and only the unplaceable parent says it may be an edge; dropping it
    leaves Y visible on B alone.
    """
    a = _store(tmp_path, "a", "device-a")
    b = _store(tmp_path, "b", "device-b")
    target = a.put("global", fact("Deploy mercury on friday"))
    replacement = a.put("global", fact("Deploy mercury on sunday", replaces=target["id"]))
    a.put(
        "global",
        fact(),
        entity=replacement["id"],
        expected_version=replacement["version"],
        forget=True,
    )
    history = {op["op_id"]: op for op in _all_ops(a)}
    tomb = next(op for op in history.values() if op["kind"] == "tombstone")
    put = history[replacement["version"]]
    origins = {"device-a"}
    b.apply_ops([tomb], origins=origins, contiguous=False)
    b.apply_ops([put], origins=origins, contiguous=False)
    b.apply_ops([history[target["version"]]], origins=origins, contiguous=False)

    assert replacement["version"] in _ids(b)
    assert a.search("mercury") == []
    assert b.search("mercury") == a.search("mercury")


@pytest.mark.parametrize("order", ["move_first", "same_frame", "destination_last"])
def test_a_moved_facts_source_revision_arriving_after_the_move_is_admitted(tmp_path, order):
    """`show().moved_from` reads the source revision through the destination's parent edge.

    Z writes the fact; C moves it. B receives C's source tombstone before Z's put,
    so the put lands on an entity with a STORED tombstone. It is still half of a
    live cross-entity edge, so it must be stored or B loses the "moved from" link.
    Each order is caught by a different check: the stored edge (``move_first``),
    the in-batch child (``same_frame``), or only the move-shaped tombstone when
    the destination has not arrived yet (``destination_last``).
    """
    z = _store(tmp_path, "z", "device-z")
    c = _store(tmp_path, "c", "device-c")
    b = _store(tmp_path, "b", "device-b")
    scopes = ("global", "team")
    source = z.put("global", fact("Moved fact"))
    _push(z, c, {}, scopes)
    receipt = c.move(source["id"], "team", expected_version=source["version"], scope="global")
    destination = receipt["moved"][0]["to"].split("/", 1)[1]
    if order == "same_frame":
        _push(c, b, {}, scopes)
    elif order == "move_first":
        _push(c, b, {"device-z": 1}, scopes)
        _push(z, b, {}, scopes)
    else:
        # B syncs only `global` first: the tombstone arrives, the destination does not.
        _push(c, b, {"device-z": 1}, ("global",))
        _push(z, b, {}, ("global",))
        assert source["version"] in _ids(b)
        _push(c, b, {"device-z": 1}, scopes)

    assert source["version"] in _ids(b)
    assert b.show("team", destination)["moved_from"] == c.show("team", destination)["moved_from"]
    assert b.show("team", destination)["moved_from"] == {"scope": "global", "id": source["id"]}


def test_an_echo_through_the_real_receive_path_stays_compacted(tmp_path, monkeypatch):
    """Review round 1, m1: the property the PR exists for, over `batch()` -> `receive()`.

    The peer holds the full history; the receiver compacted. Every frame goes
    through the receiver's real `receive()` (vector validation, cursor commit) and
    must leave the journal compacted, the cursor at the sender's vector, and
    `stream-generation` untouched (a bump makes every peer resend everything).
    """
    from types import SimpleNamespace

    from local_operator.network import memory as network_memory

    receiver = _store(tmp_path, "receiver", "device-a")
    # Live facts carry one revision: a re-admitted superseded revision is genuine
    # gap-filling under the current rule and bumps the generation, which would
    # hide whether the DROPPED ops did.
    for index in range(30):
        item = receiver.put("global", fact(f"Fact {index} at docs/{index}.md"))
        if index % 3:
            edited = _edit(receiver, item["id"], item["version"], f"Edited {index}")
            receiver.put(
                "global",
                fact(),
                entity=item["id"],
                expected_version=edited["version"],
                forget=True,
            )
    peer = _store(tmp_path, "peer", "device-b")
    origins, scopes = {"device-a", "device-b"}, ["global"]
    vector: dict[str, int] = {}
    while True:
        payload = network_memory.batch(receiver, vector, scopes, origins)
        if payload["after"] == vector:
            break
        peer.apply_ops(payload["ops"] + payload["proofs"], origins=origins, contiguous=False)
        vector = payload["after"]
    _compacted(receiver)
    kept = _ids(receiver)
    with receiver.connection() as db:
        generation = db.execute("SELECT value FROM meta WHERE key='stream-generation'").fetchone()
    monkeypatch.setattr(network_memory, "_store", lambda root: receiver)
    server = SimpleNamespace(root=receiver.root.parent)

    frames, cursor = 0, network_memory.cursor(receiver, "device-b", scopes)
    while True:
        payload = network_memory.batch(peer, cursor, scopes, origins)
        if payload["after"] == cursor:
            break
        network_memory.receive(server, "device-b", payload, scopes, origins)
        cursor, frames = network_memory.cursor(receiver, "device-b", scopes), frames + 1

    assert frames >= 1
    assert cursor == {"device-a": 70}  # 30 puts + 20 edits + 20 forgets
    assert _ids(receiver) == kept, "40 forgotten puts were pushed back and re-admitted"
    with receiver.connection() as db:
        assert (
            db.execute("SELECT value FROM meta WHERE key='stream-generation'").fetchone()
            == generation
        )


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
    # Nothing left dropped under a tombstone came back. Superseded revisions of
    # live facts may (a later op can name them; see the M1/M2 tests).
    # Only ids[3] was forgotten on left when it compacted. ids[5]'s tombstone came
    # from right in the SAME batch as the echo, which deliberately does not count.
    back = [op for op in _all_ops(left) if op["op_id"] in dropped]
    assert back, "fixture no longer exercises the echo of superseded revisions"
    assert all(op["entity_id"] != ids[3] for op in back)
