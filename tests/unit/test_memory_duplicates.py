"""``consolidate``: lexical duplicate preview, and the promise that it changes nothing.

The detector is exercised through the shipped human runner (``memory_cli.run``) for
the read-only guarantee, and directly for grouping edges that the renderer would
hide (scope separation, limits, evidence samples).
"""

import json
import sqlite3

import pytest

from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.memory_cli import render_human, run
from local_operator.memory_duplicates import duplicates

UNRELATED = [
    "Dentist appointment moved to Thursday",
    "Passport renewal needs two photographs",
    "Sourdough starter lives in the blue jar",
    "Landlord phone number is in the desk drawer",
]


@pytest.fixture
def rig(tmp_path):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("cli-session", "", "")
    session_dir = root / "sessions" / "cli-session"
    store = MemoryStore(root, identity, session_dir, origin="cli-device")
    return root, identity, session_dir, store


def human(text, root, identity, session_dir=None, prefix="/memory"):
    out, failed = run(text, config=root, identity=identity, session_dir=session_dir, prefix=prefix)
    return render_human(out, failed, prefix), failed


def payload(text, root, identity, session_dir=None):
    out, failed = run(text, config=root, identity=identity, session_dir=session_dir)
    assert not failed, out
    return json.loads(out)


def logical_dump(store):
    """Every durable row. A file hash cannot see a write still living in WAL."""
    db = sqlite3.connect(store.path)
    try:
        ops = db.execute(
            "SELECT op_id,entity_id,origin,seq,scope,owner,kind,data FROM ops ORDER BY op_id"
        ).fetchall()
        current = db.execute(
            "SELECT entity_id,scope,owner,op_id,fact,tokens FROM current ORDER BY entity_id"
        ).fetchall()
        meta = db.execute("SELECT key,value FROM meta ORDER BY key").fetchall()
        recall = db.execute(
            "SELECT entity_id,title,tags,body FROM recall ORDER BY entity_id"
        ).fetchall()
    finally:
        db.close()
    return ops, current, meta, recall


def snapshot(store, items):
    return [
        (store.show("global", item["id"])["version"], store.show("global", item["id"])["status"])
        for item in items
    ]


def test_exact_duplicate_pair_is_grouped_and_unrelated_facts_are_not(rig):
    root, identity, session_dir, store = rig
    # Two rows with the same text but distinct provenance: an exact pair, not the
    # store's idempotent re-store (which returns the SAME entity for identical facts).
    first = store.put(
        "global",
        {
            "title": "Wifi password",
            "body": "The guest wifi is in the kitchen drawer",
            "source": {"author": "user"},
        },
    )
    second = store.put(
        "global",
        {
            "title": "Wifi password",
            "body": "The guest wifi is in the kitchen drawer",
            "source": {"author": "agent"},
        },
    )
    unrelated = [store.put("global", {"title": title, "body": title}) for title in UNRELATED]

    preview = payload("consolidate --scope global", root, identity, session_dir)

    assert preview["action"] == "consolidate" and preview["read_only"] is True
    assert preview["fact_count"] == 6
    assert len(preview["groups"]) == 1
    group = preview["groups"][0]
    assert {member["id"] for member in group["members"]} == {first["id"], second["id"]}
    assert group["exact"] is True and group["similarity"] == 1.0
    pair = group["pairs"][0]
    assert pair["exact"] is True
    assert pair["shared"], "an exact pair must show the shared evidence, not an empty sample"
    assert not pair["a_only"] and not pair["b_only"]
    ids = {row["id"] for row in unrelated}
    assert not ids & {member["id"] for member in group["members"]}


def test_near_duplicates_group_with_a_measure_and_what_differs(rig):
    root, identity, session_dir, store = rig
    left = store.put("global", {"title": "Staging host", "body": "Staging host is stage-2"})
    right = store.put("global", {"title": "Staging host", "body": "Staging host is stage-2 now"})
    store.put("global", {"title": "Dentist appointment", "body": "Thursday at 10"})

    preview = payload("consolidate --scope global", root, identity, session_dir)

    assert len(preview["groups"]) == 1
    group = preview["groups"][0]
    assert {member["id"] for member in group["members"]} == {left["id"], right["id"]}
    assert group["exact"] is False
    pair = group["pairs"][0]
    assert 0.7 <= pair["similarity"] < 1.0
    assert pair["title_equal"] is True
    assert "now" in pair["a_only"] + pair["b_only"]
    assert "staging" in pair["shared"] or "host" in pair["shared"]

    # The guide promises shared tokens in the human view; keep the promise real.
    text, failed = human("consolidate --scope global", root, identity, session_dir)
    assert not failed and "shared:" in text


def test_punctuation_only_titles_do_not_form_a_group(rig):
    root, identity, session_dir, store = rig
    # Review round 1 measured a 254-member "identical" group of punctuation-only
    # titles: their normalised key is empty, and two empty token sets score 1.0.
    for title in ("...", "!!!", "??"):
        store.put(
            "global",
            {"title": title, "body": "...", "source": {"author": "user"}},
        )

    preview = payload("consolidate --scope global", root, identity, session_dir)

    assert preview["groups"] == []


def test_groups_do_not_cross_scopes(rig):
    root, identity, session_dir, store = rig
    store.put("global", {"title": "Wifi password", "body": "The guest wifi is in the drawer"})
    store.put("session", {"title": "Wifi password", "body": "The guest wifi is in the drawer"})

    preview = payload("consolidate", root, identity, session_dir)

    assert preview["groups"] == []
    assert preview["fact_count"] == 2


def test_output_is_bounded_by_limit(rig):
    root, identity, session_dir, store = rig
    # Distinct texts per group: identical bodies across groups would chain into one
    # transitive group and stop testing the limit at all.
    for title, body in (
        ("Wifi password", "Guest wifi code is 4411"),
        ("Dentist appointment", "Dentist on Thursday at ten"),
        ("Landlord phone", "Landlord number is in the desk drawer"),
    ):
        store.put("global", {"title": title, "body": body, "source": {"author": "user"}})
        store.put("global", {"title": title, "body": body, "source": {"author": "agent"}})

    preview = payload("consolidate --scope global --limit 2", root, identity, session_dir)

    assert preview["group_count"] == 3
    assert preview["limited"] is True
    assert len(preview["groups"]) == 2
    text, failed = human("consolidate --scope global --limit 2", root, identity, session_dir)
    assert not failed
    assert "3 duplicate group(s) found · 2 shown (limit 2)" in text
    assert "raise --limit to view them" in text


def test_consolidate_is_read_only_end_to_end(rig):
    root, identity, session_dir, store = rig
    items = [
        store.put(
            "global",
            {
                "title": "Wifi password",
                "body": "The guest wifi is in the drawer",
                "source": {"author": "user"},
            },
        ),
        store.put(
            "global",
            {
                "title": "Wifi password",
                "body": "The guest wifi is in the drawer",
                "source": {"author": "agent"},
            },
        ),
        store.put("global", {"title": "Dentist appointment", "body": "Thursday at 10"}),
    ]
    before_rows = logical_dump(store)
    before = snapshot(store, items)

    text, failed = human(
        "consolidate --scope global", root, identity, session_dir, prefix="lop memory"
    )

    assert not failed
    assert "Read-only duplicate preview" in text
    assert "Nothing was changed" in text
    assert logical_dump(store) == before_rows
    assert snapshot(store, items) == before
    # The follow-up it names is the existing human verbs, never an implied merge.
    assert "lop memory show" in text and "lop memory forget" in text
    assert "edit" in text


def test_empty_store_preview_changes_nothing(tmp_path):
    identity = MemoryIdentity()
    root = tmp_path / "home" / ".local-operator"
    text, failed = human("consolidate", root, identity)
    assert not failed
    assert "No identical or near-duplicate candidates" in text
    assert not (root / "memory").exists(), "a read-only preview must not create a store"


def test_duplicates_groups_exact_and_similar_transitively():
    rows = [
        {
            "id": "a",
            "scope": "global",
            "version": "va",
            "title": "Port",
            "body": "Port is 80",
            "tokens": 3,
            "status": "active",
        },
        {
            "id": "b",
            "scope": "global",
            "version": "vb",
            "title": "Port",
            "body": "Port is 80",
            "tokens": 3,
            "status": "active",
        },
        {
            "id": "c",
            "scope": "global",
            "version": "vc",
            "title": "Port",
            "body": "Port is 80 now",
            "tokens": 4,
            "status": "active",
        },
        {
            "id": "d",
            "scope": "global",
            "version": "vd",
            "title": "Landlord",
            "body": "Desk drawer",
            "tokens": 3,
            "status": "active",
        },
    ]
    preview = duplicates(rows)
    assert preview["group_count"] == 1
    assert {member["id"] for member in preview["groups"][0]["members"]} == {"a", "b", "c"}
