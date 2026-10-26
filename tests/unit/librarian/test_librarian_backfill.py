"""The Librarian's one-shot edit re-attribution, driven through its real path.

``local_operator/librarian/backfill.py`` exists because four days of her writes
were recorded as an ordinary agent's before the ``library`` tool derived her
actor from the caller's session. These tests drive the real function against a
real store and a real journal ON DISK: the predicate is about what a serialized
tool call contains, so an in-memory stand-in would test the wrong thing.

The journal shapes are copied from the real one (a native ``library`` op, and
eval-bridge cells whose code names her assets), because the two put a value in
different escaping contexts — which is the only reason the match is made against
the escaped form.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from local_operator.librarian import backfill, state
from local_operator.library.metadata import MetadataStore, reset_schema_readiness
from local_operator.session import compression

LIB = "f3cce3edac6a"
A = "la_07d83ff15563c0a2"
B = "la_c5d8fc4a5f0a862f"
C = "la_c12371dbc4be1b46"
D = "la_b5589e87d25cd6ee"
E = "la_f68836cae7c1647f"
T = "la_238504aacc9f8894"

NAME_A = "Hyundai POC News re-screen runner"
DESC_A = "Paced, self-guarding Python runner that re-screens News."
NAME_B_BEFORE = "ENG-3708 weekly"
NAME_B_AFTER = "ENG-3708 weekly rollup"
NAME_C = "Operator's own title"
NAME_D = "Locked asset"
NAME_T = "Hyundai POC batch runner"


def journal(root: Path, calls: list[dict[str, Any]], *, session_id: str = LIB) -> Path:
    """Write a journal the way the runtime does: assistant rows carrying calls."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"a{index}",
            "ts": float(index),
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [],
                "tool_calls": [call],
            },
        }
        for index, call in enumerate(calls)
    ]
    path = directory / "transcript.jsonl"
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def her_calls(tag_id: str) -> list[dict[str, Any]]:
    """The three call shapes her journal held, with the values she wrote.

    ``tag_id`` is spelled out inside the cell, the way her own cells held it
    (``sets={asset: [id]}``, ids taken from a ``tags list`` call): the pass
    matches a tag assignment by that id, so a journal that only named the asset
    would prove nothing about which set she applied.
    """
    return [
        {
            "id": "toolu_1",
            "name": "library",
            "arguments": {
                "op": "update",
                "asset_id": A,
                "name": NAME_A,
                "description": DESC_A,
            },
        },
        {
            "id": "toolu_2",
            "name": "eval",
            "arguments": {
                "code": (
                    f'upd=[("{B}","{NAME_B_BEFORE}"),("{C}","{NAME_C}"),("{D}","{NAME_D}"),'
                    f'("{E}","something she never wrote")]\n'
                    "for a,n in upd:\n"
                    '    r=tool("library",op="update",asset_id=a,name=n)\n'
                )
            },
        },
        {
            "id": "toolu_3",
            "name": "eval",
            "arguments": {
                "code": (
                    f'upd=[("{T}","{NAME_T}")]\n'
                    "for a,n in upd:\n"
                    '    r=tool("library",op="update",asset_id=a,name=n)\n'
                    f'sets={{"{T}":["{tag_id}"]}}\n'
                    "for a,t in sets.items():\n"
                    '    r=tool("library",op="tags",action="set",asset_id=a,tags=t,primary=t[0])\n'
                )
            },
        },
    ]


def her_tag_id(store: MetadataStore) -> str:
    """The id the registry minted for ``hyundai-poc`` — what her cell passed."""
    return next(tag["id"] for tag in store.list_tags() if tag["name"] == "hyundai-poc")


def seed(root: Path) -> MetadataStore:
    """One store holding every case the predicate has to tell apart.

    The process-wide schema-readiness cache is cleared first: pytest hands two
    tests the same ``tmp_path`` when their names truncate to the same 30
    characters, and a recycled directory holds no schema even though the cache
    still claims one (measured here: a 0-byte store and "no such table").
    """
    reset_schema_readiness()
    store = MetadataStore(root)
    # A: her work, still standing — both fields, and no tag edit, so the
    # timestamp it moves to is exactly the one its audit rows carry.
    store.ensure_asset(A, "p:/a", derived_name="a.png")
    store.update_fields(A, {"name": NAME_A}, actor="agent", at=100.0)
    store.update_fields(A, {"description": DESC_A}, actor="agent", at=200.0)
    # B: she named it, an ordinary agent renamed it afterwards.
    store.ensure_asset(B, "p:/b", derived_name="b.png")
    store.update_fields(B, {"name": NAME_B_BEFORE}, actor="agent", at=300.0)
    store.update_fields(B, {"name": NAME_B_AFTER}, actor="agent", at=400.0)
    # C: the operator's own title; she named the same asset in her journal.
    store.ensure_asset(C, "p:/c", derived_name="c.png")
    store.update_fields(C, {"name": NAME_C}, actor="user", at=500.0)
    # D: locked by the operator, so no pass may move it.
    store.ensure_asset(D, "p:/d", derived_name="d.png")
    store.update_fields(D, {"name": NAME_D}, actor="agent", at=600.0)
    store.update_fields(D, {"locked": True}, actor="user", at=700.0)
    # E: her call names the id but never held the value that stands on the row.
    store.ensure_asset(E, "p:/e", derived_name="e.png")
    store.update_fields(E, {"name": "an agent's own title"}, actor="agent", at=800.0)
    # T: tagged by her, in a call that also names the asset and its title.
    store.ensure_asset(T, "p:/t", derived_name="t.png")
    store.update_fields(T, {"name": NAME_T}, actor="agent")
    tag = store.create_tag("hyundai-poc")
    store.set_asset_tags(T, [tag["id"]], actor="agent")
    return store


@pytest.fixture
def attributed(isolated_root) -> tuple[Path, MetadataStore]:
    """Her state, her journal and the store, with the pass NOT yet run."""
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    journal(isolated_root, her_calls(her_tag_id(store)))
    return isolated_root, store


def test_matched_rows_flip_with_her_own_timestamp(attributed) -> None:
    """The matched rows move, dated by the edit that made them, tags and all."""
    root, store = attributed
    edits_before = len(store.edits(A, limit=500))

    assert backfill.reattribute_organized(root) == 2

    row = store.asset(A)
    assert row is not None
    assert row["name_source"] == "librarian" and row["description_source"] == "librarian"
    assert row["organized_by"] == "librarian"
    # MAX(edits.at) for that asset where actor='agent' — her original moment,
    # never the clock of this pass that noticed it.
    assert row["organized_at"] == 200.0
    assert store.asset_tags(A) == []
    # The audit log is re-read, not appended to.
    assert len(store.edits(A, limit=500)) == edits_before
    # The guard records that the pass ran and how much it moved.
    guard = json.loads(store.state_value(backfill.GUARD_KEY) or "{}")
    assert guard["matched"] == 2 and guard["at"]


def test_her_tag_assignment_moves_with_the_row(attributed) -> None:
    """A tag SHE applied — id and all — is re-sourced, and its row organized."""
    root, store = attributed
    assert [row["source"] for row in store.asset_tags(T)] == ["agent"]

    assert backfill.reattribute_organized(root) == 2

    tagged = store.asset(T)
    assert tagged is not None
    assert tagged["name_source"] == "librarian"
    assert tagged["organized_by"] == "librarian"
    # Her NAMING edit, not the tag set written seconds later: the stamp is the
    # moment she organized the asset, and a later agent edit is not it.
    assert tagged["organized_at"] == store.last_edit_at(T, actor="agent", fields=("name",))
    assert tagged["organized_at"] < store.last_edit_at(T, actor="agent")
    assert [row["source"] for row in store.asset_tags(T)] == ["librarian"]


def test_a_later_ordinary_agents_tag_set_stays_theirs(isolated_root) -> None:
    """The tags half is a VALUE match, not "her call mentions this asset".

    The finding: requiring only the asset id in a tags call re-sourced every
    agent tag row on the row she named — including a set another agent applied
    afterwards (a delete plus re-insert), which this pass may not claim. Only
    the ids her own call spells out move; the asset itself is still hers.
    """
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    journal(isolated_root, her_calls(her_tag_id(store)))
    theirs = store.create_tag("eng-3631")
    store.set_asset_tags(T, [theirs["id"]], actor="agent")

    assert backfill.reattribute_organized(isolated_root) == 2

    tagged = store.asset(T)
    assert tagged is not None and tagged["organized_by"] == "librarian"
    assert [row["id"] for row in store.asset_tags(T)] == [theirs["id"]]
    assert [row["source"] for row in store.asset_tags(T)] == ["agent"]


def test_a_native_tags_call_is_read_as_a_tags_op(isolated_root) -> None:
    """The op is an ARGUMENT in a native call, not text inside code.

    One literal would only ever match one of the two forms; the real journal
    held both, and a native call whose tags were not re-sourced would leave her
    own tag assignments attributed to an agent.
    """
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    journal(
        isolated_root,
        [
            {
                "id": "toolu_n",
                "name": "library",
                "arguments": {"op": "update", "asset_id": T, "name": NAME_T},
            },
            {
                "id": "toolu_m",
                "name": "library",
                "arguments": {
                    "op": "tags",
                    "action": "set",
                    "asset_id": T,
                    "tags": [her_tag_id(store)],
                },
            },
        ],
    )
    assert backfill.reattribute_organized(isolated_root) == 1
    assert [row["source"] for row in store.asset_tags(T)] == ["librarian"]


def tags_cell(tag_id: str) -> dict[str, Any]:
    """One eval cell whose tags op is written as a DICT — the missed spelling."""
    return {
        "id": "toolu_dict",
        "name": "eval",
        "arguments": {
            "code": (
                f'upd=[("{T}","{NAME_T}")]\n'
                "for a,n in upd:\n"
                '    r=tool("library",op="update",asset_id=a,name=n)\n'
                f'    r=tool("library",{{"op": "tags", "action": "set", '
                f'"asset_id": "{T}", "tags": ["{tag_id}"]}})\n'
            )
        },
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"op": "tags"},  # a native call: the op is an argument
        {"code": 'r=tool("library", op="tags", asset_id="la_x")'},
        {"code": "r=tool('library', op='tags', asset_id='la_x')"},
        {"code": 'r=tool("library", {"op": "tags", "asset_id": "la_x"})'},
        {"code": "r=tool('library', {'op': 'tags', 'asset_id': 'la_x'})"},
    ],
)
def test_every_spelling_of_a_tags_op_is_read_as_one(arguments: dict[str, Any]) -> None:
    """The op reaches a journal as an argument, or as text in four spellings.

    The finding: one literal matched one of them, and the miss is SILENT — the
    rows simply stay an agent's: nothing errors, nothing is logged.
    """
    assert backfill._is_tag_call({"name": "library", "arguments": arguments}) is True


@pytest.mark.parametrize(
    "code",
    [
        'stop="tags"',  # a word ending in ``op`` is not the op
        "opts='tags'",
        'tool("library", op="update", asset_id="la_x")',
        'tool("library", op="tagged", asset_id="la_x")',
    ],
)
def test_a_near_miss_is_not_a_tags_op(code: str) -> None:
    """Only the op itself counts, in either quote style and either position."""
    assert backfill._is_tag_call({"name": "library", "arguments": {"code": code}}) is False


def test_a_dict_form_tags_op_in_a_cell_still_moves_her_tag_row(isolated_root) -> None:
    """End to end, through the real pass: the spelling the old literal missed."""
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    journal(isolated_root, [tags_cell(her_tag_id(store))])

    assert backfill.reattribute_organized(isolated_root) == 1
    assert [row["source"] for row in store.asset_tags(T)] == ["librarian"]


def test_a_word_ending_in_op_is_not_a_tags_op(isolated_root) -> None:
    """``stop="tags"`` is not ``op="tags"``: no row moves on that alone.

    A bare substring match calls this cell a tag write — the id is in it — and
    re-sources a row on no evidence, which is the direction of error this pass
    must not make.
    """
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    journal(
        isolated_root,
        [
            {
                "id": "toolu_stop",
                "name": "eval",
                "arguments": {
                    "code": (
                        f'upd=[("{T}","{NAME_T}")]\n'
                        "for a,n in upd:\n"
                        '    r=tool("library",op="update",asset_id=a,name=n)\n'
                        'stop="tags"\n'
                        f'print("{T}", "{her_tag_id(store)}")\n'
                    )
                },
            }
        ],
    )
    assert backfill.reattribute_organized(isolated_root) == 1
    assert [row["source"] for row in store.asset_tags(T)] == ["agent"]


def test_a_row_an_agent_re_edited_afterwards_is_left_alone(attributed) -> None:
    """Her old value no longer stands, so the row keeps its agent provenance."""
    root, store = attributed
    backfill.reattribute_organized(root)
    row = store.asset(B)
    assert row is not None
    assert row["name"] == NAME_B_AFTER
    assert row["name_source"] == "agent"
    assert row["organized_at"] is None


def test_user_locked_and_never_written_rows_are_untouched(attributed) -> None:
    """Three ways an id in her journal is not proof of her work."""
    root, store = attributed
    backfill.reattribute_organized(root)
    for asset_id in (C, D, E):
        row = store.asset(asset_id)
        assert row is not None, asset_id
        assert row["organized_at"] is None, asset_id
        assert row["name_source"] in ("user", "agent"), asset_id
        assert row["organized_by"] is None, asset_id


def test_a_second_run_is_a_no_op(attributed) -> None:
    """The guard makes the pass one read, and the rows are unchanged by it."""
    root, store = attributed
    assert backfill.reattribute_organized(root) == 2
    before = store.asset(A)
    assert backfill.reattribute_organized(root) == 0
    assert store.asset(A) == before


def test_the_guard_short_circuits_before_the_journal_is_read(attributed, monkeypatch) -> None:
    """A finished pass costs ONE READ: it never re-parses her transcript.

    This is the ledger the trigger in ``bootstrap.ensure_session`` is bought
    with — every boot asks, and only the first one pays.
    """
    root, store = attributed
    assert backfill.reattribute_organized(root) == 2

    def explode(*_args, **_kwargs):
        raise AssertionError("a finished pass must not re-read the journal")

    monkeypatch.setattr(backfill, "_calls", explode)
    assert backfill.reattribute_organized(root) == 0


def test_a_missing_transcript_writes_the_guard_with_nothing_matched(isolated_root) -> None:
    """She exists, her journal does not: the pass finishes and records that."""
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    assert backfill.reattribute_organized(isolated_root) == 0
    guard = json.loads(store.state_value(backfill.GUARD_KEY) or "{}")
    assert guard["matched"] == 0 and guard["at"]
    row = store.asset(A)
    assert row is not None and row["organized_at"] is None


def test_an_install_without_her_writes_the_guard_and_moves_nothing(isolated_root) -> None:
    """No ``state.json`` at all is the ordinary install: nothing is hers."""
    store = seed(isolated_root)
    assert backfill.reattribute_organized(isolated_root) == 0
    assert store.state_value(backfill.GUARD_KEY) is not None
    row = store.asset(A)
    assert row is not None and row["name_source"] == "agent"


def test_a_torn_journal_line_costs_the_pass_a_row_not_an_error(attributed) -> None:
    """A crash mid-append is what a real journal looks like at the tail."""
    root, store = attributed
    path = root / "sessions" / LIB / "transcript.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "torn", "payload": {"role": "assistant", "tool_calls": ["library"')
    assert backfill.reattribute_organized(root) == 2
    row = store.asset(A)
    assert row is not None and row["organized_at"] == 200.0


def test_her_archived_journal_is_read_without_being_re_inflated(isolated_root) -> None:
    """The pass is a PURE READER of a compressed transcript.

    It reads her archive through ``transcript_for_read`` and must leave it
    exactly as it found it: no plain journal back in the session directory, and
    the archive byte-identical.
    """
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    directory = journal(isolated_root, her_calls(her_tag_id(store))).parent
    assert compression.compress_session(directory).status == "compressed"
    archive = compression.archive_path(directory)
    assert archive is not None and not (directory / compression.PLAIN_FILENAME).exists()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    assert backfill.reattribute_organized(isolated_root) == 2

    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest
    assert not (directory / compression.PLAIN_FILENAME).exists()
    assert store.asset(A) is not None


def test_an_unmakeable_cache_copy_never_restores_the_journal_in_place(
    isolated_root, monkeypatch
) -> None:
    """With no cache copy to be had, the pass gives up instead of inflating it.

    The finding: without ``read_only=True`` the reader falls into
    ``restore_for_write``, which publishes the plain file back into the session
    directory WITHOUT the writer lock — a pure reader undoing the compression on
    exactly the low-disk device the archive exists for. Refused, the typed error
    reaches the boot hook's best-effort handler: the guard stays unwritten, so
    the next boot (with a cache it can write) retries the pass.
    """
    state.write_state(isolated_root, {"session_id": LIB})
    store = seed(isolated_root)
    directory = journal(isolated_root, her_calls(her_tag_id(store))).parent
    assert compression.compress_session(directory).status == "compressed"

    def no_cache(*_args: object, **_kwargs: object) -> Path:
        raise compression.ArchiveCacheUnavailable("the volume filled")

    monkeypatch.setattr(compression, "_materialise", no_cache)
    with pytest.raises(compression.ArchiveCacheUnavailable):
        backfill.reattribute_organized(isolated_root)

    assert not (directory / compression.PLAIN_FILENAME).exists()
    assert compression.archive_path(directory) is not None
    assert store.state_value(backfill.GUARD_KEY) is None


def test_the_stamp_falls_back_when_no_naming_edit_row_survives(attributed) -> None:
    """A store whose audit row for her naming edit is gone still gets a stamp.

    The narrowed timestamp is the naming edit; only when the audit log holds no
    name or description row for the asset at all does its newest agent edit do,
    because a stamp a little early beats none.
    """
    root, store = attributed
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM edits WHERE asset_id = ? AND fields LIKE '%name%'", (T,))
    assert store.last_edit_at(T, actor="agent", fields=("name", "description")) is None

    assert backfill.reattribute_organized(root) == 2

    row = store.asset(T)
    assert row is not None
    assert row["organized_at"] == store.last_edit_at(T, actor="agent")
