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

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.librarian import backfill, state
from local_operator.library.metadata import MetadataStore

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


def her_calls() -> list[dict[str, Any]]:
    """The three call shapes her journal held, with the values she wrote."""
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
                    f'sets={{"{T}":["hyundai-poc"]}}\n'
                    "for a,t in sets.items():\n"
                    '    r=tool("library",op="tags",action="set",asset_id=a,tags=t,primary=t[0])\n'
                )
            },
        },
    ]


def seed(root: Path) -> MetadataStore:
    """One store holding every case the predicate has to tell apart."""
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
    journal(isolated_root, her_calls())
    return isolated_root, seed(isolated_root)


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
    """A tag SHE applied is re-sourced, and the row it sits on is organized."""
    root, store = attributed
    assert [row["source"] for row in store.asset_tags(T)] == ["agent"]

    assert backfill.reattribute_organized(root) == 2

    tagged = store.asset(T)
    assert tagged is not None
    assert tagged["name_source"] == "librarian"
    assert tagged["organized_by"] == "librarian"
    assert tagged["organized_at"] == store.last_edit_at(T, actor="agent")
    assert [row["source"] for row in store.asset_tags(T)] == ["librarian"]


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
                "arguments": {"op": "tags", "action": "set", "asset_id": T, "tags": []},
            },
        ],
    )
    assert backfill.reattribute_organized(isolated_root) == 1
    assert [row["source"] for row in store.asset_tags(T)] == ["librarian"]


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
