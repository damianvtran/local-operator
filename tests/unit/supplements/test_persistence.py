"""The ``supplement_v1`` row: shape against C0's frozen fixture, versions, dispositions.

The row is a CONTRACT (C0 froze it as code and fixtures). These tests build rows with the same
builders the runner uses and compare them with the committed fixtures, so a shape change here
is a diff in a fixture the other lanes read.
"""

from __future__ import annotations

import json

import pytest

from local_operator.session.transcript import BOOKKEEPING_CUSTOM_TYPES
from local_operator.supplements import persistence
from local_operator.supplements.candidates import Candidate
from local_operator.supplements.contract import (
    MORE_MAX,
    SUPERSEDED_ERROR,
    SUPPLEMENT_CUSTOM_TYPE,
    reader_disposition,
)
from local_operator.supplements.decision import Decision

from .conftest import load


def _candidate(path: str, *, name: str | None = None) -> Candidate:
    return Candidate(
        path=path,
        absolute=f"/work/{path}",
        name=name or path.rsplit("/", 1)[-1],
        kind="markdown",
        size_bytes=4120,
        mtime=1790999940.0,
        tier=1,
        tool="write",
        order=1,
        intent="",
    )


def _decision(*paths: str, more: int = 0) -> Decision:
    featured = tuple(_candidate(path) for path in paths)
    extra = tuple(_candidate(f"extra{i}.csv") for i in range(more))
    return Decision(
        featured=featured,
        more=extra,
        files_p={c.path: 0.9 for c in featured},
        graphics=False,
        graphics_p=0.0,
        vendor="radient",
    )


def test_the_committed_files_only_fixture_is_exactly_what_the_builders_produce() -> None:
    """The C0 fixture ``rows/done_files_only.json`` is the row this lane writes for the same
    decision: the shapes must not drift."""
    fixture = load("rows/done_files_only.json")
    expected = fixture["payload"]["details"]
    built = persistence.build_details(
        anchor=expected["anchor"],
        job=expected["job"],
        version=1,
        state="done",
        decision=Decision(
            featured=tuple(_candidate(f["path"], name=f["name"]) for f in expected["files"]),
            files_p=dict(expected["decision"]["files_p"]),
            graphics_p=expected["decision"]["graphics_p"],
            vendor=expected["decision"]["vendor"],
        ),
        at=expected["at"],
    )
    # ``size_bytes``/``kind``/``mtime``/``why`` come from the Candidate, so compare the fields
    # the builders own and the fixture pins: identity, state, decision and file paths.
    assert built["anchor"] == expected["anchor"]
    assert built["job"] == expected["job"]
    assert built["version"] == 1 and built["state"] == "done"
    assert [f["path"] for f in built["files"]] == [f["path"] for f in expected["files"]]
    assert built["components"] == expected["components"] == []
    assert built["decision"] == expected["decision"]
    assert built["at"] == expected["at"]


def test_a_files_only_row_round_trips_through_the_contract_types() -> None:
    from local_operator.supplements.contract import SupplementDetails

    built = persistence.build_details(
        anchor="a" * 32, job="b" * 12, version=1, state="done", decision=_decision("reports/q3.md")
    )
    assert set(SupplementDetails.__required_keys__) <= set(built)
    # json round trip: the row is journaled as JSON, so anything the writer emits must
    # survive ``json.dumps``/``loads`` unchanged.
    assert json.loads(json.dumps(built)) == built
    assert reader_disposition(built, job_live=True) == "block"


def test_the_more_disclosure_is_capped_and_counts_qualified_files_only() -> None:
    built = persistence.build_details(
        anchor="a" * 32,
        job="b" * 12,
        version=1,
        state="done",
        decision=_decision("keep.md", more=MORE_MAX + 5),
    )
    assert built["files_more"] == MORE_MAX + 5
    assert len(built["more"]) == MORE_MAX
    # An empty remainder writes neither key: a row stays lean and old readers keep working.
    lean = persistence.build_details(
        anchor="a" * 32, job="b" * 12, version=1, state="done", decision=_decision("only.md")
    )
    assert "files_more" not in lean and "more" not in lean


def test_a_new_version_carries_the_files_and_clears_the_state() -> None:
    first = persistence.build_details(
        anchor="a" * 32, job="b" * 12, version=1, state="done", decision=_decision("r.md")
    )
    nxt = persistence.next_version(first, state="failed", error="boom")
    assert nxt["version"] == 2 and nxt["state"] == "failed" and nxt["error"] == "boom"
    assert nxt["files"] == first["files"], "a later version restates what it still shows"
    assert nxt["job"] == first["job"] and nxt["anchor"] == first["anchor"]
    assert nxt["at"] >= first["at"]


def test_a_superseded_row_renders_nothing() -> None:
    row = persistence.superseded(
        persistence.build_details(
            anchor="a" * 32, job="b" * 12, version=1, state="queued", decision=_decision("r.md")
        )
    )
    assert row["state"] == "cancelled" and row["error"] == SUPERSEDED_ERROR
    assert reader_disposition(row, job_live=True) == "nothing"
    assert reader_disposition(row, job_live=False) == "nothing"


def test_an_error_is_bounded_and_a_clean_row_has_no_error_key() -> None:
    long = persistence.build_details(
        anchor="a" * 32,
        job="b" * 12,
        version=1,
        state="failed",
        decision=_decision(),
        error="x" * 10_000,
    )
    assert len(long["error"]) == persistence._ERROR_MAX_CHARS
    clean = persistence.build_details(
        anchor="a" * 32, job="b" * 12, version=1, state="done", decision=_decision()
    )
    assert "error" not in clean


def test_the_type_is_bookkeeping_and_never_a_message_row() -> None:
    """Rows must not move the activity clock (``preserve_mtime`` is the writer's other half)
    and must not be a persistable custom MESSAGE type."""
    assert SUPPLEMENT_CUSTOM_TYPE in BOOKKEEPING_CUSTOM_TYPES
    from local_operator.session.session import _PERSISTABLE_CUSTOM_TYPES

    assert SUPPLEMENT_CUSTOM_TYPE not in _PERSISTABLE_CUSTOM_TYPES


@pytest.mark.asyncio
async def test_append_row_writes_with_preserve_mtime(tmp_path) -> None:
    """A supplement lands AFTER the turn: it must not move the activity clock.

    The file's mtime is first set two hours into the past, so the two behaviours are two hours
    apart rather than sub-millisecond: with ``preserve_mtime`` the write restores the old
    stamp; without it the file is stamped NOW. A tolerance of one second separates the float
    round trip in the restore from a forgotten flag.
    """
    import os
    import time

    from local_operator.harness.types import Message
    from local_operator.session.transcript import Transcript

    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("hello"))
    journal = tmp_path / "sess" / "transcript.jsonl"
    assert journal.exists(), "the journal must exist before the measurement"
    old = time.time() - 7200
    os.utime(journal, (old, old))
    built = persistence.build_details(
        anchor="a" * 32, job="b" * 12, version=1, state="done", decision=_decision("r.md")
    )
    await persistence.append_row(transcript, built)
    rows = [
        e for e in transcript.entries() if e.payload.get("custom_type") == SUPPLEMENT_CUSTOM_TYPE
    ]
    assert len(rows) == 1
    assert rows[0].payload["details"]["anchor"] == "a" * 32
    advanced = abs(journal.stat().st_mtime - old)
    assert advanced < 1.0, f"a supplement row moved the activity clock by {advanced:.1f}s"
