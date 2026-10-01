"""The ``--failed`` / ``--paused`` listing sets: membership and the cap order.

The sets a bulk resume will select from are answered out of the attention
store's own completion kinds — :data:`FAILED_OUTCOME_KINDS` is ``error`` and
:data:`PAUSED_OUTCOME_KINDS` is ``{interrupted, retired}``, the two kinds the
product already paints together as "Unseen interruption". These cells pin the
vocabulary, the FILTER-BEFORE-LIMIT ordering (an old member must survive a cap
that newer non-members would fill), and the CLI's flag behaviour.

Every store here is a real directory with real transcripts; the outcome rows are
written through the real :class:`AttentionStore`, because the whole point of the
mapping is that it reads the store the rest of the product reads.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from local_operator import cli
from local_operator.info.collect import (
    FAILED_OUTCOME_KINDS,
    PAUSED_OUTCOME_KINDS,
    session_rows,
    stored_sessions_by_outcome,
)
from local_operator.session.attention import AttentionStore


def _session(root: Path, session_id: str, *, age_s: float) -> Path:
    """One user session directory, its transcript stamped ``age_s`` ago."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = directory / "transcript.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    stamp = time.time() - age_s
    import os

    os.utime(transcript, (stamp, stamp))
    (directory / "title.json").write_text(
        json.dumps({"title": f"session {session_id}"}), encoding="utf-8"
    )
    return directory


def _publish(root: Path, session_id: str, kind: str, *, reason: str = "") -> None:
    """One outcome row for ``session/<id>`` through the real store."""
    AttentionStore(root / "attention.db").publish(
        f"session/{session_id}",
        str(uuid.uuid4()),
        f"e-{session_id}",
        kind,
        reason=reason,
    )


def test_the_set_vocabulary_is_the_products_own_pair() -> None:
    """The mapping is a pinned constant, not a scatter of string literals.

    ``error`` is the failure half; ``{interrupted, retired}`` is the
    interruption pair ``catalog._stop_label`` already renders as one word. A
    future kind (``closed`` is the live example) is in NEITHER set by default,
    which is the safe direction: a set a resume path selects from must not grow
    a member because a new kind was invented elsewhere.
    """
    assert FAILED_OUTCOME_KINDS == frozenset({"error"})
    assert PAUSED_OUTCOME_KINDS == frozenset({"interrupted", "retired"})
    assert FAILED_OUTCOME_KINDS.isdisjoint(PAUSED_OUTCOME_KINDS)


def test_membership_is_the_recorded_kind_and_nothing_else(tmp_path: Path) -> None:
    """Only the recorded kind decides membership: no outcome row, ``complete``
    and ``closed`` are all outside both sets; ``error`` is failed alone; the
    interruption pair is paused alone.
    """
    for index, (session_id, kind) in enumerate(
        [
            ("a" * 12, "error"),
            ("b" * 12, "interrupted"),
            ("c" * 12, "retired"),
            ("d" * 12, "complete"),
            ("e" * 12, "closed"),
            ("f" * 12, None),  # no outcome row at all
        ]
    ):
        _session(tmp_path, session_id, age_s=100.0 + index)
        if kind is not None:
            _publish(tmp_path, session_id, kind)

    failed = {
        session_id for session_id, _ in stored_sessions_by_outcome(tmp_path, FAILED_OUTCOME_KINDS)
    }
    paused = {
        session_id for session_id, _ in stored_sessions_by_outcome(tmp_path, PAUSED_OUTCOME_KINDS)
    }
    assert failed == {"a" * 12}
    assert paused == {"b" * 12, "c" * 12}


def test_a_member_older_than_the_cap_still_appears(tmp_path: Path) -> None:
    """THE FILTER-BEFORE-LIMIT CELL. Sixty plain sessions are newer than the one
    failed session, so an unfiltered ``limit=50`` excludes it; the filtered
    answer must still find it. A filter applied AFTER the cap would return
    nothing here — exactly the silently-wrong failure this rule exists to stop.
    """
    for index in range(60):
        _session(tmp_path, f"u{index:010x}", age_s=10.0 + index)
    _session(tmp_path, "f" * 12, age_s=10_000.0)  # the oldest, and the only failure
    _publish(tmp_path, "f" * 12, "error", reason="killed")

    unfiltered = stored_sessions_by_outcome(tmp_path, frozenset(), limit=50)
    assert "f" * 12 not in {session_id for session_id, _ in unfiltered}, "outside the plain cap"

    matched = stored_sessions_by_outcome(tmp_path, FAILED_OUTCOME_KINDS, limit=50)
    assert [session_id for session_id, _ in matched] == ["f" * 12]

    one = stored_sessions_by_outcome(tmp_path, FAILED_OUTCOME_KINDS, limit=1)
    assert [session_id for session_id, _ in one] == ["f" * 12], "found under the smallest cap"


def test_the_cap_keeps_the_newest_members(tmp_path: Path) -> None:
    """A cap bounds the ANSWER, not the search: six failures with distinct ages
    and ``limit=2`` yields the two newest of them, in recency order.
    """
    for index in range(6):
        _session(tmp_path, f"f{index:010x}", age_s=100.0 * (index + 1))
        _publish(tmp_path, f"f{index:010x}", "error")

    matched = stored_sessions_by_outcome(tmp_path, FAILED_OUTCOME_KINDS, limit=2)
    assert [session_id for session_id, _ in matched] == ["f0000000000", "f0000000001"]


def test_both_flags_select_the_union(tmp_path: Path) -> None:
    """The union is what "show me everything stopped or broken" means, and it is
    the shape a bulk resume with both flags selects from.
    """
    _session(tmp_path, "a" * 12, age_s=100.0)
    _publish(tmp_path, "a" * 12, "error")
    _session(tmp_path, "b" * 12, age_s=200.0)
    _publish(tmp_path, "b" * 12, "interrupted")
    _session(tmp_path, "c" * 12, age_s=300.0)

    union = stored_sessions_by_outcome(
        tmp_path, FAILED_OUTCOME_KINDS | PAUSED_OUTCOME_KINDS, limit=None
    )
    assert {session_id for session_id, _ in union} == {"a" * 12, "b" * 12}


def test_live_ids_are_dropped_before_the_cap(tmp_path: Path) -> None:
    """A running session is not a stored member, and excluding it must happen
    BEFORE the cap — otherwise a live session could consume one of the cap's
    slots and push a real member off the page.
    """
    _session(tmp_path, "a" * 12, age_s=1.0)  # newest, but live
    _session(tmp_path, "b" * 12, age_s=100.0)
    _publish(tmp_path, "b" * 12, "error")

    matched = stored_sessions_by_outcome(
        tmp_path, FAILED_OUTCOME_KINDS, exclude_ids={"a" * 12}, limit=1
    )
    assert [session_id for session_id, _ in matched] == ["b" * 12]


def test_session_rows_answers_exactly_the_set(tmp_path: Path) -> None:
    """Through the public row builder a set filter yields the matched STORED
    rows only — the live fleet is not mixed in, and each row carries the kind
    that selected it.
    """
    _session(tmp_path, "a" * 12, age_s=100.0)
    _publish(tmp_path, "a" * 12, "error")
    _session(tmp_path, "b" * 12, age_s=200.0)
    _publish(tmp_path, "b" * 12, "retired")

    rows = session_rows(
        tmp_path, include_stored=True, stored_limit=50, stored_kinds=FAILED_OUTCOME_KINDS
    )
    assert [(row["session_id"], row["completion_kind"]) for row in rows] == [("a" * 12, "error")]

    paused = session_rows(
        tmp_path, include_stored=True, stored_limit=50, stored_kinds=PAUSED_OUTCOME_KINDS
    )
    assert [(row["session_id"], row["completion_kind"]) for row in paused] == [
        ("b" * 12, "retired")
    ]


def test_the_cli_flags_select_the_set_and_imply_the_store(tmp_path: Path, monkeypatch) -> None:
    """``lop sessions --failed`` reaches ``session_rows`` with the failed set and
    ``include_stored`` on, without the caller having to also pass ``--all``.
    """
    captured: dict[str, object] = {}

    def fake_session_rows(root, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return []

    monkeypatch.setattr("local_operator.info.collect.session_rows", fake_session_rows)
    monkeypatch.setattr(cli, "config_dir", lambda: tmp_path)

    code = cli.sessions_command(
        argparse.Namespace(
            json=True,
            sessions_command=None,
            all=False,
            limit=None,
            paused=False,
            failed=True,
        )
    )
    assert code == 0
    assert captured["include_stored"] is True
    assert captured["stored_kinds"] == FAILED_OUTCOME_KINDS

    captured.clear()
    cli.sessions_command(
        argparse.Namespace(
            json=True, sessions_command=None, all=False, limit=None, paused=True, failed=True
        )
    )
    assert captured["stored_kinds"] == FAILED_OUTCOME_KINDS | PAUSED_OUTCOME_KINDS


def test_a_store_without_an_attention_db_is_in_neither_set(tmp_path: Path) -> None:
    """The honest reading of "no outcome recorded" is "not known", not
    membership: with no store at all, both sets are empty and nothing raises.
    """
    _session(tmp_path, "a" * 12, age_s=100.0)
    assert stored_sessions_by_outcome(tmp_path, FAILED_OUTCOME_KINDS) == []
    assert stored_sessions_by_outcome(tmp_path, PAUSED_OUTCOME_KINDS) == []
