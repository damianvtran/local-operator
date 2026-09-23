"""The goal record's sidecar: atomicity, tolerance, and what a resume rebuilds.

Two properties are the reason this file exists rather than trusting the
functions by inspection: the journal must be written on TRANSITION only (the
judge moves on every turn end, so a per-tick write would be pure I/O for a value
that did not move), and an unreadable sidecar must cost the RECORD and never the
resume — a session whose sidecar was cut mid-write still has to open.

The ``Session`` objects here are built with ``object.__new__`` around the real
methods, the pattern the neighbouring goal tests use: the methods under test are
the real ones, without booting a transcript, a provider registry or a tool
context to prove a file write.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, cast

from local_operator.resume import (
    GOAL_SIDECAR_NAME,
    read_goal_record,
    write_goal_record,
    write_session_attachment,
)
from local_operator.session.goal import GoalJudgeState, GoalState
from local_operator.session.session import Session


class _Transcript:
    """The one attribute the two record methods read off a transcript."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory


def _stub_session(directory: Path, *, restoring: bool = False) -> Session:
    session = object.__new__(Session)
    session._transcript = cast(Any, _Transcript(directory))
    session._goal_state = GoalState()
    session._restoring_attachment = restoring
    # Counted, and stubbed rather than run: publishing is the mutation path's
    # business (pinned by the host tests), and a real refresh needs a store.
    session.refresh_frontend_state = lambda: None  # type: ignore[method-assign]
    return session


def test_write_is_atomic_and_leaves_no_temp_behind(tmp_path: Path) -> None:
    write_goal_record(tmp_path, {"goal": "A", "status": "active"})

    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [GOAL_SIDECAR_NAME], "a pid-named temp survived the replace"
    assert json.loads((tmp_path / GOAL_SIDECAR_NAME).read_text())["goal"] == "A"


def test_write_preserves_the_directory_mtime(tmp_path: Path) -> None:
    before = tmp_path.stat().st_mtime
    time.sleep(0.02)

    write_goal_record(tmp_path, {"goal": "A"})

    # Journalling a goal transition is bookkeeping ABOUT a session, never
    # activity IN it: a mark-done must not reorder the /resume picker.
    assert tmp_path.stat().st_mtime == before


def test_a_missing_sidecar_reads_as_none(tmp_path: Path) -> None:
    assert read_goal_record(tmp_path) is None


def test_a_corrupt_or_non_mapping_document_reads_as_none(tmp_path: Path) -> None:
    sidecar = tmp_path / GOAL_SIDECAR_NAME
    for raw in ("", "{", '{"goal": "x"', "[]", "null", "7"):
        sidecar.write_text(raw, encoding="utf-8")
        assert read_goal_record(tmp_path) is None, raw


def test_a_cut_multibyte_character_does_not_raise(tmp_path: Path) -> None:
    # ``errors="replace"`` is load-bearing: a process killed mid-write can cut the
    # file inside a multi-byte character, and a UnicodeDecodeError is a ValueError
    # that would sail past an ``except OSError`` and take down the whole resume.
    (tmp_path / GOAL_SIDECAR_NAME).write_bytes('{"goal": "caf\xc3'.encode("latin-1"))

    assert read_goal_record(tmp_path) is None


def test_persist_writes_the_holders_record_and_is_suppressed_in_a_restore(
    tmp_path: Path,
) -> None:
    session = _stub_session(tmp_path)
    session._goal_state.arm("ship it")

    session._persist_goal_record()
    stored = read_goal_record(tmp_path)
    assert stored is not None and stored["goal"] == "ship it"

    # A restore is a read of disk state, not a user action: journalling the value
    # straight back would rewrite the file the restore just read.
    (tmp_path / GOAL_SIDECAR_NAME).unlink()
    restoring = _stub_session(tmp_path, restoring=True)
    restoring._goal_state.arm("another")
    restoring._persist_goal_record()
    assert read_goal_record(tmp_path) is None


def test_restore_rebuilds_status_judge_and_history(tmp_path: Path) -> None:
    source = GoalState()
    source.arm("A")
    source.mark_done("the judge said so")
    source.dismiss()
    source.arm("B")
    source.judge = GoalJudgeState(
        state="continuing", run=2, verdict="continue", reason="r", failures=1
    )
    write_goal_record(tmp_path, source.to_payload())
    write_session_attachment(tmp_path, team="", agent="", goal="B")
    session = _stub_session(tmp_path)

    session._restore_goal_record()

    holder = session._goal_state
    assert holder.text == "B"
    assert holder.status == "active"
    assert holder.token == source.token
    assert holder.judge.to_payload() == source.judge.to_payload()
    assert [entry.to_wire() for entry in holder.history] == [
        entry.to_wire() for entry in source.history
    ]


def test_restore_never_journals_what_it_just_read(tmp_path: Path) -> None:
    source = GoalState()
    source.arm("A")
    write_goal_record(tmp_path, source.to_payload())
    before = (tmp_path / GOAL_SIDECAR_NAME).read_text()
    session = _stub_session(tmp_path)

    session._restore_goal_record()

    assert (tmp_path / GOAL_SIDECAR_NAME).read_text() == before


def test_restore_keeps_the_attachments_text_when_both_documents_carry_one(
    tmp_path: Path,
) -> None:
    # The attachment is written by EVERY build, including the ones that predate
    # this record, so a divergence means an older build moved the goal after the
    # record was written and the record's text is the stale one.
    session = _stub_session(tmp_path)
    session._goal_state.set("the newer one")
    write_session_attachment(tmp_path, team="", agent="", goal="the newer one")
    stale = GoalState()
    stale.arm("the older one")
    write_goal_record(tmp_path, stale.to_payload())

    session._restore_goal_record()

    assert session._goal_state.text == "the newer one"


def test_restore_fills_the_text_when_the_attachment_is_empty(tmp_path: Path) -> None:
    session = _stub_session(tmp_path)
    source = GoalState()
    source.arm("only in the record")
    write_goal_record(tmp_path, source.to_payload())

    session._restore_goal_record()

    assert session._goal_state.text == "only in the record"


def test_a_directory_with_no_record_leaves_the_pre_lifecycle_state(tmp_path: Path) -> None:
    write_session_attachment(tmp_path, team="", agent="", goal="standing work")
    session = _stub_session(tmp_path)

    session._restore_goal_record()

    # Only attachment.json: no status (the FOLD applies the migration default at
    # read time, not the restore), no judge and no history.
    assert session._goal_state.status == ""
    assert session._goal_state.judge == GoalJudgeState()
    assert session._goal_state.history == []
    assert read_goal_record(tmp_path) is None


def test_a_transition_journals_once_per_transition(tmp_path: Path) -> None:
    session = _stub_session(tmp_path)
    writes: list[Any] = []
    real_write = write_goal_record

    def _spy(directory: Path, payload: dict[str, Any]) -> None:
        writes.append(dict(payload))
        real_write(directory, payload)

    import local_operator.resume as resume_module

    resume_module.write_goal_record = _spy  # type: ignore[assignment]
    try:
        session.arm_goal("ship it")
        session.note_goal_judge(state="judging")
        session.note_goal_judge(state="continuing", run=1)
        session.mark_goal_done("done")
    finally:
        resume_module.write_goal_record = real_write  # type: ignore[assignment]

    # Four transitions, four writes — and each one carries the state at the
    # moment it happened, so the file is never a tick behind:
    assert len(writes) == 4
    assert writes[0]["status"] == "active"
    assert writes[1]["judge"]["state"] == "judging"
    assert writes[2]["judge"]["run"] == 1
    assert writes[3]["status"] == "done"


def test_a_restored_judge_keeps_its_breaker_counter(tmp_path: Path) -> None:
    source = GoalState()
    source.arm("A")
    source.judge = GoalJudgeState(state="stalled", run=3, verdict="unknown", reason="r", failures=3)
    write_goal_record(tmp_path, source.to_payload())

    session = _stub_session(tmp_path)
    session._restore_goal_record()

    # The counter is what makes the breaker survive a restart; losing it would
    # hand a broken provider a fresh set of strikes on every boot.
    assert session._goal_state.judge.failures == 3
    assert session.goal_judge is not None
    assert "failures" not in session.goal_judge, "an internal count must not ride the frame"


def test_an_unreadable_record_costs_the_record_and_not_the_resume(tmp_path: Path) -> None:
    (tmp_path / GOAL_SIDECAR_NAME).write_text("{not json", encoding="utf-8")
    write_session_attachment(tmp_path, team="", agent="", goal="standing work")
    session = _stub_session(tmp_path)

    session._restore_goal_record()

    # The session opens exactly as a pre-lifecycle one would: the goal text still
    # arrives through its own restore, and no record is invented for it.
    assert session._goal_state.status == ""
    assert session._goal_state.history == []
    assert os.path.isdir(tmp_path)
