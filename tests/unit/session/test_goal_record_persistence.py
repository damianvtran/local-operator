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

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, cast

from local_operator.prompts_api import build_system_blocks
from local_operator.resume import (
    GOAL_SIDECAR_NAME,
    read_goal_record,
    write_goal_record,
    write_session_attachment,
)
from local_operator.session.goal import GoalJudgeState, GoalState
from local_operator.session.goal_judge import GoalJudge
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
    # The turn counter the judge samples for its staleness baseline.
    session._generation = 0
    # The attributes `Session.__init__` seeds that the attachment write reads.
    # `arm_goal` journals the goal through `_persist_attachment` as well as the
    # record (the goal rides the same tail as the team/agent briefs), and that
    # path resolves `active_team_name` — which reads `active_team` and then, when
    # that is empty, the carried `_unresolved_team`. A double built with
    # `object.__new__` skips `__init__`, so it has to seed them itself or the
    # real write path raises on the first goal transition.
    session.active_team = None
    session._unresolved_team = ""
    session._unresolved_agent = ""
    # Counted, and stubbed rather than run: publishing is the mutation path's
    # business (pinned by the host tests), and a real refresh needs a store.
    session.refresh_frontend_state = lambda: None  # type: ignore[method-assign]
    return session


#: A verdict that SETTLES the goal it was asked about — the shortest way to prove
#: a judge really ran, rather than that a goal merely looked judgeable.
_ACHIEVED = "VERDICT: ACHIEVED\nall the work is done"


def _judge_over(session: Session) -> tuple[GoalJudge, list[str]]:
    """A REAL ``GoalJudge`` over a real holder, wired the way both hosts wire it.

    Both production constructors (``serving.py`` for the runtime, ``app.py`` for
    the TUI) read the goal, the status, the token and the judge record off the
    live host through exactly these callables, so a judgement driven through this
    one is the judgement those hosts would take: nothing here re-derives "is this
    goal worth judging" on its own.

    Returns the driver beside the list of prompts its judge was called with, which
    is the evidence that it ran at all.
    """
    calls: list[str] = []

    async def judge(prompt_text: str) -> str:
        calls.append(prompt_text)
        return _ACHIEVED

    async def prompt(text: str) -> None:
        raise AssertionError("an ACHIEVED verdict admits no continuation")

    def settled(reason: str) -> None:
        # A statement, not a lambda: `settled` is typed `Callable[[str], None]`
        # and `mark_goal_done` returns the entry it recorded, which the driver
        # has no use for (the serving handle's own `settled` drops it the same way).
        session.mark_goal_done(reason)

    driver = GoalJudge(
        judge=judge,
        prompt=prompt,
        changed=lambda fields: session.note_goal_judge(**fields),
        settled=settled,
        goal=lambda: session.goal,
        status=lambda: session.goal_status,
        token=lambda: session.goal_token,
        serial=lambda: session.goal_turn_serial,
        judge_state=lambda: session.goal_judge_state,
        loop_running=lambda: False,
    )
    return driver, calls


def _goal_prompt_block(session: Session) -> str:
    """The system blocks this session's goal would produce, joined.

    The gate under test lives in the prompt builder (``goal_status != "done"``),
    so asking it here is the difference between "the record says active" and "the
    model is actually told about this objective".
    """
    return "\n".join(
        build_system_blocks(
            [],
            "",
            "",
            "2026-09-23",
            goal=session.goal,
            goal_status=session.goal_status,
        )
    )


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


def test_a_cleared_goal_journals_the_record_so_a_resume_cannot_resurrect_it(
    tmp_path: Path,
) -> None:
    """A clear takes BOTH halves with it: the tail AND the sidecar.

    The attachment is not enough on its own, and that is the test above's rule
    seen from the other side: ``_restore_goal_record`` fills the goal's TEXT from
    ``goal.json`` whenever the attachment carries none, so a clear that journalled
    only the attachment left the objective in the sidecar and the next resume read
    it straight back. Measured end to end before this was pinned: a
    ``lop exec --resume SID --clear-goal`` followed by ``lop exec --resume SID
    --loop 1`` STARTED a loop instead of raising the refusal ``--loop`` owes a
    session with no objective (``exec_startup``).
    """
    session = _stub_session(tmp_path)
    # ``arm_goal`` is ``/goal <text>`` — the path that journals the record as well
    # as the attachment, and therefore the state ``--clear-goal`` clears from.
    session.arm_goal("land the OAuth refresh fix")
    armed = read_goal_record(tmp_path)
    assert armed is not None
    assert armed["goal"] == "land the OAuth refresh fix"

    session.set_goal("")

    cleared = read_goal_record(tmp_path)
    assert cleared is not None
    assert cleared["goal"] == ""
    # A resumed session must not be handed the cleared objective back — the fold
    # and the judge both read this text as standing work.
    resumed = _stub_session(tmp_path)
    resumed._restore_goal_record()
    assert resumed.goal == ""


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


def test_set_goal_arms_a_fresh_goal_the_judge_can_run_on(tmp_path: Path) -> None:
    """Agent review MAJOR-3, the `lop --goal` / mobile-relay half.

    `set_goal` is the plain "replace the text" act behind `--goal` and the mobile
    relay. The judge's `_enabled` requires a TOKEN, and it also requires the
    status to read as active — so a goal every surface reported as `active` and
    nothing could ever run on was the silent half of "the goal sitting inert".
    Both conditions now come from ONE place: the arming (see the MAJOR-5 pins
    below), rather than a text write plus a token minted afterwards.
    """
    session = _stub_session(tmp_path)

    stored = session.set_goal("land the OAuth refresh fix")

    assert stored == "land the OAuth refresh fix"
    assert session.goal_token, "the judge can run on a plainly-set goal"
    assert session.goal_status == "active"
    # Clearing it takes the token with it: a goal that left has no identity.
    session.set_goal("")
    assert session.goal_token == ""


def test_set_goal_over_a_settled_goal_starts_a_new_life_and_is_judged(
    tmp_path: Path,
) -> None:
    """Agent review round 2, MAJOR-5 — the defect the round was sent to audit.

    A plain `set_goal` over a SETTLED goal wrote only the text, so the new
    objective inherited `status="done"`: the judge refused it, the card struck it
    through as achieved, and the prompt withheld `<goal>` on this feature's own
    gate — a goal silently inert, which is the failure mode the judged goal
    exists to remove. Pinned BOTH ways: the settled goal is still withheld and
    still unjudged, and the objective that replaced it is judged and in the
    prompt.
    """
    session = _stub_session(tmp_path)
    session.set_goal("Ship safely")
    session.mark_goal_done("the judge said so")
    settled_token = session.goal_token

    # The negative control, taken while the goal really is settled: nothing runs,
    # and the model is told nothing. Everything below has to move — if it does not,
    # a test that only ever asserted the positive would still pass on a build that
    # never armed anything.
    settled_driver, settled_calls = _judge_over(session)
    asyncio.run(
        settled_driver.on_turn_end(error=False, aborted=False, serial=session.goal_turn_serial)
    )
    assert settled_calls == [], "a settled goal is not judged"
    assert "<goal>" not in _goal_prompt_block(session)

    session.set_goal("Land the new billing migration")

    # A NEW LIFE, taken together: text, lifecycle, judge, token and history.
    assert session.goal == "Land the new billing migration"
    assert session.goal_status == "active", "no branch may leave the new goal settled"
    assert (
        session.goal_token and session.goal_token != settled_token
    ), "the token is the staleness guard, so a replaced goal mints a fresh one"
    assert session.goal_judge_state.state == "waiting", "a goal IS active work"
    # The settled goal the user asked to keep is KEPT, and it is not lost by the
    # replacement: `GoalState.arm` settles a done goal where it already stands.
    assert [entry["text"] for entry in session.goal_history] == ["Ship safely"]
    assert "<goal>" in _goal_prompt_block(session)
    assert "Land the new billing migration" in _goal_prompt_block(session)

    # The two durable halves are written from that same holder, so they cannot
    # disagree about which goal is current — which they did: the attachment
    # carried the new text while `goal.json` still carried the settled goal.
    record = read_goal_record(tmp_path)
    assert record is not None
    assert record["goal"] == "Land the new billing migration"
    assert record["status"] == "active"
    attachment = json.loads((tmp_path / "attachment.json").read_text())
    assert attachment["goal"] == "Land the new billing migration"

    # ...and the judge really runs on it: its own verdict settles THIS goal.
    driver, calls = _judge_over(session)
    asyncio.run(driver.on_turn_end(error=False, aborted=False, serial=session.goal_turn_serial))
    assert calls, "a goal set after a settled one IS judged"
    assert session.goal_status == "done"
    assert [entry["text"] for entry in session.goal_history] == [
        "Land the new billing migration",
        "Ship safely",
    ]


def test_set_goal_over_an_active_goal_supersedes_it_and_re_mints(tmp_path: Path) -> None:
    """MAJOR-5's third consequence: the outgoing ACTIVE goal and its token.

    Same branch, different previous state: over a goal that was still `active`,
    the old code kept the departed goal's token AND skipped the `superseded`
    entry `/goal B` records — so a replaced goal dropped no in-flight verdict
    (the one thing the token exists for) and left no trace it had been replaced.
    The plain set now behaves exactly as `/goal B` does.
    """
    session = _stub_session(tmp_path)
    session.set_goal("First objective")
    first_token = session.goal_token

    session.set_goal("Second objective")

    assert session.goal == "Second objective"
    assert session.goal_status == "active"
    assert session.goal_token != first_token
    assert [entry["text"] for entry in session.goal_history] == ["First objective"]
    assert [entry["status"] for entry in session.goal_history] == ["superseded"]
    record = read_goal_record(tmp_path)
    assert record is not None and record["goal"] == "Second objective"
    assert record["token"] == session.goal_token


def test_a_plain_set_over_a_settled_goal_resumes_as_the_new_active_goal(
    tmp_path: Path,
) -> None:
    """The same rule across a boot, read by a REAL Session — the `--resume` face.

    On disk this is what the disagreement cost: `_restore_attachment` wins the
    text and `_restore_goal_record` then applies the RECORD's lifecycle on top of
    it, so a resume of the old shape handed the new objective the departed goal's
    `done`.
    """
    from tests.e2e.harness import ScriptedStream, build_session

    directory = tmp_path / "session"
    directory.mkdir()
    written = _stub_session(directory)
    written.set_goal("Ship safely")
    written.mark_goal_done("the judge said so")
    written.set_goal("Land the new billing migration")

    session = build_session(directory, ScriptedStream([]))

    assert session.goal == "Land the new billing migration"
    assert session.goal_status == "active", "the restore must not settle the new goal"
    assert session.goal_token, "a resumed goal is judged, not merely displayed"
    assert [entry["text"] for entry in session.goal_history] == ["Ship safely"]
    assert "<goal>" in _goal_prompt_block(session)


def test_a_goal_restored_from_a_pre_lifecycle_build_mints_a_token(tmp_path: Path) -> None:
    """Agent review MAJOR-3, the R9 migration half — driven through a REAL Session.

    A build that predates the record leaves the goal's text in `attachment.json`
    and writes no `goal.json` at all, which is the shape the fold reports as
    `active`. Reproduced here the way a user meets it: build a session over that
    directory and ask whether anything could ever run on the goal.
    """
    from tests.e2e.harness import ScriptedStream, build_session

    directory = tmp_path / "session"
    directory.mkdir()
    write_session_attachment(directory, team="", agent="", goal="land the OAuth refresh fix")

    session = build_session(directory, ScriptedStream([]))

    assert session.goal == "land the OAuth refresh fix"
    assert session.goal_token, "a restored goal is judged, not merely displayed"
    assert session.goal_status in {"", "active"}


def test_a_record_on_disk_keeps_its_own_token(tmp_path: Path) -> None:
    """The mint is for a goal that never had an identity — not a re-issue.

    A record written by a build that HAS the record carries the token its
    in-flight verdict was captured against, and the restore must hand back the
    same one; re-minting on every resume would drop a verdict the goal is still
    the same goal for (the reasoning `_restore_goal_record` already documents).
    """
    from tests.e2e.harness import ScriptedStream, build_session

    directory = tmp_path / "session"
    directory.mkdir()
    write_session_attachment(directory, team="", agent="", goal="A")
    write_goal_record(
        directory,
        {
            "goal": "A",
            "status": "active",
            "token": "token-from-disk",
            "judge": {"state": "waiting"},
            "history": [],
        },
    )

    session = build_session(directory, ScriptedStream([]))

    assert session.goal == "A"
    assert session.goal_token == "token-from-disk"
