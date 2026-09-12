"""The cut-off taxonomy: what counts as an error, and what names the cause.

The rule the operator asked for is "make sure that errors are properly called
out in all situations". The behaviour these tests pin is the default flip: a
turn cut off by anything the harness cannot attribute to a deliberate stop is an
ERROR carrying a cause, and only positive evidence of a user's own stop records
an interruption. The regression guard is the other direction — a real stop must
never be relabelled a failure, which is the one misclassification the design
calls worse than the bug it fixes.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentEndEvent, Message, ModelSpec, TextContent
from local_operator.incidents import format_cut_off_notice, render_cut_off_reason
from local_operator.session.attention import AttentionStore, conversation_identity
from local_operator.session.frontend_state import JobState
from local_operator.session.session import _default_convert_to_llm
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="mock")


async def _no_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
    """A stream double that never yields: these tests do not run a turn.

    An async GENERATOR rather than a plain callable, because that is what
    ``stream_fn``'s annotation actually requires — a stub that only returns
    ``None`` type-checks as a mistake and would raise the first time a turn
    did run, which is exactly the kind of double that hides a broken test.
    """
    return
    yield  # pragma: no cover — the ``yield`` is what makes this a generator


def _make_session(directory: Path, *, stream: Any = None):
    from local_operator.session.session import Session

    return Session(
        model=MODEL,
        stream_fn=stream or _no_stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
    )


def _rendered(history: object) -> str:
    parts: list[str] = []
    for message in _default_convert_to_llm(list(history)):  # type: ignore[arg-type]
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


# -- the live classifier -----------------------------------------------------


def test_classify_cut_off_leaves_a_deliberate_stop_alone(tmp_path: Path) -> None:
    """Design test 7a: no recorded cause → exactly today's shape.

    ``aborted=True, error=None`` is what a user's Esc produces, and the whole
    taxonomy change rests on this staying true.
    """
    session = _make_session(tmp_path / "sess")
    event = AgentEndEvent(messages=[], aborted=True)
    classed = session._classify_cut_off(event)
    assert classed is event
    assert classed.aborted is True
    assert classed.error is None


def test_classify_cut_off_rewrites_an_involuntary_abort(tmp_path: Path) -> None:
    """Design test 7b: a noted cause becomes an error the old surfaces understand."""
    session = _make_session(tmp_path / "sess")
    session.note_cut_off("runtime-shutdown")
    classed = session._classify_cut_off(AgentEndEvent(messages=[], aborted=True))
    assert classed.aborted is False
    assert classed.error == format_cut_off_notice("runtime-shutdown")
    assert classed.cut_off_cause == "runtime-shutdown"
    assert classed.cut_off == render_cut_off_reason("runtime-shutdown")


def test_classify_cut_off_does_not_overwrite_a_provider_error(tmp_path: Path) -> None:
    """A real provider diagnosis outranks "the runtime went away"."""
    session = _make_session(tmp_path / "sess")
    session.note_cut_off("runtime-killed")
    event = AgentEndEvent(messages=[], aborted=False, error="quota exhausted")
    assert session._classify_cut_off(event) is event


def test_a_deliberate_stop_outranks_a_later_cut_off_note(tmp_path: Path) -> None:
    """The stop rung is followed by the dispose rung, and the stop wins.

    Both converge on the same clean-exit ordering, so the dispose rung ALWAYS
    runs; without this precedence a user's `/stop` would be reported as the
    runtime shutting down.
    """
    session = _make_session(tmp_path / "sess")
    session.note_deliberate_stop()
    session.note_cut_off("runtime-shutdown")
    classed = session._classify_cut_off(AgentEndEvent(messages=[], aborted=True))
    assert classed.aborted is True
    assert classed.error is None


def test_note_cut_off_keeps_the_first_and_most_specific_cause(tmp_path: Path) -> None:
    """An exit is a sequence of rungs; the earliest note is the specific one."""
    session = _make_session(tmp_path / "sess")
    session.note_cut_off("runtime-retired", " (1.0@a → 1.1@b)")
    session.note_cut_off("runtime-shutdown")
    assert session._cut_off_cause == "runtime-retired"
    assert session._cut_off_detail == " (1.0@a → 1.1@b)"


# -- the durable outcome -----------------------------------------------------


@pytest.mark.asyncio
async def test_publish_names_the_cause_for_a_cut_off(tmp_path: Path) -> None:
    """Design test 8: kind, cause and reason all land in the store and journal."""
    from local_operator.session.attention import ATTENTION_CUSTOM_TYPE

    directory = tmp_path / "sessions" / "s1"
    session = _make_session(directory)
    session._attention_run_token = str(uuid.uuid4())
    session._attention_run_settled = False
    session.note_cut_off("runtime-shutdown")
    session._attention_outcome = session._classify_cut_off(AgentEndEvent(messages=[], aborted=True))
    await session._publish_attention_outcome()

    state = AttentionStore().state(conversation_identity(directory))
    assert state["kind"] == "error"
    assert state["cause"] == "runtime-shutdown"
    assert state["reason"] == render_cut_off_reason("runtime-shutdown")
    entry = Transcript(directory).latest_custom(ATTENTION_CUSTOM_TYPE)
    assert entry is not None
    assert entry["cause"] == "runtime-shutdown"
    assert entry["reason"] == state["reason"]


@pytest.mark.asyncio
async def test_publish_keeps_a_deliberate_stop_interrupted(tmp_path: Path) -> None:
    """Design test 9 — the regression guard for the whole taxonomy change."""
    directory = tmp_path / "sessions" / "s2"
    session = _make_session(directory)
    session._attention_run_token = str(uuid.uuid4())
    session._attention_run_settled = False
    session._attention_outcome = AgentEndEvent(messages=[], aborted=True)
    await session._publish_attention_outcome()

    state = AttentionStore().state(conversation_identity(directory))
    assert state["kind"] == "interrupted"
    assert state["cause"] == "user-stop"
    assert state["reason"] == render_cut_off_reason("user-stop")
    incidents = [
        entry
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]
    assert incidents == [], "a deliberate stop is not something to explain to the model"


@pytest.mark.asyncio
async def test_a_restored_cut_off_reaches_the_next_turns_history(tmp_path: Path) -> None:
    """Design test 10: the replay is the model's only account of what happened."""
    directory = tmp_path / "sessions" / "s3"
    directory.mkdir(parents=True)
    token = str(uuid.uuid4())
    transcript = Transcript(directory)
    await transcript.append_custom(
        "attention_started", {"conversation_id": "session/s3", "token": token}
    )
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="partial")])
    )

    session = _make_session(directory)
    assert session._restored_cut_off is not None
    await session.async_init()
    rendered = _rendered(session._context.messages)
    assert "[session incident]" in rendered
    assert "cut-off" in rendered
    assert "do not assume the request completed" in rendered

    # And only once: a second boot must not narrate the same run again.
    second = _make_session(directory)
    await second.async_init()
    incidents = [
        entry
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]
    assert len(incidents) == 1


# -- restored subagent rows (D3) ---------------------------------------------


def _row(job_id: str, status: str = "running") -> JobState:
    return JobState(id=job_id, type="task", status=status, label=job_id)


def _child_dir(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / "children" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_restored_rows_prefer_the_records_settled_outcome(tmp_path: Path) -> None:
    """D3 rule 1: a record that says ``completed`` is a fact, not an inference."""
    from local_operator.session.restored_rows import resolve_restored_rows

    rows = resolve_restored_rows(
        [_row("a")],
        records=[{"job_id": "a", "outcome": "completed", "session_dir": str(tmp_path)}],
    )
    assert rows[0].status == "completed"
    assert rows[0].restored is True
    assert rows[0].cut_off_cause == ""


def test_restored_rows_read_the_childs_tail_when_the_record_did_not_settle(
    tmp_path: Path,
) -> None:
    """D3 rule 2: the child's own journal decides mid-turn from finished."""
    from local_operator.session.restored_rows import resolve_restored_rows

    mid = _child_dir(tmp_path, "mid")
    transcript = Transcript(mid)
    asyncio.run(
        transcript.append_message(
            Message(role="tool", content=[TextContent(text="ok")], tool_call_id="c1")
        )
    )
    done = _child_dir(tmp_path, "done")
    asyncio.run(
        Transcript(done).append_message(
            Message(
                role="assistant", content=[TextContent(text="all finished")], stop_reason="stop"
            )
        )
    )

    rows = resolve_restored_rows(
        [_row("mid"), _row("done")],
        records=[
            {"job_id": "mid", "outcome": None, "session_dir": str(mid)},
            {"job_id": "done", "outcome": None, "session_dir": str(done)},
        ],
    )
    assert rows[0].status == "interrupted"
    assert rows[0].cut_off_cause == "owner-lost"
    assert rows[1].status == "completed"
    assert rows[1].cut_off_cause == ""


def test_restored_rows_without_records_keep_todays_behaviour(tmp_path: Path) -> None:
    """D3 rule 3: no record and no transcript is still ``interrupted``."""
    from local_operator.session.restored_rows import resolve_restored_rows

    rows = resolve_restored_rows([_row("orphan")])
    assert rows[0].status == "interrupted"
    assert rows[0].cut_off_cause == "owner-lost"


def test_a_settled_row_is_never_relitigated(tmp_path: Path) -> None:
    """A terminal status the last runtime settled is left exactly as it was."""
    from local_operator.session.restored_rows import resolve_restored_rows

    rows = resolve_restored_rows(
        [_row("c", "completed")],
        records=[{"job_id": "c", "outcome": "failed", "session_dir": str(tmp_path)}],
    )
    assert rows[0].status == "completed"


# -- the retirement latch (D4) ----------------------------------------------


class _LatchHost:
    """``ServingSessionHandle.begin_retire``'s real logic over a stub predicate.

    The method is what the two retire paths and the quiet idle exit call, and
    its contract is one synchronous step: commit iff idle, refuse admissions
    from that instant. Exercising it against a stub keeps the assertion on the
    latch rather than on a runtime boot, which the e2e stage covers.
    """

    from local_operator.session.runtime.serving import ServingSessionHandle as _H

    begin_retire = _H.begin_retire
    _retiring_refusal = _H._retiring_refusal

    #: Typed ``Any`` on purpose: the real attribute holds a ``Session``, and the
    #: tests below substitute a recorder that only implements ``note_cut_off``.
    _session: Any = None

    def __init__(self, *, reason: str = "") -> None:
        self._retiring_cause = ""
        self.reason = reason
        self.notes: list[tuple[str, str]] = []

    def may_refresh(self) -> str:
        return self.reason


class _NoteSession:
    def __init__(self) -> None:
        self.notes: list[tuple[str, str]] = []

    def note_cut_off(self, cause: str, detail: str = "") -> None:
        self.notes.append((cause, detail))


def test_begin_retire_commits_when_idle_and_names_the_cause() -> None:
    host = _LatchHost()
    session = _NoteSession()
    host._session = session
    assert host.begin_retire("runtime-retired", " (1.0@a → 1.1@b)") is True
    assert host._retiring_cause == "runtime-retired"
    assert session.notes == [("runtime-retired", " (1.0@a → 1.1@b)")]
    assert "runtime-retired" in host._retiring_refusal()


def test_begin_retire_refuses_when_a_turn_is_held() -> None:
    """Design test 11: a busy runtime never commits, so no admission is refused."""
    host = _LatchHost(reason="a turn is running")
    assert host.begin_retire("runtime-retired") is False
    assert host._retiring_cause == ""


def test_begin_retire_survives_a_failing_predicate() -> None:
    """An unanswerable probe keeps the runtime rather than retiring unguarded."""

    class _Boom(_LatchHost):
        def may_refresh(self) -> str:
            raise RuntimeError("boom")

    host = _Boom()
    assert host.begin_retire("runtime-retired") is False
    assert host._retiring_cause == ""


# -- the torn-install classification (D4/§5.2) -------------------------------


def test_classify_import_failure_names_a_moved_install(monkeypatch) -> None:
    """Design test 13: an ImportError plus a moved stamp is an install race."""
    from local_operator import update

    boot = update.BuildStamp(version="1.0.0", source_ref="aaa")
    current = update.BuildStamp(version="1.1.0", source_ref="bbb")
    monkeypatch.setattr(update, "installed_build", lambda *_a, **_k: current)
    exc = ImportError(
        "cannot import name '_journal_injection_ids' from 'local_operator.session.transcript'"
    )
    reason = update.classify_import_failure(exc, "local_operator.mobile.durable", boot=boot)
    assert reason is not None
    assert "being replaced on disk" in reason
    assert "1.0.0@aaa" in reason and "1.1.0@bbb" in reason


def test_classify_import_failure_ignores_a_genuine_packaging_bug(monkeypatch) -> None:
    """Same stamp → the miss is ours; it must stay an ordinary traceback."""
    from local_operator import update

    boot = update.BuildStamp(version="1.0.0", source_ref="aaa")
    monkeypatch.setattr(update, "installed_build", lambda *_a, **_k: boot)
    exc = ImportError("cannot import name 'nope' from 'local_operator.session.transcript'")
    assert (
        update.classify_import_failure(exc, "local_operator.session.transcript", boot=boot) is None
    )


def test_classify_import_failure_ignores_an_unrelated_error(monkeypatch) -> None:
    from local_operator import update

    boot = update.BuildStamp(version="1.0.0", source_ref="aaa")
    monkeypatch.setattr(
        update, "installed_build", lambda *_a, **_k: update.BuildStamp("2.0.0", "zzz")
    )
    assert update.classify_import_failure(ValueError("nope"), "local_operator.x", boot=boot) is None
    assert (
        update.classify_import_failure(
            ImportError("cannot import name 'requests' from 'requests'"),
            "requests",
            boot=boot,
        )
        is None
    )
    # And with no baseline at all there is nothing to compare: refuse to guess.
    assert update.classify_import_failure(ImportError("x"), "local_operator.x", boot=None) is None


# -- the dispose route (review round 1, BLOCKER-1) ---------------------------
#
# A bare `/stop` on a TUI-OWNED session does not travel through the `stop`
# control op: it disposes the session in-process. `Session.dispose()` notes
# `disposed` unconditionally (correct for the teardown rungs, wrong for a
# deliberate one), so whether the user's own cancel is reported as a failure is
# decided by ONE thing — whether the caller recorded the verdict first. The
# pre-existing guard set `_attention_outcome` directly and the e2e cell used
# `stop_session`, so neither exercised this route at all; these two tests pin
# both directions of it against the REAL `dispose()`, which is what makes the
# classification difference the fix rests on observable rather than assumed.


async def _dispose_with_unsent_run(directory: Path, *, deliberate: bool) -> None:
    """Boot a real session with an in-flight run, then dispose it."""
    directory.mkdir(parents=True, exist_ok=True)
    session = _make_session(directory)
    await session.async_init()
    # What `_run_turn` does at the head of a turn: a token is minted and the run
    # is left UNSETTLED, so the dispose publisher sees a turn that never
    # reported an outcome. Driving this by state rather than by running a turn
    # keeps the test about the classification, which is the seam in question.
    session._attention_run_token = str(uuid.uuid4())
    session._attention_run_settled = False
    if deliberate:
        session.note_deliberate_stop()
    await session.dispose()


@pytest.mark.asyncio
async def test_the_dispose_route_publishes_the_users_own_stop_as_an_interruption(
    tmp_path: Path,
) -> None:
    """BLOCKER-1: the TUI's `/stop` must not come back as `kind=error`."""
    directory = tmp_path / "sessions" / "stopped"
    await _dispose_with_unsent_run(directory, deliberate=True)

    state = AttentionStore().state(conversation_identity(directory))
    assert state["kind"] == "interrupted", state
    assert state["cause"] == "user-stop", state
    assert state["reason"] == render_cut_off_reason("user-stop")
    incidents = [
        entry
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]
    assert incidents == [], "the user's own stop is not a failure to explain"


@pytest.mark.asyncio
async def test_an_unnoted_dispose_is_still_a_cut_off_error(tmp_path: Path) -> None:
    """The control, and the reason this is a caller's verdict rather than a guess.

    An INVOLUNTARY teardown (a reload, an unmount, ``_mobile_teardown``) is a
    turn cut off under the user, and it must keep reading as one. Without this
    half, "note the stop everywhere" would look equivalent to "never report a
    disposed turn", and the two differ on exactly the teardowns nobody asked
    for.
    """
    directory = tmp_path / "sessions" / "torn"
    await _dispose_with_unsent_run(directory, deliberate=False)

    state = AttentionStore().state(conversation_identity(directory))
    assert state["kind"] == "error", state
    assert state["cause"] == "disposed", state
    assert state["reason"], "a cut-off must name a reason"


@pytest.mark.asyncio
async def test_the_handles_cancel_rungs_record_the_deliberate_verdict(tmp_path: Path) -> None:
    """``abort`` (the phone's stop button) and ``cancel`` are deliberate too.

    Both were one teardown away from the same false error: each ends the turn
    aborted with no error, and the next dispose would have published it as a
    cut-off. The verdict is recorded at the act, so the ordering cannot matter.
    Real session, real handle — the flag asserted here is the one the
    classification reads.
    """
    from local_operator.session.runtime.serving import ServingSessionHandle

    directory = tmp_path / "sessions" / "cancel-rungs"
    directory.mkdir(parents=True)
    session = _make_session(directory)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))

    await handle.abort()
    assert session._deliberate_stop_noted is True
    assert session._cut_off_cause == ""

    # And the supervisor's boundary-respecting rung, which ends the turn the
    # same way one tool later.
    session._deliberate_stop_noted = False
    receipt = await handle.cancel_gracefully("cancelled by supervisor")
    assert session._deliberate_stop_noted is True, receipt


# -- an aborted turn that also failed (review round 1, MINOR-1) --------------


def test_an_aborted_turn_that_also_failed_keeps_the_providers_diagnosis(
    tmp_path: Path,
) -> None:
    """`_classify_cut_off`'s own contract, which its guard used to break.

    The loop emits a provider failure on an aborted turn itself
    (``aborted=True, error=<stream_error>``), and the guard only spared
    ``error and not aborted`` — so a cut-off that coincided with a real provider
    fault overprinted the vendor's specific message with a generic sentence.
    """
    session = _make_session(tmp_path / "sess")
    session.note_cut_off("runtime-shutdown")
    event = AgentEndEvent(messages=[], aborted=True, error="quota exhausted: your credit is gone")
    classed = session._classify_cut_off(event)
    assert classed is event
    assert classed.error == "quota exhausted: your credit is gone"
    assert classed.cut_off_cause == ""


@pytest.mark.asyncio
async def test_the_durable_outcome_agrees_with_an_aborted_provider_error(
    tmp_path: Path,
) -> None:
    """And the STORE agrees, which is why the publisher reads the event.

    The publisher used to key the cut-off on the local flag, so the store named
    the cut-off sentence while the transcript held the provider's — one turn,
    two stories, on the surface that outlives the process.
    """
    directory = tmp_path / "sessions" / "aborted-error"
    session = _make_session(directory)
    session._attention_run_token = str(uuid.uuid4())
    session._attention_run_settled = False
    session.note_cut_off("runtime-shutdown")
    session._attention_outcome = session._classify_cut_off(
        AgentEndEvent(messages=[], aborted=True, error="quota exhausted: your credit is gone")
    )
    await session._publish_attention_outcome()

    state = AttentionStore().state(conversation_identity(directory))
    assert state["kind"] == "error"
    assert state["cause"] == ""
    assert state["reason"] == "quota exhausted: your credit is gone"
    incidents = [
        entry
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]
    assert incidents == [], "a provider error is not a cut-off incident"


# -- the cut-off the dying runtime could not narrate (QA Q-2) ---------------


@pytest.mark.asyncio
async def test_a_cut_off_the_dying_runtime_could_not_journal_is_narrated_on_restore(
    tmp_path: Path,
) -> None:
    """The update/shutdown family reaches the MODEL too, one boot later.

    A runtime that publishes its own outcome and then dies cannot journal it:
    ``journal_incident`` refuses once ``_disposed`` is set, and the dispose rung
    sets that before the turn's ``finally`` publishes. The saved marker is
    replayed verbatim, so the successor is the only writer left that can tell
    the model — and it must, or "continue" after an update means re-guessing a
    request that was cut in half (QA round 1, Q-2; review MINOR-2).
    """
    from local_operator.session.attention import ATTENTION_CUSTOM_TYPE

    directory = tmp_path / "sessions" / "shutdown"
    directory.mkdir(parents=True)
    token = str(uuid.uuid4())
    transcript = Transcript(directory)
    await transcript.append_custom(
        "attention_started",
        {"conversation_id": conversation_identity(directory), "token": token},
    )
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="half a reply")])
    )
    # The dying runtime's own marker: `_publish_attention_outcome` wrote this
    # immediately before its process ended.
    await transcript.append_custom(
        ATTENTION_CUSTOM_TYPE,
        {
            "conversation_id": conversation_identity(directory),
            "token": token,
            "anchor": "completion-1",
            "kind": "error",
            "cause": "runtime-shutdown",
            "reason": render_cut_off_reason("runtime-shutdown"),
        },
    )
    # The run record of the process that wrote it, now dead.
    record = tmp_path / "runtimes" / f"{token}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{}", encoding="utf-8")

    session = _make_session(directory)
    await session.async_init()
    try:
        incidents = [
            entry
            for entry in Transcript(directory).entries()
            if entry.payload.get("custom_type") == "session_incident"
        ]
        assert len(incidents) == 1, "the successor did not narrate the cut-off"

        rendered = _rendered(session._context.messages)
        assert "[session incident]" in rendered
        assert "cut-off" in rendered

        # Once per token: a second boot must not narrate it again.
        again = _make_session(directory)
        await again.async_init()
        try:
            assert (
                len(
                    [
                        entry
                        for entry in Transcript(directory).entries()
                        if entry.payload.get("custom_type") == "session_incident"
                    ]
                )
                == 1
            )
        finally:
            await again.dispose()
    finally:
        await session.dispose()


# -- the OWNER path's restore (review round 1, UX U2) ------------------------


async def _seed_roster_sidecar(directory: Path, children: Path) -> None:
    """The sidecar exactly as ``_persist_subagent_roster`` writes it.

    The ``jobs`` rows go through ``_subagent_job_row``, the ALLOWLIST projection
    the owner persists — not ``JobState.model_dump()``, which carries
    frontend-only fields the strict ``AsyncJob`` sidecar rejects (a probe that
    seeded frontend rows here exercised nothing at all: every row was dropped as
    malformed and the owner path restored an EMPTY roster).
    """
    from local_operator.harness.jobs import AsyncJob
    from local_operator.session.session import (
        SUBAGENT_ROSTER_SIDECAR,
        _subagent_job_row,
    )

    for name in ("mid", "done"):
        (children / name).mkdir(parents=True, exist_ok=True)
    # A child cut off mid-turn: its last row is a tool RESULT with nothing after.
    await Transcript(children / "mid").append_message(
        Message(role="tool", content=[TextContent(text="ok")], tool_call_id="call-1")
    )
    # A child that finished: its last row is a settled assistant message.
    await Transcript(children / "done").append_message(
        Message(role="assistant", content=[TextContent(text="all finished")])
    )

    def row(job_id: str, label: str) -> dict[str, Any]:
        return _subagent_job_row(
            AsyncJob(id=job_id, type="task", status="running", label=label, start_time=0.0)
        )

    payload = {
        "version": 1,
        "generation": 7,
        "jobs": [
            row("job-settled", "scan filings"),
            row("job-midturn", "draft the memo"),
            row("job-finished", "collect sources"),
            row("job-orphan", "orphan child"),
        ],
        "records": [
            {
                "job_id": "job-settled",
                "outcome": "completed",
                "settled_at": 2.0,
                "session_dir": str(children / "settled"),
            },
            {
                "job_id": "job-midturn",
                "outcome": None,
                "settled_at": None,
                "session_dir": str(children / "mid"),
            },
            {
                "job_id": "job-finished",
                "outcome": None,
                "settled_at": None,
                "session_dir": str(children / "done"),
            },
        ],
        "accounting": [],
    }
    (children / "unused").mkdir(parents=True, exist_ok=True)
    (directory / SUBAGENT_ROSTER_SIDECAR).write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.asyncio
async def test_the_owner_path_restores_the_records_resolved_rows(tmp_path: Path) -> None:
    """UX U2: the successor's restore must not flatten what the records settled.

    The cold viewer resolves a restored row against the roster records; the
    runtime that opens a moment later restores the SAME rows into
    ``AsyncJobManager``, and its table is what the session publishes. Resolved on
    one path only, the user saw the good rows for under a second — measured at
    t+1.2 s ``completed|interrupted|completed|interrupted``, by t+1.9 s all four
    back to a blanket ``interrupted`` with the cause dropped. Both writers now
    run the same resolver, so this asserts the owner's table AND the wire shape
    the dock paints.
    """
    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR  # noqa: F401

    directory = tmp_path / "sessions" / "owner-restore"
    directory.mkdir(parents=True)
    await _seed_roster_sidecar(directory, tmp_path / "children")

    from tests.e2e.harness import ScriptedStream, build_session, text_turn

    session = build_session(directory, ScriptedStream([text_turn("ok")]))
    await session.async_init()
    try:
        rows = {job.id: job for job in session.jobs.list()}
        assert rows["job-settled"].status == "completed", "a settled record is a fact"
        assert rows["job-finished"].status == "completed", "the child's own tail settled it"
        assert rows["job-midturn"].status == "interrupted"
        assert rows["job-midturn"].cut_off_cause == "owner-lost"
        assert rows["job-orphan"].status == "interrupted"
        assert rows["job-orphan"].cut_off_cause == "owner-lost"
        for job in rows.values():
            assert job.restored is True

        # The wire rows the dock reads, which is where the cause used to be
        # dropped even when the resolver had set it.
        wire = {job.id: job for job in (JobState.from_job(job) for job in session.jobs.list())}
        assert wire["job-midturn"].cut_off_cause == "owner-lost"
        assert wire["job-settled"].cut_off_cause == ""
    finally:
        await session.dispose()

    # And the SIDECAR is unchanged: the strict ``AsyncJob`` rows an older owner
    # validates must not grow a field it does not know, or it drops the row.
    persisted = json.loads((directory / "subagent-roster.v1.json").read_text(encoding="utf-8"))
    assert persisted["jobs"], "the fixture wrote no rows"
    assert all("cut_off_cause" not in row for row in persisted["jobs"])
