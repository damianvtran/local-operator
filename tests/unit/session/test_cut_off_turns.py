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
import uuid
from pathlib import Path

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)
from local_operator.incidents import format_cut_off_notice, render_cut_off_reason
from local_operator.session.attention import AttentionStore, conversation_identity
from local_operator.session.frontend_state import JobState
from local_operator.session.session import _default_convert_to_llm
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="mock")


def _make_session(directory: Path, *, stream: object | None = None):
    from local_operator.session.session import Session

    return Session(
        model=MODEL,
        stream_fn=stream or (lambda *_args, **_kwargs: None),
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
    from local_operator.session.attached import _restored_job_rows

    rows = _restored_job_rows(
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
    from local_operator.session.attached import _restored_job_rows

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

    rows = _restored_job_rows(
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
    from local_operator.session.attached import _restored_job_rows

    rows = _restored_job_rows([_row("orphan")])
    assert rows[0].status == "interrupted"
    assert rows[0].cut_off_cause == "owner-lost"


def test_a_settled_row_is_never_relitigated(tmp_path: Path) -> None:
    """A terminal status the last runtime settled is left exactly as it was."""
    from local_operator.session.attached import _restored_job_rows

    rows = _restored_job_rows(
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

    def __init__(self, *, reason: str = "") -> None:
        self._retiring_cause = ""
        self._session = None
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
