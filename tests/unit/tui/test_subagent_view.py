"""The full-page subagent view.

The surface this replaced was a small centred modal that painted its own
one-line rows. What was wrong with it was not styling: prose was clipped at
the card's right edge instead of wrapped, a long run stopped mid-list with
nothing saying more existed, the conversation behind it was blacked out, and
no row on screen said how to get out. So the assertions here are about those
four things and about the mode's contract — the composer refuses input while
the page is open, and leaving puts the conversation back exactly as it was.

The fold (:func:`fold_trajectory`) is a pure function of the retained event
list and is tested as one; everything else is driven through a mounted
``OperatorApp``, because a mode that is half-wired passes every widget-level
test there is.
"""

from __future__ import annotations

from typing import Any

import pytest
from rich.cells import cell_len

from local_operator.tui.app import SUBAGENT_LAYOUT_CLASS, OperatorApp
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.subagent_panel import GLYPH_DONE
from local_operator.tui.widgets.subagent_view import (
    COLLAPSE_AFFORDANCE,
    EXPAND_HINT,
    INSTRUCTION_ROWS,
    LEDGER_GONE_NOTE,
    TRAJECTORY_MAX_EVENTS,
    TRUNCATION_NOTE,
    InstructionBlock,
    SubagentView,
    fold_trajectory,
)
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock

from .test_band_panels import FakeSession, _async_factory, _fake_jobs, _Job


def _text(mid: str, body: str) -> list[dict[str, Any]]:
    """A complete assistant message, the way the child stream emits one."""
    return [
        {"type": "message_start", "message": {"role": "assistant", "id": mid}},
        {"type": "message_update", "message": {"role": "assistant", "id": mid}, "delta": body},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "id": mid,
                "content": [{"type": "text", "text": body}],
            },
        },
    ]


def _call(call_id: str, name: str, **args: Any) -> dict[str, Any]:
    return {
        "type": "tool_execution_start",
        "tool_call_id": call_id,
        "tool_name": name,
        "args": args,
    }


def _result(call_id: str, name: str, text: str = "", is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "tool_execution_end",
        "tool_call_id": call_id,
        "tool_name": name,
        "is_error": is_error,
        "result": {
            "tool_call_id": call_id,
            "tool_name": name,
            "content": [{"type": "text", "text": text}],
            "is_error": is_error,
        },
    }


#: One ordinary child run: a sentence, a tool that worked, a tool that failed.
TRAJECTORY = [
    {"type": "agent_start"},
    *_text("m1", "Reading the ingest path."),
    _call("c1", "read", path="pipeline/ingest.py"),
    _result("c1", "read", "def ingest(batch):\n    ..."),
    _call("c2", "bash", command="pytest -q"),
    _result("c2", "bash", "2 failed", is_error=True),
    *_text("m2", "Two tests fail on the retry budget."),
]


def _job_with(trajectory: list[dict[str, Any]], status: str = "completed") -> Any:
    job = _Job("sub-1", "audit the ingest path", status=status)
    if status != "running":
        job.settled_at = job.start_time + 42
    job.trajectory = trajectory
    return job


# -- the fold ---------------------------------------------------------------


def test_fold_produces_prose_and_tool_rows_in_call_order() -> None:
    """Deltas accumulate into one row per message and tools keep their outcome.

    Order is the child's, not the results': one assistant turn issues a batch
    and the answers can come back in any order, so a fold keyed on the result
    would reorder the transcript it is meant to be replaying.
    """
    entries = fold_trajectory(TRAJECTORY, settled=True)
    assert [(entry.kind, entry.tool_name or entry.text) for entry in entries] == [
        ("text", "Reading the ingest path."),
        ("tool", "read"),
        ("tool", "bash"),
        ("text", "Two tests fail on the retry budget."),
    ]
    assert entries[1].outcome == "success"
    assert entries[2].outcome == "error"
    assert entries[2].result_text == "2 failed"


def test_fold_keeps_a_live_call_live_and_settles_it_when_the_job_is_over() -> None:
    """A call with no end event means two different things, and the difference
    is whether the child is still running: still executing, or killed mid-turn.
    Painting the second as running would claim work inside a dead session."""
    events = [*_text("m1", "working"), _call("c1", "bash", command="sleep 60")]
    assert fold_trajectory(events, settled=False)[-1].outcome is None
    assert fold_trajectory(events, settled=True)[-1].outcome == "interrupted"


def test_fold_reports_the_childs_own_failure_and_its_retries() -> None:
    """A failed child's page must say WHY. The reason arrives as `agent_end`,
    and before this it lived only on the band row the reader had left."""
    events = [
        {"type": "retry_start", "attempt": 2, "error": "429", "fallback_model": "x/mini"},
        {"type": "agent_end", "error": "context window exceeded"},
    ]
    texts = [entry.text for entry in fold_trajectory(events, settled=True)]
    assert texts == ["retry 2: 429 → falling back to x/mini", "context window exceeded"]
    assert [entry.notice_kind for entry in fold_trajectory(events, settled=True)] == [
        "warning",
        "error",
    ]


def test_fold_admits_that_the_engine_dropped_the_start_of_a_long_run() -> None:
    """At the retention cap the beginning of the run is GONE, and a transcript
    that hides that invites conclusions drawn from a deleted opening."""
    short = fold_trajectory(TRAJECTORY, settled=True)
    assert TRUNCATION_NOTE not in [entry.text for entry in short]

    capped = [*_text("m0", "first step")]
    while len(capped) < TRAJECTORY_MAX_EVENTS:
        capped.append({"type": "turn_start"})
    folded = fold_trajectory(capped, settled=True)
    assert folded[0].text == TRUNCATION_NOTE
    assert folded[0].notice_kind == "note"


def test_fold_survives_junk_without_raising() -> None:
    """Entries come from another session's internals. This surface is
    observability, and observability may not take the app down."""
    junk: list[Any] = [
        None,
        "not an event",
        {"type": "message_end"},  # no message at all
        {"type": "tool_execution_end", "tool_call_id": "never-started"},
        {"type": "message_end", "message": {"role": "user", "content": []}},
        {"type": "notice", "text": ""},
    ]
    assert fold_trajectory(junk, settled=True) == []


# -- the page ---------------------------------------------------------------


async def _open(pilot: Any, app: OperatorApp, job: Any) -> SubagentView:
    for _ in range(80):
        await pilot.pause()
        if app._session is not None:
            break
    app._append_block(UserBlock("audit the ingest path"))
    app._refresh_band()
    await pilot.pause()
    app._open_subagent_view(str(job.id))
    await pilot.pause()
    return app.query_one(SubagentView)


@pytest.mark.asyncio
async def test_the_page_renders_the_childs_messages_and_tool_calls() -> None:
    """The child's work is rendered with the MAIN conversation's blocks, which
    is the whole promise of the redesign: prose is an AssistantBlock that
    wraps, a tool is a ToolCard carrying its own outcome, and a failure is the
    same error row the live transcript would have drawn."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        blocks = view._body.blocks()
        assert [type(block).__name__ for block in blocks] == [
            "AssistantBlock",
            "ToolCard",
            "ToolCard",
            "AssistantBlock",
        ]
        cards = [block for block in blocks if isinstance(block, ToolCard)]
        assert [card.tool_name for card in cards] == ["read", "bash"]
        assert [card._state for card in cards] == ["success", "error"]
        # No duration is invented for a call this app never timed.
        assert all(card._duration is None for card in cards)
        prose = [block for block in blocks if isinstance(block, AssistantBlock)]
        assert prose[0].text() == "Reading the ingest path."

        page = " ".join(view.rendered_rows())
        assert "audit the ingest path" in page  # the title names the subagent
        assert "Two tests fail on the retry budget." in page


@pytest.mark.asyncio
async def test_the_page_states_the_way_out_and_that_it_is_read_only() -> None:
    """The complaint that started this was "there is no stated way out". The
    hint is chrome, not scrollback, so it cannot scroll away."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        hint = view.rendered_rows()[-1]
        assert "esc" in hint
        assert "back to conversation" in hint
        assert "read-only" in hint


@pytest.mark.asyncio
async def test_the_way_out_survives_a_terminal_too_narrow_for_the_sentence() -> None:
    """Every rung of the shed ladder keeps `esc`. A page that drops the only
    key that leaves it is the bug, not a tidy narrow layout."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        for width in (120, 60, 40, 24, 12, 4):
            assert "esc" in view._hint_row(width).plain, width


@pytest.mark.asyncio
async def test_the_composer_refuses_input_while_the_page_is_open() -> None:
    """Read-only, and visibly so: the editor keeps whatever was typed, takes
    no more of it, and cannot hold focus — a caret blinking in a field that
    ignores every key is the most misleading thing this mode could paint."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one(Editor)
        editor.text = "half-written prompt"
        app._refresh_band()
        await pilot.pause()
        app._open_subagent_view("sub-1")
        await pilot.pause()

        assert editor.read_only is True
        assert editor.can_focus is False
        assert app.screen.has_class(SUBAGENT_LAYOUT_CLASS)
        before = editor.text
        await pilot.press("x", "y", "z")
        assert editor.text == before == "half-written prompt"


@pytest.mark.asyncio
async def test_leaving_restores_the_conversation_and_the_composer() -> None:
    """Esc goes back, and back means back: the main transcript was only
    hidden, so its blocks, and the half-typed prompt, are exactly as left."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._append_block(UserBlock("audit the ingest path"))
        app._append_block(NoticeBlock("subagent started", "info"))
        editor = app.query_one(Editor)
        editor.text = "half-written prompt"
        transcript = app._transcript_view()
        before = list(transcript.blocks())
        app._refresh_band()
        await pilot.pause()

        app._open_subagent_view("sub-1")
        await pilot.pause()
        assert transcript.display is False

        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(SubagentView)
        assert transcript.display is True
        assert transcript.blocks() == before
        assert editor.read_only is False
        assert editor.can_focus is True
        assert editor.text == "half-written prompt"
        assert not app.screen.has_class(SUBAGENT_LAYOUT_CLASS)
        assert session.aborts == [], "esc left the page; it must not stop the parent's turn"


@pytest.mark.asyncio
async def test_the_page_follows_a_running_subagent() -> None:
    """A running child keeps working after the page opens. The old modal
    snapshotted the trajectory at open and then sat still."""
    job = _Job("sub-1", "audit the ingest path", status="running")
    job.trajectory = [*_text("m1", "Reading the ingest path.")]
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        # Prose, then the live tail row that terminates a running page.
        prose, tail = view._body.blocks()
        assert isinstance(prose, AssistantBlock)
        assert isinstance(tail, NoticeBlock)

        job.trajectory.append(_call("c1", "bash", command="pytest -q"))
        app._refresh_band()
        await pilot.pause()
        blocks = view._body.blocks()
        assert isinstance(blocks[1], ToolCard), [type(b).__name__ for b in blocks]
        assert isinstance(blocks[2], NoticeBlock)
        assert blocks[1]._state == "running"  # still going, and it says so
        # The prose block is the SAME widget, not a re-render of it. A refresh
        # that rebuilds the page throws away the reader's scroll position and
        # every expanded card; the diff exists to make that impossible.
        assert blocks[0] is prose

        job.trajectory.append(_result("c1", "bash", "12 passed"))
        job.status = "completed"
        job.settled_at = job.start_time + 12
        app._refresh_band()
        await pilot.pause()
        # The tail row is gone with the run, and the card was updated in place
        # rather than duplicated behind a second copy.
        settled_prose, card = view._body.blocks()
        assert isinstance(card, ToolCard)
        assert card._state == "success"
        assert settled_prose is prose
        assert "completed" in view.rendered_rows()[0]


@pytest.mark.asyncio
async def test_a_swept_job_says_so_instead_of_claiming_to_be_running() -> None:
    """Retention evicts settled jobs. A page open across that boundary must
    not keep reporting a child that the ledger no longer has."""
    session = FakeSession()
    session.jobs = _fake_jobs()  # nothing on the ledger at all
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._open_subagent_view("sub-gone")
        await pilot.pause()
        view = app.query_one(SubagentView)
        rows = view.rendered_rows()
        # No glyph, no clock, no count: all three would be invented, and the
        # fallthrough glyph was the app's own success tick.
        assert rows[0] == "Subagent · sub-gone  no longer on the ledger"
        assert GLYPH_DONE not in rows[0]
        assert LEDGER_GONE_NOTE in " ".join(rows)


@pytest.mark.asyncio
async def test_opening_another_subagent_retargets_the_same_page() -> None:
    """The band stays live under the page precisely so a reader can hop
    between children; hopping must not stack two pages or leave the previous
    child's rows underneath the new one's."""
    first = _job_with([*_text("m1", "first child")], status="completed")
    second = _Job("sub-2", "second subagent", status="completed")
    second.settled_at = second.start_time + 3
    second.trajectory = [*_text("m9", "second child")]
    session = FakeSession()
    session.jobs = _fake_jobs(first, second)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, first)
        assert "first child" in " ".join(view.rendered_rows())

        app._open_subagent_view("sub-2")
        await pilot.pause()
        assert len(app.query(SubagentView)) == 1
        page = " ".join(app.query_one(SubagentView).rendered_rows())
        assert "second child" in page
        assert "first child" not in page


@pytest.mark.asyncio
async def test_a_long_transcript_scrolls_instead_of_stopping() -> None:
    """The reported frame simply ended mid-list with no scrollbar and no way
    to reach the rest. The body is a real transcript, so it scrolls, and it
    opens at the BOTTOM — the latest step is the one being watched for."""
    events: list[dict[str, Any]] = []
    for index in range(40):
        events.append(_call(f"c{index}", "read", path=f"pipeline/step_{index}.py"))
        events.append(_result(f"c{index}", "read", "…"))
    job = _job_with(events)
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        await pilot.pause()
        body = view._body
        assert len(body.blocks()) == 40
        assert body.virtual_size.height > body.size.height, "premise: this overflows"
        assert body.show_vertical_scrollbar is True
        assert isinstance(body, TranscriptView)
        # Focus is on the body, so the keys the hint advertises reach it.
        assert app.focused is body
        assert body.scroll_offset.y >= body.virtual_size.height - body.size.height - 2


#: Both halves of the reload switch: `/new` drops the context, `/resume` keeps
#: it. The page's answer must not depend on which - it is a window onto a job
#: ledger that BOTH dispose, so a page left standing would report `gone` for
#: every job in it. Parametrized rather than asserted once, because the two
#: paths diverge inside `_reload_session` and only one of them was covered.
@pytest.mark.parametrize("keep_context", [False, True])
@pytest.mark.asyncio
async def test_replacing_the_session_leaves_the_page_first(keep_context: bool) -> None:
    """`/new` and `/resume` dispose the session, and the job ledger with it.
    A page left standing would report `gone` for a child it had been reading a
    moment earlier, over a conversation it no longer describes — with the
    composer still refusing input."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        assert view.is_mounted

        # The seam `/new` and `/resume` both reach; the commands themselves need
        # a resume factory this fake host does not have, so the switch is driven
        # directly - the page's behaviour is what is under test, not the command
        # plumbing that reaches it.
        app.run_worker(app._reload_session(keep_context=keep_context), thread=False)
        for _ in range(80):
            await pilot.pause()
            if not app.query(SubagentView):
                break
        assert not app.query(SubagentView)
        assert app.query_one(Editor).read_only is False
        assert not app.screen.has_class(SUBAGENT_LAYOUT_CLASS)


def test_fold_survives_a_scalar_content_where_a_list_belongs() -> None:
    """`content` is type-checked, not truthiness-checked: a scalar there is
    not falsy, and iterating one raised out of the 1 Hz refresh timer — a
    repeating handler exception rather than the skipped row this promises."""
    junk: list[Any] = [
        {"type": "message_end", "message": {"role": "assistant", "id": "m", "content": 5}},
        {"type": "tool_execution_start", "tool_call_id": "c", "tool_name": "read", "args": {}},
        {"type": "tool_execution_end", "tool_call_id": "c", "result": {"content": 7}},
    ]
    entries = fold_trajectory(junk, settled=True)
    assert [entry.kind for entry in entries] == ["tool"]
    assert entries[0].result_text == ""
    assert fold_trajectory(object(), settled=True) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_capped_run_keeps_what_it_showed_when_the_window_rolls_past_it() -> None:
    """The retained trajectory is a ROLLING window: past the cap the engine
    deletes the oldest events, so re-folding hands back a first message whose
    opening deltas are gone. Mirroring that would make the page lose content
    as the child works — the reader watched it arrive, and the engine
    forgetting it does not make it false."""
    job = _Job("sub-1", "long run", status="running")
    job.trajectory = [*_text("m1", "the opening paragraph"), *_text("m2", "the second")]
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        prose = view._body.blocks()[0]
        assert isinstance(prose, AssistantBlock)
        assert prose.text() == "the opening paragraph"
        # The window rolls: every event of the first message is evicted.
        del job.trajectory[:3]
        app._refresh_band()
        await pilot.pause()
        kept = view._body.blocks()[0]
        assert kept is prose
        assert isinstance(kept, AssistantBlock)
        assert kept.text() == "the opening paragraph"


@pytest.mark.asyncio
async def test_an_append_during_the_mode_lands_in_the_conversation() -> None:
    """Two `TranscriptView`s exist while the page is open, and the parent's
    turn keeps streaming into the hidden one. Resolving a transcript by TYPE
    would put the parent's next block on the child's page (or raise); this is
    the contract `_transcript_view()` exists for."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        block = NoticeBlock("the parent kept working", "info")
        app._append_block(block)
        await pilot.pause()
        assert block in app._transcript_view().blocks()
        assert block not in view._body.blocks()


@pytest.mark.asyncio
async def test_esc_stops_the_agent_once_the_page_is_gone() -> None:
    """The half of the precedence rule that can silently rot: the first Esc
    leaves, the NEXT one means what it always meant."""
    streaming: list[bool] = []

    class _Session(FakeSession):
        @property
        def is_streaming(self) -> bool:
            return bool(streaming)

    session = _Session()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(pilot, app, _job_with(TRAJECTORY))
        await pilot.press("escape")
        await pilot.pause()
        assert session.aborts == []
        streaming.append(True)
        await pilot.press("escape")
        await pilot.pause()
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_ctrl_l_returns_to_the_conversation_before_clearing_it() -> None:
    """Ctrl+L is global and acts on the CONVERSATION. Wiping a transcript the
    user cannot currently see would read as the key having done nothing."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(pilot, app, _job_with(TRAJECTORY))
        await pilot.press("ctrl+l")
        await pilot.pause()
        assert not app.query(SubagentView)
        assert app._transcript_view().display is True
        assert app.query_one(Editor).read_only is False


@pytest.mark.asyncio
async def test_the_page_opens_with_the_instruction_the_parent_delegated() -> None:
    """The page used to open on the child's first reply, so it showed an agent
    working with no statement of what it had been asked to do — the one thing a
    reader cannot infer from the rest of the transcript. The instruction is the
    USER turn of this conversation and gets the block a user turn gets."""
    job = _job_with(TRAJECTORY)
    job.prompt = (
        "Review the remediation commit on the MR.\n\n"
        "Check every finding from round 1 is dispositioned, and flag any that "
        "were silently dropped."
    )
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        blocks = view._body.blocks()
        first = blocks[0]
        assert isinstance(first, UserBlock), [type(b).__name__ for b in blocks]
        assert first.text().startswith("Review the remediation commit")

        # It is mounted ONCE: a later event must not re-mount the instruction.
        job.trajectory.append(_call("c9", "read", path="notes.md"))
        app._refresh_band()
        await pilot.pause()
        assert view._body.blocks()[0] is first
        assert sum(1 for b in view._body.blocks() if isinstance(b, UserBlock)) == 1


@pytest.mark.asyncio
async def test_a_job_with_no_recorded_prompt_spends_no_row_on_one() -> None:
    """`AsyncJob.prompt` is populated by the subagent launcher only, so a job
    from anywhere else has none — and an empty user block would read as an
    instruction that was blank rather than one that was never recorded."""
    job = _job_with(TRAJECTORY)
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        assert not any(isinstance(b, UserBlock) for b in view._body.blocks())


@pytest.mark.asyncio
async def test_clicking_the_exit_hint_leaves_the_page() -> None:
    """The hints were inert text next to a mouse that could not use them, on a
    page that is otherwise entirely mouse-reachable. Clicking is additive — the
    key still works and the hint still names it."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        await pilot.click(view._exit_hint)
        await pilot.pause()
        assert not app.query(SubagentView)
        assert app.query_one(Editor).read_only is False
        assert session.aborts == [], "leaving is not stopping the parent's turn"


@pytest.mark.asyncio
async def test_clicking_the_arrows_scrolls_the_transcript_both_ways() -> None:
    """Two targets, not one: a single `↑↓` hint would have to guess which way a
    click meant, and a scroll affordance that guesses is one a reader stops
    trusting after the first wrong jump."""
    events: list[dict[str, Any]] = []
    for index in range(40):
        events.append(_call(f"c{index}", "read", path=f"pipeline/step_{index}.py"))
        events.append(_result(f"c{index}", "read", "…"))
    job = _job_with(events)
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        await pilot.pause()
        tail = view._body.scroll_offset.y
        assert tail > 0, "premise: the page opens scrolled to the tail"

        assert view._up_hint.has_class("actionable"), "there IS somewhere to scroll back to"
        await pilot.click(view._up_hint)
        await pilot.pause()
        lifted = view._body.scroll_offset.y
        assert lifted < tail, "the up arrow moves the reader back up the run"

        await pilot.click(view._down_hint)
        await pilot.pause()
        assert view._body.scroll_offset.y > lifted, "and the down arrow moves them forward"


@pytest.mark.asyncio
async def test_only_the_hints_that_do_something_answer_the_pointer() -> None:
    """A highlight that leads nowhere is the reported "nothing happens when I
    click" bug one step earlier, so the state hint stays inert."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        assert view._state_hint.rendered().strip().endswith("read-only")
        assert not view._state_hint.has_class("actionable")
        assert view._exit_hint.has_class("actionable")
        # This run's transcript fits the viewport, so paging is a no-op and
        # BOTH arrows are inert: a lit target that does nothing is the reported
        # "nothing happens when I click" bug one step earlier.
        assert view._body.max_scroll_y == 0, "premise: nothing to scroll"
        assert not view._up_hint.has_class("actionable")
        assert not view._down_hint.has_class("actionable")

        # The real gesture, not a method call: a handler Textual failed to
        # dispatch would otherwise go unnoticed.
        resting = view._exit_hint._build(
            view._exit_hint._label, lead=view._exit_hint._lead, hovered=False
        )
        await pilot.hover(view._exit_hint)
        lit = view._exit_hint._text()
        assert lit.plain == resting.plain, "hover changes ink, never the words"
        assert lit.spans != resting.spans, "an actionable hint answers the pointer"

        await pilot.hover(view._state_hint)
        assert view._state_hint._hovered is False, "an inert hint never lights"
        assert view._exit_hint._hovered is False, "and leaving restores the resting tone"


BRIEF = "\n\n".join(
    [
        "Review the remediation commit on MR !412.",
        "Every finding from round 1 must be either fixed with a commit SHA, "
        "rejected with a reason, or deferred with a ticket.",
        "Flag any that were silently dropped rather than restating the reply.",
        "Do not run the full suite; the pipeline already did.",
        "Report back with a table, one row per finding.",
    ]
)


@pytest.mark.asyncio
async def test_a_long_brief_is_folded_to_a_summary_and_opens_on_demand() -> None:
    """Pasted whole, a delegated brief broke the page in both directions: eight
    rows took 36% of the body at 120 columns, and on a nine-step child it sat
    eighteen rows above a viewport that opens at the tail — present, invisible,
    and opening mid-sentence either way."""
    job = _job_with(TRAJECTORY)
    job.prompt = BRIEF
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        brief = view._body.blocks()[0]
        assert isinstance(brief, InstructionBlock)
        rows = brief._rows(80)
        # The summary, plus one row saying what expanding would cost. Bare
        # `⟨expand⟩` does not distinguish two more lines from fifty.
        assert len(rows) == INSTRUCTION_ROWS + 1
        assert rows[0].startswith("Review the remediation commit")
        assert rows[-1].startswith(EXPAND_HINT)
        assert "more line" in rows[-1]
        full = len(UserBlock(BRIEF)._rows(80))
        assert f"{full - INSTRUCTION_ROWS} more lines" in rows[-1]

        await pilot.click(brief)
        await pilot.pause()
        opened = brief._rows(80)
        assert len(opened) == full + 1, "every row of the brief, plus the fold-back"
        assert opened[-1] == COLLAPSE_AFFORDANCE
        assert "Report back with a table" in " ".join(opened)

        await pilot.click(brief)
        await pilot.pause()
        assert len(brief._rows(80)) == INSTRUCTION_ROWS + 1, "and it folds back"


@pytest.mark.asyncio
async def test_a_brief_that_already_fits_is_offered_no_affordance() -> None:
    """An expander with nothing to open is the "nothing happens when I click"
    bug one step earlier, and the row it would spend is a row of the child's."""
    job = _job_with(TRAJECTORY)
    job.prompt = "Audit the ingest path."
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        brief = view._body.blocks()[0]
        assert isinstance(brief, InstructionBlock)
        rows = brief._rows(80)
        assert rows == ["Audit the ingest path."]
        assert not any(EXPAND_HINT in row for row in rows)


@pytest.mark.asyncio
async def test_the_title_drops_whole_fields_rather_than_cutting_a_value() -> None:
    """`⣿ running · 23…` cannot be told from 23s or 23m, and `⣷ runn…` spends
    four cells to say less than the glyph already said. A field that will not
    fit is worth less than the one before it, so it leaves whole."""
    job = _Job("sub-1", "RetryBudgetScout", status="running")
    job.trajectory = [*_text("m1", "working"), _call("c1", "read", path="a.py")]
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        elapsed = view._elapsed
        for width in (120, 80, 60, 48, 40, 32, 24, 16, 10, 6):
            row = view._title_row(width, "⣾", tools=1).plain
            assert len(row) <= width or cell_len(row) <= width, (width, row)
            # Whatever is still on the row is COMPLETE: no field is half a
            # value. `running` never appears as a prefix of itself, and the
            # duration never loses its unit.
            if elapsed[:-1] and elapsed[:-1] in row:
                assert elapsed in row, (width, row)
            for prefix in ("runn", "run "):
                assert not row.endswith(prefix + "…"), (width, row)
            assert "…" not in row.split("  ")[-1] or "tool" not in row, (width, row)
        # The glyph is the one field that never leaves: it is the only one that
        # still answers "is this running" in a colourless frame at 6 cells.
        assert "⣾" in view._title_row(6, "⣾", tools=1).plain


@pytest.mark.asyncio
async def test_the_scroll_caption_keeps_the_space_every_other_hint_has() -> None:
    """It is the one hint with no key in front of it, so without its own
    leading space the row read `↑↓scroll` while every other hint in the app —
    and every hint on the aside overlay — reads `key label`."""
    session = FakeSession()
    session.jobs = _fake_jobs(_job_with(TRAJECTORY))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, _job_with(TRAJECTORY))
        assert "↑↓ scroll" in view.rendered_rows()[-1]
        assert "↑↓scroll" not in view.rendered_rows()[-1]
