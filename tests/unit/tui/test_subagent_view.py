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

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from rich.cells import cell_len

from local_operator.harness.comms import (
    HUB_COMMUNICATION_CUSTOM_TYPE,
    HUB_MESSAGE_TYPE,
    SubagentComms,
)
from local_operator.harness.jobs import (
    CANCELLED_BEFORE_START,
    AsyncJob,
    AsyncJobManager,
)
from local_operator.harness.types import CustomMessage, Message, TextContent, ToolCall
from local_operator.session.session import Session
from local_operator.session.transcript import (
    Transcript,
    TranscriptEntry,
    TranscriptPage,
)
from local_operator.tui.app import SUBAGENT_LAYOUT_CLASS, OperatorApp
from local_operator.tui.widgets.assistant import FALLBACK_WIDTH, AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.subagent_panel import (
    GLYPH_DONE,
    SubagentPanel,
    SubagentRow,
)
from local_operator.tui.widgets.subagent_view import (
    COLLAPSE_AFFORDANCE,
    EXPAND_HINT,
    HISTORY_ERROR_NOTE,
    HISTORY_START_NOTE,
    HISTORY_UNAVAILABLE_NOTE,
    INSTRUCTION_ROWS,
    LEDGER_GONE_NOTE,
    TRAJECTORY_MAX_EVENTS,
    TRUNCATION_NOTE,
    InstructionBlock,
    SubagentView,
    entry_block,
    fold_trajectory,
)
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptView,
    UserBlock,
    WorkingBlock,
)

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


def test_fold_accepts_immutable_mapping_and_sequence_contracts() -> None:
    """Canonical follower trajectories stay tuple-backed and fully renderable."""
    from local_operator.session.frontend_state import (
        FrontendSessionState,
        FrontendStateStore,
        JobState,
    )
    from local_operator.tui.widgets.subagent_panel import row_facts

    state = FrontendStateStore(
        FrontendSessionState(
            session_id="follower",
            epoch="owner",
            jobs=[
                JobState(
                    id="sub-1",
                    type="task",
                    label="immutable child",
                    latest_details={"progress": "reading files"},
                    trajectory=TRAJECTORY,
                )
            ],
        )
    ).state
    job = state.jobs[0]
    assert isinstance(job.latest_details, Mapping)
    assert isinstance(job.trajectory, Sequence)
    assert row_facts(job, fallback_id=job.id, current=False).activity == "reading files"
    entries = fold_trajectory(job.trajectory, settled=False)
    assert [entry.kind for entry in entries] == ["text", "tool", "tool", "text"]
    assert entries[0].text == "Reading the ingest path."


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


@pytest.mark.parametrize(
    "duration",
    [True, False, "1.25", {"seconds": 1.25}, float("nan"), float("inf"), -float("inf"), -0.1],
)
def test_fold_rejects_invalid_tool_durations_without_crashing(duration: object) -> None:
    events = [
        _call("e1", "edit", path="a.py", old_text="old", new_text="new"),
        {**_result("e1", "edit", "Done!"), "duration_s": duration},
    ]
    assert fold_trajectory(events, settled=True)[0].duration_s is None


def test_fold_uses_valid_result_duration_when_event_duration_is_invalid() -> None:
    result = _result("e1", "edit", "Done!")
    result["result"]["duration_s"] = 2.5
    events = [
        _call("e1", "edit", path="a.py", old_text="old", new_text="new"),
        {**result, "duration_s": float("nan")},
    ]
    assert fold_trajectory(events, settled=True)[0].duration_s == 2.5


def test_fold_preserves_tool_duration_diff_counts_and_diff_only_expansion() -> None:
    events = [
        _call("e1", "edit", path="a.py", old_text="old", new_text="new"),
        {
            **_result("e1", "edit", "Done!"),
            "duration_s": 1.25,
            "result": {
                **_result("e1", "edit", "Done!")["result"],
                "details": {
                    "added": 1,
                    "removed": 1,
                    "diff": ["---", "+++", "@@ -1 +1 @@", "-old", "+new"],
                },
            },
        },
    ]
    entry = fold_trajectory(events, settled=True)[0]
    assert entry.duration_s == 1.25
    card = entry_block(entry)
    assert isinstance(card, ToolCard)
    assert card._added == 1
    assert card._removed == 1
    assert card._duration == 1.25
    assert card._diff == ["---", "+++", "@@ -1 +1 @@", "-old", "+new"]
    assert card._output == ["Done!"]


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


def test_fold_narrates_a_fallback_route_change_once() -> None:
    events = [
        {
            "type": "notice",
            "text": "provider failure — falling back to x/mini",
            "kind": "warning",
        },
        {
            "type": "model_change",
            "provider": "x",
            "model_id": "mini",
            "is_fallback": True,
        },
    ]

    entries = fold_trajectory(events, settled=False)

    assert [(entry.text, entry.notice_kind) for entry in entries] == [
        ("provider failure — falling back to x/mini", "warning")
    ]


def test_fold_keeps_each_notice_in_a_fallback_cascade_without_model_change_duplicates() -> None:
    events: list[dict[str, Any]] = []
    targets = ["x/mini", "y/medium", "z/max"]
    for selector in targets:
        provider, model_id = selector.split("/", 1)
        events.extend(
            [
                {
                    "type": "notice",
                    "text": f"provider failure — falling back to {selector}",
                    "kind": "warning",
                },
                {
                    "type": "model_change",
                    "provider": provider,
                    "model_id": model_id,
                    "is_fallback": True,
                },
            ]
        )

    entries = fold_trajectory(events, settled=False)

    assert [entry.text for entry in entries] == [
        f"provider failure — falling back to {selector}" for selector in targets
    ]
    assert [entry.notice_kind for entry in entries] == ["warning"] * len(targets)


def test_fold_narrates_primary_recovery_once() -> None:
    events = [
        {"type": "notice", "text": "back to primary/model", "kind": "info"},
        {
            "type": "model_change",
            "provider": "primary",
            "model_id": "model",
            "is_fallback": False,
        },
    ]

    entries = fold_trajectory(events, settled=False)

    assert [(entry.text, entry.notice_kind) for entry in entries] == [
        ("back to primary/model", "info")
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
async def test_peer_keys_cycle_all_siblings_in_opposite_directions() -> None:
    jobs = [
        _Job(job_id, label) for job_id, label in (("a", "alpha"), ("b", "beta"), ("c", "gamma"))
    ]
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    comms = SubagentComms(cast(Session, session))
    for job in jobs:
        comms.record_launch(job.id, job.label)
    session._subagent_comms = comms
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, jobs[1])
        assert view.job_id == "b"

        await pilot.press("]")
        await pilot.pause()
        assert app.query_one(SubagentView).job_id == "c"
        await pilot.press("]")
        await pilot.pause()
        assert app.query_one(SubagentView).job_id == "a"
        await pilot.press("[")
        await pilot.pause()
        assert app.query_one(SubagentView).job_id == "c"
        await pilot.press("[")
        await pilot.pause()
        assert app.query_one(SubagentView).job_id == "b"


@pytest.mark.asyncio
async def test_narrow_subagent_mode_preserves_a_useful_transcript_viewport() -> None:
    jobs = [_job_with(TRAJECTORY), _Job("peer", "peer")]
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    comms = SubagentComms(cast(Session, session))
    for job in jobs:
        comms.record_launch(job.id, job.label)
    session._subagent_comms = comms
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        view = await _open(pilot, app, jobs[0])
        for _ in range(3):
            await pilot.pause()
        assert app.screen.has_class("subagent-compact")
        assert app.query_one("#band").display is False
        assert view._body.size.height >= 8, view._body.size
        assert app.screen.virtual_size.height <= app.screen.size.height


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (60, 22)])
async def test_opening_from_expanded_roster_preserves_child_viewport(size) -> None:
    jobs = [_job_with(TRAJECTORY, status="completed") for _ in range(100)]
    for index, job in enumerate(jobs):
        job.id = f"sub-{index:03d}"
        job.label = f"task {index:03d}"
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        panel = app.query_one(SubagentPanel)
        assert panel._expanded is True

        app._open_subagent_view(jobs[-1].id)
        for _ in range(4):
            await pilot.pause()
        view = app.query_one(SubagentView)

        assert panel._expanded is False
        assert view.size.height > 0
        assert view._body.size.height > 0
        assert app.screen.virtual_size.height <= app.screen.size.height


@pytest.mark.asyncio
async def test_escape_from_child_restores_composer_when_roster_row_was_hidden() -> None:
    jobs = [_job_with(TRAJECTORY, status="completed") for _ in range(30)]
    for index, job in enumerate(jobs):
        job.id = f"sub-{index:03d}"
        job.label = f"task {index:03d}"
    session = FakeSession()
    session.jobs = _fake_jobs(*jobs)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.text = "draft "
        await pilot.press("ctrl+g")
        await pilot.pause()
        for _ in range(10):
            await pilot.press("down")
        selected = app.focused
        assert isinstance(selected, SubagentRow)
        await pilot.press("enter")
        for _ in range(4):
            await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        assert selected.display is False
        assert app.focused is editor
        await pilot.press("x")
        await pilot.pause()
        assert editor.text == "xdraft "
        await pilot.press("down")
        await pilot.pause()
        assert app.focused is editor


async def _wait_history(pilot: Any, view: SubagentView) -> None:
    started = view._history_loading
    for _ in range(100):
        await asyncio.sleep(0.005)
        await pilot.pause()
        started = started or view._history_loading or bool(view._history_entries)
        if started and not view._history_loading:
            return
    raise AssertionError("history worker did not settle")


async def _wait_geometry_settled(
    pilot: Any, body: Any, *, cycles: int = 4, limit: int = 200
) -> None:
    """Pause until the body's geometry stops changing for ``cycles`` cycles.

    A prepend restores the anchor across TWO layout passes: gap settlement
    re-decides margins against real widths, and the pass that applies them
    republishes positions. A fixed number of pauses bets on which pass wins and
    loses that bet under parallel load, so the wait is on the observable itself.
    """
    last: tuple[int, float] | None = None
    stable = 0
    for _ in range(limit):
        await pilot.pause()
        current = (body.virtual_size.height, body.scroll_y)
        stable = stable + 1 if current == last else 0
        last = current
        if stable >= cycles:
            return
    raise AssertionError("body geometry never settled")


@pytest.mark.asyncio
async def test_durable_history_opens_at_tail_and_pages_to_the_start(tmp_path) -> None:
    transcript = Transcript(tmp_path / "child")
    for index in range(230):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert view._body.scroll_y >= view._body.max_scroll_y - 1
        assert "durable 229" in " ".join(view.rendered_rows())
        assert "durable 0" not in " ".join(view.rendered_rows())

        while not view._history_exhausted:
            view.action_home()
            await _wait_history(pilot, view)
        page = " ".join(view.rendered_rows())
        assert "durable 0" in page
        assert HISTORY_START_NOTE in page
        assert len(view._history_ids) == 230
        assert view._state_hint.rendered().endswith("transcript start · read-only")
        assert view._state_hint.region.width >= len("transcript start · read-only")


@pytest.mark.asyncio
async def test_history_home_key_lands_on_newly_loaded_page(tmp_path) -> None:
    """The binding path must not sample the old tail before Home settles."""
    transcript = Transcript(tmp_path / "child")
    for index in range(140):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert view._body.scroll_y >= view._body.max_scroll_y - 1
        before_max = view._body.max_scroll_y

        await pilot.press("home")
        await _wait_history(pilot, view)
        for _ in range(5):
            await pilot.pause()

        growth = view._body.max_scroll_y - before_max
        # Block gap settlement can add a few rows after anchor restoration; the
        # invariant is that Home lands with the prepended page, not at the tail.
        assert growth <= view._body.scroll_y <= growth + 4
        assert view._body.scroll_y < view._body.max_scroll_y - 1


@pytest.mark.asyncio
async def test_history_prepend_preserves_anchor_and_home_dedupes_requests(
    tmp_path, monkeypatch
) -> None:
    transcript = Transcript(tmp_path / "child")
    for index in range(140):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        calls = 0
        original = __import__(
            view.__module__, fromlist=["read_transcript_page"]
        ).read_transcript_page

        def slow(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(view.__module__ + ".read_transcript_page", slow)
        view._initial_tail_pending = True
        view._body.scroll_home(animate=False)
        await _wait_geometry_settled(pilot, view._body)
        view._initial_tail_pending = False
        # Preserve the actual visible block and its viewport-relative row.
        before = view._body.scroll_y
        anchor_block = next(
            block for block in view._body.blocks() if block.virtual_region.bottom > before
        )
        before_gap = anchor_block.virtual_region.y - before
        view.action_home()
        view.action_home()
        await _wait_history(pilot, view)
        await _wait_geometry_settled(pilot, view._body)
        assert calls == 1
        assert anchor_block in view._body.blocks()
        assert anchor_block.virtual_region.y - view._body.scroll_y == before_gap


@pytest.mark.asyncio
async def test_history_settles_tool_split_across_durable_page_boundary(tmp_path) -> None:
    """A result at newer[0] must settle its call once older[-1] is loaded."""
    transcript = Transcript(tmp_path / "child")
    for index in range(98):
        await transcript.append_message(Message.assistant(f"older {index}"))
    await transcript.append_message(
        Message(
            role="assistant",
            content=[],
            tool_calls=[ToolCall(id="boundary-call", name="bash", arguments={"command": "true"})],
        )
    )
    await transcript.append_message(
        Message(
            role="tool",
            tool_call_id="boundary-call",
            tool_name="bash",
            content=[TextContent(text="done")],
        )
    )
    for index in range(99):
        await transcript.append_message(Message.assistant(f"newer {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert all(entry.key != "boundary-call" for entry in view._history_entries)

        await pilot.press("home")
        await _wait_history(pilot, view)

        call = next(entry for entry in view._history_entries if entry.key == "boundary-call")
        assert call.outcome == "success"
        assert call.result_text == "done"


@pytest.mark.asyncio
async def test_durable_edit_and_write_restore_parent_diff_cards_and_durations(tmp_path) -> None:
    """Completed child tools use the parent's ToolCard metadata, not raw args.

    Diff details and execution time live in the durable tool message's harness
    payload. Dropping that payload produced a checked row whose expansion was
    the model's raw ``old_text``/``new_text`` fields and whose duration was
    blank, even though the parent transcript had already rendered both.
    """
    transcript = Transcript(tmp_path / "child")
    expected = {
        "edit": (1, 1, ["--- a.py", "+++ a.py", "@@ -1 +1 @@", "-old", "+new"]),
        "write": (2, 0, ["--- /dev/null", "+++ b.py", "@@ -0,0 +1,2 @@", "+one", "+two"]),
    }
    durations = {"edit": 1.25, "write": 2.5}
    for name, (added, removed, diff) in expected.items():
        call_id = f"{name}-call"
        await transcript.append_message(
            Message.assistant(
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        name=name,
                        arguments={
                            "path": f"{name}.py",
                            "old_text": "raw-old",
                            "new_text": "raw-new",
                        },
                    )
                ]
            )
        )
        result = Message(
            role="tool",
            tool_call_id=call_id,
            tool_name=name,
            content=[TextContent(text="Done!")],
            provider_payload={
                "details": {"added": added, "removed": removed, "diff": diff},
                "duration_s": durations[name],
            },
        )
        await transcript.append_message(result)

    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        cards = {
            block.tool_name: block for block in view._body.blocks() if isinstance(block, ToolCard)
        }
        for name, (added, removed, diff) in expected.items():
            assert cards[name]._added == added
            assert cards[name]._removed == removed
            assert cards[name]._diff == diff
            assert cards[name]._duration == durations[name]
            assert cards[name]._output == ["Done!"]


@pytest.mark.asyncio
async def test_durable_tool_rejects_bool_and_malformed_duration(tmp_path) -> None:
    transcript = Transcript(tmp_path / "child")
    invalid = (
        True,
        "1.25",
        {"seconds": 1.25},
        float("nan"),
        float("inf"),
        -float("inf"),
        -0.1,
    )
    for index, duration in enumerate(invalid):
        call_id = f"bad-duration-{index}"
        await transcript.append_message(
            Message.assistant(tool_calls=[ToolCall(id=call_id, name="write")])
        )
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id=call_id,
                tool_name="write",
                provider_payload={"duration_s": duration},
            )
        )
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        cards = [block for block in view._body.blocks() if isinstance(block, ToolCard)]
        assert len(cards) == len(invalid)
        assert all(card._duration is None for card in cards)


@pytest.mark.asyncio
async def test_history_replacement_adopts_changed_payloads_with_preserved_ids(tmp_path) -> None:
    """Compaction can rewrite content without minting new entry identities."""
    transcript = Transcript(tmp_path / "child")
    original = await transcript.append_message(Message.assistant("before replacement"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)

        replacement_payload = dict(original.payload)
        replacement_payload["content"] = [{"type": "text", "text": "after replacement"}]
        replacement = TranscriptEntry(
            id=original.id,
            ts=original.ts,
            type=original.type,
            payload=replacement_payload,
        )
        view._apply_history_page(
            view._history_generation,
            TranscriptPage((replacement,), has_more=False, reconciled=True),
            anchor=0.0,
            initial=False,
        )
        await pilot.pause()

        rendered = " ".join(view.rendered_rows())
        assert "after replacement" in rendered
        assert "before replacement" not in rendered
        assert view._history_ids == {original.id}
        assert [row.id for row in view._history_rows] == [original.id]


@pytest.mark.asyncio
async def test_history_job_switch_discards_a_late_page(tmp_path, monkeypatch) -> None:
    first = Transcript(tmp_path / "first")
    second = Transcript(tmp_path / "second")
    await first.append_message(Message.assistant("from first"))
    await second.append_message(Message.assistant("from second"))
    job_a = _Job("a", "first", status="completed")
    job_b = _Job("b", "second", status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job_a, job_b)
    directories = {"a": first.directory, "b": second.directory}
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, job_id: directories[job_id]}
    )()
    original = __import__(
        SubagentView.__module__, fromlist=["read_transcript_page"]
    ).read_transcript_page

    def delayed(directory, **kwargs):
        if str(directory) == str(first.directory):
            import time

            time.sleep(0.05)
        return original(directory, **kwargs)

    monkeypatch.setattr(SubagentView.__module__ + ".read_transcript_page", delayed)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job_a)
        app._open_subagent_view("b")
        await _wait_history(pilot, view)
        for _ in range(20):
            await pilot.pause()
        page = " ".join(view.rendered_rows())
        assert "from second" in page
        assert "from first" not in page


@pytest.mark.asyncio
@pytest.mark.parametrize("fact_first", [False, True])
async def test_durable_communication_dedupes_across_page_boundary(
    tmp_path, fact_first: bool
) -> None:
    """The human-facing fact supersedes replay XML in either page order."""
    transcript = Transcript(tmp_path / "child")
    replay = CustomMessage(
        id="q1",
        custom_type=HUB_MESSAGE_TYPE,
        attribution="user",
        details={
            "direction": "to_child",
            "body": "Which file?",
            "expects_reply": True,
            "text": "<parent-message>Which file?</parent-message>",
        },
    )

    async def append_fact() -> None:
        await transcript.append_custom(
            HUB_COMMUNICATION_CUSTOM_TYPE,
            {
                "direction": "to_child",
                "body": "Which file?",
                "kind": "ask",
                "communication_id": "q1",
            },
        )

    if fact_first:
        await append_fact()
    else:
        await transcript.append_message(replay)
    # Ninety-nine rows put the two representations on opposite sides of the
    # 100-row durable page boundary without relying on private reader details.
    for index in range(99):
        await transcript.append_message(Message.assistant(f"boundary filler {index}"))
    if fact_first:
        await transcript.append_message(replay)
    else:
        await append_fact()

    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        await pilot.press("home")
        await _wait_history(pilot, view)

        notices = [
            entry.text
            for entry in view._history_entries
            if entry.text == "Parent · asked\nWhich file?"
        ]
        assert notices == ["Parent · asked\nWhich file?"]
        assert "<parent-message>" not in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_durable_communications_are_human_facing_and_correlated(tmp_path) -> None:
    transcript = Transcript(tmp_path / "child")
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {
            "direction": "to_child",
            "body": "Which file failed?",
            "kind": "ask",
            "communication_id": "q1",
        },
    )
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {"direction": "to_parent", "body": "tests/test_api.py", "reply_to": "q1"},
    )
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {
            "direction": "to_child",
            "body": "Focus on retries",
            "kind": "steer",
            "communication_id": "s1",
        },
    )
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        communications = [
            (entry.kind, entry.text)
            for entry in view._history_entries
            if entry.kind in ("parent_message", "subagent_message")
        ]
        assert communications == [
            ("parent_message", "Parent · asked\nWhich file failed?"),
            ("subagent_message", "Subagent · replied\n\ntests/test_api.py"),
            ("parent_message", "Parent · redirected\nFocus on retries"),
        ]
        assert "<parent-message>" not in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_history_unavailable_and_error_retry_keep_trajectory_fallback(
    tmp_path, monkeypatch
) -> None:
    job = _job_with(TRAJECTORY, status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    child_dir = tmp_path / "child"
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: child_dir}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()
        assert "Reading the ingest path." in " ".join(view.rendered_rows())

        child_dir.mkdir()
        view._history_unavailable = False
        attempts = 0

        def flaky(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("disk busy")
            return __import__(
                "local_operator.session.transcript", fromlist=["read_transcript_page"]
            ).read_transcript_page(*args, **kwargs)

        monkeypatch.setattr(SubagentView.__module__ + ".read_transcript_page", flaky)
        # Inject the failure deterministically; the regression below exercises
        # the real mounted Home binding for the explicit retry itself.
        view.action_home()
        await _wait_history(pilot, view)
        assert HISTORY_ERROR_NOTE in view._history_state_text()
        failed_attempts = attempts
        # Repaints and the top-edge watcher may run freely, but the advertised
        # retry must remain a user decision rather than an implicit hot loop.
        for _ in range(50):
            await asyncio.sleep(0.005)
            await pilot.pause()
        assert attempts == failed_attempts == 1

        view.focus_body()
        await pilot.pause()
        assert app.focused is view._body
        await pilot.press("home")
        for _ in range(100):
            await asyncio.sleep(0.005)
            await pilot.pause()
            if attempts == 2 and not view._history_loading:
                break
        assert attempts == 2
        assert view._history_unavailable


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
async def test_historical_attempt_opens_and_selects_the_current_visible_row() -> None:
    session = FakeSession()
    manager = AsyncJobManager()
    current = AsyncJob(
        id="current",
        type="task",
        status="completed",
        start_time=1.0,
        settled_at=2.0,
        label="continued child",
        logical_id="/tmp/child",
        attempt_aliases=["historical"],
    )
    manager.restore([current])
    session.jobs = manager
    comms = SubagentComms(session)  # type: ignore[arg-type]
    comms.restore(
        [
            {
                "job_id": "current",
                "label": "continued child",
                "session_dir": "/tmp/child",
                "attempt_aliases": ["historical"],
            }
        ]
    )
    session._subagent_comms = comms

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        app._open_subagent_view("historical")
        await pilot.pause()

        view = app.query_one(SubagentView)
        panel = app._subagent_panel
        assert view.job_id == "current"
        assert panel is not None
        assert list(panel._rows) == ["current"]
        assert panel._rows["current"].current is True


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
async def test_canonical_progress_refresh_keeps_child_page_live() -> None:
    """The single canonical invalidation updates an already-open child page."""
    from local_operator.session.frontend_state import FrontendSessionState, JobState

    job = _Job("sub-1", "audit the ingest path", status="running")
    job.trajectory = [*_text("m1", "Reading the ingest path.")]
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        prose = view._body.blocks()[0]
        job.trajectory.append(_call("c1", "bash", command="pytest -q"))

        # This is the frame the 50 ms jobs coalescer publishes to both owner and
        # follower. No raw SubagentProgress repaint participates any more.
        app._apply_frontend_state(
            FrontendSessionState(
                session_id="sess",
                epoch="owner",
                jobs=[
                    JobState.from_job(job).model_copy(
                        update={"latest_details": {"progress": "running pytest"}}
                    )
                ],
            )
        )
        await pilot.pause()
        blocks = view._body.blocks()
        assert blocks[0] is prose
        assert isinstance(blocks[1], ToolCard)
        assert blocks[1]._state == "running"
        assert isinstance(blocks[2], WorkingBlock)


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
        assert isinstance(tail, WorkingBlock)

        job.trajectory.append(_call("c1", "bash", command="pytest -q"))
        app._refresh_band()
        await pilot.pause()
        blocks = view._body.blocks()
        assert isinstance(blocks[1], ToolCard), [type(b).__name__ for b in blocks]
        assert isinstance(blocks[2], WorkingBlock)
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
async def test_the_title_says_a_cancelled_child_never_ran() -> None:
    """``⊘ cancelled · 1m36s`` presents waiting time as work time.

    A job cancelled before its runner was entered ran for zero seconds and
    spent nothing; its duration is how long it WAITED. Beside a page whose
    sibling rows read ``⣷ running · 7m53s``, the bare word paired with that
    number says an operator killed a child a minute and a half into its work.
    The manager records the distinction on the row
    (``CANCELLED_BEFORE_START``, keyed on ``started_at``) and the title spends
    it.
    """
    parked = _Job("sub-parked", "flaky test bisect", status="cancelled")
    parked.settled_at = parked.start_time + 96.0
    parked.result_text = CANCELLED_BEFORE_START
    mid_run = _Job("sub-midrun", "docs sweep agent", status="cancelled")
    mid_run.settled_at = mid_run.start_time + 96.0
    # Representational, not load-bearing: the view reads ``result_text``, but
    # this row's premise is "its runner began", and ``started_at`` is the fact
    # that says so on a real job — left unset, the row encodes the OPPOSITE.
    mid_run.started_at = mid_run.start_time
    session = FakeSession()
    session.jobs = _fake_jobs(parked, mid_run)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, parked)
        title = view.rendered_rows()[0]
        assert CANCELLED_BEFORE_START in title, title
        # A child whose runner DID begin keeps the bare word: it worked, and
        # that duration is work time. ``started_at`` (set above) is what
        # separates it from the parked row — ``queued`` would not, since an
        # admitted-but-never-entered job is not parked either.
        app._open_subagent_view("sub-midrun")
        await pilot.pause()
        title = view.rendered_rows()[0]
        assert "cancelled" in title
        assert CANCELLED_BEFORE_START not in title, title


@pytest.mark.asyncio
async def test_a_narrow_title_shortens_the_state_word_before_losing_it() -> None:
    """The phrase is 27 cells where the bare word is 9, and a single-rung
    ladder dropped the state ENTIRELY at the widths between the two.

    At those widths the page showed a glyph and a duration with no word on
    screen — for the one state whose whole point is being said. The state is
    shortened before it is dropped, so it is the last thing to go, not the
    first.
    """
    parked = _Job("sub-parked", "flaky test bisect", status="cancelled")
    parked.settled_at = parked.start_time + 96.0
    parked.result_text = CANCELLED_BEFORE_START
    session = FakeSession()
    session.jobs = _fake_jobs(parked)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(48, 24)) as pilot:
        view = await _open(pilot, app, parked)
        title = view.rendered_rows()[0]

    assert "cancelled" in title, title
    assert CANCELLED_BEFORE_START not in title, "premise: too narrow for the phrase"


@pytest.mark.asyncio
async def test_the_title_never_loses_its_state_as_it_widens() -> None:
    """A wider title must never lose a state word a narrower one carried.

    The label budget omitted the glyph cell, so a rung whose label consumed its
    budget exactly was rejected and the ladder fell through: the state visible
    at 35 cells and gone again at 36-40, where it still fit. The assertion is
    deliberately about STATE preservation, not every field: the ladder may
    trade a duration for the longer, more precise phrase as it widens.
    """
    parked = _Job("sub-parked", "flaky test bisect", status="cancelled")
    parked.settled_at = parked.start_time + 96.0
    parked.result_text = CANCELLED_BEFORE_START
    session = FakeSession()
    session.jobs = _fake_jobs(parked)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 24)) as pilot:
        view = await _open(pilot, app, parked)

        state_seen = False
        for width in range(20, 110):
            row = view._title_row(width, "⠋", 0).plain
            has_state = "cancelled" in row
            if state_seen:
                assert (
                    has_state
                ), f"width {width} lost the state its narrower neighbour kept: {row!r}"
            state_seen = state_seen or has_state


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
async def test_durable_history_cannot_move_the_delegation_after_child_work(tmp_path) -> None:
    """Disk history may arrive after ``show`` but chronology cannot follow it.

    The prompt is the user turn that caused the durable assistant rows. Loading
    history asynchronously used to prepend those rows ahead of the already
    mounted instruction, leaving the delegation at the transcript end.
    """
    transcript = Transcript(tmp_path / "child")
    await transcript.append_message(Message.assistant("First child response."))
    await transcript.append_message(Message.assistant("Final child response."))
    job = _job_with([], status="completed")
    job.prompt = "Delegated parent instruction."
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        entries = [entry for entry in view._pending if entry.key != "__working__"]
        assert [(entry.kind, entry.text) for entry in entries] == [
            ("prompt", "Delegated parent instruction."),
            ("text", "First child response."),
            ("text", "Final child response."),
        ]
        assert isinstance(view._body.blocks()[0], InstructionBlock)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_role", "durable_prompt"),
    [
        ("task", "Delegated parent instruction."),
        ("coder", "[role: coder]\nRole guidance.\n\nDelegated parent instruction."),
        ("scout", "[scout mode: read only.]\n\nDelegated parent instruction."),
        (
            "coder",
            "[team: lopdev]\n\nYou are coder on this team.\n\n"
            "[role: coder]\nRole guidance.\n\nDelegated parent instruction.",
        ),
        ("dashboard-specialist", "Specialist guidance.\n\nDelegated parent instruction."),
    ],
)
async def test_durable_launch_turn_is_replaced_in_place(
    tmp_path, agent_role: str, durable_prompt: str
) -> None:
    transcript = Transcript(tmp_path / "child")
    await transcript.append_message(Message.assistant("Before launch row."))
    launch = await transcript.append_message(Message.user(durable_prompt))
    await transcript.append_message(Message.assistant("After launch row."))
    job = _job_with([], status="completed")
    job.prompt = "Delegated parent instruction."
    job.effective_prompt = durable_prompt
    job.launch_message_id = launch.id
    job.agent_role = agent_role
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        entries = [entry for entry in view._pending if entry.key != "__working__"]
        assert [(entry.key, entry.kind, entry.text) for entry in entries] == [
            (entries[0].key, "text", "Before launch row."),
            (launch.id, "prompt", "Delegated parent instruction."),
            (entries[2].key, "text", "After launch row."),
        ]
        assert sum(entry.kind == "prompt" for entry in entries) == 1


@pytest.mark.asyncio
async def test_resumed_view_projects_every_collapsed_launch_prompt(tmp_path) -> None:
    """After #314 collapses attempts, ALL durable launch rows go concise.

    The shared transcript holds one ``subagent-launch:<id>`` turn per attempt.
    The comms node maps each to its concise prompt, so a resumed view must
    replace every one, never leaking an earlier attempt's role/system preamble
    as a plain user row (review round 4 R4-1).
    """
    transcript = Transcript(tmp_path / "child")
    first_launch = await transcript.append_message(
        Message.user("[role: reviewer]\nSYSTEM PREAMBLE\nOriginal task.")
    )
    await transcript.append_message(Message.assistant("first attempt"))
    second_launch = await transcript.append_message(
        Message.user("[role: reviewer]\nSYSTEM PREAMBLE\nWrap up.")
    )
    job = _job_with([], status="completed")
    # The live job row only carries the CURRENT (newest) attempt.
    job.prompt = "Wrap up."
    job.effective_prompt = "[role: reviewer]\nSYSTEM PREAMBLE\nWrap up."
    job.launch_message_id = second_launch.id
    job.agent_role = "reviewer"
    # The comms node carries every collapsed launch identity's concise prompt.
    node = type(
        "Node",
        (),
        {
            "label": "reviewer",
            "prompt": "Wrap up.",
            "effective_prompt": "[role: reviewer]\nSYSTEM PREAMBLE\nWrap up.",
            "launch_message_id": second_launch.id,
            "agent_role": "reviewer",
            "effort": "",
            "session_id": None,
            "session_dir": transcript.directory,
            "launch_prompts": {
                first_launch.id: "Original task.",
                second_launch.id: "Wrap up.",
            },
        },
    )()
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms",
        (),
        {
            "session_dir_of": lambda self, _job_id: transcript.directory,
            "node": lambda self, _job_id: node,
            "ancestors": lambda self, _job_id: [],
            "job": lambda self, _job_id: job,
        },
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        entries = [entry for entry in view._pending if entry.key != "__working__"]
        kinds = {entry.key: entry for entry in entries}
        # Both launch rows are concise, neither leaks its preamble.
        assert kinds[first_launch.id].kind == "prompt"
        assert kinds[first_launch.id].text == "Original task."
        assert kinds[second_launch.id].kind == "prompt"
        assert kinds[second_launch.id].text == "Wrap up."
        assert "SYSTEM PREAMBLE" not in " ".join(entry.text for entry in entries)
        # No leftover synthetic head: the current launch matched in history.
        assert not any(entry.key == "__prompt__" for entry in entries)
        assert sum(entry.kind == "prompt" for entry in entries) == 2


@pytest.mark.asyncio
async def test_specialist_identity_cannot_replace_a_later_suffix_collision(tmp_path) -> None:
    transcript = Transcript(tmp_path / "child")
    effective = "Specialist guidance.\n\nDelegated parent instruction."
    launch = await transcript.append_message(Message.user(effective))
    await transcript.append_message(Message.assistant("Work happened."))
    collision = await transcript.append_message(
        Message.user("Please quote:\n\nDelegated parent instruction.")
    )
    job = _job_with([], status="completed")
    job.prompt = "Delegated parent instruction."
    job.effective_prompt = effective
    job.launch_message_id = launch.id
    job.agent_role = "dashboard-specialist"
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        entries = [entry for entry in view._pending if entry.key != "__working__"]
        replacement = next(entry for entry in entries if entry.key == launch.id)
        assert replacement.kind == "prompt"
        later = next(entry for entry in entries if entry.key == collision.id)
        assert later.kind == "user"
        assert later.text == "Please quote:\n\nDelegated parent instruction."
        assert sum(entry.kind == "prompt" for entry in entries) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "later_text",
    ["Delegated parent instruction.", "  Delegated   parent instruction.  "],
)
async def test_launch_id_wins_over_exact_and_whitespace_equivalent_duplicates(
    tmp_path, later_text: str
) -> None:
    transcript = Transcript(tmp_path / "child")
    launch = await transcript.append_message(Message.user("Delegated parent instruction."))
    later = await transcript.append_message(Message.user(later_text))
    job = _job_with([], status="completed")
    job.prompt = "Delegated parent instruction."
    job.effective_prompt = "Delegated parent instruction."
    job.launch_message_id = launch.id
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        entries = [entry for entry in view._pending if entry.key != "__working__"]
        assert next(entry for entry in entries if entry.key == launch.id).kind == "prompt"
        duplicate = next(entry for entry in entries if entry.key == later.id)
        assert duplicate.kind == "user"
        assert duplicate.text == later_text.strip()


@pytest.mark.asyncio
async def test_legacy_whitespace_equivalent_prompt_is_not_value_matched(tmp_path) -> None:
    transcript = Transcript(tmp_path / "child")
    durable = await transcript.append_message(Message.user("  Delegated   parent instruction.  "))
    job = _job_with([], status="completed")
    job.prompt = "Delegated parent instruction."
    job.effective_prompt = ""
    job.launch_message_id = ""
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        entries = [entry for entry in view._pending if entry.key != "__working__"]
        assert entries[0].key == "__prompt__"
        assert next(entry for entry in entries if entry.key == durable.id).kind == "user"


@pytest.mark.asyncio
async def test_legacy_duplicate_prompt_is_left_ambiguous_and_synthetic(tmp_path) -> None:
    transcript = Transcript(tmp_path / "child")
    first = await transcript.append_message(Message.user("Delegated parent instruction."))
    second = await transcript.append_message(Message.user("Delegated parent instruction."))
    job = _job_with([], status="completed")
    job.prompt = "Delegated parent instruction."
    job.effective_prompt = ""
    job.launch_message_id = ""
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        entries = [entry for entry in view._pending if entry.key != "__working__"]
        assert entries[0].key == "__prompt__"
        assert next(entry for entry in entries if entry.key == first.id).kind == "user"
        assert next(entry for entry in entries if entry.key == second.id).kind == "user"


@pytest.mark.asyncio
async def test_paged_legacy_duplicates_never_replace_history_or_move_anchor(tmp_path) -> None:
    """A tail page cannot prove an apparent match is unique in older history."""
    transcript = Transcript(tmp_path / "child")
    earlier = await transcript.append_message(Message.user("Delegated parent instruction."))
    for index in range(118):
        await transcript.append_message(Message.assistant(f"older {index}"))
    later = await transcript.append_message(Message.user("Delegated parent instruction."))
    for index in range(20):
        await transcript.append_message(Message.assistant(f"tail {index}"))
    job = _job_with([], status="completed")
    job.prompt = "Delegated parent instruction."
    job.effective_prompt = ""
    job.launch_message_id = ""
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        initial = [entry for entry in view._pending if entry.key != "__working__"]
        assert initial[0].key == "__prompt__"
        assert next(entry for entry in initial if entry.key == later.id).kind == "user"

        view._body.scroll_home(animate=False)
        await pilot.pause()
        anchor = view._body.scroll_y
        while not view._history_exhausted:
            view.action_home()
            await _wait_history(pilot, view)
        complete = [entry for entry in view._pending if entry.key != "__working__"]
        assert complete[0].key == "__prompt__"
        assert next(entry for entry in complete if entry.key == earlier.id).kind == "user"
        assert next(entry for entry in complete if entry.key == later.id).kind == "user"
        # Prepending older pages retains the reader's content-relative anchor;
        # the offset grows rather than snapping back to the synthetic head.
        assert view._body.scroll_y >= anchor


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
async def test_the_title_names_a_non_default_role_and_the_effort_tier() -> None:
    """The header surfaces WHAT kind of child this is and at what level, not
    only its label: a scout on the `hi` tier reads `Subagent · scout · <label>
    … running · hi · <elapsed>`. Both come off the job, recorded at launch."""
    job = _Job("sub-scout", "RetryBudgetScout", status="running")
    job.agent_role = "scout"
    job.effort = "hi"
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        wide = view._title_row(120, "⣾", tools=0).plain
        assert "scout" in wide
        assert "hi" in wide
        # Role rides the breadcrumb (before the label); effort rides the status
        # group (after the state word), which is where the band puts it too.
        assert wide.index("scout") < wide.index("RetryBudgetScout")
        assert wide.index("running") < wide.index(" hi")


@pytest.mark.asyncio
async def test_the_default_task_role_is_not_printed_as_noise() -> None:
    """Every child is a task unless told otherwise, so the word says nothing a
    reader did not already assume. A `task` role is suppressed; the effort tier
    beside it is still shown, because the level is not a default."""
    job = _Job("sub-1", "IngestAuditor", status="running")
    job.agent_role = "task"
    job.effort = "med"
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        wide = view._title_row(120, "⣾", tools=0).plain
        assert "task" not in wide
        assert "med" in wide


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, glyph, word",
    [
        # A SHORT state word (`running`, 7 cells) and a LONG one (`completed`,
        # 9 cells). The long word is the regression guard for MINOR 1: the role
        # chrome (`scout · `, 8 cells) is shorter than ` completed`, so an
        # innermost keep-role offer let a wordless-but-role-kept row win over a
        # worded-but-roleless one at ~30 cells — the invariant held for
        # `running` and broke for `completed`, which the old single-status test
        # could not see. The glyph mirrors what `status_glyph` yields so the row
        # measures the way it paints.
        ("running", "⣾", "running"),
        ("completed", "✓", "completed"),
    ],
)
async def test_the_role_yields_before_the_state_word_as_the_row_tightens(
    status: str, glyph: str, word: str
) -> None:
    """Role is identity sugar the label already half-carries, so it is strictly
    more disposable than the status fields: wherever the role is on the row the
    state word is too (the role is never kept at the cost of the state, even
    when the role chrome is shorter than the word), every field on the row is
    whole, and the role is gone entirely once the row is narrow enough that only
    the identity of the page survives."""
    job = _Job("sub-scout", "RetryBudgetScout", status=status)
    job.agent_role = "scout"
    job.effort = "hi"
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        wide = view._title_row(120, glyph, tools=0).plain
        assert "scout" in wide  # the role shows when there is room
        for width in range(20, 121):
            row = view._title_row(width, glyph, tools=0).plain
            assert cell_len(row) <= width, (width, row)
            # The role never survives a width where the state word had to go —
            # not even where the role is the cheaper of the two to keep.
            if "scout" in row:
                assert word in row, (width, row)
        # At the narrow end the row keeps only what identifies the page; the
        # role has left with the rest of the qualifiers.
        assert "scout" not in view._title_row(24, glyph, tools=0).plain


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


# -- streaming reconciliation -----------------------------------------------
#
# The page used to re-derive its whole transcript and REMOUNT the tail on every
# refresh tick, where the main conversation mutates one long-lived block in
# place. A growing message never compares equal to itself, so its block was
# destroyed and rebuilt per tick: measured on the predecessor at 240 mounts and
# 239 removes over 120 ticks, with a new WorkingBlock (and therefore a restarted
# shimmer sweep and clock) every time. The tests below fail on that code.


def _stream(mid: str, body: str) -> list[dict[str, Any]]:
    """An assistant message still being streamed — start, then one delta."""
    return [
        {"type": "message_start", "message": {"role": "assistant", "id": mid}},
        {"type": "message_update", "message": {"role": "assistant", "id": mid}, "delta": body},
    ]


class _MountCounter:
    """Counts what a refresh actually costs the container."""

    def __init__(self, view: SubagentView) -> None:
        self.mounts = 0
        self.removes = 0
        body = view._body
        append, remove = body.append_block, body.remove_block

        def append_block(block: Any) -> Any:
            self.mounts += 1
            return append(block)

        def remove_block(block: Any) -> Any:
            self.removes += 1
            return remove(block)

        body.append_block = append_block  # type: ignore[method-assign]
        body.remove_block = remove_block  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_a_streaming_message_keeps_one_block_across_refreshes() -> None:
    """The row being streamed is UPDATED, not rebuilt.

    Block identity is the assertion because it is what the user feels: a
    rebuilt AssistantBlock is constructed unmounted (so it folds at the
    fallback width for a frame, which reads as the message flashing narrow)
    and it discards the incremental markdown cache, so the settling final
    response is re-lexed in full on every tick.
    """
    # Seeded with its first token rather than empty so the row already EXISTS
    # when the counter is installed. From an empty message the block's first
    # appearance would land inside the loop and count as a mount, which is a
    # legitimate one — and an assertion that has to tolerate it cannot tell a
    # first mount from a duplicate row appended beside the streaming one.
    text = "opening "
    job = _job_with(_stream("m1", text), status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await pilot.pause()
        counter = _MountCounter(view)
        identities: set[int] = set()
        # Bound before the loop so the assertions below are reachable even if
        # the body never runs: an empty loop would otherwise leave `block`
        # unbound and turn a real failure into a NameError at the assert.
        block: AssistantBlock | None = None
        for index in range(12):
            text += f"token{index} "
            job.trajectory = _stream("m1", text)
            app._refresh_subagent_view()
            await pilot.pause()
            block = next(b for b in view._body.blocks() if isinstance(b, AssistantBlock))
            identities.add(id(block))
        assert len(identities) == 1, "the streaming block was rebuilt"
        assert counter.removes == 0, f"{counter.removes} rows torn down while streaming"
        # Identity alone would not catch a DUPLICATE row appended beside the
        # streaming one — the set stays size 1 while the page grows a second
        # copy of the message. The sibling settling test asserts this too.
        assert counter.mounts == 0, f"{counter.mounts} rows remounted while streaming"
        # The message on screen is the whole accumulated text, not a fragment.
        assert block is not None
        assert block.text().startswith("opening")
        assert "token0 " in block.text() and block.text().endswith("token11")


@pytest.mark.asyncio
async def test_the_working_tail_survives_every_refresh() -> None:
    """The tail line owns an animation and a clock, and both restart when the
    widget does. It is pinned (`TranscriptView.pin_tail`, the same lever the
    main conversation uses) so rows mount ABOVE it instead of replacing it."""
    job = _job_with(_stream("m1", "opening"), status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await pilot.pause()
        identities: set[int] = set()
        events = list(job.trajectory)
        for index in range(6):
            # Rows keep arriving UNDER the tail: each pass adds a settled tool,
            # which is exactly the case that used to drag the tail down with it.
            events.append(_call(f"c{index}", "read", path=f"f{index}.py"))
            events.append(_result(f"c{index}", "read", "ok"))
            job.trajectory = list(events)
            app._refresh_subagent_view()
            await pilot.pause()
            tail = [b for b in view._body.blocks() if isinstance(b, WorkingBlock)]
            assert len(tail) == 1, "the page grew a second working line"
            identities.add(id(tail[0]))
        assert len(identities) == 1, "the working line was rebuilt under the reader"
        # It is still LAST: the whole point of pinning it.
        assert isinstance(view._body.blocks()[-1], WorkingBlock)


@pytest.mark.asyncio
async def test_a_settling_mid_list_tool_does_not_rebuild_the_rows_beneath_it() -> None:
    """One parallel batch settling oldest-first used to rebuild the whole
    suffix under each result — 272 mounts for 16 pairs, quadratic in the rows
    below the one that changed. Settling a card is now an update to that card."""
    events: list[dict[str, Any]] = []
    for index in range(4):
        events.extend(_text(f"m{index}", f"step {index}"))
        events.append(_call(f"c{index}", "bash", command=f"step {index}"))
    job = _job_with(list(events), status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        await pilot.pause()
        cards = {id(block) for block in view._body.blocks() if isinstance(block, ToolCard)}
        assert len(cards) == 4
        counter = _MountCounter(view)
        for index in range(4):
            events.append(_result(f"c{index}", "bash", "ok"))
            job.trajectory = list(events)
            app._refresh_subagent_view()
            await pilot.pause()
        assert counter.mounts == 0, f"{counter.mounts} rows remounted by settling tools"
        assert counter.removes == 0, f"{counter.removes} rows torn down by settling tools"
        # Same widgets, now carrying their outcome and the executor's duration.
        settled = [block for block in view._body.blocks() if isinstance(block, ToolCard)]
        assert {id(card) for card in settled} == cards
        assert all(card.is_finalized() for card in settled)


@pytest.mark.asyncio
async def test_a_row_folds_at_the_body_width_it_is_mounted_into() -> None:
    """No block is built at the 80-column fallback and re-folded a frame later.

    That flash was visible on ANY mount path, not only the streaming one: a
    block authors its own rows and PINS its height to the count of them, so a
    fallback-width build pinned a fallback-width height until the resize
    landed. On a 140-column terminal the block measurably built at 80 and
    settled at 134.

    The probe is installed BEFORE the page opens, which is the whole point.
    Round 1 of review caught the predecessor of this test arming it after
    `_open`, so every fold performed during the initial mount was invisible to
    the assertion: the test passed while the page still built its first blocks
    at 80 and re-folded them a frame later, painted (measured on 2 of 20 opens
    at 140x34 — `on_mount` runs before the body has a region, so the width
    every source could report is 0). The mount path is the one a reader
    actually sees flash, so it is the one that has to be covered.

    The page is opened and FILLED in one synchronous beat, which is the
    ordering `_open_subagent_view` produces whenever its deferred fill wins the
    race with layout. Left to the scheduler the race resolves the harmless way
    on most runs (2 of 20 opens showed the flash), so driving the ordering is
    what makes this a regression test rather than a coin toss.
    """
    prose = "a paragraph of the child's prose that folds differently at 80 columns. "
    job = _job_with(_stream("m1", prose * 3) + _stream("m2", prose * 2), status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))

    # Every fold performed by ANY path, armed before the page exists. The
    # predecessor of this test installed its probe AFTER the page was open, so
    # the mount folds it names were invisible to it; it passed while the flash
    # still happened. The streaming path is covered too: a rebuilt block is
    # constructed with no parent to ask for a width, so it folded at 80 and
    # re-folded once `on_resize` landed — a frame of narrow text per tick.
    folds: list[int] = []
    original = AssistantBlock._apply_rows

    def record(self: AssistantBlock, text: Any) -> Any:
        folds.append(self._flat_width())
        return original(self, text)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(AssistantBlock, "_apply_rows", record)
    try:
        async with app.run_test(size=(140, 34)) as pilot:
            for _ in range(80):
                await pilot.pause()
                if app._session is not None:
                    break
            app._append_block(UserBlock("audit the ingest path"))
            app._refresh_band()
            await pilot.pause()

            app._open_subagent_view(str(job.id))
            view = app.query_one(SubagentView)
            # The condition under test: mounted, not yet laid out, so every
            # width the page could read is 0 and a naive fold falls to 80.
            assert view._body.scrollable_content_region.width == 0
            app._refresh_subagent_view(str(job.id))
            for _ in range(4):
                await pilot.pause()

            body_width = view._body.scrollable_content_region.width
            assert body_width > 80, "the fixture needs a body wider than the fallback"
            mount_folds = list(folds)

            job.trajectory = _stream("m1", prose * 4) + _stream("m2", prose * 2)
            app._refresh_subagent_view()
            for _ in range(4):
                await pilot.pause()

            assert mount_folds, "the mount path performed no fold to check"
            assert folds != mount_folds, "the streaming path performed no fold to check"
            # Stated as the FALLBACK rather than as "== body_width" so the
            # failure names the defect: a fold at 80 is the flash, whatever
            # else the ladder may legitimately report mid-layout.
            assert FALLBACK_WIDTH not in folds, folds
            assert all(width == body_width for width in folds), folds
            block = next(b for b in view._body.blocks() if isinstance(b, AssistantBlock))
            assert block._built_width == body_width
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_a_settled_message_is_committed_in_the_block_it_streamed_into() -> None:
    """Settling COMMITS the row rather than rebuilding it.

    The streaming splice concatenates a frozen prefix with the volatile tail,
    which cannot produce the blank row between two paragraphs until the whole
    message is re-rendered once (`AssistantBlock.finalize_text`). The main
    conversation does that at `message_end`; this page has no such event, so it
    commits on the refresh that observes the job settle. Without it a finished
    child kept its mid-stream fold forever — a paragraph break the same text
    shows in the main transcript, missing here.
    """
    body = "First paragraph here.\n\nSecond paragraph here."
    job = _job_with(_stream("m1", body), status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(140, 34)) as pilot:
        view = await _open(pilot, app, job)
        await pilot.pause()
        block = next(b for b in view._body.blocks() if isinstance(b, AssistantBlock))
        streaming = str(block.renderable)
        assert not block.is_finalized()

        # The child stops. Note the trajectory does NOT change: the settle is
        # the job's status moving, which is the case a text-only diff misses.
        job.status = "completed"
        job.settled_at = job.start_time + 5
        app._refresh_subagent_view()
        await pilot.pause()

        settled = next(b for b in view._body.blocks() if isinstance(b, AssistantBlock))
        assert id(settled) == id(block), "the message was rebuilt to settle it"
        assert settled.is_finalized()
        rows = str(settled.renderable).split("\n")
        assert any(not row.strip() for row in rows), rows
        assert len(rows) == len(streaming.split("\n")) + 1


@pytest.mark.asyncio
async def test_a_finished_message_is_committed_while_the_child_is_still_working() -> None:
    """Completeness is a property of the ROW, not of the job (review round 1, M1).

    A running child's transcript is mostly finished messages — the prose it
    writes between tool calls, each of which already had its `message_end`.
    Deriving "may this be committed?" from the job status alone left every one
    of them rendering with the streaming splice's concatenated fold, which
    cannot produce the blank row between two paragraphs. Worse, `_apply_rows`
    pins height to the row count, so those rows were pinned SHORT and grew
    when the job eventually settled: measured 5 -> 6 and 2 -> 3 rows on this
    exact fixture, a one-row content shift per message at settle, which is the
    reflow class this page's in-place reconciliation exists to remove.

    So the assertion is the pair: the paragraph break is present WHILE the
    child works, and the row count does not move when it stops.
    """
    events = _text("m1", "Plan:\n\n- step one\n- step two\n\nNow running it.")
    events += _text("m2", "That worked.\n\nMoving on to the next file.")
    # A call the child is still running, so the job is unambiguously live
    # while both messages above are complete.
    events.append(_call("c0", "bash", command="pytest -q"))
    job = _job_with(events, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(140, 34)) as pilot:
        view = await _open(pilot, app, job)
        for _ in range(4):
            await pilot.pause()

        live = [b for b in view._body.blocks() if isinstance(b, AssistantBlock)]
        assert len(live) == 2, live
        for block in live:
            assert block.is_finalized(), "a finished message was left uncommitted"
            rows = str(block.renderable).split("\n")
            assert any(not row.strip() for row in rows), rows
        live_rows = [len(str(block.renderable).split("\n")) for block in live]

        # The child stops. The trajectory does not change — only the status —
        # which is exactly the refresh that used to grow every one of these
        # rows by one.
        job.status = "completed"
        job.settled_at = job.start_time + 5
        app._refresh_subagent_view()
        for _ in range(4):
            await pilot.pause()

        settled = [b for b in view._body.blocks() if isinstance(b, AssistantBlock)]
        assert [id(b) for b in settled] == [id(b) for b in live], "rows were rebuilt at settle"
        settled_rows = [len(str(block.renderable).split("\n")) for block in settled]
        assert (
            settled_rows == live_rows
        ), f"content reflowed at settle: {live_rows} -> {settled_rows}"


@pytest.mark.asyncio
async def test_an_evicted_message_end_does_not_uncommit_a_settled_row() -> None:
    """A fold that LOST the `message_end` must not un-finalize the row.

    The trajectory is a rolling window, so `message_end` is as evictable as
    any other event: a later fold of the same message can report it as still
    streaming. Accepting that would un-finalize a committed block, and
    `update_entry_block` answers a finalized block whose text moved with a
    REMOUNT — so a row that had settled correctly would be torn down and
    rebuilt under the reader. `_supersedes` refuses the downgrade for the same
    reason it refuses a shorter text.
    """
    body = "First paragraph.\n\nSecond paragraph."
    job = _job_with(_text("m1", body) + [_call("c0", "bash", command="go")], status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(140, 34)) as pilot:
        view = await _open(pilot, app, job)
        for _ in range(4):
            await pilot.pause()
        block = next(b for b in view._body.blocks() if isinstance(b, AssistantBlock))
        assert block.is_finalized()
        rows = len(str(block.renderable).split("\n"))

        # The window rolls: the same message is now reported WITHOUT its end.
        job.trajectory = _stream("m1", body) + [_call("c0", "bash", command="go")]
        app._refresh_subagent_view()
        for _ in range(4):
            await pilot.pause()

        after = next(b for b in view._body.blocks() if isinstance(b, AssistantBlock))
        assert id(after) == id(block), "the settled row was rebuilt by a worse fold"
        assert after.is_finalized(), "an evicted message_end un-committed the row"
        assert len(str(after.renderable).split("\n")) == rows
        # The page's own record, which is where the refusal happens: the block
        # above stays finalized either way (a finalized block answers an
        # unchanged text with agreement), so asserting only on the widget
        # would pass with the guard removed.
        assert view._known["m1"].complete, "the page forgot the row was complete"


@pytest.mark.asyncio
async def test_a_history_page_lands_below_the_truncation_note(tmp_path) -> None:
    """The prepend index accounts for the head block (review round 1, N1).

    `prefix` is counted over ENTRIES, which exclude the truncation note, but
    `TranscriptView.insert_blocks` indexes the BODY's list, which holds that
    note at 0 — `_sync_body` mounts it outside the diffed sequence and keeps it
    out of `self._blocks` (measured: `view._blocks` 168 against a body of 169).
    Both a truncated trajectory and a durable history page are needed at once
    for the skew to show, which is why it went unnoticed; without the offset
    the history page is inserted ABOVE the note that is supposed to head the
    page.
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(140):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    # Over the cap, so the page carries its truncation note as a head block.
    events: list[dict[str, Any]] = []
    while len(events) < TRAJECTORY_MAX_EVENTS + 10:
        events.extend(_text(f"m{len(events)}", f"live {len(events)}"))
    job = _job_with(events, status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert view._head_block is not None, "the fixture needs a truncated trajectory"
        assert view._body.blocks()[0] is view._head_block

        before = len(view._body.blocks())
        view.action_home()
        await _wait_history(pilot, view)
        await _wait_geometry_settled(pilot, view._body)
        assert len(view._body.blocks()) > before, "no history page was prepended"

        # The note still heads the page, and nothing was inserted above it.
        assert view._body.blocks()[0] is view._head_block, "a history page landed above the note"


@pytest.mark.asyncio
async def test_a_history_page_lands_in_order_when_the_cap_is_crossed_mid_read(tmp_path) -> None:
    """The prepend offset keys on the note's POSITION, not its existence.

    `_head_block` is created lazily and appended at whatever length the body
    has reached, so it heads the list only when the child was ALREADY
    truncated when the page opened. A child that is under the cap at open and
    crosses it while the reader watches mounts the note mid-list (measured:
    index 5 of 257) — and correcting for a note that is not above the rows
    pushes the prepended page one row too LOW, which reorders the durable
    window (`durable 40` above `durable 0`).

    That is the mirror image of round 1's N1 and it is the case `origin/main`
    happened to get right, so it needs its own guard: the two scenarios are
    disjoint and a fix conditioned on existence alone passes the other test
    while breaking this one (review round 2, M3).
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(140):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    # UNDER the cap at open: no truncation note exists yet.
    job = _job_with(_text("m0", "live 0"), status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert view._head_block is None, "the fixture must start under the cap"

        # Now cross the cap under the reader, which appends the note mid-list.
        events: list[dict[str, Any]] = []
        while len(events) < TRAJECTORY_MAX_EVENTS + 10:
            events.extend(_text(f"m{len(events)}", f"live {len(events)}"))
        job.trajectory = events
        app._refresh_subagent_view()
        for _ in range(4):
            await pilot.pause()
        assert view._head_block is not None, "the fixture needs to cross the cap"
        assert (
            view._body.blocks()[0] is not view._head_block
        ), "this test only means anything while the note is NOT at index 0"

        view.action_home()
        await _wait_history(pilot, view)
        await _wait_geometry_settled(pilot, view._body)

        # The durable window stays in order: the page that was paged in must
        # not be split around the row it belongs above.
        durable = [row for row in view.rendered_rows() if row.strip().startswith("durable ")]
        ordering = [int(row.strip().split()[1]) for row in durable]
        assert ordering == sorted(ordering), f"the prepended page landed out of order: {ordering}"
