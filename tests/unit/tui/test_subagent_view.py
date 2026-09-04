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
import copy
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pytest
from rich.cells import cell_len
from textual.events import MouseScrollUp

from local_operator.harness.comms import (
    HUB_COMMUNICATION_CUSTOM_TYPE,
    HUB_MESSAGE_TYPE,
    SubagentComms,
)
from local_operator.harness.jobs import (
    CANCELLED_BEFORE_START,
    TRAJECTORY_SEQ_KEY,
    AsyncJob,
    AsyncJobManager,
)
from local_operator.harness.types import (
    CustomMessage,
    Message,
    NoticeEvent,
    TextContent,
    ToolCall,
)
from local_operator.session.session import Session
from local_operator.session.transcript import (
    Transcript,
    TranscriptEntry,
    TranscriptPage,
)
from local_operator.tui.app import SUBAGENT_LAYOUT_CLASS, OperatorApp
from local_operator.tui.widgets import subagent_view
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
    READ_ONLY_NOTE,
    TRAJECTORY_MAX_EVENTS,
    TRUNCATION_NOTE,
    InstructionBlock,
    SubagentView,
    _mark_consecutive_notices,
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
    """A job carrying its OWN copy of ``trajectory``.

    The copy is the fix for a cross-test leak, not tidiness. Callers pass the
    module-level :data:`TRAJECTORY`, and three tests mutate what they get back
    to drive a live update: ``job.trajectory.append(...)`` at the "mounted
    once" and "coalesced frame" cases, and ``del job.trajectory[:3]`` at the
    eviction case. Assigning by reference made those edits land on the SHARED
    list, so every test that ran afterwards folded a trajectory with an extra
    tool call in it, or with its first message missing.

    That is why this file failed only in company and passed alone, and why the
    failure moved around: which test sees the corruption depends on execution
    order, so xdist changes the symptom (2 failures at ``-n4``, 1 at ``-n0``)
    without changing the cause. Observed as
    ``assert [...'ToolCard'] == [...'AssistantBlock']`` and
    ``assert 'history unavailable' in 'read-only'``.

    A shallow copy is sufficient and deliberate: the leak is list membership
    (append/delete), and no test mutates an event dict in place. Copying the
    list here rather than asking each test to remember means a new test cannot
    reintroduce this by writing the obvious thing.
    """
    job = _Job("sub-1", "audit the ingest path", status=status)
    if status != "running":
        job.settled_at = job.start_time + 42
    job.trajectory = list(trajectory)
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


def _evicting_refreshes(
    trajectory: list[dict[str, Any]],
    ticks: int,
    *,
    stamped: bool = False,
) -> dict[str, Any]:
    """Refresh the page ``ticks`` times while the engine evicts from the front.

    Reproduces the two halves of the live coupling that produced duplicate
    rows: the writer appends one event and drops the oldest (keeping the list
    at the cap), and the view accumulates folded rows by key and never removes
    one. Returns the accumulated ``key -> entry`` map the page would hold, so a
    test can count how many ROWS one event produced.
    """
    known: dict[str, Any] = {}
    seq = len(trajectory)
    for tick in range(ticks):
        for entry in fold_trajectory(trajectory):
            if entry.head:
                continue
            known.setdefault(entry.key, entry)
        filler: dict[str, Any] = {
            "type": "message_start",
            "message": {"role": "assistant", "id": f"filler-{tick}"},
        }
        if stamped:
            filler[TRAJECTORY_SEQ_KEY] = seq
        seq += 1
        trajectory.append(filler)
        overflow = len(trajectory) - TRAJECTORY_MAX_EVENTS
        if overflow > 0:
            del trajectory[:overflow]
    return known


def _filler_events(count: int, *, stamped: bool = False) -> list[dict[str, Any]]:
    """Events that hold a trajectory slot without painting a row."""
    events: list[dict[str, Any]] = []
    for index in range(count):
        event: dict[str, Any] = {
            "type": "message_start",
            "message": {"role": "assistant", "id": f"seed-{index}"},
        }
        if stamped:
            event[TRAJECTORY_SEQ_KEY] = index
        events.append(event)
    return events


@pytest.mark.parametrize("stamped", [False, True])
def test_one_notice_stays_one_row_while_the_engine_evicts(stamped: bool) -> None:
    """A single error notice must not multiply as the trajectory rotates.

    The retained window evicts from the FRONT, so every append shifts a
    surviving event one slot left. Keying a notice by that offset re-spelled
    the SAME error on every 1 Hz refresh, and because the page only ever adds
    rows, each new spelling mounted another identical copy — the reported
    defect was eleven stacked "Invalid arguments" lines under a running tool.

    Parametrized over both identity sources: the writer's monotonic stamp, and
    the content fallback a trajectory from an older release still has to use.
    """
    error = "Invalid arguments: argument 'edits' does not match type array"
    trajectory = _filler_events(TRAJECTORY_MAX_EVENTS - 1, stamped=stamped)
    notice: dict[str, Any] = {"type": "notice", "kind": "error", "text": error}
    if stamped:
        notice[TRAJECTORY_SEQ_KEY] = TRAJECTORY_MAX_EVENTS - 1
    trajectory.append(notice)

    known = _evicting_refreshes(trajectory, 12, stamped=stamped)
    rows = [entry for entry in known.values() if entry.kind == "notice" and entry.text == error]
    assert len(rows) == 1


@pytest.mark.parametrize("stamped", [False, True])
def test_two_distinct_notices_with_identical_text_stay_two_rows(stamped: bool) -> None:
    """Deduplication must not be bought by collapsing distinct events.

    The same error legitimately occurring twice is two things that happened,
    and a page that folds them into one row under-reports the child's run.
    """
    error = "Invalid arguments: argument 'edits' does not match type array"
    trajectory: list[dict[str, Any]] = []
    for index in range(2):
        notice: dict[str, Any] = {"type": "notice", "kind": "error", "text": error}
        if stamped:
            notice[TRAJECTORY_SEQ_KEY] = index
        trajectory.append(notice)

    rows = [
        entry for entry in _mark_consecutive_notices(fold_trajectory(trajectory)) if not entry.head
    ]
    # Two rows still, because collapsing them would under-report the run
    # (the data model is right). They are no longer byte-identical: a
    # consecutive identical pair is the visual signature of the old
    # duplicate-render bug, so each row carries its ordinal (issue #405).
    assert len(rows) == 2
    assert len({entry.key for entry in rows}) == 2
    assert rows[0].text == f"{error}  (1/2)"
    assert rows[1].text == f"{error}  (2/2)"


def test_consecutive_identical_notices_carry_an_ordinal() -> None:
    """A genuine double failure must not look like the old duplicate-row bug.

    Mutation: drop the ordinal and this assertion dies. A count-collapse
    (``×2`` on one row) would also die — the data model keeps two rows.
    """
    error = "Invalid arguments: argument 'edits' does not match type array"
    trajectory = [
        {TRAJECTORY_SEQ_KEY: 0, "type": "notice", "kind": "error", "text": error},
        {TRAJECTORY_SEQ_KEY: 1, "type": "notice", "kind": "error", "text": error},
        {TRAJECTORY_SEQ_KEY: 2, "type": "notice", "kind": "error", "text": error},
    ]
    rows = [
        entry for entry in _mark_consecutive_notices(fold_trajectory(trajectory)) if not entry.head
    ]
    assert [entry.text for entry in rows] == [
        f"{error}  (1/3)",
        f"{error}  (2/3)",
        f"{error}  (3/3)",
    ]
    # Intervening prose breaks the run: two identical notices with a
    # sentence between them already read as two attempts, so they stay
    # unmarked. Numbering those would invent a sequence the page has no
    # right to.
    mixed = [
        {TRAJECTORY_SEQ_KEY: 0, "type": "notice", "kind": "error", "text": error},
        {
            TRAJECTORY_SEQ_KEY: 1,
            "type": "message_end",
            "message": {
                "role": "assistant",
                "id": "m1",
                "content": [{"type": "text", "text": "retrying"}],
            },
        },
        {TRAJECTORY_SEQ_KEY: 2, "type": "notice", "kind": "error", "text": error},
    ]
    mixed_rows = [
        entry for entry in _mark_consecutive_notices(fold_trajectory(mixed)) if not entry.head
    ]
    notices = [entry for entry in mixed_rows if entry.kind == "notice"]
    assert [entry.text for entry in notices] == [error, error]


@pytest.mark.asyncio
async def test_the_page_paints_ordinals_on_a_consecutive_identical_pair() -> None:
    """The overlay has to reach the mounted page, not just the pure fold.

    ``_supersedes`` refuses notice revisions, so marking inside the fold
    would freeze the first unmarked row and leave its twin as ``(2/2)``.
    Marking at assemble time is what makes both rows update. Mutation:
    mark only in ``fold_trajectory`` and this assertion dies on the
    first row still reading the bare error.
    """
    error = "Invalid arguments: argument 'edits' does not match type array"
    events = [
        {TRAJECTORY_SEQ_KEY: 0, "type": "notice", "kind": "error", "text": error},
        {TRAJECTORY_SEQ_KEY: 1, "type": "notice", "kind": "error", "text": error},
    ]
    job = _job_with(events, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        view = await _open(pilot, app, job)
        for _ in range(6):
            await pilot.pause()
        rows = [row for row in view.rendered_rows() if error in row]
        assert rows == [f"{error}  (1/2)", f"{error}  (2/2)"]


def test_id_less_anchored_key_cannot_collide_with_a_real_id() -> None:
    """An id-less fallback must not share a key with a producer-issued id.

    On the pre-separator spelling, an id-less message stamped at sequence 5
    keyed as ``ms5``; a child that emitted a real ``message.id`` of literally
    ``"ms5"`` folded onto the same row (issue #408, confirmed on base). The
    colon makes that collision unrepresentable: ``m:s5`` cannot equal ``ms5``.
    The same shape holds for an id-less tool start (``t:s9`` vs ``ts9``).

    Mutation: revert the separator and this assertion dies — the two
    messages collapse to one row whose text is the REAL-ID one.
    """
    messages = fold_trajectory(
        [
            {
                TRAJECTORY_SEQ_KEY: 5,
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ID-LESS"}],
                },
            },
            {
                TRAJECTORY_SEQ_KEY: 6,
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "id": "ms5",
                    "content": [{"type": "text", "text": "REAL-ID"}],
                },
            },
        ],
        settled=True,
    )
    texts = [entry for entry in messages if entry.kind == "text"]
    assert {entry.text for entry in texts} == {"ID-LESS", "REAL-ID"}
    assert {entry.key for entry in texts} == {"m:s5", "ms5"}

    tools = fold_trajectory(
        [
            {
                TRAJECTORY_SEQ_KEY: 9,
                "type": "tool_execution_start",
                "tool_name": "read",
                "args": {"path": "a"},
            },
            {
                TRAJECTORY_SEQ_KEY: 10,
                "type": "tool_execution_start",
                "tool_call_id": "ts9",
                "tool_name": "bash",
                "args": {"command": "b"},
            },
        ],
        settled=True,
    )
    tool_rows = [entry for entry in tools if entry.kind == "tool"]
    assert {entry.tool_name for entry in tool_rows} == {"read", "bash"}
    assert {entry.key for entry in tool_rows} == {"t:s9", "ts9"}


def test_id_less_message_and_tool_survive_eviction_without_duplicating() -> None:
    """The message and tool fallbacks carried the same positional defect.

    A child that streams prose with no message id, or a tool call with no
    ``tool_call_id``, was keyed by offset too — so an unstamped trajectory
    duplicated its text block and its tool card exactly as it duplicated a
    notice. They are one defect and are fixed by one anchor.
    """
    trajectory = _filler_events(TRAJECTORY_MAX_EVENTS - 3)
    trajectory += [
        {"type": "message_start", "message": {"role": "assistant"}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "prose with no id"}],
            },
        },
        {"type": "tool_execution_start", "tool_name": "bash", "args": {"command": "ls"}},
    ]

    known = _evicting_refreshes(trajectory, 12)
    texts = [entry for entry in known.values() if entry.kind == "text"]
    tools = [entry for entry in known.values() if entry.kind == "tool"]
    assert [entry.text for entry in texts] == ["prose with no id"]
    assert [entry.tool_name for entry in tools] == ["bash"]


#: A realistic long agent error: nine identical stack frames, so two different
#: failures agree for ~900 characters and differ only in the final line. This
#: is the shape that made a head-only fingerprint drop a row (round 2, F4/D5).
_SHARED_TRACEBACK = "Traceback (most recent call last):\n" + "".join(
    f'  File "/app/local_operator/mod{index}.py", line {100 + index}, in handler\n'
    f"    result = step{index}(payload, context)\n"
    for index in range(9)
)


def _long_notice(tail: str) -> dict[str, Any]:
    return {"type": "notice", "kind": "error", "text": _SHARED_TRACEBACK + tail}


def test_unstamped_fingerprint_is_bounded_for_a_1hz_path() -> None:
    """The fallback may not pay for an unbounded payload on a 1 Hz path.

    Fingerprinting whole events made a window of id-less notices carrying
    20 KB each cost ~133 ms per fold (review round 1, F2). The bound is
    asserted through the observable property rather than a timing number: the
    hashed projection of a value cannot grow with the payload.
    """
    small = _long_notice("KeyError: 'tool_call_id'")
    huge = {"type": "notice", "kind": "error", "text": "y" * 200_000}

    assert subagent_view._DIGEST_VALUE_CHARS < len(huge["text"])
    # Stable for equal input — the property row identity depends on.
    assert subagent_view._digest(small) == subagent_view._digest(dict(small))
    assert subagent_view._digest(huge) == subagent_view._digest(dict(huge))
    # A difference inside the head is seen, as is one only in length.
    assert subagent_view._digest(huge) != subagent_view._digest(
        {"type": "notice", "kind": "error", "text": "y" * 199_999}
    )


def test_two_different_errors_sharing_a_long_prefix_both_reach_the_page() -> None:
    """Distinct events must not share a key, INCLUDING across folds.

    The regression this pins (round 2, F4/D5): bounding the fingerprint to a
    value's head alone made two different failures from one deep stack share a
    fingerprint. Within a single fold that was harmless, because the occurrence
    ordinal separated them — which is exactly what the previous version of this
    test asserted, and why it passed while the defect was live.

    Across folds it is content LOSS. ``_Anchors`` is built per fold with an
    empty ``_seen``, so once the earlier event has been front-evicted the later
    one takes ordinal 0, collides with the key the page already holds, and —
    since notices never supersede — is silently discarded. The reader is shown
    one stale error where two different things failed.

    Both regimes are asserted here, because only the second one ever broke.
    """
    first = _long_notice("KeyError: 'tool_call_id'")
    second = _long_notice("ValueError: malformed duration")
    assert first["text"][:512] == second["text"][:512]  # the aliasing shape

    # Co-resident: never at risk, kept so a fix cannot regress it.
    resident = [entry for entry in fold_trajectory([first, second]) if not entry.head]
    assert len({entry.key for entry in resident}) == 2

    # Cross-fold: the first error is evicted before the second ever happens,
    # so the page can only tell them apart by their fingerprints.
    trajectory: list[dict[str, Any]] = [first]
    trajectory += _filler_events(TRAJECTORY_MAX_EVENTS - 1)
    known = _evicting_refreshes(trajectory, TRAJECTORY_MAX_EVENTS + 5)
    assert not any(event.get("text") == first["text"] for event in trajectory)

    trajectory.append(second)
    del trajectory[: max(0, len(trajectory) - TRAJECTORY_MAX_EVENTS)]
    for entry in fold_trajectory(trajectory):
        if not entry.head:
            known.setdefault(entry.key, entry)

    endings = sorted(
        entry.text.splitlines()[-1] for entry in known.values() if entry.kind == "notice"
    )
    assert endings == ["KeyError: 'tool_call_id'", "ValueError: malformed duration"]


def test_a_repeated_identical_notice_still_folds_to_one_row_across_folds() -> None:
    """The D2 ceiling, pinned so the F4 fix does not quietly widen it.

    Identical wording that is never co-resident still collapses to one row —
    documented, deliberate, and the safe direction. The point of this test is
    that the fix for F4 separates DIFFERENT content without also making the
    same content register twice.
    """
    error = _long_notice("KeyError: 'tool_call_id'")
    trajectory: list[dict[str, Any]] = [dict(error)]
    trajectory += _filler_events(TRAJECTORY_MAX_EVENTS - 1)
    known = _evicting_refreshes(trajectory, TRAJECTORY_MAX_EVENTS + 5)

    trajectory.append(dict(error))
    del trajectory[: max(0, len(trajectory) - TRAJECTORY_MAX_EVENTS)]
    for entry in fold_trajectory(trajectory):
        if not entry.head:
            known.setdefault(entry.key, entry)

    notices = [entry for entry in known.values() if entry.kind == "notice"]
    assert len(notices) == 1


def test_a_large_payload_nested_in_a_dict_is_bounded_and_still_distinguished() -> None:
    """The bound must reach values that are not top-level strings (round 2, F5).

    ``repr`` builds its entire result before anything can slice it, so a large
    payload inside ``message`` or ``args`` — which is where the id-less message
    and tool paths actually carry one — was materialized in full on every fold.
    The bounded repr must still tell two such events apart, or F5's fix would
    reintroduce F4 by another route.
    """
    payload = "x" * 20_000
    first = {
        "type": "message_start",
        "message": {"role": "assistant", "content": [{"type": "text", "text": payload + "ALPHA"}]},
    }
    second = {
        "type": "message_start",
        "message": {"role": "assistant", "content": [{"type": "text", "text": payload + "BETA"}]},
    }

    assert subagent_view._digest(first) != subagent_view._digest(second)
    assert subagent_view._digest(first) == subagent_view._digest(copy.deepcopy(first))
    # The bounded repr never renders the whole payload.
    assert len(subagent_view._VALUE_REPR.repr(first["message"])) < len(payload)


def test_relay_stamps_a_sequence_that_eviction_cannot_renumber() -> None:
    """The writer's half of the contract the fold depends on.

    The stamp counts events RELAYED, not events retained, so it keeps rising
    past the cap and an evicted event's number is never reissued to a later
    one. Without that, a reader keying by it would still collide after a
    rotation.
    """
    from local_operator.harness.subagent import TRAJECTORY_CAP, _make_relay

    job = AsyncJob(id="j1", type="task", status="running", label="child", start_time=1.0)
    job.trajectory = []

    async def _emit(_event: Any) -> None:
        return None

    relay = _make_relay(
        "j1", "child", job, cast(Any, None), _emit, lambda _text: None, {"text": "", "error": None}
    )

    async def _drive() -> None:
        for index in range(TRAJECTORY_CAP + 5):
            await relay(NoticeEvent(text=f"notice {index}", kind="error"))

    asyncio.run(_drive())

    assert job.trajectory is not None
    stamps = [event[TRAJECTORY_SEQ_KEY] for event in job.trajectory]
    # Evicted to the cap, still numbered by relay order, and strictly rising.
    assert len(stamps) == TRAJECTORY_CAP
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
    assert stamps[-1] == TRAJECTORY_CAP + 4


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
    """Open the child page once the boot worker has adopted a session.

    Waits on the WORKER, not on a frame budget. ``on_mount`` hands
    ``_boot_session`` to ``run_worker(group="session")`` so first paint does
    not wait on the factory; every assertion below reads through that session,
    so an unbooted app is a different test. The previous 80-pause loop was a
    bet that 80 frames outlast the factory, the same class #461 converted
    everywhere else in this file.
    """
    workers = [w for w in app.workers if w.group == "session"]
    if workers:
        await app.workers.wait_for_complete(workers)
    assert app._session is not None, (
        "no session worker is pending and no session was adopted — the "
        "boot worker never ran, so waiting here would have waited on nothing"
    )
    app._append_block(UserBlock("audit the ingest path"))
    app._refresh_band()
    await pilot.pause()
    app._open_subagent_view(str(job.id))
    await pilot.pause()
    return app.query_one(SubagentView)


@pytest.mark.parametrize("size", [(100, 30), (140, 40)])
@pytest.mark.asyncio
async def test_the_page_takes_the_whole_view_when_opened_from_the_splash(
    size: tuple[int, int],
) -> None:
    """The child page gets the same geometry from the splash as over a
    conversation.

    ``Screen.boot`` is a whole second layout (centred width-clamped input card,
    plus rows reserved below it in the dock's own padding), and this mode
    replaces the region it lays out, so with both applied the page rendered into
    the leftovers around a card that still held its clamp.

    BOTH dimensions are asserted because the collision changes shape with the
    terminal: at 140x40 it costs the page rows, while at 100x30 the composition
    reserves none and the damage is purely horizontal — the shell clamped to 73
    cells at column 12 instead of 96 at column 1. A rows-only assertion passes
    against the broken tree at that size (review round 1, F1/F2).
    """

    async def measure(seed_conversation: bool) -> tuple[int, int, int, int]:
        session = FakeSession()
        session.jobs = _fake_jobs(_Job("sub-1", "audit the ingest path"))
        app = OperatorApp(_async_factory(session))
        async with app.run_test(size=size) as pilot:
            for _ in range(80):
                await pilot.pause()
                if app._session is not None:
                    break
            if seed_conversation:
                app._append_block(UserBlock("audit the ingest path"))
                await pilot.pause()
                assert not app.screen.has_class("boot"), size
            else:
                # Opened straight off the splash: no block is appended, so the
                # boot layout is still up. This is the route that was broken.
                assert app.screen.has_class("boot"), size
            app._refresh_band()
            await pilot.pause()
            app._open_subagent_view("sub-1")
            await pilot.pause()
            await pilot.pause()
            view = app.query_one(SubagentView)
            dock = app.query_one("#input-dock")
            shell = app.query_one("#input-shell")
            assert not app.screen.has_class("boot-card"), (
                f"{size}: the boot card's clamp survived into the page "
                f"(#input-shell is {shell.size.width} cells at x={shell.region.x})"
            )
            assert dock.outer_size.height == dock.size.height, size
            assert app.screen.virtual_size.height <= app.screen.size.height, size
            return (view.size.height, view.size.width, shell.size.width, shell.region.x)

    assert await measure(False) == await measure(True), size


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
    """Block until the history page worker has settled.

    The page is read off the loop (``asyncio.to_thread`` inside a
    ``subagent-history`` worker), and the flags the callers assert on
    (``_history_unavailable``, ``_history_error``, ``_history_entries``) are
    written in that worker's completion path. Waiting on the WORKER is waiting
    on those writes; the previous 100-cycle poll was spending a frame budget
    and calling it settled when the budget ran out, which is why
    ``test_history_unavailable_and_error_retry_keep_trajectory_fallback``
    failed under xdist as ``assert 'history unavailable' in 'read-only'`` —
    the state it asserted had simply not been written yet.

    The selection is checked rather than passed because Textual's
    ``wait_for_complete`` treats an empty list as falsy and falls back to
    waiting on EVERY worker. Empty here legitimately means the read already
    finished, so the pause below is enough to let its callback land.
    """
    workers = [w for w in view.workers if w.group == "subagent-history"]
    if workers:
        await view.workers.wait_for_complete(workers)
    await pilot.pause()


async def _wait_landing_settled(
    pilot: Any, view: SubagentView, *, cycles: int = 4, limit: int = 200
) -> None:
    """Block until the initial landing has been DECIDED, then settled.

    The landing is the end of a deferred chain, not a state that exists when
    the history worker returns: ``_settle_initial_landing`` re-arms the
    one-shot inside a ``call_after_refresh``, ``_reconcile_current_body``
    schedules ``_schedule_landing_snap`` through another, and that schedules
    ``_snap_landing_to_row_head`` through a third. Until the last of those
    runs, the body is wherever sticky-tail following left it — which on a
    short viewport is the raw tail, three rows into the wrapping notice.

    A fixed pause budget is a bet that the chain drains inside it. The bet
    holds on an idle box (the snap lands before the first pause here) and
    loses on a contended CI runner, where it failed four times with the
    identical signature ``offset=28, owner_top=25, max=28`` — offset equal to
    ``max_scroll_y``, i.e. the untouched tail, the snap never having run.
    That reproduces exactly by forcing the state the numbers describe
    (``_tail_anchor.acquire()`` then ``_scroll_to_tail()``).

    So wait on the two observables the assertions actually depend on: the
    one-shot being spent (``_landing_snap_pending`` false) and the geometry
    then holding still, since the snap can be followed by a further layout
    pass.

    ``_snap_landing_to_row_head`` does NOT clear the flag on every path out
    of it — it returns with the one-shot still armed when the body is not yet
    mounted or laid out, and when the history page is still pending. Those
    are exactly the states this helper must keep waiting through, so the
    flag's meaning here is "the landing has been decided", not "the snap
    function has run". The ceiling is therefore a deadlock guard rather than
    a timing assumption: a slow machine costs nothing, and a landing that
    never resolves fails naming both flags instead of hanging.
    """
    body = view._body
    stable = 0
    last: tuple[int, float] | None = None
    for _ in range(limit):
        await pilot.pause()
        current = (body.virtual_size.height, body.scroll_y)
        decided = not view._landing_snap_pending and not view._initial_tail_pending
        # BOTH, and on the same frame. The flag alone is not enough: the snap
        # is called several times (measured: three on this fixture, the first
        # two returning early while the body is still short), and the call
        # that runs against a still-growing layout can find the current offset
        # already sitting on a head, take ``target == offset``, and clear the
        # one-shot without scrolling or releasing the anchor. A later pass
        # then grows the extent, sticky-follow moves to the new tail, and the
        # landing is a wrap fragment again with the flag long since spent —
        # observed here as the precondition failing at ``offset=28,
        # owner_top=25``, the same numbers CI reported.
        #
        # Requiring the extent to have held still for ``cycles`` frames while
        # the flag is clear is what makes "decided" mean decided against the
        # FINAL layout rather than against whichever one the one-shot met.
        stable = stable + 1 if decided and current == last else 0
        last = current
        if stable >= cycles:
            return
    raise AssertionError(
        "the landing never settled: "
        f"snap_pending={view._landing_snap_pending}, "
        f"tail_pending={view._initial_tail_pending}, "
        f"virtual_height={body.virtual_size.height}, scroll_y={body.scroll_y}"
    )


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
        # Exhausting the history and REPAINTING the hint are separate frames:
        # `_wait_history` waits on the page worker, and the hint's text is
        # recomputed on a later refresh, so a slow runner reads the previous
        # "loading earlier…" caption (CI shard 3, 3.12). Poll for the settled
        # caption rather than asserting on whichever frame arrives first.
        settled = f"{HISTORY_START_NOTE} · {READ_ONLY_NOTE}"
        for _ in range(50):
            if view._state_hint.rendered().endswith(settled):
                break
            await pilot.pause()
        page = " ".join(view.rendered_rows())
        assert "durable 0" in page
        assert HISTORY_START_NOTE in page
        assert len(view._history_ids) == 230
        assert view._state_hint.rendered().endswith(settled)
        assert view._state_hint.region.width >= len(settled)


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
async def test_history_arriving_at_the_top_loads_one_page_then_stops(tmp_path) -> None:
    """Scrolling to the top loads history per ARRIVAL — never a cascade.

    The pre-fix trigger was LEVEL-triggered on ``scroll_y <= 1`` with no
    programmatic-scroll guard, so it loaded on every watch firing while the
    reader sat at the top: the anchor restore a prepend performs, the settle
    frames after it, and a wheel held against the clamped top. The trigger
    must be an EDGE, and the property that makes it one is that NO page may
    load without the reader having actually travelled away from the top rows
    since the previous page — the operator's complaint was loading "without
    requiring me to scroll up to the top again on the new height".

    Driven with real wheel events as a CONTINUED drag that does not stop at
    the first mount (review round 1, M1: stopping there ends the gesture
    exactly where a cascade would begin, so that shape passed pre-fix). Every
    disk read is audited against the offset's peak since the previous read:
    a read with no intervening travel is the bug.

    Honest scope note: on THIS view a page is 100 rows against a ~28-row
    viewport, so a continued drag displaces the reader a full page between
    arrivals and the no-travel violation is frame-timing-dependent — the
    deterministic pin for it is the latch state machine in
    ``test_page_back_latch.py`` (which fails 3/3 on the pre-fix tree). This
    test holds the same contract end-to-end: no no-travel read survives a
    long drag, nothing loads once the gesture is over, and Home still buys
    exactly one page.
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(320):
        await transcript.append_message(Message.assistant(f"durable {index}"))
    job = _job_with([], status="completed")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    # The DISK READ is the unit counted (not `_history_ids`): a duplicate
    # request inside one gesture is deduped by `_history_loading` and never
    # lands, so the ID set cannot see it. Peak tracking supplies the travel
    # audit: the highest offset observed since the previous read is how far
    # the reader actually went between pages.
    # The verdict is evaluated AFTER the drag, not at read time: a page that
    # mounts displaces the reader DOWN only once its insert settles, which can
    # land a frame after the read that caused it. Judging "no travel" at read
    # time therefore called a legitimate mount-displacement "no travel" — a
    # measurement race, not a cascade (verified by logging: y=101 was observed
    # between two reads the audit had flagged). Each read records the peak
    # since the PREVIOUS read, extended forward to the next observation, and
    # the assertion judges the completed timeline.
    reads = {"n": 0, "first": True, "peaks": [], "peak": 0.0}
    original_read = subagent_view.read_transcript_page
    real_scroll = SubagentView._scroll_changed

    def counting_read(*args: Any, **kwargs: Any) -> Any:
        if not reads["first"]:
            reads["peaks"].append(reads["peak"])
        reads["first"] = False
        reads["n"] += 1
        reads["peak"] = 0.0
        return original_read(*args, **kwargs)

    def auditing_scroll(self: SubagentView, *args: Any) -> None:
        reads["peak"] = max(reads["peak"], self._body.scroll_y)
        real_scroll(self, *args)

    # BOTH patches are restored in the finally below. A module-attribute
    # swap that leaks contaminates every later test in the same xdist worker:
    # `read_transcript_page` is module state, not instance state, so a leaked
    # counter keeps wrapping (and keeps mutating the shared `reads` dict) for
    # the rest of the worker's lifetime (review round 2, M5).
    subagent_view.read_transcript_page = counting_read
    SubagentView._scroll_changed = auditing_scroll  # type: ignore[method-assign]
    try:
        async with app.run_test(size=(90, 28)) as pilot:
            view = await _open(pilot, app, job)
            await _wait_history(pilot, view)
            reads["n"] = 0  # the opening page's read is not the gesture's
            reads["peak"] = 0.0

            # A CONTINUED drag in the rhythm a trackpad actually delivers:
            # several notches per frame, running straight through every mount
            # it causes — the shape that cascaded pre-fix (a level trigger
            # mounted a page per clamped notch once the reader reached the
            # top). No early stop: stopping at the first mount ends the
            # gesture exactly where the cascade would begin.
            for _ in range(40):
                for _ in range(5):
                    view._body.post_message(
                        MouseScrollUp(
                            widget=view._body,
                            button=0,
                            shift=False,
                            meta=False,
                            ctrl=False,
                            x=10,
                            y=10,
                            delta_x=0,
                            delta_y=-2,
                        )
                    )
                reads["peak"] = max(reads["peak"], view._body.scroll_y)
                await pilot.pause()
            # ...and then HOLD the wheel against the clamped top: notches keep
            # arriving while the offset is pinned at 0 and cannot move. A
            # clamped notch is not travel and must not earn a page — this is
            # the exact half of the reported loop where chunks loaded "one
            # after another" with the reader parked at the top.
            for _ in range(60):
                for _ in range(5):
                    view._body.post_message(
                        MouseScrollUp(
                            widget=view._body,
                            button=0,
                            shift=False,
                            meta=False,
                            ctrl=False,
                            x=10,
                            y=10,
                            delta_x=0,
                            delta_y=-2,
                        )
                    )
                # The audit must sample the hold too: a page mounted during
                # the hold displaces the reader DOWN, and that displacement is
                # exactly the travel that legitimises the next page. Sampling
                # only the burst made a mount-during-hold look like no travel
                # at all — a measurement gap, not a cascade.
                reads["peak"] = max(reads["peak"], view._body.scroll_y)
                await pilot.pause()
            await _wait_history(pilot, view)
            await _wait_geometry_settled(pilot, view._body)
            # The interval of the LAST read extends forward to here: its
            # mount's displacement is observed by the settle, not by the drag.
            reads["peaks"].append(reads["peak"])

            # The contract: pages may load during a long drag (each genuine
            # re-arrival at the top earns one), but NEVER without the reader
            # having travelled away from the top since the previous page.
            # `peaks[i]` is the highest offset observed between read i and
            # read i+1 (the first entry is the pre-first-read interval).
            no_travel = sum(1 for peak in reads["peaks"] if peak <= 1)
            assert no_travel == 0, (
                f"{no_travel} of {reads['n']} pages loaded with no travel "
                "away from the top rows — the level-trigger cascade; "
                f"peaks={reads['peaks']}"
            )
            assert reads["n"] >= 1, "the drag reached the top and loaded a page"

            # Once the gesture is OVER, parked wherever it left them, no
            # further page loads however long the view idles.
            final_ids = len(view._history_ids)
            for _ in range(80):
                await pilot.pause()
            assert len(view._history_ids) == final_ids

            # A deliberate discrete act — the advertised Home key — still
            # buys exactly one more page (the pinned per-press contract).
            if not view._history_exhausted:
                before = len(view._history_ids)
                view.action_home()
                await _wait_history(pilot, view)
                await _wait_geometry_settled(pilot, view._body)
                for _ in range(10):
                    await pilot.pause()
                assert len(view._history_ids) > before
    finally:
        SubagentView._scroll_changed = real_scroll  # type: ignore[method-assign]
        subagent_view.read_transcript_page = original_read


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


def _steer_envelope(body: str) -> str:
    """The model-facing wrapper ``SubagentComms._format_to_child`` builds for a
    steer, restated here so the view tests pin the render contract against the
    exact shape a persisted steer row carries."""
    return (
        "<parent-message>\n"
        "This changes your instructions. Apply it from now on, and drop work it "
        "makes pointless.\n\n"
        f"{body}\n"
        "</parent-message>"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("correlated", [True, False])
async def test_a_persisted_steer_never_renders_the_model_facing_envelope(
    tmp_path, correlated: bool
) -> None:
    """A hub steer persists as a plain user Message carrying the model-facing
    ``<parent-message>`` XML. The human-facing fact supersedes it, so the page
    shows exactly one ``Parent · redirected`` row and never the XML.

    ``correlated`` covers both transcript generations: a steer persisted after
    the id fix carries its fact's id (the primary correlation), while a LEGACY
    steer carries a fresh id that can only match by body text. Both must render
    one fact row and no envelope.
    """
    transcript = Transcript(tmp_path / "child")
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {
            "direction": "to_child",
            "body": "Focus on retries",
            "kind": "steer",
            "communication_id": "s1",
        },
    )
    steer_id = "s1" if correlated else "legacy-uuid"
    await transcript.append_message(Message.user(_steer_envelope("Focus on retries"), id=steer_id))

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

        redirected = [
            entry.text
            for entry in view._history_entries
            if entry.text == "Parent · redirected\nFocus on retries"
        ]
        assert redirected == ["Parent · redirected\nFocus on retries"]
        assert "<parent-message>" not in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_two_identical_legacy_steers_render_two_rows(tmp_path) -> None:
    """Steering the same words twice must render TWO rows.

    The legacy (body-text) correlation arm used a set with a membership-only
    check, so a second steer of identical text matched the FIRST steer's fact
    and was dropped: two redirections delivered, one shown. The fact multiset
    is consumed one count per matched row, which keeps the rendered count equal
    to the delivered count. Legacy ids throughout, since that is the only
    generation where body text does the matching.
    """
    transcript = Transcript(tmp_path / "child")
    # ONE fact for TWO envelope rows: the page-boundary shape the fallback arm
    # exists for (the second steer's fact sits on an unloaded page). This is
    # what exposes the collapse — with a fact per row the set and the multiset
    # agree, because the fact rows alone already make up the count.
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {
            "direction": "to_child",
            "body": "Focus on retries",
            "kind": "steer",
            "communication_id": "s-unmatched",
        },
    )
    for index in range(2):
        await transcript.append_message(
            Message.user(_steer_envelope("Focus on retries"), id=f"legacy-{index}")
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

        redirected = [
            entry.text
            for entry in view._history_entries
            if entry.text == "Parent · redirected\nFocus on retries"
        ]
        assert redirected == [
            "Parent · redirected\nFocus on retries",
            "Parent · redirected\nFocus on retries",
        ]
        assert "<parent-message>" not in " ".join(view.rendered_rows())


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_first", [True, False])
async def test_a_mixed_vintage_pair_of_identical_steers_renders_two_rows(
    tmp_path, legacy_first: bool
) -> None:
    """A child steered identically BEFORE and AFTER the id fix shows two rows.

    ``communication_bodies`` is a budget of facts available to supersede a row,
    and a fact must be spent at most once and only by ITS OWN row. Having both
    correlation arms decrement that budget during the ordered walk broke the
    second half of that invariant: whichever row the walk reached first spent
    the other's fact. So the fix is order-dependent unless the id matches are
    resolved in the pre-pass, which is why this case is parametrised on the row
    order rather than seeded in one.

    ``legacy_first`` is the ordering that actually occurs on disk and the one
    that stayed broken after the first attempt: the legacy row is OLDER — it
    was persisted before steers carried their fact's id — so it is reached
    first, consumes the id-correlated row's fact by body text, and the id row is
    then suppressed on identity anyway. Two redirections delivered, one
    rendered. Durable page boundaries are arbitrary, so neither order may be
    assumed.

    This is the mixed-vintage shape specifically: one fact carries a
    ``communication_id`` matching its own row, while the legacy row's own fact
    sits on an unloaded page (the page boundary the fallback arm exists for).
    """
    transcript = Transcript(tmp_path / "child")
    await transcript.append_custom(
        HUB_COMMUNICATION_CUSTOM_TYPE,
        {
            "direction": "to_child",
            "body": "Focus on retries",
            "kind": "steer",
            "communication_id": "m1",
        },
    )
    # Post-upgrade: the envelope carries its fact's id, so it correlates by id.
    # Pre-upgrade: same words, unrelated id, and no fact of its own in the
    # window. It must fall through to the labelled fallback, not be eaten by
    # the id-matched row's fact.
    order = ["legacy", "m1"] if legacy_first else ["m1", "legacy"]
    for message_id in order:
        await transcript.append_message(
            Message.user(_steer_envelope("Focus on retries"), id=message_id)
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

        redirected = [
            entry.text
            for entry in view._history_entries
            if entry.text == "Parent · redirected\nFocus on retries"
        ]
        assert redirected == [
            "Parent · redirected\nFocus on retries",
            "Parent · redirected\nFocus on retries",
        ]
        assert "<parent-message>" not in " ".join(view.rendered_rows())


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_first", [True, False])
async def test_two_facts_for_an_id_row_and_a_legacy_row_render_two_rows(
    tmp_path, legacy_first: bool
) -> None:
    """Each of two identical steers has its OWN fact loaded: still two rows.

    This is the case that rules out the cheap repair of finding 11 — withholding
    from the legacy budget every fact whose ``communication_id`` names a loaded
    row. That fixes the legacy-first pair above but over-withholds here: the id
    row's fact is correctly held back, the legacy row then finds nothing left to
    spend even though its own fact IS loaded, and it renders a third row beside
    the two facts. Withholding has to be per-fact — one fact reserved for the
    one row that claims it — not per-body, which is only visible when the count
    of facts and the count of claiming rows differ.
    """
    transcript = Transcript(tmp_path / "child")
    for communication_id in ("b1", "b2"):
        await transcript.append_custom(
            HUB_COMMUNICATION_CUSTOM_TYPE,
            {
                "direction": "to_child",
                "body": "Focus on retries",
                "kind": "steer",
                "communication_id": communication_id,
            },
        )
    # Only `b1` has a row carrying its id; `b2`'s row predates the id fix, so
    # `b2` is the fact the body-text arm must be left to spend on it.
    order = ["legacy", "b1"] if legacy_first else ["b1", "legacy"]
    for message_id in order:
        await transcript.append_message(
            Message.user(_steer_envelope("Focus on retries"), id=message_id)
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

        redirected = [
            entry.text
            for entry in view._history_entries
            if entry.text == "Parent · redirected\nFocus on retries"
        ]
        assert redirected == [
            "Parent · redirected\nFocus on retries",
            "Parent · redirected\nFocus on retries",
        ]
        assert "<parent-message>" not in " ".join(view.rendered_rows())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("steer", "expects_reply", "label"),
    [
        (True, False, "Parent · redirected"),
        (False, True, "Parent · asked"),
        (False, False, "Parent"),
    ],
)
async def test_the_no_fact_fallback_labels_an_envelope_by_its_own_kind(
    tmp_path, steer: bool, expects_reply: bool, label: str
) -> None:
    """The fallback arm — an envelope row whose communication fact is NOT in
    the loaded window (it may sit on an unloaded page) — renders the extracted
    body under the label matching the envelope's OWN kind.

    It used to hardcode ``Parent · redirected`` for all three, so a note or a
    question would have been reported as a redirection the parent never made.
    The instruction line names the kind, so the fallback has no need to guess.
    This is also the only direct coverage of the arm itself: the correlated and
    legacy-with-fact cases both take the suppression path instead.
    """
    transcript = Transcript(tmp_path / "child")
    envelope = SubagentComms._format_to_child(
        "Focus on retries", expects_reply=expects_reply, steer=steer
    )
    await transcript.append_message(Message.user(envelope, id="lonely-envelope"))

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

        parents = [entry.text for entry in view._history_entries if entry.kind == "parent_message"]
        assert parents == [f"{label}\nFocus on retries"]
        assert "<parent-message>" not in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_a_user_quoting_the_envelope_keeps_their_own_words(tmp_path) -> None:
    """A human asking about the wrapper is NOT a parent communication.

    Extraction requires the exact instruction preamble the builder emits, so a
    user who pastes the tag shape into their own message keeps their words and
    their ``user`` row rather than having them re-rendered as something the
    parent said.
    """
    transcript = Transcript(tmp_path / "child")
    quoted = "<parent-message>\nwhy does my log show this?\n\nsecret plan\n</parent-message>"
    await transcript.append_message(Message.user(quoted, id="human-1"))

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

        rows = [(entry.kind, entry.text) for entry in view._history_entries]
        assert ("user", quoted) in rows
        assert not [row for row in rows if row[0] == "parent_message"]


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
        # Opening the page kicks off the first history read, and the missing
        # directory is discovered on that worker — so the note is not painted
        # until it settles. Asserting straight after _open read whatever state
        # happened to be current one frame in.
        await _wait_history(pilot, view)
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()
        assert "Reading the ingest path." in " ".join(view.rendered_rows())

        child_dir.mkdir()
        # No manual clear of ``_history_unavailable`` here any more. It used to
        # be needed because that flag was a load gate nothing in the product
        # ever cleared, so the Home retry below could not have run — the crutch
        # was itself the evidence for the latch this file now pins. ``Home`` is
        # an explicit reader gesture and is admitted regardless of the note.
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
        # The advertised retry must remain a user decision rather than an
        # implicit hot loop. After the worker has settled, a handful of extra
        # loop turns is enough for a hot-looping retry to have fired; a 250 ms
        # wall-clock window was the previous discriminator and is what went
        # red under swap thrashing (#403).
        for _ in range(8):
            await pilot.pause()
        assert attempts == failed_attempts == 1

        view.focus_body()
        await pilot.pause()
        assert app.focused is view._body
        await pilot.press("home")
        # Wait on the retry actually happening (``attempts`` advancing), not
        # on a 500 ms budget and not on a worker that may not have been
        # spawned yet. Bounded by loop turns: contention stretches how long
        # a turn takes, not how many the key-press needs.
        for _ in range(200):
            if attempts == 2:
                break
            await pilot.pause()
        else:
            raise AssertionError("the explicit Home retry never ran")
        await _wait_history(pilot, view)
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
async def test_the_title_tracks_failed_attempts_without_mislabeling_in_flight_calls() -> None:
    """Attempts stay counted while outcomes settle from the folded tool row.

    A start is not a failure, an error adds the compact warning immediately,
    and a later success remains an attempt without inflating that warning.
    """
    job = _Job("sub-1", "RetryBudgetScout", status="running")
    job.trajectory = [_call("c1", "edit", path="one.py")]
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        title = view.rendered_rows()[0]
        assert "1 tool" in title
        assert "failed" not in title

        job.trajectory.append(_result("c1", "edit", "invalid patch", is_error=True))
        app._refresh_band()
        await pilot.pause()
        title = view.rendered_rows()[0]
        assert "1 tool · 1 failed" in title

        job.trajectory.extend(
            [_call("c2", "read", path="two.py"), _result("c2", "read", "contents")]
        )
        app._refresh_band()
        await pilot.pause()
        title = view.rendered_rows()[0]
        assert "2 tools · 1 failed" in title


@pytest.mark.asyncio
async def test_failed_tool_summary_drops_as_one_truthful_field() -> None:
    """The ladder keeps both counts whole or drops the outcome field entirely."""
    job = _Job("sub-1", "RetryBudgetScout", status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        view = await _open(pilot, app, job)
        rows = {
            width: view._title_row(width, "⣾", tools=2, failed_tools=2).plain
            for width in range(20, 121)
        }
        assert any("2 tools · 2 failed" in row for row in rows.values())
        assert any("failed" not in row for row in rows.values())
        for width, row in rows.items():
            assert cell_len(row) <= width, (width, row)
            if "failed" in row:
                assert "2 tools · 2 failed" in row, (width, row)


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
    """A history prepend on a truncated child keeps durable rows in order.

    The truncation note is page chrome (``_truncation``), not a body block,
    so `prefix` and the body's index agree — there is no head-offset to
    apply. The previous write-up described the note occupying body index 0
    while being excluded from `_entries`; pinning it out of the column
    removed that skew (review round 1, N1).
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
        assert view._truncation.display, "the truncation note is page chrome, not a body row"

        before = len(view._body.blocks())
        view.action_home()
        await _wait_history(pilot, view)
        await _wait_geometry_settled(pilot, view._body)
        assert len(view._body.blocks()) > before, "no history page was prepended"

        # The note is still page chrome (not a body row), and the prepended
        # history did not displace it or land above the first durable row.
        assert view._truncation.display
        assert view._body.blocks()[0] is not view._head_block


@pytest.mark.asyncio
async def test_a_history_page_lands_in_order_when_the_cap_is_crossed_mid_read(tmp_path) -> None:
    """Crossing the cap mid-read still prepends history in order.

    The truncation note is now pinned at the head the moment it appears
    (issue #407), so a child that crosses the cap while the page is open
    no longer mounts the note mid-list. The prepend offset still keys on
    the note's POSITION, and this case is the one that used to break when
    that position was mid-list and the offset assumed it was not
    (`durable 40` above `durable 0`, review round 2, M3).
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

        # Now cross the cap under the reader. The note is PINNED at the
        # head (issue #407) rather than appended mid-list, so it occupies
        # index 0 the moment it exists — the same position a truncated-at-
        # open child already had. The prepend offset still keys on that
        # position, which is why this case used to break when the note
        # sat mid-list and the offset assumed it did not.
        events: list[dict[str, Any]] = []
        while len(events) < TRAJECTORY_MAX_EVENTS + 10:
            events.extend(_text(f"m{len(events)}", f"live {len(events)}"))
        job.trajectory = events
        app._refresh_subagent_view()
        for _ in range(4):
            await pilot.pause()
        assert view._head_block is not None, "the fixture needs to cross the cap"
        assert view._truncation.display, "crossing the cap must pin the note as chrome"

        view.action_home()
        await _wait_history(pilot, view)
        await _wait_geometry_settled(pilot, view._body)

        # The durable window stays in order: the page that was paged in must
        # not be split around the row it belongs above.
        durable = [row for row in view.rendered_rows() if row.strip().startswith("durable ")]
        ordering = [int(row.strip().split()[1]) for row in durable]
        assert ordering == sorted(ordering), f"the prepended page landed out of order: {ordering}"


def _long_error_notice() -> dict[str, Any]:
    """An error that wraps past a 62x24 body's viewport.

    A two-line wrap still fits above the working line on a 9-row body, so
    sticky-tail following never bisects it. The glyph only sits above the
    fold when the notice itself is taller than the viewport-minus-tail,
    which is the D3 frame: first visible line is a hanging continuation.
    """
    return {
        "type": "notice",
        "kind": "error",
        "text": (
            "Invalid arguments: argument 'edits' does not match type array "
            "while calling edit on local_operator/tui/widgets/subagent_view.py "
            "with a payload that also failed validation on every subsequent "
            "field of the same call: path, old_text, new_text, replace_all, "
            "and the trailing context the child included to justify the edit"
        ),
    }


@pytest.mark.asyncio
async def test_the_truncation_note_stays_in_the_viewport_when_the_body_scrolls() -> None:
    """The note is page chrome, not a transcript row.

    Mounted in the body it left the screen the moment content exceeded the
    viewport — which is exactly when it is explaining why the run starts
    mid-sentence (issue #407). Mutation: put it back in the body as an
    ordinary head block and ``TRUNCATION_NOTE`` drops out of the visible
    chrome once the body is scrolled to its tail.
    """
    events: list[dict[str, Any]] = []
    while len(events) < TRAJECTORY_MAX_EVENTS:
        events.extend(_text(f"m{len(events)}", f"step {len(events)} applying the review fixes"))
    events.append(_long_error_notice())
    job = _job_with(events, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(62, 24)) as pilot:
        view = await _open(pilot, app, job)
        for _ in range(8):
            await pilot.pause()
        assert view._head_block is not None, "the fixture needs a truncated trajectory"
        assert view._truncation.display
        # Follow the tail the way a live page does. The note is a sibling
        # of the body, so its region.y cannot follow the body's scroll.
        view._body.scroll_end(animate=False)
        await _wait_geometry_settled(pilot, view._body)
        assert view._truncation.display
        assert view._truncation.region.y < view._body.region.y, (
            f"truncation chrome is not above the body: "
            f"note.y={view._truncation.region.y} body.y={view._body.region.y}"
        )
        assert TRUNCATION_NOTE in " ".join(view.rendered_rows())
        # And it is NOT in the scrolling column, so a Home/End cycle
        # cannot hide it.
        assert view._head_block not in view._body.blocks()
        # One CONTENT row at 62 columns: the previous wording wrapped so
        # `kept` sat alone on line 2 (design round 1, D2). Region height
        # is 2 because the stylesheet pads one row above the chrome;
        # wrapping is a newline in the renderable.
        painted = str(view._truncation.renderable)
        assert "\n" not in painted, (
            f"truncation note wrapped at width {view._truncation.region.width}: " f"{painted!r}"
        )


def _landing_owner_top(view: SubagentView) -> tuple[float, float | None]:
    """Scroll offset and the top of the block that currently owns it."""
    body = view._body
    offset = body.scroll_offset.y
    owner_top = None
    for block in body.blocks():
        if block is view._head_block:
            continue
        top = block.virtual_region.y
        if top <= offset < block.virtual_region.bottom:
            owner_top = top
            break
    return offset, owner_top


@pytest.mark.asyncio
async def test_a_narrow_viewport_opens_on_a_row_head_not_a_wrap_fragment(
    tmp_path,
) -> None:
    """At 62x24 the first visible transcript line must be a row HEAD.

    Sticky-tail following can bisect a wrapping notice so the glyph sits
    above the fold and the first glance is a hanging-indented continuation
    in the notice's own red (issue #407). The wrap is correct; the landing
    is not.

    A real child always has a transcript directory. The comms-less fixture
    this used to drive snapped the trajectory-only body, then the initial
    history page grew the extent and sticky-follow put the fragment back
    (review round 1, F1). Mutation: keep the one-shot burning on the
    pre-history body and this assertion dies with ``into > 0``.
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(20):
        await transcript.append_message(
            Message.assistant(f"durable {index}: applying the review fixes to the widget.")
        )
    events: list[dict[str, Any]] = []
    # Two short live rows above the wrapping notice: enough that sticky-tail
    # following overflows, not so many that the notice sits entirely
    # below the fold. The notice itself is taller than the 9-row body,
    # which is what puts a continuation line at the top of the viewport
    # before the snap.
    for index in range(2):
        events.extend(_text(f"m{index}", f"Step {index}: applying the review fixes to the widget."))
    events.append(_call("c1", "edit", path="local_operator/tui/widgets/subagent_view.py"))
    events.append(
        _result(
            "c1",
            "edit",
            "Invalid arguments: argument 'edits' does not match type array",
            is_error=True,
        )
    )
    events.append(_long_error_notice())
    job = _job_with(events, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(62, 24)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        await _wait_landing_settled(pilot, view)
        offset, owner_top = _landing_owner_top(view)
        assert owner_top is not None, (
            f"no block owns the landing offset {offset}; "
            f"starts={[b.virtual_region.y for b in view._body.blocks()[:12]]}"
        )
        assert owner_top == offset, (
            f"landing sits {offset - owner_top} rows into a block "
            f"(offset={offset}, owner_top={owner_top}, max={view._body.max_scroll_y})"
        )
        # Not just ANY row head — the wrapping error is the first glance
        # a tail-following open is trying to show. A one-shot spent on the
        # pre-history body freezes the offset on an earlier durable row
        # that also happens to be a head, which is how the comms-less
        # fixture hid F1.
        notice = next(
            block
            for block in view._body.blocks()
            if isinstance(block, NoticeBlock) and "Invalid arguments" in block.text()
        )
        assert notice.virtual_region.y == offset, (
            f"landed on {type(view._body.blocks()[0]).__name__} at {offset}, "
            f"not the wrapping notice at {notice.virtual_region.y}"
        )


@pytest.mark.asyncio
async def test_the_landing_survives_the_next_extent_change(tmp_path) -> None:
    """A row that arrives after the landing must not drag it back onto a wrap.

    The snap pulls the offset back to a row head, which leaves the viewport
    ABOVE the tail — so sticky-tail following has to be released, or the very
    next extent change calls ``_scroll_to_tail`` and puts the fragment straight
    back. ``_snap_landing_to_row_head`` does release it, and the comment there
    names this exact regression (F1, review round 1), but nothing exercised it:
    removing the ``_tail_anchor.release()`` call leaves the whole file green on
    ``main``, because the sibling test above asserts the landing and then stops
    looking.

    This is that missing half. It opens the same 62x24 page, waits for the same
    landing, then appends one live row — the ordinary thing a running child
    does — and requires the first visible line to still be a row head.

    Mutation: delete the ``release()`` call and this fails with the offset
    three rows inside the notice, which is the shape issue #407 reported.
    """
    transcript = Transcript(tmp_path / "child")
    for index in range(20):
        await transcript.append_message(
            Message.assistant(f"durable {index}: applying the review fixes to the widget.")
        )
    events: list[dict[str, Any]] = []
    for index in range(2):
        events.extend(_text(f"m{index}", f"Step {index}: applying the review fixes to the widget."))
    events.append(_call("c1", "edit", path="local_operator/tui/widgets/subagent_view.py"))
    events.append(
        _result(
            "c1",
            "edit",
            "Invalid arguments: argument 'edits' does not match type array",
            is_error=True,
        )
    )
    events.append(_long_error_notice())
    job = _job_with(events, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: transcript.directory}
    )()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(62, 24)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        await _wait_landing_settled(pilot, view)

        landed, landed_top = _landing_owner_top(view)
        if landed_top != landed:
            # The landing itself did not come out on a row head. That is the
            # SIBLING test's subject — it asserts exactly this and fails with
            # the offending offsets — so failing here too would report one
            # defect twice while making this test's own subject (whether a
            # SETTLED landing survives growth) unreachable. Skipping keeps the
            # two subjects separate: if the landing regresses, the sibling
            # goes red and names it; this one simply has nothing to say.
            pytest.skip(
                f"landing did not settle on a row head (offset={landed}, "
                f"owner_top={landed_top}); that is the sibling test's subject"
            )
        # Following must be off after a snap that has landed above the tail.
        # Asserted as the STATE, not inferred from the offset, so a failure
        # here names the cause rather than its symptom. This is the assertion
        # that kills the F1 mutant, and it does not depend on the timing
        # above: once the landing IS on a head short of the tail, the anchor
        # must be released or the growth below will drag it back.
        assert not view._body._tail_anchor.following, (
            "the landing snap left sticky-tail following armed; the next "
            "extent change will scroll back onto the wrap fragment"
        )

        # One more live row: the ordinary growth a running child produces,
        # published the way the jobs coalescer publishes it rather than by
        # poking the view, so this exercises the real repaint path.
        from local_operator.session.frontend_state import FrontendSessionState, JobState

        job.trajectory.extend(_text("m9", "Step 9: still working through the widget rewrite."))
        app._apply_frontend_state(
            FrontendSessionState(
                session_id="sess",
                epoch="owner",
                jobs=[JobState.from_job(job)],
            )
        )
        await _wait_geometry_settled(pilot, view._body)

        offset, owner_top = _landing_owner_top(view)
        assert owner_top is not None, (
            f"no block owns the offset {offset} after growth; "
            f"starts={[b.virtual_region.y for b in view._body.blocks()[:12]]}"
        )
        assert owner_top == offset, (
            f"growth dragged the viewport {offset - owner_top} rows into a block "
            f"(offset={offset}, owner_top={owner_top}, max={view._body.max_scroll_y})"
        )


@pytest.mark.asyncio
async def test_the_unavailable_note_clears_when_the_transcript_appears(tmp_path) -> None:
    """The launch race, end to end: the note must not outlive the absence.

    ``SubagentComms.attach`` binds ``session_dir`` when the child session is
    constructed, but ``Transcript`` creates ``transcript.jsonl`` on the first
    append, so a page opened in that window reads a valid directory with no
    file in it. The reported symptom was the footer saying "history
    unavailable" under a fully rendered trajectory for the rest of the page's
    life, because the flag that failure set was also the gate on ever loading
    again.

    Driven through the app's own refresh (``_refresh_subagent_view``, the 1 Hz
    poll's entry point) rather than by poking the flag, since the claim is
    that the PRODUCT re-examines the directory.
    """
    job = _job_with(TRAJECTORY, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    child_dir = tmp_path / "child"
    # attach() made the directory; the child's first append has not landed.
    child_dir.mkdir()
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: child_dir}
    )()

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        # Correct at this instant: there genuinely is no transcript yet.
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()

        transcript = Transcript(child_dir)
        await transcript.append_message(Message.assistant("durable row", id="durable-1"))

        app._refresh_subagent_view(view.job_id)
        await _wait_history(pilot, view)

        assert HISTORY_UNAVAILABLE_NOTE not in view._history_state_text()
        assert "durable row" in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_a_child_with_no_directory_never_probes_and_keeps_the_note(tmp_path) -> None:
    """The permanent half: no directory is an absence no refresh revises.

    A child that never started a durable session has nothing to read now and
    nothing to read later, so the note is the truth and the page must not
    spend a disk read per refresh discovering that again — the hot loop the
    module's other retry latches exist to prevent.
    """
    job = _job_with(TRAJECTORY, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    session._subagent_comms = type("Comms", (), {"session_dir_of": lambda self, _job_id: None})()

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()
        assert view._history_absent_final

        probes = 0
        real_probe = view._transcript_file_exists

        def counted() -> bool:
            nonlocal probes
            probes += 1
            return real_probe()

        view._transcript_file_exists = counted  # type: ignore[method-assign]
        for _ in range(5):
            app._refresh_subagent_view(view.job_id)
            await pilot.pause()

        assert probes == 0, "a child with no directory must not be probed at all"
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()


@pytest.mark.asyncio
async def test_a_missing_transcript_costs_one_stat_per_refresh_and_no_read(
    tmp_path, monkeypatch
) -> None:
    """The transient half must not become a retry storm.

    A directory that exists and stays empty is re-examined on every refresh —
    that is what makes the note self-correcting — but the re-examination is a
    ``stat``, and the expensive page read is only requested once the cheap
    answer says there is something to read. A swept child whose transcript was
    genuinely deleted sits here indefinitely, so this is the case that decides
    whether "self-correcting" costs anything.
    """
    job = _job_with(TRAJECTORY, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: child_dir}
    )()

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()

        reads = 0
        real_read = subagent_view.read_transcript_page

        def counted_read(*args, **kwargs):
            nonlocal reads
            reads += 1
            return real_read(*args, **kwargs)

        monkeypatch.setattr(subagent_view, "read_transcript_page", counted_read)
        for _ in range(6):
            app._refresh_subagent_view(view.job_id)
            await pilot.pause()
        await _wait_history(pilot, view)
        assert reads == 0, "an empty directory must not be re-read from disk each refresh"
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()

        # The moment the file appears, the SAME refresh path picks it up, and
        # exactly one read pays for it.
        transcript = Transcript(child_dir)
        await transcript.append_message(Message.assistant("late row", id="late-1"))
        app._refresh_subagent_view(view.job_id)
        await _wait_history(pilot, view)

        assert reads == 1
        assert HISTORY_UNAVAILABLE_NOTE not in view._history_state_text()
        assert "late row" in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_a_swept_child_with_a_deleted_transcript_reads_honestly(tmp_path) -> None:
    """``gone`` semantics must survive the re-look.

    Retention sweeps a settled child five minutes after it finishes, and its
    session directory can be removed with it. The page then has no durable
    rows to offer and says so — while ``LEDGER_GONE_NOTE`` continues to
    terminate the body, because the rows above it were true when captured.
    """
    job = _job_with(TRAJECTORY, status="gone")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    child_dir = tmp_path / "swept"
    child_dir.mkdir()
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: child_dir}
    )()

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)

        # The directory outlived the file, which is the sweep this covers.
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()
        for _ in range(3):
            app._refresh_subagent_view(view.job_id)
            await pilot.pause()
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()
        assert LEDGER_GONE_NOTE in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_a_transient_probe_error_does_not_disable_the_re_look(tmp_path, monkeypatch) -> None:
    """A background peek that fails must not spend the reader's error latch.

    ``_history_error`` is a one-way gate: ``_maybe_load_history`` admits only
    an explicit ``Home`` past it. That is right for a read the reader ASKED
    for — they are owed the outcome of their gesture — but the re-look added
    for the launch race issues reads nobody requested, so one transient
    ``OSError`` on a speculative probe used to latch the gate on the reader's
    behalf and permanently downgrade a self-correcting page to a manual one,
    silently (review round 1, R2 / QA Q10).

    Pins the whole property, not just the flag: after the failed probe the
    footer still says what it said before, later refreshes still issue reads
    (the page is still looking), and the durable row lands on its own without
    anyone pressing anything.
    """
    job = _job_with(TRAJECTORY, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: child_dir}
    )()

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()

        reads = 0
        failed = 0
        succeeded = 0
        real_read = subagent_view.read_transcript_page

        def flaky(*args, **kwargs):
            nonlocal reads, failed, succeeded
            reads += 1
            # Transient, and on the FIRST probe only — a network home
            # directory hiccuping once is the reachable production shape.
            #
            # Counted by OUTCOME rather than by total, because the total is a
            # scheduling detail: a loaded shard can settle an extra refresh
            # before an assertion is reached. "One probe failed" and "a later
            # read succeeded" are the two facts this test is actually about,
            # and both are stable under load.
            if reads == 1:
                failed += 1
                raise OSError("transient NFS hiccup")
            succeeded += 1
            return real_read(*args, **kwargs)

        monkeypatch.setattr(subagent_view, "read_transcript_page", flaky)

        # The file appearing is what triggers the probe, and that probe fails.
        transcript = Transcript(child_dir)
        await transcript.append_message(Message.assistant("durable row", id="durable-1"))
        app._refresh_subagent_view(view.job_id)
        # Wait on the PROBE HAVING RUN, not on a frame budget. `_wait_history`
        # only awaits workers that ALREADY exist, so a refresh whose worker has
        # not been spawned yet returns having waited on nothing and leaves
        # `reads` at 0 — which is exactly how this went red on a loaded CI
        # shard. Bounded by loop turns, so contention stretches how long a turn
        # takes rather than how many the probe needs.
        for _ in range(200):
            await _wait_history(pilot, view)
            if failed:
                break
        else:
            raise AssertionError("the probe never ran")

        # `>= 1`, not `== 1`: the probe having run and failed is the
        # precondition this test needs, and pinning the exact COUNT pins a
        # scheduling detail instead. A loaded shard can settle two refreshes
        # (the explicit one and the spinner's) before the assertion is
        # reached, which is not a defect in anything this test is about --
        # the count itself is owned by
        # `test_a_persistently_failing_probe_costs_no_extra_reads_per_refresh`,
        # where a ceiling is the actual subject.
        assert failed == 1, "the probe must actually have run and failed"
        # The failed peek concluded nothing, so the previous conclusion stands
        # rather than being replaced by an error the reader never provoked.
        assert HISTORY_ERROR_NOTE not in view._history_state_text()
        assert HISTORY_UNAVAILABLE_NOTE in view._history_state_text()
        assert not view._history_error

        # The point of the fix: the page is STILL looking. No Home, no keypress.
        app._refresh_subagent_view(view.job_id)
        # Same edge as above: the re-look happening IS the fix, so wait on it
        # rather than on a fixed number of frames.
        for _ in range(200):
            await _wait_history(pilot, view)
            if succeeded:
                break
        else:
            raise AssertionError("the re-look never ran after the failed probe")

        # Same reason as above: what this pins is that the re-look HAPPENED
        # after a failed probe, which is the whole point of the fix. One more
        # read than the failed probe proves it; the exact total does not.
        assert succeeded >= 1, "a transient probe failure must not stop the re-look"
        assert HISTORY_UNAVAILABLE_NOTE not in view._history_state_text()
        assert "durable row" in " ".join(view.rendered_rows())


@pytest.mark.asyncio
async def test_a_persistently_failing_probe_costs_no_extra_reads_per_refresh(
    tmp_path, monkeypatch
) -> None:
    """Recovering from a failed probe must not become the retry storm.

    The tempting fix for R2 — admitting ``recheck`` past the error guard —
    also clears the latch, but a read that keeps failing would then cost one
    disk read per refresh forever. Restoring the previous conclusion instead
    keeps the cheap ``stat`` as the gate, so a persistent failure costs one
    read per refresh at most and the note never flaps (QA Q13 measured 0 extra
    reads over 60 refreshes on the reachable permission-denied path).
    """
    job = _job_with(TRAJECTORY, status="running")
    session = FakeSession()
    session.jobs = _fake_jobs(job)
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    session._subagent_comms = type(
        "Comms", (), {"session_dir_of": lambda self, _job_id: child_dir}
    )()

    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        view = await _open(pilot, app, job)
        await _wait_history(pilot, view)

        transcript = Transcript(child_dir)
        await transcript.append_message(Message.assistant("durable row", id="durable-1"))

        reads = 0

        def always_fails(*args, **kwargs):
            nonlocal reads
            reads += 1
            raise OSError("permission denied")

        monkeypatch.setattr(subagent_view, "read_transcript_page", always_fails)

        footers = set()
        for _ in range(20):
            app._refresh_subagent_view(view.job_id)
            await _wait_history(pilot, view)
            footers.add(view._history_state_text())

        # One read per REFRESH is the floor the stat gate cannot lower here:
        # the file genuinely is there, so the cheap probe says "go look" every
        # time and the read is the only way to learn it still fails. Measured
        # at ~1.1 per refresh — the settle that follows an abandoned probe can
        # re-enter the refresh path once — which matches the rate QA measured
        # on the analogous stat/read disagreement and did not file.
        #
        # The storm this excludes is the one an error-guard fix would create:
        # unbounded growth with the EVENT rate rather than the refresh rate.
        assert reads <= 2 * 20, f"probe storm: {reads} reads over 20 refreshes"

        # The bound above is necessary but NOT sufficient: the rejected
        # error-guard fix also satisfies it at this refresh count (review
        # round 2, R5). What actually separates the two is the SHAPE of the
        # growth — the storm rises with the event stream, this does not — so
        # assert on that directly. `show()` is what a relayed event drives;
        # driving it 100x more must not buy 100x the reads, because the
        # in-flight guard serialises probes and only a settled one re-arms.
        reads_after_refresh_phase = reads
        for _ in range(200):
            app._refresh_subagent_view(view.job_id)
        await _wait_history(pilot, view)
        burst_reads = reads - reads_after_refresh_phase
        assert burst_reads <= 20, (
            f"probe cost tracks the EVENT rate, not the refresh rate: "
            f"{burst_reads} reads from 200 back-to-back shows"
        )

        # And the reader sees one stable, truthful note throughout — no blink
        # between the absence and an error nobody asked to hear about.
        assert footers == {f"{HISTORY_UNAVAILABLE_NOTE} · {READ_ONLY_NOTE}"}, footers
