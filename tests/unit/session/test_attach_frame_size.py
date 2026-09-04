"""Attach frames must fit the control socket, however busy the session is.

The runtime's reader drops any line past ``server._MAX_LINE_BYTES`` (1 MiB),
and a job's retained trajectory is bounded in COUNT (``TRAJECTORY_CAP`` = 500
events) but not in BYTES — each event holds a whole tool result. Ten children
at the cap serialize to ~3.1 MB, so before trajectories were taken out of the
snapshot the first frame of a busy session could not be sent at all and the
session simply could not be attached to: 12 of 17 sessions on the reference
machine failed exactly this way.

These are hard size assertions rather than "it worked" assertions, because the
failure they guard is silent — an oversized frame is a dropped line, not an
error anybody reports.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, Usage
from local_operator.session.frontend_state import (
    USAGE_COMPONENT_CAP,
    FrontendSessionState,
    FrontendStateStore,
    FrontendUsage,
    JobState,
    McpServerState,
    TodoItemState,
    TodoPhaseState,
    _folded_components,
    filter_update_trajectories,
    oversized_frame_report,
    sync_wire_payload,
)
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import _MAX_LINE_BYTES, RuntimeServer
from local_operator.tui.costs import job_cost
from tests.unit.session.runtime.test_server import FakeHandle

#: A tool result big enough to be realistic. The point of the cap is that ONE
#: event carries an unbounded payload, so a small filler would test the row
#: count rather than the thing that overflows.
_RESULT_CHARS = 400


def _event(index: int) -> dict[str, Any]:
    return {
        "type": "tool_execution_end",
        "generation": 1,
        "tool_call_id": f"call_{index:06d}",
        "tool_name": "bash",
        "intent": "Checking something moderately descriptive here",
        "result": {"content": [{"type": "text", "text": "x" * _RESULT_CHARS}]},
        "_traj_seq": index,
    }


def _jobs(count: int, rows: int, *, start: int = 0) -> list[JobState]:
    return [
        JobState(
            id=f"job{index}",
            type="task",
            label=f"child {index}",
            status="running",
            trajectory=[_event(row) for row in range(start, start + rows)],
        )
        for index in range(count)
    ]


def _line_bytes(frame: dict[str, Any]) -> int:
    """Exactly what the socket writes: one JSON line plus its delimiter."""
    return len(json.dumps(frame).encode()) + 1


async def _record(root: Path):  # noqa: ANN202
    for _ in range(100):
        rows = registry.scan(root)
        if rows and rows[0][1] == "live":
            return rows[0][0]
        await asyncio.sleep(0.02)
    raise AssertionError("record did not publish")


async def _never():
    raise AssertionError("takeover was not expected")


def test_ten_jobs_at_the_cap_overflow_the_line_limit_without_the_fix() -> None:
    """The regression this guards is real, not hypothetical.

    Asserting the UNFIXED size keeps the other tests meaningful: if a future
    change made trajectories small enough to fit anyway, the fix's own tests
    would pass for the wrong reason and this one would fail loudly instead.
    """
    store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", jobs=_jobs(10, 500))
    )
    subscription = store.subscribe(lambda _update: None)
    naive = _line_bytes({"op": "frontend_sync", "data": subscription.sync.model_dump(mode="json")})
    assert naive > _MAX_LINE_BYTES, (
        "the fixture no longer reproduces the oversized frame; "
        f"{naive} bytes is under the {_MAX_LINE_BYTES} limit"
    )


def test_sync_for_ten_jobs_at_the_cap_fits_the_line_limit() -> None:
    store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", jobs=_jobs(10, 500))
    )
    subscription = store.subscribe(lambda _update: None)
    frame = {"op": "frontend_sync", "data": sync_wire_payload(subscription.sync)}
    assert _line_bytes(frame) < _MAX_LINE_BYTES

    jobs = frame["data"]["snapshot"]["jobs"]
    assert len(jobs) == 10
    # The rows are gone but the COUNT survives, which is what lets the viewer
    # say "loading 500 events" instead of rendering the child as empty.
    assert all(job["trajectory"] == [] for job in jobs)
    assert [job["trajectory_length"] for job in jobs] == [500] * 10


def test_delta_burst_across_unwatched_jobs_fits_the_line_limit() -> None:
    """The snapshot is only half the budget; a mid-turn burst is the other.

    Ten children each appending 200 events in one tick overflows the limit as
    surely as the snapshot did, and a viewer reading ONE child's page must not
    pay for the other nine.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1", jobs=_jobs(10, 0)))
    store.subscribe(lambda _update: None)
    update = store.mutate(jobs=_jobs(10, 200))
    assert update is not None
    payload = update.model_dump(mode="json")

    unfiltered = _line_bytes({"op": "frontend_update", "data": payload})
    assert unfiltered > _MAX_LINE_BYTES, "the burst fixture no longer overflows"

    watching_one = filter_update_trajectories(payload, {"job3"}.__contains__)
    assert _line_bytes({"op": "frontend_update", "data": watching_one}) < _MAX_LINE_BYTES
    assert list(watching_one["job_trajectory_appends"]) == ["job3"]

    watching_none = filter_update_trajectories(payload, lambda _job_id: False)
    assert _line_bytes({"op": "frontend_update", "data": watching_none}) < _MAX_LINE_BYTES
    assert watching_none["job_trajectory_appends"] == {}
    # Row counts still ride the roster, so a page opened later knows what to
    # fetch rather than resuming from a hole.
    assert [job["trajectory_length"] for job in watching_none["changes"]["jobs"]] == [200] * 10


def test_unfiltered_update_is_returned_unchanged_when_nothing_needs_dropping() -> None:
    """The common delta carries no trajectories and must not pay for a copy."""
    payload = {"epoch": "e1", "sequence": 3, "changes": {"streaming": True}}
    assert filter_update_trajectories(payload, lambda _job_id: False) is payload


# ---------------------------------------------------------------------------
# The CLASS, not the next instance.
#
# Trajectories were the first unbounded per-turn list to overflow this frame.
# ``usage_components`` was the second, and it shipped a release in which the
# reference machine's largest sessions could not be attached to at all. The
# tests below are written against the SHAPE — "an attach frame stays under the
# cap however long the conversation ran and however many children it launched"
# — so a third such field fails CI here rather than in a user's terminal.
# ---------------------------------------------------------------------------


#: The serving identity folding keys on. Both fields must match for two
#: receipts to fold together.
_PROVIDER = "anthropic"
_MODEL_ID = "claude-opus-4-8-20260101"


def _priced_spec() -> ModelSpec:
    """A spec the cost table can price, so receipts actually accrue.

    ``accrue_usage`` only appends to ``usage_components`` when it can put a
    number on the call; an unpriceable model records PARTIAL knowledge and
    appends nothing, which would make the cap assertions below pass trivially.
    """
    return ModelSpec(
        provider="anthropic",
        model_id="claude-opus-4-8-20260101",
        display_name="Opus",
        context_window=1_000_000,
        max_output_tokens=64_000,
    )


def _receipt(index: int) -> Usage:
    """One provider receipt, the size the real ones are.

    Real receipts carry the serving identity and a per-call price, which is
    what makes them ~275 bytes each rather than a handful. A tiny filler would
    test the row count instead of the thing that overflows.
    """
    return Usage(
        input_tokens=12_000 + index,
        output_tokens=800 + index,
        cache_read_tokens=9_000,
        cache_write_tokens=1_200,
        context_tokens=180_000 + index,
        usd_cost=0.0123,
        provider="anthropic",
        model_id="claude-opus-4-8-20260101",
    )


def test_a_long_conversations_receipts_are_capped_at_accumulation() -> None:
    """The list must not grow without bound as turns accumulate.

    Asserted on the STORE rather than on a byte size: the property that makes
    the frame fit is that the list stops growing, and a byte assertion would
    pass for the wrong reason the day a receipt gets smaller.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))
    session = SimpleNamespace(effective_model=_priced_spec())
    for index in range(USAGE_COMPONENT_CAP * 3):
        store.accrue_usage(session, _receipt(index))

    components = store.state.usage_components
    assert len(components) == USAGE_COMPONENT_CAP
    # Newest-wins, oldest evicted — the same discipline AsyncJob.trajectory
    # uses. The most recent receipts are the ones a mixed-provider aggregate
    # needs, so keeping the head instead would keep the useless half.
    assert components[-1].input_tokens == 12_000 + (USAGE_COMPONENT_CAP * 3 - 1)


def test_capping_receipts_cannot_move_a_number_the_ui_paints() -> None:
    """The cap is only safe because the painted figures are running totals.

    ``cumulative_parent_cost`` accrues per call and is never re-derived by
    summing ``usage_components``. If that ever changed, this fails and the cap
    has to grow a running aggregate beside the bounded tail.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))
    session = SimpleNamespace(effective_model=_priced_spec())
    for index in range(USAGE_COMPONENT_CAP * 2):
        store.accrue_usage(session, _receipt(index))

    state = store.state
    # Every receipt carried a provider-reported price, so the lifetime cost is
    # the full count's worth even though only the tail survives.
    assert len(state.usage_components) == USAGE_COMPONENT_CAP
    assert state.cumulative_parent_cost == pytest.approx(0.0123 * USAGE_COMPONENT_CAP * 2)
    # Occupancy is a LEVEL, not a sum, and comes from the newest receipt.
    assert state.context_tokens == 180_000 + (USAGE_COMPONENT_CAP * 2 - 1)


def test_a_restored_fat_checkpoint_is_capped_on_the_way_in() -> None:
    """Every transcript written before the cap still carries the fat list.

    This is the case the operator's machine is actually in: 4,910 sessions
    whose newest checkpoint holds thousands of receipts. Without capping on
    RESTORE, the first resume of such a session rebuilds the oversized state in
    memory and re-emits exactly the frame that could not be sent.
    """

    class _Transcript:
        def latest_custom(self, _custom_type: str) -> dict[str, Any]:
            fat = FrontendSessionState(
                session_id="s1",
                epoch="old",
                usage_components=[
                    FrontendUsage.model_validate(_receipt(index).model_dump(mode="json"))
                    for index in range(2_685)
                ],
            )
            return {"state": fat.model_dump(mode="json")}

    session = SimpleNamespace(session_id="s1", _transcript=_Transcript())
    store = FrontendStateStore.from_checkpoint(session)

    assert len(store.state.usage_components) == USAGE_COMPONENT_CAP
    frame = {
        "op": "frontend_sync",
        "data": sync_wire_payload(store.subscribe(lambda _u: None).sync),
    }
    assert _line_bytes(frame) < _MAX_LINE_BYTES


#: Every collection-typed field on ``FrontendSessionState``, with the reason it
#: cannot blow the frame. A field NOT in here fails
#: ``test_every_collection_field_is_classified`` the moment it is added, which is
#: what makes the guard cover fields nobody thought to write a fixture for.
#:
#: "bounded" fields are bounded by something that is not conversation length or
#: child count — config, the code, the user's own hand, or an explicit fold.
_BOUNDED_COLLECTION_FIELDS = {
    "context_breakdown": "one entry per tool; bounded by the tool inventory",
    "child_costs": "one float per job; O(1) bytes each",
    "queued_steering": "drains every turn",
    "live_events": "explicitly bounded by _fold_live_event",
    "todos": "the user's own list, written by hand",
    "wakes": "the user's own schedules",
    "mcp_servers": "one row per configured server",
    "slash_capabilities": "one row per SLASH_COMMANDS entry",
    "model_catalogue": "one row per catalogue model; bounded by the provider",
    # One startup report from the MCP wiring pass, not an accumulator: it is
    # REPLACED on each wiring round rather than appended to, and its size is a
    # function of how many servers are configured.
    "mcp_startup": "one MCP wiring report; replaced, never appended",
}

#: Fields that grow with use and are therefore bounded HERE, by this module.
#: Each must be exercised by the class guard below.
_CAPPED_COLLECTION_FIELDS = {
    "usage_components": "capped at accumulation (USAGE_COMPONENT_CAP)",
    "jobs": "trajectories stripped, receipts folded, free text clipped on the wire",
}


def _collection_fields() -> set[str]:
    """Collection-typed fields on the state model, read from the model itself."""
    fields: set[str] = set()
    for name, field in FrontendSessionState.model_fields.items():
        annotation = str(field.annotation)
        if "list[" in annotation or "dict[" in annotation:
            fields.add(name)
    return fields


def test_every_collection_field_is_classified() -> None:
    """A NEW unbounded field must not be able to slip past the class guard.

    The previous version of the guard was a hand-built fixture that populated
    ten fields by name, so a field added later defaulted to empty, contributed
    zero bytes, and passed blind — it re-caught the two fields already known
    rather than closing the shape (review round 1, C3).

    Driving the classification off ``model_fields`` inverts that: adding a
    collection field to ``FrontendSessionState`` fails HERE until someone
    states which side of the line it is on, and if it is capped it must also
    be exercised by the frame guard below.
    """
    classified = set(_BOUNDED_COLLECTION_FIELDS) | set(_CAPPED_COLLECTION_FIELDS)
    unclassified = _collection_fields() - classified
    assert not unclassified, (
        f"unclassified collection field(s) on FrontendSessionState: {sorted(unclassified)}. "
        "Every collection field must be listed in _BOUNDED_COLLECTION_FIELDS (with the "
        "reason it cannot grow with conversation length or child count) or in "
        "_CAPPED_COLLECTION_FIELDS (and exercised by the frame guard). This is the "
        "check that stops the next unbounded field reaching a user's terminal."
    )
    # The reverse direction: a field removed from the model must not leave a
    # stale entry here implying coverage that no longer exists.
    stale = classified - _collection_fields()
    assert not stale, f"stale entries for fields that no longer exist: {sorted(stale)}"


def test_the_attach_frame_fits_for_a_session_that_ran_all_year() -> None:
    """The class guard: EVERY collection field maxed at once, still under the cap.

    Populated from ``_collection_fields()`` rather than by hand, so a field
    added to the model is filled here automatically and this assertion is what
    catches it. The reference machine's real session serialized 1,052,296
    bytes through this path.
    """
    jobs = [
        job.model_copy(
            update={
                # The per-job twin of usage_components: a child's own folded
                # receipts, which is what kept 18 stripped-trajectory jobs at
                # 196 KB on the reference machine.
                "usage": Usage(
                    input_tokens=1_000,
                    cost_components=[_receipt(index) for index in range(400)],
                ),
                # Unbounded in BYTES: whole child outputs. This is what pushed
                # the roster over the limit at ~130 rows (C4).
                "result_text": "r" * 40_000,
                "prompt": "p" * 40_000,
                "error_text": "e" * 40_000,
            }
        )
        for job in _jobs(200, 500)
    ]
    # Every collection field, filled past anything a real session reaches.
    populated: dict[str, Any] = {
        "jobs": jobs,
        "usage_components": [
            FrontendUsage.model_validate(_receipt(index).model_dump(mode="json"))
            for index in range(5_000)
        ],
        "child_costs": {f"job{index}": 1.25 for index in range(2_000)},
        "context_breakdown": {f"tool_{index}": 1_000 for index in range(2_000)},
        "queued_steering": [{"id": str(index), "text": "q" * 200} for index in range(200)],
        "live_events": [{"type": "message_update", "text": "e" * 200} for index in range(200)],
        "todos": [
            TodoPhaseState(
                name=f"phase {index}",
                items=[TodoItemState(text="t" * 200, status="pending")],
            )
            for index in range(200)
        ],
        "wakes": [],
        "mcp_servers": [
            McpServerState(name=f"server-{index}", status="connected") for index in range(200)
        ],
        "mcp_startup": {f"server-{index}": {"error": "e" * 200} for index in range(200)},
        "slash_capabilities": [],
        "model_catalogue": [
            {"provider": "p", "model_id": f"model-{index}", "context_window": 200_000}
            for index in range(1_000)
        ],
    }
    missing = _collection_fields() - set(populated)
    assert not missing, (
        f"collection field(s) not exercised by the class guard: {sorted(missing)}. "
        "Add them to `populated` so this assertion actually covers them."
    )
    state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        conversation_title="a" * 500,
        goal="g" * 2_000,
        cwd="/" + "d" * 500,
        **populated,
    )
    store = FrontendStateStore(state)
    frame = {
        "op": "frontend_sync",
        "data": sync_wire_payload(store.subscribe(lambda _u: None).sync),
    }

    size = _line_bytes(frame)
    assert size < _MAX_LINE_BYTES, (
        f"the attach frame is {size:,} bytes, over the {_MAX_LINE_BYTES:,} limit. "
        "Some field in FrontendSessionState grows without bound and is not "
        "capped at accumulation or stripped in sync_wire_payload. "
        f"{oversized_frame_report(frame, _MAX_LINE_BYTES)}"
    )


@pytest.mark.parametrize(
    "provider,model_id",
    [
        # BOTH wire shapes, deliberately. The first version of this test used
        # only Anthropic — the one wire where `_cache_tokens_are_inside_input`
        # is False and the `max(0, input - read - write)` subtraction never
        # runs — so it could not fail on any of the four divergences review
        # round 1 found (C1). OpenAI-shaped is where that subtraction is live.
        ("anthropic", "claude-opus-4-8-20260101"),
        ("openai", "gpt-5.6-sol"),
    ],
)
def test_folding_a_jobs_receipts_does_not_change_what_it_cost(provider: str, model_id: str) -> None:
    """The per-job list is priced, so it is folded rather than capped.

    ``job_cost`` sums ``usage.cost_components``: dropping a row there would
    undercount a child's spend, which is a wrong number on screen rather than
    a large frame. Folding must therefore reproduce per-receipt pricing
    exactly — see `_folded_components` for the two operations that do not
    commute with summation and how each is handled.

    Measured against the reference machine's real roster: 14 jobs, 104
    components folding to 1, worst cost difference $0.00.
    """
    reported = [
        Usage(
            input_tokens=1_000,
            output_tokens=100,
            cache_read_tokens=400,
            usd_cost=0.25,
            provider=provider,
            model_id=model_id,
        )
        for _ in range(40)
    ]
    # No usd_cost: priced from tokens at the model's rate, which only folds
    # correctly if the tokens are summed rather than the prices.
    estimated = [
        Usage(
            input_tokens=2_000,
            output_tokens=300,
            cache_read_tokens=800,
            cache_write_tokens=100,
            provider=provider,
            model_id=model_id,
        )
        for _ in range(40)
    ]
    usage = Usage(input_tokens=1, cost_components=[*reported, *estimated])
    label = f"{provider}/{model_id}"

    before = job_cost(SimpleNamespace(usage=usage, model_label=label), default_model_label=label)
    folded = usage.model_copy(update={"cost_components": _folded_components(usage.cost_components)})
    after = job_cost(SimpleNamespace(usage=folded, model_label=label), default_model_label=label)

    assert before is not None
    assert after == pytest.approx(before)
    # Bounded by DISTINCT IDENTITIES (reported and estimated are two buckets),
    # not by call count — which is what makes it survive a deep roster.
    assert len(folded.cost_components) == 2


def test_an_oversized_frame_is_reported_with_the_field_responsible() -> None:
    """The diagnosis, not just the refusal.

    An oversized frame used to present as a slow owner: unreadable line, dead
    pump, 15 s timeout, silent degrade. The report is what makes the next
    occurrence one log line to find instead of a profiling session, so it must
    name the offending field rather than only the size.
    """
    fits = {"op": "frontend_sync", "data": {"snapshot": {"todos": []}}}
    assert oversized_frame_report(fits, _MAX_LINE_BYTES) is None

    huge = {
        "op": "frontend_sync",
        "data": {"snapshot": {"usage_components": ["x" * 40] * 40_000, "cwd": "/tmp"}},
    }
    report = oversized_frame_report(huge, _MAX_LINE_BYTES)
    assert report is not None
    assert "usage_components" in report
    assert "n=40000" in report
    assert "1,048,576" in report


@pytest.mark.asyncio
async def test_attach_succeeds_against_a_session_that_exceeded_the_old_limit(
    tmp_path: Path, monkeypatch
) -> None:
    """The end-to-end claim: this session could not be attached to before.

    Drives the real ``RuntimeServer`` over a real socket with a roster whose
    naive snapshot is ~3 MB, then fetches one child's window on demand and
    checks the rows arrive intact.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    handle._frontend.mutate(jobs=_jobs(10, 500))
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        # Attach itself is the assertion: an oversized frame never arrives, so
        # before the fix this connect timed out waiting for the sync.
        assert len(remote.frontend_state.jobs) == 10
        job = remote.jobs.get("job4")
        assert job is not None
        assert list(job.trajectory) == []
        assert job.trajectory_length == 500

        assert await remote.load_job_trajectory("job4") is True
        loaded = remote.jobs.get("job4")
        assert loaded is not None
        assert len(loaded.trajectory) == 500
        assert loaded.trajectory[0]["tool_call_id"] == "call_000000"
        assert loaded.trajectory[-1]["tool_call_id"] == "call_000499"
        # One page cannot carry 500 rows; the loader must have paged.
        pages = [call for call in handle.calls if call[0] == "job_trajectory"]
        assert len(pages) > 1
        # Unopened children stay unfetched: the whole point is that a viewer
        # pays for the page it is reading and nothing else.
        unopened = remote.jobs.get("job5")
        assert unopened is not None
        assert list(unopened.trajectory) == []
        assert unopened.trajectory_length == 500
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_live_appends_reach_only_the_watched_job(tmp_path: Path, monkeypatch) -> None:
    """``watch_job`` is what makes the delta stream affordable."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    handle._frontend.mutate(jobs=_jobs(2, 1))
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        assert await remote.load_job_trajectory("job0") is True

        handle._frontend.mutate(jobs=_jobs(2, 4))
        for _ in range(100):
            watched = remote.jobs.get("job0")
            if watched is not None and len(watched.trajectory) == 4:
                break
            await asyncio.sleep(0.02)

        watched = remote.jobs.get("job0")
        assert watched is not None
        assert len(watched.trajectory) == 4, "appends for the open page must arrive"
        unwatched = remote.jobs.get("job1")
        assert unwatched is not None
        assert list(unwatched.trajectory) == [], "an unopened page must cost nothing"
        # The count still tells the truth for the job nobody is watching.
        assert unwatched.trajectory_length == 4
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_the_turn_end_checkpoint_does_not_grow_with_the_conversation() -> None:
    """The transcript pays for this row once per turn, forever.

    The attach frame is only half the budget the unbounded lists blew. The
    same state is appended to the transcript at EVERY turn end, so a list that
    grows with conversation length is re-serialized in full on every turn:
    quadratic. On the reference machine those rows were 64.3% of a 103 MB
    transcript, and ``usage_components`` alone was 48.2% of the whole file.

    Asserted as a RATIO between an early and a late checkpoint rather than as
    an absolute size, so the test states the property (the row stops growing)
    instead of pinning a byte count that legitimate new fields would break.
    """

    class _Transcript:
        def __init__(self) -> None:
            self.rows: list[int] = []

        async def append_custom(self, _custom_type: str, details: dict[str, Any]) -> None:
            self.rows.append(len(json.dumps(details).encode()))

    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))
    session = SimpleNamespace(effective_model=_priced_spec())
    transcript = _Transcript()

    for turn in range(40):
        for call in range(20):
            store.accrue_usage(session, _receipt(turn * 20 + call))
        await store.checkpoint(transcript)

    # Compared between two points that are BOTH past the cap's saturation
    # point, because the row legitimately grows while the bounded tail is
    # still filling. What must not happen is continued growth after that.
    settled, late = transcript.rows[19], transcript.rows[-1]
    assert late <= settled * 1.05, (
        f"the checkpoint grew from {settled:,} to {late:,} bytes between turn 20 "
        "and turn 40; something in the durable state accumulates with "
        "conversation length and is re-written in full on every turn"
    )


def test_the_roster_text_budget_is_shared_rather_than_per_row() -> None:
    """A per-row cap alone moves the threshold; a shared budget closes it.

    The defect this guards (review round 1, C4) is that ``jobs`` grew the frame
    linearly with roster depth — ~130 settled children with 4 KB of text each
    overflowed the limit with no receipt involved. Capping each row's text is
    not sufficient on its own: it lowers the constant and leaves the growth.

    Asserted as the STRUCTURAL property rather than a byte count: doubling the
    roster must not double the text on the wire. Rows are never dropped, so
    every child is still described.
    """

    def _text_bytes(count: int) -> tuple[int, int]:
        jobs = [
            job.model_copy(update={"result_text": "r" * 8_000, "prompt": "p" * 8_000})
            for job in _jobs(count, 0)
        ]
        store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1", jobs=jobs))
        rows = sync_wire_payload(store.subscribe(lambda _u: None).sync)["snapshot"]["jobs"]
        text = sum(len(row.get("result_text") or "") + len(row.get("prompt") or "") for row in rows)
        return len(rows), text

    shallow_rows, shallow_text = _text_bytes(20)
    deep_rows, deep_text = _text_bytes(200)

    # Every child is present at both depths: a dropped row would read as a
    # child that never ran, which the reader cannot tell from the truth.
    assert shallow_rows == 20
    assert deep_rows == 200

    # PER-ROW text must SHRINK as the roster deepens — that is the shared
    # budget working. Without it each row keeps its full per-field cap and the
    # per-row figure is flat while the total climbs linearly.
    assert deep_text / deep_rows < shallow_text / shallow_rows / 2, (
        f"per-row text was {shallow_text / shallow_rows:,.0f} chars at 20 rows and "
        f"{deep_text / deep_rows:,.0f} at 200; the budget is not being shared"
    )

    # The residual is stated rather than hidden: because rows are never dropped
    # and each keeps a legible floor, total text still rises with depth, just
    # far below linear. 10x the roster costs well under 10x the text.
    assert deep_text < shallow_text * 10 * 0.5, (
        f"text grew from {shallow_text:,} to {deep_text:,} chars over a 10x deeper "
        "roster; that is close enough to linear that the budget is not binding"
    )


@pytest.mark.parametrize(
    "case,provider,rows",
    [
        # The four divergences review round 1 reproduced (C1). Each one made
        # the folded total differ from the per-receipt total, and none could
        # be caught by a single-identity Anthropic fixture.
        (
            "negative usd_cost is rejected by _usage_cost, so it must not fold "
            "into the reported bucket",
            "anthropic",
            [{"usd_cost": 0.30}, {"usd_cost": -1.0}],
        ),
        (
            "non-finite usd_cost (json Infinity is wire-reachable) likewise",
            "anthropic",
            [{"usd_cost": 0.30}, {"usd_cost": float("inf")}],
        ),
        (
            "a negative token count ('-1 for unknown') is floored per receipt, "
            "and max(0,a)+max(0,b) != max(0,a+b)",
            "anthropic",
            [{"input_tokens": 1_000}, {"input_tokens": -1}],
        ),
        (
            "openai-shaped, cache_read > input on ONE receipt: the per-receipt "
            "subtraction floors at zero and does not commute",
            "openai",
            [
                {"input_tokens": 1_000, "cache_read_tokens": 200},
                {"input_tokens": 50, "cache_read_tokens": 900},
            ],
        ),
    ],
)
def test_folding_malformed_receipts_still_prices_identically(
    case: str, provider: str, rows: list[dict[str, Any]]
) -> None:
    """Folding must agree with per-receipt pricing on MALFORMED input too.

    `clients.py` sanitises negatives and non-finites at the wire, so these
    shapes need a malformed stored checkpoint or a non-wire producer rather
    than a live provider response — but the checkpoint on disk is exactly an
    untrusted stored input, and "lossless" was stated unconditionally. These
    are the cases that falsified it.
    """
    model_id = "claude-opus-4-8-20260101" if provider == "anthropic" else "gpt-5.6-sol"
    components = [
        Usage(output_tokens=100, provider=provider, model_id=model_id, **row) for row in rows
    ]
    label = f"{provider}/{model_id}"

    def _price(items: list[Usage]) -> float | None:
        total = 0.0
        for item in items:
            one = job_cost(
                SimpleNamespace(usage=item, model_label=label), default_model_label=label
            )
            if one is None:
                return None
            total += one
        return total

    before = _price(components)
    after = _price(list(_folded_components(components)))
    assert before is not None and after is not None
    assert after == pytest.approx(before, abs=1e-9), case
