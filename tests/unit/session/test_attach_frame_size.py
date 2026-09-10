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
import base64
import contextlib
import io
import json
import logging
import random
import re
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.types import ImageContent, ModelSpec, ToolResult, Usage
from local_operator.media import sniff_image
from local_operator.mobile.attach_client import (
    _FRAME_ENCODING_SLACK_BYTES,
    _MIN_VIABLE_IMAGE_BUDGET_BYTES,
    _READ_LIMIT_BYTES,
    AttachClient,
    OversizedRequest,
    _frame_overhead_bytes,
    _megabytes,
    _RefitReport,
    _text_is_the_bulk_refusal,
)
from local_operator.session.attention import AttentionStore
from local_operator.session.frontend_state import (
    _DERIVED_STATE_LABELS,
    _MODEL_CATALOGUE_LINE_LIMIT,
    _SHAREABLE_STATE_FIELDS,
    LIVE_EVENT_BLOCK_ELIDED_PLACEHOLDER,
    LIVE_EVENT_END_ROWS_MAX,
    LIVE_EVENT_TEXT_FLOOR_CHARS,
    LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS,
    MODEL_CATALOGUE_FLOOR_ROWS,
    USAGE_COMPONENT_CAP,
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
    FrontendSync,
    FrontendUpdate,
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
from local_operator.session.goal_loop import (
    LOOP_GOAL_CHARS,
    LOOP_REASON_CHARS,
    MAX_LOOP_ITERATIONS,
)
from local_operator.session.history_window import DisplayHistoryWindow
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import _MAX_LINE_BYTES, RuntimeServer
from local_operator.tui.costs import job_cost
from tests.unit.session.runtime.test_server import FakeHandle

#: A tool result big enough to be realistic. The point of the cap is that ONE
#: event carries an unbounded payload, so a small filler would test the row
#: count rather than the thing that overflows.
_RESULT_CHARS = 400


def _catalogue_row(index: int) -> dict[str, Any]:
    """One catalogue row exactly as ``refresh_model_catalogue`` builds it.

    Every key that method emits, with values the length real ones have
    (aggregator selectors are long: ``anthropic/claude-sonnet-4.5``). A row
    trimmed to the three interesting keys serializes at a quarter of the real
    weight, which is how this guard previously certified a frame that could not
    actually be sent.
    """
    return {
        "provider": "openrouter",
        "model_id": f"anthropic/claude-model-variant-{index}",
        "label": f"Claude Model Variant {index}",
        "context_window": 200_000,
        "default_context_window": 200_000,
        "max_context_window": 1_000_000,
        "input_price": 3.0,
        "output_price": 15.0,
        "connected": True,
        "aggregated": True,
        "routed": False,
    }


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
    # One replacement snapshot, never a per-turn receipt/history collection:
    # bounded conversation/token/anchor/kind plus a two-integer SQLite watermark.
    "attention": "fixed-shape latest outcome and read watermark, not accumulated history",
    "context_breakdown": "one entry per tool; bounded by the tool inventory",
    "child_costs": "one float per job; O(1) bytes each",
    "queued_steering": "drains every turn",
    # Was "explicitly bounded by _fold_live_event", which stopped being true the
    # moment a `tool_execution_end` began RETAINING its row instead of erasing
    # it (so a viewer reconnecting mid-turn can settle a card for work that
    # finished while it was away). The fold self-bounded only because completed
    # calls left nothing behind; with the end retained the field accumulates one
    # full tool result per call and measured 2 MB on a 1 MiB frame. Now bounded
    # at the WIRE like `model_catalogue`: oldest ends evicted past
    # LIVE_EVENT_END_ROWS_MAX, remaining result text clipped to a shared frame
    # budget. Every retained row keeps the identity and outcome a card needs.
    "live_events": "clipped at the wire: end rows capped newest-first, result text budgeted",
    "todos": "the user's own list, written by hand",
    "wakes": "the user's own schedules",
    "mcp_servers": "one row per configured server",
    "slash_capabilities": "one row per SLASH_COMMANDS entry",
    # Was described here as "bounded by the provider", which was not a bound at
    # all: nothing in this process capped it, and a real provider's list is
    # large enough to overflow the socket line on its own. Now bounded at the
    # wire by a RESIDUAL budget, with `model_catalogue_truncated` telling the
    # reader when the list it received is a prefix.
    "model_catalogue": "clipped at the wire to the frame's remaining bytes; truncation flagged",
    # One startup report from the MCP wiring pass, not an accumulator: it is
    # REPLACED on each wiring round rather than appended to, and its size is a
    # function of how many servers are configured.
    "mcp_startup": "one MCP wiring report; replaced, never appended",
    # A FIXED set of five scalar keys (status/completed/goal/iterations/reason),
    # REPLACED on each publish rather than appended to, so it does not grow with
    # iteration count: a 100-iteration loop and a 1-iteration loop serialize the
    # same shape, only `completed` differs. BOTH free-text values are clipped
    # where they enter the state: the judge's `reason` at LOOP_REASON_CHARS in
    # `_parse_loop_verdict`, and the user's `goal` at LOOP_GOAL_CHARS in
    # `GoalLoop.start`.
    "loop": "five scalar keys, replaced per publish; both free-text values clipped at entry",
}

#: Fields that grow with use and are therefore bounded HERE, by this module.
#: Each must be exercised by the class guard below.
_CAPPED_COLLECTION_FIELDS = {
    "usage_components": "capped at accumulation (USAGE_COMPONENT_CAP)",
    "jobs": "trajectories stripped, receipts folded, free text clipped on the wire",
}


#: Every free-text field on ``JobState``, with what bounds it ON THE WIRE.
#:
#: The state-level table above cannot see these. ``jobs`` is one entry there, so
#: a new field on the ROW model is invisible to ``_collection_fields()`` no
#: matter how much it costs — and a row field is multiplied by roster depth,
#: which is the more dangerous of the two shapes.
#:
#: That blindness shipped a regression. ``launch_message_id`` was added to
#: ``JobState`` as a per-row scalar bounded by nothing; at 46.7 B on every task
#: child it cut the maximum attachable roster from 812 rows to 769 against a
#: guard that passes with 3,316 bytes to spare, and no assertion here could
#: fail (review round 1 blocker). Scalars are listed as well as collections
#: because that regression WAS a scalar: "collection" is not the property that
#: makes a row field dangerous, being per-row is.
_BOUNDED_JOB_FIELDS = {
    # Identity and O(1) scalars: bounded by their own shape, not by use.
    "id": "one job id",
    "type": "one of a fixed set of job types",
    "status": "one of a fixed set of statuses",
    "queued": "bool",
    "label": "the caller's short label",
    "agent": "one role name",
    "intent": "one short intent line",
    "model_label": "one model identity",
    "context_window": "int",
    "direct_cost": "float",
    "direct_cost_knowledge": "enum",
    "start_time": "float",
    "started_at": "float",
    "settled_at": "float",
    "trajectory_length": "int",
    "output_seq": "int",
    "restored": "bool",
    "parent_job_id": "one job id",
    "session_id": "one session id",
    "session_dir": "one filesystem path",
    "agent_role": "one role name",
    "effort": "one tier name",
    "output_tail": "bounded by the job runner's own tail buffer",
    "latest_details": "one progress payload, replaced not appended",
    "usage": "folded by _fold_job_usage_in_place",
    "attempt_aliases": "one id per collapsed resume attempt",
}

#: Row fields bounded by THIS module, each of which the frame guard must
#: populate at its worst case.
_CAPPED_JOB_FIELDS = {
    "trajectory": "stripped entirely on the wire",
    "todos": "nulled on the wire; fetched per job",
    "descendant_usage": "folded by _fold_job_usage_in_place",
    "result_text": "clipped to JOB_RESULT_WIRE_CHARS and the frame text share",
    "prompt": "clipped to JOB_PROMPT_WIRE_CHARS and the frame text share",
    "error_text": "clipped to JOB_ERROR_WIRE_CHARS and the frame text share",
    "launch_prompts": "per-entry, per-row and roster-shared budgets",
    "launch_message_id": "omitted when derivable from the job id",
}


def _collection_fields() -> set[str]:
    """Collection-typed fields on the state model, read from the model itself."""
    fields: set[str] = set()
    for name, field in FrontendSessionState.model_fields.items():
        annotation = str(field.annotation)
        if "list[" in annotation or "dict[" in annotation:
            fields.add(name)
    return fields


def _job_fields() -> set[str]:
    """Every field on the ROW model, read from the model itself.

    Not filtered to collections, unlike :func:`_collection_fields`: the
    regression this exists to catch was a per-row STRING. What makes a row field
    dangerous is that the frame pays for it once per child.
    """
    return set(JobState.model_fields)


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


def test_shareable_state_fields_are_real_and_immutable() -> None:
    """The copy-free allow-list must name real fields that cannot be mutated.

    ``read_field`` hands out the store's OWN object, so the allow-list is the
    entire safety argument. Round 2 found it asserting neither half: it listed
    ``conversation_name`` (the field is ``conversation_title``), which passed
    the ``KeyError`` gate and then raised ``AttributeError`` from ``getattr``
    on a per-frame path, and it admitted ``selected_model``/``effective_model``
    /``last_usage`` — non-frozen models whose instance a caller could rewrite,
    removing an invariant ``state`` provides by copying.

    Driven off ``model_fields`` and the annotation, in the same shape as the
    collection-field guard above: a renamed field fails here rather than at a
    user's terminal, and a field whose type becomes mutable cannot re-enter the
    set without someone stating why sharing it is safe.
    """
    immutable = (str, bool, int, float)

    unknown = _SHAREABLE_STATE_FIELDS - set(FrontendSessionState.model_fields)
    assert not unknown, (
        f"copy-free field(s) that do not exist on FrontendSessionState: {sorted(unknown)}. "
        "`read_field` checks membership BEFORE `getattr`, so a name that is not a real "
        "field passes the guard and then raises AttributeError on a per-frame path."
    )

    mutable: list[str] = []
    for name in sorted(_SHAREABLE_STATE_FIELDS):
        annotation = FrontendSessionState.model_fields[name].annotation
        types = {
            arg for arg in getattr(annotation, "__args__", (annotation,)) if arg is not type(None)
        }
        if not types or not all(isinstance(t, type) and issubclass(t, immutable) for t in types):
            mutable.append(name)
    assert not mutable, (
        f"copy-free field(s) whose value is not deeply immutable: {mutable}. "
        "`read_field` shares the store's own object, so anything a caller can mutate "
        "in place — a model, a dict, a list — must read through `state` instead and "
        "pay for its deep copy."
    )


def test_every_job_row_field_is_classified() -> None:
    """The same closure for the ROW model, which the state-level guard cannot see.

    ``_collection_fields`` reads ``FrontendSessionState``, where the entire
    roster is one entry (``jobs``). A field added to ``JobState`` is therefore
    invisible to it however much the frame pays for it — and a row field is
    multiplied by roster depth, so it is the more dangerous of the two shapes.

    That gap shipped a measured regression: ``launch_message_id`` went on as a
    per-row scalar bounded by nothing, cost 46.7 B on every task child, and cut
    the maximum attachable roster from 812 rows to 769 while every assertion in
    this file stayed green (review round 1 blocker). Adding a field to
    ``JobState`` now fails HERE until someone states which side of the line it
    is on.
    """
    classified = set(_BOUNDED_JOB_FIELDS) | set(_CAPPED_JOB_FIELDS)
    unclassified = _job_fields() - classified
    assert not unclassified, (
        f"unclassified field(s) on JobState: {sorted(unclassified)}. Every row field must "
        "be listed in _BOUNDED_JOB_FIELDS (with the reason it cannot grow with "
        "conversation length or child count) or in _CAPPED_JOB_FIELDS (and exercised by "
        "the frame guard). The roster is paid for ONCE PER CHILD, so an unbounded row "
        "field costs the attach ceiling far faster than a state-level one."
    )
    stale = classified - _job_fields()
    assert not stale, f"stale entries for row fields that no longer exist: {sorted(stale)}"


def test_the_attach_frame_fits_for_a_session_that_ran_all_year(tmp_path: Path) -> None:
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
                # The PER-ROW launch identity and its collapsed-attempt map,
                # which the state-level classifier structurally cannot reach.
                # Unbounded these cost 9,290 B (identity alone) and 29,758 B
                # (both) across these 200 rows, either of which overflows a
                # frame that passed with 3,316 B to spare — the round-1 blocker.
                #
                # The identity is deliberately NOT the derivable
                # ``subagent-launch:<job id>``: elision would drop it and the
                # guard would measure nothing. Every real producer emits the
                # derivable form (verified across launch, #314 resume-fold and
                # the persist/restore round trip), so this is the worst case
                # that cannot be elided away, held against the wire's own bounds.
                #
                # These two are what the OTHER maxed fields have to leave room
                # for. Their aggregate is capped
                # (JOB_LAUNCH_IDS_FRAME_BUDGET_CHARS +
                # JOB_LAUNCH_PROMPTS_FRAME_BUDGET_CHARS ≈ 26 KB however deep the
                # roster), so the per-row text below is trimmed by that much to
                # keep the fixture's total at the same worst case it has always
                # asserted rather than a strictly larger one.
                "launch_message_id": f"subagent-launch:resumed-{index:04d}",
                # Twelve collapsed attempts of 4 KB each: past both the per-row
                # entry cap and the roster-shared budget, so most rows here land
                # on the elided and dropped tiers, which is the shape a deep
                # roster actually puts on the wire.
                "launch_prompts": {
                    f"subagent-launch:attempt-{index:04d}-{attempt}": "L" * 4_000
                    for attempt in range(12)
                },
            }
        )
        for index, job in enumerate(_jobs(200, 500))
    ]
    # Exercise the real authority's complete, non-empty snapshot: an earlier
    # read plus a newer unread outcome. A new attention payload field must get
    # an explicit size-policy review instead of hiding behind dict[str, Any].
    attention_store = AttentionStore(tmp_path / "attention.db")
    read_token = str(uuid.uuid4())
    attention_store.publish("session/s1", read_token, "old-result", "complete")
    attention_store.acknowledge("session/s1", read_token)
    attention = attention_store.publish("session/s1", str(uuid.uuid4()), "new-result", "complete")
    assert set(attention) == {
        "conversation_id",
        "completion_token",
        "anchor_id",
        "kind",
        "unseen",
        "revision",
    }
    assert attention["unseen"] and attention["revision"] == [2, 1]

    # Every collection field, filled past anything a real session reaches.
    populated: dict[str, Any] = {
        "attention": attention,
        "jobs": jobs,
        "usage_components": [
            FrontendUsage.model_validate(_receipt(index).model_dump(mode="json"))
            for index in range(5_000)
        ],
        "child_costs": {f"job{index}": 1.25 for index in range(2_000)},
        "context_breakdown": {f"tool_{index}": 1_000 for index in range(2_000)},
        "queued_steering": [{"id": str(index), "text": "q" * 200} for index in range(200)],
        # COMPLETED TOOL CALLS, not `message_update` rows. The old fixture used
        # 200 `message_update`s, which `_fold_live_event` dedupes to a single
        # row by phase — so it could never exercise the shape that actually
        # accumulates, and the whole suite passed while `live_events` was
        # contributing 2 MB to a 1 MiB frame. A retained `tool_execution_end`
        # is the row that grows one-per-call with a full tool result attached;
        # this fixture is what makes `assert_frame_fits` able to see it.
        #
        # Each row carries all THREE payload shapes a real turn produces, because
        # a bound written against one of them silently misses the others. The
        # first cut of this bound clipped `content[].text` and let both of the
        # rest through: an image block keeps its base64 under `data` (one
        # permitted image is ~1.4 MB encoded, over the whole line limit by
        # itself) and `details` carries the MCP bridge's `server_result` dump
        # (50 calls measured 2.6 MB). A fixture that only holds text can only
        # ever catch the bug that was already fixed.
        "live_events": [
            row
            for index in range(1_000)
            for row in (
                {
                    "type": "tool_execution_start",
                    "tool_call_id": f"call-{index}",
                    "tool_name": "read",
                },
                {
                    "type": "tool_execution_end",
                    "tool_call_id": f"call-{index}",
                    "tool_name": "read",
                    "is_error": False,
                    "result": {
                        "tool_call_id": f"call-{index}",
                        "tool_name": "read",
                        "content": [
                            {"type": "text", "text": "e" * 60_000},
                            {
                                "type": "image",
                                "data": "A" * 1_400_000,
                                "mime_type": "image/png",
                            },
                        ],
                        "details": {"server_result": {"blob": "D" * 50_000}},
                        "is_error": False,
                    },
                },
            )
        ],
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
        # A loop at the iteration ceiling with its judge reason at the clip.
        # Filled at the BOUND rather than past it, because that is the largest
        # value the producer can actually publish: the key set is fixed and
        # `_parse_loop_verdict` clips the one free-text value, so a bigger
        # fixture here would assert against a state the runtime cannot reach.
        "loop": {
            "status": "running",
            "completed": MAX_LOOP_ITERATIONS,
            "iterations": MAX_LOOP_ITERATIONS,
            "goal": "g" * LOOP_GOAL_CHARS,
            "reason": "r" * LOOP_REASON_CHARS,
        },
        "slash_capabilities": [],
        # PRODUCTION-SHAPED rows, and far more of them than any provider lists.
        # The old fixture used a 3-key, 63 B row while `refresh_model_catalogue`
        # builds an 11-key, ~267 B one — understating a real row 4.24x, so the
        # guard measured a catalogue a quarter of its true weight and passed a
        # frame the socket could not carry. The count is past the ~1,410 one QA
        # backend published because this field takes a RESIDUAL wire budget
        # (`MODEL_CATALOGUE_FLOOR_ROWS`): the assertion below is that the frame
        # fits HOWEVER long the owner's list is, so the fixture has to be longer
        # than the budget can hold rather than tuned to sit under it.
        "model_catalogue": [_catalogue_row(index) for index in range(5_000)],
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

    assert frame["data"]["snapshot"]["attention"] == attention
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


@pytest.mark.parametrize("invalidate", ["", "epoch", "removed", "identity"])
@pytest.mark.asyncio
async def test_watched_todo_fetch_cannot_roll_back_newer_state(tmp_path, monkeypatch, invalidate):
    import threading

    entered, release = threading.Event(), threading.Event()

    def plan(text):  # noqa: ANN001, ANN202
        return [{"name": "Work", "items": [{"text": text, "status": "pending", "reason": ""}]}]

    class DetailHandle(FakeHandle):
        async def job_trajectory(self, job_id, offset, limit):  # noqa: ANN001, ANN202
            page = await super().job_trajectory(job_id, offset, limit)
            snapshot = self._frontend.state
            job = next(row for row in snapshot.jobs if row.id == job_id)
            page.update(
                detail_job_id=job_id,
                detail_session_id=job.session_id,
                detail_epoch=snapshot.epoch,
                detail_sequence=snapshot.sequence,
                todos=job.model_dump(mode="json")["todos"],
            )
            entered.set()
            await asyncio.to_thread(release.wait, 10)
            return page

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = DetailHandle()
    jobs = [
        row.model_copy(update={"session_id": f"child-{i}", "todos": plan("Earlier")})
        for i, row in enumerate(_jobs(2, 1))
    ]
    handle._frontend.mutate(jobs=jobs)
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        remote = await RemoteSession.connect(
            await _record(tmp_path), "s1", config_dir=tmp_path, takeover_factory=_never
        )

        def todos_for(job_id: str):  # noqa: ANN202
            assert remote is not None
            job = remote.jobs.get(job_id)
            assert job is not None
            return job.todos

        assert todos_for("job0") is None
        loading = asyncio.create_task(remote.load_job_trajectory("job0"))
        assert await asyncio.to_thread(entered.wait, 10)
        changed = asyncio.Event()
        subscription = remote.subscribe_frontend(lambda _: changed.set())
        try:
            updated = [job.model_copy(update={"todos": plan("Newer")}) for job in jobs]
            handle._frontend.mutate(jobs=updated)
            await asyncio.wait_for(changed.wait(), 10)
            assert todos_for("job0") == plan("Newer")
            assert todos_for("job1") is None
            state = remote.frontend_state
            if invalidate == "epoch":
                remote._install_frontend(state.model_copy(update={"epoch": "new-owner"}))
            elif invalidate == "removed":
                remote._install_frontend(state.model_copy(update={"jobs": []}))
            elif invalidate == "identity":
                remote._install_frontend(
                    state.model_copy(
                        update={
                            "jobs": [
                                row.model_copy(update={"session_id": "resumed-child"})
                                for row in state.jobs
                            ]
                        }
                    )
                )
            release.set()
            assert await loading is (not bool(invalidate))
            if not invalidate:
                assert todos_for("job0") == plan("Newer")
                changed.clear()
                handle._frontend.mutate(jobs=[job.model_copy(update={"todos": []}) for job in jobs])
                await asyncio.wait_for(changed.wait(), 10)
                assert todos_for("job0") == []
        finally:
            subscription.unsubscribe()
    finally:
        release.set()
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


@pytest.mark.asyncio
async def test_an_unreadable_frame_fails_fast_instead_of_waiting_out_the_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    """The user must not sit through 15 s of silence for a frame we cannot read.

    Making the pump honest internally was only half the fix: `_await_frontend`
    still waited out its full 15 s timeout, because nothing failed the pending
    future when the connection died (UX round 1, U2). The oversized frame is
    known unreadable within milliseconds, so the wait must end then — and the
    reason must be the one the pump produced, not a generic timeout, or the
    copy that explains what happened never reaches a surface (design round 1,
    D5).

    Driven against the REAL server over a REAL socket with a genuinely
    oversized frame, because the bug is in how the two halves interact.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    # Defeat the wire bounds deliberately: an extra field the stripper does not
    # know about, carrying more than the line limit. This is the shape of the
    # NEXT unbounded field, which is exactly what must not hang.
    handle._frontend.mutate(cwd="x" * (_MAX_LINE_BYTES + 1024))
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        record = await _record(tmp_path)
        started = asyncio.get_running_loop().time()
        with pytest.raises(ConnectionError) as caught:
            await RemoteSession.connect(record, "s1", config_dir=tmp_path, takeover_factory=_never)
        elapsed = asyncio.get_running_loop().time() - started

        # The 15 s sync timeout is the backstop for a silent owner, not the
        # budget for a failure we already detected. Generous bound: the claim
        # is "does not wait out the timeout", not a performance figure.
        assert elapsed < 5.0, (
            f"an unreadable frame took {elapsed:.1f}s to report; the connection died "
            "immediately and the wait should have ended with it"
        )
        # And the reason names what actually happened.
        assert "too large" in str(caught.value), (
            f"the failure reported {caught.value!r}, which does not tell the user "
            "the frame could not be read"
        )
    finally:
        registrant.close()


def test_the_loop_goal_is_clipped_where_it_enters_the_state() -> None:
    """`goal` is bounded by the producer, not merely by the fixture above.

    The frame guard asserted against a 2,000-char goal while nothing enforced
    2,000: `GoalLoop.start` took the slash argument verbatim, bounded only by
    the desktop route's `max_length=200_000`. The guard was therefore testing a
    limit that did not exist, and an oversized attach frame is a DROPPED LINE --
    a session that cannot be attached to (review round 2, MINOR-1).
    """
    import asyncio

    from local_operator.session.goal_loop import GoalLoop

    async def start_a_huge_goal() -> dict[str, object]:
        async def never_runs(_text: str) -> str:
            # The loop is cancelled before a turn completes; this exists to
            # satisfy the judge signature, not to be called.
            raise AssertionError("the loop should not reach its judge")

        loop = GoalLoop(
            prompt=lambda _text: asyncio.sleep(0),
            judge=never_runs,
            abort=lambda: None,
            changed=lambda _state: None,
        )
        # The largest value the ROUTE accepts, which is what the producer must
        # defend against -- two orders of magnitude past the guard's fixture.
        state = loop.start("G" * 200_000, "")
        await loop.cancel()
        return state

    state = asyncio.run(start_a_huge_goal())
    assert len(str(state["goal"])) == LOOP_GOAL_CHARS


def _catalogue_frame(rows: int, jobs: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(frame, snapshot)`` for a session offering ``rows`` models."""
    state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        jobs=_jobs(jobs, 50) if jobs else [],
        model_catalogue=[_catalogue_row(index) for index in range(rows)],
    )
    store = FrontendStateStore(state)
    payload = sync_wire_payload(store.subscribe(lambda _u: None).sync)
    return {"op": "frontend_sync", "data": payload}, payload["snapshot"]


def test_a_real_providers_catalogue_rides_the_wire_whole() -> None:
    """The bound must not cost the picker models a provider actually lists.

    The reason this file's guard cannot be satisfied with a flat cap. Sizing a
    constant cap against the 200-child worst case below yields ~241 rows, which
    would hide two thirds of a real provider's list to make a fixture pass —
    the same defect as trimming the fixture, one level down. So the budget is
    residual, and this is the half that keeps it honest: an ordinary session
    (the deepest roster observed on the reference machine is 19) sends every
    row, flag clear, at both a real provider's ~600 and the 1,410 one QA
    backend published.
    """
    for rows in (600, 1_410):
        frame, snapshot = _catalogue_frame(rows, jobs=19)
        assert len(snapshot["model_catalogue"]) == rows, rows
        assert snapshot["model_catalogue_truncated"] is False, rows
        assert _line_bytes(frame) < _MAX_LINE_BYTES, rows


def test_a_catalogue_too_large_for_the_frame_is_clipped_and_says_so() -> None:
    """Clipping is bounded, ordered, and never silent.

    A short list that claims to be complete is the failure the flag exists for:
    the picker would say "these are your models" while omitting hundreds, and
    the user would conclude a model they can really switch to does not exist.
    """
    frame, snapshot = _catalogue_frame(5_000, jobs=200)
    kept = snapshot["model_catalogue"]
    assert _line_bytes(frame) < _MAX_LINE_BYTES
    assert len(kept) < 5_000
    assert snapshot["model_catalogue_truncated"] is True
    # A PREFIX in the owner's order, not an arbitrary slice: that order is the
    # one the picker renders, so the rows kept are the most relevant ones.
    assert kept == [_catalogue_row(index) for index in range(len(kept))]
    # Never emptied. An empty picker claims the session can switch to nothing,
    # which is the same undetectable lie as a silently short list.
    assert len(kept) >= MODEL_CATALOGUE_FLOOR_ROWS


def test_an_untruncated_catalogue_leaves_the_flag_alone() -> None:
    """The flag describes the list that SHIPPED, so it stays false when whole."""
    _frame, snapshot = _catalogue_frame(10)
    assert snapshot["model_catalogue"] and snapshot["model_catalogue_truncated"] is False


def test_an_oversized_frame_does_not_call_a_complete_catalogue_partial() -> None:
    """An oversized frame is not evidence that the catalogue was clipped.

    When the overflow comes from other fields and the catalogue already sits at
    or below :data:`MODEL_CATALOGUE_FLOOR_ROWS`, the search keeps every row. The
    flag is set before the search so each measurement pays for the key that
    ships, so without withdrawing it the reader would be told models are missing
    when none are -- the same lie as a silently short list, inverted, sending
    the user hunting for a model that was never omitted.
    """
    # The overflow has to come from a field the wire does NOT budget, or the
    # branch under test is never entered: the roster is text-budgeted, so even
    # 800 jobs reach only ~519 KB. Todos are unbounded, which is also how a
    # real session gets here.
    state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        todos=[
            TodoPhaseState(
                name="p" * 200,
                items=[TodoItemState(text="t" * 300) for _ in range(6)],
            )
            for _ in range(600)
        ],
        model_catalogue=[_catalogue_row(index) for index in range(MODEL_CATALOGUE_FLOOR_ROWS)],
    )
    store = FrontendStateStore(state)
    payload = sync_wire_payload(store.subscribe(lambda _update: None).sync)
    snapshot = payload["snapshot"]

    # The fixture must actually overflow, or this test proves nothing.
    frame_bytes = len(json.dumps({"op": "frontend_sync", "data": payload}).encode()) + 1
    assert frame_bytes > _MODEL_CATALOGUE_LINE_LIMIT, frame_bytes

    assert len(snapshot["model_catalogue"]) == MODEL_CATALOGUE_FLOOR_ROWS
    assert snapshot.get("model_catalogue_truncated", False) is False


def test_the_catalogue_budget_tracks_the_socket_line_limit() -> None:
    """The mirrored limit must equal the reader's, or the budget is fiction.

    ``frontend_state`` cannot import ``runtime.server`` (server imports it), so
    the limit is mirrored. This is the pin that keeps the copy honest.
    """
    assert _MODEL_CATALOGUE_LINE_LIMIT == _MAX_LINE_BYTES


def test_the_client_read_limit_tracks_the_socket_line_limit() -> None:
    """The reader's limit must equal the writer's, or one side is wrong.

    Same pin as the catalogue budget above, for the third copy of this value.
    ``attach_client`` declares its own ``_READ_LIMIT_BYTES`` because the
    transport is deliberately import-light, so nothing but this assertion keeps
    the two honest. Drift IS caught today, but only incidentally, by e2e tests
    whose failure message never names the cause: a client limit BELOW the
    server's turns a legal frame into "owner sent a frame too large to read",
    and one ABOVE it means the guard this module tests stops matching what the
    reader will actually refuse.
    """
    assert _READ_LIMIT_BYTES == _MAX_LINE_BYTES


@pytest.mark.asyncio
async def test_attach_succeeds_against_an_owner_offering_thousands_of_models(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end over a real socket: a huge catalogue attaches, and says it clipped.

    The serialization tests above measure the frame; this one proves the claim
    that matters — that a viewer can actually CONNECT to an owner whose model
    catalogue would otherwise overflow the control socket's line. Before the
    wire bound this frame was silently dropped by the reader, so the connect
    timed out and degraded to a cold session (the failure mode
    ``oversized_frame_report`` exists to explain).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    handle._frontend.mutate(
        jobs=_jobs(200, 500),
        model_catalogue=[_catalogue_row(index) for index in range(5_000)],
    )
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        # The attach completing at all is the assertion.
        catalogue = remote.frontend_state.model_catalogue
        assert 0 < len(catalogue) < 5_000
        assert remote.frontend_state.model_catalogue_truncated is True
        # The rows that survived are usable, in the owner's own order.
        assert catalogue[0] == _catalogue_row(0)
        assert remote.runtime_model_catalogue()[0]["model_id"] == _catalogue_row(0)["model_id"]
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_an_oversized_delta_keeps_the_socket_and_the_follower_resyncs(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end over a REAL socket: the connection survives AND state recovers.

    The serialization tests above prove the frame validates. This one proves the
    two claims that actually matter to the operator, neither of which a shape
    assertion can reach:

    1. The socket stays up. Before the fix the follower's ``model_validate``
       raised, the pump's catch-all named it ``owner frame could not be
       applied`` and dropped the connection — the reported failure.
    2. The follower ends up with the owner's CANONICAL state, not silently
       stale state. The degraded frame consumed a sequence, so the gap check
       stays satisfied forever; without a forced re-snapshot the viewer would
       keep a title from before the shed delta and never learn otherwise. That
       is the difference between a fixed bug and a hidden one.

    The fixture degrades a DELTA while leaving the snapshot small — a huge
    catalogue mutation is 1.5 MB on the wire and re-syncs to ~1.2 KB — which is
    the real shape: the snapshot the recovery rides on has to fit.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        _assert_reachable(remote)
        assert remote.frontend_state.conversation_title == "fake"

        # One mutation whose DELTA cannot ride the wire. `conversation_title`
        # rides along in the same delta: it is the field whose loss proves the
        # follower would otherwise be stale, because it survives into the
        # snapshot while the catalogue is what made the frame oversized.
        handle._frontend.mutate(
            model_catalogue=[_catalogue_row(index) for index in range(5_000)],
            conversation_title="after the degrade",
        )

        # The owner's delta really was over the limit, so this test cannot pass
        # by the frame simply fitting.
        assert (
            _line_bytes(
                {
                    "op": "frontend_update",
                    "data": {
                        "epoch": handle._frontend.state.epoch,
                        "sequence": handle._frontend.state.sequence,
                        "changes": {
                            "model_catalogue": [_catalogue_row(index) for index in range(5_000)],
                            "conversation_title": "after the degrade",
                        },
                    },
                }
            )
            > _MAX_LINE_BYTES
        )

        # Wait on the RESULT, never on the clock: the recovery is an async task
        # the frame scheduled, so poll the property it repairs.
        for _ in range(200):
            if remote.frontend_state.conversation_title == "after the degrade":
                break
            await asyncio.sleep(0.02)

        # 1. The connection is still up. This is the regression assertion.
        client = remote._client
        assert client is not None and client.connected, "the degraded delta killed the socket"

        # 2. Canonical state matches the OWNER, rather than being stale at the
        #    pre-degrade value while the sequence marches on.
        owner = handle._frontend.state
        assert remote.frontend_state.conversation_title == owner.conversation_title
        assert remote.frontend_state.sequence == owner.sequence
        # The catalogue came back too (clipped by the wire budget, as ever), so
        # the recovery restored the shed body rather than only the scalar.
        assert remote.frontend_state.model_catalogue
        assert remote.frontend_state.model_catalogue[0] == _catalogue_row(0)

        # And the stream is still LIVE after recovery: a later ordinary delta
        # applies on top, proving the sequence cursor was not left desynced.
        handle._frontend.mutate(conversation_title="still live")
        for _ in range(200):
            if remote.frontend_state.conversation_title == "still live":
                break
            await asyncio.sleep(0.02)
        assert remote.frontend_state.conversation_title == "still live"
        assert client.connected
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0.0, 0.05])
async def test_a_burst_of_degraded_deltas_coalesces_into_one_resync(
    tmp_path: Path, monkeypatch, delay: float
) -> None:
    """Frames arriving DURING a refresh fold into it rather than each buying one.

    The frame that degrades is by definition a frame the session is big enough
    to produce repeatedly — the operator hit this with nine subagents running —
    so answering a 1.5 MB frame with a sync RPC per frame while a refresh is
    already running would make the overload worse.

    THE BOUND IS IN-FLIGHT COALESCING, NOT A RATE LIMIT, and this test says so
    because an earlier version of it did not: it asserted ``syncs <= 3`` while
    emitting all 20 mutations without awaiting, which pins only the unpaced
    regime. QA swept the spacing and found 1 sync back to back but 20 syncs at
    50 ms apart. That is correct — a frame arriving after a snapshot completed
    shed state that snapshot did not cover, so it genuinely owes another sync —
    and the paced case is asserted below so the claim and the code agree.

    What must hold at EVERY spacing is that the invalidation does not exceed the
    work: ``_display_revision`` is a TUI cache key, so a bump per frame rather
    than per reload moves the refresh storm from the RPC layer into the
    presentation layer, where this test would not have seen it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        _assert_reachable(remote)
        client = remote._client
        assert client is not None

        syncs = 0
        original = client.frontend_sync

        async def counting_sync() -> Any:
            nonlocal syncs
            syncs += 1
            return await original()

        monkeypatch.setattr(client, "frontend_sync", counting_sync)
        assert client.frontend_sync is counting_sync, "the sync counter did not land"

        # ``_load_frontend_history`` is where a reload actually happens, and it
        # owns the one legitimate ``_display_revision`` bump. Counting it is
        # what lets the invalidation assertion below mean "per reload" rather
        # than "a number that looked small in the unpaced case".
        loads = 0
        original_load = remote._load_frontend_history

        async def counting_load(frontend: Any) -> None:
            nonlocal loads
            loads += 1
            return await original_load(frontend)

        monkeypatch.setattr(remote, "_load_frontend_history", counting_load)
        assert remote._load_frontend_history is counting_load, "the load counter did not land"

        revision_before = remote.display_history_revision

        # Every delta must be GENUINELY oversized, so the catalogue has to
        # DIFFER each time: re-sending an identical list produces no field diff
        # and a small delta, which would let this pass for free.
        rounds = 20
        oversized = 0
        for index in range(rounds):
            update = handle._frontend.mutate(
                model_catalogue=[_catalogue_row(row + index * 10_000) for row in range(5_000)],
                conversation_title=f"title-{index}",
            )
            assert update is not None
            if (
                _line_bytes({"op": "frontend_update", "data": update.model_dump(mode="json")})
                > _MAX_LINE_BYTES
            ):
                oversized += 1
            if delay:
                await asyncio.sleep(delay)
        # The fixture has to keep producing frames the wire cannot carry, or
        # every assertion below passes for free against ordinary deltas.
        assert oversized == rounds, f"only {oversized}/{rounds} frames were oversized"

        for _ in range(400):
            if (
                remote.frontend_state.conversation_title == f"title-{rounds - 1}"
                and not remote._frontend_resync_pending
            ):
                break
            await asyncio.sleep(0.02)

        # Converged on the owner's latest, not on some intermediate the
        # coalescing happened to stop at. True at BOTH spacings.
        owner = handle._frontend.state
        assert remote.frontend_state.conversation_title == owner.conversation_title
        assert remote.frontend_state.sequence == owner.sequence
        assert client.connected

        # THE INVALIDATION MUST TRACK THE WORK, NOT THE FRAMES. This is the
        # assertion the previous version lacked: the RPCs coalesced while
        # ``_display_revision`` was bumped once per degraded frame, so the TUI
        # dropped its cached sidebar presentation ~20x per burst for a single
        # reload. Tying it to ``loads`` rather than to a constant keeps it
        # honest at any spacing, since the sync count itself is legitimately
        # scheduling dependent.
        assert remote.display_history_revision - revision_before == loads, (
            f"{rounds} degraded frames bumped the display revision "
            f"{remote.display_history_revision - revision_before} times "
            f"for {loads} history reloads"
        )
        assert 0 < syncs <= rounds
        assert loads == syncs
        if not delay:
            # Back to back, the in-flight slot is what does the work: the whole
            # burst folds into a single pass.
            assert syncs <= 3, f"{rounds} unpaced degraded frames caused {syncs} round trips"
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


class _WindowedHandle(FakeHandle):
    """A ``FakeHandle`` whose followers negotiate the WINDOWED history path.

    The plain double offers no ``history_page``, so ``RuntimeServer`` never
    advertises ``display-history-window-v1`` and every follower built on it
    lands in ``_display_window_supported=False`` — the legacy/full-replay case.
    That left the MODERN viewer, which is what the TUI actually builds, with no
    coverage of the degrade path at all, and the two are not interchangeable
    here: ``_invalidate_display_history`` returns early for the legacy follower
    while ``_display_invalidated`` latches for the windowed one, so the blast
    radius of a failed re-sync differs by viewer.

    THE WINDOW IS CAPTURED FROM A REAL ``Transcript``, NOT SYNTHESISED, and the
    reason is a fixture defect worth naming (review round 2, Q). The first
    version hardcoded ``history_generation=1`` against a snapshot carrying 0.
    ``_load_frontend_history`` takes ``_loaded_history_generation`` from the
    WINDOW, so that follower was permanently invalidated: every ordinary delta
    re-entered ``_invalidate_display_history`` and started a fresh refresh,
    which serviced any orphaned re-sync debt AS A SIDE EFFECT. The regression
    guard for a blocker therefore could not fail — measured green on the
    pre-fix tree, which has no retry mechanism at all.

    That species of fixture defect is immune to the usual defences: the
    instrument is live, the mutation lands, and going green on both trees reads
    as "already fixed" rather than "my fixture is curing the bug". The defence
    is a precondition on the precondition — assert the starting state is one
    PRODUCTION CAN REACH, not merely that it was set. ``_assert_reachable``
    below is that assertion, and this handle earns it by serving
    ``display_window(...)``, the exact function ``Session.history_page`` calls,
    over a real transcript. Real rows also make paging genuine (the surfaced
    symptom was "Could not load earlier messages") rather than a window whose
    empty message list makes every paging assertion vacuous.
    """

    #: Enough rows to exceed ``DISPLAY_HISTORY_MESSAGES`` (120), so the first
    #: window is a real PAGE with a ``before_token`` behind it rather than the
    #: whole history — the shape the TUI actually pages through.
    _ROWS = 270

    def __init__(self, directory: Path) -> None:
        super().__init__()
        from local_operator.session.transcript import Transcript

        self.transcript = Transcript(directory)
        self.page_calls: list[str] = []

    async def seed(self) -> None:
        """Commit real rows and point the snapshot cursor at the last of them.

        Separate from ``__init__`` because ``append_messages`` is async. The
        cursor must equal the window's ``through_id`` or the follower refuses
        the window and silently falls back to the legacy path.
        """
        from local_operator.harness.types import Message

        await self.transcript.append_messages(
            [Message.user(f"row {index}") for index in range(self._ROWS)]
        )
        entries = self.transcript.entries()
        assert len(entries) == self._ROWS, "the transcript did not take the rows"
        self._frontend.mutate(history_cursor=entries[-1].id)

    def _window(self, before: str | None = None, anchor: str = "") -> DisplayHistoryWindow:
        from local_operator.session.history_window import display_window

        state = self._frontend.state
        return display_window(
            self.transcript,
            conversation_id=state.session_id,
            owner_epoch=state.epoch,
            through_id=state.history_cursor,
            before=before,
            anchor=anchor,
        )

    def subscribe_frontend(self, on_update, *, display_window=False):  # noqa: ANN001, ANN202
        subscription = self._frontend.subscribe(on_update)
        if display_window:
            subscription.sync.display_history = self._window()
        return subscription

    async def history_page(self, before: str, anchor: str = "") -> dict[str, Any]:
        # Presence of this method is what makes the server advertise the
        # windowed capability. Recorded so a paging assertion can prove the RPC
        # was actually served rather than answered from the resident window.
        self.page_calls.append(before)
        return self._window(before=before, anchor=anchor).model_dump(mode="json")


async def _windowed_handle(tmp_path: Path) -> _WindowedHandle:
    """A seeded windowed handle whose window is canaried before any test uses it.

    A window that came back empty, or as ``reset``, would make every paging and
    staleness assertion downstream pass vacuously.
    """
    handle = _WindowedHandle(tmp_path / "transcript")
    await handle.seed()
    probe = handle._window()
    assert probe.status == "ok", f"window status {probe.status}"
    assert probe.total_message_count == _WindowedHandle._ROWS
    assert probe.messages, "the window carries no rows — the instrument is dead"
    assert probe.before_token, "the window has no before_token — paging is unreachable"
    return handle


def _assert_reachable(remote: RemoteSession) -> None:
    """Refuse a fixture whose starting state PRODUCTION CANNOT REACH.

    Review round 2, Q. A fixture pairing a window generation with a disagreeing
    snapshot generation leaves the follower permanently invalidated, so ordinary
    deltas keep restarting refreshes that repair the very defect the test exists
    to catch — an instrument that does not merely fail to observe but
    ACCIDENTALLY REPAIRS. A real owner serves both numbers from the same
    ``Transcript._history_generation``, so they agree at connect by
    construction; anything else is a state that exists only in the harness.

    Asserted after connect on every follower these tests build, legacy included,
    so the pairing cannot silently drift again.
    """
    snapshot_generation = remote._read_state_field("history_generation")
    assert snapshot_generation == remote._loaded_history_generation, (
        "unreachable fixture: the follower is permanently invalidated, so ordinary "
        f"deltas will repair the staleness under test (snapshot {snapshot_generation} "
        f"!= loaded {remote._loaded_history_generation})"
    )
    assert remote.display_history_current, "unreachable fixture: invalidated at connect"


@pytest.mark.asyncio
async def test_a_windowed_follower_also_resyncs_after_a_degraded_delta(
    tmp_path: Path, monkeypatch
) -> None:
    """The modern windowed viewer must recover too, not just the legacy one.

    Every other test here runs against a handle with no ``history_page``, so
    the follower negotiates full replay. The TUI negotiates the WINDOW, and the
    two paths differ exactly where this fix lives: ``_invalidate_display_history``
    is a no-op for the legacy follower and latches ``_display_invalidated`` for
    the windowed one. A fix proven only on the legacy path is proven on the
    viewer the operator was least likely to be running.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = await _windowed_handle(tmp_path)
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        assert "display-history-window-v1" in record.capabilities
        remote = await RemoteSession.connect(
            record,
            "s1",
            config_dir=tmp_path,
            takeover_factory=_never,
            display_window=True,
        )
        # The whole point of this test: without this the run silently repeats
        # the legacy-path coverage the other tests already have.
        assert remote._display_window_supported, "the follower did not take the windowed path"
        _assert_reachable(remote)
        # Real rows arrived, so the window under test is the paging shape the
        # TUI builds rather than an empty envelope that satisfies the validator.
        assert remote.display_history_window()
        assert remote.history_before_token

        handle._frontend.mutate(
            model_catalogue=[_catalogue_row(index) for index in range(5_000)],
            conversation_title="after the degrade",
        )

        for _ in range(400):
            if remote.frontend_state.conversation_title == "after the degrade":
                break
            await asyncio.sleep(0.02)

        client = remote._client
        assert client is not None and client.connected, "the degraded delta killed the socket"
        owner = handle._frontend.state
        assert remote.frontend_state.conversation_title == owner.conversation_title
        assert remote.frontend_state.sequence == owner.sequence
        assert remote.frontend_state.model_catalogue
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("windowed", [False, True])
async def test_a_failed_resync_is_retried_rather_than_left_permanently_stale(
    tmp_path: Path, monkeypatch, windowed: bool
) -> None:
    """A live socket showing WRONG data is worse than a dead one.

    THE BUG THIS PINS (review round 1, B1). ``_frontend_resync_pending`` is
    cleared before the capture and ``_refresh_display_history`` re-raises on
    failure, so the first version of this fix dropped the debt on the floor: one
    transient ``frontend_sync`` failure left the follower stale FOREVER behind a
    gap check that agreed with the owner at every later sequence. Measured on
    that tree: socket alive, sequence 11/11, follower still showing its
    pre-degrade title through ten subsequent healthy deltas.

    Recovery cannot ride on ``ensure_display_current`` either — that is a TUI
    navigation hook the CLI, the session factory and the desktop bridge never
    call — so the retry has to be the transport's own.

    Run on BOTH follower shapes because the failure's blast radius depends on
    which one the viewer negotiated.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = await _windowed_handle(tmp_path) if windowed else FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record,
            "s1",
            config_dir=tmp_path,
            takeover_factory=_never,
            display_window=windowed,
        )
        assert remote._display_window_supported is windowed
        # The windowed leg used to run on a fixture whose window/snapshot
        # generations disagreed, which permanently invalidated the follower and
        # made every ordinary delta below restart a refresh that repaired the
        # staleness under test. This assertion is what keeps this test able to
        # FAIL (review round 2, Q).
        _assert_reachable(remote)
        client = remote._client
        assert client is not None

        failures = 0
        original = client.frontend_sync

        async def failing_sync() -> Any:
            nonlocal failures
            failures += 1
            raise ConnectionError("injected sync failure")

        monkeypatch.setattr(client, "frontend_sync", failing_sync)
        # ASSERT THE INJECTION LANDED. A harness that silently fails to patch
        # turns this test into a green run of the unmodified path.
        assert client.frontend_sync is failing_sync

        handle._frontend.mutate(
            model_catalogue=[_catalogue_row(index) for index in range(5_000)],
            conversation_title="after the degrade",
        )

        for _ in range(400):
            if failures:
                break
            await asyncio.sleep(0.02)
        # The failure must actually have FIRED, or the retry below proves
        # nothing about the failure path.
        assert failures >= 1, "the injected sync failure never fired"

        # Restore a working owner: the retry, not another degraded frame, is
        # what has to repair this. Ordinary deltas on OTHER fields keep flowing
        # so a follower that merely applies them still fails the assertion.
        monkeypatch.setattr(client, "frontend_sync", original)
        for index in range(10):
            handle._frontend.mutate(cwd=f"/tmp/normal-{index}")
            await asyncio.sleep(0.02)

        # Wait for the END STATE, never for a clock: the retry is a backoff task
        # and the last ordinary delta can still be in flight behind it, so a
        # poll on the title alone reads a moment mid-recovery.
        for _ in range(600):
            if (
                remote.frontend_state.conversation_title == "after the degrade"
                and remote.frontend_state.sequence == handle._frontend.state.sequence
                and not remote._frontend_resync_pending
                and remote._degraded_resync_retry_task is None
            ):
                break
            await asyncio.sleep(0.02)

        owner = handle._frontend.state
        assert client.connected, "the retry cost the socket"
        assert remote.frontend_state.conversation_title == owner.conversation_title, (
            "the follower is permanently stale behind a satisfied gap check: "
            f"{remote.frontend_state.conversation_title!r} != {owner.conversation_title!r}"
        )
        assert remote.frontend_state.sequence == owner.sequence
        assert remote.frontend_state.cwd == "/tmp/normal-9"
        # The debt is settled rather than merely quiet, and no backoff timer is
        # left running behind it.
        assert not remote._frontend_resync_pending
        assert remote._degraded_resync_retry_task is None

        # And the stream is still LIVE after the retry, which is what proves the
        # recovery did not leave the sequence cursor desynced.
        handle._frontend.mutate(conversation_title="after the retry")
        for _ in range(400):
            if remote.frontend_state.conversation_title == "after the retry":
                break
            await asyncio.sleep(0.02)
        assert remote.frontend_state.conversation_title == "after the retry"
        assert client.connected
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("windowed", [False, True])
async def test_a_degrade_folded_into_an_in_flight_refresh_is_still_retried(
    tmp_path: Path, monkeypatch, windowed: bool
) -> None:
    """The debt survives being handed to a refresh THIS PATH DID NOT START.

    THE BUG THIS PINS (review round 2, B2). ``_resync_after_degraded_delta``
    delegates to an already-running refresh instead of starting its own, and the
    retry used to be attached only to the task the degrade path created. A
    refresh started by ``_invalidate_display_history`` carried a LOG-ONLY
    callback, so a degraded frame arriving during it handed its debt to a task
    that re-armed nothing and scheduled no retry — the round-1 defect, one route
    over, with a live socket, a satisfied gap check and permanently stale
    canonical fields. Reachable in production from a ``history_generation`` move
    and from ``CompactionEndEvent``, which is exactly what a long busy session
    (the session shape that emits oversized deltas) does.

    A cached task holding a FAILURE is not a cache, it is a latch. The fix is
    that the retry belongs to the SLOT: whatever fills ``_display_refresh_task``
    carries the debt-aware callback.

    THE TRAP THIS TEST AVOIDS. Pre-populating the slot with an already-completed
    task would take the same early return and pass without ever exercising the
    failure path. So the refresh here is held open on a real gate, the test
    asserts it was genuinely IN FLIGHT when the degrade folded into it, and
    asserts it then genuinely FAILED.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = await _windowed_handle(tmp_path) if windowed else FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record,
            "s1",
            config_dir=tmp_path,
            takeover_factory=_never,
            display_window=windowed,
        )
        assert remote._display_window_supported is windowed
        _assert_reachable(remote)
        client = remote._client
        assert client is not None

        original = client.frontend_sync
        entered = asyncio.Event()
        release = asyncio.Event()
        failures = 0

        async def blocking_failing_sync() -> Any:
            nonlocal failures
            failures += 1
            entered.set()
            # Hold the pass open so the degraded frame below lands while it is
            # genuinely running, which is the whole reachability condition.
            await release.wait()
            raise ConnectionError("injected sync failure")

        monkeypatch.setattr(client, "frontend_sync", blocking_failing_sync)
        assert client.frontend_sync is blocking_failing_sync

        # Start a refresh through the OTHER door — the one whose callback used
        # to only log. The legacy follower's ``_invalidate_display_history``
        # returns early by design, so the flag is lifted just long enough to
        # fill the slot: the point of the legacy leg is the DELEGATION branch in
        # ``_resync_after_degraded_delta``, which is shape-independent.
        supported = remote._display_window_supported
        remote._display_window_supported = True
        remote._invalidate_display_history()
        remote._display_window_supported = supported
        await asyncio.wait_for(entered.wait(), timeout=5)

        in_flight = remote._display_refresh_task
        assert in_flight is not None and not in_flight.done(), (
            "the refresh was not in flight, so this run never reaches the delegating "
            "branch this test exists for"
        )

        handle._frontend.mutate(
            model_catalogue=[_catalogue_row(index) for index in range(5_000)],
            conversation_title="after the degrade",
        )
        # A frame arriving mid-pass is BUFFERED, not applied — that is how it
        # reaches the delegating branch. It is replayed by the failure path
        # below, at which point the failing task still occupies the slot, so
        # ``_resync_after_degraded_delta`` hands it the debt and returns. Waiting
        # on the buffer (never on the clock) is what makes the ordering certain
        # rather than assumed.
        for _ in range(400):
            buffered = remote._pending_frontend_updates
            if buffered and any(update.degraded for update in buffered):
                break
            await asyncio.sleep(0.02)
        buffered = remote._pending_frontend_updates
        assert buffered and any(update.degraded for update in buffered), (
            "the degraded frame never reached the in-flight pass's buffer, so this run "
            "does not exercise the delegating branch this test exists for"
        )
        assert not in_flight.done(), "the pass finished early; the fold never happened"

        release.set()
        await asyncio.gather(in_flight, return_exceptions=True)
        # It genuinely FAILED. A pass that succeeded would settle the debt by
        # doing the work, proving nothing about the failure path.
        assert failures >= 1
        assert in_flight.done() and in_flight.exception() is not None
        # The debt landed on a task THIS PATH DID NOT START and which has now
        # failed: the slot still holds it, so the degrade path took its early
        # return rather than starting a pass of its own. That is the exact state
        # in which the debt used to be orphaned — `pending=True retry=None`.
        assert remote._display_refresh_task is in_flight, (
            "the degrade started its own pass, so the delegating branch — where the "
            "debt was orphaned — was never exercised"
        )
        assert remote._frontend_resync_pending, "the degraded frame never registered a debt"
        assert remote._degraded_resync_retry_task is not None, (
            "the debt was orphaned: the in-flight pass failed with a re-sync owed and "
            "armed no retry"
        )

        # Restore a healthy owner. Nothing else touches the follower: no further
        # deltas, so only the retry can repair this.
        monkeypatch.setattr(client, "frontend_sync", original)

        for _ in range(600):
            if (
                remote.frontend_state.conversation_title == "after the degrade"
                and not remote._frontend_resync_pending
                and remote._degraded_resync_retry_task is None
            ):
                break
            await asyncio.sleep(0.02)

        owner = handle._frontend.state
        assert client.connected, "the retry cost the socket"
        assert remote.frontend_state.conversation_title == owner.conversation_title, (
            "the debt was orphaned in the in-flight refresh: the follower is permanently "
            f"stale at {remote.frontend_state.conversation_title!r}"
        )
        assert remote.frontend_state.sequence == owner.sequence
        # The shed BODY came back, not just the scalar that also rides the
        # ordinary stream — a title alone is a way to misread a stale follower
        # as converged.
        assert remote.frontend_state.model_catalogue
        assert not remote._frontend_resync_pending
        assert remote._degraded_resync_retry_task is None

        # Navigation is usable again rather than latched on a stored error.
        await remote.ensure_display_current()
        if isinstance(handle, _WindowedHandle):
            # The surfaced symptom was "Could not load earlier messages": prove
            # real paging still works, over real transcript rows.
            served = len(handle.page_calls)
            assert remote.history_before_token
            rows = await remote.load_older_display_page()
            assert rows, "paging returned nothing after the recovery"
            assert len(handle.page_calls) > served, "the page was not served by the owner"
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("windowed", [False, True])
async def test_a_quiescent_follower_recovers_without_waiting_for_more_traffic(
    tmp_path: Path, monkeypatch, windowed: bool
) -> None:
    """Recovery must not depend on a delta that may never come.

    THE FINDING THIS PINS (QA round 2, Q3) — the same mechanism as B2 seen from
    its second face. When the debt was orphaned in a refresh started by the
    other path, a follower still LOOKED like it recovered as long as traffic
    kept arriving: ``_display_invalidated`` stayed latched, so the next delta of
    any kind restarted a pass that serviced the orphan as a side effect. A
    genuinely quiet session had nothing to ride on and stayed stale — measured
    at 45 s of quiescence with 0 of 5000 catalogue rows on a live socket at
    sequence 3/3.

    So this test sends NO traffic after the failure, and asserts the owner
    stayed quiet. The retry timer is the only thing that can repair it, which is
    the property Q3 asks for.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = await _windowed_handle(tmp_path) if windowed else FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record,
            "s1",
            config_dir=tmp_path,
            takeover_factory=_never,
            display_window=windowed,
        )
        assert remote._display_window_supported is windowed
        _assert_reachable(remote)
        client = remote._client
        assert client is not None

        original = client.frontend_sync
        failures = 0

        async def failing_sync() -> Any:
            nonlocal failures
            failures += 1
            raise ConnectionError("injected sync failure")

        monkeypatch.setattr(client, "frontend_sync", failing_sync)
        assert client.frontend_sync is failing_sync

        handle._frontend.mutate(
            model_catalogue=[_catalogue_row(index) for index in range(5_000)],
            conversation_title="after the degrade",
        )
        for _ in range(400):
            if failures:
                break
            await asyncio.sleep(0.02)
        assert failures >= 1, "the injected sync failure never fired"

        # THE INTERLEAVING (this is what orphans the debt, not the failure
        # alone). Inside the backoff, an ordinary delta that MOVES
        # ``history_generation`` re-enters ``_invalidate_display_history`` and
        # takes over the refresh slot. When that pass carried a log-only
        # callback, ``_retry_degraded_resync`` deferred to it, the pass failed,
        # and the debt died there — after which only more traffic could repair
        # it.
        assert remote._degraded_resync_retry_task is not None, (
            "no backoff timer to interleave with, so this run cannot reproduce the "
            "hand-off that orphans the debt"
        )
        if windowed:
            # Only the windowed follower can be interleaved with:
            # ``_invalidate_display_history`` early-returns for the legacy one,
            # so the generation move is a no-op there and asserting a takeover
            # would be asserting against a path that does not exist. The legacy
            # leg below still proves quiescent recovery, just from the simpler
            # state where the retry is the only actor.
            taken_over = remote._display_refresh_task
            handle._frontend.mutate(history_generation=1, cwd="/tmp/generation-moved")
            for _ in range(400):
                current = remote._display_refresh_task
                if current is not None and current is not taken_over:
                    break
                await asyncio.sleep(0.02)
            current = remote._display_refresh_task
            # The SLOT genuinely changed hands to a pass this path did not
            # start. A count of injected failures would not prove this — the
            # backoff's own attempts also raise, so it would go green on the
            # broken tree for the wrong reason.
            assert current is not None and current is not taken_over, (
                "the generation move never took over the refresh slot, so the debt was "
                "never handed to the other path"
            )
            await asyncio.gather(current, return_exceptions=True)
            assert current.exception() is not None, (
                "the interleaved pass succeeded, so it settled the debt by doing the "
                "work rather than by orphaning it"
            )

        monkeypatch.setattr(client, "frontend_sync", original)
        # Deliberately no mutations from here on. The owner is quiet, which is
        # the condition under which the orphaned debt was never serviced.
        quiet_sequence = handle._frontend.state.sequence
        for _ in range(600):
            if (
                remote.frontend_state.conversation_title == "after the degrade"
                and not remote._frontend_resync_pending
                and remote._degraded_resync_retry_task is None
            ):
                break
            await asyncio.sleep(0.02)

        assert handle._frontend.state.sequence == quiet_sequence, (
            "the owner emitted deltas during the quiet window, so this run cannot "
            "distinguish the retry from a delta-driven refresh"
        )
        assert client.connected
        assert remote.frontend_state.conversation_title == "after the degrade"
        assert remote.frontend_state.model_catalogue, "the shed body never came back"
        assert not remote._frontend_resync_pending
        assert remote._degraded_resync_retry_task is None
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_a_display_refresh_failure_still_reports_when_nothing_canonical_is_owed(
    tmp_path: Path, monkeypatch
) -> None:
    """Fixing a raise must not remove it: ``ensure_display_current`` still raises.

    Moving the retry onto the slot made ``_invalidate_display_history`` share
    the degrade path's failure callback, so the risk is the mirror of B2 — a
    display-drift failure quietly retried into silence instead of reported to
    navigation, which is how stale rows get painted. It must still surface (the
    fence stays closed, the error reaches the caller) when NO canonical re-sync
    is owed and the owner is genuinely unreachable, and it must still be a
    reporting surface rather than a retrying one.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = await _windowed_handle(tmp_path)
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never, display_window=True
        )
        assert remote._display_window_supported
        _assert_reachable(remote)
        client = remote._client
        assert client is not None

        # A healthy owner with nothing owed RETURNS. Without this the raise
        # below would pass for the wrong reason.
        await remote.ensure_display_current()

        failures = 0
        original = client.frontend_sync

        async def failing_sync() -> Any:
            nonlocal failures
            failures += 1
            raise ConnectionError("injected sync failure")

        monkeypatch.setattr(client, "frontend_sync", failing_sync)
        remote._invalidate_display_history()
        task = remote._display_refresh_task
        assert task is not None
        await asyncio.gather(task, return_exceptions=True)
        assert failures >= 1, "the injected sync failure never fired"

        # No canonical debt was ever armed, so this is pure display drift.
        assert not remote._frontend_resync_pending
        # No retry timer either: this surface reports, and one timer per failed
        # navigation would be an RPC stream the operator never asked for.
        assert remote._degraded_resync_retry_task is None
        assert not remote.display_history_current, "the invalidation fence was left open"
        with pytest.raises(ConnectionError):
            await remote.ensure_display_current()

        # The raise is the CONTRACT here, so it must survive the owner coming
        # back: `ensure_display_current` awaits the stored task first, so this
        # surface stays latched on the failed pass until something invalidates
        # again. That is pre-existing behaviour on the display-drift path —
        # unchanged by moving the retry onto the slot — and it is what makes the
        # canonical debt's own retry necessary rather than optional.
        monkeypatch.setattr(client, "frontend_sync", original)
        with pytest.raises(ConnectionError):
            await remote.ensure_display_current()
        # A fresh invalidation is the documented way out, and it works.
        remote._invalidate_display_history()
        refreshed = remote._display_refresh_task
        assert refreshed is not None and refreshed is not task
        await refreshed
        assert remote.display_history_current
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


def test_a_malformed_non_degraded_update_is_refused_rather_than_silently_applied():
    """Relaxing ``changes`` must not extend past the frame that needs it.

    THE FINDING THIS PINS (review round 1, M2). ``changes`` was made optional
    for every update, and the model allows extras — so a frame that misspelled
    the key validated as an empty change set, applied as "nothing moved",
    consumed a sequence and kept the gap check happy with no log and no
    ``degraded`` flag. That is the same silent drift the degraded frame causes,
    reintroduced from a different cause: a loud failure traded for a quiet one.

    Refusing the frame is not the same as tearing down the socket over a
    degraded one — the degrade path labels its own frames, so what is rejected
    here is a frame claiming to be a complete delta while carrying no body,
    which no correct owner emits.
    """
    from pydantic import ValidationError

    # The reviewer's exact frame.
    with pytest.raises(ValidationError, match="only a degraded frame may shed its body"):
        FrontendUpdate.model_validate(
            {"epoch": "e1", "sequence": 6, "chnages": {"streaming": True}}
        )

    # And the same shape with the key simply absent.
    with pytest.raises(ValidationError, match="only a degraded frame may shed its body"):
        FrontendUpdate.model_validate({"epoch": "e1", "sequence": 6})

    # A DEGRADED frame may still omit it — the fix must survive its own guard.
    degraded = FrontendUpdate.model_validate(
        {"epoch": "e1", "sequence": 6, "degraded": True, "degraded_reason": "too large"}
    )
    assert degraded.changes == {}

    # An EXPLICIT empty change set is still legal: "nothing moved" is a
    # different statement from "the body was shed", which is why the flag
    # exists rather than being inferred from emptiness.
    empty = FrontendUpdate.model_validate({"epoch": "e1", "sequence": 6, "changes": {}})
    assert empty.changes == {}
    assert empty.degraded is False

    # An ordinary delta is untouched.
    ordinary = FrontendUpdate.model_validate(
        {"epoch": "e1", "sequence": 7, "changes": {"conversation_title": "t"}}
    )
    assert ordinary.changes == {"conversation_title": "t"}
    assert ordinary.degraded is False


def test_a_refused_update_never_reaches_the_store_or_consumes_a_sequence():
    """The refusal's POINT is that no sequence is spent on a frame nobody read.

    A silently-applied malformed frame is worse than a rejected one precisely
    because it advances the cursor: the follower then agrees with the owner
    about sequencing forever while disagreeing about content. The receiver
    refuses the frame instead, which the transport already recovers from by
    re-syncing.
    """
    from pydantic import ValidationError

    store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", conversation_title="before")
    )
    start = store.state.sequence

    with pytest.raises(ValidationError):
        store.apply_update(
            FrontendUpdate.model_validate(
                {"epoch": "e1", "sequence": start + 1, "chnages": {"conversation_title": "typo"}}
            )
        )

    assert store.state.sequence == start
    assert store.state.conversation_title == "before"


def _oversized_event_frame() -> dict[str, Any]:
    """An ``event`` frame the size the operator's real transcript produced.

    A single 1,039,374-byte transcript row serialized to a 1,129,319-byte
    ``event`` frame — past the 1 MiB line limit both sides enforce.
    """
    from local_operator.session.runtime.server import _MAX_LINE_BYTES

    return {
        "op": "event",
        "data": {
            "type": "message_update",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "x" * (_MAX_LINE_BYTES + 80_000)}],
            },
            "delta": "x",
        },
    }


def _live_end(call_id: str, *, text: str) -> dict[str, Any]:
    return {
        "type": "tool_execution_end",
        "tool_call_id": call_id,
        "tool_name": "read",
        "is_error": False,
        "result": {
            "tool_call_id": call_id,
            "tool_name": "read",
            "content": [{"type": "text", "text": text}],
            "is_error": False,
        },
    }


def test_oversized_event_frame_degrades_instead_of_killing_the_socket():
    """``event`` is guarded the way ``frontend_sync`` already was.

    An oversized line makes the client's ``readline`` raise
    ``LimitOverrunError``, and the overrun does NOT consume the buffer, so every
    later read re-raises: the pump dies and the viewer paints "owner sent a
    frame too large to read". Degrading one delta is safe — the viewer recovers
    through ``frontend_sync`` plus durable history — while killing the socket
    is not.
    """
    from local_operator.session.remote import deserialize_event
    from local_operator.session.runtime.server import (
        _MAX_LINE_BYTES,
        relay_frame_or_degraded,
    )

    frame = _oversized_event_frame()
    assert len(json.dumps(frame).encode()) + 1 > _MAX_LINE_BYTES

    sendable = relay_frame_or_degraded(frame, _MAX_LINE_BYTES)

    # It now fits, so the client can read it...
    assert len(json.dumps(sendable).encode()) + 1 <= _MAX_LINE_BYTES
    assert sendable["op"] == "event"
    # ...and it is still a VALID event: the client deserializes every relayed
    # payload, and a bare marker would raise there and be silently swallowed.
    event = deserialize_event(sendable["data"])
    assert event.type == "notice"


def test_oversized_frontend_update_keeps_its_sequence():
    """Canonical deltas are not replacement-safe, so the placeholder is not one.

    The client closes the connection on an ``epoch``/``sequence`` gap by design.
    A placeholder that dropped those fields would trip that check and kill the
    connection this guard exists to save, so only the oversized BODY is shed.
    """
    from local_operator.session.runtime.server import (
        _MAX_LINE_BYTES,
        relay_frame_or_degraded,
    )

    frame = {
        "op": "frontend_update",
        "data": {
            "epoch": "epoch-1",
            "sequence": 42,
            "snapshot": {"jobs": "y" * (_MAX_LINE_BYTES + 10_000)},
        },
    }

    sendable = relay_frame_or_degraded(frame, _MAX_LINE_BYTES)

    assert len(json.dumps(sendable).encode()) + 1 <= _MAX_LINE_BYTES
    assert sendable["data"]["epoch"] == "epoch-1"
    assert sendable["data"]["sequence"] == 42
    assert sendable["data"]["degraded"] is True


def test_a_degraded_frontend_update_still_validates_as_one():
    """The degrade path's own frame must satisfy the model the follower parses.

    THE BUG THIS PINS. ``changes`` was a REQUIRED field, so the stand-in built
    by ``relay_frame_or_degraded`` — which sheds the body and keeps only
    sequencing — raised ``ValidationError`` inside the follower's
    ``_on_frontend_update``. ``AttachClient._pump``'s catch-all turned that into
    ``owner frame could not be applied: ...`` and tore the socket down, so the
    guard written specifically to avoid killing the connection killed it one
    layer further down. The operator saw exactly this string.
    """
    from local_operator.session.runtime.server import (
        _MAX_LINE_BYTES,
        relay_frame_or_degraded,
    )

    frame = {
        "op": "frontend_update",
        "data": {
            "epoch": "epoch-1",
            "sequence": 42,
            "changes": {"cwd": "y" * (_MAX_LINE_BYTES + 10)},
        },
    }

    sendable = relay_frame_or_degraded(frame, _MAX_LINE_BYTES)

    # The exact call the follower makes. Before the fix this raised.
    update = FrontendUpdate.model_validate(sendable["data"])
    assert update.degraded is True
    assert update.degraded_reason
    # Empty rather than absent, so `payload["changes"]` readers keep working.
    assert update.changes == {}
    assert update.model_dump(mode="json")["changes"] == {}


def test_a_degraded_delta_advances_the_sequence_without_touching_fields():
    """Applying a shed body as an empty change set would be a silent corruption.

    The owner consumed the sequence, so it must advance here or every later
    delta is refused as a gap. But the fields the frame carried are UNKNOWN,
    not unchanged — writing an empty change set over canonical state and
    calling it applied is what leaves a follower permanently, quietly wrong.
    """
    store = FrontendStateStore(
        FrontendSessionState(session_id="s1", epoch="e1", conversation_title="before")
    )
    seen: list[FrontendUpdate] = []
    store.subscribe(seen.append)
    start = store.state.sequence

    state = store.apply_update(
        FrontendUpdate(
            epoch="e1",
            sequence=start + 1,
            degraded=True,
            degraded_reason="too large",
        )
    )

    assert state.sequence == start + 1
    assert state.conversation_title == "before"
    # Published, so an in-process subscriber and the resync path agree that a
    # delta was shed rather than one of them believing state is complete.
    assert [u.degraded for u in seen] == [True]


def test_ordinary_relay_frames_pass_through_untouched():
    """The guard is on the hot path, so the common case must not rewrite."""
    from local_operator.session.runtime.server import (
        _MAX_LINE_BYTES,
        relay_frame_or_degraded,
    )

    frame = {"op": "event", "data": {"type": "notice", "text": "hello", "kind": "note"}}
    assert relay_frame_or_degraded(frame, _MAX_LINE_BYTES) is frame


@pytest.mark.asyncio
async def test_compaction_never_assembles_an_unreadable_frame() -> None:
    """Merging individually-legal frames must not produce an oversized one.

    The guard runs at ``_enqueue_client_frame``, but ``_compact_event_queue``
    runs AFTER it and is the one operation that makes a frame bigger than
    anything the guard was shown: it concatenates ``delta`` fields and
    re-queues the result, which ``_send_to`` then writes with no size re-check.
    Measured on the real path, 20 frames of 908,157 B — every one legal on its
    own — merged to 1,060,157 B and killed a real pump with the precise failure
    the relay guard exists to prevent.

    The merge is refused rather than degraded because refusing is lossless:
    both frames are individually sendable.
    """
    from local_operator.session.runtime.server import _EVENT_QUEUE_MAX

    conn = SimpleNamespace(event_queue=asyncio.Queue(maxsize=_EVENT_QUEUE_MAX))

    # Ordinary streaming shape: a near-limit accumulated message with small
    # deltas, which is exactly the burst compaction exists to absorb.
    chunk = "z" * 1024
    # Near the limit but under it: a long assistant message still streaming.
    # Each frame is legal; 64 concatenated deltas are what push the merge over.
    accumulated = "y" * 1_020_000
    for _ in range(_EVENT_QUEUE_MAX):
        conn.event_queue.put_nowait(
            {
                "op": "event",
                "data": {
                    "type": "message_update",
                    "message": {"id": "m1", "role": "assistant", "text": accumulated},
                    "delta": chunk,
                },
            }
        )

    # Precondition: every input frame is individually legal, so the guard at
    # the chokepoint passes all of them and cannot be what saves us here.
    for frame in list(conn.event_queue._queue):
        assert len(json.dumps(frame).encode()) + 1 <= _MAX_LINE_BYTES

    before = "".join(str(f["data"]["delta"]) for f in list(conn.event_queue._queue))

    server = RuntimeServer.__new__(RuntimeServer)
    server._compact_event_queue(cast(Any, conn))

    # Whatever came out, every frame must be readable by the client.
    out = list(conn.event_queue._queue)
    assert out, "compaction must not empty the queue"
    for frame in out:
        size = len(json.dumps(frame).encode()) + 1
        assert size <= _MAX_LINE_BYTES, f"compaction emitted an unreadable {size}-byte frame"

    # LOSSLESSNESS is the property that makes refusing (rather than degrading)
    # defensible, and without this assertion the test above still passes
    # against a compaction that keeps the first frame and discards 63 deltas.
    # The delta stream is append-only, so concatenating it must be unchanged.
    assert "".join(str(f["data"]["delta"]) for f in out) == before


def _live_start(call_id: str) -> dict[str, Any]:
    return {"type": "tool_execution_start", "tool_call_id": call_id, "tool_name": "read"}


def _bounded_live_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run rows through the real wire boundary and hand back what survived."""
    state = FrontendSessionState(session_id="s1", epoch="e1", live_events=rows)
    payload = sync_wire_payload(
        FrontendSync(epoch=state.epoch, sequence=state.sequence, snapshot=state, live_cursor=None)
    )
    return payload["snapshot"]["live_events"]


def test_a_retained_tool_end_keeps_what_settles_the_card() -> None:
    """Truncation may take the payload, never the identity or the outcome.

    The seed exists so a viewer that reconnects mid-turn can settle a card for
    a call that finished while it was away. A bound that dropped ``is_error``
    or the id would leave the card live and hand it back to the retirement
    pass as ``⊘ interrupted`` — the artefact the retention removes.
    """
    survivors = _bounded_live_events([_live_start("c1"), _live_end("c1", text="R" * 500_000)])

    ends = [row for row in survivors if row["type"] == "tool_execution_end"]
    assert len(ends) == 1
    assert ends[0]["tool_call_id"] == "c1"
    assert ends[0]["is_error"] is False
    text = ends[0]["result"]["content"][0]["text"]
    assert text.endswith("…"), "a clipped preview must be marked as clipped"
    # A lone row gets the WHOLE frame budget; the floor is the guarantee for a
    # turn of many calls, not a ceiling for one.
    assert len(text) <= LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS + 1
    assert text.startswith("RRR"), "the surviving preview must be the head of the result"


def test_the_text_floor_holds_when_a_turn_has_very_many_calls() -> None:
    """Every retained end stays legible however many calls the turn ran.

    A share divided by call count alone would shrink to nothing on a long
    turn, leaving cards that settle with an empty result. The floor is what
    keeps a clipped card readable rather than merely present.
    """
    rows: list[dict[str, Any]] = []
    for index in range(LIVE_EVENT_END_ROWS_MAX):
        rows.append(_live_end(f"c{index}", text="y" * 50_000))

    survivors = _bounded_live_events(rows)
    lengths = [len(row["result"]["content"][0]["text"]) for row in survivors]

    assert len(lengths) == LIVE_EVENT_END_ROWS_MAX
    assert min(lengths) >= LIVE_EVENT_TEXT_FLOOR_CHARS


def test_an_evicted_tool_end_takes_its_start_with_it() -> None:
    """Never leave a start whose end was dropped: that is a stranded spinner.

    Evicting the end alone would leave a card the viewer paints live and can
    never settle, which is the same ``⊘ interrupted`` outcome by another route.
    A start with NO end is a call still running and must always survive.
    """
    rows: list[dict[str, Any]] = []
    for index in range(LIVE_EVENT_END_ROWS_MAX + 50):
        rows.append(_live_start(f"done-{index}"))
        rows.append(_live_end(f"done-{index}", text="x" * 100))
    rows.append(_live_start("still-running"))

    survivors = _bounded_live_events(rows)
    ends = {row["tool_call_id"] for row in survivors if row["type"] == "tool_execution_end"}
    starts = {row["tool_call_id"] for row in survivors if row["type"] == "tool_execution_start"}

    assert len(ends) == LIVE_EVENT_END_ROWS_MAX
    # Newest kept: those are the cards most likely still on screen unsettled.
    assert "done-149" in ends and "done-0" not in ends
    # No start outlives its own end...
    assert starts - ends == {"still-running"}
    # ...and the call that never ended keeps its card.
    assert "still-running" in starts


@pytest.mark.asyncio
async def test_attach_succeeds_mid_turn_against_an_owner_with_a_heavy_seed(
    tmp_path: Path, monkeypatch
) -> None:
    """End to end over a real socket: a heavy in-flight seed still attaches.

    The serialization tests above measure the frame; this one proves the claim
    that matters. A viewer reconnecting into a long turn is the exact case the
    retained ``tool_execution_end`` was added for, so it is the case that must
    not be broken by the retention's own weight: without the wire bound this
    ``frontend_sync`` runs to megabytes, the reader refuses the line, and the
    connect degrades to the cold session the whole PR exists to prevent.

    The surviving seed is asserted to be USABLE, not merely present — each end
    keeps the id and outcome ``on_tool_ended`` needs to settle its card.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    seed: list[dict[str, Any]] = []
    for index in range(600):
        seed.append(_live_start(f"call-{index}"))
        # Every payload shape a real turn produces, not just text: an image
        # block alone is over the line limit, and `details` is what the MCP
        # bridge fills. A seed of pure text cannot prove the socket survives.
        end = _live_end(f"call-{index}", text="R" * 20_000)
        end["result"]["content"].append(
            {"type": "image", "data": "A" * 1_400_000, "mime_type": "image/png"}
        )
        end["result"]["details"] = {"server_result": {"blob": "D" * 50_000}}
        seed.append(end)
    handle._frontend.mutate(jobs=_jobs(200, 500), live_events=seed)

    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        # Connecting at all is the assertion the bound exists to protect.
        live = remote.frontend_state.live_events
        ends = [row for row in live if row.get("type") == "tool_execution_end"]
        assert 0 < len(ends) <= LIVE_EVENT_END_ROWS_MAX
        # Newest survive: those are the cards still unsettled on screen.
        assert ends[-1]["tool_call_id"] == "call-599"
        # And every survivor can still settle its card.
        assert all(row["tool_call_id"] and "is_error" in row for row in ends)
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


def _live_image_end(call_id: str, *, b64: str) -> dict[str, Any]:
    """A completed `read` of a PNG — the shape with no ``text`` key at all."""
    return {
        "type": "tool_execution_end",
        "tool_call_id": call_id,
        "tool_name": "read",
        "is_error": False,
        "result": {
            "tool_call_id": call_id,
            "tool_name": "read",
            "content": [{"type": "image", "data": b64, "mime_type": "image/png"}],
            "is_error": False,
        },
    }


def test_an_image_result_cannot_ride_the_seed_verbatim() -> None:
    """A block with no ``text`` key must still be bounded.

    ``ImageContent`` carries base64 under ``data``. One permitted image is
    ~1.4 MB encoded — larger than the whole socket line limit on its own — so
    a bound that inspects ``text`` does not merely clip it badly, it never
    sees it. The row survives (it is what settles the card) and the block is
    replaced by a marker rather than deleted, so the card does not read as a
    tool that returned nothing.
    """
    survivors = _bounded_live_events([_live_image_end("img", b64="A" * 1_400_000)])

    end = survivors[0]
    assert end["tool_call_id"] == "img" and end["is_error"] is False
    block = end["result"]["content"][0]
    assert LIVE_EVENT_BLOCK_ELIDED_PLACEHOLDER in block["text"]
    assert "A" * 1_000 not in json.dumps(end), "the base64 payload must not reach the wire"
    assert len(json.dumps(end)) < LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS


def test_fat_result_details_cannot_ride_the_seed_verbatim() -> None:
    """``result.details`` is bounded too — the MCP bridge fills it.

    ``details`` holds the bridge's whole ``server_result`` dump and is not
    rendered on the card, so it is the cheapest thing to shed and the first
    thing dropped. 50 such calls measured 2.6 MB before this.
    """
    row = _live_end("mcp", text="ok")
    row["result"]["details"] = {"server_result": {"blob": "D" * 500_000}}

    end = _bounded_live_events([row])[0]

    assert end["result"]["details"] is None
    assert "D" * 1_000 not in json.dumps(end)
    # The card still settles and still shows what the tool said.
    assert end["result"]["content"][0]["text"] == "ok"


def test_a_small_ordinary_result_is_left_exactly_alone() -> None:
    """The bound is a ceiling, not a rewrite: the common row must be untouched.

    Worth asserting because every other test here drives the clipping paths;
    without this, a bound that mangled ordinary results would look correct.
    """
    row = _live_end("ok", text="3 files changed")
    row["result"]["details"] = {"added": 3, "removed": 1}

    end = _bounded_live_events([row])[0]

    assert end["result"]["content"][0]["text"] == "3 files changed"
    assert end["result"]["details"] == {"added": 3, "removed": 1}


def test_the_headline_of_a_clipped_result_survives_beside_an_image() -> None:
    """A mixed result keeps its text preview AND settles, in one row.

    The realistic shape after a browser screenshot or an image ``read``: text
    the user wants to read, plus a payload that cannot ride. Both paths run on
    the same row here, which is what the round-3 defect got wrong.
    """
    row = _live_end("mixed", text="HEADLINE. " + "z" * 300_000)
    row["result"]["content"].append(
        {"type": "image", "data": "B" * 1_400_000, "mime_type": "image/png"}
    )
    row["result"]["details"] = {"server_result": {"blob": "D" * 200_000}}

    end = _bounded_live_events([row])[0]
    text_block, image_block = end["result"]["content"]

    assert text_block["text"].startswith("HEADLINE. ")
    assert text_block["text"].endswith("…")
    assert LIVE_EVENT_BLOCK_ELIDED_PLACEHOLDER in image_block["text"]
    assert end["result"]["details"] is None
    assert len(json.dumps(end)) < LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS + 1_000


def test_block_count_is_bounded_the_way_block_size_is() -> None:
    """Many blocks in one result must not multiply the row's share.

    The floor is a promise that the CARD stays legible, so it is spent once
    per row. Granted per block it becomes an entitlement instead of a ceiling:
    N blocks cost N x floor and the row has no bound at all. That regression
    was introduced once by moving the floor one loop level inward during a
    rewrite, and it measured 1,102,743 B at the 100-row cap against a
    1,048,576-byte limit while the per-row form stayed flat.

    Pinned here because ``content`` is typed ``list[dict[str, Any]]`` and this
    module deliberately does not enumerate what rides in it — a producer that
    starts emitting many blocks is a change in DATA, which no test of block
    SIZE would catch.
    """
    row = _live_end("many", text="a" * 5_000)
    row["result"]["content"] = [{"type": "text", "text": "a" * 5_000} for _ in range(200)]

    end = _bounded_live_events([row])[0]
    cost = len(json.dumps(end))

    # The whole row lands within one share plus its own envelope — NOT within
    # 200 shares. The exact ceiling is not the assertion; not scaling with
    # block count is.
    assert cost < LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS * 2
    # The TEXT the row carries is one share's worth in total, spent in order
    # until it runs out. Asserting the sum rather than any per-block length is
    # the point: the defect was that each block could re-claim the floor, so
    # the total is what distinguishes a shared ceiling from N entitlements.
    total_text = sum(len(block.get("text") or "") for block in end["result"]["content"])
    assert total_text <= LIVE_EVENT_TEXT_FRAME_BUDGET_CHARS
    # Blocks past the budget are clipped to nothing rather than each keeping a
    # floor's worth. Under the per-block floor all 200 sat at or above it.
    starved = [
        block
        for block in end["result"]["content"]
        if len(block.get("text") or "") < LIVE_EVENT_TEXT_FLOOR_CHARS
    ]
    assert len(starved) > 150, "the floor was granted per block, not per row"
    # ...and the row still settles its card.
    assert end["tool_call_id"] == "many" and end["is_error"] is False


def test_the_row_cost_of_many_blocks_does_not_grow_with_their_number() -> None:
    """The structural invariant behind the test above, stated as a comparison.

    A ceiling that holds at 10 blocks and quietly scales at 2,000 is the
    defect this pins; comparing the two is what distinguishes a real bound
    from a number that merely happened to fit.
    """
    costs = []
    for count in (10, 2_000):
        row = _live_end(f"n{count}", text="a" * 5_000)
        row["result"]["content"] = [{"type": "text", "text": "a" * 5_000} for _ in range(count)]
        costs.append(len(json.dumps(_bounded_live_events([row])[0])))

    ten, many = costs
    # 200x the blocks must not mean anything like 200x the bytes. Only the
    # extra blocks' fixed envelopes may grow; their TEXT comes out of the one
    # shared budget, so the ratio stays near 1 rather than tracking the count.
    # Under the per-block floor this grew without limit, which is why the
    # COMPARISON is the assertion and neither number alone would serve.
    assert many < ten * 3, f"row cost scaled with block count: {ten} -> {many}"


def test_the_elided_marker_separates_itself_from_a_caption() -> None:
    """The marker must not run into the text of the block before it.

    ``ToolResult.text`` joins content blocks with ``""``, so a caption
    followed by a shed image rendered as
    ``screenshot of the page[dropped from the reconnect snapshot…]`` — the
    marker read as part of what the tool said (round-4 Q4-2, caught in a
    rendered frame). Asserted through the real ``ToolResult.text`` rather
    than on the block dict, because the join is where the defect lived.
    """
    row = _live_end("shot", text="screenshot of the page")
    row["result"]["content"].append(
        {"type": "image", "data": "A" * 1_400_000, "mime_type": "image/png"}
    )

    end = _bounded_live_events([row])[0]
    rendered = ToolResult.model_validate(end["result"]).text

    assert "page\n[dropped" in rendered
    assert "page[dropped" not in rendered


def test_a_lone_elided_marker_does_not_open_with_a_blank_line() -> None:
    """The separator is conditional: nothing before it means nothing to separate.

    An unconditional leading newline would open the card with an empty row
    for the common single-image result, trading one cosmetic defect for
    another.
    """
    end = _bounded_live_events([_live_image_end("img", b64="A" * 1_400_000)])[0]
    rendered = ToolResult.model_validate(end["result"]).text

    assert rendered == LIVE_EVENT_BLOCK_ELIDED_PLACEHOLDER
    assert not rendered.startswith("\n")


def test_derived_state_labels_are_real_and_immutable() -> None:
    """The copy-free LABEL list must name real properties returning fresh strings.

    The sibling of ``test_shareable_state_fields_are_real_and_immutable`` for
    ``read_label``, and it exists because that guard cannot see these: it is
    driven off ``model_fields``, and a derived label is a PROPERTY, so a label
    added to ``_DERIVED_STATE_LABELS`` is invisible to every assertion above.

    Two properties are checked, and both are the safety argument rather than
    tidiness. A name that is not a real property would pass the membership gate
    and then raise ``AttributeError`` on a per-frame path — the exact shape
    round 2 caught in the field allow-list (``conversation_name`` for
    ``conversation_title``). And a property returning one of the state's OWN
    objects would share it with none of the field allow-list's scrutiny, which
    is precisely what keeping the specs off that list exists to prevent.
    """
    unknown = [
        name for name in sorted(_DERIVED_STATE_LABELS) if not hasattr(FrontendSessionState, name)
    ]
    assert not unknown, (
        f"copy-free label(s) that are not properties of FrontendSessionState: {unknown}. "
        "`read_label` checks membership BEFORE `getattr`, so a name that is not real "
        "passes the guard and then raises AttributeError on a per-frame path."
    )

    for name in sorted(_DERIVED_STATE_LABELS):
        attribute = getattr(FrontendSessionState, name)
        assert isinstance(attribute, property), (
            f"{name!r} is not a derived property. `read_label` exists for values COMPUTED "
            "per read; a stored field belongs in _SHAREABLE_STATE_FIELDS, where the "
            "immutability guard can see it."
        )

    # A fresh value per read is what makes sharing safe here: the caller holds
    # a string the property just built, never the spec it was built from.
    state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        selected_model=FrontendModelSpec(provider="openai", model_id="gpt-4o"),
        effective_model=FrontendModelSpec(provider="anthropic", model_id="claude"),
    )
    store = FrontendStateStore(state)
    for name in sorted(_DERIVED_STATE_LABELS):
        value = store.read_label(name)
        assert isinstance(value, str), f"{name!r} must return a str, got {type(value)!r}"


def test_read_label_refuses_a_name_that_is_not_a_derived_label() -> None:
    """The gate is enforced, not documented — including against real properties.

    ``FrontendSessionState`` has other properties, and a future one could
    return a mutable object. Membership rather than "is it a property" is what
    keeps such a value off this path by default.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))

    with pytest.raises(KeyError):
        store.read_label("jobs")
    with pytest.raises(KeyError):
        store.read_label("selected_model")
    with pytest.raises(KeyError):
        store.read_label("not_a_property_at_all")


def test_read_label_refuses_a_label_that_is_not_a_str() -> None:
    """An allow-listed name whose value is not a `str` must FAIL, not stringify.

    Review round 1 (MINOR): the boundary used to read
    ``return str(getattr(...))``, so a label property whose return type drifted
    — a tuple-shaped label rendering ``(a, b)`` — passed silently. The
    coercion could never share state (``str()`` always builds fresh); what it
    hid was the CONTRACT: a non-str is exactly the shape the allow-list's
    closure test exists to catch, and the boundary is where a drift it missed
    has to stop. Now a `TypeError`, the label sibling of ``read_field``'s
    ``KeyError`` on a non-allowlisted name.

    Reached by swapping the store's state for one whose label is not a str,
    which exercises the boundary directly without mutating the pydantic model
    class behind it.
    """
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))
    store._state = SimpleNamespace(model_label=("openai", "gpt-4o"))  # type: ignore[assignment]

    with pytest.raises(TypeError):
        store.read_label("model_label")

    # The boundary still serves a genuine str from a genuine state.
    store._state = FrontendSessionState(
        session_id="s1",
        epoch="e1",
        selected_model=FrontendModelSpec(provider="openai", model_id="gpt-4o"),
    )
    assert store.read_label("model_label") == "openai/gpt-4o"


def _spec_state() -> FrontendSessionState:
    """A state carrying both specs, a roster and a usage row."""
    return FrontendSessionState(
        session_id="s1",
        epoch="e1",
        conversation_title="a named conversation",
        goal="ship it",
        active_agent="coder",
        active_team="lopdev",
        history_generation=4,
        selected_model=FrontendModelSpec(provider="openai", model_id="gpt-4o"),
        effective_model=FrontendModelSpec(provider="anthropic", model_id="claude"),
        last_usage=FrontendUsage(input_tokens=10, output_tokens=5),
        jobs=[JobState(id="j1", type="task", status="running")],
    )


def _remote_with_state(tmp_path: Path, state: FrontendSessionState) -> RemoteSession:
    """A viewer bound to a REAL store, with no socket.

    The accessors under test read only the store, so a live connection would
    add a fixture without adding coverage.
    """
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=_never)
    remote._frontend_store = FrontendStateStore(state)
    return remote


def test_copy_free_accessors_return_what_the_clone_returned(tmp_path: Path) -> None:
    """Converting an accessor to the copy-free path must not change its answer.

    The whole value of the conversion is that it is invisible to callers: the
    band paints the same string, the refresh compares the same epoch. Asserted
    against ``frontend_state`` — the clone the accessors used to read — so this
    is a direct equivalence rather than a restatement of the new code.
    """
    state = _spec_state()
    remote = _remote_with_state(tmp_path, state)
    snapshot = remote.frontend_state

    assert remote.model_label == snapshot.model_label == "openai/gpt-4o"
    assert remote.effective_model_label == snapshot.effective_model_label == "anthropic/claude"
    assert remote.goal == snapshot.goal == "ship it"
    assert remote.conversation_name == snapshot.conversation_title == "a named conversation"
    assert remote.active_agent == snapshot.active_agent == "coder"
    assert remote.active_team_name == snapshot.active_team == "lopdev"
    assert remote.epoch == snapshot.epoch == "e1"
    assert remote._read_state_field("history_generation") == snapshot.history_generation == 4


def test_copy_free_accessors_cannot_corrupt_the_store(tmp_path: Path) -> None:
    """A caller mutating a returned value must not reach canonical state.

    This is the invariant the whole-state clone provided and the reason the
    allow-list is a safety argument rather than a convenience list. Every
    converted accessor returns either an immutable scalar or a freshly built
    string, so the mutation a caller CAN perform is rebinding its own local
    name — which is asserted here by re-reading the store afterwards.
    """
    remote = _remote_with_state(tmp_path, _spec_state())

    label = remote.model_label
    label += "/tampered"
    effective = remote.effective_model_label
    effective += "/tampered"
    goal = remote.goal
    goal += " and then some"

    assert remote.model_label == "openai/gpt-4o"
    assert remote.effective_model_label == "anthropic/claude"
    assert remote.goal == "ship it"

    # The specs themselves must never leave the store through a label read: a
    # label that handed back the spec's own string would still be safe (str is
    # immutable), but a label read must not be a route to the OBJECT.
    store = remote._frontend_store
    assert store is not None
    canonical = store._state.selected_model
    assert canonical is not None
    assert canonical.model_id == "gpt-4o"


def test_model_and_usage_accessors_still_deep_copy(tmp_path: Path) -> None:
    """The accessors NOT converted must keep the clone that protects them.

    ``model``/``effective_model`` hand out a non-frozen ``ModelSpec`` and
    ``restored_usage`` a ``Usage`` the harness accumulates in place with ``+=``
    (``harness/jobs.py``, ``harness/subagent.py``). Sharing either instance is
    the invariant loss review round 2 (Q6/F4) rejected, so these must remain on
    the copying path even though their sibling label accessors no longer are.

    Asserted by MUTATING what they return and re-reading the store, which fails
    on a shared instance and passes on a copy — the property itself rather than
    the implementation detail that currently provides it.
    """
    remote = _remote_with_state(tmp_path, _spec_state())
    store = remote._frontend_store
    assert store is not None

    spec = remote.model
    spec.model_id = "tampered"
    assert store._state.selected_model is not None
    assert store._state.selected_model.model_id == "gpt-4o"
    assert remote.model.model_id == "gpt-4o"

    effective = remote.effective_model
    effective.model_id = "tampered"
    assert store._state.effective_model is not None
    assert store._state.effective_model.model_id == "claude"

    usage = remote.restored_usage()
    assert usage is not None
    usage.input_tokens += 1_000_000
    canonical_usage = store._state.last_usage
    assert canonical_usage is not None
    assert canonical_usage.input_tokens == 10

    # And the label accessors keep reporting canonical state after all of that,
    # which is the pairing that matters: the fast path must not be a way to
    # observe a corruption the slow path prevented.
    assert remote.model_label == "openai/gpt-4o"
    assert remote.effective_model_label == "anthropic/claude"


# ---------------------------------------------------------------------------
# The INBOUND twin: a frame the CLIENT sends that the owner cannot read.
#
# Everything above guards frames travelling owner -> viewer. The same 1 MiB
# line limit applies to the other direction and had no protection at all: the
# runtime's reader loop called `readline()` OUTSIDE the `try` that catches
# `ValueError`, so an over-limit inbound line escaped through the reader task
# and the `finally` dropped the connection. One pasted screenshot over ~780 KB
# of source was enough, which the operator experienced as "sending a message
# with an image exits lop and I have to run `lop --resume`".
# ---------------------------------------------------------------------------


def _png_bytes(width: int, height: int) -> bytes:
    """A PNG of CONTINUOUS-TONE content, which is what makes it a real fixture.

    The content matters more than the size here. Pure noise and a flat fill are
    both pathological for the ladder under test — noise defeats every codec and
    a flat fill compresses to nothing, so either one tests the fixture rather
    than the refit. Blurred noise behaves the way a photograph or a screenshot
    does: PNG stores it badly (no flat runs to pack) and JPEG stores it well,
    which is the whole premise of preferring a re-encode over lost pixels.
    """
    Image = pytest.importorskip("PIL.Image")
    ImageFilter = pytest.importorskip("PIL.ImageFilter")
    rng = random.Random(width * height)
    coarse = Image.frombytes(
        "RGB",
        (width // 4, height // 4),
        bytes(rng.getrandbits(8) for _ in range((width // 4) * (height // 4) * 3)),
    )
    image = coarse.resize((width, height), Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.2))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _wire_image(width: int, height: int) -> dict[str, str]:
    return {
        "mime_type": "image/png",
        "data_b64": base64.b64encode(_png_bytes(width, height)).decode("ascii"),
    }


class _ImageRecordingHandle(FakeHandle):
    """A handle that KEEPS the images it was given.

    ``FakeHandle`` records only the text, and the whole question here is what
    pixels survived the wire — so a test asserting on its calls could not tell
    a delivered image from a dropped one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.received: list[tuple[str, list[Any]]] = []

    async def prompt(self, text, images=None, command_id=None):  # noqa: ANN001, ANN202
        self.received.append((text, list(images or [])))
        return await super().prompt(text, images, command_id)

    async def steer(self, text, images=None):  # noqa: ANN001, ANN202
        self.received.append((f"steer:{text}", list(images or [])))
        return await super().steer(text, images)

    async def slash_images(self, command, args, images):  # noqa: ANN001, ANN202
        self.received.append((f"slash:{command}", list(images or [])))
        return "slash ok"


def test_an_oversized_prompt_frame_is_the_shape_that_killed_the_session() -> None:
    """The regression is real: an ordinary pasted screenshot overflows the line.

    The sibling of ``test_ten_jobs_at_the_cap_overflow_the_line_limit_without_
    the_fix`` for the inbound direction. Asserting the UNGUARDED size keeps the
    tests below honest — if images ever got small enough to fit anyway, those
    would pass for the wrong reason and this one would fail loudly instead.
    """
    image = _wire_image(1400, 1400)
    naive = _line_bytes(
        {
            "op": "prompt",
            "req": 1,
            "command_id": str(uuid.uuid4()),
            "text": "what does this show?",
            "images": [image],
        }
    )
    assert naive > _MAX_LINE_BYTES, (
        "the fixture no longer reproduces the oversized inbound frame; "
        f"{naive} bytes is under the {_MAX_LINE_BYTES} limit"
    )


@pytest.mark.asyncio
async def test_an_oversized_inbound_frame_does_not_kill_the_session(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """THE BUG, end to end: the connection must survive a frame it cannot read.

    Driven against the REAL ``RuntimeServer`` over a REAL socket with a
    deliberately unguarded write, which is what an OLD client (or any peer that
    is not ``AttachClient``) does. Before the fix the reader loop's
    ``readline`` raised ``ValueError`` past the ``ConnectionResetError`` handler
    and the ``finally`` dropped the client — the session death the operator hit.

    The assertion that matters is the SECOND message: surviving the bad frame is
    only useful if the session is still usable afterwards, and that is precisely
    what the operator lost.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    writer = None
    try:
        record = await _record(tmp_path)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=_MAX_LINE_BYTES
        )
        writer.write(json.dumps({"key": record.control_key, "client": "attach"}).encode() + b"\n")
        await writer.drain()
        welcome = json.loads((await asyncio.wait_for(reader.readline(), timeout=5)).decode())
        assert welcome.get("op") in ("projection", "welcome")

        # Unguarded on purpose: this is the frame the fixed client would never
        # send and an older one still does.
        oversized = {
            "op": "prompt",
            "req": 1,
            "command_id": str(uuid.uuid4()),
            "text": "here is a screenshot",
            "images": [{"mime_type": "image/png", "data_b64": "A" * (_MAX_LINE_BYTES + 350_000)}],
        }
        assert _line_bytes(oversized) > _MAX_LINE_BYTES
        with caplog.at_level(logging.ERROR, logger="local_operator.session.runtime.server"):
            writer.write(json.dumps(oversized).encode() + b"\n")
            await writer.drain()

            # THE NEXT MESSAGE, which is the whole point. On the pre-fix tree
            # the socket is already gone and this ack never arrives.
            follow_up = {
                "op": "prompt",
                "req": 2,
                "command_id": str(uuid.uuid4()),
                "text": "the next message",
            }
            writer.write(json.dumps(follow_up).encode() + b"\n")
            await writer.drain()
            reply = None
            deadline = asyncio.get_running_loop().time() + 10
            while asyncio.get_running_loop().time() < deadline:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                assert line, "the runtime closed the connection; the oversized frame killed it"
                frame = json.loads(line.decode())
                if frame.get("req") == 2:
                    reply = frame
                    break
            assert reply is not None, "the follow-up was never answered"
            assert reply.get("op") == "ack", f"the follow-up was refused: {reply}"

        # The oversized frame was DISCARDED, not delivered half-parsed.
        assert [text for text, _ in handle.received] == ["the next message"]
        # And it was reported loudly enough to find, naming the limit.
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert "line limit" in logged, f"the discard was not reported: {logged!r}"
        assert str(_MAX_LINE_BYTES) in logged
    finally:
        if writer is not None:
            writer.close()
        registrant.close()


@pytest.mark.asyncio
async def test_a_large_pasted_image_is_refitted_and_actually_arrives(
    tmp_path: Path, monkeypatch
) -> None:
    """The product half: an ordinary paste must SEND, not be refused.

    The operator pastes screenshots routinely, and the sizes that overflow the
    frame are ordinary — one composer-bounded render measured 1.23 MB of base64
    on this machine. So the client refits rather than rejecting, and the
    assertion is that the image genuinely reaches the owner: a guard that
    silently dropped the attachment would pass a "did not crash" test while
    losing the user's work.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")
        image = _wire_image(1400, 1400)
        assert _line_bytes({"op": "prompt", "images": [image]}) > _MAX_LINE_BYTES

        detail = await client.prompt("what does this show?", images=[image])
        assert detail == "prompt ok"
        assert client.connected, "the send severed the connection"

        text, delivered = handle.received[-1]
        assert text == "what does this show?"
        assert len(delivered) == 1, "the attachment was dropped instead of resized"
        arrived = delivered[0]
        payload = getattr(arrived, "data", None) or arrived.get("data_b64")
        # It really is an image, and it really is smaller than the budget.
        decoded = base64.b64decode(payload)
        info = sniff_image(decoded)
        assert info is not None, "what arrived is not a decodable image"
        assert len(payload) < _MAX_LINE_BYTES

        # The session keeps working afterwards, which is the operator's actual
        # complaint — the send used to be the thing that ended it.
        assert await client.prompt("and a follow-up") == "prompt ok"
        assert client.connected
    finally:
        client.close()
        registrant.close()


@pytest.mark.asyncio
async def test_the_refit_log_names_the_payload_and_the_frame_as_what_they_are(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """QA round 3, Q1: the one fix in the round-2 delta that had no test.

    The refit log line carries TWO quantities, and round 2 fixed the second
    one reporting the first: the clause reading ``to N bytes of image payload``
    said ``refitted_size`` — the WHOLE frame, the user's text included — which
    understated-by-conflation was harmless beside a fixed 64 KiB reserve and
    became a 3.25x-16.75x overstatement of the attachments once up to 1 MiB of
    text sat inside the number. Every other fix in that delta gained a test
    that goes red when reverted; this one did not, so the conflation could
    silently return. This pins the logged payload figure against the bytes
    the owner actually received — exact, not approximate, because both sides
    of the comparison are measured on the same send.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    # Substantial text is the regime the old conflation overstated: the frame
    # figure must dwarf the payload figure, or the assertion below could pass
    # with the two quantities swapped back together.
    text = "context: " + ("the model must read this. " * 12_000)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")
        image = _wire_image(1400, 1400)
        assert _line_bytes({"op": "prompt", "text": text, "images": [image]}) > _MAX_LINE_BYTES

        with caplog.at_level(logging.WARNING, logger="local_operator.mobile.attach_client"):
            assert await client.prompt(text, images=[image]) == "prompt ok"
        assert client.connected, "the send severed the connection"
    finally:
        client.close()
        registrant.close()

    refit = [record for record in caplog.records if "bytes of image payload" in record.getMessage()]
    assert len(refit) == 1, f"expected one refit line, got {len(refit)}"
    logged = refit[0].getMessage()
    match = re.search(
        r"resized \d+ image\(s\) to (\d+) bytes of image payload and the frame "
        r"is now (\d+) bytes",
        logged,
    )
    assert match is not None, f"the refit line does not carry both figures: {logged!r}"
    logged_payload, logged_frame = (int(match.group(1)), int(match.group(2)))

    _prompt_text, delivered = handle.received[-1]
    assert len(delivered) == 1, "the attachment was dropped instead of resized"
    payload = getattr(delivered[0], "data", None) or delivered[0].get("data_b64")
    assert isinstance(payload, str) and payload, "no image payload reached the owner"
    assert logged_payload == len(payload), (
        f"the log's image-payload figure is {logged_payload:,} bytes but what "
        f"arrived is {len(payload):,} — the line is reporting the frame again"
    )
    assert logged_frame > logged_payload, (
        "the frame and payload figures agree, so the two quantities the line "
        "exists to distinguish have collapsed back into one"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["prompt", "steer", "slash"])
async def test_every_image_bearing_op_is_guarded_not_just_prompt(
    tmp_path: Path, monkeypatch, op: str
) -> None:
    """``steer`` and ``slash`` carry images too, on the same socket.

    Guarding ``prompt`` alone would leave two live routes to the same session
    death: steering mid-turn with a pasted image, and a slash command that
    takes attachments (``slash_images``). Parametrized rather than copied so a
    FOURTH image-bearing op cannot quietly skip the guard — it shares the one
    seam in ``_request_frame``.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")
        images = [_wire_image(1400, 1400)]
        assert _line_bytes({"op": op, "images": images}) > _MAX_LINE_BYTES

        if op == "prompt":
            await client.prompt("look", images=images)
        elif op == "steer":
            await client.steer("look", images=images)
        else:
            await client.slash("compact", "", images=images)

        assert client.connected, f"{op} with an oversized image severed the connection"
        _text, delivered = handle.received[-1]
        assert len(delivered) == 1, f"{op} lost its attachment"
    finally:
        client.close()
        registrant.close()


@pytest.mark.asyncio
async def test_several_images_share_one_frame_budget(tmp_path: Path, monkeypatch) -> None:
    """N images must fit TOGETHER, not each against the whole limit.

    The subtle way to get this wrong: refit every image against the full 1 MiB
    and each one passes while the frame still overflows. Two composer-bounded
    screenshots already serialize past the limit on this machine, so the
    multi-image case is the ordinary one rather than an edge.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")
        images = [_wire_image(900 + index * 60, 900) for index in range(4)]

        assert await client.prompt("compare these", images=images) == "prompt ok"
        assert client.connected

        _text, delivered = handle.received[-1]
        assert len(delivered) == 4, "an attachment was dropped rather than resized"
        rebuilt = [
            {
                "mime_type": getattr(image, "mime_type", None) or image.get("mime_type"),
                "data_b64": getattr(image, "data", None) or image.get("data_b64"),
            }
            for image in delivered
        ]
        # The FRAME is what had to fit, so that is what is measured.
        assert _line_bytes({"op": "prompt", "req": 1, "images": rebuilt}) <= _MAX_LINE_BYTES
    finally:
        client.close()
        registrant.close()


@pytest.mark.asyncio
async def test_an_unsendable_message_is_refused_by_name_not_silently_lost(
    tmp_path: Path, monkeypatch
) -> None:
    """A refusal must reach the USER, and must not take the session with it.

    The user's composer content is their work: the two ways to get this wrong
    are to drop it silently and to kill the connection reporting it. So the
    refusal is an exception the TUI turns into a notice (and a restored draft),
    the wording names what to do, and the session is still usable after it.

    Two shapes, because they need different sentences: prose over the limit
    (nothing to resize) and an attachment that is not a decodable image at all
    — telling someone to shrink bytes that were never an image sends them off
    fixing the wrong thing.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")

        with pytest.raises(OversizedRequest) as prose:
            await client.prompt("x" * (_MAX_LINE_BYTES + 200_000))
        assert "shorten" in str(prose.value), str(prose.value)
        # The LIMIT, in the same scale as the size beside it. Asserted as "the
        # sentence carries the ceiling" rather than against a raw byte count:
        # both figures print as MB now, because a refusal that quotes
        # `1,341,208 bytes` against an image refusal quoting MB makes the user
        # compare two scales to understand one failure (design round 1, D1).
        assert _megabytes(_MAX_LINE_BYTES) in str(prose.value), "the refusal hides the actual limit"
        assert client.connected, "reporting the refusal killed the connection"

        junk = {
            "mime_type": "image/png",
            "data_b64": base64.b64encode(b"\x00" * (_MAX_LINE_BYTES + 200_000)).decode("ascii"),
        }
        with pytest.raises(OversizedRequest) as unreadable:
            await client.prompt("look at this", images=[junk])
        # Named by POSITION so it points at a specific composer chip, and NOT
        # described as merely "too large", which would be the wrong remedy.
        assert "image 1" in str(unreadable.value), str(unreadable.value)
        assert "not a readable image" in str(unreadable.value), str(unreadable.value)
        assert client.connected

        # Nothing was delivered, and the session still works.
        assert handle.received == []
        assert await client.prompt("a normal message") == "prompt ok"
    finally:
        client.close()
        registrant.close()


@pytest.mark.asyncio
async def test_an_unreadable_attachment_keeps_its_own_sentence_when_the_text_is_the_bulk(
    tmp_path: Path, monkeypatch
) -> None:
    """Review round 3, NIT-1: two failures, two remedies, and the rewrite must
    not collapse them.

    When the text has eaten the frame's budget, :func:`fit_request_frame`
    re-blames a SIZE failure on the text — the copy call review round 2
    (MAJOR-4) argued for, because a per-image share rounding to ``0.0 MB``
    points at a screenshot when only shortening the prompt can help.

    That rewrite used to catch the UNREADABLE refusal too, since it is also an
    ``OversizedRequest``. So a corrupt attachment on a text-heavy frame told
    the user to "shorten the text or send the images on their own" — advice
    that cannot work, because those bytes are not an image at any size and no
    amount of shortening makes them one. The remedy there is always "remove
    it", whatever the budget looks like, which is why the refusal got its own
    class (``UnreadableImageRequest``) rather than a message check.

    The sibling test above covers the same junk payload with a ROOMY budget;
    this one is the text-dominated budget, the only regime where the rewrite
    runs. Both must produce the same sentence.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")

        # THE FIXTURE MUST REACH THE REWRITE, or this asserts nothing: the
        # text has to leave less than `_MIN_VIABLE_IMAGE_BUDGET_BYTES` for the
        # images, which is the only condition under which the text-blaming
        # sentence is selected at all.
        text = "x" * 1_015_000
        junk = {
            "mime_type": "image/png",
            "data_b64": base64.b64encode(b"\x00" * 40_000).decode("ascii"),
        }
        probe = {"op": "prompt", "req": 1, "text": text, "images": [junk]}
        assert (
            len(json.dumps(probe).encode()) + 1 > _READ_LIMIT_BYTES
        ), "the fixture no longer crosses the line limit, so the refit never runs"
        overhead = _frame_overhead_bytes(probe)
        budget = _READ_LIMIT_BYTES - overhead - _FRAME_ENCODING_SLACK_BYTES
        assert 0 < budget < _MIN_VIABLE_IMAGE_BUDGET_BYTES, (
            f"the fixture leaves {budget} bytes for its images; the rewrite "
            "under test only runs below the text-is-the-bulk gate"
        )

        with pytest.raises(OversizedRequest) as refusal:
            await client.prompt(text, images=[junk])
        sentence = str(refusal.value)
        assert "not a readable image" in sentence, (
            f"the unreadable attachment was re-blamed on the text, sending the "
            f"user off shortening a prompt that was never the problem: {sentence}"
        )
        assert "image 1" in sentence, f"the refusal does not name the chip: {sentence}"
        assert "text alone fills" not in sentence, f"the text rewrite replaced it: {sentence}"
        assert client.connected, "reporting the refusal killed the connection"
        assert handle.received == []

        # And the session still works afterwards.
        assert await client.prompt("a normal message") == "prompt ok"
    finally:
        client.close()
        registrant.close()


def test_a_refused_request_leaves_nothing_pending() -> None:
    """A refusal must not park a future nobody will ever resolve.

    Found while testing the refusal path: the request id was registered in
    ``_pending`` BEFORE the fit could raise, so a refused send left a future
    that nothing completes. It sat until teardown, took the disconnect's
    ``ConnectionError``, and logged "exception was never retrieved" for a
    refusal the caller had already handled cleanly.
    """

    async def scenario() -> int:
        client = AttachClient(lambda _projection: None, lambda _reason: None)
        client._connected = True
        # A writer that would FAIL the test by being used: a refused request
        # must never reach the socket at all.
        client._writer = cast(Any, SimpleNamespace(write=_forbidden_write, drain=None))
        with pytest.raises(OversizedRequest):
            await client.prompt("x" * (_MAX_LINE_BYTES + 200_000))
        return len(client._pending)

    assert asyncio.run(scenario()) == 0


def _forbidden_write(_payload: bytes) -> None:
    raise AssertionError("a refused request must not be written to the socket")


def test_the_refit_prefers_the_codec_over_the_users_pixels() -> None:
    """Fidelity means legible text, and pixels carry that while the codec does not.

    The ladder's first rung re-encodes at UNCHANGED dimensions, so an ordinary
    screenshot clears the budget with every pixel intact. Asserted as a
    property rather than against a byte count: the claim is "it did not have to
    downscale", which is what the user would notice.
    """
    from local_operator.imaging import refit_image_to_budget

    source = _png_bytes(1200, 800)
    data_b64 = base64.b64encode(source).decode("ascii")
    # A budget the PNG cannot meet, so the refit genuinely has to do something.
    budget = len(data_b64) // 2
    result = refit_image_to_budget(data_b64, "image/png", budget)
    assert result is not None, "an ordinary screenshot must not be refused"
    refitted_b64, _mime = result
    assert len(refitted_b64) <= budget
    info = sniff_image(base64.b64decode(refitted_b64))
    assert info is not None
    assert (info.width, info.height) == (1200, 800), (
        "the refit gave up pixels when a re-encode would have been enough; "
        f"it delivered {info.width}x{info.height}"
    )


def test_the_every_rung_jpeg_rule_has_its_own_test() -> None:
    """A 1400x1400 continuous-tone frame against 240 KB must FIT, not be refused.

    The self-reported defect this pins: the JPEG re-encode was applied only to
    the no-resize rung, so every descending rung handed back a PNG of content
    PNG cannot compress, the ladder ran out of rungs, and the image was refused
    while the real remedy had never been tried below full size.

    Its own test, named for the claim. The regression was previously caught only
    as a side effect of ``test_several_images_share_one_frame_budget``, whose
    stated subject is the shared budget — so a legitimate refactor of the budget
    split could rewrite that test and silently drop coverage of this rule
    (review round 1, NIT-1). Asserted as a PROPERTY (it fits at all) rather than
    against a byte count or a specific rung, so the ladder stays free to change.
    """
    from local_operator.imaging import refit_image_to_budget

    data_b64 = base64.b64encode(_png_bytes(1400, 1400)).decode("ascii")
    result = refit_image_to_budget(data_b64, "image/png", 240 * 1024)

    assert result is not None, (
        "a 1400x1400 frame against a 240 KB budget was refused; the JPEG "
        "re-encode is not being applied at every rung"
    )
    refitted_b64, mime = result
    assert len(refitted_b64) <= 240 * 1024
    assert mime == "image/jpeg", f"the descending rung returned {mime}, not a re-encode"


@pytest.mark.asyncio
async def test_a_long_prompt_beside_a_screenshot_still_sends(tmp_path: Path, monkeypatch) -> None:
    """The PR's OWN headline repro: a big screenshot over ~780 KB of source.

    The image budget used to be ``limit - 64 KiB``, a constant, while the text
    it had to cover is bounded only by ``MAX_CLIPBOARD_TEXT_BYTES`` (1 MiB). Any
    prompt past ~64 KiB therefore had its images fitted against room that was
    never available, the post-refit re-measure caught the overflow, and the
    whole message was refused — including the exact "one pasted screenshot over
    ~780 KB of source" this module's docstring names as the bug being fixed
    (review round 1, MAJOR-1).

    Parametrized over the text sizes that used to fail, because the threshold is
    what regressed: 32 KiB passed before this fix and everything above it did
    not.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")
        image = _wire_image(1672, 941)

        for text_bytes in (100 * 1024, 300 * 1024, 780 * 1024):
            prompt = "x" * text_bytes
            assert await client.prompt(prompt, images=[image]) == "prompt ok", (
                f"a {text_bytes // 1024} KiB prompt beside a screenshot was refused; "
                "the image budget is not being measured against the real text"
            )
            delivered_text, delivered = handle.received[-1]
            assert delivered_text == prompt, "the text was altered to make it fit"
            assert len(delivered) == 1, "the attachment was dropped rather than resized"
        assert client.connected
    finally:
        client.close()
        registrant.close()


@pytest.mark.asyncio
async def test_a_small_image_hands_its_unused_budget_to_a_large_one(
    tmp_path: Path, monkeypatch
) -> None:
    """An icon must not reserve the same share as a screenshot and waste it.

    A flat ``budget // len(images)`` split gave a 32x32 icon the same share as a
    1600x1000 screenshot, and the icon handed nothing back: the screenshot lost
    half its pixels while most of the frame went unspent (review round 1,
    MINOR-2). Asserted against the SHARE the flat split would have imposed
    rather than a fixed size, so the test states the property — the big image
    got more than an equal share — instead of pinning today's ladder rung.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")

        assert (
            await client.prompt(
                "what is this", images=[_wire_image(32, 32), _wire_image(1600, 1000)]
            )
            == "prompt ok"
        )
        _text, delivered = handle.received[-1]
        assert len(delivered) == 2

        payloads = [getattr(image, "data", None) or image.get("data_b64") for image in delivered]
        flat_share = (_MAX_LINE_BYTES - len("what is this")) // 2
        assert len(payloads[0]) < flat_share // 4, "the icon grew to fill its share"
        assert len(payloads[1]) > flat_share, (
            "the screenshot was held to an equal share while the icon's went "
            f"unspent: it got {len(payloads[1])} of a {flat_share} flat share"
        )
    finally:
        client.close()
        registrant.close()


@pytest.mark.asyncio
async def test_a_refusal_quotes_the_marker_the_user_can_see(tmp_path: Path, monkeypatch) -> None:
    """ "image 3" must name the chip on screen, not the wire position.

    Marker numbers do not renumber when an attachment is deleted, so after three
    pastes and two backspaces the survivor's chip reads ``[Image #3]`` while its
    wire position is 1 — and the refusal said "image 1", pointing at a chip that
    is not on the user's screen (design round 1, D4).

    Also pins the two numbers the sentence carries (design round 1, D1): the
    IMAGE's size rather than its base64 (which inflates by 4/3 and overstated a
    2.4 MB file as 3.2 MB), and the ceiling that actually applied, without which
    the user cannot tell whether cropping would help.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    registrant = RuntimeServer(_ImageRecordingHandle(), kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")
        # THIRTY-TWO attachments, not the sixteen this test was written with.
        # The smallest-first reclaim (review round 1, MINOR-2) raised capacity,
        # so the old fixture REFITS successfully at this head and a test built
        # on it asserts nothing about a refusal at all (QA round 2). The
        # `pytest.raises` below is what keeps that honest.
        images = [{**_wire_image(1024, 1024), "marker": 20 + index} for index in range(32)]

        with pytest.raises(OversizedRequest) as refusal:
            await client.prompt("compare these", images=images)

        message = str(refusal.value)
        named = re.search(r"image (\d+)", message)
        assert named is not None, f"the refusal named no image at all: {message}"
        quoted = int(named.group(1))
        assert quoted >= 20, (
            "the refusal quoted a wire position rather than the block's marker: " f"{message}"
        )
        assert "per-image budget" in message, f"the refusal hides the ceiling: {message}"
        assert "32 attachments" in message, f"the refusal hides the share's cause: {message}"
        assert client.connected
    finally:
        client.close()
        registrant.close()


def test_the_composer_stamps_the_chip_number_onto_every_image_it_sends() -> None:
    """The PRODUCER half of the refusal's marker, pinned from a real draft.

    ``_refit_images`` prefers ``image["marker"]`` and falls back to the wire
    position. That lookup shipped in round 1 with NOTHING populating it:
    ``ImageContent`` had no marker field and ``_image_to_wire`` emitted only
    ``data_b64``/``mime_type``, so every refusal still quoted the position and
    the fix was the old behaviour under a new name (design round 2, D8).

    Starting from a DRAFT rather than a wire dict is the whole point. The
    round-1 test hand-injected ``"marker"`` into its blocks, which pins the
    lookup and can never see the producer go missing; this walks
    ``resolve_markers`` -> ``_image_to_wire``, so deleting either end fails it.

    The gap is what makes it a real test: marker numbers do not renumber on
    delete, so a draft that lost ``#2`` sends chips 1 and 3 from wire positions
    0 and 1.
    """
    from local_operator.session.remote import _image_to_wire
    from local_operator.tui.widgets.editor import Attachment, resolve_markers

    def _attachment(index: int) -> Attachment:
        payload = base64.b64encode(f"png{index}".encode()).decode()
        return Attachment(
            ImageContent(data=payload, mime_type="image/png"),
            f"[Image #{index}]",
        )

    # Three pastes, then the middle chip deleted: the draft cites #1 and #3.
    attachments = {1: _attachment(1), 2: _attachment(2), 3: _attachment(3)}
    text = "compare [Image #1] and [Image #3]"

    images = resolve_markers(text, attachments)
    assert [image.marker for image in images] == [1, 3], (
        "resolve_markers did not carry the chip number onto the images it "
        f"resolved: {[image.marker for image in images]}"
    )

    blocks = [_image_to_wire(image) for image in images]
    assert [block.get("marker") for block in blocks] == [
        1,
        3,
    ], f"_image_to_wire dropped the marker on the way to the socket: {blocks}"
    # The gap is real: wire position 1 is the chip labelled #3.
    assert blocks[1]["marker"] == 3

    # A producer with no chips to name leaves the key OFF rather than sending
    # null, so the position fallback stays in charge for the phone relay.
    bare = _image_to_wire(ImageContent(data="AAAA", mime_type="image/png"))
    assert "marker" not in bare, f"a marker-less image put a key on the wire: {bare}"


def test_the_marker_never_reaches_a_provider_or_a_transcript() -> None:
    """``marker`` is presentation state, and must not become conversation content.

    It rides ``ImageContent`` so the transport can name a chip, which puts it
    one field away from every provider payload, transcript row and context
    hash. ``exclude=True`` is what keeps it out; this pins that, because a
    later change dropping the flag would silently start writing composer state
    into persisted history and into the bytes sent to a model.
    """
    image = ImageContent(data="AAAA", mime_type="image/png", marker=7)

    assert image.marker == 7, "the field must be readable in-process"
    assert "marker" not in image.model_dump()
    assert "marker" not in image.model_dump(exclude_defaults=True)
    assert "marker" not in image.model_dump(mode="json")
    assert "marker" not in image.model_dump_json()


@pytest.mark.asyncio
async def test_a_refit_that_could_fit_is_never_refused_for_being_predicted_to_fail(
    tmp_path: Path, monkeypatch
) -> None:
    """The small-budget branch picks a SENTENCE; it must not veto the send.

    An earlier version short-circuited to a refusal whenever the text left less
    than ``_MIN_VIABLE_IMAGE_BUDGET_BYTES`` for the images, justified as a
    guarantee that the refit would fail anyway. That premise holds for
    continuous-tone photographs and is false for flat screenshots and line art,
    which compress to a fraction of it — so messages the refit would have sent
    were refused unsent (review round 2, MAJOR-4), the same defect MAJOR-1
    filed against the old fixed reserve, one layer up.

    A flat image is the fixture precisely because it is the shape the constant
    mispredicts.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    registrant = RuntimeServer(_ImageRecordingHandle(), kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")

        # A SMOOTH GRADIENT: heavy as PNG on the wire, so the frame really
        # crosses the limit, but it re-encodes to a fraction of the budget the
        # gate predicted it could not meet. That combination is the whole
        # finding — a flat screenshot is small enough that the frame never
        # crosses at all and the early return, not the gate, decides.
        Image = pytest.importorskip("PIL.Image")
        canvas = Image.new("RGB", (3000, 2000))
        pixels = canvas.load()
        for y in range(canvas.height):
            for x in range(canvas.width):
                pixels[x, y] = (
                    (x * 255) // canvas.width,
                    (y * 255) // canvas.height,
                    ((x + y) * 255) // (canvas.width + canvas.height),
                )
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        gradient = {
            "mime_type": "image/png",
            "data_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

        # THE FIXTURE MUST REACH THE BRANCH, or this asserts nothing. A mutation
        # run caught exactly that: an earlier fixture never crossed the limit,
        # so `fit_request_frame` returned at its first length check and the test
        # passed with the defect restored.
        text = "x" * 1_015_000
        frame = {
            "op": "prompt",
            "req": 1,
            "command_id": "00000000-0000-4000-8000-000000000000",
            "text": text,
            "images": [gradient],
        }
        assert (
            len(json.dumps(frame).encode()) + 1 > _READ_LIMIT_BYTES
        ), "the fixture no longer exceeds the line limit, so the refit never runs"
        overhead = _frame_overhead_bytes(frame)
        budget = _READ_LIMIT_BYTES - overhead - _FRAME_ENCODING_SLACK_BYTES
        assert 0 < budget < 32 * 1024, (
            f"the fixture leaves {budget} bytes for its images; the branch under "
            "test only runs for a positive budget under the gate"
        )

        assert await client.prompt(text, images=[gradient]) == "prompt ok"
        assert client.connected

        # THE OTHER DIRECTION: when the text genuinely leaves no room and the
        # refit really cannot fit the image, the refusal still fires and still
        # blames the text rather than the attachment.
        heavy = _wire_image(1400, 1400)
        with pytest.raises(OversizedRequest) as refusal:
            await client.prompt("x" * 1_045_000, images=[heavy])
        assert "text alone fills" in str(
            refusal.value
        ), f"a text-dominated refusal blamed the attachment: {refusal.value}"
        assert client.connected
    finally:
        client.close()
        registrant.close()


@contextlib.contextmanager
def _refit_report(entries: tuple[_RefitReport, ...]):
    """Publish ``entries`` as the current task's refit report, then clear it.

    ``_report_wire_refit_for`` CONSUMES the report, so a test that set the
    ContextVar and left it would leak into the next one; the reset token is
    what keeps these independent of ordering.
    """
    from local_operator.mobile.attach_client import _REFIT_REPORT

    token = _REFIT_REPORT.set(entries)
    try:
        yield
    finally:
        _REFIT_REPORT.reset(token)


def test_the_refit_caption_saturates_on_a_mixed_aspect_paste() -> None:
    """The caption's cost is bounded by SHAPES, not by the attachment count.

    ``_report_wire_refit_for`` used to join one ``#N WxH -> WxH`` clause per
    downscaled image, in ``_refit_images``' internal smallest-first walk. The
    rung that earns a caption is by construction the many-attachment case, so
    the row a user actually got was three lines of 28 at 100x30 and **five of
    20 at 60x22** — a quarter of a narrow viewport spent on a ``note`` about a
    message that was delivered fine — enumerated in an order the user cannot
    see, with six of eight clauses the identical string (design round 2, D9;
    QA round 2, Q5).

    The honest bound is RUNGS x ASPECT RATIOS, not rungs alone. A rung caps
    the LONG edge and the refit preserves aspect ratio, so grouping on the
    exact delivered ``WxH`` splits once per distinct shape in the paste — and
    this test's earlier version built every fixture as a SQUARE, which
    collapsed ``(w, h)`` onto the rung count by construction and passed
    vacuously against a docstring promising "one row at 100 cols for ANY
    number of attachments" (review round 3, MAJOR-5). Re-fixtured with the
    aspect mix an operator actually pastes, that claim failed at every count
    above two.

    What this asserts is the property that survives, and it is the one that
    distinguishes the count form from the per-image enumeration D9 killed:
    given a fixed set of delivered SHAPES, the clause count does not grow
    with the attachment count. The fixture cycles ten real delivered sizes,
    so at 300 attachments the row is the same length as at 10 — where the
    enumeration form would have 300 clauses. Through the real fitter the
    clause count moves within a measured envelope rather than holding exactly
    (4-9 clauses over n=10..30 on a five-aspect paste, because which rung a
    shape lands on tracks the per-image budget); ``_report_wire_refit_for``'s
    docstring carries those figures and the row budget they imply.
    """
    from local_operator.tui.app import OperatorApp

    def caption(entries: list[Any]) -> str:
        captured: list[str] = []
        source = cast(Any, SimpleNamespace())
        app = cast(Any, OperatorApp.__new__(OperatorApp))
        app._notice_for = lambda _source, text, _kind: captured.append(text)
        with _refit_report(tuple(entries)):
            OperatorApp._report_wire_refit_for(app, source)
        return captured[0] if captured else ""

    # Delivered sizes the ladder really produces for composer-bounded source
    # shapes — 16:9 window captures, 9:19.5 phone shots, squares, 3:2 and 4:3
    # photos. NINE of them, because that is the most the real fitter produced
    # over a five-aspect paste swept from n=10 to 30 (measured: 4-9 clauses,
    # 97-168 characters). A fixture wider than reality would make the row
    # budget below a guess rather than a measurement.
    mixed_delivered = (
        (768, 432),
        (512, 288),
        (354, 768),
        (236, 512),
        (768, 768),
        (512, 512),
        (768, 512),
        (512, 342),
        (512, 384),
    )

    def entry(index: int, size: tuple[int, int]) -> _RefitReport:
        width, height = size
        # The composer-bounded source that produced this delivered size: the
        # same aspect at the 1024px ingest edge, so every entry really lost
        # pixels (a paste is bounded BEFORE the transport refit sees it).
        if width >= height:
            source_width, source_height = 1024, round(height * 1024 / width)
        else:
            source_width, source_height = round(width * 1024 / height), 1024
        return _RefitReport(
            marker=index + 1,
            width=width,
            height=height,
            source_width=source_width,
            source_height=source_height,
        )

    # SATURATION: past the point where every shape in the mix has appeared,
    # adding attachments — a hundred, three hundred — adds no clause and keeps
    # the row inside the measured budget.
    for count in (10, 24, 60, 300):
        entries = [entry(i, mixed_delivered[i % len(mixed_delivered)]) for i in range(count)]
        row = caption(entries)
        assert row.startswith(
            f"{count} image"
        ), f"the caption does not lead with the count it is claiming: {row!r}"
        clauses = row.split(": ", 1)[1].split(", ")
        assert len(clauses) == len(mixed_delivered), (
            f"{count} attachments produced {len(clauses)} clauses where the mix "
            f"has {len(mixed_delivered)} shapes — the caption is grouping by "
            f"something other than the delivered size, or growing again: {row!r}"
        )
        # THE CLAIM GUARD, in the idiom `test_ten_jobs_at_the_cap_overflow_the_
        # line_limit_without_the_fix` sets: assert the shape that DISPROVES the
        # old docstring, so "bounded by the ladder's rungs" can never silently
        # return. A composer-bounded paste reaches three rungs (768/512/384 —
        # never the 1024 one, because the ingest bound already applied), and
        # this paste produces nine clauses from them (review round 3, MAJOR-5).
        from local_operator.imaging import IMAGE_WIRE_REFIT_EDGES

        reachable_rungs = sum(
            1 for edge in IMAGE_WIRE_REFIT_EDGES if edge is not None and edge < 1024
        )
        assert len(clauses) > reachable_rungs, (
            f"{len(clauses)} clauses from {reachable_rungs} reachable rungs no "
            "longer exceeds the rung count, so this fixture has stopped being "
            "the mixed-aspect paste it exists to be"
        )
        # A CHARACTER-COUNT PROXY for the rendered row count, not the row count
        # itself: the notice's own glyph and spine indent cost a couple of
        # columns this does not model, and the authority on real geometry is
        # the rendered frames in the design and QA rounds. The budget is the
        # measured envelope — the real fitter over a five-aspect paste swept
        # n=10..30 topped out at 9 clauses and 168 characters, 2 rows at 100
        # cols and 3 at 60, and this fixture reproduces that shape at 167-178
        # characters — so the headroom is measurement, not luck.
        assert len(row) <= 2 * 98, (
            f"the caption is {len(row)} characters at {count} attachments, past "
            f"the measured 2-row budget at 100 columns: {row!r}"
        )
        assert len(row) <= 4 * 58, (
            f"the caption is {len(row)} characters at {count} attachments, past "
            f"the measured 4-row budget at 60 columns: {row!r}"
        )

    # THE COMMON PASTE is a run of same-shape screenshots, and it stays ONE
    # row at 100 cols at any count — the single-shape collapse QA measured to
    # 300 attachments, which the mixed fixtures above must not be read as
    # replacing.
    for count in (2, 8, 300):
        entries = [
            entry(i, (512, 288) if i % 2 else (768, 432))  # one shape, two rungs
            for i in range(count)
        ]
        row = caption(entries)
        clauses = row.split(": ", 1)[1].split(", ")
        assert len(clauses) == 2, f"one shape reached {len(clauses)} rungs: {row!r}"
        assert len(row) < 98, f"a single-shape paste wraps at 100 columns: {row!r}"

    # NO GLYPH ON THIS ROW AT ALL (design round 3, D14 — round 2's D10 had it
    # bound to the count). The chips carry their own ``↓`` for the INGEST
    # bound, and a caption ``↓`` for the TRANSPORT refit put two shrink counts
    # within one glyph of each other on a partial refit.
    #
    # Asserted on the BARE arrow, not on ``RESIZED_MARK`` (which is ``" ↓"``,
    # space included): the finding is that this row shares no glyph with the
    # chips, so any spelling of the arrow is a regression. Pinning the exact
    # constant would let a space-less one back in — a canary caught precisely
    # that, a mutation restoring ``\u2193`` without the leading space sailing
    # through an assertion written against ``RESIZED_MARK``.
    from local_operator.tui.widgets.editor import RESIZED_MARK

    row = caption(
        [
            entry(0, (512, 512)),
            entry(1, (384, 384)),
        ]
    )
    assert RESIZED_MARK.strip() not in row, f"the resize glyph is back on the caption: {row!r}"

    # IN THE USER'S ORDER: groups appear by their lowest chip number, not in
    # the refit's internal smallest-first walk.
    row = caption(
        [
            entry(8, (384, 384)),
            entry(1, (512, 512)),
        ]
    )
    assert row.index("512x512") < row.index(
        "384x384"
    ), f"the caption enumerates in the refit's order rather than the user's: {row!r}"


def test_the_size_scale_never_prints_a_smaller_number_in_a_bigger_unit() -> None:
    """``_megabytes`` must not move DOWN as the byte count goes up.

    The switch used to be keyed on the rounded MB figure, which put the
    boundary mid-KB: ``104,857`` printed ``102 KB`` and ``104,858`` printed
    ``0.1 MB``, so a one-byte step showed a smaller number in a larger unit and
    a user comparing two refusals a minute apart read them backwards (review
    round 2, NIT-2). Round 3 added the same guard one scale DOWN: below a full
    KB the raw byte count prints, because ``1023 B`` reading ``0 KB`` repeats
    the "did not fit in nothing" defect on the per-image share path the split
    can genuinely reach (review round 3, NIT-2).
    """

    def as_bytes(rendered: str) -> float:
        figure, unit = rendered.split()
        return float(figure) * 1024 ** {"B": 0, "KB": 1, "MB": 2}[unit]

    previous = -1.0
    for size in range(0, 3 * 1024 * 1024, 311):
        current = as_bytes(_megabytes(size))
        assert current >= previous, (
            f"{size} bytes rendered as {_megabytes(size)}, which is smaller than "
            "the figure printed for fewer bytes"
        )
        previous = current

    # The boundaries are whole units, so both sides of each read as the same
    # quantity.
    assert _megabytes(1023) == "1023 B"
    assert _megabytes(1024) == "1 KB"
    assert _megabytes(1024 * 1024 - 1) == "1024 KB"
    assert _megabytes(1024 * 1024) == "1.0 MB"


def test_the_text_bulk_refusal_compares_like_with_like() -> None:
    """Design round 3, D17: the two figures the sentence compares share a unit.

    "fills 1000 KB of the 1.0 MB limit" asked the user to convert units
    mid-sentence between exactly the two figures the sentence exists to
    compare. The text figure now renders in the limit's scale — MB — which is
    safe ONLY because of where this sentence fires: the budget precondition
    (under `_MIN_VIABLE_IMAGE_BUDGET_BYTES`, asserted below rather than
    assumed) puts the overhead within a few KB of the whole limit, so its MB
    figure can never round below 0.9 and the ``0.0 MB`` class `_megabytes`
    exists to avoid is unreachable here. The ROOM figure keeps `_megabytes`'
    own scale because it is genuinely small — KB is the honest unit for a few
    KB, and "no room" already covers a sub-KB remainder.
    """

    def figures(overhead: int, budget: int) -> tuple[str, str]:
        sentence = str(_text_is_the_bulk_refusal(overhead, budget))
        text_figure = sentence.split(" fills ", 1)[1].split(" of the", 1)[0]
        limit_figure = sentence.split("the ", 1)[1].split(" limit", 1)[0]
        return text_figure, limit_figure

    # The reachable band for this sentence, both edges: a budget just under
    # the gate with the text a hair under the limit, and one past it.
    for overhead, budget in ((1_000_000, 30_000), (1_011_713, 1_023), (1_100_000, 0)):
        assert budget < _MIN_VIABLE_IMAGE_BUDGET_BYTES, (
            f"the fixture ({overhead:,} overhead, {budget:,} budget) is not in "
            "the band this sentence fires in, so the 0.9-MB-floor argument "
            "does not hold for it"
        )
        text_figure, limit_figure = figures(overhead, budget)
        both_mb = text_figure.endswith("MB") and limit_figure.endswith("MB")
        assert both_mb, f"the compared figures mix units: {text_figure!r} vs {limit_figure!r}"


@pytest.mark.asyncio
async def test_only_a_downscale_is_reported_to_the_user(tmp_path: Path, monkeypatch) -> None:
    """A codec swap keeps every pixel and must stay silent; a downscale must not.

    The product call design round 1 (D2) argued rather than deferred: because the
    composer bounds every paste to 1024px, the common rung is a re-encode at
    unchanged dimensions, and a notice on that would be pure noise on a routine
    gesture. Losing pixels is different in kind — measured at ~40% edge energy —
    so that rung, and only that rung, is surfaced.

    Asserted on ``downscaled``, which is the predicate the transcript row is
    gated on, and measured from the DELIVERED bytes on both sides so a rung
    change cannot make the two cases agree by accident.
    """
    from local_operator.mobile.attach_client import taken_refit_report

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    registrant = RuntimeServer(_ImageRecordingHandle(), kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")

        # Two composer-bounded screenshots: over the limit together, but the
        # codec alone closes the gap, so every pixel survives.
        await client.prompt("look", images=[_wire_image(1024, 576) for _ in range(2)])
        codec_only = taken_refit_report()
        assert not [entry for entry in codec_only if entry.downscaled], (
            "a codec-only refit was reported as a downscale; the transcript "
            f"would put a notice on a routine paste: {codec_only}"
        )

        # Far past what the codec can absorb, so pixels genuinely go.
        await client.prompt("look", images=[_wire_image(1400, 1400) for _ in range(8)])
        downscaled = [entry for entry in taken_refit_report() if entry.downscaled]
        assert downscaled, "pixels were lost and nothing was reported"
        for entry in downscaled:
            assert entry.width * entry.height < entry.source_width * entry.source_height
            assert entry.marker >= 1

        # CONSUMED, not merely read: a stale report would caption the next,
        # untouched message with this one's resize.
        assert taken_refit_report() == ()
    finally:
        client.close()
        registrant.close()


@pytest.mark.asyncio
async def test_a_junk_scalar_frame_does_not_kill_the_connection(
    tmp_path: Path, monkeypatch
) -> None:
    """A frame that parses as a bare JSON scalar must be dropped, not fatal.

    ``json.loads`` succeeds on any scalar, so a discarded oversized line whose
    surviving tail happens to read ``12345``/``null``/``true``/``"s"``/``[1,2]``
    reached ``_on_request`` and hit ``.get`` on an int/None/bool/str/list. The
    resulting ``AttributeError`` escaped the reader loop's
    ``ConnectionResetError``/``BrokenPipeError`` handler and the ``finally``
    dropped the client — the exact session death this module exists to prevent,
    made reachable by the new discard path (review round 1, MAJOR-2).

    Driven over a REAL socket, and the assertion is not merely "still
    connected": the NEXT message must be DELIVERED, which is what the operator
    actually lost.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = _ImageRecordingHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    client = AttachClient(lambda _projection: None, lambda _reason: None)
    try:
        record = await _record(tmp_path)
        await client.connect(record, "s1")
        assert client._writer is not None

        for raw in ("12345", "null", "true", '"str"', "[1,2]"):
            # The fixture only tests what it claims if it really is a non-dict.
            assert not isinstance(json.loads(raw), dict), f"{raw} is not a scalar frame"
            client._writer.write(raw.encode() + b"\n")
            await client._writer.drain()

            assert (
                await client.prompt(f"after {raw}") == "prompt ok"
            ), f"a {raw} frame killed the connection; the next message was lost"
            assert handle.received[-1][0] == f"after {raw}"
            assert client.connected
    finally:
        client.close()
        registrant.close()
