"""Child-attribution v1: the progress stamp, the two receipts, the reconcile pass.

Drives REAL parent/child Sessions through a scripted provider (the pattern
``test_comms.py`` established) wherever a claim is about the live path, and
drives the new module directly where the claim is about the artifacts. Nothing
touches the network; every case writes only under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.comms import ChildInfo, SubagentComms
from local_operator.harness.types import (
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.session import subagent_ledger as ledger
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import execute_hub, execute_jobs

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


# --- scripted children -------------------------------------------------------


def _tool_call(name: str, args: dict[str, Any]):
    async def gen():
        yield StreamToolCallDelta(
            index=0, id=f"c-{name}", name=name, argument_delta=json.dumps(args)
        )
        yield StreamEndEvent(stop_reason="toolUse")

    return gen()


def _text(body: str):
    async def gen():
        yield StreamTextDelta(delta=body)
        yield StreamEndEvent(stop_reason="stop")

    return gen()


class HangingChild:
    """First call parks the child in one long tool call (a live lane that never
    reports again); later calls hang so it never settles."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: ChatRequest, signal: Any = None):
        self.calls += 1
        if self.calls == 1:
            return _tool_call("bash", {"command": "sleep 3600"})

        async def hang():
            await asyncio.Event().wait()
            if False:
                yield

        return hang()


class CompletingChild:
    """One prose answer and done — the healthy child the negatives are read."""

    def __call__(self, request: ChatRequest, signal: Any = None):
        return _text("all done")


def make_parent(tmp_path: Path, provider: Any, name: str = "parent") -> Session:
    return Session(
        model=MODEL,
        stream_fn=provider,
        tools=[],
        transcript=Transcript(tmp_path / name),
        system_blocks_provider=lambda: ["parent", "env"],
        cwd=str(tmp_path),
    )


def ctx_for(session: Session) -> ToolContext:
    return ToolContext(
        cwd=str(session._cwd), subagent_comms=session.subagent_comms, jobs=session.jobs
    )


async def wait_until(predicate, timeout: float = 15.0) -> None:
    for _ in range(int(timeout * 20)):
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition never became true")


async def wait_child_dir(session: Session, job_id: str, timeout: float = 15.0) -> Path:
    for _ in range(int(timeout * 20)):
        d = session.subagent_comms.session_dir_of(job_id)
        if d is not None:
            return Path(d)
        await asyncio.sleep(0.05)
    raise AssertionError("child dir never appeared")


def body(result: Any) -> str:
    """The text of a tool result, asserted rather than assumed (the pattern
    ``test_comms.py`` uses)."""
    from local_operator.harness.types import TextContent

    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def job_of(session: Session, job_id: str) -> Any:
    """The live job row, asserted present — a vanished row is a real failure,
    not an ``Optional`` for the test to tolerate."""
    job = session.jobs.get(job_id)
    assert job is not None, f"job {job_id} is not registered"
    return job


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    # The team's standing rule: never let an inherited cmux/LOP var reach a
    # headless run that boots a TUI or a fork.
    for key in list(__import__("os").environ):
        if key.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(key, raising=False)
    return tmp_path


# --- the module: receipts ----------------------------------------------------


def test_lane_receipt_round_trips_and_withdraws(tmp_path: Path) -> None:
    payload = ledger.build_lane_payload(
        job_id="j1",
        label="lane",
        agent_role="coder",
        child_session_id="abcd1234",
        parent_session_id="p1",
        parent_job_id=None,
        started_at=100.0,
    )
    written = ledger.write_lane_receipt(tmp_path, payload)
    assert written is not None and written.exists()
    receipts = ledger.read_lane_receipts(tmp_path)
    assert receipts and receipts[0]["job_id"] == "j1"
    assert receipts[0]["artifact"] == ledger.LANE_ARTIFACT
    assert receipts[0]["bound_s"] == ledger.SUBAGENT_LANE_BOUND_S
    ledger.withdraw_lane_receipt(tmp_path, "j1")
    assert ledger.read_lane_receipts(tmp_path) == []


def test_stop_receipt_round_trips(tmp_path: Path) -> None:
    payload = ledger.build_stop_payload(
        job_id="j2",
        label="lane",
        child_session_id="abcd",
        parent_session_id="p1",
        actor="user-escape",
        mechanism="cancel",
        reason="interrupted",
    )
    assert ledger.write_stop_receipt(tmp_path, payload) is not None
    got = ledger.read_stop_receipts(tmp_path)
    assert got and got[0]["actor"] == "user-escape" and got[0]["deliberate"] is True


def test_absent_and_malformed_receipts_are_tolerated(tmp_path: Path) -> None:
    assert ledger.read_lane_receipts(tmp_path) == []
    assert ledger.read_stop_receipts(tmp_path) == []
    (tmp_path / f"{ledger.LANE_PREFIX}bad{ledger.RECEIPT_SUFFIX}").write_text("{not json")
    assert ledger.read_lane_receipts(tmp_path) == []
    # A withdraw of something that is not there must not raise either.
    ledger.withdraw_lane_receipt(tmp_path, "nope")
    ledger.withdraw_stop_receipt(tmp_path, "nope")


def test_two_attempts_accumulate_rather_than_overwrite(tmp_path: Path) -> None:
    for job_id in ("a", "b"):
        ledger.write_lane_receipt(
            tmp_path,
            ledger.build_lane_payload(
                job_id=job_id,
                label=job_id,
                agent_role="",
                child_session_id="c",
                parent_session_id="p",
                parent_job_id=None,
                started_at=1.0,
            ),
        )
    assert {r["job_id"] for r in ledger.read_lane_receipts(tmp_path)} == {"a", "b"}


# --- the module: idle clause -------------------------------------------------


def test_idle_clause_is_none_without_a_stamp() -> None:
    # "No progress recorded" must never read as "stalled" — the load-bearing rule.
    assert ledger.idle_clause(None, now=1000.0) is None


def test_idle_clause_recent_and_over_bound() -> None:
    assert ledger.idle_clause(997.0, now=1000.0) == "idle 3s"
    # The overdue form names the bound and marks the crossing (D1/D6).
    over = ledger.idle_clause(1000.0 - 47 * 60, now=1000.0, bound_s=900.0)
    assert over == "no progress for >15m (stall bound 15m)"
    # Exactly at the bound is NOT over it.
    assert ledger.idle_clause(1000.0 - 900, now=1000.0, bound_s=900.0) == "idle 15m"
    # D1's measured collision: 899 s and 901 s against a 900 s bound must not
    # render the same string, which is what "no progress for 15m (bound 15m)" did.
    under = ledger.idle_clause(1000.0 - 899, now=1000.0, bound_s=900.0)
    just_over = ledger.idle_clause(1000.0 - 901, now=1000.0, bound_s=900.0)
    assert under != just_over, (under, just_over)
    assert under == "idle 15m" and just_over == "no progress for >15m (stall bound 15m)"
    # N2's floor: a row that reported a fraction of a second ago says nothing.
    assert ledger.idle_clause(999.9, now=1000.0) is None
    assert ledger.idle_clause(999.0, now=1000.0) == "idle 1s"


# --- the module: reconcile ---------------------------------------------------


def test_reconcile_recovers_attribution_from_a_stop_receipt(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    ledger.write_stop_receipt(
        child,
        ledger.build_stop_payload(
            job_id="j",
            label="l",
            child_session_id="child",
            parent_session_id="p",
            actor="mobile-stop",
            mechanism="cancel",
            reason="stopped from mobile",
        ),
    )
    out = ledger.reconcile_lane_evidence(
        [{"job_id": "j", "session_dir": str(child), "outcome": "cancelled"}]
    )
    assert len(out) == 1
    assert out[0].ended_by == "mobile-stop"
    assert out[0].cancel_reason == "stopped from mobile"
    assert out[0].never_settled is False


def test_reconcile_flags_a_lane_that_never_settled(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    ledger.write_lane_receipt(
        child,
        ledger.build_lane_payload(
            job_id="j",
            label="l",
            agent_role="",
            child_session_id="child",
            parent_session_id="p",
            parent_job_id=None,
            started_at=1.0,
        ),
    )
    out = ledger.reconcile_lane_evidence(
        [{"job_id": "j", "session_dir": str(child), "outcome": None}]
    )
    assert out and out[0].never_settled is True


def test_reconcile_leaves_a_settled_record_alone(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    ledger.write_lane_receipt(
        child,
        ledger.build_lane_payload(
            job_id="j",
            label="l",
            agent_role="",
            child_session_id="child",
            parent_session_id="p",
            parent_job_id=None,
            started_at=1.0,
        ),
    )
    # A recorded outcome means the lane DID settle: never flag it, even with a
    # receipt still on disk (belt-and-braces remains after a crash mid-settle).
    out = ledger.reconcile_lane_evidence(
        [{"job_id": "j", "session_dir": str(child), "outcome": "completed"}]
    )
    assert out == []


def test_reconcile_has_no_opinion_without_a_session_dir() -> None:
    assert ledger.reconcile_lane_evidence([{"job_id": "j", "outcome": None}]) == []


# --- live path: the stamp ----------------------------------------------------


@pytest.mark.asyncio
async def test_progress_stamps_job_and_record_and_never_zero(iso) -> None:
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="lane", prompt="do a long thing")
    await wait_until(lambda: (job_of(parent, job_id).last_progress_at or 0) > 0)
    job = job_of(parent, job_id)
    record = parent.subagent_comms._record(job_id)
    assert record is not None
    assert job.last_progress_at is not None and job.last_progress_at > 0
    assert record.last_progress_at is not None and record.last_progress_at > 0
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_a_job_that_never_reports_has_no_stamp(iso) -> None:
    """``None``, not ``0`` — the distinction the reader rule depends on."""
    parent = make_parent(iso, CompletingChild())
    await parent.async_init()
    job_id = parent.jobs.register("bash", "quiet", _never_runs)
    job = parent.jobs.get(job_id)
    assert job is not None and job.last_progress_at is None
    assert ledger.idle_clause(job.last_progress_at, now=time.time()) is None
    await asyncio.wait_for(parent.dispose(), timeout=30)


async def _never_runs(job_id, signal, report_progress):  # noqa: ANN001, ANN202
    return "n/a"


# --- live path: surfaces -----------------------------------------------------


@pytest.mark.asyncio
async def test_jobs_ops_list_renders_the_idle_clause(iso) -> None:
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="hung-lane", prompt="do a long thing")
    await wait_until(lambda: (job_of(parent, job_id).last_progress_at or 0) > 0)
    # Push the stamp past the floor (N2) so the clause is exercised deterministically
    # rather than by sleeping: the reading is a function of the stamp, not the clock.
    job_of(parent, job_id).last_progress_at -= 5.0
    result = await execute_jobs("c", {"op": "list"}, None, None, ctx_for(parent))
    text = body(result)
    assert "idle 5s" in text
    assert "hung-lane" in text
    # D7: the clause sits with the columns, BEFORE the label, so a scanning eye
    # finds it in the same region as status/age rather than past a variable label.
    row = next(line for line in text.splitlines() if "hung-lane" in line)
    assert row.index("idle 5s") < row.index("hung-lane")
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_hub_ops_list_renders_idle_and_attribution(iso) -> None:
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="cancel-lane", prompt="do a long thing")
    await wait_child_dir(parent, job_id)
    await asyncio.sleep(0.2)
    job_of(parent, job_id).last_progress_at -= 5.0

    listed = await execute_hub("c", {"op": "list"}, None, None, ctx_for(parent))
    assert "idle 5s" in body(listed)

    await execute_hub(
        "c",
        {"op": "cancel", "to": job_id, "message": "operator asked to stop it"},
        None,
        None,
        ctx_for(parent),
    )
    after = await execute_hub("c", {"op": "list"}, None, None, ctx_for(parent))
    text = body(after)
    # The verb is mechanism-neutral (D2) and the actor is humanized (D4); the raw
    # token still rides the details payload.
    assert "ended by the parent: operator asked to stop it" in text
    assert "parent-hub" not in text
    await asyncio.wait_for(parent.dispose(), timeout=30)


# --- live path: receipts -----------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_withdraws_the_lane_receipt_and_leaves_a_stop_receipt(iso) -> None:
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="lane", prompt="do a long thing")
    child_dir = await wait_child_dir(parent, job_id)
    assert ledger.read_lane_receipts(child_dir)  # staged at attach

    await execute_hub(
        "c", {"op": "cancel", "to": job_id, "message": "stop"}, None, None, ctx_for(parent)
    )
    assert ledger.read_lane_receipts(child_dir) == []  # withdrawn on settle
    stops = ledger.read_stop_receipts(child_dir)
    assert stops and stops[0]["actor"] == "parent-hub" and stops[0]["reason"] == "stop"
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_a_refused_cancel_leaves_no_stop_receipt(iso) -> None:
    parent = make_parent(iso, CompletingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="lane", prompt="quick")
    await wait_until(lambda: job_of(parent, job_id).status != "running", timeout=20)
    child_dir = await wait_child_dir(parent, job_id)
    result = await execute_hub(
        "c", {"op": "cancel", "to": job_id, "message": "too late"}, None, None, ctx_for(parent)
    )
    assert "failed" in body(result) or "already" in body(result)
    assert ledger.read_stop_receipts(child_dir) == []
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_a_completed_child_leaves_no_lane_receipt_and_no_stall_clause(iso) -> None:
    parent = make_parent(iso, CompletingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="healthy", prompt="quick")
    await wait_until(lambda: job_of(parent, job_id).status == "completed", timeout=20)
    child_dir = await wait_child_dir(parent, job_id)
    assert ledger.read_lane_receipts(child_dir) == []
    after = await execute_hub("c", {"op": "list"}, None, None, ctx_for(parent))
    assert "idle " not in body(after)
    await asyncio.wait_for(parent.dispose(), timeout=30)


# --- record + sidecar --------------------------------------------------------


def test_record_outcome_does_not_clobber_attribution(tmp_path, monkeypatch) -> None:
    """Attribution is the actor's statement, not a property of the outcome.

    Mirrors the deliberate non-clearance ``record_outcome`` already performs on
    ``paused``: the actor's stamp must survive the settle that follows it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    comms = make_parent(tmp_path, CompletingChild()).subagent_comms
    comms.record_launch("j1", "lane")
    record = comms._record("j1")
    assert record is not None
    record.ended_by = "user-escape"
    record.cancel_reason = "interrupted"
    record.paused = True  # the other field record_outcome leaves alone
    comms.record_outcome("j1", "cancelled")
    assert record.ended_by == "user-escape"
    assert record.cancel_reason == "interrupted"
    assert record.paused is True


def test_snapshot_and_restore_round_trip_the_new_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    from local_operator.session.session import Session

    session = make_parent(tmp_path, CompletingChild())
    comms = session.subagent_comms
    comms.record_launch("j1", "lane")
    record = comms._record("j1")
    assert record is not None
    record.session_dir = tmp_path  # snapshot() drops a record with none
    record.outcome = "cancelled"
    record.last_progress_at = 123.5
    record.ended_by = "mobile-stop"
    record.cancel_reason = "stopped from mobile"
    rows = comms.snapshot()

    other = SubagentComms(Session.__new__(Session))
    other.restore(rows)
    round_tripped = other._record("j1")
    assert round_tripped is not None
    assert round_tripped.last_progress_at == 123.5
    assert round_tripped.ended_by == "mobile-stop"
    assert round_tripped.cancel_reason == "stopped from mobile"
    # The boot-derived flag is NEVER restored: it is recomputed each boot.
    assert round_tripped.lane_never_settled is False


def test_childinfo_carries_the_new_fields() -> None:
    info = ChildInfo(job_id="j", label="l", status="running", resumable=False, age_s=1.0)
    assert info.last_progress_at is None
    assert info.ended_by == "" and info.cancel_reason == ""


@pytest.mark.asyncio
async def test_reconcile_flags_a_hard_kill_shaped_record(iso) -> None:
    """The parent-gone shape: a lane receipt, no stop receipt, no outcome."""
    parent = make_parent(iso, CompletingChild())
    await parent.async_init()
    child_dir = iso / "sessions" / "orphanchild"
    child_dir.mkdir(parents=True, exist_ok=True)
    ledger.write_lane_receipt(
        child_dir,
        ledger.build_lane_payload(
            job_id="orphan",
            label="orphan",
            agent_role="",
            child_session_id="orphanchild",
            parent_session_id="gone",
            parent_job_id=None,
            started_at=1.0,
        ),
    )
    comms = parent.subagent_comms
    comms.record_launch("orphan", "orphan")
    orphan = comms._record("orphan")
    assert orphan is not None
    orphan.session_dir = child_dir
    parent._reconcile_lane_evidence()
    assert orphan.lane_never_settled is True
    await asyncio.wait_for(parent.dispose(), timeout=30)


# --- boot: the reconcile pass reads a real sidecar ---------------------------


@pytest.mark.asyncio
async def test_a_restart_recovers_stop_attribution_and_flags_a_never_settled_lane(
    iso,
) -> None:
    """The design's restart cell, driven through the REAL boot path.

    A previous process persisted the roster sidecar; two children are on disk —
    one stopped deliberately (its sidecar row never captured the actor, because
    the crash landed between the stamp and the persist) and one whose lane
    receipt survived a hard parent death. ``_load_subagent_roster`` is the exact
    method a boot calls, so this asserts the recovery where it actually happens
    rather than through a hand-rolled reconcile call.
    """
    import json

    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR

    parent_dir = iso / "sessions" / "bootparent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    stopped_dir = iso / "sessions" / "stoppedchild"
    orphan_dir = iso / "sessions" / "orphanchild"
    stopped_dir.mkdir(parents=True, exist_ok=True)
    orphan_dir.mkdir(parents=True, exist_ok=True)

    # The stopped child left a stop receipt but its sidecar row lost the actor.
    ledger.write_stop_receipt(
        stopped_dir,
        ledger.build_stop_payload(
            job_id="stopped1",
            label="stopped-lane",
            child_session_id="stoppedchild",
            parent_session_id="bootparent",
            actor="mobile-stop",
            mechanism="cancel",
            reason="user stopped it from the phone",
        ),
    )
    # The orphaned lane left only its launch receipt.
    ledger.write_lane_receipt(
        orphan_dir,
        ledger.build_lane_payload(
            job_id="orphan1",
            label="orphan-lane",
            agent_role="",
            child_session_id="orphanchild",
            parent_session_id="bootparent",
            parent_job_id=None,
            started_at=1.0,
        ),
    )

    sidecar = {
        "version": 1,
        "generation": 7,
        "jobs": [],
        "records": [
            {
                "job_id": "stopped1",
                "label": "stopped-lane",
                "session_dir": str(stopped_dir),
                "outcome": "cancelled",
                "settled": True,
            },
            {
                "job_id": "orphan1",
                "label": "orphan-lane",
                "session_dir": str(orphan_dir),
                "outcome": None,
                "settled": False,
            },
        ],
        "accounting": {},
    }
    (parent_dir / SUBAGENT_ROSTER_SIDECAR).write_text(json.dumps(sidecar))

    booted = Session(
        model=MODEL,
        stream_fn=CompletingChild(),
        tools=[],
        transcript=Transcript(parent_dir),
        system_blocks_provider=lambda: [],
        cwd=str(iso),
    )
    booted._load_subagent_roster()
    comms = booted.subagent_comms
    stopped = comms._record("stopped1")
    orphan = comms._record("orphan1")
    assert stopped is not None and orphan is not None
    # Attribution recovered from the artifact, not the sidecar (which lost it).
    assert stopped.ended_by == "mobile-stop"
    assert stopped.cancel_reason == "user stopped it from the phone"
    # The never-settled lane is flagged, and the roster surface says so.
    assert orphan.lane_never_settled is True
    listed = await execute_hub("c", {"op": "list"}, None, None, ctx_for(booted))
    text = body(listed)
    assert "ended by the phone: user stopped it from the phone" in text
    assert ledger.LANE_NEVER_SETTLED_DETAIL in text
    assert "launched here and never settled" in text


# --- settle-item (i): does dispose await child runner-settle? ----------------


@pytest.mark.asyncio
async def test_dispose_awaits_child_settle_so_the_lane_receipt_is_withdrawn(iso) -> None:
    """Settle-item (i), answered with evidence rather than prose.

    The design left open whether ``jobs.dispose`` awaits each child runner's
    settle before process exit — which decides whether the settle arm that
    withdraws the lane receipt reliably runs on a dispose-time cancel, or
    whether the leftover receipt is the NORMAL outcome for that path.

    Evidence: after ``Session.dispose()`` returns, the hanging child's lane
    receipt is GONE (its runner's finally ran) and a stop receipt naming
    ``parent-teardown`` is present (attribution was stamped before the cancel).
    If dispose did not await settle, a lane receipt would survive every clean
    quit and the never-settled reading would be worthless.
    """
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="teardown-lane", prompt="do a long thing")
    await wait_until(lambda: (job_of(parent, job_id).last_progress_at or 0) > 0)
    child_dir = await wait_child_dir(parent, job_id)
    assert ledger.read_lane_receipts(child_dir)  # staged while running

    await asyncio.wait_for(parent.dispose(), timeout=30)

    assert ledger.read_lane_receipts(child_dir) == []  # settle arm ran
    stops = ledger.read_stop_receipts(child_dir)
    assert stops and stops[0]["actor"] == "parent-teardown"


# --- Q1 / R-MINOR-1: a REFUSED stop must leave no attribution anywhere --------


def _spy_cancel(session: Session, seen: list[tuple[str, str]]) -> None:
    """Wrap ``jobs.cancel`` recording the record's stamp AT CALL TIME.

    This is what makes the STAGING ORDER observable: the design's rule is that
    the acting party attests BEFORE it acts, so at the moment the cancel is
    issued the record must already name the actor. An assertion taken after the
    ``await`` returns cannot see the order at all (review round 1, R-MINOR-4).
    """
    real = session.jobs.cancel

    async def wrapper(job_id: str, *args: Any, **kwargs: Any) -> bool:
        comms = session._subagent_comms
        record = comms._record(job_id) if comms is not None else None
        seen.append((job_id, "" if record is None else record.ended_by))
        return await real(job_id, *args, **kwargs)

    session.jobs.cancel = wrapper  # type: ignore[method-assign]


def _no_stamp(record: Any) -> bool:
    return record is not None and record.ended_by == "" and record.cancel_reason == ""


@pytest.mark.asyncio
async def test_a_refused_cancel_through_the_jobs_tool_leaves_no_attribution(iso) -> None:
    """Q1's deterministic repro: cancelling an already-completed row is NORMAL
    (the tool reports "was not cancelled"), and it must not stamp the record."""
    parent = make_parent(iso, CompletingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="settle-qa", prompt="quick")
    await wait_until(lambda: job_of(parent, job_id).status == "completed", timeout=20)
    child_dir = await wait_child_dir(parent, job_id)

    result = await execute_jobs(
        "c", {"op": "cancel", "job_id": job_id}, None, None, ctx_for(parent)
    )
    assert "was not cancelled" in body(result)
    assert _no_stamp(parent.subagent_comms._record(job_id))
    assert ledger.read_stop_receipts(child_dir) == []
    after = await execute_hub("c", {"op": "list"}, None, None, ctx_for(parent))
    assert "ended by" not in body(after)
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_a_refused_cancel_on_the_escape_path_leaves_no_attribution(iso) -> None:
    """Esc-Esc on a child that has already settled: the receipt AND the record
    stamp must both roll back."""
    parent = make_parent(iso, CompletingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="settle-escape", prompt="quick")
    await wait_until(lambda: job_of(parent, job_id).status == "completed", timeout=20)
    child_dir = await wait_child_dir(parent, job_id)

    await parent._cancel_job_quietly(job_id, "esc-esc", by="user-escape")

    assert _no_stamp(parent.subagent_comms._record(job_id))
    assert ledger.read_stop_receipts(child_dir) == []
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_a_refused_cancel_through_the_hub_race_leaves_no_attribution(iso) -> None:
    """The settlement race in ``comms.cancel``: the status check passes, then the
    manager refuses. Both halves must roll back."""
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="race", prompt="long")
    await wait_child_dir(parent, job_id)
    child_dir = await wait_child_dir(parent, job_id)

    async def refuse(_job_id: str, *args: Any, **kwargs: Any) -> bool:
        return False

    parent.jobs.cancel = refuse  # type: ignore[method-assign]
    delivery = await parent.subagent_comms.cancel(job_id, by="parent-hub", reason="raced")
    assert delivery.outcome == "failed"
    assert _no_stamp(parent.subagent_comms._record(job_id))
    assert ledger.read_stop_receipts(child_dir) == []
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_a_refused_pause_leaves_no_attribution_and_reports_failure(iso) -> None:
    """R-MINOR-1's worst site: ``pause`` threw the manager's refusal away, so a
    refused pause left a ``stopped by`` stamp on a still-running child."""
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="pause-refused", prompt="long")
    child_dir = await wait_child_dir(parent, job_id)

    async def refuse(_job_id: str, *args: Any, **kwargs: Any) -> bool:
        return False

    parent.jobs.cancel = refuse  # type: ignore[method-assign]
    delivery = await parent.subagent_comms.pause(job_id, by="parent-hub", reason="park")
    assert delivery.outcome == "failed"
    record = parent.subagent_comms._record(job_id)
    assert _no_stamp(record)
    assert record is not None and record.paused is False
    assert ledger.read_stop_receipts(child_dir) == []
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_a_teardown_that_refuses_a_stamp_is_rolled_back(iso, monkeypatch) -> None:
    """R-MINOR-1's fourth site: teardown stamped every running child and could not
    see a per-job refusal. A child that settles instead of being cancelled must
    not keep a ``parent-teardown`` attestation."""
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="teardown-race", prompt="long")
    child_dir = await wait_child_dir(parent, job_id)

    real_cancel = parent.jobs.cancel

    async def settle_instead(job_id_: str, *args: Any, **kwargs: Any) -> bool:
        job = parent.jobs.get(job_id_)
        if job is not None:
            job.status = "completed"
        return False

    parent.jobs.cancel = settle_instead  # type: ignore[method-assign]
    try:
        await asyncio.wait_for(parent.dispose(), timeout=30)
    finally:
        parent.jobs.cancel = real_cancel  # type: ignore[method-assign]

    assert _no_stamp(parent.subagent_comms._record(job_id))
    assert ledger.read_stop_receipts(child_dir) == []


# --- R-MINOR-4: the stamp must PRECEDE the act, at every site ----------------


@pytest.mark.asyncio
@pytest.mark.parametrize("site", ["hub", "escape", "jobs-tool", "teardown"])
async def test_the_stamp_precedes_the_cancel_at_every_site(iso, site: str) -> None:
    """The ordering rule is contract: the acting party attests BEFORE it acts,
    because a wedged child cannot record its own stop and a cancel that raises
    must not lose the attribution. Asserted at the moment ``jobs.cancel`` runs,
    so moving ``begin_stop`` below the ``await`` fails this cell (R-MINOR-4)."""
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label=f"order-{site}", prompt="long")
    await wait_child_dir(parent, job_id)

    seen: list[tuple[str, str]] = []
    _spy_cancel(parent, seen)

    if site == "hub":
        await execute_hub(
            "c", {"op": "cancel", "to": job_id, "message": "hub"}, None, None, ctx_for(parent)
        )
    elif site == "escape":
        await parent._cancel_job_quietly(job_id, "esc", by="user-escape")
    elif site == "jobs-tool":
        await execute_jobs("c", {"op": "cancel", "job_id": job_id}, None, None, ctx_for(parent))
    else:
        # teardown: dispose() awaits each runner's cancel, so the spy observes the
        # cancel the same way the other three sites do.
        await asyncio.wait_for(parent.dispose(), timeout=30)

    assert seen, f"jobs.cancel was never called for site {site}"
    stamped_at_cancel = {by for _jid, by in seen}
    assert (
        stamped_at_cancel and "" not in stamped_at_cancel
    ), f"site {site} issued the cancel before stamping the actor: {seen}"


# --- R-MINOR-2: the receipt's reason is capped like the record's -------------


def test_the_stop_reason_is_capped_on_disk_as_well_as_on_the_record(tmp_path) -> None:
    payload = ledger.build_stop_payload(
        job_id="j",
        label="l",
        child_session_id="c",
        parent_session_id="p",
        actor="parent-hub",
        reason="x" * (ledger.REASON_CAP * 3),
    )
    assert len(payload["reason"]) == ledger.REASON_CAP


# --- D2: a PAUSED row must not read "stopped" --------------------------------


@pytest.mark.asyncio
async def test_a_paused_row_uses_a_mechanism_neutral_verb(iso) -> None:
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="pause-lane", prompt="long")
    await wait_child_dir(parent, job_id)
    await parent.subagent_comms.pause(job_id, by="parent-hub", reason="pausing to free the slot")

    text = body(await execute_hub("c", {"op": "list"}, None, None, ctx_for(parent)))
    assert "paused" in text
    assert "ended by the parent: pausing to free the slot" in text
    # A paused child is halted-and-RESUMABLE: the attribution verb must not claim
    # it was stopped, which contradicted the row's own adjective (D2).
    assert "stopped by" not in text
    await asyncio.wait_for(parent.dispose(), timeout=30)


# --- D3: the hard-death row must not claim an attribution --------------------


@pytest.mark.asyncio
async def test_a_settled_record_with_no_outcome_reads_gone_not_cancelled(iso) -> None:
    """D3, fixed at the ladder's single rung.

    A [redacted] record is stamped ``settled`` by ``restore`` itself, so the hard
    parent-death shape lands on the last rung with NO outcome. It used to read
    ``cancelled`` — an attribution nobody made, contradicted on the same row by the
    reconcile detail. The design's §2.5 table expects ``gone``.
    """
    import json

    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR

    parent_dir = iso / "sessions" / "d3parent"
    parent_dir.mkdir(parents=True, exist_ok=True)
    child_dir = iso / "sessions" / "d3child"
    child_dir.mkdir(parents=True, exist_ok=True)
    # A real hard-killed child HAS a transcript (its lane ran): give it one, so the
    # resumable assertion below is about the STATUS WORD rather than about the
    # roster's own transcript probe.
    from local_operator.session.transcript import TRANSCRIPT_FILENAME

    (child_dir / TRANSCRIPT_FILENAME).write_text(
        json.dumps({"type": "message", "role": "user", "content": "go"}) + "\n"
    )
    ledger.write_lane_receipt(
        child_dir,
        ledger.build_lane_payload(
            job_id="dead1",
            label="dead-lane",
            agent_role="",
            child_session_id="d3child",
            parent_session_id="d3parent",
            parent_job_id=None,
            started_at=1.0,
        ),
    )
    (parent_dir / SUBAGENT_ROSTER_SIDECAR).write_text(
        json.dumps(
            {
                "version": 1,
                "generation": 3,
                "jobs": [],
                "records": [
                    {
                        "job_id": "dead1",
                        "label": "dead-lane",
                        "session_dir": str(child_dir),
                        "outcome": None,
                        "settled": True,
                    }
                ],
                "accounting": {},
            }
        )
    )
    booted = Session(
        model=MODEL,
        stream_fn=CompletingChild(),
        tools=[],
        transcript=Transcript(parent_dir),
        system_blocks_provider=lambda: [],
        cwd=str(iso),
    )
    booted._load_subagent_roster()
    info = next(row for row in booted.subagent_comms.roster() if row.job_id == "dead1")
    assert info.status == "gone", info
    assert info.status != "cancelled"
    assert info.detail is not None and "never settled" in info.detail
    # Still resumable: the word changed, the capability did not.
    assert info.resumable is True


# --- R-MINOR-3: teardown attribution for a GRANDCHILD ------------------------


@pytest.mark.asyncio
async def test_a_child_session_s_own_teardown_stamps_its_running_grandchild(iso) -> None:
    """The mechanism R-MINOR-3 doubted, exercised directly.

    ``_stamp_teardown_attribution`` walks only THIS session's manager, and a
    grandchild's row lives on its parent's — so the reviewer read a torn-down
    grandchild as landing on the never-settled path. It does not: teardown is
    RECURSIVE. ``Session.dispose`` (root) cancels its child's runner, whose
    ``finally`` calls ``_dispose_child`` → ``child.dispose()``, and THAT dispose
    stamps the child's own running rows with ``parent-teardown`` before cancelling
    them. This cell drives the inner half directly: a child Session holding a
    running grandchild, disposed on its own, must leave the grandchild a
    ``parent-teardown`` receipt and no never-settled reading.
    """
    from local_operator.harness.jobs import AsyncJob

    child = make_parent(iso, HangingChild(), name="childsess")
    await child.async_init()
    grandchild_job = child._launch_subagent(label="grandchild", prompt="long")
    await wait_child_dir(child, grandchild_job)
    grandchild_dir = await wait_child_dir(child, grandchild_job)

    # A settle that never happens: the point is the STAMP, not the outcome.
    async def never_settles(job_id: str, *args: Any, **kwargs: Any) -> bool:
        job = child.jobs.get(job_id)
        if isinstance(job, AsyncJob):
            job.status = "cancelled"
        return True

    child.jobs.cancel = never_settles  # type: ignore[method-assign]
    await child.dispose()

    stamps = ledger.read_stop_receipts(grandchild_dir)
    assert stamps and stamps[0]["actor"] == "parent-teardown", stamps
