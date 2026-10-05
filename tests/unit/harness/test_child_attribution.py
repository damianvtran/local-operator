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
    over = ledger.idle_clause(1000.0 - 47 * 60, now=1000.0, bound_s=900.0)
    assert over == "no progress for 47m (bound 15m)"
    # Exactly at the bound is NOT over it.
    assert ledger.idle_clause(1000.0 - 900, now=1000.0, bound_s=900.0) == "idle 15m"


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
    result = await execute_jobs("c", {"op": "list"}, None, None, ctx_for(parent))
    text = body(result)
    assert "idle " in text
    assert "hung-lane" in text
    await asyncio.wait_for(parent.dispose(), timeout=30)


@pytest.mark.asyncio
async def test_hub_ops_list_renders_idle_and_attribution(iso) -> None:
    parent = make_parent(iso, HangingChild())
    await parent.async_init()
    job_id = parent._launch_subagent(label="cancel-lane", prompt="do a long thing")
    await wait_child_dir(parent, job_id)
    await asyncio.sleep(0.2)

    listed = await execute_hub("c", {"op": "list"}, None, None, ctx_for(parent))
    assert "idle " in body(listed)

    await execute_hub(
        "c",
        {"op": "cancel", "to": job_id, "message": "operator asked to stop it"},
        None,
        None,
        ctx_for(parent),
    )
    after = await execute_hub("c", {"op": "list"}, None, None, ctx_for(parent))
    text = body(after)
    assert "stopped by parent-hub: operator asked to stop it" in text
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
    assert "stopped by mobile-stop: user stopped it from the phone" in text
    assert ledger.LANE_NEVER_SETTLED_DETAIL in text


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
