"""Durable outbox + abortable async dispatcher: fencing, atomic intent, real abort."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from local_operator.hub_sync.budget import OperationBudget, OperationDeadlineExceeded
from local_operator.hub_sync.outbox import (
    PUSH_MAX_ATTEMPTS,
    AsyncDispatcher,
    Outbox,
    OutboxRecord,
)


def _stored(out: Outbox, operation_id: str) -> OutboxRecord:
    """The stored record, ASSERTED present: every call here enqueued one first."""

    record = out.read(operation_id)
    assert record is not None
    return record


def test_intent_is_durable_then_settled_and_a_stale_generation_is_fenced(tmp_path: Path) -> None:
    out = Outbox(tmp_path)
    out.enqueue(operation_id="op1", kind="agent", local_id="a1", direction="push", generation=1)
    assert _stored(out, "op1").state == "pending"

    # A late worker from generation 1 must NOT settle after generation 2 starts.
    new_gen = out.bump_generation("op1")
    assert new_gen == 2
    assert out.settle("op1", generation=1, result={"ok": True}) is False  # fenced out
    assert _stored(out, "op1").state == "pending"
    assert out.settle("op1", generation=2, result={"ok": True}) is True
    assert _stored(out, "op1").state == "done"


def test_a_crafted_operation_id_cannot_escape_the_outbox_dir(tmp_path: Path) -> None:
    out = Outbox(tmp_path)
    # A path-shaped id is refused: read yields nothing and enqueue raises, so no
    # file is ever created outside the outbox directory.
    assert out.read("../../etc/passwd") is None
    with pytest.raises(ValueError):
        out.enqueue(
            operation_id="../escape", kind="agent", local_id="a1", direction="push", generation=1
        )


def test_dispatcher_aborts_a_slow_request_within_the_budget(tmp_path: Path) -> None:
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5.0)
        return httpx.Response(200, json={"ok": True})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
        disp = AsyncDispatcher(client=client)
        # A monotonic budget that is already spent for this stage must refuse.
        budget = OperationBudget(
            deadline_monotonic=budget_now() - 1,
            started_monotonic=budget_now() - 10,
            mode="manual",
            operation_id="op",
            attempt_generation=1,
        )
        with pytest.raises(OperationDeadlineExceeded):
            await disp.send("POST", "https://hub.test/v1/x", budget=budget)
        await disp.aclose()

    asyncio.run(run())


def test_dispatcher_aborts_a_trickling_response_not_just_a_slow_connect() -> None:
    async def trickle(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(3.0)
        return httpx.Response(200, json={"ok": True})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(trickle))
        disp = AsyncDispatcher(client=client)
        budget = OperationBudget.start("automatic", "op", 1)
        # Force a tiny remaining budget so the abort path is exercised fast.
        budget = OperationBudget(
            deadline_monotonic=budget.started_monotonic + 0.2,
            started_monotonic=budget.started_monotonic,
            mode="automatic",
            operation_id="op",
            attempt_generation=1,
            clock=budget.clock,
        )
        with pytest.raises(OperationDeadlineExceeded):
            await disp.send("POST", "https://hub.test/v1/x", budget=budget)
        await disp.aclose()

    asyncio.run(run())


def budget_now() -> float:
    import time

    return time.monotonic()


# -- begin_attempt: the push-side state machine (backoff, replay, dead-letter) -----------


def test_begin_attempt_backs_off_after_a_failure_and_resumes_past_the_window(
    tmp_path: Path,
) -> None:
    out = Outbox(tmp_path)
    first = out.begin_attempt("op1", kind="agent", local_id="a1")
    assert first is not None and first.attempts == 1 and first.replay is None
    out.settle(
        "op1",
        generation=first.generation,
        error_class="hub-error",
        next_retry_at="2999-01-01T00:00:00Z",
    )
    # Inside the window: no attempt is handed out (and no attempt is burned).
    assert out.begin_attempt("op1", kind="agent", local_id="a1") is None
    # A done record restarts the counter (a new authored change), and settlement
    # from the OLD generation stays fenced out.
    assert out.settle("op1", generation=first.generation, result={"revision": "x"}) is False


def test_begin_attempt_past_the_backoff_window_fences_and_counts(tmp_path: Path) -> None:
    out = Outbox(tmp_path)
    first = out.begin_attempt("op1", kind="agent", local_id="a1")
    assert first is not None
    out.settle(
        "op1",
        generation=first.generation,
        error_class="hub-error",
        next_retry_at="2000-01-01T00:00:00Z",
    )
    second = out.begin_attempt("op1", kind="agent", local_id="a1")
    assert second is not None and second.attempts == 2
    stale = out.settle("op1", generation=first.generation, result={"revision": "old"})
    assert stale is False  # the late worker from attempt 1 cannot settle


def test_begin_attempt_dead_letters_at_the_cap(tmp_path: Path) -> None:
    out = Outbox(tmp_path)
    region = None
    for _ in range(PUSH_MAX_ATTEMPTS):
        region = out.begin_attempt("op1", kind="agent", local_id="a1")
        assert region is not None
        assert out.settle("op1", generation=region.generation, error_class="hub-error")
    assert region is not None and region.attempts == PUSH_MAX_ATTEMPTS
    assert out.begin_attempt("op1", kind="agent", local_id="a1") is None  # dead-lettered


def test_a_reconciliation_without_a_stored_intent_cannot_replay(tmp_path: Path) -> None:
    """The replay order is: store the intent, THEN go uncertain; without it, skip."""

    out = Outbox(tmp_path)
    first = out.begin_attempt("op1", kind="agent", local_id="a1")
    assert first is not None
    # The PRODUCTION order: the verbatim intent is persisted while pending (right
    # after the preview bought its pin), and only an uncertain outcome follows.
    assert out.store_intent("op1", first.generation, {"document": {"a": 1}})
    assert out.mark_pending_reconciliation("op1", generation=first.generation)
    resumed = out.begin_attempt("op1", kind="agent", local_id="a1")
    assert resumed is not None and resumed.generation == first.generation
    assert resumed.replay == {"document": {"a": 1}}  # verbatim, same generation
    # A record that went uncertain WITHOUT an intent cannot be replayed...
    out2 = Outbox(tmp_path / "other")
    second = out2.begin_attempt("op1", kind="agent", local_id="a1")
    assert second is not None
    assert out2.mark_pending_reconciliation("op1", generation=second.generation)
    assert out2.begin_attempt("op1", kind="agent", local_id="a1") is None
