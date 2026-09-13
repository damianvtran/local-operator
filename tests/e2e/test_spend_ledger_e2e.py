"""The ledger's money survives a real cold open — and only where it is earned.

The unit suite pins the arithmetic; this drives the ASSEMBLED path the operator
actually sees: a real ``Session`` writing its own transcript, then a real
``AttachedSession.cold`` over that journal in a scratch config dir — the boot
state of ``lop``, where there is no runtime to ask and the only source of money
is the file. Two claims, one per direction:

- a session WITH a record paints its exact recalled total and NO ``≥`` (the mark
  is now reserved for a total that really is missing rows);
- a session WITHOUT one — the pre-ledger population, 92.3% of the real store —
  keeps today's behaviour: the one receipt, priced, marked ``≥``, and its dollars
  intact.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.harness.types import (
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import CostKnowledge
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tui.widgets.session_panel import SessionDiagnostics
from tests.e2e.test_viewer_attach_e2e import _never_take_over

pytestmark = pytest.mark.e2e

MODEL = ModelSpec(provider="test", model_id="ledger", context_window=100_000)


def _receipt(usd: float) -> Usage:
    return Usage(
        provider="openrouter",
        model_id="test/x",
        input_tokens=100,
        output_tokens=10,
        context_tokens=100,
        usd_cost=usd,
    )


async def _run_ledger_turn(directory: Path, usd: float) -> Session:
    """One REAL turn through a REAL session, then wait for its record to land."""

    def stream(request, signal=None):
        async def gen():
            yield StreamTextDelta(delta="ack")
            yield StreamEndEvent(stop_reason="stop", usage=_receipt(usd))

        return gen()

    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(directory),
    )
    await session._run_turn([Message.user("ledger probe")])
    async with asyncio.timeout(20):
        while not session._spend_recorded:
            await asyncio.sleep(0.01)
    return session


async def _seed_pre_ledger(directory: Path, usd: float) -> None:
    """A conversation with usage rows and NO spend record: the old shape."""
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("before the ledger"))
    await transcript.append_message(Message.assistant("hi", usage=_receipt(usd)))
    transcript.flush()


@pytest.mark.asyncio
async def test_cold_open_recalls_the_record_without_a_floor_mark(
    headless_tui_env: Path, workspace: Path
) -> None:
    directory = headless_tui_env / "sessions" / "ledgersess01"
    directory.mkdir(parents=True)
    owner = await _run_ledger_turn(directory, 0.0021)
    try:
        assert owner.spend.micro == 2_100
        cold = await AttachedSession.cold(
            "ledgersess01",
            config_dir=headless_tui_env,
            cwd=str(workspace),
            takeover_factory=_never_take_over,
        )
        try:
            state = cold.frontend_state
            # The recalled total, not one receipt priced and marked: the point of
            # the whole change, seen from the surface with no runtime.
            assert state.cumulative_parent_cost == pytest.approx(0.0021)
            assert state.cost_knowledge is CostKnowledge.EXACT
            spend = cold.restored_spend()
            assert spend is not None and spend.micro == 2_100
            diagnostics = SessionDiagnostics.capture(cold)
            assert diagnostics.spend_micro == 2_100
            assert diagnostics.spend_knowledge == "exact"
        finally:
            await cold.dispose()
    finally:
        await owner.dispose()


@pytest.mark.asyncio
async def test_cold_open_of_a_pre_ledger_session_keeps_its_floor_and_money(
    headless_tui_env: Path, workspace: Path
) -> None:
    directory = headless_tui_env / "sessions" / "ledgersess02"
    await _seed_pre_ledger(directory, 0.5)
    cold = await AttachedSession.cold(
        "ledgersess02",
        config_dir=headless_tui_env,
        cwd=str(workspace),
        takeover_factory=_never_take_over,
    )
    try:
        state = cold.frontend_state
        # No record: today's behaviour, and the dollars are not lost.
        assert state.cumulative_parent_cost == pytest.approx(0.5)
        assert state.cost_knowledge is CostKnowledge.FLOOR
        # And nothing claims an exact figure it does not have.
        assert cold.restored_spend() is None
        assert SessionDiagnostics.capture(cold).spend_micro is None
    finally:
        await cold.dispose()


@pytest.mark.asyncio
async def test_a_rebuilt_pre_ledger_session_publishes_an_exact_total(
    headless_tui_env: Path, workspace: Path
) -> None:
    """The one-time rebuild, exercised through the adopt seam it starts from."""
    directory = headless_tui_env / "sessions" / "ledgersess03"
    await _seed_pre_ledger(directory, 1.25)
    session = Session(
        model=MODEL,
        stream_fn=lambda *_a, **_k: None,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(directory),
        has_ui=True,
    )
    session.refresh_frontend_usage()  # the seam the app calls on adopt
    async with asyncio.timeout(30):
        while session._spend_tasks:
            await asyncio.sleep(0.01)
    spend = session.restored_spend()
    assert spend is not None and spend.rebuilt is True
    # One row, nothing dropped: the reconstruction is the whole bill, so the
    # figure is EXACT and carries no mark.
    assert spend.micro == 1_250_000
    assert spend.knowledge() is CostKnowledge.EXACT
    assert session.frontend_state.cumulative_parent_cost == pytest.approx(1.25)
    await session.dispose()
