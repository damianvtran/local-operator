"""The terminal tells the user when this machine's remote access needs a person.

A parked connector cannot say anything itself: parking means it exited
SUCCESSFULLY — the one exit a supervisor reads as "do not retry me" — so no
process is left running, and the file it left behind
(`~/.local-operator/tunnel/state.json`) is the only thing that knows. These
tests drive the REAL app (the one that loads `local_operator.tcss`) because the
question is what a user sees, and assert the three rules the surface has:

* one card per EPISODE, not per poll — the interval is 5 s and the card lives
  10 s, so re-raising on every tick would hold it on screen forever;
* the nag gate: never for a machine with no tunnel, never for one the operator
  deliberately stopped;
* withdrawal when the park clears, because a card still claiming remote access
  is off contradicts the sign-in the user just completed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from rich.cells import cell_len

from local_operator.tui.app import OperatorApp, tunnel_park_card
from local_operator.tui.widgets.status_line import StatusLine
from local_operator.tui.widgets.toast import Toast
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolate_sources(tmp_path, monkeypatch):
    # A headless pilot that inherits the operator's CMUX_* variables can rename
    # their real cmux workspaces; HOME is redirected too because the config dir
    # alone leaves the cache pointed at the real home (AGENTS.md, "Isolating a
    # run").
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)


def _tunnel_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "config" / "tunnel"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _park(tmp_path: Path, *, reason: str = "login_required", stopped: bool = False) -> None:
    """The on-disk shape the connector leaves behind, written by hand.

    The store itself is exercised in `tests/unit/test_tunnels.py`; what matters
    here is that the terminal reads the FILES and nothing else, so the fixture
    is the files. The remedy's command still comes from the vocabulary rather
    than from a literal, because it is the command the card is judged on: a
    fixture that hard-coded one would test a park no connector writes.
    """
    from local_operator.tunnels import gateway

    directory = _tunnel_dir(tmp_path)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "tunnel_id": "tunnel-1",
                "credential_id": 7,
                "gateway_port": 4100,
                "stopped": stopped,
                "record": {"id": "tunnel-1", "status": "active"},
            }
        )
    )
    (directory / "state.json").write_text(
        json.dumps(
            {
                "state": "parked",
                "reason": reason,
                "detail": "the connector stopped and will not retry by itself",
                "remedy": {
                    "command": gateway.TERMINAL_REMEDY.get(reason, "lop tunnel status"),
                    "url": "https://example.invalid",
                },
                "credential_id": 7,
                "at": 1_800_000_000,
                "first_at": 1_800_000_000,
                "attempts": 1,
                "logged_at": 1_800_000_000,
                "logged_attempts": 1,
            }
        )
    )


async def _boot():
    app = OperatorApp(lambda: _factory(FakeSession()))
    return app


def _band(app: OperatorApp) -> StatusLine:
    """The app's status band, asserted present.

    ``_status`` is optional on the app until mount, and this suite drives the
    real app: asserting once here rather than at each of the seven reads is both
    shorter and the narrowing pyright needs (a member access is re-widened by any
    call on its object, and these tests call the poll between reads).
    """
    assert app._status is not None
    return app._status


@pytest.mark.asyncio
async def test_a_park_is_announced_once_and_withdrawn_when_it_clears(tmp_path) -> None:
    """One card, the login remedy, and it goes away when the park does."""
    from local_operator.tunnels import state

    _park(tmp_path)
    app = await _boot()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        toast = app.query_one(Toast)

        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display, "an app that starts parked must say so"
        assert toast.message == "! /login radient — Radient sign-in expired"
        # …and the BAND carries it too, which is the half that survives the
        # card's ten seconds (D4). Read out of the rendered row rather than
        # asked of `is_showing`, which reports whether the LADDER shed a rung
        # and not whether the segment had anything to paint. `_status` is
        # optional on the app until mount; this one asserts it, the way the
        assert "remote access off" in _band(app).render_text(120).plain

        # The same episode, polled again: the card is left alone. `show`
        # re-arms its own dismissal timer, so re-raising here would hold a
        # ten-second card on screen for as long as the condition lasted.
        first_generation = toast.generation
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.generation == first_generation, "a second poll re-raised the card"
        assert "remote access off" in _band(app).render_text(120).plain

        # The user signs in: the park is gone from disk (the connector cleared
        # it, or `lop tunnel status` did), and both claims go with it.
        state.clear()
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is False, "a cleared park must retract its claim"
        assert "remote access off" not in _band(app).render_text(120).plain

        # …and a park that comes BACK is news again.
        _park(tmp_path)
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is True
        assert "remote access off" in _band(app).render_text(120).plain


@pytest.mark.asyncio
async def test_the_nag_gate_never_mentions_a_tunnel_that_is_not_in_use(tmp_path) -> None:
    """No tunnel, or a deliberately stopped one, is silent — card AND band.

    Both are states where nothing has gone wrong: a machine that never enrolled
    a tunnel has no remote access to restore, and `stopped` is a decision the
    operator already made. Nagging about either is how a notice trains people to
    ignore it — and the band is the harder half of the gate, because it does not
    expire: a rung left standing over a tunnel the user switched off would sit
    there for the life of the session.
    """
    from local_operator.tunnels import state

    app = await _boot()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        toast = app.query_one(Toast)

        # No tunnel configured at all.
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is False
        assert "remote access off" not in _band(app).render_text(120).plain

        # A park behind a deliberate stop, which is not a park to act on. The
        # connector clears it itself when it runs; the terminal must not depend
        # on having run.
        _park(tmp_path, stopped=True)
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is False
        assert "remote access off" not in _band(app).render_text(120).plain
        assert state.read() is not None, "the fixture really did leave a park on disk"


@pytest.mark.asyncio
async def test_another_reason_names_its_own_command(tmp_path) -> None:
    """The login is the common case, not the only one: a console re-enrolment
    says which local command clears it, from the park's own remedy — and the
    command comes FIRST, because it is the part a clamp must never eat."""
    _park(tmp_path, reason="reenrolment_required")
    app = await _boot()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        toast = app.query_one(Toast)
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is True
        assert toast.message == "! lop tunnel connect — remote access is off"
        assert "/login radient" not in toast.message


def test_the_card_wording_is_pinned_to_the_vocabulary_it_mirrors() -> None:
    """The two tables in `app.py` name the vocabulary's own strings, so pin them.

    The card cannot import `tunnels.gateway` at module scope (the poll's own
    import is lazy: that module drags httpx and starlette onto every session's
    boot path), so its reasons and commands are literals — the same trade
    `toast.py` makes for its mirrored MCP copy, and the same reason it needs a pin
    rather than a comment. A renamed park code or remedy is exactly the drift the
    round-1 review opened with (one condition, two commands); this is what makes
    that drift fail here instead of reaching a user.
    """
    from local_operator.tui.app import TUNNEL_CARD_COMMANDS, TUNNEL_CARD_REASONS
    from local_operator.tui.widgets.status_line import ICON_APPROVALS
    from local_operator.tunnels import gateway

    # The three codes the connector can park with (see `service.py`), by the
    # vocabulary's own names rather than by a second spelling of them.
    assert set(TUNNEL_CARD_REASONS) == {
        gateway.LOGIN_REQUIRED,
        gateway.REENROLMENT_REQUIRED,
        gateway.LOCAL_PREREQUISITE,
    }
    assert set(TUNNEL_CARD_COMMANDS) == {gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED]}
    # The one mapping this app performs, exercised end to end rather than trusted.
    assert (
        tunnel_park_card(gateway.LOGIN_REQUIRED, gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED])
        == f"{ICON_APPROVALS} /login radient — Radient sign-in expired"
    )


def test_every_card_branch_leads_with_its_command_and_fits_one_row() -> None:
    """D3/D9 held to numbers rather than to a frame: width, order and glyph.

    50 cells is the budget rather than the 58-cell content box, because the box
    shrinks with the terminal (`toast_max_width`) while a card that has to be
    re-read by someone the incident happened to should not start clamping until
    well below the ordinary size. Measured with `cell_len`, which is the one
    width model this codebase uses — the em dash is one cell and the glyph is
    one cell, and neither is obvious from `len()`.
    """
    from local_operator.tui.app import tunnel_park_card
    from local_operator.tui.widgets.status_line import ICON_APPROVALS

    cases = [
        ("login_required", "lop login radient", "/login radient — Radient sign-in expired"),
        ("reenrolment_required", "lop tunnel connect", "lop tunnel connect — remote access is off"),
        (
            "local_prerequisite",
            "lop tunnel install",
            "lop tunnel install — a prerequisite is missing",
        ),
    ]
    for reason, command, body in cases:
        card = tunnel_park_card(reason, command)
        assert card == f"{ICON_APPROVALS} {body}", card
        assert cell_len(card) <= 50, (reason, cell_len(card))
        # The command leads, the reason trails: the tail is what a clamp may lose.
        assert card.index(body.split(" — ")[0]) == cell_len(ICON_APPROVALS) + 1

    # A reason this build has no wording for still reads as something, and an
    # unknown command is shown as it stands — a shell command in a terminal is
    # runnable, so there is nothing to guess at. `reason_label` returns an
    # unknown code verbatim rather than as "unknown", which is why the tail here
    # is the code itself.
    assert tunnel_park_card("some_new_code", "lop tunnel status") == (
        f"{ICON_APPROVALS} lop tunnel status — some_new_code"
    )
    # A park with no remedy at all names no command rather than an empty one.
    assert tunnel_park_card("login_required", "") == (f"{ICON_APPROVALS} Radient sign-in expired")
