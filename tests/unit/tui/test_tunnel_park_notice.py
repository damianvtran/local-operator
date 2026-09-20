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

from local_operator.tui.app import OperatorApp
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
    is the files.
    """
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
                "remedy": {"command": "lop tunnel connect", "url": "https://example.invalid"},
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
        assert toast.message == "Radient sign-in needed — run /login radient"

        # The same episode, polled again: the card is left alone. `show`
        # re-arms its own dismissal timer, so re-raising here would hold a
        # ten-second card on screen for as long as the condition lasted.
        first_generation = toast.generation
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.generation == first_generation, "a second poll re-raised the card"

        # The user signs in: the park is gone from disk (the connector cleared
        # it, or `lop tunnel status` did), and the claim on screen goes with it.
        state.clear()
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is False, "a cleared park must retract its claim"

        # …and a park that comes BACK is news again.
        _park(tmp_path)
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is True


@pytest.mark.asyncio
async def test_the_nag_gate_never_mentions_a_tunnel_that_is_not_in_use(tmp_path) -> None:
    """No tunnel, or a deliberately stopped one, is silent.

    Both are states where nothing has gone wrong: a machine that never enrolled
    a tunnel has no remote access to restore, and `stopped` is a decision the
    operator already made. Nagging about either is how a notice trains people to
    ignore it.
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

        # A park behind a deliberate stop, which is not a park to act on. The
        # connector clears it itself when it runs; the terminal must not depend
        # on having run.
        _park(tmp_path, stopped=True)
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is False
        assert state.read() is not None, "the fixture really did leave a park on disk"


@pytest.mark.asyncio
async def test_another_reason_names_its_own_command(tmp_path) -> None:
    """The login is the common case, not the only one: a console re-enrolment
    says which local command clears it, from the park's own remedy."""
    _park(tmp_path, reason="reenrolment_required")
    app = await _boot()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        toast = app.query_one(Toast)
        app._poll_tunnel_park()
        await pilot.pause()
        assert toast.display is True
        assert "needs re-enrolment" in toast.message
        assert "/login radient" not in toast.message
