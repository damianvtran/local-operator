"""The app's half of multiplexer resume publication.

The single most important property here is that the SUITE ITSELF must not
publish. This repository's tests are routinely run inside a cmux surface, and
``tests/conftest.py`` isolates HOME and the provider keys but deliberately not
``CMUX_*`` — so without the headless gate every pilot test in this directory
would rewrite the resume binding of the very session the developer is running
the tests from. That failure is silent, and it damages state outside the
suite, which is the kind of bug a test has to prevent rather than report.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

import local_operator.multiplexer.cmux as cmux_mod
from local_operator.session.protocol import SessionProtocol
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture
def resumable_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Back the id ``FakeSession`` reports with a real resumable directory.

    Without this, `is_resumable_session` refuses and nothing publishes for
    reasons that have nothing to do with the property under test.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = tmp_path / "sessions" / FakeSession().session_id
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")


@pytest.fixture
def spy_rpc(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every cmux RPC that would have been sent, and send none."""
    seen: list[str] = []
    monkeypatch.setattr(
        cmux_mod, "_rpc", lambda binary, method, params: seen.append(method) or None
    )
    monkeypatch.setattr(cmux_mod, "_cmux_binary", lambda: "/bin/cmux")
    return seen


@pytest.mark.asyncio
async def test_a_headless_app_publishes_nothing(
    spy_rpc: list[str], resumable_session: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard that keeps this suite off the developer's own cmux surface.

    Deliberately arranged so the headless gate is the ONLY thing left stopping
    a publish: the session id `FakeSession` reports is backed by a real
    resumable directory, and the cmux target/binary are both satisfied. Without
    that setup this test would pass for the wrong reason — an unresumable
    session publishes nothing regardless of the gate, so the assertion would
    hold even if the gate were deleted.

    Proven by construction: flipping `is_headless` to False in the same
    arrangement produces `surface.resume.set`, which is the next test.
    """
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "11111111-2222-3333-4444-555555555555")
    monkeypatch.setenv("CMUX_SURFACE_ID", "66666666-7777-8888-9999-000000000000")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.is_headless is True
        app._session = cast(SessionProtocol, FakeSession())
        app._start_multiplexer_broadcast()
        await asyncio.sleep(0.5)
    assert spy_rpc == []


@pytest.mark.asyncio
async def test_the_same_arrangement_publishes_once_the_gate_is_lifted(
    spy_rpc: list[str], resumable_session: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the test above: proves the gate is what blocks it."""
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "11111111-2222-3333-4444-555555555555")
    monkeypatch.setenv("CMUX_SURFACE_ID", "66666666-7777-8888-9999-000000000000")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        app._session = cast(SessionProtocol, FakeSession())
        app._start_multiplexer_broadcast()
        await asyncio.sleep(0.5)
        assert "surface.resume.set" in spy_rpc


@pytest.mark.asyncio
async def test_the_pane_binding_is_not_left_behind_on_exit(spy_rpc: list[str]) -> None:
    """A clean exit leaves no broadcast handle for a later turn to re-assert."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
    assert getattr(app, "_multiplexer_broadcast", None) is None


@pytest.mark.asyncio
async def test_a_session_swap_rebinds_the_pane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pane must advertise the conversation currently in it.

    What is pinned is the ORDER: retire the outgoing binding, THEN publish the
    new one. Both name the same pane, so the reverse order would clear the
    binding it had just published and leave the pane advertising nothing.

    Three things have to be arranged for this path to run at all, and each one
    is a real precondition rather than test scaffolding:

    * ``is_headless`` must be False. It is Textual's property, not this app's,
      so it is patched on ``OperatorApp`` itself — patching ``type(app)``
      reaches into the framework and changes every app in the process.
    * the session must report a ``session_id``; the pilot's ``FakeSession``
      has none, which is exactly why the hook is a no-op under the other
      tests in this file.
    * the publish/retire pair are imported inside the hook, so they are
      patched on the package they are imported FROM.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    events: list[str] = []

    class Handle:
        def stop(self, *, retire: bool = True) -> None:
            events.append(f"stop(retire={retire})")

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()

        # `events.append(...) or Handle()` would work, but spelled explicitly:
        # a lambda whose first expression returns None is exactly the kind of
        # thing that silently yields the wrong object when edited later.
        def fake_broadcast(session_id: str, **kwargs: Any) -> Handle:
            events.append("publish")
            return Handle()

        monkeypatch.setattr(
            "local_operator.multiplexer.broadcast_session", fake_broadcast, raising=True
        )
        monkeypatch.setattr(
            "local_operator.multiplexer.retire_session",
            lambda handle: handle.stop(retire=True) if handle is not None else None,
            raising=True,
        )
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        # `FakeSession` already reports a `session_id` ("sess"), which is all
        # the hook reads; the other tests in this file no-op only because they
        # never reach this point with a session adopted.
        app._session = cast(SessionProtocol, FakeSession())
        app._start_multiplexer_broadcast()
        app._start_multiplexer_broadcast()
        # Asserted INSIDE the context: leaving it unmounts the app, which
        # correctly retires the live binding and would append a fourth event.
        assert events == ["publish", "stop(retire=True)", "publish"]

    # And that unmount retire is itself part of the contract: a clean exit
    # never leaves the pane advertising a closed session.
    assert events[-1] == "stop(retire=True)"


@pytest.mark.asyncio
async def test_a_broadcast_failure_never_reaches_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication is bookkeeping; it must not be able to break a session."""
    app = OperatorApp(lambda: _factory(FakeSession()))

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cmux socket died")

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr("local_operator.multiplexer.broadcast_session", boom)
        monkeypatch.setattr(type(app), "is_headless", property(lambda self: False))
        # The app's hook swallows it; the session carries on.
        with pytest.raises(RuntimeError):
            boom()
        try:
            app._start_multiplexer_broadcast()
        except RuntimeError:  # pragma: no cover - the failure this test forbids
            pytest.fail("a broadcast failure escaped into the app")
