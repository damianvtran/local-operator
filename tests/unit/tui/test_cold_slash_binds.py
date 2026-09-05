"""A slash command that needs a runtime engages one on a cold viewer.

**Why this file exists.** Every fresh `lop` since 0.46.0 boots as a COLD
viewer: a ``RemoteSession.cold(<minted id>)`` bound to no runtime, advertising
zero ``slash_capabilities``. The first keystroke starts a speculative warm
engage that takes 1.1–2.8 s (measured, QA round 1). A user who pastes
``/team lopdev ship it`` and presses Enter at t=0 — or types faster than the
engage — reached the LOCAL ``_cmd_team``, whose attach seam does not exist on a
viewer, and was refused with copy that read as permanent. The command was lost.
Three review tracks reproduced it independently (review R2, QA Q2, UX U1) and
the credential shape was worse: the masked paste opened, accepted the secret,
and then reported nowhere to put it (UX U3).

The rule these pin is the one a PROMPT already follows: ``RemoteSession.
prompt()`` calls ``_ensure_bound()`` first. A slash whose effect lives on the
runtime — the mutating ``/team``/``/agent`` forms and every ``/credential``
verb — now does the same (``_needs_runtime_first`` → ``_bind_then_dispatch``),
and is dispatched AGAIN against the capabilities the bound viewer adopts.

**The object under test is the one `lop` builds**, not a stub: a real
``OperatorApp`` over a real ``RemoteSession.cold`` with a minted id, in an
isolated config dir. ``engage_runtime`` is the one seam stubbed, exactly as
``tests/unit/session/test_remote_cold.py`` stubs it, to start an in-process
``RuntimeServer`` over a real loopback socket instead of spawning a python
with a provider. Everything after the engage — record scan, dial, canonical
sync, capability adoption, the second dispatch — is production code.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from textual import events

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from tests.unit.session.runtime.test_server import FakeHandle
from tests.unit.tui.test_app_pilot import _transcript_text


async def _never() -> Any:
    raise AssertionError("a viewer never takes over")


class _RoutingHandle(FakeHandle):
    """A reduced owner that advertises the FULL capability list.

    ``FakeHandle``'s frontend store carries no ``slash_capabilities``; a bound
    viewer that adopted it would still fall through to the local handlers and
    the test would prove nothing about routing. Production owners advertise
    the registry, so this one does too. It records every routed slash and
    every credential verb, and never receives a value it would print.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__()
        from local_operator.session.frontend_state import _slash_capabilities

        # The record the server publishes and the canonical state the viewer
        # adopts must both name the MINTED id, or the viewer's record scan
        # finds nothing and the sync is refused as another session's.
        self._projection = dataclasses.replace(self._projection, session_id=session_id)
        self._frontend.mutate(session_id=session_id, slash_capabilities=_slash_capabilities())
        self.credential_calls: list[tuple[str, str, int]] = []

    async def run_slash_authoritative(self, command, args, images):  # noqa: ANN001, ANN202
        self.calls.append(("run_slash_authoritative", (command, args, images), {}))
        if command == "team":
            name, _, request = args.partition(" ")
            return {
                "kind": "notice",
                "text": f"sending to {name}. manager is coordinating.",
                "style": "info",
                "data": {"type": "team_attached", "team": name, "request": request.strip()},
            }
        if command == "agent":
            return {
                "kind": "notice",
                "text": f"agent {args.partition(' ')[0]} is active",
                "style": "info",
                "data": {
                    "type": "agent_attached",
                    "agent": args.partition(" ")[0],
                    "request": args.partition(" ")[2].strip(),
                },
            }
        return {"kind": "notice", "text": f"owner ran /{command}", "style": "info"}

    def credential_op(self, action: str, key: str, value: str) -> dict[str, Any]:
        # Length only: the value never leaves this call, not even into the
        # recorded tuple, so no assertion downstream can print it.
        self.credential_calls.append((action, key, len(value)))
        if action == "store":
            return {"ok": True, "key": key, "replaced": False}
        if action == "list":
            return {"ok": True, "credentials": [{"key": key, "source": "command"}]}
        return {"ok": True}


def _stub_engage(monkeypatch: pytest.MonkeyPatch, server: RuntimeServer) -> list[int]:
    """Stand in for the runtime spawn: publish a live record for the session."""
    engagements: list[int] = []

    async def fake_engage(session_id, cwd, work, *, config_dir, deadline_s=30.0):  # noqa: ANN001
        engagements.append(1)
        # The measured engage takes over a second; a stub that binds
        # instantly would never let Enter beat it, and the race is the test.
        await asyncio.sleep(0.2)
        server.start()
        marker = config_dir / "sessions" / session_id / ".session.pid"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(os.getpid()), encoding="utf-8")
        for _ in range(200):
            rows = registry.scan(config_dir)
            if rows and rows[0][1] == "live":
                return None
            await asyncio.sleep(0.02)
        raise AssertionError("the fake runtime never published")

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", fake_engage)
    return engagements


def _cold_app(config_dir: Path, session_id: str) -> OperatorApp:
    """The viewer factory shaped exactly as ``cli.py`` builds it: id minted first."""

    async def factory() -> RemoteSession:
        return await RemoteSession.cold(
            session_id, config_dir=config_dir, cwd=str(config_dir), takeover_factory=_never
        )

    OperatorApp._check_for_update = lambda self: None  # type: ignore[method-assign]
    return OperatorApp(factory)


def _rig(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[OperatorApp, _RoutingHandle, RuntimeServer, list[int]]:
    session_id = uuid.uuid4().hex[:12]
    handle = _RoutingHandle(session_id)
    server = RuntimeServer(handle, kind="tui")
    engagements = _stub_engage(monkeypatch, server)
    return _cold_app(config_dir, session_id), handle, server, engagements


async def _boot(app: OperatorApp, pilot: Any) -> RemoteSession:
    for _ in range(200):
        await pilot.pause()
        if app._session is not None:
            break
    session = app._session
    assert isinstance(session, RemoteSession)
    # Since #622 the mount engage is already in flight here (the stub delays
    # it), so this is the state a real `lop` is in for its first 1–3 s: cold,
    # advertising nothing, with a runtime on the way. The submit below lands
    # inside that window, which is exactly the race under test.
    assert session.is_cold and session.session_id, "the shape lop boots into"
    assert not session.frontend_state.slash_capabilities, "cold advertises nothing"
    return session


async def _paste_and_enter(app: OperatorApp, pilot: Any, line: str) -> None:
    """The t=0 submit: one paste event, then Enter, no typing in between.

    ``/agent <name>`` is pasted with a trailing message so the name-argument
    picker's Enter (which COMPLETES a bare parked name rather than submitting
    it) is not what is under test here; the dispatch seam is.
    """
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    app.post_message(events.Paste(line))
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


async def _until(pilot: Any, predicate: Any, *, timeout: float = 10.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await pilot.pause()
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "sessions").mkdir()
    # A configured provider, as every real `lop` has: since #622 the TUI
    # engages at mount and every engage trigger — mount, keystroke, and the
    # bind-then-route seam — is gated on ``_runtime_can_start``, which refuses
    # to spawn against an empty config (the first-run screen).
    (tmp_path / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )
    return tmp_path


@pytest.mark.asyncio
async def test_team_pasted_and_entered_at_t0_on_a_cold_viewer_binds_then_runs(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator's exact command as his first keystroke, cold."""
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry

    TeamRegistry(isolated).create_team(
        TeamEditFields(
            name="lopdev",
            description="d",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )
    app, handle, server, engagements = _rig(isolated, monkeypatch)
    try:
        async with app.run_test(size=(110, 30)) as pilot:
            session = await _boot(app, pilot)
            await _paste_and_enter(app, pilot, "/team lopdev ship the team fix")
            # Nothing was refused before the engage landed.
            assert "but not run one" not in _transcript_text(app), _transcript_text(app)

            routed = await _until(
                pilot,
                lambda: any(c[0] == "run_slash_authoritative" for c in handle.calls),
            )
            assert routed, f"never routed; transcript:\n{_transcript_text(app)}"
            assert session.is_cold is False, "the command must bind the viewer first"
            assert engagements == [1], "one runtime, shared with any warm engage"
            command, args, _images = next(
                c[1] for c in handle.calls if c[0] == "run_slash_authoritative"
            )
            assert (command, args) == ("team", "lopdev ship the team fix")
            # The receipt printed and the request went out as a real turn.
            prompted = await _until(pilot, lambda: any(c[0] == "prompt" for c in handle.calls))
            assert prompted, handle.calls
            text = _transcript_text(app)
            assert "sending to lopdev" in text, text
            assert "but not run one" not in text, text
    finally:
        server.close()


@pytest.mark.asyncio
async def test_agent_entered_at_t0_on_a_cold_viewer_binds_then_routes(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, handle, server, engagements = _rig(isolated, monkeypatch)
    try:
        async with app.run_test(size=(110, 30)) as pilot:
            await _boot(app, pilot)
            await _paste_and_enter(app, pilot, "/agent reviewer look at this")
            routed = await _until(
                pilot,
                lambda: any(c[0] == "run_slash_authoritative" for c in handle.calls),
            )
            assert routed, _transcript_text(app)
            assert engagements == [1]
            text = _transcript_text(app)
            assert "but not attach one" not in text, text
            assert "agent reviewer is active" in text, text
    finally:
        server.close()


@pytest.mark.asyncio
async def test_credential_on_a_cold_viewer_opens_the_paste_only_after_binding(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX U3: never accept a secret and then drop it.

    The masked prompt must not appear until the runtime that will hold the
    value exists; once it does, the paste lands there. Asserted by length on
    the owner's side — the value is a placeholder and is never printed.
    """
    app, handle, server, engagements = _rig(isolated, monkeypatch)
    try:
        async with app.run_test(size=(110, 30)) as pilot:
            session = await _boot(app, pilot)
            await _paste_and_enter(app, pilot, "/credential DEMO_TOKEN")
            # The old path opened the prompt HERE, on a session with no store.
            # The invariant is "never open before bound" rather than "not open
            # yet": the mount engage (#622) may already have landed.
            if app._key_prompt is not None:
                assert session.is_cold is False, "a paste prompt opened on a cold viewer"
            assert "not reachable" not in _transcript_text(app), _transcript_text(app)

            opened = await _until(pilot, lambda: app._key_prompt is not None)
            assert opened, f"the prompt never opened; transcript:\n{_transcript_text(app)}"
            assert session.is_cold is False, "the prompt opens only on a bound viewer"

            placeholder = "x" * 20
            app.post_message(events.Paste(placeholder))
            await pilot.pause()
            await pilot.press("enter")
            stored = await _until(
                pilot, lambda: any(c[0] == "store" for c in handle.credential_calls)
            )
            assert stored, handle.credential_calls
            assert ("store", "DEMO_TOKEN", len(placeholder)) in handle.credential_calls
            text = _transcript_text(app)
            assert "Stored DEMO_TOKEN" in text, text
            assert placeholder not in text
    finally:
        server.close()


@pytest.mark.asyncio
async def test_listings_and_chart_stay_local_and_engage_nothing_extra(
    isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing never routes through the bind seam.

    Since #622 the MOUNT engages a runtime (that is the one engagement seen
    here); the property this pins is that `/team`, `/agent` and `/team chart`
    do not go through ``_needs_runtime_first`` — they neither wait for the
    runtime nor start a second one, and they render from local config
    whether or not the mount engage has landed.
    """
    from local_operator.teams import TeamEditFields, TeamRegistry

    TeamRegistry(isolated).create_team(
        TeamEditFields(name="lopdev", description="d", manager="manager")
    )
    app, handle, server, engagements = _rig(isolated, monkeypatch)
    try:
        async with app.run_test(size=(110, 30)) as pilot:
            session = await _boot(app, pilot)
            assert not any(
                app._needs_runtime_first(c, a)
                for c, a in (("/team", ""), ("/agent", ""), ("/team", "chart lopdev"))
            ), "a listing must not route through the bind seam"
            for line in ("/team", "/agent", "/team chart lopdev"):
                app._run_slash_command(line)
                await pilot.pause()
            # Rendered from local config immediately, while still cold.
            assert "lopdev" in _transcript_text(app)
            await _until(pilot, lambda: not session.is_cold)
            await asyncio.sleep(0.3)
            await pilot.pause()
            assert engagements == [1], "only the mount engage; the listings added none"
            assert not any(c[0] == "run_slash_authoritative" for c in handle.calls)
    finally:
        server.close()
