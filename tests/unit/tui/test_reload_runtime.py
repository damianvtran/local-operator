"""Ownership and teardown boundaries for terminal-only replacement."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from local_operator.session.attached import AttachedSession
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def remote(tmp_path):
    return AttachedSession(
        config_dir=tmp_path, session_id="reload001", takeover_factory=AsyncMock()
    )


@pytest.mark.parametrize("state", ["streaming", "compacting", "loop"])
@pytest.mark.parametrize("ownership", ["remote", "local", "takeover", "unsaved"])
def test_relaunch_admission(tmp_path, monkeypatch, state, ownership):
    session = remote(tmp_path) if ownership != "local" else FakeSession()
    app = OperatorApp(lambda: _factory(FakeSession()))
    app._session = session
    if state == "streaming":
        if isinstance(session, AttachedSession):
            session._streaming = True
        else:
            session.streaming = True
    elif state == "compacting":
        app._compacting = True
    else:
        app._loop_running = True
    if ownership == "takeover":
        assert isinstance(session, AttachedSession)
        session._takeover_target = FakeSession()
    monkeypatch.setattr(
        app, "_resumable_session_id", lambda: "" if ownership == "unsaved" else "reload001"
    )
    refusal = app._relaunch_refusal()
    assert bool(refusal) == (ownership != "remote")
    if ownership == "unsaved":
        assert "saved" in refusal


def test_background_shell_blocks_relaunch(tmp_path):
    app = OperatorApp(lambda: _factory(FakeSession()))
    source = SessionInteraction(remote(tmp_path))
    source.shell.worker = SimpleNamespace(is_finished=False)
    app._interactions[id(source.session)] = source
    assert "local shell" in app._relaunch_refusal()
    source.shell.worker.is_finished = True
    assert app._relaunch_refusal() == ""


@pytest.mark.asyncio
async def test_detach_covers_background_sources_once(tmp_path):
    app = OperatorApp(lambda: _factory(FakeSession()))
    visible, background, takeover = [remote(tmp_path) for _ in range(3)]
    takeover._takeover_target = FakeSession()
    mocks = [AsyncMock() for _ in range(3)]
    for session, mock in zip((visible, background, takeover), mocks):
        session.detach_viewer_gates = mock
        app._interactions[id(session)] = SessionInteraction(session)
    app._session = visible
    await app._detach_relaunch_gates()
    mocks[0].assert_awaited_once_with(preserve_answers=True)
    mocks[1].assert_awaited_once_with(preserve_answers=True)
    mocks[2].assert_not_awaited()
