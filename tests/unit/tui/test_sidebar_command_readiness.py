"""Every human owner mutation shares the saved-view readiness boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from local_operator.harness.types import ModelSpec
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from tests.unit.tui.test_app_pilot import FakeSession, _factory


class MutationAttempt(AssertionError):
    pass


class ControlSession(FakeSession):
    @property
    def model(self):
        return ModelSpec(provider="test", model_id="model", reasoning_effort="low")

    def abort(self, reason="interrupted"):
        raise MutationAttempt("unready abort reached the owner facade")

    def set_model(self, model, *, explicit=False):
        raise MutationAttempt("unready model mutation reached the owner facade")


@pytest.mark.parametrize("boundary", ["display_only", "command_frame_pending"])
@pytest.mark.parametrize("operation", ["interrupt", "effort", "fast", "model"])
def test_direct_owner_controls_refuse_both_readiness_boundaries(monkeypatch, boundary, operation):
    session = ControlSession()
    app = OperatorApp(lambda: _factory(session))
    app._session = session
    source = SessionInteraction(session)
    app._interaction = source
    app._interactions[id(session)] = source
    setattr(source, boundary, True)
    monkeypatch.setattr(app, "_notice", Mock())
    if operation == "interrupt":
        app._interrupt()
    elif operation == "effort":
        assert not app._apply_effort("high")
    elif operation == "fast":
        assert not app._apply_fast_mode(True)
    else:
        app._activate_resolved_model(session, "test", "other", session.model, False, app._notice)


def test_model_activation_rechecks_captured_source_after_selection_changed(monkeypatch):
    previous = ControlSession()
    current = FakeSession()
    app = OperatorApp(lambda: _factory(current))
    app._session = current
    app._interaction = SessionInteraction(current)
    app._interactions[id(previous)] = SessionInteraction(previous)
    monkeypatch.setattr(app, "_notice", Mock())
    app._activate_resolved_model(previous, "test", "other", previous.model, False, app._notice)


@pytest.mark.parametrize("boundary", ["display_only", "command_frame_pending"])
def test_aside_fork_does_not_schedule_unready_owner_work(monkeypatch, boundary):
    session = ControlSession()
    app = OperatorApp(lambda: _factory(session))
    app._session = session
    app._interaction = SessionInteraction(session)
    setattr(app._interaction, boundary, True)
    panel = SimpleNamespace(
        is_open=True, fork_messages=lambda: [("Question", "Answer")], set_notice=Mock()
    )
    monkeypatch.setattr(app, "_aside_panel", lambda: panel)
    scheduled = Mock(side_effect=lambda coroutine, **kwargs: coroutine.close())
    monkeypatch.setattr(app, "run_worker", scheduled)
    app.action_fork_aside()
    scheduled.assert_not_called()
    assert "Connect" in panel.set_notice.call_args.args[0]


def test_ready_control_can_reach_the_instrumented_owner(monkeypatch):
    session = ControlSession()
    app = OperatorApp(lambda: _factory(session))
    app._session = session
    app._interaction = SessionInteraction(session)
    monkeypatch.setattr(app, "_notice", Mock())
    with pytest.raises(MutationAttempt, match="model mutation"):
        app._apply_effort("high")
