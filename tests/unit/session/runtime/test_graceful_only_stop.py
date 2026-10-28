"""A tool-authorized graceful stop cannot inherit the CLI's signal ladder."""

from unittest.mock import AsyncMock, Mock

import pytest

from local_operator.paths import config_dir
from local_operator.session.runtime import control, registry
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_control import _bare_record, _record_for, _serve
from tests.unit.session.runtime.test_server import FakeHandle


@pytest.mark.asyncio
async def test_graceful_only_never_escalates_an_unanswered_owner(tmp_path, monkeypatch):
    record = _bare_record()
    exchange = AsyncMock(return_value=None)
    signal = AsyncMock(return_value=False)
    monkeypatch.setattr(control, "_exchange", exchange)
    monkeypatch.setattr(control, "_signal_and_confirm", signal)
    monkeypatch.setattr(registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(control, "_confirmed_session_id", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(control, "_write_stop_marker", Mock())
    monkeypatch.setattr(control, "_withdraw_staged_stop_marker", Mock())
    outcome = await control.stop_session(record, _root=tmp_path, graceful_only=True)
    assert outcome.method == "refused"
    assert exchange.call_args.args[1] == {"op": "stop_if_idle"}
    signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_graceful_only_cannot_combine_with_force(tmp_path, monkeypatch):
    exchange = AsyncMock()
    monkeypatch.setattr(control, "_exchange", exchange)
    outcome = await control.stop_session(
        _bare_record(), _root=tmp_path, graceful_only=True, force=True
    )
    assert outcome.method == "refused"
    exchange.assert_not_awaited()


class _IdleStopHandle(FakeHandle):
    # Exercise the production atomic latch, not a test reimplementation of it.
    begin_retire = ServingSessionHandle.begin_retire
    # The borrowed latch reads and writes these on ``self``. FakeHandle does not
    # declare them, so pyright cannot see the access and the type-check job went
    # red on this new file. The values match the real handle's defaults.
    _exit_committed: bool = False
    _disposing: bool = False

    def __init__(self, busy):
        super().__init__()
        self.busy = busy
        self.stops = []

    def may_refresh(self):
        return "busy" if self.busy else ""

    def request_stop(self):
        assert self._exit_committed
        self.stops.append(True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_real_socket_stop_checks_and_latches_owner_idle(busy, monkeypatch):
    handle = _IdleStopHandle(busy)
    server, record = await _serve(handle)
    signals = AsyncMock(side_effect=AssertionError("idle-only stop must never signal"))
    monkeypatch.setattr(control, "_signal_and_confirm", signals)
    monkeypatch.setattr(
        control, "_await_stopped", AsyncMock(side_effect=lambda *a: bool(handle.stops))
    )
    try:
        outcome = await control.stop_session(
            _record_for(record), timeout_s=1, _root=config_dir(), graceful_only=True
        )
        assert outcome.method == ("refused" if busy else "socket")
        assert handle.stops == ([] if busy else [True])
        assert bool(getattr(handle, "_exit_committed", False)) is not busy
        signals.assert_not_awaited()
    finally:
        server.close()


@pytest.mark.asyncio
async def test_real_socket_old_owner_is_refused_without_fallback(monkeypatch):
    # An owner without the latch cannot attest that its stop preserves work.
    handle = FakeHandle()
    stop = Mock(side_effect=AssertionError("must not stop old owner"))
    monkeypatch.setattr(handle, "request_stop", stop, raising=False)
    server, record = await _serve(handle)
    signals = AsyncMock(side_effect=AssertionError("must not signal old owner"))
    monkeypatch.setattr(control, "_signal_and_confirm", signals)
    try:
        outcome = await control.stop_session(
            _record_for(record), timeout_s=1, _root=config_dir(), graceful_only=True
        )
        assert outcome.method == "refused"
        stop.assert_not_called()
        signals.assert_not_awaited()
    finally:
        server.close()


@pytest.mark.asyncio
async def test_legacy_force_false_is_not_a_no_signal_policy(tmp_path, monkeypatch):
    # Negative control for the original bug: the unchanged CLI path still
    # escalates with force=False. No OS signal is delivered by this test.
    sent = AsyncMock(return_value=False)
    monkeypatch.setattr(control, "_exchange", AsyncMock(return_value=None))
    monkeypatch.setattr(control, "_signal_and_confirm", sent)
    monkeypatch.setattr(control, "_confirmed_session_id", AsyncMock(return_value=(True, "")))
    monkeypatch.setattr(registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(control, "_write_stop_marker", Mock())
    monkeypatch.setattr(control, "_withdraw_staged_stop_marker", Mock())
    await control.stop_session(_bare_record(), _root=tmp_path, force=False)
    assert sent.await_count == 2
    assert sent.await_args_list[-1].args[1] == control.procstate.hard_kill_signal()
