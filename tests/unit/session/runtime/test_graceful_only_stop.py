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


@pytest.mark.asyncio
async def test_acked_late_exit_is_pending_not_refused(tmp_path, monkeypatch):
    """An accepted stop that outlives the wait is a stop, not a refusal.

    Agent review round 1 D1 / UX round 1 U1: the owner latched its retirement
    and ran ``request_stop``, so the marker records a deliberate stop; reporting
    "refused" made a reader plan around a session that was already leaving, and
    withdrawing the marker recorded the user's own stop as an unexplained exit.
    """
    record = _bare_record()
    monkeypatch.setattr(control, "_exchange", AsyncMock(return_value={"op": "ack", "req": 1}))
    monkeypatch.setattr(control, "_await_stopped", AsyncMock(return_value=False))
    marker = Mock()
    withdraw = Mock(side_effect=AssertionError("an acked stop keeps its marker"))
    monkeypatch.setattr(control, "_write_stop_marker", marker)
    monkeypatch.setattr(control, "_withdraw_staged_stop_marker", withdraw)
    # Parked, because a runtime that is leaving will not fire its schedules.
    park = AsyncMock(return_value=2)
    monkeypatch.setattr(control, "_park_wakes", park)
    monkeypatch.setattr(control, "_park_monitors", park)
    monkeypatch.setattr(registry, "pid_alive", lambda pid: True)
    signal = AsyncMock(side_effect=AssertionError("idle-only stop must never signal"))
    monkeypatch.setattr(control, "_signal_and_confirm", signal)

    outcome = await control.stop_session(record, timeout_s=1, _root=tmp_path, graceful_only=True)

    assert outcome.method == "pending"
    assert outcome.reason == ""
    assert outcome.wakes_dormant == 2
    assert marker.call_count == 1
    withdraw.assert_not_called()
    signal.assert_not_awaited()
    # The receipt says the act happened; it does not say it failed.
    assert "accepted" in outcome.line
    assert "refused" not in outcome.line


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply, reason",
    [
        (None, control.REASON_UNANSWERED),
        ({"op": "error", "req": 1, "reason": "busy"}, control.REASON_BUSY),
        ({"op": "error", "req": 1, "reason": "wake"}, control.REASON_WAKE),
        ({"op": "error", "req": 1, "reason": "old_owner"}, control.REASON_OLD_OWNER),
        # An owner too old to send the field is not invented into a cause.
        ({"op": "error", "req": 1}, control.REASON_BUSY),
    ],
)
async def test_refusal_carries_the_owner_reason(tmp_path, monkeypatch, reply, reason):
    """One refusal sentence per cause, because each has its own next step.

    UX round 1 U2 / design round 1 D2: busy, a wake inside the hold window, an
    older owner and silence all rendered identically and named no action, even
    though the owner already said which one applied.
    """
    monkeypatch.setattr(control, "_exchange", AsyncMock(return_value=reply))
    monkeypatch.setattr(registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(control, "_write_stop_marker", Mock())
    monkeypatch.setattr(control, "_withdraw_staged_stop_marker", Mock())
    signal = AsyncMock(side_effect=AssertionError("idle-only stop must never signal"))
    monkeypatch.setattr(control, "_signal_and_confirm", signal)

    outcome = await control.stop_session(
        _bare_record(), timeout_s=1, _root=tmp_path, graceful_only=True
    )

    assert outcome.method == "refused"
    assert outcome.reason == reason
    signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_if_idle_refusal_names_its_cause_on_the_wire(monkeypatch):
    """The cause reaches the caller as a machine token, not as prose to parse.

    The owner is the only party that knows WHY it refused, and a second probe by
    the caller would describe state the deciding sample never saw.
    """
    handle = _IdleStopHandle(busy=True)
    server, record = await _serve(handle)
    try:
        reply = await control._exchange(
            _record_for(record), {"op": "stop_if_idle"}, reply_timeout_s=5
        )
    finally:
        server.close()
    assert reply is not None
    assert reply["op"] == "error"
    assert reply["reason"] == control.REASON_BUSY
    assert handle.stops == []


@pytest.mark.asyncio
async def test_stop_if_idle_reports_a_wake_due_as_its_own_cause(monkeypatch):
    """A wake inside the hold window is not "busy" and must not say so.

    The owner is idle in that case, and the same stop succeeds once the wake has
    fired, so the retry advice is different (UX round 1 U2).
    """
    handle = _IdleStopHandle(busy=False)
    monkeypatch.setattr(handle, "may_refresh", lambda: "wake due within the warm window")
    server, record = await _serve(handle)
    try:
        reply = await control._exchange(
            _record_for(record), {"op": "stop_if_idle"}, reply_timeout_s=5
        )
    finally:
        server.close()
    assert reply is not None
    assert reply["reason"] == control.REASON_WAKE
    assert handle.stops == []


@pytest.mark.asyncio
async def test_draining_skip_offers_the_tool_no_force_flag(tmp_path, monkeypatch):
    """A surface must not name an action it cannot accept.

    Design round 1 D3: ``_command="sessions tool"`` does not start with "/", so
    the draining receipt appended the CLI's ``--force`` to an agent that has no
    such parameter and whose contract forbids the escalation.
    """
    record = _bare_record()
    record.leaving = "leaves at its next turn boundary"
    monkeypatch.setattr(registry, "pid_alive", lambda pid: True)
    monkeypatch.setattr(control, "_drain_stalled", lambda rec: "")
    monkeypatch.setattr(control, "_held_unconfirmed", lambda rec: "")
    outcome = await control.stop_session(
        record,
        _root=tmp_path,
        graceful_only=True,
        _command="sessions tool",
    )
    assert outcome.method == "draining"
    assert "--force" not in outcome.line
    assert "nothing to do" in outcome.line
