"""Pin cwd-move teardown's Darwin zombie race without timing a real exit.

The kernel can refuse killpg for an exited, unreaped child after Popen.poll
reported it live. Only a reaped child AND a vanished group justify recovery;
permission failures must never hide a living child or surviving descendants.
"""

from __future__ import annotations

import signal
import subprocess
from unittest.mock import Mock, call

import pytest

from tests.e2e.test_cwd_move_e2e import _reap


@pytest.mark.parametrize("returncode", [0, 1, -signal.SIGKILL])
def test_reap_accepts_permission_race_only_after_reaping_and_group_disappears(
    monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    child = Mock(spec=subprocess.Popen, pid=12345)
    child.poll.side_effect = [None, returncode]
    child.wait.return_value = returncode
    denied = PermissionError("Darwin unreaped zombie")

    def killpg(pid: int, sig: int) -> None:
        assert pid == child.pid
        if sig == signal.SIGKILL:
            raise denied
        assert sig == 0
        # The proof must follow reaping, not merely an exit observation.
        child.wait.assert_called_once_with(timeout=30)
        raise ProcessLookupError("reaped group is gone")

    kill = Mock(side_effect=killpg)
    monkeypatch.setattr("tests.e2e.test_cwd_move_e2e.os.killpg", kill)

    _reap(child)

    assert child.poll.call_count == 2
    child.wait.assert_called_once_with(timeout=30)
    assert kill.call_args_list == [call(child.pid, signal.SIGKILL), call(child.pid, 0)]


def test_reap_propagates_permission_error_for_a_living_child_that_never_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denied SIGKILL on a child that never exits is a failure, never accepted.

    The contract changed in PR #4003: Darwin EPERMs a group whose leader is
    mid-exit while ``poll()`` still reports it running, and a viewer-less
    self-move's runtime retires itself right after ``STAGE_HANDOFF``, so ``_reap``
    now waits ``exit_wait_s`` for the leader before judging the denial. This pins
    the guarantee that wait must not erode: a leader that is still running when
    the bound expires surfaces the ORIGINAL PermissionError, and no group probe
    runs, so there is no path by which a live group could be accepted.
    """
    child = Mock(spec=subprocess.Popen, pid=12345)
    child.poll.side_effect = [None, None]
    child.wait.side_effect = subprocess.TimeoutExpired(cmd="runtime", timeout=7.5)
    denied = PermissionError("live child cannot be killed")
    kill = Mock(side_effect=denied)
    monkeypatch.setattr("tests.e2e.test_cwd_move_e2e.os.killpg", kill)

    with pytest.raises(PermissionError) as failure:
        _reap(child, exit_wait_s=7.5)

    assert failure.value is denied
    # Only the bounded exit wait, distinct from the 30 s reap wait so the bound is
    # observably the one honoured.
    child.wait.assert_called_once_with(timeout=7.5)
    # No ``killpg(pid, 0)`` probe: the acceptance path is never reached.
    kill.assert_called_once_with(child.pid, signal.SIGKILL)


@pytest.mark.parametrize("group_survives", [False, True])
def test_reap_waits_out_an_exiting_group_then_requires_it_gone(
    monkeypatch: pytest.MonkeyPatch, group_survives: bool
) -> None:
    """Darwin's mid-exit EPERM is accepted only once the leader exits AND the group is gone.

    ``poll()`` stays None (the leader is in P_WEXIT, not yet a zombie) and the
    SIGKILL is denied. ``_reap`` waits for the leader, reaps it, then probes the
    group: ESRCH means the exit finished and teardown is done; a group that still
    answers the probe holds a descendant that may own the lease, so the original
    denial propagates.
    """
    child = Mock(spec=subprocess.Popen, pid=12345)
    child.poll.side_effect = [None, None]
    child.wait.return_value = 0
    denied = PermissionError("exiting group denied the signal")

    def killpg(pid: int, sig: int) -> None:
        assert pid == child.pid
        if sig == signal.SIGKILL:
            raise denied
        assert sig == 0
        # The probe must follow the leader's exit and its reaping.
        assert child.wait.call_args_list == [call(timeout=7.5), call(timeout=30)]
        if not group_survives:
            raise ProcessLookupError("exited group is gone")

    kill = Mock(side_effect=killpg)
    monkeypatch.setattr("tests.e2e.test_cwd_move_e2e.os.killpg", kill)

    if group_survives:
        with pytest.raises(PermissionError) as failure:
            _reap(child, exit_wait_s=7.5)
        assert failure.value is denied
    else:
        _reap(child, exit_wait_s=7.5)

    assert child.wait.call_args_list == [call(timeout=7.5), call(timeout=30)]
    assert kill.call_args_list == [call(child.pid, signal.SIGKILL), call(child.pid, 0)]


@pytest.mark.parametrize("probe_denied", [False, True])
def test_reap_rejects_surviving_group_or_persistent_permission_error(
    monkeypatch: pytest.MonkeyPatch, probe_denied: bool
) -> None:
    child = Mock(spec=subprocess.Popen, pid=12345)
    child.poll.side_effect = [None, 0]
    child.wait.return_value = 0
    denied = PermissionError("initial kill denied")
    probe_error = PermissionError("group probe denied") if probe_denied else None
    kill = Mock(side_effect=[denied, probe_error])
    monkeypatch.setattr("tests.e2e.test_cwd_move_e2e.os.killpg", kill)

    with pytest.raises(PermissionError) as failure:
        _reap(child)

    assert failure.value is (probe_error if probe_denied else denied)
    child.wait.assert_called_once_with(timeout=30)
    assert kill.call_args_list == [call(child.pid, signal.SIGKILL), call(child.pid, 0)]


def test_reap_still_kills_and_waits_for_a_live_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = Mock(spec=subprocess.Popen, pid=12345)
    child.poll.return_value = None
    child.wait.return_value = -signal.SIGKILL
    kill = Mock()
    monkeypatch.setattr("tests.e2e.test_cwd_move_e2e.os.killpg", kill)

    _reap(child)

    kill.assert_called_once_with(child.pid, signal.SIGKILL)
    child.wait.assert_called_once_with(timeout=30)
