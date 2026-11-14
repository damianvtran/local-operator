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


def test_reap_propagates_permission_error_for_living_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = Mock(spec=subprocess.Popen, pid=12345)
    child.poll.side_effect = [None, None]
    denied = PermissionError("live child cannot be killed")
    kill = Mock(side_effect=denied)
    monkeypatch.setattr("tests.e2e.test_cwd_move_e2e.os.killpg", kill)

    with pytest.raises(PermissionError) as failure:
        _reap(child)

    assert failure.value is denied
    child.wait.assert_not_called()
    kill.assert_called_once_with(child.pid, signal.SIGKILL)


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
