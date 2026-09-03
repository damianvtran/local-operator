from __future__ import annotations

import os
import time
from pathlib import Path

from local_operator.browser_bridge import state
from local_operator.browser_bridge.protocol import PROTO_VERSION


def record(**updates: object) -> state.BridgeState:
    values = {
        "pid": os.getpid(),
        "port": 4099,
        "session_key": "k" * 32,
        "proto": PROTO_VERSION,
        "extension_connected": True,
        "paired": True,
    }
    values.update(updates)
    return state.BridgeState.model_validate(values)


def test_state_is_private_atomic_and_available(tmp_path: Path) -> None:
    target = state.publish(record(), tmp_path)
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert state.available(tmp_path)


def test_availability_matrix(tmp_path: Path, monkeypatch) -> None:
    assert not state.available(tmp_path)
    state.publish(record(extension_connected=False), tmp_path)
    assert not state.available(tmp_path)
    # An unpaired but connected extension advertises the tool so execution can
    # return the actionable `lop browser pair` diagnostic from the protocol.
    state.publish(record(paired=False), tmp_path)
    assert state.available(tmp_path)
    current = record()
    state.publish(current, tmp_path)
    current.heartbeat_at = time.time() - state.HEARTBEAT_TIMEOUT_S - 1
    # publish refreshes the heartbeat by contract; write stale JSON directly to
    # isolate the reader's classification without weakening the publisher.
    state.state_path(tmp_path).write_text(current.model_dump_json())
    assert not state.available(tmp_path)
    monkeypatch.setattr(state, "pid_alive", lambda _pid: False)
    current.heartbeat_at = time.time()
    state.state_path(tmp_path).write_text(current.model_dump_json())
    assert not state.available(tmp_path)


def test_liveness_distinguishes_absent_fresh_and_stale(tmp_path: Path, monkeypatch) -> None:
    """The three-state classification RC2 depends on.

    ``available()`` collapses to a bool for the cheap session-construction
    gate, but the browser path needs to tell "no daemon" from "daemon alive,
    file stale" — only the latter is worth a socket probe.
    """
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT

    state.publish(record(), tmp_path)
    assert state.liveness(tmp_path)[0] is state.Liveness.FRESH

    current = record()
    state.publish(current, tmp_path)
    current.heartbeat_at = time.time() - state.HEARTBEAT_TIMEOUT_S - 1
    state.state_path(tmp_path).write_text(current.model_dump_json())
    status, read_back = state.liveness(tmp_path)
    assert status is state.Liveness.STALE
    assert read_back is not None
    # The age is what `lop browser status` renders and what the demotion hint
    # quotes, so it must be a real elapsed figure, not a flag.
    assert state.heartbeat_age(read_back) > state.HEARTBEAT_TIMEOUT_S

    # A stale file whose process is GONE is absent, not stale: there is
    # nothing alive to acquit, so no probe should ever be spent on it.
    monkeypatch.setattr(state, "pid_alive", lambda _pid: False)
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT


def test_heartbeat_age_never_reports_negative(tmp_path: Path) -> None:
    """Clock skew must not render as a negative age in diagnostics."""
    current = record(heartbeat_at=time.time() + 30)
    assert state.heartbeat_age(current) == 0.0
