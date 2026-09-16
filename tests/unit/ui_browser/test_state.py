"""The desktop app's browser-host discovery record, and its three-state predicate."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from local_operator.ui_browser import state


def record(**updates: object) -> state.UiHostState:
    values: dict[str, object] = {
        "pid": os.getpid(),
        "port": 52133,
        "session_key": "k" * 32,
        "proto": 1,
        "app_version": "0.21.0",
    }
    values.update(updates)
    return state.UiHostState.model_validate(values)


def test_record_is_private_and_in_its_own_namespace(tmp_path: Path) -> None:
    target = state.publish(record(), tmp_path)
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700
    # A namespace of its own, NOT beside the bridge daemon's file: `lop browser
    # status` and the daemon's install/cleanup paths all assume
    # `run/browser/bridge.json` is the daemon's, so a second process's record
    # there invites exactly the confusion a cleanup sweep would cause.
    assert target.relative_to(tmp_path).as_posix() == "run/ui-browser/host.json"
    assert not (tmp_path / "run" / "browser").exists()


def test_read_creates_nothing(tmp_path: Path) -> None:
    """Detection may never mutate the filesystem: the ENOSPC lesson, shared.

    `state_path` is pure path arithmetic because routing a reader through the
    writer's mkdir turned an ENOSPC handler into a second OSError raised from
    inside the first one.
    """
    before = sorted(p.name for p in tmp_path.iterdir())
    assert state.read(tmp_path) is None
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT
    assert state.state_path(tmp_path).exists() is False
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_liveness_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # absent
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT

    # malformed JSON
    path = state.state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert state.read(tmp_path) is None
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT

    # dead pid
    state.publish(record(pid=999_999_999), tmp_path)
    assert state.liveness(tmp_path)[0] is state.Liveness.ABSENT

    # fresh
    current = record()
    state.publish(current, tmp_path)
    assert state.liveness(tmp_path)[0] is state.Liveness.FRESH

    # stale heartbeat, live pid: genuinely unknown, so only STALE
    stale = state.read(tmp_path)
    assert stale is not None
    stale.heartbeat_at = time.time() - state.HEARTBEAT_TIMEOUT_S - 1
    path.write_text(stale.model_dump_json())
    assert state.liveness(tmp_path)[0] is state.Liveness.STALE
    assert state.available(tmp_path) is False
    assert state.advertisable(tmp_path) is True

    # proto mismatch is NOT absence: a version-skewed host is a real host that
    # can explain itself through the typed error, and unadvertising it would
    # leave the agent with no browser tool and no explanation for a running host.
    monkeypatch.setattr(state, "pid_alive", lambda _pid: True)
    skewed = record(proto=99)
    state.publish(skewed, tmp_path)
    assert state.liveness(tmp_path)[0] is state.Liveness.FRESH
    assert state.available(tmp_path) is True


def test_an_extension_connected_field_is_not_required(tmp_path: Path) -> None:
    """The one place this predicate must differ from the bridge's.

    `browser_bridge.state.liveness` folds `extension_connected` in, because the
    daemon's browser is a separate process the user can close while the daemon
    stays up. The UI host has no such third party — its views are its own
    children — so reusing that predicate verbatim would classify a perfectly
    healthy app as ABSENT forever.
    """
    state.publish(record(), tmp_path)
    assert state.available(tmp_path) is True
    raw = state.state_path(tmp_path).read_text()
    assert "extension_connected" not in raw


def test_ui_specific_fields_survive_the_read(tmp_path: Path) -> None:
    """`read` must not silently drop the host's own fields.

    `BridgeState` declares `extra="ignore"`, so reading this file through the
    bridge's model would parse cleanly and lose `host`, `app_version`,
    `profile_dir` and `agent_tabs` with no error at all.
    """
    state.publish(record(profile_dir="/tmp/profile", tabs=2, agent_tabs=1), tmp_path)
    current = state.read(tmp_path)
    assert current is not None
    assert current.host == "ui"
    assert current.app_version == "0.21.0"
    assert current.profile_dir == "/tmp/profile"
    assert current.tabs == 2 and current.agent_tabs == 1
