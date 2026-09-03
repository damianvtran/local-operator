"""Regression tests for the round-1 REVIEW remediation on PR #563 (F2/F3, Q1).

Each test here fails on the branch as it stood at review time. They guard the
three defects that reviewers found in the incident fix itself — the gaps were
in *reaching* the fix, not in the fix, which is exactly the kind of hole a unit
test closes cheaply and a manual pass keeps re-opening.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.browser_bridge import state
from local_operator.browser_bridge.daemon import BridgeService, ExtensionLink
from local_operator.browser_bridge.protocol import PROTO_VERSION
from local_operator.tools import builtin


def _record(**updates: Any) -> state.BridgeState:
    values: dict[str, Any] = {
        "pid": os.getpid(),
        "port": 4099,
        "session_key": "k" * 32,
        "proto": PROTO_VERSION,
        "extension_connected": True,
        "paired": True,
    }
    values.update(updates)
    return state.BridgeState.model_validate(values)


def _publish_stale(root: Path) -> None:
    """A daemon whose pid is alive but whose heartbeat writer has stopped."""
    current = _record()
    state.publish(current, root)
    current.heartbeat_at = time.time() - state.HEARTBEAT_TIMEOUT_S - 1
    # publish() refreshes the heartbeat by contract, so the stale JSON is
    # written directly rather than weakening the publisher for a test.
    state.state_path(root).write_text(current.model_dump_json())


# --------------------------------------------------------------------------
# F2 / Q2 — the stale-but-alive rescue must be REACHABLE on an extension-only
# host. Gating used the FRESH-only check, so a session on a host with no cmux
# got no browser tool at all and `execute_browser` — where the socket probe and
# the demotion diagnostic both live — was never reached.
# --------------------------------------------------------------------------


def test_stale_daemon_is_advertisable_even_though_it_is_not_available(tmp_path: Path) -> None:
    _publish_stale(tmp_path)
    status, _ = state.liveness(tmp_path)
    assert status is state.Liveness.STALE
    # The strict gate still says no: backend selection must keep asking for a
    # daemon that is known-good, and the socket probe does the acquitting.
    assert state.available(tmp_path) is False
    # The gating gate says yes, so the tool reaches execute_browser.
    assert state.advertisable(tmp_path) is True


def test_advertisable_is_false_when_the_daemon_is_really_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Advertising is weaker than availability, not unconditional."""
    assert state.advertisable(tmp_path) is False
    _publish_stale(tmp_path)
    monkeypatch.setattr(state, "pid_alive", lambda _pid: False)
    assert state.advertisable(tmp_path) is False
    state.publish(_record(extension_connected=False), tmp_path)
    assert state.advertisable(tmp_path) is False


def test_stale_but_alive_daemon_still_advertises_the_browser_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The finding itself: extension-only host + stale heartbeat => tool exists.

    Without the fix `build_browser_tool` returns None here, the session runs
    its whole life with no browser tool, and the agent cannot even discover the
    healthy daemon sitting behind the stale file.
    """
    _publish_stale(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)

    assert builtin.bridge_browser_available() is False
    assert builtin.bridge_browser_advertisable() is True
    tool = builtin.build_browser_tool(None)
    assert tool is not None
    assert tool.name == "browser"


def test_gating_opens_no_socket_and_spawns_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The constraint that made the gate file-only: it runs per session.

    Gating must stay synchronous and non-blocking, so it may not open a socket
    (the acquitting probe belongs on the browser path) — a regression here taxes
    the construction of every session on the machine.
    """
    import socket as socket_module

    _publish_stale(tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)

    def no_sockets(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("tool gating opened a socket")

    monkeypatch.setattr(socket_module, "socket", no_sockets)
    monkeypatch.setattr(socket_module, "create_connection", no_sockets)

    started = time.monotonic()
    assert builtin.build_browser_tool(None) is not None
    # Generous: the point is "no network round-trip", not a benchmark.
    assert time.monotonic() - started < 0.5


# --------------------------------------------------------------------------
# F3 — a REDACTED surface handle must never key a driven record.
# --------------------------------------------------------------------------


def test_redacted_handle_does_not_fork_a_second_driven_record() -> None:
    """`status` with no tab returns a redacted token; it must not become a key.

    Without the fix this leaves two records for one tab, and the redacted one
    survives the real close as a phantom advertising a dead URL — the exact
    ghost this PR exists to remove.
    """
    link = ExtensionLink()
    link.note_driven("bridge:7:abcdef0123456789", "https://example.com", "Example")
    # What a handle-less `status` reports back through the worker.
    link.note_driven("bridge:7:abcdef\u2026", "https://example.com", "Example")

    assert list(link.driven) == ["bridge:7:abcdef0123456789"]
    assert len(link.driven) == 1

    link.note_closed("bridge:7:abcdef0123456789")
    assert link.driven == {}
    assert link.current_url == ""


def test_redacted_close_does_not_silently_blank_every_record() -> None:
    """A redacted handle proves nothing, so it is treated as no handle."""
    link = ExtensionLink()
    link.note_driven("bridge:1:aaaaaaaaaaaa", "https://one.example", "One")
    link.note_driven("bridge:2:bbbbbbbbbbbb", "https://two.example", "Two")
    assert len(link.driven) == 2
    link.note_closed("bridge:9:cccccc\u2026")
    # Cannot name a surface => the safe reading is "nothing is provably driven"
    # rather than keeping entries that may be ghosts.
    assert link.driven == {}


def test_keyed_close_leaves_another_peers_unkeyed_record_alone() -> None:
    """Q3: the docstring promised isolation the code did not deliver.

    A handle-carrying close used to also pop the unkeyed record, so in a
    mixed-version pair one session closing its own tab blanked an OLDER
    extension's still-live entry.
    """
    link = ExtensionLink()
    link.note_driven("", "https://legacy.example", "Legacy")  # old extension
    link.note_driven("bridge:3:dddddddddddd", "https://new.example", "New")
    link.note_closed("bridge:3:dddddddddddd")
    assert list(link.driven) == [""]
    assert link.current_url == "https://legacy.example"


# --------------------------------------------------------------------------
# Q1 — the ENOSPC error path must not perform the write that is failing.
# --------------------------------------------------------------------------


def test_state_path_creates_nothing(tmp_path: Path) -> None:
    """Readers and diagnostics must not mkdir; only the writer may."""
    root = tmp_path / "fresh"
    path = state.state_path(root)
    assert path.name == state.STATE_FILENAME
    assert not root.exists(), "state_path() created the config dir"
    assert not path.parent.exists(), "state_path() created the run dir"
    # The writer still creates it, so publishing is unaffected.
    state.publish(_record(), root)
    assert path.exists()


def test_publish_safely_absorbs_enospc_on_a_config_dir_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1: building the ENOSPC log line used to raise a second ENOSPC.

    `state_path()` went through `run_dir()`, which mkdirs — so on a fresh
    config dir with a full disk the handler for "cannot write, disk is full"
    itself tried to write, and `publish_safely` raised from inside the branch
    whose whole contract is that it does not. The daemon then failed to boot,
    in exactly the disk-full scenario this change is about.
    """
    root = tmp_path / "never-created"
    service = BridgeService(root=root)

    def full_disk(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    # Fail the real write the way a full disk does, leaving the run dir absent.
    monkeypatch.setattr(state, "publish", full_disk)
    monkeypatch.setattr("local_operator.browser_bridge.daemon.state_store.publish", full_disk)

    assert service.publish_safely() is False  # absorbed, not raised
    assert service._publish_failures == 1
    assert not root.exists(), "the ENOSPC handler created the directory it could not write"
