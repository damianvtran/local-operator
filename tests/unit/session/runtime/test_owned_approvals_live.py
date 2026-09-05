"""``OwnedSessionHandle``'s approval gate follows ``tool_approval_mode`` live.

The gate the RUNTIME's tools actually consult is ``_auto_approve`` on the
handle (``_install_gates``), read per decision. ``follow_config`` hangs a
listener on the process ``ConfigWatcher`` so a ``config.yml`` write from any
process moves it within a poll — in both directions, overriding a per-session
``/approvals`` toggle — while two limits hold: a card already PARKED is never
auto-answered or auto-denied, and an explicit ``--yolo`` pin ignores the key.

``poll_now()`` is the tick; nothing here waits on the clock.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import ConfigWatcher, _reset_for_tests
from local_operator.session.runtime.owned import OwnedSessionHandle
from tests.unit.session.runtime.test_owned import FakeSession


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _write_elsewhere(config_dir, key: str, value: Any) -> None:
    """A write shaped like another process's: below the notify hook."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io._store(ConfigManager(config_dir), setting.path, value)


def _handle(
    tmp_path, *, auto_approve: bool, pinned: bool = False
) -> tuple[OwnedSessionHandle, FakeSession, ConfigWatcher, list[Any]]:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value(
        "tool_approval_mode", "auto" if auto_approve else "ask"
    )
    session = FakeSession()
    emitted: list[Any] = []

    async def _emit(event: object) -> None:
        emitted.append(event)

    session._emit = _emit
    handle = OwnedSessionHandle(
        session,
        asyncio.get_running_loop(),
        cwd=str(tmp_path),
        auto_approve=auto_approve,
        approval_pinned=pinned,
    )
    watcher = ConfigWatcher(config_dir)
    handle.follow_config(watcher)
    return handle, session, watcher, emitted


@pytest.mark.asyncio
async def test_a_disk_write_loosens_the_gate_at_the_next_decision(tmp_path) -> None:
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=False)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    change = watcher.poll_now()
    assert change is not None and "tool_approval_mode" in change.changed_keys

    assert handle._auto_approve is True
    # The NEXT decision answers inline — no card parked.
    assert await handle._approval_gate("bash", "rm -rf build/") is True
    assert handle._fold.projection.pending is None
    # The receipt comes from the process that owns the gate.
    await asyncio.sleep(0)
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("tool approvals: auto" in t and "config.yml changed" in t for t in texts), texts
    assert [getattr(e, "kind", "") for e in emitted if "auto" in getattr(e, "text", "")] == [
        "warning"
    ]
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_disk_write_tightens_the_gate_and_the_next_decision_parks(tmp_path) -> None:
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=True)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()

    assert handle._auto_approve is False
    pending = asyncio.ensure_future(handle._approval_gate("write", "a file"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None, "the tightened gate did not park a card"
    request_id = handle._fold.projection.pending.request_id
    await handle.approval_answer(request_id, False, False)
    assert await pending is False
    await asyncio.sleep(0)
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("tool approvals: ask" in t and "prompt again" in t for t in texts), texts
    await handle.dispose()


@pytest.mark.asyncio
async def test_the_disk_overrides_a_per_session_approvals_toggle(tmp_path) -> None:
    """A pane the operator put in ``ask`` with ``/approvals ask`` flips to
    ``auto`` when the FILE says auto: the disk write is the machine-wide
    intent, and the alternative makes "all my agents" false in exactly the
    case the operator described. ``/approvals ask`` restores it in one step."""
    handle, session, watcher, _emitted = _handle(tmp_path, auto_approve=False)

    from local_operator.session.frontend_state import SlashResult

    # /approvals auto in this session, then /approvals ask again — per-session.
    handle._approvals_slash(session, "auto", SlashResult)
    assert handle._auto_approve is True
    handle._approvals_slash(session, "ask", SlashResult)
    assert handle._auto_approve is False

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    assert handle._auto_approve is True
    handle._approvals_slash(session, "ask", SlashResult)
    assert handle._auto_approve is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_parked_prompt_is_left_for_the_human(tmp_path) -> None:
    """The gate reads the flag when a DECISION is made. A card already on
    screen when the file loosens is neither auto-approved nor dismissed; a
    card on screen when it tightens is not auto-denied. The human answers."""
    handle, _session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    parked = asyncio.ensure_future(handle._approval_gate("bash", "touch /tmp/x"))
    await asyncio.sleep(0)
    pending = handle._fold.projection.pending
    assert pending is not None

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is True
    assert not parked.done(), "a loosening auto-answered a card the human was looking at"
    assert handle._fold.projection.pending is not None
    assert handle._fold.projection.pending.request_id == pending.request_id

    # The human decides, and only then does the future settle.
    await handle.approval_answer(pending.request_id, False, False)
    assert await parked is False
    # A brand-new decision after the human's answer follows the file.
    assert await handle._approval_gate("bash", "next") is True
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_yolo_pin_ignores_the_key(tmp_path) -> None:
    """``lop exec --control --yolo``: an explicit flag on this run outranks a
    default in a file. Nothing moves, nothing is announced."""
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=True, pinned=True)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    assert handle._auto_approve is True
    assert await handle._approval_gate("bash", "anything") is True
    await asyncio.sleep(0)
    assert emitted == []
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_no_op_write_and_an_unknown_mode_leave_the_gate_alone(tmp_path) -> None:
    handle, _session, watcher, emitted = _handle(tmp_path, auto_approve=False)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "sometimes")
    watcher.poll_now()
    assert handle._auto_approve is False
    await asyncio.sleep(0)
    assert emitted == []
    await handle.dispose()


@pytest.mark.asyncio
async def test_dispose_unsubscribes_and_follow_config_is_idempotent(tmp_path) -> None:
    handle, _session, watcher, _emitted = _handle(tmp_path, auto_approve=False)
    handle.follow_config(watcher)
    assert len(watcher._listeners) == 1
    await handle.dispose()
    assert watcher._listeners == []
    # A tick after dispose reaches nothing.
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    assert handle._auto_approve is False
