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
async def test_a_loosening_does_not_revoke_an_explicit_per_session_ask(tmp_path) -> None:
    """THE ASYMMETRIC RULE, loosening half (review R1, UX U1).

    A human who typed ``/approvals ask`` in this session keeps that gate when
    the FILE later says ``auto``, and reads a keep notice naming the way to
    adopt the file. The operator asked for settings to REACH running sessions;
    they did not ask for a file write to revoke a hardening a human typed into
    a specific pane. This mirrors the model half of the same change
    (``Session._explicit_model_choice`` and its ``keeping …`` notice) on the
    more dangerous of the two keys.
    """
    handle, session, watcher, emitted = _handle(tmp_path, auto_approve=False)

    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "ask", SlashResult)
    assert handle._auto_approve is False
    emitted.clear()

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)

    assert handle._auto_approve is False, "a file write revoked a hardening the human typed"
    # The gate really is still armed, not merely flagged: a decision parks.
    parked = asyncio.ensure_future(handle._approval_gate("bash", "rm -rf build/"))
    await asyncio.sleep(0)
    assert handle._fold.projection.pending is not None
    await handle.approval_answer(handle._fold.projection.pending.request_id, False, False)
    assert await parked is False

    texts = [getattr(e, "text", "") for e in emitted]
    assert any(
        "keeping tool approvals: ask" in t and "/approvals auto adopts it" in t for t in texts
    ), texts
    # And the named way out works in one step.
    handle._approvals_slash(session, "auto", SlashResult)
    assert handle._auto_approve is True
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_tightening_follows_the_file_even_over_an_explicit_choice(tmp_path) -> None:
    """THE ASYMMETRIC RULE, tightening half. Safety propagates without
    exception: a session that explicitly chose ``auto`` still follows the file
    to ``ask``, because a user who ends up safer than they asked is never the
    wrong surprise. This is the direction the rule does NOT make conditional."""
    # File AND gate start at `auto`, so the disk write below is a real
    # transition; the session's own `/approvals auto` is what records the
    # explicit choice the tightening then has to override.
    handle, session, watcher, emitted = _handle(tmp_path, auto_approve=True)

    from local_operator.session.frontend_state import SlashResult

    handle._approvals_slash(session, "auto", SlashResult)
    assert handle._auto_approve is True
    assert handle._explicit_approvals_mode == "auto"
    emitted.clear()

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is False, "a tightening was refused; safety must always propagate"
    texts = [getattr(e, "text", "") for e in emitted]
    assert any("tool approvals: ask" in t and "config.yml changed" in t for t in texts), texts

    # ...AND THE SESSION CAN STILL FOLLOW THE FILE BACK (review round 2, R6).
    # The runtime half of the same regression: the only mode this human ever
    # typed is `auto`, so there is no hardening for the loosening guard to
    # protect, and a guard that read merely "this session chose something"
    # pinned it to `ask` permanently. The FILE owns the value once it moves the
    # gate, which is why the recorded mode is cleared — that is also what keeps
    # the keep notice's "set with /approvals in this session" a true statement.
    assert (
        handle._explicit_approvals_mode is None
    ), "a file write left the session claiming the human had typed the mode"
    emitted.clear()
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    await asyncio.sleep(0)
    assert handle._auto_approve is True, (
        "a session whose human only ever chose `auto` stayed pinned to `ask` after a "
        "file tightening; no later file write could ever move it back (R6)"
    )
    texts = [getattr(e, "text", "") for e in emitted]
    assert not any("keeping tool approvals" in t for t in texts), texts
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_session_that_never_chose_follows_the_file_in_both_directions(tmp_path) -> None:
    """The operator's own case, unchanged: a pane nobody typed ``/approvals``
    into is exactly the "goes into effect for all my agents" pane, and it
    follows the file loosening AND tightening. The keep rule is scoped to a
    deliberate in-session choice and must not leak into this path."""
    handle, _session, watcher, _emitted = _handle(tmp_path, auto_approve=False)

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    watcher.poll_now()
    assert handle._auto_approve is True

    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "ask")
    watcher.poll_now()
    assert handle._auto_approve is False
    await handle.dispose()


@pytest.mark.asyncio
async def test_a_bare_approvals_reports_a_divergence_against_the_file(
    tmp_path, monkeypatch
) -> None:
    """The reporting half (UX U1 step 3 / U2). With the asymmetric rule a
    session can legitimately hold a mode the file disagrees with, so the one
    surface whose job is "what is in effect and why" compares against the FILE
    rather than a cached default — otherwise it reports a matched pair for
    exactly the state it exists to disclose."""
    handle, session, watcher, _emitted = _handle(tmp_path, auto_approve=False)

    from local_operator.config_watch import process_watcher
    from local_operator.session.frontend_state import SlashResult

    # The report reads the REGISTERED watcher's last-good snapshot (never a
    # fresh `ConfigManager`, which could move a malformed file aside from a
    # print path) and resolves it through `paths.config_dir()`, so the env var
    # has to name this scratch dir. `_handle`'s own bare watcher stays the one
    # driving the gate; this one is what the REPORT reads.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(watcher.config_dir))
    registered = process_watcher(watcher.config_dir)

    handle._approvals_slash(session, "ask", SlashResult)
    _write_elsewhere(watcher.config_dir, "tool_approval_mode", "auto")
    registered.poll_now()
    watcher.poll_now()  # the gate's own watcher; the keep rule holds `ask`

    reported = handle._approvals_slash(session, "", SlashResult)
    text = getattr(reported, "text", "")
    assert "tool approvals: ask (this session)" in text, text
    assert "config.yml says auto" in text, text
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
