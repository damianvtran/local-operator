"""A ``display.*`` write from another process moves a RUNNING runtime's folds.

Design review round 1 on #1746 (D1): the live fold runs in the SESSION process
(the runtime child for phone-started sessions; every ``exec`` run), reads the
process-cached display flags per decision (``note_peer_message``,
``_tool_row``), and nothing subscribed the config watcher — so a
``display.*`` edit by another process did not reach it at all and needed a
runtime restart. ``attach_display_config_watch`` is the seam; these tests
drive it and then the fold DECISIONS themselves, with the write shaped like
another process's (``settings_io._store``, below the notify hook) and
``poll_now()`` as the tick. No clock is involved.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.cross_session import cross_session_hidden
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.types import SessionProjection
from local_operator.session.runtime import serving

KEY = "display.hide_cross_session"


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture(autouse=True)
def _fresh_display_cache():
    from local_operator.tui.settings import settings_reload

    settings_reload()
    yield
    settings_reload()


@pytest.fixture(autouse=True)
def _fresh_attach_registry(monkeypatch: pytest.MonkeyPatch):
    """The attach guard is process state; each test needs a cold one."""
    monkeypatch.setattr(serving, "_display_watch_attached", set())


def _write_elsewhere(config_dir: Path, key: str, value: object) -> None:
    """A write shaped like another process's: a fresh manager, no notify hook."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io._store(ConfigManager(config_dir), setting.path, value)


def _fold() -> ProjectionFold:
    return ProjectionFold(
        SessionProjection(
            session_id="prop",
            pid=0,
            kind="tui",
            conversation_name="prop",
            cwd="/tmp",
            model_label="test/model",
        )
    )


@pytest.mark.asyncio
async def test_a_display_write_moves_the_live_fold_without_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decisions that run AFTER the write follow it; earlier rows stay.

    Forward-only is the shared contract with the TUI (``tui/settings.py``
    ``_DEFAULT_NOTES``): a row already painted is never torn down by a flip,
    and a re-seed under the new value is what applies it backwards.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    serving.attach_display_config_watch(config_dir)

    fold = _fold()
    # Control: flag off — the live decisions paint.
    fold.note_peer_message("first", sender={"pid": 7})
    assert fold.projection.transcript[-1].kind == "peer_message"
    assert fold._tool_row("cs1", "send") is not None

    _write_elsewhere(config_dir, KEY, True)
    change = process_watcher(config_dir).poll_now()
    assert change is not None and KEY in change.changed_keys

    # No restart: the next decisions read the new value, and the rows painted
    # above are exactly as they were (forward-only).
    fold.note_peer_message("second", sender={"pid": 7})
    assert [entry.kind for entry in fold.projection.transcript] == ["peer_message", "tool"]
    assert fold._tool_row("cs2", "send") is None
    # The gate is exact: a non-send call still mints its row.
    assert fold._tool_row("r1", "read") is not None


@pytest.mark.asyncio
async def test_the_attach_seed_reflects_the_new_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-seed (what a re-engage/dial re-fold runs) reads the flag current.

    ``fold_history`` is the attach seed path; this is the "reconnect shows the
    new value" half of the acceptance — the fold that runs after the write
    drops both hidden kinds.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
    from local_operator.harness.types import (
        CustomMessage,
        Message,
        TextContent,
        ToolCall,
    )

    history = [
        Message.user("go", id="u1"),
        CustomMessage(
            custom_type=PEER_MESSAGE_MESSAGE_TYPE,
            attribution="user",
            details={"text": "<wrapped>hi</wrapped>", "body": "hi there", "sender": {}},
        ),
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[ToolCall(id="cs1", name="send", arguments={"text": "hi"})],
            stop_reason="toolUse",
        ),
    ]

    serving.attach_display_config_watch(config_dir)
    fold = _fold()
    fold.fold_history(list(history))
    assert [entry.kind for entry in fold.projection.transcript] == [
        "user",
        "peer_message",
        "tool",
    ]

    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None

    reseeded = _fold()
    reseeded.fold_history(list(history))
    assert [entry.kind for entry in reseeded.projection.transcript] == [
        "user"
    ], "a re-seed after the write must drop the peer receipt and the send row"


@pytest.mark.asyncio
async def test_the_follower_is_idempotent_per_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One subscription per directory: a second attach must not stack.

    The settings cache is process-global, so a stacked listener would only run
    the same drop twice per change. The count is the observable.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    import local_operator.tui.settings as settings_mod

    reloads: list[int] = []
    real = settings_mod.settings_reload

    def counted() -> None:
        reloads.append(1)
        real()

    monkeypatch.setattr(settings_mod, "settings_reload", counted)

    serving.attach_display_config_watch(config_dir)
    serving.attach_display_config_watch(config_dir)

    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None
    assert len(reloads) == 1
    assert cross_session_hidden() is True


@pytest.mark.asyncio
async def test_spawn_owned_session_attaches_the_follower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child runtime that serves a phone-started session follows display.*.

    ``spawn_owned_session`` is that child's only constructor; the session is
    stubbed (the subject is the attach, not the composition root) so the test
    stays cheap and deterministic.
    """
    from local_operator import session_factory
    from tests.unit.session.runtime.test_exec_control import FakeSession

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    async def fake_create_session(*args: object, **kwargs: object) -> FakeSession:
        return FakeSession()

    monkeypatch.setattr(session_factory, "create_session", fake_create_session)

    handle = await serving.spawn_owned_session(
        asyncio.get_running_loop(), cwd=str(tmp_path), provider="test", model_id="mock"
    )
    try:
        assert cross_session_hidden() is False
        _write_elsewhere(config_dir, KEY, True)
        assert process_watcher(config_dir).poll_now() is not None
        assert cross_session_hidden() is True
    finally:
        dispose = getattr(handle, "dispose", None)
        if callable(dispose):
            maybe = dispose()
            if asyncio.iscoroutine(maybe):
                await maybe
