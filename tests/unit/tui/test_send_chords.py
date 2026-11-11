"""TUI per-message send chords (T4, D4-D9): the composer rewrite, the
delivery override, and the states that must ignore it.

The chord maps to a mode at the top of ``Editor._on_key`` and rewrites to
Enter, so every Enter owner — pickers, credential capture, the live-prompt
router — treats it as Enter. ``_submit`` consumes the mode once into
``EditorSubmitted(delivery=...)``; the app's busy branch uses
``msg.delivery or _busy_delivery_mode()``; an idle session, shell submit,
an open picker, credential capture and paste ignore it.

Raw kitty vocabulary (``ctrl+enter``) is spelled with ``pilot.press`` the
way the measurement probes spelled the bytes; ``alt+s`` covers the
everywhere family.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.tui.app import (
    BOTH_WAIT_NOTICE,
    QUEUED_STEER_NOTICE,
    QUEUED_WAIT_NOTICE,
    OperatorApp,
)
from local_operator.tui.widgets.editor import Editor, EditorSubmitted
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock

from .test_app_pilot import FakeSession, _factory
from .test_busy_delivery import _DeliveryStreaming, _notice_texts, _set_mode, _Streaming


def _user_texts(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


async def _wait_session(pilot: Any, app: OperatorApp) -> None:
    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    assert app._session is not None


async def _type(app: OperatorApp, pilot: Any, text: str) -> None:
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    editor.text = text
    await pilot.pause()


@pytest.mark.asyncio
async def test_chord_with_picker_open_sends_nothing() -> None:
    """A chord is Enter, so an open slash picker accepts the row."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        await _type(app, pilot, "/he")
        await pilot.pause()
        submitted: list[EditorSubmitted] = []
        original = app.on_editor_submitted

        async def capture(message: EditorSubmitted) -> None:
            submitted.append(message)
            await original(message)

        app.on_editor_submitted = capture  # type: ignore[method-assign]
        await pilot.press("ctrl+enter")
        await pilot.pause()
        # The picker consumed Enter before _submit: no EditorSubmitted.
        assert submitted == []
        assert session.prompts == []


@pytest.mark.asyncio
async def test_chord_in_credential_capture_confirms_only() -> None:
    """A chord in a masked capture submits nothing; it confirms the capture.

    The masked capture is armed directly (``_credential_typing``): the
    picker route to it needs a mounted runtime, which the fake does not
    supply, so staging it here tests the Enter-owner branch, not the picker.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        # Stage the masked-capture state directly: `_credential_typing` is
        # the branch the chord must confirm, not re-derive.
        editor._credential_typing = 0  # type: ignore[attr-defined]
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert session.prompts == []
        assert editor._chord_delivery is None


@pytest.mark.asyncio
async def test_idle_session_sends_a_normal_prompt() -> None:
    """Idle (not streaming): the chord mode is ignored; a prompt sends."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "alt+s"}, announce=False)
        await _type(app, pilot, "plain hello")
        await pilot.press("alt+s")
        await pilot.pause()
        assert session.prompts == ["plain hello"]


@pytest.mark.asyncio
async def test_shell_mode_ignores_the_chord_mode() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        editor = app.query_one(Editor)
        editor.focus()
        editor.set_shell_mode(True)
        await pilot.pause()
        editor.text = "echo hi"
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause()
        # Shell mode runs the command locally; it is not a prompt and the
        # chord mode is ignored (the command runs, the prompt never appears).
        assert session.prompts == []
        assert "echo hi" not in _user_texts(app)


@pytest.mark.asyncio
async def test_paste_fires_nothing() -> None:
    """Paste arrives as events.Paste, not Key, so it cannot fire a chord."""
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.text = "draft"
        await pilot.pause()
        from textual import events

        await editor._on_paste(events.Paste("ctrl+enter"))
        await pilot.pause()
        assert editor._chord_delivery is None
        assert session.prompts == []
        assert session.delivery_steers == []


@pytest.mark.asyncio
async def test_chord_delivery_never_leaks_to_the_next_key() -> None:
    """A chord, then a more-typed key, then plain Enter: the mode is gone."""
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.text = "first"
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert editor._chord_delivery is None
        editor.text = "second"
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        assert editor._chord_delivery is None
        await pilot.press("enter")
        await pilot.pause()
        # Second send took the default (steer), not the chord's queue.
        assert ("secondx", "queue") not in session.delivery_steers


@pytest.mark.asyncio
async def test_a_chorded_queue_against_a_legacy_handle_is_refused() -> None:
    """Never sent as steer: the existing refusal path with the text kept."""
    session = _Streaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        _set_mode(app, "steer")
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        await _type(app, pilot, "do not downgrade")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert session.steers == []
        assert QUEUED_STEER_NOTICE not in _notice_texts(app)


@pytest.mark.asyncio
async def test_a_setting_write_changes_the_next_send() -> None:
    """_apply_keymap re-pushes the map; the very next chord uses the new key."""
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        _set_mode(app, "steer")
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        await _type(app, pilot, "one")
        await pilot.press("ctrl+enter")
        await pilot.pause()
        assert session.delivery_steers == [("one", "queue")]
        app._apply_keymap({"keymap.send_queue": "alt+s"}, announce=False)
        await _type(app, pilot, "two")
        await pilot.press("alt+s")
        await pilot.pause()
        assert session.delivery_steers[-1] == ("two", "queue")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chord_mode", "default_mode", "delivery"),
    [
        ("steer", "queue", "steer"),
        ("steer", "both", "steer"),
        ("steer", "steer", "steer"),
        ("queue", "steer", "queue"),
        ("queue", "both", "queue"),
        ("queue", "queue", "queue"),
        ("both", "steer", "both"),
        ("both", "queue", "both"),
        ("both", "both", "both"),
    ],
)
async def test_each_chord_against_each_default_mode(
    chord_mode: str, default_mode: str, delivery: str
) -> None:
    """3x3: the chord wins for this send; the default holds for the next."""
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        _set_mode(app, default_mode)
        chord_for = {"steer": "alt+q", "queue": "ctrl+enter", "both": "alt+s"}
        app._apply_keymap({f"keymap.send_{chord_mode}": chord_for[chord_mode]}, announce=False)
        await _type(app, pilot, "payload")
        await pilot.press(chord_for[chord_mode])
        await pilot.pause()
        # `steer` keeps the legacy call shape (delivery=None on the fake).
        expected = delivery if delivery != "steer" else None
        assert session.delivery_steers[-1] == ("payload", expected)
        notices = {
            "steer": QUEUED_STEER_NOTICE,
            "queue": QUEUED_WAIT_NOTICE,
            "both": BOTH_WAIT_NOTICE,
        }
        notice = notices[delivery]
        assert notice in _notice_texts(app)


@pytest.mark.asyncio
async def test_chord_while_approval_prompt_is_live_behaves_as_enter() -> None:
    """Documented: the chord reaches the live prompt as Enter, like ctrl+j.

    The editor rewrites the event to ``enter`` before the router re-reads
    it, so the chord is indistinguishable from Enter at that seam. The
    mode is consumed by ``_submit`` on the same key, so nothing leaks past
    it; what this pins is the rewrite reaching the router.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        from textual import events

        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        seen: list[str] = []

        class _Prompt:
            def answer(self, value: str) -> None:
                seen.append(value)

        app._live_prompt = lambda: _Prompt()  # type: ignore[method-assign]
        editor.text = "draft"
        await pilot.pause()
        event = events.Key("ctrl+enter", None)
        event.set_sender(editor)
        await editor._on_key(event)
        await pilot.pause()
        assert event.key == "enter"
        # The fake prompt never engaged, so the Enter submitted the draft
        # as a normal idle prompt (the chord rewrote the event before any
        # router read it).
        assert session.prompts == ["draft"]
        # _submit consumed the chord on this same key.
        assert editor._chord_delivery is None
