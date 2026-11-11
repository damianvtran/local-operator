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
    """A chord is Enter, so an open slash picker accepts the row.

    The assertion is the ACCEPT, not the absence of a prompt: ``/he``
    highlights ``help`` unambiguously, so the chord must complete that row
    and submit the completed word — the buffer empties, the picker closes,
    and the app dispatches the slash command (no chat prompt). A picker
    that ignored the chord would leave ``/he`` in the buffer with the list
    still open and this would fail.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        editor = app.query_one(Editor)
        await _type(app, pilot, "/he")
        await pilot.pause()
        submitted_dispatch: list[str] = []
        submits: list[str] = []
        original_submit = editor._submit

        def spy_submit() -> None:
            submits.append(editor.text)
            original_submit()

        editor._submit = spy_submit  # type: ignore[method-assign]
        original = app.on_editor_submitted

        async def capture(message: EditorSubmitted) -> None:
            submitted_dispatch.append(message.text)
            await original(message)

        app.on_editor_submitted = capture  # type: ignore[method-assign]
        await pilot.press("ctrl+enter")
        await pilot.pause()
        await pilot.pause()
        # Unambiguous row accepted and its completed word submitted. (A
        # ``app.on_editor_submitted`` instance override does not intercept
        # Textual's ``post_message`` dispatch — measured here — so the
        # dispatch leg is proved by the ``_submit`` spy on the editor,
        # which the picker path calls, not by the capture wrapper.)
        assert submits == ["/help "]
        assert editor.text == ""
        assert not editor._picker.is_open()
        # The completed word is a slash command: dispatched, never prompted.
        assert session.prompts == []


@pytest.mark.asyncio
async def test_picker_held_chord_mode_never_leaks_to_the_next_send() -> None:
    """The picker-consume leak: a chord the picker holds must not steer the next send.

    An AMBIGUOUS prefix (``/lo`` -> ``loop``/``logout`` …) takes Enter as
    "extend to the common prefix" WITHOUT submitting, so ``_submit`` never
    runs to consume the chord mode and it stays set — the one path where the
    reset at the top of ``_on_key`` is the only thing standing between a
    stale ``queue`` and the next plain-Enter send. Delete that reset and
    the next send carries ``queue`` instead of the default ``steer`` here.
    Plain text typed between the chord and the send goes through ``_on_key``
    (a real ``pilot.press``), so the reset is what clears it — setting
    ``editor.text`` directly would bypass the very line under test.
    """
    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        _set_mode(app, "steer")
        app._apply_keymap({"keymap.send_queue": "ctrl+enter"}, announce=False)
        editor = app.query_one(Editor)
        await _type(app, pilot, "/lo")
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause()
        # Ambiguous Enter: no submit, the mode is still parked on the editor.
        assert editor._chord_delivery == "queue"
        assert session.delivery_steers == []
        editor.text = "second"
        await pilot.pause()
        await pilot.press("x")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        # Default (steer — the legacy call shape, delivery=None), not queue.
        assert session.delivery_steers == [("xsecond", None)]


@pytest.mark.asyncio
async def test_chord_in_credential_capture_confirms_only() -> None:
    """A chord in a masked capture submits nothing; it confirms the capture.

    The masked capture is armed directly (``_credential_typing``): the
    picker route to it needs a mounted runtime, which the fake does not
    supply, so staging it here tests the Enter-owner branch, not the picker.

    Real effect, not just "no prompt": a typed canary must mint its
    ``[Credential #1, 6 chars]`` marker and land in the editor's attachments
    as a ``PastedCredential`` — the confirm actually captured something.
    (``_mint_typed_credential`` is editor-local and needs no runtime.)
    The chord mode is NOT consumed by a mint (``_submit`` never runs), so
    the test deliberately does not assert it is ``None`` here — that is
    the next key's reset, pinned by the picker leak test.
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
        editor.text = "/credential " + "\u2022" * 6
        editor._credential_typing = 12  # type: ignore[attr-defined]
        editor._credential_typed = "canary"  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.press("ctrl+enter")
        await pilot.pause()
        await pilot.pause()
        # Capture confirmed: marker minted, capture closed, nothing sent.
        assert editor._credential_typing is None
        assert editor.text == "/credential [Credential #1, 6 chars] "
        assert [getattr(a, "value") for a in editor._attachments.values()] == ["canary"]
        assert session.prompts == []


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
    """A chord in shell mode runs the command; it never takes a delivery mode.

    Real effect: ``echo hi`` must EXECUTE — a ``! echo hi`` user row and a
    settled ``ToolResult`` — and produce no chat prompt. (The old assertion
    checked ``"echo hi" not in user_texts`` where the row is ``"! echo hi"``:
    an exact-membership check that passed whether or not anything ran.)
    """
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
        # The command runs through the real execute_bash path: wait for it.
        for _ in range(100):
            await pilot.pause()
            if app._interaction.shell.result is not None:
                break
        assert _user_texts(app) == ["! echo hi"]
        assert app._interaction.shell.result is not None
        assert session.prompts == []
        assert editor._chord_delivery is None


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
    """The REAL router is under test, not a stub: a held answer key + chord answers it.

    The editor rewrites the chord to ``enter`` BEFORE ``route_key_to_live_prompt``
    runs, so a chord arriving while an answer key is held must take the
    router's held-Enter branch and ANSWER the question — not restore ``y``
    into the buffer and submit it as a chat prompt (the exact leak the
    ``ctrl+j`` event rewrite closed, before this chord existed). Without the
    event rewrite this test fails: the router would see ``ctrl+enter``,
    restore ``y`` and the editor would submit it.

    Why the prompt object is a stub while the router is real: a mounted
    approval card needs a live session gate (an approval only exists mid-tool
    call on a real runtime), and the fake here supplies none. The router is
    reached through the real ``editor._on_key`` ->
    ``app.route_key_to_live_prompt`` path either way; the stub only has to
    offer the advertised-key surface the router reads (``answer_keys``,
    ``question_index``, ``answer_from_key``).
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
        answered: list[str] = []

        class _Prompt:
            question_index = 0
            passive = False

            def answer_keys(self):  # type: ignore[no-untyped-def]
                return ("y",)

            def answer_from_key(self, character: str) -> None:  # type: ignore[no-untyped-def]
                answered.append(character)

        prompt = _Prompt()
        app._live_prompt = lambda: prompt  # type: ignore[method-assign]
        # Park ``y`` via the real router (empty buffer + advertised key).
        y_event = events.Key("y", "y")
        y_event.set_sender(editor)
        assert app.route_key_to_live_prompt(y_event) is True
        assert app._held_answer_key is not None
        # The chord: rewritten to Enter, then answered by the held branch.
        await pilot.press("ctrl+enter")
        await pilot.pause()
        await pilot.pause()
        assert answered == ["y"]
        # Answered, not submitted: no prompt, no restored character.
        assert session.prompts == []
        assert editor.text == ""
        assert app._held_answer_key is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored", "sequence", "event_key", "mode", "expected_delivery"),
    [
        # Every spelling variant of the same chord must fire the same mode.
        # Storing ``ctrl+alt+b`` (the way a human types it) while the parser
        # emits ``alt+ctrl+b`` (alphabetical modifier order) is the round-1
        # F1/Q1 defect: the chord never fired. Each row drives the REAL
        # ``_apply_keymap`` (which builds the editor's chord map from the
        # stored spelling), the REAL ``XTermParser`` (which decodes the
        # raw bytes to the event key) and the REAL ``Editor._on_key`` —
        # so replacing any of the three canonicalisation sites with plain
        # ``normalize_key`` fails the corresponding row.
        ("ctrl+alt+b", "\x1b[98;7u", "alt+ctrl+b", "queue", "queue"),
        ("alt+ctrl+b", "\x1b[98;7u", "alt+ctrl+b", "queue", "queue"),
        ("shift+ctrl+enter", "\x1b[13;6u", "ctrl+shift+enter", "both", "both"),
        ("ctrl+shift+enter", "\x1b[13;6u", "ctrl+shift+enter", "both", "both"),
        ("shift+alt+enter", "\x1b[13;4u", "alt+shift+enter", "steer", None),
        ("alt+shift+enter", "\x1b[13;4u", "alt+shift+enter", "steer", None),
        # A legacy-family chord that works in both protocols.
        ("alt+s", "alt+s", "alt+s", "queue", "queue"),
    ],
)
async def test_stored_spelling_fires_on_raw_bytes(
    stored: str, sequence: str, event_key: str, mode: str, expected_delivery: Any
) -> None:
    """The reviewer's probe, as a real test: stored spelling → raw bytes → steered mode.

    ``\x1b[13;5u`` (``ctrl+alt+enter``) is not in this table on purpose: the
    parser emits ``ctrl+enter`` for it (measured), and the grammar refuses
    that chord, so it can never be stored — a row for it would assert a
    chord map that cannot exist.
    """
    from textual._xterm_parser import XTermParser

    session = _DeliveryStreaming()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _wait_session(pilot, app)
        _set_mode(app, "steer")
        app._apply_keymap({f"keymap.send_{mode}": stored}, announce=False)
        editor = app.query_one(Editor)
        await _type(app, pilot, "payload")
        parser = XTermParser()
        # ``feed`` returns a GENERATOR, and reading it twice (once for the
        # assert, once for the dispatch) consumes it — materialise once.
        messages = list(parser.feed(sequence))
        if sequence.startswith("\x1b["):
            # Raw kitty bytes: the parser's event is the real one.
            assert [m.key for m in messages] == [event_key]
            for message in messages:
                await editor._on_key(message)
        else:
            await pilot.press(event_key)
        await pilot.pause()
        await pilot.pause()
        # The chord fired: its mode rode the steer, not the default.
        assert session.delivery_steers == [("payload", expected_delivery)], (
            stored,
            event_key,
            session.delivery_steers,
        )
        assert session.prompts == []
