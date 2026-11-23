"""The read-only composer's keyed copy and existing live-repaint seam.

Only English source is authored here. A synthetic English resolver marker proves
that repaint re-resolves the hint rather than reusing an import-time string;
this is wiring evidence, not target-locale or native-rendering approval.
"""

from __future__ import annotations

import pytest

from local_operator.i18n import text
from local_operator.i18n.keys import tui_editor_readonly
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import editor as editor_mod
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def test_readonly_hint_preserves_english_and_the_keyboard_token() -> None:
    before = text.active_locale()
    try:
        text.set_active_locale("en")
        message = tui_editor_readonly.hint(key="esc")
        assert message.code == "tui.editor_readonly.hint"
        assert editor_mod.read_only_placeholder() == "Read-only · esc back"
        assert len(editor_mod.read_only_placeholder()) == 20
    finally:
        text.set_active_locale(before)


@pytest.mark.asyncio
async def test_readonly_hint_re_resolves_on_live_repaint_without_enabling_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = text.active_locale()
    text.set_active_locale("en")
    app = OperatorApp(lambda: _factory(FakeSession()))
    try:
        async with app.run_test(size=(110, 34)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            app._set_composer_read_only(True)
            assert editor.placeholder == "Read-only · esc back"
            original_tr = editor_mod.tr
            calls = []

            def resolve(namespace):
                if namespace != "tui.editor_readonly":
                    return original_tr(namespace)

                def hint(message):
                    calls.append(message)
                    return "[English resolver refresh] Read-only · esc back"

                return hint

            monkeypatch.setattr(editor_mod, "tr", resolve)
            app._apply_language("en")
            await pilot.pause()
            assert editor.placeholder == "[English resolver refresh] Read-only · esc back"
            assert calls and calls[-1] == tui_editor_readonly.hint(key="esc")
            assert editor.read_only and not editor.can_focus
            draft = editor.text
            await pilot.press("x")
            assert editor.text == draft

            # A different mode still owns its custom hint. Translation must not
            # widen the refresh rule to overwrite secret/credential prompts.
            app._set_composer_read_only(False)
            editor.placeholder = "[credential prompt sentinel]"
            app._apply_language("en")
            assert editor.placeholder == "[credential prompt sentinel]"
    finally:
        text.set_active_locale(before)
