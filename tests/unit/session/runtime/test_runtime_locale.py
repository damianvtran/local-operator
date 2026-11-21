"""The session RUNTIME speaks the configured language to its viewers.

Desktop fr census (2026-11), row 10: a French desktop showed "Saved model
information is incomplete; using test/test-model. ..." although
``tui.notices.model_info_incomplete`` has a French string. Two causes, both
here: the runtime child never activated the configured locale (only ``cli.py``
and ``exec_worker`` did), and the notice's ``text`` is pinned to English on
purpose (it is persisted to the transcript), while the viewers paint ``text``
verbatim. The fix renders a catalogued notice's ``text`` in the runtime's
active locale at the relay boundary (``serving._notice_text_for_viewer``),
activates the locale at ``process.main``, and follows later switches
(``serving.attach_language_config_watch``).

The active locale is process-global; ``tests/conftest.py``'s
``restore_active_locale`` restores it after each test.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.harness.types import NoticeEvent
from local_operator.i18n.keys import tui_notices
from local_operator.i18n.text import active_locale, set_active_locale
from local_operator.session.runtime import process, serving
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import _notice

_EN = (
    "Saved model information is incomplete; using test/test-model. "
    "This selection is now saved for future resumes."
)
_FR = (
    "Les informations de modèle enregistrées sont incomplètes ; utilisation de "
    "test/test-model. Ce choix est désormais enregistré pour les reprises à venir."
)


def _relay() -> tuple[list[dict[str, Any]], Any]:
    """A handle's real ``subscribe_events`` relay around a fake session.

    The fake session has no ``Transcript``, so ``subscribe_events`` returns
    right after installing its per-event serializer — the exact code a desktop
    viewer's frames go through — and the test captures that serializer.
    """
    captured: dict[str, Any] = {}

    def subscribe(callback: Any) -> Any:
        captured["handler"] = callback
        return lambda: None

    handle = object.__new__(ServingSessionHandle)
    handle._session = SimpleNamespace(session_id="locale-fixture", subscribe=subscribe)
    frames: list[dict[str, Any]] = []
    handle.subscribe_events(frames.append)
    return frames, captured["handler"]


def _incomplete_notice() -> NoticeEvent:
    # Built by the SAME helper ``Session._ensure_selected_model`` emits through.
    return _notice(tui_notices.model_info_incomplete(model="test/test-model"))


def test_a_runtime_notice_reaches_the_viewer_in_french_and_the_event_stays_english() -> None:
    frames, handler = _relay()
    event = _incomplete_notice()
    set_active_locale("fr")

    handler(event)

    assert frames[-1]["text"] == _FR
    # Identity survives for a locale-aware re-render.
    assert frames[-1]["code"] == "tui.notices.model_info_incomplete"
    assert frames[-1]["params"] == {"model": "test/test-model"}
    # The object the transcript persists is untouched: the journal stays English.
    assert event.text == _EN


def test_english_runtime_is_byte_identical_to_before() -> None:
    frames, handler = _relay()
    set_active_locale("en")

    handler(_incomplete_notice())

    assert frames[-1]["text"] == _EN


def test_uncatalogued_and_unknown_notices_keep_their_text() -> None:
    frames, handler = _relay()
    set_active_locale("fr")

    handler(NoticeEvent(text="free prose nobody catalogued", kind="info"))
    handler(NoticeEvent(text="English fallback", code="tui.notices.no_such_key_x", params={}))

    assert frames[-2]["text"] == "free prose nobody catalogued"
    # An unknown code renders as the code itself; the producer's sentence wins.
    assert frames[-1]["text"] == "English fallback"


def test_process_main_activates_the_resolved_locale() -> None:
    """``process.main`` calls ``activate_resolved_locale`` (it cannot be booted here).

    ``main()`` takes the lease, opens a socket and runs a session, so the call
    is pinned structurally; the real-path boot is in the PR's QA evidence.
    """
    tree = ast.parse(inspect.getsource(process.main))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "activate_resolved_locale" in called


@pytest.fixture
def _fresh_watchers(monkeypatch: pytest.MonkeyPatch):
    _reset_for_tests()
    monkeypatch.setattr(serving, "_language_watch_attached", set())
    yield
    _reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_fresh_watchers")
async def test_a_language_switch_from_another_process_moves_a_running_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("LOP_LANG", raising=False)
    monkeypatch.delenv("LOP_I18N", raising=False)
    set_active_locale("en")
    serving.attach_language_config_watch(config_dir)
    serving.attach_language_config_watch(config_dir)  # idempotent: one listener
    frames, handler = _relay()

    # Shaped like another process's write: a fresh manager, no notify hook.
    setting = settings_io.resolve_key("language")
    assert setting is not None
    settings_io._store(ConfigManager(config_dir), setting.path, "fr")
    change = process_watcher(config_dir).poll_now()
    assert change is not None and "language" in change.changed_keys

    assert active_locale() == "fr"
    handler(_incomplete_notice())
    assert frames[-1]["text"] == _FR

    # An unrelated key does not move the locale.
    set_active_locale("en")
    other = settings_io.resolve_key("display.hide_cross_session")
    assert other is not None
    settings_io._store(ConfigManager(config_dir), other.path, True)
    assert process_watcher(config_dir).poll_now() is not None
    assert active_locale() == "en"
