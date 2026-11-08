"""``resolve.config_value`` must not trust a watcher nobody is polling.

WHY THIS FILE EXISTS (the by-value / unstarted-watcher bug class). The server
creates config watchers it never starts: ``process_watcher()`` at the settings
route, the team bridge and the COS-name read each ``poll_now()`` once and stop.
``existing_watcher()`` returns ANY watcher ever created, started or not, and
``config_value`` used to prefer its ``.values`` unconditionally. An unstarted
watcher's snapshot only moves on ``poll_now()`` or the in-process
``notify_local()`` (the PATCH fast path), so a language change from the CLI, the
TUI or another client was never seen: ``/v1/capabilities``, the feed ``config``
event and the sessions-list wake label painted the old language until restart
(and ``catalog._label_locale`` memoised the stale answer under the NEW file
stamp, so it could not self-heal).

The contract pinned here: trust the watcher only while it is polling; otherwise
the file is the truth.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.i18n import catalogues, resolve
from local_operator.session import catalog

KEY = "wire.session.status.scheduled"
TEXT = {
    "en": "Scheduled ({count, plural, one {# wake} other {# wakes}})",
    "fr": "Planifié ({count, plural, one {# réveil} other {# réveils}})",
    "es": "Programado ({count, plural, one {# activación} other {# activaciones}})",
}
LABEL = {"en": "Scheduled (1 wake)", "fr": "Planifié (1 réveil)", "es": "Programado (1 activación)"}


@pytest.fixture(autouse=True)
def isolated(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """Own config dir and a fixture catalogue; no ambient language source."""
    config = tmp_path / "cfg"
    config.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.delenv("LOP_LANG", raising=False)
    monkeypatch.delenv("LOP_I18N", raising=False)
    root = tmp_path / "cat"
    for tag, text in TEXT.items():
        (root / tag).mkdir(parents=True)
        (root / tag / "wire.session.json").write_text(json.dumps({KEY: text}), encoding="utf-8")
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setattr(catalog, "_label_locale_memo", None)
    _reset_for_tests()
    yield config
    _reset_for_tests()


def _write_language(config: pathlib.Path, tag: str) -> None:
    """A write from ANOTHER process: the file moves, no watcher is told."""
    (config / "config.yml").write_text(f"values:\n  language: {tag}\n", encoding="utf-8")


def test_an_unstarted_watcher_does_not_pin_the_boot_language(isolated) -> None:
    _write_language(isolated, "fr")
    watcher = process_watcher(isolated)
    watcher.poll_now()  # what the settings route / team bridge do — and nothing more
    assert not watcher.is_polling
    assert resolve.config_value() == "fr"

    for tag in ("es", "fr", "en"):
        _write_language(isolated, tag)
        assert resolve.config_value() == tag
        assert catalog.scheduled_label(1) == LABEL[tag]


@pytest.mark.asyncio
async def test_a_started_watcher_is_still_honoured_while_it_polls(isolated) -> None:
    _write_language(isolated, "fr")
    watcher = process_watcher(isolated)
    watcher.start()
    try:
        assert watcher.is_polling
        assert resolve.config_value() == "fr"
        # The file moves but the poll tick has not run: the in-memory snapshot
        # wins, which is the whole point of preferring a live watcher (a dict
        # read instead of a file read on a hot path).
        _write_language(isolated, "es")
        assert resolve.config_value() == "fr"
        watcher.poll_now()
        assert resolve.config_value() == "es"
    finally:
        await watcher.stop()
    # Stopped -> no longer authoritative; the file is.
    assert not watcher.is_polling
    _write_language(isolated, "en")
    assert resolve.config_value() == "en"


def test_the_in_process_patch_path_still_moves_the_label(isolated) -> None:
    _write_language(isolated, "fr")
    watcher = process_watcher(isolated)
    watcher.poll_now()
    assert catalog.scheduled_label(1) == LABEL["fr"]

    manager = ConfigManager(isolated)
    setting = settings_io.resolve_key("language")
    assert setting is not None
    settings_io.write_setting(manager, setting, "es")  # fires watcher.notify_local()
    assert catalog.scheduled_label(1) == LABEL["es"]


def test_reading_the_language_never_creates_the_config_dir(tmp_path, monkeypatch) -> None:
    """The read stays side-effect-free: no watcher and no file means no directory."""
    absent = tmp_path / "never-created"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(absent))
    assert resolve.config_value() is None
    assert not absent.exists()
