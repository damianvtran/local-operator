"""The TUI / backend English the census found, keyed — and proved LIVE.

A rendered census (desktop 0.52.3 against a scratch backend, one locale at a
time) still painted English after a language switch in these places, all in
core: the `/language` receipts, the composer's resting placeholder, the
`/notifications` kind words, the retired-row status suffix, and the
backend-composed strings the desktop prints verbatim (hotkey help, voice
provider availability, import-source detail, the supervisor detail).

Two properties are pinned, mirroring ``test_compose_live_locale.py``:

* **English is byte-identical.** Every key renders, in English, exactly the
  literal that stood at its call site on ``origin/main`` (the table below IS
  those literals). Nothing about the English product moves.
* **A live switch reaches every consumer.** Every consumer module is imported
  FIRST (in English) and the locale moves inside the test, which is the real
  ordering of ``/language fr``; a test that switched before importing would
  pass against a by-value-at-import binding — the bug class that bit #3502.

``fr`` here is a FIXTURE: a temp catalogue root carrying a MARKER prefix on
each new message. No target-locale text is authored in this repository by this
change — translators supply it later — so the proof that a switch does
something is "the rendered text now carries the marker the fr file holds".
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

# Every consumer is imported (in English) BEFORE any test moves the locale.
from local_operator import keymap as keymap_mod
from local_operator import settings_io
from local_operator.audiogen import cascade as audiogen_cascade  # noqa: F401
from local_operator.harness_import.types import SourceInfo
from local_operator.i18n import catalogues
from local_operator.i18n import resolve as resolve_mod
from local_operator.i18n import text
from local_operator.i18n.keys import (
    tui_composer,
    tui_notification_kinds,
    wire_harness_import_detail,
    wire_language_receipts,
    wire_session_status,
    wire_voice_availability,
    wire_wakes_supervisor,
)
from local_operator.i18n.messages import Msg, render_resolved
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.session import catalog as catalog_mod
from local_operator.stt import cascade as stt_cascade  # noqa: F401
from local_operator.tts import cascade as tts_cascade  # noqa: F401
from local_operator.tui import app as app_mod
from local_operator.tui import notify as notify_mod
from local_operator.tui.app import OperatorApp, notification_kind_word
from local_operator.tui.widgets import editor as editor_mod
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock
from local_operator.video import VideoRoute
from local_operator.video import cascade as video_cascade
from local_operator.wakes import install as wakes_install  # noqa: F401
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: Every namespace this change adds. Seven locales carry none of them yet.
NEW_NAMESPACES = (
    "tui.chrome.remote_park",
    "tui.composer",
    "tui.notification_kinds",
    "wire.harness_import_detail",
    "wire.language_receipts",
    "wire.session_status",
    "wire.settings.keymap",
    "wire.voice_availability",
    "wire.wakes_supervisor",
)

MARKER = "⟦fr⟧ "


def _tr_en(namespace: str, message: Msg) -> str:
    text.set_active_locale("en")
    return text.tr(namespace)(message)


# -- English parity: the literal each call site held on origin/main ---------------

#: ``(namespace, message, the origin/main literal)``. The f-string receipts use
#: ``tag="en"`` / ``name="English"``: an English UI names only English in the
#: parenthesis (the name is the ACTIVE language's own), so those bytes are
#: unchanged; a non-English interface is exactly what is allowed to differ.
ENGLISH_PARITY: list[tuple[str, Msg, str]] = [
    ("tui.composer", tui_composer.placeholder_default(), "Message Local Operator…"),
    ("tui.notification_kinds", tui_notification_kinds.word_closed(), "completed"),
    (
        "tui.notification_kinds",
        tui_notification_kinds.word_retired(),
        "retired for an update",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.show_terminal(tag="en", name="English"),
        "language: en (English) — /language <value>; space browses",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.show_options(tag="en", name="English", options="auto, en, fr"),
        "language: en (English) — auto, en, fr",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.unknown_terminal(echo="'xx'"),
        "unknown language 'xx' — press space to list values",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.unknown_options(echo="'xx'", options="auto, en"),
        "unknown language 'xx' — available: auto, en",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.already(key="en", tag="en", name="English"),
        "language: already en — resolves to en (English)",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.unregistered(),
        "language: the setting is not registered in this build",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.save_failed(key="fr", error="disk full"),
        "language: could not save fr — disk full",
    ),
    (
        "wire.language_receipts",
        wire_language_receipts.saved(key="en", tag="en", name="English"),
        "language: en saved — active en (English)",
    ),
    (
        "wire.session_status",
        wire_session_status.retired_label(base="Interrupted"),
        "Interrupted — retired for an update",
    ),
    (
        "wire.session_status",
        wire_session_status.retired_label(base="Unseen interruption"),
        "Unseen interruption — retired for an update",
    ),
    (
        "wire.voice_availability",
        wire_voice_availability.radient_signed_in(),
        "Signed in to Radient.",
    ),
    (
        "wire.voice_availability",
        wire_voice_availability.radient_signed_out(),
        "Not signed in to Radient.",
    ),
    (
        "wire.voice_availability",
        wire_voice_availability.elevenlabs_stored(),
        "An ElevenLabs API key is stored.",
    ),
    (
        "wire.voice_availability",
        wire_voice_availability.elevenlabs_missing(),
        "No ElevenLabs API key is stored.",
    ),
    (
        "wire.voice_availability",
        wire_voice_availability.openai_stored(),
        "An OpenAI API key is stored.",
    ),
    (
        "wire.voice_availability",
        wire_voice_availability.openai_missing(),
        "No OpenAI API key is stored.",
    ),
    (
        "wire.voice_availability",
        wire_voice_availability.no_tts_provider(),
        "No text-to-speech provider is available: sign in to Radient, or store an "
        "ElevenLabs or OpenAI API key.",
    ),
    ("wire.voice_availability", wire_voice_availability.fal_stored(), "A FAL key is stored."),
    ("wire.voice_availability", wire_voice_availability.fal_missing(), "No FAL key is stored."),
    ("wire.voice_availability", wire_voice_availability.route_available(), "Available."),
    ("wire.voice_availability", wire_voice_availability.route_unavailable(), "Not available."),
    (
        "wire.harness_import_detail",
        wire_harness_import_detail.omp_missing(),
        "no omp agent directory found",
    ),
    (
        "wire.harness_import_detail",
        wire_harness_import_detail.claude_missing(path="/h/.claude"),
        "no Claude store at /h/.claude",
    ),
    (
        "wire.harness_import_detail",
        wire_harness_import_detail.codex_missing(path="/h/.codex"),
        "no Codex store at /h/.codex",
    ),
    (
        "wire.wakes_supervisor",
        wire_wakes_supervisor.shared_unit_mismatch(),
        "this store is not the one the shared unit serves; launchd cannot supervise it",
    ),
]

#: The eight hotkey rows' copy: ``(action id, label, help)`` exactly as
#: ``keymap.KEY_ACTIONS`` (and so origin/main's settings rows for the first
#: five) carried them. The three send rows are the composer-scope batch:
#: their English is the `wire.settings.keymap` catalogue, which is also
#: what the settings rows paint, so registry and settings agree by
#: construction.
KEYMAP_ENGLISH = {
    "keymap.new_session": ("New session", "Starts a fresh conversation, the same as /new."),
    "keymap.resume": (
        "Resume a session",
        "Opens the picker of recent conversations, the same as /resume.",
    ),
    "keymap.quick_send": (
        "Quick send",
        "Opens a small composer over other apps; messages the chief of staff.",
    ),
    "keymap.push_to_talk": ("Push to talk", "Hold to dictate (desktop); release to stop."),
    "keymap.start_recording": (
        "Start recording",
        "Starts speech-to-text recording in the desktop window.",
    ),
    "keymap.send_steer": (
        "Send as steer",
        "Sends the message now and interrupts the running step. Ctrl/Alt/Cmd+Enter only "
        "work in terminals that support enhanced keys; Alt+letter works everywhere "
        "(macOS terminals need Option-as-Meta).",
    ),
    "keymap.send_queue": (
        "Send as queue",
        "Waits for the current step to finish and never interrupts running tools. "
        "Ctrl/Alt/Cmd+Enter only work in terminals that support enhanced keys; "
        "Alt+letter works everywhere (macOS terminals need Option-as-Meta).",
    ),
    "keymap.send_both": (
        "Send as both",
        "Waits like queue, then interrupts like steer if the message has waited too long. "
        "Ctrl/Alt/Cmd+Enter only work in terminals that support enhanced keys; "
        "Alt+letter works everywhere (macOS terminals need Option-as-Meta).",
    ),
}


@pytest.mark.parametrize(
    ("namespace", "message", "literal"),
    ENGLISH_PARITY,
    ids=[message.code for _ns, message, _lit in ENGLISH_PARITY],
)
def test_english_renders_the_literal_origin_main_held(
    namespace: str, message: Msg, literal: str
) -> None:
    assert _tr_en(namespace, message) == literal


def test_the_composer_placeholder_and_kind_words_read_english_through_their_consumers() -> None:
    text.set_active_locale("en")
    assert editor_mod.default_placeholder() == "Message Local Operator…"
    assert editor_mod.DEFAULT_PLACEHOLDER == "Message Local Operator…"
    assert notification_kind_word("closed") == "completed"
    assert notification_kind_word("retired") == "retired for an update"
    # An unknown kind keeps its raw store token: a receipt may be terse, never wrong.
    assert notification_kind_word("brand_new_kind") == "brand_new_kind"


def test_every_hotkey_row_keeps_its_registry_english_and_none_is_unkeyed() -> None:
    text.set_active_locale("en")
    assert {action.id for action in keymap_mod.KEY_ACTIONS} == set(KEYMAP_ENGLISH)
    for action in keymap_mod.KEY_ACTIONS:
        row = settings_io.BY_KEY[action.id]
        label, help_text = KEYMAP_ENGLISH[action.id]
        assert (row.label, row.help) == (label, help_text), action.id
        # The registry itself is unchanged: group-conflict refusals still use it.
        assert (action.label, action.help) == (label, help_text), action.id


def test_the_video_route_reasons_keep_their_english() -> None:
    text.set_active_locale("en")
    cases = {
        (VideoRoute.RADIENT, True): "Signed in to Radient.",
        (VideoRoute.RADIENT, False): "Not signed in to Radient.",
        (VideoRoute.FAL, True): "A FAL key is stored.",
        (VideoRoute.FAL, False): "No FAL key is stored.",
    }
    for (route, available), expected in cases.items():
        assert video_cascade._route_reason(route, available) == expected


def test_the_retired_row_label_keeps_its_english() -> None:
    entry = SimpleNamespace(completion_kind="retired", completion_reason="")
    stop_label = catalog_mod.CatalogEntry._stop_label
    expected = "Interrupted — retired for an update"
    assert stop_label(entry, "Interrupted") == expected  # type: ignore[arg-type]
    assert stop_label(entry, "Unseen interruption") == (  # type: ignore[arg-type]
        "Unseen interruption — retired for an update"
    )


def test_an_unkeyed_import_detail_stays_the_english_diagnostic() -> None:
    plain = SourceInfo(source="omp", label="omp", root=None, detected=False, detail="thread index")
    assert plain.display_detail() == "thread index"
    keyed = SourceInfo(
        source="claude",
        label="Claude",
        root=None,
        detected=False,
        detail="no Claude store at /h/.claude",
        detail_key="claude_missing",
        detail_path="/h/.claude",
    )
    assert keyed.display_detail() == "no Claude store at /h/.claude"
    # The model-facing / default wire shape never changes with the language.
    assert keyed.to_dict()["detail"] == "no Claude store at /h/.claude"
    # No message code crosses the wire: the namespace is rendered in-process only.
    assert "detail_code" not in keyed.to_dict()


# -- the live switch -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _english_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``render_resolved`` asks the config chain (env, then the OS): pin it to English.

    ``marker_fr`` overrides this with ``fr`` for the live-switch tests.
    """
    monkeypatch.setenv("LOP_LANG", "en")


@pytest.fixture(autouse=True)
def _restore_locale() -> Iterator[None]:
    """Put the process back on English with a LIVE catalogue root.

    The move back re-runs the real refreshers, so it must read the real tree:
    the test's temp root can already be gone by teardown (see the sibling
    ``test_language_live.py``), and a refresh against a deleted tree would
    degrade every re-materialised constant to message codes.
    """
    real_root = catalogues.catalogue_root()
    yield
    patched = catalogues._CATALOGUES
    catalogues._CATALOGUES = real_root
    try:
        text.set_active_locale("en")
    finally:
        catalogues._CATALOGUES = patched


@pytest.fixture
def marker_fr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """A catalogue root with a MARKER ``fr`` for each new namespace.

    ``LOP_LANG=fr`` makes ``render_resolved`` (the server-side path, which asks
    the config chain rather than the TUI's active locale) resolve ``fr`` too.
    Returns the marker so callers assert on the bytes the fr file really holds.
    """
    real = catalogues.catalogue_root()
    root = tmp_path / "catalogues"
    shutil.copytree(real / "en", root / "en")
    (root / "fr").mkdir()
    for namespace in NEW_NAMESPACES:
        english = json.loads((real / "en" / f"{namespace}.json").read_text(encoding="utf-8"))
        (root / "fr" / f"{namespace}.json").write_text(
            json.dumps({code: MARKER + source for code, source in english.items()}),
            encoding="utf-8",
        )
    monkeypatch.setattr(catalogues, "_CATALOGUES", root)
    monkeypatch.setenv("LOP_LANG", "fr")
    return MARKER


def test_a_live_switch_repaints_the_composer_and_kind_words_and_hotkey_help(
    marker_fr: str,
) -> None:
    text.set_active_locale("en")
    english_placeholder = editor_mod.default_placeholder()
    english_help = settings_io.BY_KEY["keymap.push_to_talk"].help
    assert marker_fr not in english_placeholder

    text.set_active_locale("fr")  # AFTER every consumer was imported in English

    assert editor_mod.default_placeholder() == marker_fr + "Message Local Operator…"
    # The module constant follows through its registered refresher.
    assert editor_mod.DEFAULT_PLACEHOLDER == marker_fr + "Message Local Operator…"
    assert notification_kind_word("retired") == marker_fr + "retired for an update"
    assert notification_kind_word("closed") == marker_fr + "completed"
    # The frozen settings registry re-materialises hotkey help in place.
    assert settings_io.BY_KEY["keymap.push_to_talk"].help == marker_fr + english_help
    assert settings_io.BY_KEY["keymap.start_recording"].label.startswith(marker_fr)

    text.set_active_locale("en")
    assert editor_mod.default_placeholder() == english_placeholder
    assert settings_io.BY_KEY["keymap.push_to_talk"].help == english_help


def test_a_fresh_composer_widget_is_built_in_the_active_language(marker_fr: str) -> None:
    """The constructor default was ``= DEFAULT_PLACEHOLDER`` (bound at class creation)."""
    text.set_active_locale("fr")
    assert Editor().placeholder == marker_fr + "Message Local Operator…"
    assert Editor().resting_placeholder == marker_fr + "Message Local Operator…"


def test_backend_composed_display_text_follows_the_configured_language(
    marker_fr: str,
) -> None:
    """The server path: no TUI, ``render_resolved`` asks the config chain."""
    assert resolve_mod.resolve_language() == "fr"
    assert video_cascade._route_reason(VideoRoute.FAL, False) == marker_fr + "No FAL key is stored."
    assert video_cascade._route_reason(VideoRoute.RADIENT, True) == (
        marker_fr + "Signed in to Radient."
    )
    keyed = SourceInfo(
        source="codex",
        label="Codex",
        root=None,
        detected=False,
        detail="no Codex store at /h/.codex",
        detail_key="codex_missing",
        detail_path="/h/.codex",
    )
    assert keyed.to_dict(localized=True)["detail"] == marker_fr + "no Codex store at /h/.codex"
    assert keyed.to_dict()["detail"] == "no Codex store at /h/.codex"  # the model's copy
    assert render_resolved(wire_wakes_supervisor.shared_unit_mismatch()) == (
        marker_fr + "this store is not the one the shared unit serves; launchd cannot supervise it"
    )
    entry = SimpleNamespace(completion_kind="retired", completion_reason="")
    assert catalog_mod.CatalogEntry._stop_label(entry, "Interrupted") == (  # type: ignore[arg-type]
        marker_fr + "Interrupted — retired for an update"
    )


@pytest.mark.asyncio
async def test_the_tts_cascade_reasons_follow_the_language_through_the_real_resolver(
    marker_fr: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Through ``resolve_voice_path`` itself, against a real (empty) AuthStore."""
    from local_operator.providers.auth_store import AuthStore

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "empty-config"))
    for var in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "RADIENT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    store = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path / "config")
    try:
        resolution = await tts_cascade.resolve_voice_path(
            config_dir=tmp_path / "config",
            base_url="https://api.radienthq.com/v1",
            store=store,
        )
    finally:
        store.close()
    assert [rung.reason for rung in resolution.rungs] == [
        marker_fr + "Not signed in to Radient.",
        marker_fr + "No ElevenLabs API key is stored.",
        marker_fr + "No OpenAI API key is stored.",
    ]
    assert resolution.reason == marker_fr + (
        "No text-to-speech provider is available: sign in to Radient, or store an "
        "ElevenLabs or OpenAI API key."
    )


@pytest.fixture
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "cfg"
    path.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(path))
    return path


def _notices(app: OperatorApp) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


@pytest.mark.asyncio
async def test_a_running_switch_repaints_the_composer_placeholder_and_the_receipt(
    marker_fr: str, config_dir: Path
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        for _ in range(60):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one(Editor)
        # Booted in English (LOP_LANG=fr resolves fr, but nothing has moved the
        # process yet): the painted placeholder is the English resting copy.
        text.set_active_locale("en")
        editor.placeholder = editor.resting_placeholder
        assert editor.placeholder == "Message Local Operator…"

        app._run_slash_command("/language fr")
        await pilot.pause()
        await pilot.pause()

        # (1) the composer's PAINTED placeholder moved with the language.
        assert editor.placeholder == marker_fr + "Message Local Operator…"
        # (2) the receipt is composed after the switch, in the new language, and
        # names the language by its own (catalogue-independent) native name.
        assert marker_fr + "language: fr saved — active fr (Français)" in _notices(app)

        # A composer showing ANOTHER mode's placeholder is not clobbered.
        editor.placeholder = "Read-only · esc back"
        app._apply_language("en")
        assert editor.placeholder == "Read-only · esc back"


def test_the_language_name_is_data_not_an_english_word() -> None:
    names = {tag: resolve_mod.language_native_name(tag) for tag in resolve_mod.SUPPORTED_LOCALES}
    assert names["en"] == "English"
    assert names["fr"] == "Français"
    # Pinned to the manifest, so a locale added there cannot be missing here.
    manifest = json.loads(
        (Path(__file__).resolve().parents[3] / "i18n" / "languages.json").read_text(
            encoding="utf-8"
        )
    )
    assert {row["code"]: row["native"] for row in manifest["languages"]} == names
    assert (
        resolve_mod.language_native_name("xx") == "English"
    )  # unknown degrades like language_name


def test_app_module_has_no_import_time_copy_of_the_kind_words() -> None:
    """The dict holds message FACTORIES; a dict of rendered words would freeze English."""
    for factory in app_mod.NOTIFICATION_KIND_WORDS.values():
        assert callable(factory)


# -- the "parked on another device" banners ----------------------------------------

#: What ``tui.notify``'s ``REMOTE_PARK_*`` constants held on origin/main, formatted
#: with the device name the callers pass.
_DEVICE = "Mac mini"
_APPROVAL_REMAINDER = (
    "Allowing it happens on Mac mini (Touch ID) or on a phone paired with it. "
    "If nothing there can check a signature: ask Local Operator to set up operator "
    "authority there (one approval and one admin password prompt) — if that device is "
    "running an older build, updating it is what fixes that."
)
_APPROVAL_REMEDY = (
    "Allowing it happens on Mac mini (Touch ID) or on a phone paired with it. If Mac mini "
    "cannot check a signature yet, this machine's operator anchor is not installed there: "
    "run `lop operator anchor export --file anchor.json` on this machine, copy that file "
    "to Mac mini, then run `lop operator install --from anchor.json` on Mac mini once "
    "(one admin password prompt). That lets Mac mini check a signature made on this "
    "machine; it grants nothing else."
)


def test_the_remote_park_helpers_keep_their_origin_main_english() -> None:
    text.set_active_locale("en")
    assert notify_mod.remote_park_item_title(_DEVICE, "approval") == (
        "Waiting for approval on Mac mini"
    )
    assert notify_mod.remote_park_item_title(_DEVICE, "ask") == (
        "Waiting for your answer on Mac mini"
    )
    assert notify_mod.remote_park_item_body(_DEVICE, "approval") == (
        "Open the session here to deny it.\n\n" + _APPROVAL_REMEDY
    )
    assert notify_mod.remote_park_item_body(_DEVICE, "ask") == "Open the session here to answer it."
    assert notify_mod.remote_park_card(_DEVICE, "approval") == (
        "Waiting for approval on Mac mini\nOpen the session here to deny it. " + _APPROVAL_REMAINDER
    )
    assert notify_mod.remote_park_card(_DEVICE, "approval", name="Nightly", attached=True) == (
        "Nightly\nWaiting for approval on Mac mini\n"
        "Its gate card has not reached this view — deny it here once it does. "
        + _APPROVAL_REMAINDER
    )
    assert notify_mod.remote_park_card(_DEVICE, "ask") == (
        "Waiting for your answer on Mac mini\nOpen the session here to answer it."
    )
    assert notify_mod.remote_park_banner(_DEVICE, "approval") == (
        "Waiting for approval on Mac mini — allow there or on its paired phone; "
        "deny after opening."
    )
    assert notify_mod.remote_park_banner(_DEVICE, "ask") == (
        "Waiting for your answer on Mac mini — open it to answer."
    )
    assert notify_mod.remote_park_hints(_DEVICE, "approval") == (
        "runs on Mac mini — allow there (Touch ID) or its paired phone; deny works here",
        "runs on Mac mini — allow there (Touch ID); deny works here",
        "runs on Mac mini — allow there; deny works here",
        "runs on Mac mini",
    )
    assert notify_mod.remote_park_hints(_DEVICE, "ask") == (
        "runs on Mac mini — answer it from here",
        "runs on Mac mini",
    )


def test_a_live_switch_reaches_the_park_banners(marker_fr: str) -> None:
    text.set_active_locale("en")
    english_banner = notify_mod.remote_park_banner(_DEVICE, "ask")
    text.set_active_locale("fr")  # AFTER notify was imported in English
    assert notify_mod.remote_park_banner(_DEVICE, "ask") == marker_fr + english_banner
    assert notify_mod.remote_park_item_title(_DEVICE, "approval") == (
        marker_fr + "Waiting for approval on Mac mini"
    )
    assert notify_mod.remote_park_hints(_DEVICE, "ask")[-1] == marker_fr + "runs on Mac mini"
    card = notify_mod.remote_park_card(_DEVICE, "approval")
    assert card.startswith(marker_fr + "Waiting for approval on Mac mini\n" + marker_fr)
