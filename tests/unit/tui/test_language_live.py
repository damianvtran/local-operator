"""The live language path (slice C): a change swaps resolved copy, no restart.

What this file pins, in the order the mechanism moves:

* a `/language` switch in a RUNNING app re-materialises the modules'
  catalogued constants — the splash's, the band's, keymap's tip templates —
  and the surfaces that resolve per call follow on their next render (the
  band rendered through its real entry point paints the new copy);
* a write from another process (the config watcher's ``disk`` delivery)
  applies the same way and is announced (``applied: language``);
* a write from this process (the /settings page's facade, or the command's
  own echo) applies silently, because the writer's receipt is the line the
  user reads.

`fr` here is a FIXTURE locale, not a shipped one: the mechanism ships dark
(only `en` is in the ledger), so the proof that a move does something is a
temp catalogue root with marker strings plus a widened ``shipped_locales`` —
the same root-keying ``test_text.py`` exercises. The autouse fixture moves
the process back to `en` afterwards; the modules' constants end where they
started because the move back re-runs the real refreshers on the real root.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator import keymap as keymap_mod
from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.i18n import catalogues
from local_operator.i18n import resolve as resolve_mod
from local_operator.i18n import text
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui import app as app_mod
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import status_line as status_line_mod
from local_operator.tui.widgets import welcome as welcome_mod
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.settings_view import SettingsView
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _set_editor_line


@pytest.fixture(autouse=True)
def _restore_locale() -> Iterator[None]:
    """The active locale is PROCESS state the app moves; put it back.

    The move back runs the real refreshers, so it must read a LIVE catalogue
    root. The test body's own root (a `tmp_path` tree it patched in) can be
    GONE by teardown time — `tmp_path`'s cleanup and the monkeypatch undo
    finalise in an order this fixture cannot see — and a refresh against a
    deleted tree degrades every re-materialised constant to message codes
    (the failure this pin was written from: welcome's pool read as
    `tui.welcome.tip.resume`). So the move back pins the root to the one
    THIS fixture captured at setup, while it was still the real tree, and
    hands the patched value back afterwards for monkeypatch's own undo.
    """
    real_root = catalogues.catalogue_root()
    yield
    patched = catalogues._CATALOGUES
    catalogues._CATALOGUES = real_root
    try:
        text.set_active_locale("en")
    finally:
        catalogues._CATALOGUES = patched


@pytest.fixture(autouse=True)
def _fresh_registry() -> Iterator[None]:
    _reset_for_tests()
    yield
    _reset_for_tests()


def _fr_catalogues(tmp_path: Path) -> Path:
    """A catalogue root: the real `en` trees plus marker `fr` namespaces."""
    root = tmp_path / "catalogues"
    (root / "en").mkdir(parents=True)
    (root / "fr").mkdir(parents=True)
    real = catalogues.catalogue_root()
    # "tui.app" rides along: the config-change receipt is a catalogue key now
    # (slice 9), so a fixture root without it would render the key CODE instead
    # of the notice text — the extraction moved the literal, and every root that
    # renders a receipt has to carry the namespace it moved into.
    for namespace in ("tui.welcome", "tui.status", "tui.app"):
        shutil.copyfile(real / "en" / f"{namespace}.json", root / "en" / f"{namespace}.json")
    # Marker fr values for the same namespace too: the receipt in THIS test is
    # built before the move, so it renders en — but a receipt resolved under
    # the fixture's fr locale must not fall back to a key code either.
    shipped = json.loads((real / "en" / "tui.app.json").read_text(encoding="utf-8"))
    (root / "fr" / "tui.app.json").write_text(
        json.dumps({key: "FR:" + value for key, value in shipped.items()}),
        encoding="utf-8",
    )
    (root / "fr" / "tui.status.json").write_text(
        json.dumps(
            {
                "tui.status.model.connecting": "FR:connecting…",
                "tui.status.model.setup": "FR:setup",
                "tui.status.tunnel.parked": "FR:parked",
                "tui.status.agents": "{count, plural, one {# FR-agent} other {# FR-agents}}",
            }
        ),
        encoding="utf-8",
    )
    (root / "fr" / "tui.welcome.json").write_text(
        json.dumps(
            {
                "tui.welcome.tip.paste": "FR:paste",
                "tui.welcome.tip.mesh": "FR:mesh",
                "tui.welcome.tip.setup": "FR:{login} one browser sign-in",
                "tui.welcome.tip.resume": "FR:resume",
                "tui.welcome.tip.key_new_session": "FR '{key}' starts",
                "tui.welcome.tip.key_resume": "FR '{key}' resumes",
                "tui.welcome.tip.hotkeys": "FR:hotkeys",
                "tui.welcome.hint.command_picker": "FR:commands",
                "tui.welcome.hint.all_commands": "FR:all commands",
                "tui.welcome.hint.quit": "FR:quit",
                "tui.welcome.hint.sign_in": "FR:sign in",
                "tui.welcome.hint.other_provider": "FR:other provider",
                "tui.welcome.status.login_required.short": "FR:not logged in",
            }
        ),
        encoding="utf-8",
    )
    return root


def _ship_fr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point resolution at the fixture root and pretend `fr` passed audit.

    ``LOP_LANG=en`` keeps the chain deterministic under the fixture (config
    tag first, then env): boot resolves `en`, and the switch below resolves
    `fr` because the CONFIG tag beats the env once it is written.
    """
    monkeypatch.setattr(catalogues, "_CATALOGUES", _fr_catalogues(tmp_path))
    monkeypatch.setattr(resolve_mod, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.setenv("LOP_LANG", "en")


async def _boot(pilot, app: OperatorApp, *, watcher: bool = False) -> None:
    for _ in range(200):
        await pilot.pause()
        if app._session is not None and (not watcher or app._unsubscribe_config_watch is not None):
            return
    raise AssertionError("the app never booted")


def _notices(app: OperatorApp) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


def _write_elsewhere(config_dir: Path, value: str) -> None:
    """A write shaped like another process's: below the notify hook."""
    setting = settings_io.resolve_key("language")
    assert setting is not None
    settings_io._store(ConfigManager(config_dir), setting.path, value)


@pytest.mark.asyncio
async def test_a_running_switch_swaps_the_band_and_the_splash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        _ship_fr(monkeypatch, tmp_path)
        # Before the switch, the same call sites resolve the en copy.
        assert status_line_mod.format_tunnel(True) == "remote access off"

        app._run_slash_command("/language fr")
        await pilot.pause()

        # The constants re-materialised: the splash's...
        assert welcome_mod.MODEL_PENDING == "FR:connecting…"
        assert welcome_mod.MODEL_SETUP == "FR:setup"
        assert welcome_mod.TIP_PASTE == "FR:paste"
        assert welcome_mod.TIP_MESH == "FR:mesh"
        assert welcome_mod.TIPS[0] == "FR:resume"
        assert welcome_mod.HINTS[0][1] == "FR:commands"
        assert welcome_mod.WARNING_SHORT == "FR:not logged in"
        # (ICU quoting is consumed by the runtime: the fixture's `'{key}'`
        # resolves to the literal `{key}` the pool then substitutes at render.)
        assert ("FR {key} starts", "keymap.new_session") in welcome_mod.KEYED_TIPS
        # ...the keyed-tip LOOKUP follows the pool: _resolve_tip must recognise
        # the FRESH template (regression: the template->binding map was
        # import-time only, so a live switch painted a literal `{key}` on the
        # splash until restart).
        resolved_fr = welcome_mod._resolve_tip("FR {key} starts")
        assert "{key}" not in resolved_fr, resolved_fr
        assert resolved_fr.startswith("FR ") and resolved_fr.endswith(" starts")
        # ...the band's, and keymap's tip templates (welcome's pool reads them).
        assert status_line_mod.TUNNEL_PARKED_TEXT == "FR:parked"
        assert keymap_mod.BY_ID["keymap.resume"].tip == "FR {key} resumes"
        # The width budget re-materialises through its builder too (round-1
        # F2: it was the one mirror without one).
        assert welcome_mod.TIP_MIN_WIDTH == welcome_mod._tip_min_width()
        # Per-call resolutions follow without any re-materialiser.
        assert status_line_mod.format_tunnel(True) == "FR:parked"
        assert status_line_mod.format_agents(2) == "2 FR-agents"
        # The band rendered through its own entry point paints the new copy.
        status = app._status
        assert status is not None
        status.update(tunnel_parked=True)
        assert "FR:parked" in status.render_text(200).plain
        # The receipt names the resolved language.
        assert any("language: fr saved — active fr (French)" in n for n in _notices(app))

        # A switch BACK is the same mechanism in reverse (and proves the
        # constants were not merely frozen at import).
        app._run_slash_command("/language en")
        await pilot.pause()
        assert welcome_mod.MODEL_PENDING == "connecting…"
        assert status_line_mod.TUNNEL_PARKED_TEXT == "remote access off"
        assert status_line_mod.format_agents(2) == "2 agents"
        # The keyed-tip lookup re-materialised in reverse too.
        resolved_en = welcome_mod._resolve_tip("{key} starts a new conversation")
        assert "{key}" not in resolved_en, resolved_en


@pytest.mark.asyncio
async def test_a_change_from_another_process_applies_and_is_announced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app, watcher=True)
        _ship_fr(monkeypatch, tmp_path)
        _write_elsewhere(config_dir, "fr")
        process_watcher(config_dir).poll_now()
        await pilot.pause()
        # The TUI-owned apply ran off the watcher's delivery...
        assert welcome_mod.MODEL_PENDING == "FR:connecting…"
        assert status_line_mod.TUNNEL_PARKED_TEXT == "FR:parked"
        # ...and the announcement names the key that moved.
        notices = [n for n in _notices(app) if "config.yml changed" in n]
        # EN, not the marker locale: the receipt is built BEFORE the move is
        # applied (it announces the change in the language in force when it
        # fires, and the new language governs the next paint). So this asserts
        # the en bytes of the extracted key, which is the invariant the
        # extraction must preserve.
        assert notices == ["config.yml changed: applied: language"], notices


@pytest.mark.asyncio
async def test_a_write_from_this_process_applies_silently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app, watcher=True)
        _ship_fr(monkeypatch, tmp_path)
        # The /settings page's shape: the facade, notifying this process
        # LOCALLY (the command applies directly; this is the row's write).
        setting = settings_io.resolve_key("language")
        assert setting is not None
        settings_io.write_setting(ConfigManager(config_dir), setting, "fr")
        await pilot.pause()
        assert welcome_mod.MODEL_PENDING == "FR:connecting…"
        assert status_line_mod.TUNNEL_PARKED_TEXT == "FR:parked"
        # Silent: the page (or the command) already showed its own receipt.
        assert [n for n in _notices(app) if "config.yml changed" in n] == []


@pytest.mark.asyncio
async def test_an_open_settings_page_repaints_on_a_watcher_locale_move(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """QA round 1, Q2: a move written by ANOTHER process must repaint the page.

    Reproduced before the fix: the registry re-materialised (the model strings
    behind its frozen rows moved), but the open page kept its PAINTED strips and
    its row cache, so a Spanish page did not appear until the next keypress.
    ``es`` is a SHIPPED locale, so this drives the real catalogues rather than a
    fixture root. The frame is read without any key press between the write and
    the assert — that is the whole claim.
    """
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    monkeypatch.setenv("LOP_LANG", "en")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await _boot(pilot, app, watcher=True)
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        before = view.render_lines_for_test()
        assert any(line.startswith("Model") for line in before), before[:5]
        assert not any("Proveedor predeterminado" in line for line in before)

        _write_elsewhere(config_dir, "es")
        process_watcher(config_dir).poll_now()
        await pilot.pause()

        after = view.render_lines_for_test()
        assert any(line.startswith("Modelo") for line in after), after[:5]
        assert any("Proveedor predeterminado" in line for line in after), after[:8]
        # The English header is gone from the frame, not merely outnumbered.
        assert not any(line.startswith("Model ") for line in after), after[:5]


@pytest.mark.asyncio
async def test_the_app_paints_the_swapped_words_through_the_welcome_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F1 (round 1): the app must not carry its own bound copies of the words.

    Reproduced before the fix: after a locale move, ``welcome.MODEL_PENDING``
    read the new catalogue while the app's from-imported copy stayed at the
    boot ``en``, so the band painted the old word. The pin drives the two REAL
    paint paths (``_reset_band_for_swap``, ``_enter_setup_state``) and reads
    the band through its public render entry; the ``hasattr`` asserts keep the
    from-import class from coming back.
    """
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app)
        _ship_fr(monkeypatch, tmp_path)
        app._run_slash_command("/language fr")
        await pilot.pause()
        status = app._status
        assert status is not None
        app._reset_band_for_swap()
        assert "FR:connecting…" in status.render_text(200).plain
        app._enter_setup_state()
        assert "FR:setup" in status.render_text(200).plain
    # And the class itself: no from-imported copies survive on the app module.
    assert not hasattr(app_mod, "MODEL_PENDING")
    assert not hasattr(app_mod, "MODEL_SETUP")


@pytest.mark.asyncio
async def test_a_language_move_reaches_a_mounted_picker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The picker follows ``/language`` only because the app RE-HANDS its rows.

    ``CommandPicker.set_commands`` stores a ``list(...)`` COPY, and the painted
    rows are a second copy derived only by ``sync()``: the registry's in-place
    refresher reaches neither, so a mounted picker painted the boot language
    until the next keystroke (QA round 1, Q1). This drives the real app with the
    list ALREADY OPEN — one ``_set_editor_line`` before the move and none after
    — so it fails if ``self._refresh_offered_commands()`` leaves ``_apply_language``
    (nothing is re-handed) AND if ``Editor.set_commands`` stops re-deriving the
    open list (the copy is re-handed but not repainted). It deliberately reads
    the SHIPPED French copy, so the asserted row is the production sentence and
    not a fixture marker.
    """
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        _set_editor_line(editor, "/")
        await pilot.pause()
        picker = editor.picker
        # The boot rows are the English ones this process started in, painted.
        at_boot = [row.plain for row in picker.render_rows(picker.size.width or 100)]
        assert any("List all commands" in row for row in at_boot), at_boot

        app._apply_language("fr")
        await pilot.pause()
        # No keystroke, no re-sync: the list is already on screen, and the move
        # must repaint it in place.
        painted = [row.plain for row in picker.render_rows(picker.size.width or 100)]
    assert any("Lister toutes les commandes" in row for row in painted), painted
    assert not any("List all commands" in row for row in painted), painted
