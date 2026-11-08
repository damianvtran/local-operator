"""``/language``: the command surface (slice C).

The command is the discoverable half of the ``language`` setting; its tests
are about the four shapes a user can meet — bare (what am I on), a write
(persists, receipts), a no-op (already there), and a refusal (unknown, or a
supported locale no wave has shipped yet). The LIVE half — a switch or a
foreign write moving what the running TUI resolves — is
``test_language_live.py``; both files share the isolated config dir, so a
committed switch never touches the operator's real config.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.cells import cell_len

from local_operator.config import ConfigManager
from local_operator.i18n import catalogues as i18n_catalogues
from local_operator.i18n import resolve
from local_operator.i18n import text as i18n_text
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tui.app import OperatorApp
from local_operator.tui.autocomplete import ArgumentChoice
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _restore_locale() -> Iterator[None]:
    """The active locale is PROCESS state a committed switch moves; put it back.

    ``test_a_shipped_wave_locale_switches_and_persists`` leaves the process in
    ``fr``, and every later test in the same worker then resolves French copy:
    CI shard ``test (3.12, 1)`` failed twice on
    ``test_marker_chip.py::test_the_receipt_row_is_inked_as_chrome_not_as_the_user_s_prose``
    reading ``1 image jointe`` (PR #2377, reproduced locally on main ``8492cebb5``
    by running this file before that test). The sibling ``test_language_live.py``
    restores the same way; this file never patches the catalogue root, so a plain
    move back to ``en`` is enough.
    """
    yield
    i18n_text.set_active_locale("en")


@pytest.fixture()
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point the config at a temp dir — a committed switch WRITES one.

    Also empties the process's config-watcher registry on both sides of the test.
    The registry (``config_watch._watchers``) is keyed by DIRECTORY PATH and
    outlives the app that created a watcher; ``language`` is read from that
    watcher's snapshot (``resolve.config_value``). pytest's
    ``tmp_path_retention_policy = "failed"`` deletes a passing test's directory,
    and its numbered-dir allocator then hands the SAME path to the next test with
    the same name prefix — so a parametrised case inherited the previous case's
    snapshot (``language: en``), ``/language en`` answered "already", wrote
    nothing, and ``_stored`` read an empty file (``assert None == 'en'``). Which
    case follows which depends on xdist's worksteal scheduling, which is why it
    failed only occasionally under the default ``-n auto``; serially, with
    ``-k "refusal_fits and -en]" -n0``, it reproduces every time
    (``[cjk30-en]`` after ``[ascii40-en]``).
    """
    from local_operator import config_watch

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    config_watch._reset_for_tests()
    yield tmp_path
    config_watch._reset_for_tests()


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _type(pilot, app: OperatorApp, text: str) -> None:
    # Set the line AND park the caret at the end, as typing it would: the slash
    # pickers are caret-anchored (inline detection), so a text-set that leaves
    # the caret at the origin sits before the slash where no command is live.
    editor = app.query_one(Editor)
    editor.text = text
    editor.move_cursor(editor._end_of_buffer())
    editor._sync_picker()
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [block.text() or "" for block in app.query(NoticeBlock)]


def _notices_since(app: OperatorApp, before: int) -> list[str]:
    """Every notice the command printed after ``before`` rows already existed.

    Structural, not lexical: a prefix list ("language:", "langue", ...) only
    recognises the receipts it was written for, so a translated refusal whose
    first word is not on the list (zh-CN ``未知语言``, ru ``неизвестный``, ur
    ``نامعلوم``, hi ``अज्ञात``) silently escaped the cell bound it was meant to
    enforce. Taking "whatever was appended since the baseline" needs no
    knowledge of any locale's wording.
    """
    return _notices(app)[before:]


def _receipt_cap(key: str) -> int:
    """The ``maxCells`` the EN context sidecar declares for a receipt key.

    Read from the sidecar the translators are briefed with, so the test and the
    brief cannot drift apart on the number.
    """
    path = Path(i18n_catalogues.__file__).parent / "catalogues" / "en"
    sidecar = json.loads((path / "wire.language_receipts.context.json").read_text(encoding="utf-8"))
    cap = sidecar[f"wire.language_receipts.{key}"]["maxCells"]
    assert isinstance(cap, int), (key, cap)
    return cap


def _choice_map(picker) -> dict[str, ArgumentChoice]:
    """The open argument list as name -> row, narrowed for the type checker."""
    rows: dict[str, ArgumentChoice] = {}
    for name, choice in picker.suggestions():
        assert isinstance(choice, ArgumentChoice), choice
        rows[name] = choice
    return rows


def _stored(config_dir: Path) -> str | None:
    value = ConfigManager(config_dir).get_config().values.get("language")
    return value if isinstance(value, str) else None


@pytest.mark.asyncio
async def test_bare_language_names_the_active_language(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        before = len(_notices(app))
        app._run_slash_command("/language")
        await pilot.pause()
        texts = _notices(app)
        # Resolved language, its English name, the value space, and the space
        # as the browse — the accepted command shape.
        assert any(
            text.startswith("language: en (English)")
            and "/language <value>; space browses" in text
            and "; space" in text
            for text in texts
        ), texts
        for text in _notices_since(app, before):
            assert cell_len(text) <= 70, text
        # The bare receipt names the space browse instead of enumerating the
        # value space (which grows with every shipped locale). This loop only
        # proves the receipt tracks the switch; the cell bound for EVERY
        # shipped locale is enforced by the parametrised tests at the end of
        # this file, which is where a translation that overruns the row fails.
        for value in ("en", "fr", "es", "ru", "vi", "ur", "hi", "auto"):
            app._run_slash_command(f"/language {value}")
            await pilot.pause()
            app._run_slash_command("/language")
            await pilot.pause()
            # LOOSENED from the English ``startswith("language: <tag> (")``: the
            # switch re-renders the receipt in the chosen locale (the keyed
            # wire.language_receipts.show_terminal), so the leading word is the
            # translation's own and is no longer asserted here. What is still
            # checked is the resolved tag with its opening parenthesis and the
            # literal ``/language <value>`` command, which the translator
            # brief keeps verbatim. The full English sentence stays pinned for
            # ``en`` by the strict assertion above. This is a weaker check than
            # before, on purpose: wording belongs to the catalogues; the
            # width contract is the parametrised tests below.
            shown = "en" if value == "auto" else value
            assert any(
                f" {shown} (" in text and "/language <value>" in text
                for text in _notices_since(app, before)
            ), (value, _notices(app))
        for text in _notices_since(app, before):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_a_switch_persists_the_same_key_the_settings_row_writes(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        before = len(_notices(app))
        app._run_slash_command("/language en")
        await pilot.pause()
        # ONE truth: the registry key the /settings row owns, written through
        # the same facade (`settings_io.write_setting`).
        assert _stored(config_dir) == "en"
        assert any(
            text.startswith("language: en saved — active en (English)") for text in _notices(app)
        ), _notices(app)
        for text in _notices_since(app, before):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_switching_to_auto_when_already_auto_writes_nothing(
    config_dir: Path,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        before = len(_notices(app))
        app._run_slash_command("/language auto")
        await pilot.pause()
        # No config value existed; `auto` IS the absent value, so the no-op
        # receipt must not flip it into an explicitly stored one.
        assert _stored(config_dir) is None
        assert any("language: already auto" in text for text in _notices(app)), _notices(app)
        for text in _notices_since(app, before):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_the_lang_alias_resolves_to_the_same_command(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        before = len(_notices(app))
        app._run_slash_command("/lang en")
        await pilot.pause()
        # The one-resolver rule: `/lang` reaches the same handler, the same
        # write and the same receipt.
        assert _stored(config_dir) == "en"
        assert any(
            text.startswith("language: en saved — active en (English)") for text in _notices(app)
        ), _notices(app)
        for text in _notices_since(app, before):
            assert cell_len(text) <= 70, text


@pytest.mark.asyncio
async def test_the_space_opens_the_choices_on_the_current_value(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _type(pilot, app, "/language ")
        picker = editor.picker
        assert picker.is_open(), "the space must open the choices list"
        rows = _choice_map(picker)
        assert list(rows) == [
            "auto",
            "en",
            "fr",
            "es",
            "zh-CN",
            "ru",
            "vi",
            "ur",
            "hi",
        ], list(rows)
        assert picker.highlighted_name() == "auto", "the browse starts where the user is"
        assert rows["auto"].detail == "← current"
        # After a committed switch the marker moves with the value.
        app._run_slash_command("/language en")
        await pilot.pause()
        editor.focus()
        await _type(pilot, app, "/language ")
        picker = editor.picker
        rows = _choice_map(picker)
        assert picker.highlighted_name() == "en"
        assert rows["en"].detail == "← current"
        assert rows["auto"].detail == ""


@pytest.mark.asyncio
async def test_unknown_and_unshipped_values_are_refused_with_the_affordance(
    config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every supported locale now ships (Chinese after its 2026-10-24
    # disposition), so the supported-but-unshipped refusal is exercised by
    # narrowing the generated set in isolation. Eligibility, not recognition,
    # must still refuse the write.
    monkeypatch.setattr(
        resolve, "shipped_locales", lambda: ("en", "fr", "es", "ru", "vi", "ur", "hi")
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        before = len(_notices(app))
        for value in ("klingon", "zh-CN"):
            app._run_slash_command(f"/language {value}")
            await pilot.pause()
            receipt = f"unknown language '{value}' — press space to list values"
            assert receipt in _notices(app), _notices(app)
            assert _stored(config_dir) is None
        for text in _notices_since(app, before):
            assert cell_len(text) <= 70, text
        # The echoed token is bounded by CELLS, not characters: a wide token
        # would otherwise push the refusal past the row.
        for token in ("語" * 30, "\\" * 20):
            app._run_slash_command("/language " + token)
            await pilot.pause()
            long_notice = _notices(app)[-1]
            assert "…" in long_notice, long_notice
            assert cell_len(long_notice) <= 70, long_notice


@pytest.mark.asyncio
async def test_a_shipped_wave_locale_switches_and_persists(config_dir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        before = len(_notices(app))
        # Wave 1 shipped es+fr: the call that was refused pre-wave now keys its
        # catalogue and stores the value through the shared facade.
        app._run_slash_command("/language fr")
        await pilot.pause()
        assert _stored(config_dir) == "fr"
        # The fr receipt is the shipped wire.language_receipts.saved copy, with
        # French typography (a narrow no-break space before the colon).
        assert any(
            text.startswith("langue\u202f: fr enregistrée — fr (Français) active")
            for text in _notices(app)
        ), _notices(app)
        for text in _notices_since(app, before):
            assert cell_len(text) <= 70, text


# Worst-case clipped echoes. The app clips the typed value to 20 cells and
# adds two quotes (``_quoted_echo_within_cells``), so the widest ``{echo}`` is
# 22 cells. An ASCII run reaches that exactly; CJK and backslashes (repr
# doubles them) exercise the width-vs-character-count paths.
_WORST_CASE_ECHOES = ("a" * 40, "語" * 30, "\\" * 40)


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", resolve.shipped_locales())
async def test_the_bare_receipt_fits_one_notice_row_in_every_shipped_locale(
    config_dir: Path, locale: str
) -> None:
    """Bare ``/language`` is one row within the sidecar cap in each locale.

    The locale list is the shipped manifest (``resolve.shipped_locales``), so a
    newly shipped locale is covered without editing this file. The handler is
    the real one; the locale is entered through ``/language <locale>`` exactly
    as a user would.
    """
    cap = _receipt_cap("show_terminal")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command(f"/language {locale}")
        await pilot.pause()
        assert resolve.resolve_language() == locale
        blocks_before = len(list(app.query(NoticeBlock)))
        app._run_slash_command("/language")
        await pilot.pause()
        blocks = list(app.query(NoticeBlock))[blocks_before:]
        assert len(blocks) == 1, [b.text() for b in blocks]
        text = blocks[0].text() or ""
        assert not blocks[0].spans_multiple_rows(), text
        assert "\n" not in text, text
        assert cell_len(text) <= cap, (locale, cell_len(text), cap, text)


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", resolve.shipped_locales())
@pytest.mark.parametrize("echo", _WORST_CASE_ECHOES, ids=("ascii40", "cjk30", "backslash40"))
async def test_the_unknown_value_refusal_fits_one_notice_row_in_every_shipped_locale(
    config_dir: Path, locale: str, echo: str
) -> None:
    """The refusal of a long mistyped value is one row within the cap, per locale.

    The English-only version of this check let es (78 cells) and vi (82) ship
    over the 70-cell row: those catalogues spend more cells around ``{echo}``,
    so only a run in each locale sees it. Nothing is filtered by wording — the
    refusal is whatever the handler appended — so an untranslated prefix cannot
    hide a row from the bound.
    """
    cap = _receipt_cap("unknown_terminal")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await _boot(pilot, app)
        app._run_slash_command(f"/language {locale}")
        await pilot.pause()
        assert resolve.resolve_language() == locale
        blocks_before = len(list(app.query(NoticeBlock)))
        app._run_slash_command("/language " + echo)
        await pilot.pause()
        blocks = list(app.query(NoticeBlock))[blocks_before:]
        assert len(blocks) == 1, [b.text() for b in blocks]
        text = blocks[0].text() or ""
        assert "…" in text, text  # the echo really was clipped (the worst case)
        assert not blocks[0].spans_multiple_rows(), text
        assert "\n" not in text, text
        assert cell_len(text) <= cap, (locale, cell_len(text), cap, text)
        assert _stored(config_dir) == locale  # a refusal writes nothing
