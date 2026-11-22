"""The session sidebar's residual chrome, keyed under ``tui.session_sidebar``.

WHY THIS FILE EXISTS. The operator's goal is that switching language leaves no
English in the interface, and rendered frames still showed the sidebar's
section headings ("Active Sessions"), its body states, footer notes and the
wedged tooltip clause ("· lop stop if it stays silent"). None of those reached
the i18n scanner: they flow through ``truncate_cells``, a dict lookup or a
module constant, which ``scripts/i18n/check.py``'s collector cannot see (its
``_arguments`` docstring names the blind spot). So "the file is activated at
zero findings" proved nothing here; this file is the guard instead.

Two claims, each driven through the REAL render paths (``render()``,
``_describe``, ``_asks_note``, ``_hub_note``, the footer), not the catalogue:

1. English is byte-identical to the pre-extraction literals. The expected
   strings are spelled out literally on purpose — reading them back from the
   catalogue would let a wording change slip through as "still equal".
2. With a synthetic ``fr`` carrying only this namespace, every one of those
   surfaces paints the fr text, the ★/⌥ glyphs stay outside the words, and the
   ``lop stop`` command token survives verbatim. The fixture copies the real
   English catalogues so every OTHER namespace keeps rendering; only this fixture
   ever makes the synthetic locale eligible.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TypedDict

import pytest

from local_operator.i18n import catalogues, resolve, text
from local_operator.i18n.keys import tui_session_sidebar as sidebar_keys
from local_operator.resume import SessionRow
from local_operator.session.catalog import CatalogEntry, wedged_status
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.session_sidebar import peer_reason_words
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_session_sidebar import _plain, _sidebar_with, _sub

NS = "tui.session_sidebar"
EN_ROOT = Path(catalogues.__file__).resolve().parent / "catalogues"
#: The reason gloss is `peer_reason_words`'s (a later slice), not this namespace's.
REASON = peer_reason_words("offline")
BASE: dict[str, str] = json.loads((EN_ROOT / "en" / f"{NS}.json").read_text("utf-8"))


def _fr(value: str) -> str:
    """Prefix every literal run but keep ICU syntax intact.

    Only the leading text and each plural branch's text get the marker, so the
    message still parses and placeholders still substitute.
    """
    if value.startswith("{count, plural"):
        return value.replace("one {", "one {FR ").replace("other {", "other {FR ")
    return "FR " + value


def _remote(sid: str, *, network: str = "", stale: bool = False, label: str = "") -> CatalogEntry:
    now = time.time()
    return CatalogEntry(
        SessionRow(
            sid,
            now,
            f"Session {sid}",
            locality="remote",
            owner_device="d_aaaa1111",
            owner_device_name=label,
            owner_network_name=network,
            reachable=False,
            unreachable_reason="offline",
            placement_stale=stale,
        )
    )


def _wedged() -> CatalogEntry:
    return CatalogEntry(SessionRow("quiet0000001", time.time(), "Quiet", live_state="wedged"))


def _untitled() -> CatalogEntry:
    return CatalogEntry(SessionRow("noname000001", time.time(), ""))


class _Surfaces(TypedDict):
    """What ``_surfaces`` reads off the widget, typed so each assertion narrows.

    ``remote`` mixes shapes on purpose: the two reachable-state rows keep two
    tooltip lines (owner + reason), the two stale rows keep one.
    """

    frame: list[str]
    wedged: str
    remote: list[list[str] | str]
    notes: tuple[str, str]
    notes_zero: tuple[str, str]
    footer_error: str
    footer_opening: str
    states: list[list[str]]


def _tooltip(sidebar, entry: CatalogEntry) -> str:
    """``_describe`` is ``str | None`` (None only for no entry); assert it painted."""
    described = sidebar._describe(entry)
    assert described is not None
    return described


async def _surfaces(pilot, app) -> _Surfaces:
    """Every keyed surface, read from what the widget actually paints."""
    sidebar = await _sidebar_with(
        pilot,
        app,
        [_plain("act", active=True), _plain("old"), _sub("subrun1"), _untitled()],
        pins=("old",),
        show_subagents=True,
        total=1,
    )
    frame = [line.rstrip() for line in sidebar.render().plain.splitlines()]
    wedged = _tooltip(sidebar, _wedged()).splitlines()[1]
    remote: list[list[str] | str] = [
        sidebar._describe(_remote("r1", label="mbp")).splitlines()[2:4],
        sidebar._describe(_remote("r2", label="mbp", network="mesh")).splitlines()[2:4],
        sidebar._describe(_remote("r3", label="mbp", stale=True)).splitlines()[2],
        sidebar._describe(_remote("r4", label="mbp", network="mesh", stale=True)).splitlines()[2],
    ]
    sidebar._asks_total, sidebar._hub_total = 4, 2
    notes = (sidebar._asks_note(), sidebar._hub_note())
    sidebar._asks_total = sidebar._hub_total = 0
    notes_zero = (sidebar._asks_note(), sidebar._hub_note())
    sidebar.error = "boom"
    footer_error = sidebar.render().plain.splitlines()[-1].strip()
    sidebar.error = ""
    sidebar.requested_id = "act"
    footer_opening = sidebar.render().plain.splitlines()[-1].strip()
    sidebar.requested_id = None
    # Body states: an empty list in each of the three quiet states.
    sidebar.set_entries([])
    states: list[list[str]] = []
    for error, loading in (("boom", False), ("", True), ("", False)):
        sidebar.error, sidebar._catalog_loading = error, loading
        lines = [line.strip() for line in sidebar.render().plain.splitlines() if line.strip()]
        states.append(lines[:2])
    sidebar.error, sidebar._catalog_loading = "", False
    return _Surfaces(
        frame=frame,
        wedged=wedged,
        remote=remote,
        notes=notes,
        notes_zero=notes_zero,
        footer_error=footer_error,
        footer_opening=footer_opening,
        states=states,
    )


async def _overflow_painted(pilot, app) -> str:
    """The painted `+N more pinned` note: pinned rows that rank below the page.

    Read from the frame, not the catalogue, so it proves `render()` routes the
    note through the key (and the raw-digit `shown` argument).
    """
    entries = [_plain(f"p{i:02d}") for i in range(40)]
    sidebar = await _sidebar_with(pilot, app, entries, pins=("p37", "p38", "p39"))
    lines = [line.strip() for line in sidebar.render().plain.splitlines()]
    notes = [line for line in lines if "more pinned" in line]
    assert len(notes) == 1, lines
    return notes[0]


@pytest.mark.asyncio
async def test_english_bytes_match_the_pre_extraction_literals(fr_sidebar) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        got = await _surfaces(pilot, app)
        frame = got["frame"]
        for heading in ("★ Pinned", "Active Sessions", "Previous Sessions", "⌥ Subagent Runs"):
            assert heading in [line.strip() for line in frame], frame
        assert any("Untitled conversation" in line for line in frame), frame
        assert got["wedged"] == f"{wedged_status()} · lop stop if it stays silent"
        assert got["remote"] == [
            ["on mbp", f"unreachable · {REASON}"],
            ["on mbp · mesh", f"unreachable · {REASON}"],
            "on mbp (last known state)",
            "on mbp · mesh (last known state)",
        ]
        assert got["notes"] == ("asks: 4", "hub: 2")
        assert got["notes_zero"] == ("", "")
        assert got["footer_error"] == "Refresh failed"
        assert got["footer_opening"] == "Opening…"
        error, loading, empty = got["states"]
        assert error == ["Sessions", "Could not load conversations"]
        assert loading[0] == "Sessions" and loading[1].endswith(" Loading conversations…")
        assert empty == ["Sessions", "No conversations yet"]
        assert await _overflow_painted(pilot, app) == "+3 more pinned — scroll"
        sidebar = app._session_sidebar
        nameless = _tooltip(sidebar, _remote("r5")).splitlines()[2]
        # owner_label falls back to the device id prefix, so the "unnamed
        # device" key is reached only with no device id at all.
        assert nameless == "on d_aaaa11"
        bare = CatalogEntry(
            SessionRow("r6", time.time(), "x", locality="remote", placement_stale=False)
        )
        assert _tooltip(sidebar, bare).splitlines()[2] == "on unnamed device"


@pytest.fixture
def fr_sidebar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    # The loader rejects an English symlink escaping this fixture root and
    # silently renders raw IDs instead. Copy only the tiny catalogue data (not
    # dependencies or context sidecars); dependency sharing still applies.
    shutil.copytree(
        EN_ROOT / "en", tmp_path / "en", ignore=shutil.ignore_patterns("*.context.json")
    )
    (tmp_path / "fr").mkdir()
    (tmp_path / "fr" / f"{NS}.json").write_text(
        json.dumps({k: _fr(v) for k, v in BASE.items()}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    # Isolated so `set_active_locale` re-materialises nothing against tmp_path
    # (the nine-slice fixture's review round 1, B1, explains the order defect).
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


def _assert_fixture_english() -> None:
    """Pin a literal and its provenance: raw IDs must never look like English."""
    message = sidebar_keys.section_active()
    assert text.tr(NS)(message) == "Active Sessions"
    rendered = text.tr_record(NS)(message)
    assert rendered.text == "Active Sessions"
    assert rendered.source_locale == "en"


def test_synthetic_fixture_preserves_real_english(fr_sidebar) -> None:
    root = catalogues.catalogue_root()
    assert not (root / "en").is_symlink()
    originals = sorted((EN_ROOT / "en").glob("*.json"))
    originals = [path for path in originals if not path.name.endswith(".context.json")]
    assert originals
    assert sorted(path.name for path in (root / "en").iterdir()) == [
        path.name for path in originals
    ]
    for original in originals:
        copied = root / "en" / original.name
        assert not copied.is_symlink()
        assert copied.read_bytes() == original.read_bytes()
    assert catalogues.load_catalogue("en", NS)[sidebar_keys.section_active().code] == (
        "Active Sessions"
    )
    _assert_fixture_english()


def test_original_symlink_is_rejected_and_english_guard_detects_raw_ids(
    fr_sidebar, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Recreate the original defect beside the corrected fixture, retaining the
    # real loader/resolver. Its raw-ID fallback must trip the very same guard.
    broken = tmp_path / "original-symlink"
    broken.mkdir()
    (broken / "en").symlink_to(EN_ROOT / "en", target_is_directory=True)
    monkeypatch.setattr(catalogues, "_CATALOGUES", broken)
    with pytest.raises(catalogues.CatalogueNotFound, match="escapes the catalogue root"):
        catalogues.load_catalogue("en", NS)
    rendered = text.tr_record(NS)(sidebar_keys.section_active())
    assert rendered.text == sidebar_keys.section_active().code
    assert rendered.source_locale is None
    with pytest.raises(AssertionError):
        _assert_fixture_english()


@pytest.mark.asyncio
async def test_every_surface_follows_a_live_language_change(fr_sidebar) -> None:
    _assert_fixture_english()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        text.set_active_locale("fr")
        got = await _surfaces(pilot, app)
        stripped = [line.strip() for line in got["frame"]]
        for heading in (
            "★ FR Pinned",
            "FR Active Sessions",
            "FR Previous Sessions",
            "⌥ FR Subagent Runs",
        ):
            assert heading in stripped, got["frame"]
        assert any("FR Untitled conv" in line for line in got["frame"]), got["frame"]
        assert got["wedged"] == f"FR {wedged_status()} · lop stop if it stays silent"
        assert got["remote"][0] == ["FR on mbp", f"FR unreachable · {REASON}"]
        assert got["remote"][1][0] == "FR on mbp · mesh"
        assert got["remote"][2] == "FR on mbp (last known state)"
        assert got["remote"][3] == "FR on mbp · mesh (last known state)"
        assert got["notes"] == ("FR asks: 4", "FR hub: 2")
        assert got["footer_error"] == "FR Refresh failed"
        assert got["footer_opening"] == "FR Opening…"
        error, loading, empty = got["states"]
        # The fake's "FR " prefix pushes this one past the 29-cell content
        # width, so the widget's own crop applies — which is itself the claim:
        # translated copy goes through the same truncation as English.
        assert error[0] == "FR Sessions" and error[1].startswith("FR Could not load conv")
        assert loading[1].endswith(" FR Loading conversations…")
        assert empty == ["FR Sessions", "FR No conversations yet"]
        assert await _overflow_painted(pilot, app) == "FR +3 more pinned — scroll"
        bare = CatalogEntry(SessionRow("r6", time.time(), "x", locality="remote"))
        assert _tooltip(app._session_sidebar, bare).splitlines()[2] == "FR on FR unnamed device"
        assert text._refreshers == []


@pytest.fixture
def shipped_fr(monkeypatch: pytest.MonkeyPatch):
    """The REAL shipped catalogue tree and shipped set — no synthetic locale.

    The synthetic fixture above proves the wiring; this one proves the audited
    fr translation actually reaches the painted frame once fr is eligible
    through the generated ``shipped.json`` (a missing ledger row would un-ship
    fr and the frame would silently fall back to English).
    """
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


@pytest.mark.asyncio
async def test_shipped_fr_catalogue_paints_the_sidebar(shipped_fr) -> None:
    assert "fr" in resolve.shipped_locales()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        text.set_active_locale("fr")
        got = await _surfaces(pilot, app)
        assert "Sessions actives" in [line.strip() for line in got["frame"]], got["frame"]
        # `lop stop` is a command the operator types, so it must stay literal
        # inside the translated clause rather than being translated with it.
        assert got["wedged"] == f"{wedged_status()} · lop stop si le silence persiste"


def _drawn_cells(strip, note: str) -> range:
    """The terminal cells ``note`` occupies in the painted footer ``strip``.

    Located by cropping the rendered Strip in CELLS (Textual's own cell
    arithmetic, the one that puts glyphs on screen), never through the span
    helpers under test, so a bug in their index arithmetic cannot agree with
    itself here.
    """
    from rich.cells import cell_len

    width = cell_len(note)
    for start in range(strip.cell_length - width + 1):
        if strip.crop(start, start + width).text == note:
            return range(start, start + width)
    raise AssertionError(f"{note!r} not drawn in {strip.text!r}")


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["en", "zh-CN", "hi"])
async def test_footer_counter_presses_land_on_their_own_drawn_cells(shipped_fr, locale) -> None:
    """PR #4346 R1/Q1: every footer column presses its own counter or nothing.

    Translating the asks/hub notes put double-width (zh-CN `待答：4`) and
    zero-width combining (hi) glyphs on the line, and the hit tests compared
    the span helpers' STRING indices with pointer CELLS: in zh-CN the `4` of
    the asks note opened Hub links and no cell of `链接：2` answered. The sweep
    goes through the real press resolution QA used — ``App.mouse_position`` →
    ``focus_on_click`` — at every column of the footer row, and English is the
    control that the conversion leaves untouched.
    """
    from textual.geometry import Offset

    assert locale in resolve.shipped_locales()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(140, 34)) as pilot:
        await pilot.pause()
        text.set_active_locale(locale)
        sidebar = await _sidebar_with(pilot, app, [_plain("a", active=True)], total=1)
        sidebar.set_asks_total(4)
        sidebar.set_hub_total(2)
        await pilot.pause()
        y = sidebar.size.height - 1
        strip = sidebar.render_line(y)
        pad = int(sidebar.styles.padding.left)
        expected = {"asks": _drawn_cells(strip, sidebar._asks_note())}
        expected["hub"] = _drawn_cells(strip, sidebar._hub_note())
        expected["chip"] = _drawn_cells(strip, "⌥1")
        got: dict[str, list[int]] = {"asks": [], "hub": [], "chip": []}
        for column in range(-pad, sidebar.size.width + pad):
            app.mouse_position = Offset(sidebar.region.x + pad + column, sidebar.region.y + y)
            sidebar.focus_on_click()
            hits = [
                name
                for name, flag in (
                    ("asks", sidebar._press_asks_hit),
                    ("hub", sidebar._press_hub_hit),
                    ("chip", sidebar._press_chip_hit),
                )
                if flag
            ]
            assert len(hits) <= 1, (column, hits)
            for name in hits:
                got[name].append(column)
        assert got == {name: list(cells) for name, cells in expected.items()}, strip.text
