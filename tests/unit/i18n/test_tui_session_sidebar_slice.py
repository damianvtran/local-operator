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
   ``lop stop`` command token survives verbatim. The fixture symlinks the real
   English tree so every OTHER namespace keeps rendering; only this fixture
   ever makes the synthetic locale eligible.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from local_operator.i18n import catalogues, resolve, text
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


async def _surfaces(pilot, app) -> dict[str, object]:
    """Every keyed surface, read from what the widget actually paints."""
    sidebar = await _sidebar_with(
        pilot,
        app,
        [_plain("act", active=True), _plain("old"), _sub("subrun1"), _untitled()],
        pins=("old",),
        show_subagents=True,
        total=1,
    )
    out: dict[str, object] = {}
    out["frame"] = [line.rstrip() for line in sidebar.render().plain.splitlines()]
    wedged = sidebar._describe(_wedged()).splitlines()[1]
    out["wedged"] = wedged
    out["remote"] = [
        sidebar._describe(_remote("r1", label="mbp")).splitlines()[2:4],
        sidebar._describe(_remote("r2", label="mbp", network="mesh")).splitlines()[2:4],
        sidebar._describe(_remote("r3", label="mbp", stale=True)).splitlines()[2],
        sidebar._describe(_remote("r4", label="mbp", network="mesh", stale=True)).splitlines()[2],
    ]
    sidebar._asks_total, sidebar._hub_total = 4, 2
    out["notes"] = (sidebar._asks_note(), sidebar._hub_note())
    sidebar._asks_total = sidebar._hub_total = 0
    out["notes_zero"] = (sidebar._asks_note(), sidebar._hub_note())
    sidebar.error = "boom"
    out["footer_error"] = sidebar.render().plain.splitlines()[-1].strip()
    sidebar.error = ""
    sidebar.requested_id = "act"
    out["footer_opening"] = sidebar.render().plain.splitlines()[-1].strip()
    sidebar.requested_id = None
    # Body states: an empty list in each of the three quiet states.
    sidebar.set_entries([])
    states = []
    for error, loading in (("boom", False), ("", True), ("", False)):
        sidebar.error, sidebar._catalog_loading = error, loading
        lines = [line.strip() for line in sidebar.render().plain.splitlines() if line.strip()]
        states.append(lines[:2])
    out["states"] = states
    sidebar.error, sidebar._catalog_loading = "", False
    return out


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
async def test_english_bytes_match_the_pre_extraction_literals() -> None:
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
        nameless = sidebar._describe(_remote("r5")).splitlines()[2]
        # owner_label falls back to the device id prefix, so the "unnamed
        # device" key is reached only with no device id at all.
        assert nameless == "on d_aaaa11"
        bare = CatalogEntry(
            SessionRow("r6", time.time(), "x", locality="remote", placement_stale=False)
        )
        assert sidebar._describe(bare).splitlines()[2] == "on unnamed device"


@pytest.fixture
def fr_sidebar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    (tmp_path / "en").symlink_to(EN_ROOT / "en", target_is_directory=True)
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


@pytest.mark.asyncio
async def test_every_surface_follows_a_live_language_change(fr_sidebar) -> None:
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
        assert app._session_sidebar._describe(bare).splitlines()[2] == "FR on FR unnamed device"
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
