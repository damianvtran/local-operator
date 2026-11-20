"""The ``/settings`` page's residual chrome, keyed under ``tui.settings_view``.

WHY THIS FILE EXISTS. The operator's goal is that switching language leaves no
English in the interface, and the settings page still painted English through
paths ``scripts/i18n/check.py``'s collector cannot see: a module constant
(the too-short notice), f-strings built from an enum value (the section's
``takes effect:`` tag) and from a pane id (``no teams configured``), and
literals passed through ``truncate_cells`` or a widest-first ladder (the
editor's key hints). "The file is activated at zero findings" proved nothing
for those, so this file is the guard instead.

Two claims, each driven through the REAL paint paths (``render_lines_for_test``,
``_row_text``, ``_editor_text``, ``rendered_pane``, the detail row), not the
catalogue:

1. English is byte-identical to the pre-extraction literals. The expected
   strings are spelled out literally on purpose — reading them back from the
   catalogue would let a wording change slip through as "still equal".
2. With a synthetic ``fr`` carrying only this namespace, every one of those
   surfaces paints the fr text while key names (``enter``, ``esc``, ``tab``,
   ``↑↓``) and commands (``/login <provider>``, ``/team``, ``/agent``) survive
   verbatim. The fixture symlinks the real English tree so every OTHER
   namespace keeps rendering; only this fixture ever makes the synthetic
   locale eligible. A mutation control proves claim 2 discriminates: restoring
   one literal makes it fail.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from local_operator.i18n import catalogues, resolve, text
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets import settings_view as sv_mod
from local_operator.tui.widgets.settings_view import SettingsView, _Row
from tests.unit.tui.test_app_pilot import FakeSession, _factory

NS = "tui.settings_view"
EN_ROOT = Path(catalogues.__file__).resolve().parent / "catalogues"
BASE: dict[str, str] = json.loads((EN_ROOT / "en" / f"{NS}.json").read_text("utf-8"))


def _fr(value: str) -> str:
    return "FR " + value


def _contract(view: SettingsView, key: str, monkeypatch: pytest.MonkeyPatch, **patch: Any) -> str:
    """The editor's key-hint tail for ``key``, painted by ``_editor_text``.

    The buffer is non-empty and the caret at its end so no placeholder ghost is
    appended; what follows the caret glyph is exactly the contract.
    """
    view._editing, view._buffer, view._caret = key, "1", 1
    for name, value in patch.items():
        monkeypatch.setattr(view, name, value)
    return view._editor_text().plain.split("▏", 1)[1]


def _suggest_ladder(view: SettingsView, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every distinct contract the dropdown ladder paints, widest first."""
    seen: list[str] = []
    for width in range(200, 20, -1):
        got = _contract(
            view,
            "retry.baseDelayMs",
            monkeypatch,
            _suggest_rows=lambda: [object()],
            _list_width=lambda w=width: w,
        )
        if got not in seen:
            seen.append(got)
    return seen


async def _surfaces(
    pilot: Any, app: OperatorApp, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    app._open_settings_view()
    view = app.query_one(SettingsView)
    await pilot.pause()
    out: dict[str, Any] = {}
    lines = view.render_lines_for_test()
    # The tag is painted at the shared value column of each section header;
    # read from there so the assertion is the tag exactly, in any language.
    out["tags"] = sorted(
        {line[sv_mod._VALUE_COLUMN :].rstrip() for line in lines if "takes effect:" in line}
    )
    out["add"] = [
        view._row_text(_Row(kind="hop_add", chain="default"), -1, 100).plain.strip(),
        view._row_text(_Row(kind="chain_add"), -1, 100).plain.strip(),
    ]
    out["plain"] = _contract(view, "retry.baseDelayMs", monkeypatch, _list_width=lambda: 200)
    out["unset"] = _contract(
        view, "network.memory.exclude_devices", monkeypatch, _list_width=lambda: 200
    )
    out["ladder"] = _suggest_ladder(view, monkeypatch)
    view._editing = None
    view.load(teams=[], agents=[], providers=[])
    await pilot.pause()
    out["teams"] = [line.strip() for line in view.rendered_pane().splitlines() if line.strip()]
    view.action_pane(1)
    await pilot.pause()
    out["agents"] = [line.strip() for line in view.rendered_pane().splitlines() if line.strip()]
    # More logged-in providers than the pane's height holds: the provider list
    # sheds rows into the `… N more` spill count.
    view.load(providers=[(f"prov{i:02d}", "signed in") for i in range(40)], teams=[], agents=[])
    await pilot.pause()
    pane = [line.strip() for line in view.rendered_pane().splitlines() if line.strip()]
    out["spill"] = [line for line in pane if "more" in line]
    return out


async def _too_short(app: OperatorApp, pilot: Any) -> str:
    app._open_settings_view()
    view = app.query_one(SettingsView)
    await pilot.pause()
    await pilot.pause()
    assert view._too_short
    return view._detail_text.plain


@pytest.mark.asyncio
async def test_english_bytes_match_the_pre_extraction_literals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        got = await _surfaces(pilot, app, monkeypatch)
    assert got["tags"] == [
        "takes effect: live",
        "takes effect: never",
        "takes effect: new launch",
        "takes effect: new sessions",
    ], got["tags"]
    assert got["add"] == ["+ add a hop", "+ add a chain"], got["add"]
    assert got["plain"] == "  enter saves · esc cancels"
    assert got["unset"] == "  enter saves · esc cancels · clear to unset"
    assert got["ladder"] == [
        "  ↑↓ suggestions · tab completes · enter saves · esc",
        "  ↑↓ browse · tab complete · enter saves",
        "  ↑↓ · tab · enter saves",
        "  enter saves · esc cancels",
    ], got["ladder"]
    # The pane is width-capped, so longer lines end in the widget's own crop;
    # prefixes are asserted there, and the crop itself shows translated copy
    # goes through the same truncation as English.
    assert got["teams"][:4] == [
        "providers",
        "none logged in — /login <" + ("…" if "" else "pro…"),
        "teams  agents  (←→)",
        "no teams configured",
    ], got["teams"]
    assert got["teams"][4] == "/team lists and attaches them", got["teams"]
    assert got["teams"][-1] == "read-only", got["teams"]
    assert len(got["spill"]) == 1 and re.fullmatch(r"… \d+ more", got["spill"][0]), got["spill"]
    assert got["agents"][3] == "no agents configured", got["agents"]
    assert got["agents"][4].startswith("/agent lists"), got["agents"]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 13)) as pilot:
        await pilot.pause()
        assert "terminal too short for settings — make it taller" in await _too_short(app, pilot)


@pytest.fixture
def fr_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only this fixture may make a synthetic locale eligible, never production."""
    # COPIED, not symlinked: `catalogues.load_catalogue` resolves the path and
    # refuses one that escapes the catalogue root, so a symlinked `en` makes
    # every OTHER namespace degrade to its raw key instead of falling back to
    # English (observed: `tui.small.settings_view.read_only_label` painted in
    # the pane). A copy keeps the fallback the product actually has.
    shutil.copytree(EN_ROOT / "en", tmp_path / "en")
    (tmp_path / "fr").mkdir()
    (tmp_path / "fr" / f"{NS}.json").write_text(
        json.dumps({k: _fr(v) for k, v in BASE.items()}, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(catalogues, "_CATALOGUES", tmp_path)
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.setattr(resolve, "shipped_locales", lambda: ("en", "fr"))
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


async def _assert_fr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every keyed surface repaints in the synthetic fr after a live switch."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        text.set_active_locale("fr")
        got = await _surfaces(pilot, app, monkeypatch)
    # The tag's prefix is "FR takes effect: " and the scope word is itself fr.
    assert got["tags"] == [
        "FR takes effect: FR live",
        "FR takes effect: FR never",
        "FR takes effect: FR new launch",
        "FR takes effect: FR new sessions",
    ], got["tags"]
    assert got["add"] == ["FR + add a hop", "FR + add a chain"], got["add"]
    assert got["plain"] == "  FR enter saves · esc cancels"
    assert got["unset"] == "  FR enter saves · esc cancels · clear to unset"
    assert got["ladder"] == [
        "  FR ↑↓ suggestions · tab completes · enter saves · esc",
        "  FR ↑↓ browse · tab complete · enter saves",
        "  FR ↑↓ · tab · enter saves",
        "  FR enter saves · esc cancels",
    ], got["ladder"]
    # The pane is width-capped, so longer lines end in the widget's own crop;
    # prefixes are asserted there, and the crop itself shows translated copy
    # goes through the same truncation as English.
    assert got["teams"][:4] == [
        "FR providers",
        "FR none logged in — /login <" + ("…" if "FR " else "pro…"),
        "FR teams  FR agents  (←→)",
        "FR no teams configured",
    ], got["teams"]
    assert got["teams"][4].startswith("FR /team lists"), got["teams"]
    # Other namespaces still fall back to English, never to a raw key.
    assert got["teams"][-1] == "read-only", got["teams"]
    assert len(got["spill"]) == 1 and re.fullmatch(r"FR … \d+ more", got["spill"][0]), got["spill"]
    assert got["agents"][3] == "FR no agents configured", got["agents"]
    assert got["agents"][4].startswith("FR /agent lists"), got["agents"]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 13)) as pilot:
        await pilot.pause()
        text.set_active_locale("fr")
        assert "FR terminal too short for settings" in await _too_short(app, pilot)


@pytest.mark.asyncio
async def test_every_surface_follows_a_live_language_change(
    fr_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _assert_fr(monkeypatch)
    assert text._refreshers == []


@pytest.mark.asyncio
async def test_mutation_control_a_restored_literal_fails_the_fr_check(
    fr_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring ONE literal (the `+ add a hop` row) must break claim 2.

    Simulated at the resolver: the add-hop key paints its English literal
    instead of resolving, exactly what reverting that call site would do.
    Without this, a fr check that passed vacuously would look identical.
    """
    real = sv_mod._t_sv

    def restored(message):
        if message.code == f"{NS}.row.add_hop":
            return "+ add a hop"
        return real(message)

    monkeypatch.setattr(sv_mod, "_t_sv", restored)
    with pytest.raises(AssertionError, match=r"'\+ add a hop', 'FR \+ add a chain'"):
        await _assert_fr(monkeypatch)


@pytest.fixture
def shipped_fr(monkeypatch: pytest.MonkeyPatch):
    """The REAL shipped catalogue tree and shipped set — no synthetic locale.

    The synthetic fixture above proves the wiring; this one proves the audited
    fr translation actually reaches the painted frame once fr is eligible
    through the generated ``shipped.json`` (a missing ledger row would un-ship
    fr and the settings page would silently fall back to English).
    """
    monkeypatch.setattr(text, "_active_locale", "en")
    monkeypatch.setattr(text, "_refreshers", [])
    monkeypatch.delenv("LOP_I18N", raising=False)
    text._sources.cache_clear()
    yield
    text.set_active_locale("en")
    text._sources.cache_clear()


@pytest.mark.asyncio
async def test_shipped_fr_catalogue_paints_the_settings_view(shipped_fr) -> None:
    assert "fr" in resolve.shipped_locales()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        text.set_active_locale("fr")
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        add_hop = view._row_text(_Row(kind="hop_add", chain="default"), -1, 100).plain.strip()
        lines = view.render_lines_for_test()
    assert add_hop == "+ ajouter une étape"
    # Same column read as the English check: the tag exactly, so a partially
    # translated tag (fr frame, English scope word) cannot pass. The audited
    # catalogue puts U+202F (narrow no-break space) before the colon, per
    # French typography; spelled as an escape so it cannot be "tidied" away.
    tags = {line[sv_mod._VALUE_COLUMN :].rstrip() for line in lines if "prise d’effet" in line}
    assert "prise d’effet\u202f: immédiat" in tags, tags
    assert not any("takes effect" in line for line in lines), lines


async def _paint_every_surface(
    view: SettingsView, pilot: Any, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Paint each keyed surface once through its real paint path.

    Any projection error (an RTL locale rejecting a control character it was
    handed, a technical span the display boundary refuses) raises out of here,
    which is the whole point: these crashes only surface at paint time.
    """
    out: dict[str, Any] = {}
    out["lines"] = view.render_lines_for_test()
    out["add"] = view._row_text(_Row(kind="hop_add", chain="default"), -1, 100).plain
    out["plain"] = _contract(view, "retry.baseDelayMs", monkeypatch, _list_width=lambda: 200)
    out["unset"] = _contract(
        view, "network.memory.exclude_devices", monkeypatch, _list_width=lambda: 200
    )
    out["ladder"] = _suggest_ladder(view, monkeypatch)
    view._editing = None
    teams = [(f"team{i:02d}", "3 members", "") for i in range(12)]
    providers = [(f"prov{i:02d}", "signed in") for i in range(40)]
    view.load(teams=teams, agents=[], providers=providers)
    await pilot.pause()
    out["teams"] = view.rendered_pane()
    view.action_pane(1)
    await pilot.pause()
    out["agents"] = view.rendered_pane()
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", resolve.shipped_locales())
async def test_every_shipped_locale_paints_the_settings_view(
    shipped_fr, locale: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for PR #4361 review R1/R2 (QA Q1/Q2): `ur` crashed on open.

    The scope word was projected for display (Urdu wraps it in bidi isolates)
    and then substituted into `takes effect: {scope}`, whose own projection
    rejects control characters; and `↑↓` was declared a technical parameter,
    which the display boundary admits only as printable ASCII. Both raised at
    paint, so only a frame painted with the REAL shipped catalogue in every
    locale can catch the class — not a synthetic one, and not English.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(180, 40)) as pilot:
        await pilot.pause()
        text.set_active_locale(locale)
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        got = await _paint_every_surface(view, pilot, monkeypatch)
    # Each scope tag sits on the shared value column; four sections carry one.
    tagged = [line for line in got["lines"] if line[sv_mod._VALUE_COLUMN :].strip()]
    assert tagged, got["lines"]
    # The widest ladder rung names the browse keys verbatim in every locale.
    assert "↑↓" in got["ladder"][0], got["ladder"]
    assert len(got["ladder"]) == 4, got["ladder"]
    # A projected string balances its isolates: every opener has a closer.
    for painted in [got["add"], got["plain"], got["unset"], *got["ladder"], *got["lines"]]:
        opens = sum(painted.count(c) for c in "\u2066\u2067\u2068")
        assert opens == painted.count("\u2069"), repr(painted)
    assert got["teams"].rstrip().endswith(got["agents"].rstrip().splitlines()[-1])


@pytest.mark.asyncio
async def test_es_roster_spill_count_never_wraps_the_tab_row(shipped_fr) -> None:
    """QA Q3 / design D3: the collapsed roster count rode the tab row unbudgeted.

    In es `equipos  agentes  (←→)  … 12 más` is 32 cells against the pane's
    31, so the row wrapped and pushed the `solo lectura` caption out of the
    pane at 120x34. Every pane line must fit the pane's width (a wrapped line
    costs a row the height budget never counted) and the caption must survive.
    """
    from rich.cells import cell_len

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.pause()
        text.set_active_locale("es")
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        view.load(
            teams=[(f"team{i:02d}", "3 members", "") for i in range(12)],
            agents=[],
            providers=[(f"prov{i:02d}", "signed in") for i in range(40)],
        )
        await pilot.pause()
        pane = view.rendered_pane().splitlines()
        width = view._pane_width()
        height = view._pane_height()
    assert all(cell_len(line) <= width for line in pane), [
        (cell_len(line), line) for line in pane if cell_len(line) > width
    ]
    assert len(pane) <= height, pane
    assert pane[-1].strip() == "solo lectura", pane
    # The roster still says it is withholding entries rather than going silent.
    assert any("más" in line and "equipos" in line for line in pane), pane
